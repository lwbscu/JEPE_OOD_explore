#!/usr/bin/env python3
"""Offline all-AND gate for the Cube off-policy V2 predictor.

V2 is evaluated on two frozen, disjoint protocols:

* the old unseeded red/blue-v2/yellow-v2 12x300 audit pools, using their
  already-cached physical outcomes; and
* the exact 2,000 clean expert-action segments frozen by Measurement 1.

The MaskedAug encoder/projector and XYZ probe are reusable only when the V2
checkpoint is elementwise identical on those two submodules.  All checkpoint,
probe, audit-pool, expert-segment, dataset, manifest, and memory-index
identities are written before physical outcome arrays or labels are joined.
Formal online evaluation is authorized only when every color passes both pool
criteria and the expert depth-5 criterion passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
LEWM_ROOT = HERE.parent
AILAB_ROOT = LEWM_ROOT.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cube_imagination_error_common as imag  # noqa: E402
import cube_probe_common as probe_common  # noqa: E402
import evaluate_cube_offpolicy as v1  # noqa: E402
import run_cube_imagination_error as measurement  # noqa: E402


DATASET = v1.DATASET
MANIFEST = v1.MANIFEST
BASE_CHECKPOINT = v1.BASE_CHECKPOINT
MASKED_PROBE = v1.MASKED_PROBE
MEMORY_INDEX = v1.MEMORY_INDEX
CHECKPOINT_ROOT = AILAB_ROOT / "checkpoints/lewm-cube-offpolicy_v2"
DEFAULT_CHECKPOINT = CHECKPOINT_ROOT / "offpolicy_v2_pred_seed3072/weights_final.pt"
OUTPUT_ROOT = AILAB_ROOT / "outputs/eval/cube/offpolicy_v2"
OFFLINE_ROOT = OUTPUT_ROOT / "offline"
TRAIN_OUTPUT_ROOT = AILAB_ROOT / "outputs/train/offpolicy_v2"
EXPERT_SEGMENTS = AILAB_ROOT / "outputs/eval/cube/imagination_error/measurement1_segments.json"

CONDITIONS = v1.CONDITIONS
AUDIT_ENVS = v1.AUDIT_ENVS
NUM_CANDIDATES = v1.NUM_CANDIDATES
MODEL_LABELS = v1.MODEL_LABELS
CSV_FIELDS = v1.CSV_FIELDS
SUCCESS_K = v1.SUCCESS_K
EXPERT_COUNT = 2_000
EXPERT_DEPTHS = tuple(range(1, imag.HORIZON + 1))
EXPERT_FIELDS = (
    "model",
    "segment_id",
    "episode_idx",
    "start_row",
    "target_row",
    "depth",
    "action_teacher_forcing",
    "latent_teacher_forcing",
    "E_roll_mm",
    "E_enc_mm",
    "E_imag_mm",
    "Delta_roll_minus_enc_mm",
    "latent_l2",
    "latent_cosine_distance",
    "roll_gt_40mm",
)

# These are frozen protocol thresholds, not values inferred from a new run.
# They are the exact/near-exact halves of the Masked same-pool rates reported
# before V2 was designed; the displayed percentages are 31.222/38.500/38.514.
RATE_THRESHOLDS = {
    "red": 0.31222222222222223,
    "blue_v2": 0.385,
    "yellow_v2": 0.3851388888888889,
}
EXPECTED_BASE = {
    "red": {
        "median_E_roll_mm": 85.72032358672334,
        "roll_gt_40mm_rate": 0.6244444444444445,
    },
    "blue_v2": {
        "median_E_roll_mm": 112.47600267594952,
        "roll_gt_40mm_rate": 0.77,
    },
    "yellow_v2": {
        "median_E_roll_mm": 123.35626844455668,
        "roll_gt_40mm_rate": 0.7702777777777777,
    },
}
EXPERT_DEPTH5_LIMIT_MM = 8.0
EXPERT_STOPLINE = 0.10
BASELINE_MEDIAN_ATOL_MM = 1e-6


def _configure_storage() -> None:
    v1._configure_storage()


def _checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = CHECKPOINT_ROOT.resolve()
    if root not in resolved.parents or resolved.name != "weights_final.pt":
        raise ValueError(f"V2 checkpoint must be weights_final.pt below {root}: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _prepare_output(path: Path, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"offline output must be a concrete child of {root}: {resolved}")
    if resolved.is_symlink():
        raise ValueError(f"refusing symlink output: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output exists but is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is nonempty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: v1._jsonable(row[field]) for field in fields})
    os.replace(temporary, path)


def _config_contract(checkpoint: Path) -> dict[str, Any]:
    return v1._config_contract(checkpoint)


def _recompute_expert_stopline(
    base_expert: Mapping[str, Any],
    final_expert: Mapping[str, Any],
    stopline: Mapping[str, Any],
    completed_stopline: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the gate-driving clean-expert regression from paired means."""
    if (
        not base_expert.get("provenance_sha256")
        or base_expert.get("provenance_sha256") != final_expert.get("provenance_sha256")
        or int(base_expert.get("num_batches", -1)) <= 0
        or int(base_expert.get("num_batches", -1)) != int(final_expert.get("num_batches", -2))
    ):
        raise RuntimeError("posthoc expert validation batches are not paired")
    base_pred = float(base_expert.get("mean", {}).get("teacher_pred_loss", np.nan))
    final_pred = float(final_expert.get("mean", {}).get("teacher_pred_loss", np.nan))
    if not np.isfinite([base_pred, final_pred]).all() or base_pred <= 0 or final_pred < 0:
        raise RuntimeError("posthoc expert teacher-pred losses are invalid")
    recomputed_increase = final_pred / base_pred - 1.0
    if (
        float(stopline.get("threshold_relative_increase", np.nan)) != EXPERT_STOPLINE
        or stopline.get("status") != "PASS"
        or stopline.get("offline_gate_authorized") is not True
        or not np.isclose(
            float(stopline.get("base_teacher_pred_loss", np.nan)),
            base_pred,
            atol=0.0,
            rtol=1e-15,
        )
        or not np.isclose(
            float(stopline.get("final_teacher_pred_loss", np.nan)),
            final_pred,
            atol=0.0,
            rtol=1e-15,
        )
        or not np.isclose(
            float(stopline.get("relative_increase", np.nan)),
            recomputed_increase,
            atol=1e-12,
            rtol=1e-12,
        )
        or recomputed_increase > EXPERT_STOPLINE
        or dict(completed_stopline) != dict(stopline)
    ):
        raise RuntimeError(
            "expert stopline is not reproducible/PASS at threshold 0.10: "
            f"recomputed_relative_increase={recomputed_increase}"
        )
    return {
        "base": base_pred,
        "final": final_pred,
        "relative_increase_recomputed": recomputed_increase,
        "threshold": EXPERT_STOPLINE,
        "status": "PASS",
        "offline_gate_authorized": True,
        "paired_batch_provenance_sha256": base_expert["provenance_sha256"],
        "num_batches": int(base_expert["num_batches"]),
    }


def _base_reference_matches(condition: str, median_mm: float, rate: float) -> bool:
    expected = EXPECTED_BASE[condition]
    return bool(
        np.isclose(
            median_mm,
            expected["median_E_roll_mm"],
            atol=BASELINE_MEDIAN_ATOL_MM,
            rtol=0,
        )
        and np.isclose(
            rate,
            expected["roll_gt_40mm_rate"],
            atol=0.0,
            rtol=0.0,
        )
    )


def _training_contract(checkpoint: Path) -> dict[str, Any]:
    """Validate the paired training stopline artifacts for this exact run."""
    checkpoint = _checkpoint(checkpoint)
    run_id = checkpoint.parent.name
    run_root = (TRAIN_OUTPUT_ROOT / run_id).resolve()
    completed_path = run_root / "completed.json"
    posthoc_path = run_root / "posthoc_validation.json"
    for path in (completed_path, posthoc_path):
        if not path.is_file() or path.parent != run_root:
            raise FileNotFoundError(f"same-run V2 training artifact missing: {path}")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
    checkpoint_identity = v1._identity(checkpoint)
    checkpoint_sha = checkpoint_identity["sha256"]
    if (
        completed.get("run_id") != run_id
        or Path(completed.get("final_weights", "")).resolve() != checkpoint
        or completed.get("final_weights_sha256") != checkpoint_sha
    ):
        raise RuntimeError("completed.json does not bind the exact requested V2 checkpoint")
    if (
        posthoc.get("format_version") != "cube_offpolicy_v2_paired_posthoc_v1"
        or posthoc.get("run_id") != run_id
        or posthoc.get("read_only") is not True
        or posthoc.get("protocol", {}).get("clean_expert") is not True
        or posthoc.get("protocol", {}).get("paired_base_final") is not True
    ):
        raise RuntimeError("posthoc_validation.json protocol is not paired clean-expert V2")
    base = posthoc.get("base", {})
    final = posthoc.get("final", {})
    if (
        base.get("checkpoint_sha256") != v1._sha256(BASE_CHECKPOINT)
        or Path(base.get("checkpoint", "")).resolve() != BASE_CHECKPOINT.resolve()
        or final.get("checkpoint_sha256") != checkpoint_sha
        or Path(final.get("checkpoint", "")).resolve() != checkpoint
    ):
        raise RuntimeError("posthoc base/final checkpoints do not match Masked/V2 identities")
    base_expert = base.get("sources", {}).get("expert", {})
    final_expert = final.get("sources", {}).get("expert", {})
    stopline = posthoc.get("expert_stopline", {})
    completed_stopline = completed.get("expert_stopline", {})
    expert_teacher_pred = _recompute_expert_stopline(
        base_expert, final_expert, stopline, completed_stopline
    )
    return {
        "format_version": "cube_offpolicy_v2_training_provenance_v1",
        "run_id": run_id,
        "checkpoint": checkpoint_identity,
        "completed": v1._identity(completed_path),
        "posthoc_validation": v1._identity(posthoc_path),
        "expert_teacher_pred": expert_teacher_pred,
    }


def _freeze_probe_contract(checkpoint: Path, output: Path) -> dict[str, Any]:
    """Bind V2 to the unchanged Masked encoder/projector and Masked XYZ probe."""
    import torch

    config_contract = _config_contract(checkpoint)
    base_state = v1._state_dict(BASE_CHECKPOINT)
    new_state = v1._state_dict(checkpoint)
    prefixes = ("encoder.", "projector.")
    base_keys = sorted(key for key in base_state if key.startswith(prefixes))
    new_keys = sorted(key for key in new_state if key.startswith(prefixes))
    if base_keys != new_keys:
        raise RuntimeError("V2 changed the encoder/projector key set; Masked probe reuse is invalid")
    differences = [
        key
        for key in base_keys
        if not torch.is_tensor(base_state[key])
        or not torch.is_tensor(new_state[key])
        or not torch.equal(base_state[key], new_state[key])
    ]
    if differences:
        raise RuntimeError(
            "V2 changed encoder/projector tensors; fresh probe required; "
            f"first differences={differences[:10]}"
        )
    probe_payload = torch.load(MASKED_PROBE, map_location="cpu", weights_only=False)
    test_median = float(probe_payload["metrics"]["test"]["xyz_error_mm"]["median"])
    if not np.isfinite(test_median) or test_median >= v1.PROBE_TEST_LIMIT_MM:
        raise RuntimeError(
            f"Masked XYZ probe quality failed: expected<15mm, actual={test_median}"
        )
    shared_sha = v1._substate_sha(base_state, prefixes)
    if shared_sha != v1._substate_sha(new_state, prefixes):
        raise RuntimeError("encoder/projector hash mismatch after tensor equality")
    payload = {
        "format_version": "cube_offpolicy_v2_probe_provenance_v1",
        "created_before_outcome_join": True,
        "reuse_reason": (
            "V2 freezes encoder and projector elementwise; one Masked XYZ probe keeps "
            "base/new physical readout paired."
        ),
        "frozen_prefixes": list(prefixes),
        "num_equal_tensors": len(base_keys),
        "shared_encoder_projector_sha256": shared_sha,
        "base_checkpoint": v1._identity(BASE_CHECKPOINT),
        "new_checkpoint": v1._identity(checkpoint),
        "config_contract": config_contract,
        "probe_checkpoint": v1._identity(MASKED_PROBE),
        "probe_original_world_model_state_sha256": probe_payload["world_model_state_sha256"],
        "probe_test_median_xyz_error_mm": test_median,
        "probe_test_limit_mm_strict": v1.PROBE_TEST_LIMIT_MM,
    }
    v1._write_json(output / "probe_provenance.json", payload)
    return payload


def _load_fixed_expert_segments(path: Path = EXPERT_SEGMENTS) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    starts = np.asarray(payload.get("start_rows"), dtype=np.int64)
    episodes = np.asarray(payload.get("episode_indices"), dtype=np.int64)
    formal_episodes = np.asarray(payload.get("formal_episodes_excluded"), dtype=np.int64)
    if (
        payload.get("format_version") != "cube_imagination_error_segments_v1"
        or int(payload.get("seed", -1)) != 42
        or int(payload.get("num_red_segments", -1)) != EXPERT_COUNT
        or starts.shape != (EXPERT_COUNT,)
        or episodes.shape != (EXPERT_COUNT,)
        or len(np.unique(starts)) != EXPERT_COUNT
        or formal_episodes.shape != (50,)
        or np.intersect1d(episodes, formal_episodes).size
    ):
        raise ValueError("fixed Measurement-1 expert segment manifest is malformed/leaky")
    return {
        "start_row": starts,
        "episode_idx": episodes,
        "formal_episodes": formal_episodes,
    }


def _freeze_inputs(checkpoint: Path, output: Path) -> dict[str, Any]:
    """Write all semantic input identities before the first outcome join."""
    import hdf5plugin  # noqa: F401
    import h5py

    training = _training_contract(checkpoint)
    v1._write_json(output / "training_provenance.json", training)
    manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = np.asarray(manifest_payload["formal_rows"], dtype=np.int64)
    if rows.shape != (50,) or len(np.unique(rows)) != 50:
        raise ValueError("formal manifest must contain 50 unique rows")
    segments = _load_fixed_expert_segments()
    with h5py.File(DATASET, "r", swmr=True) as h5:
        current_formal_episodes = np.asarray(h5["ep_idx"][rows], dtype=np.int64)
        current_segment_episodes = np.asarray(
            h5["ep_idx"][segments["start_row"]], dtype=np.int64
        )
    if not np.array_equal(current_formal_episodes, segments["formal_episodes"]):
        raise RuntimeError("expert manifest exclusion set differs from current formal50 episodes")
    if not np.array_equal(current_segment_episodes, segments["episode_idx"]):
        raise RuntimeError("expert segment episodes differ from current H5 dataset")
    if np.intersect1d(current_segment_episodes, current_formal_episodes).size:
        raise RuntimeError("formal50 episode leaked into fixed V2 expert segments")
    audit_inputs: dict[str, Any] = {}
    for condition in CONDITIONS:
        cases = []
        for env_idx in AUDIT_ENVS:
            case = v1._audit_case(condition, env_idx, int(rows[env_idx]))
            files = {
                name: case / name
                for name in ("population.npz", "physical_outcomes.npz", "candidate_outcomes.csv")
            }
            for path in files.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            # Hashing freezes bytes but does not parse/join physical labels.
            cases.append(
                {
                    "env_idx": env_idx,
                    "dataset_row": int(rows[env_idx]),
                    **{name: v1._identity(path) for name, path in files.items()},
                }
            )
        audit_inputs[condition] = cases
    payload = {
        "format_version": "cube_offpolicy_v2_offline_inputs_frozen_v1",
        "created_before_outcome_join": True,
        "created_unix_seconds": time.time(),
        "dataset": v1._identity(DATASET, include_sha256=False),
        "formal_manifest": v1._identity(MANIFEST),
        "memory_index": v1._memory_index_identity(),
        "expert_segment_manifest": v1._identity(EXPERT_SEGMENTS),
        "expert_segment_count": EXPERT_COUNT,
        "expert_formal50_exclusion_verified_against_current_h5": True,
        "expert_segment_start_rows_sha256": hashlib.sha256(
            segments["start_row"].astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "formal_rows": rows,
        "audit_envs": list(AUDIT_ENVS),
        "base_checkpoint": v1._identity(BASE_CHECKPOINT),
        "new_checkpoint": v1._identity(checkpoint),
        "training_provenance": v1._identity(output / "training_provenance.json"),
        "checkpoint_config_contract": _config_contract(checkpoint),
        "masked_probe": v1._identity(MASKED_PROBE),
        "audit_inputs": audit_inputs,
        "stored_latent_cost_allowed": False,
        "outcome_join_allowed_after_this_artifact_and_probe_provenance": True,
    }
    v1._write_json(output / "frozen_inputs.json", payload)
    return payload


def _expert_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "E_roll_mm",
        "E_enc_mm",
        "E_imag_mm",
        "Delta_roll_minus_enc_mm",
        "latent_l2",
        "latent_cosine_distance",
    )
    output: dict[str, Any] = {}
    for model in MODEL_LABELS:
        output[model] = {}
        for depth in EXPERT_DEPTHS:
            subset = [row for row in rows if row["model"] == model and row["depth"] == depth]
            output[model][str(depth)] = {
                "num_segments": len(subset),
                **{
                    metric: v1._distribution([row[metric] for row in subset])
                    for metric in metrics
                },
                "roll_gt_40mm_rate": float(np.mean([row["roll_gt_40mm"] for row in subset])),
            }
    return output


def _measure_expert(
    models: Mapping[str, tuple[Any, Mapping[str, Any]]],
    probe: Any,
    device: str,
    segment_batch_size: int,
    encoder_batch_size: int,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run exact clean Measurement-1 teacher-forcing on the frozen 2,000 rows."""
    import hdf5plugin  # noqa: F401
    import h5py
    import torch

    segments = _load_fixed_expert_segments()
    action_mean, action_scale, scaler_meta = measurement._load_action_normalizer()
    rows: list[dict[str, Any]] = []
    with h5py.File(DATASET, "r", swmr=True) as h5:
        starts_all = segments["start_row"]
        if not np.array_equal(np.asarray(h5["ep_idx"][starts_all]), segments["episode_idx"]):
            raise RuntimeError("fixed expert segments no longer map to frozen episodes")
        for depth in EXPERT_DEPTHS:
            if not np.array_equal(
                np.asarray(h5["ep_idx"][starts_all + depth * imag.ACTION_BLOCK]),
                segments["episode_idx"],
            ):
                raise RuntimeError(f"expert segment crosses episode at depth {depth}")
        for model_label, (model, _) in models.items():
            for first in range(0, EXPERT_COUNT, segment_batch_size):
                segment_ids = np.arange(first, min(first + segment_batch_size, EXPERT_COUNT))
                starts = starts_all[segment_ids]
                pixels, actions, xyz = measurement._segment_batch(
                    h5, starts, action_mean, action_scale
                )
                predicted = measurement._rollout(model, pixels[:, 0], actions, device)[:, 1:]
                actual = imag.encode_uint8(
                    model,
                    pixels[:, 1:].reshape(-1, 224, 224, 3),
                    device,
                    encoder_batch_size,
                ).reshape(len(starts), imag.HORIZON, imag.LATENT_DIM)
                with torch.inference_mode():
                    roll_xyz = probe(predicted).detach().cpu().numpy()
                    enc_xyz = probe(actual).detach().cpu().numpy()
                true_xyz = xyz[:, 1:]
                e_roll = imag.xyz_error_mm(roll_xyz, true_xyz)
                e_enc = imag.xyz_error_mm(enc_xyz, true_xyz)
                e_imag = imag.xyz_error_mm(roll_xyz, enc_xyz)
                drift = imag.latent_metrics(predicted, actual)
                for local, segment_id in enumerate(segment_ids):
                    for depth_index, depth in enumerate(EXPERT_DEPTHS):
                        roll_error = float(e_roll[local, depth_index])
                        enc_error = float(e_enc[local, depth_index])
                        rows.append(
                            {
                                "model": model_label,
                                "segment_id": int(segment_id),
                                "episode_idx": int(segments["episode_idx"][segment_id]),
                                "start_row": int(starts[local]),
                                "target_row": int(starts[local] + depth * imag.ACTION_BLOCK),
                                "depth": depth,
                                "action_teacher_forcing": True,
                                "latent_teacher_forcing": False,
                                "E_roll_mm": roll_error,
                                "E_enc_mm": enc_error,
                                "E_imag_mm": float(e_imag[local, depth_index]),
                                "Delta_roll_minus_enc_mm": roll_error - enc_error,
                                "latent_l2": float(drift["latent_l2"][local, depth_index]),
                                "latent_cosine_distance": float(
                                    drift["latent_cosine_distance"][local, depth_index]
                                ),
                                "roll_gt_40mm": bool(roll_error > 40.0),
                            }
                        )
    expected = len(MODEL_LABELS) * EXPERT_COUNT * len(EXPERT_DEPTHS)
    if len(rows) != expected:
        raise RuntimeError(f"expert row mismatch: expected={expected}, actual={len(rows)}")
    csv_path = output / "expert_measurement.csv"
    _write_csv(csv_path, rows, EXPERT_FIELDS)
    summary = {
        "format_version": "cube_offpolicy_v2_expert_measurement_v1",
        "protocol": {
            "segment_manifest": v1._identity(EXPERT_SEGMENTS),
            "num_segments": EXPERT_COUNT,
            "depths": list(EXPERT_DEPTHS),
            "seed": 42,
            "condition": "clean expert frames/actions",
            "action_teacher_forcing": True,
            "latent_teacher_forcing": False,
            "action_normalizer": scaler_meta,
            "formal50_excluded": True,
        },
        "by_model_depth": _expert_summary(rows),
        "num_csv_rows": len(rows),
        "candidate_outcomes_read": False,
        "csv": v1._identity(csv_path),
    }
    v1._write_json(output / "expert_measurement.json", summary)
    return rows, summary


def _build_gate(
    checkpoint: Path,
    errors: Mapping[str, Any],
    expert: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    training = json.loads((output / "training_provenance.json").read_text(encoding="utf-8"))
    training_gate = training["expert_teacher_pred"]
    training_pass = bool(
        training_gate.get("status") == "PASS"
        and training_gate.get("offline_gate_authorized") is True
        and float(training_gate.get("threshold", np.nan)) == EXPERT_STOPLINE
        and float(training_gate.get("relative_increase_recomputed", np.inf))
        <= EXPERT_STOPLINE
    )
    colors: dict[str, Any] = {}
    for condition in CONDITIONS:
        base = errors["masked_base"][condition]["all"]
        new = errors["offpolicy_new"][condition]["all"]
        base_expected = EXPECTED_BASE[condition]
        base_median = float(base["E_roll_mm"]["median"])
        base_rate = float(base["roll_gt_40mm_rate"])
        base_reference_pass = _base_reference_matches(condition, base_median, base_rate)
        median = float(new["E_roll_mm"]["median"])
        new_rate = float(new["roll_gt_40mm_rate"])
        rate_threshold = RATE_THRESHOLDS[condition]
        median_pass = median < 40.0
        rate_pass = new_rate <= rate_threshold
        colors[condition] = {
            "status": "PASS" if base_reference_pass and median_pass and rate_pass else "FAIL",
            "count": int(new["count"]),
            "base_median_E_roll_mm": base_median,
            "base_roll_gt_40mm_rate": base_rate,
            "expected_base_reference": EXPECTED_BASE[condition],
            "base_reference_median_atol_mm": BASELINE_MEDIAN_ATOL_MM,
            "base_reference_pass": base_reference_pass,
            "new_median_E_roll_mm": median,
            "median_threshold_mm_strict": 40.0,
            "median_pass": median_pass,
            "new_roll_gt_40mm_rate": new_rate,
            "rate_threshold_frozen": rate_threshold,
            "rate_pass": rate_pass,
        }
    depth5 = expert["by_model_depth"]["offpolicy_new"]["5"]
    expert_median = float(depth5["E_roll_mm"]["median"])
    expert_gate = {
        "status": "PASS" if expert_median <= EXPERT_DEPTH5_LIMIT_MM else "FAIL",
        "num_segments": int(depth5["num_segments"]),
        "new_depth5_median_E_roll_mm": expert_median,
        "base_depth5_median_E_roll_mm": float(
            expert["by_model_depth"]["masked_base"]["5"]["E_roll_mm"]["median"]
        ),
        "threshold_mm_inclusive": EXPERT_DEPTH5_LIMIT_MM,
        "pass": expert_median <= EXPERT_DEPTH5_LIMIT_MM,
    }
    status = "PASS" if (
        training_pass
        and all(cell["status"] == "PASS" for cell in colors.values())
        and expert_gate["status"] == "PASS"
    ) else "FAIL"
    artifact_names = (
        "frozen_inputs.json",
        "probe_provenance.json",
        "training_provenance.json",
        "expert_measurement.csv",
        "expert_measurement.json",
        "candidate_scores.csv",
        "summary.json",
    )
    return {
        "format_version": "cube_offpolicy_v2_aggregate_gate_v1",
        "status": status,
        "authorization": (
            "all three colors may enter T2 formal evaluation"
            if status == "PASS"
            else "fail-stop: no formal T2 evaluation is authorized"
        ),
        "all_requirements_AND": True,
        "all_colors_required": True,
        "primary_model": "offpolicy_new",
        "baseline_model": "masked_base",
        "pool": "same old unseeded 3 colors x fixed12 x 300",
        "checkpoint": {
            "weights": v1._identity(checkpoint),
            "config": v1._identity(checkpoint.parent / "config.json"),
            "full_config_semantically_equal_to_masked_base": True,
        },
        "colors": colors,
        "training_expert_stopline": {**training_gate, "pass": training_pass},
        "expert_manifold": expert_gate,
        "artifacts": {name: v1._identity(output / name) for name in artifact_names},
    }


def command_run(args: argparse.Namespace) -> int:
    _configure_storage()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {args.device}")
    checkpoint = _checkpoint(args.checkpoint)
    output = _prepare_output(args.output, args.overwrite)

    # No physical label parser may run before both provenance artifacts exist.
    probe_provenance = _freeze_probe_contract(checkpoint, output)
    frozen_inputs = _freeze_inputs(checkpoint, output)
    base_model, base_meta = v1._load_model(BASE_CHECKPOINT, args.device)
    new_model, new_meta = v1._load_model(checkpoint, args.device)
    probe = imag.LoadedXYZProbe(MASKED_PROBE, args.device)
    if probe.payload["world_model_state_sha256"] != base_meta["world_model_state_sha256"]:
        raise RuntimeError("Masked XYZ probe does not match loaded base checkpoint")
    models = {
        "masked_base": (base_model, base_meta),
        "offpolicy_new": (new_model, new_meta),
    }

    # Expert regression is measured before joining cached candidate outcomes.
    _, expert = _measure_expert(
        models,
        probe,
        args.device,
        args.expert_batch_size,
        args.encoder_batch_size,
        output,
    )

    formal_rows = np.asarray(frozen_inputs["formal_rows"], dtype=np.int64)
    all_rows: list[dict[str, Any]] = []
    started = time.time()
    for condition in CONDITIONS:
        for env_idx in AUDIT_ENVS:
            dataset_row = int(formal_rows[env_idx])
            case = v1._audit_case(condition, env_idx, dataset_row)
            with np.load(case / "population.npz", allow_pickle=False) as loaded:
                candidates = np.asarray(loaded["candidates_normalized"], dtype=np.float32)
                initial = np.asarray(loaded["initial_pixels"], dtype=np.uint8)
                goal_pixels = np.asarray(loaded["goal_pixels"], dtype=np.uint8)
                goal_position = np.asarray(loaded["goal_position"], dtype=np.float64)
            if candidates.shape != (NUM_CANDIDATES, 5, 25):
                raise ValueError(f"candidate pool malformed: {case}/{candidates.shape}")
            truth = v1._load_truth(case)
            for model_label, (model, _) in models.items():
                predicted = probe_common.exact_candidate_terminal_embeddings(
                    model,
                    initial,
                    candidates,
                    args.device,
                    batch_size=args.rollout_batch_size,
                )
                actual = imag.encode_uint8(
                    model,
                    truth["terminal_images"],
                    args.device,
                    args.encoder_batch_size,
                )
                with torch.inference_mode():
                    roll_xyz = probe(predicted).detach().cpu().numpy()
                    enc_xyz = probe(actual).detach().cpu().numpy()
                e_roll = imag.xyz_error_mm(roll_xyz, truth["terminal_xyz"])
                e_enc = imag.xyz_error_mm(enc_xyz, truth["terminal_xyz"])
                e_imag = imag.xyz_error_mm(roll_xyz, enc_xyz)
                drift = imag.latent_metrics(predicted, actual)
                latent_cost = v1._exact_latent_costs(
                    model,
                    initial,
                    goal_pixels,
                    candidates,
                    args.device,
                    args.cost_batch_size,
                )
                probe_cost = np.linalg.norm(
                    np.asarray(roll_xyz, dtype=np.float64) - goal_position[None], axis=1
                )
                latent_rank = v1._stable_rank(latent_cost)
                probe_rank = v1._stable_rank(probe_cost)
                for candidate_idx in range(NUM_CANDIDATES):
                    all_rows.append(
                        {
                            "condition": condition,
                            "env_idx": env_idx,
                            "dataset_row": dataset_row,
                            "candidate_idx": candidate_idx,
                            "model": model_label,
                            "E_roll_mm": float(e_roll[candidate_idx]),
                            "E_enc_mm": float(e_enc[candidate_idx]),
                            "E_imag_mm": float(e_imag[candidate_idx]),
                            "Delta_roll_minus_enc_mm": float(e_roll[candidate_idx] - e_enc[candidate_idx]),
                            "latent_l2": float(drift["latent_l2"][candidate_idx]),
                            "latent_cosine_distance": float(drift["latent_cosine_distance"][candidate_idx]),
                            "roll_gt_40mm": bool(e_roll[candidate_idx] > 40.0),
                            "final_success": bool(truth["final_success"][candidate_idx]),
                            "ever_success": bool(truth["ever_success"][candidate_idx]),
                            "min_goal_distance_m": float(truth["min_goal_distance_m"][candidate_idx]),
                            "final_goal_distance_m": float(truth["final_goal_distance_m"][candidate_idx]),
                            "latent_cost_recomputed": float(latent_cost[candidate_idx]),
                            "probe_cost_m": float(probe_cost[candidate_idx]),
                            "latent_rank": int(latent_rank[candidate_idx]),
                            "probe_rank": int(probe_rank[candidate_idx]),
                        }
                    )
    expected_rows = len(MODEL_LABELS) * len(CONDITIONS) * len(AUDIT_ENVS) * NUM_CANDIDATES
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"offline row mismatch: expected={expected_rows}, actual={len(all_rows)}")
    if v1._identity(checkpoint) != frozen_inputs["new_checkpoint"]:
        raise RuntimeError("V2 checkpoint changed during offline evaluation")
    if v1._identity(checkpoint.parent / "config.json") != frozen_inputs[
        "checkpoint_config_contract"
    ]["new_config"]:
        raise RuntimeError("V2 checkpoint config changed during offline evaluation")
    if v1._identity(MASKED_PROBE) != frozen_inputs["masked_probe"]:
        raise RuntimeError("Masked XYZ probe changed during offline evaluation")
    scores_path = output / "candidate_scores.csv"
    _write_csv(scores_path, all_rows, CSV_FIELDS)
    errors = v1._error_summary(all_rows)
    ranks = v1._rank_summary(all_rows)
    summary = {
        "format_version": "cube_offpolicy_v2_offline_summary_v1",
        "protocol": {
            "pool": "old unseeded cube_cem_300{,_blue_v2,_yellow_v2}",
            "same_actions_and_physical_outcomes_for_both_models": True,
            "candidate_count": expected_rows,
            "stored_official_latent_cost_used": False,
            "latent_cost": "recomputed through each model's real JEPA get_cost path",
            "probe_cost": "terminal predicted XYZ Euclidean distance to numeric goal",
            "stable_sort": "lexicographic (cost, candidate_idx)",
            "primary_success": "final_success",
            "sensitivity_success": "ever_success",
            "outcomes_joined_only_after_freeze": True,
            "candidate_gate_rate_thresholds": RATE_THRESHOLDS,
            "expert_gate": "fixed 2000 clean expert segments; depth5 median E_roll <=8mm",
        },
        "models": {"masked_base": base_meta, "offpolicy_new": new_meta},
        "probe_provenance": probe_provenance,
        "frozen_inputs": v1._identity(output / "frozen_inputs.json"),
        "expert_measurement": v1._identity(output / "expert_measurement.json"),
        "errors": errors,
        "reranking": ranks,
        "elapsed_seconds_candidate_pools": time.time() - started,
        "candidate_scores": v1._identity(scores_path),
    }
    v1._write_json(output / "summary.json", summary)
    gate = _build_gate(checkpoint, errors, expert, output)
    v1._write_json(output / "gate.json", gate)
    print(json.dumps({"status": gate["status"], "output": str(output)}, sort_keys=True))
    return 0


def _read_candidate_core(scores_path: Path, summary_path: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[tuple[int, int, float, bool]]] = {}
    with scores_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("candidate score schema mismatch")
        for record in reader:
            model, condition = record["model"], record["condition"]
            if model not in MODEL_LABELS or condition not in CONDITIONS:
                raise ValueError(f"unexpected candidate cell: {model}/{condition}")
            env_idx, candidate_idx = int(record["env_idx"]), int(record["candidate_idx"])
            error = float(record["E_roll_mm"])
            over = record["roll_gt_40mm"].strip().lower()
            if env_idx not in AUDIT_ENVS or not 0 <= candidate_idx < NUM_CANDIDATES:
                raise ValueError("candidate ID outside frozen pool")
            if not np.isfinite(error) or over not in {"true", "false"}:
                raise ValueError("invalid candidate gate row")
            grouped.setdefault((model, condition), []).append(
                (env_idx, candidate_idx, error, over == "true")
            )
    expected_ids = {
        (env_idx, candidate_idx)
        for env_idx in AUDIT_ENVS
        for candidate_idx in range(NUM_CANDIDATES)
    }
    expected_cells = {(model, condition) for model in MODEL_LABELS for condition in CONDITIONS}
    if set(grouped) != expected_cells:
        raise ValueError("candidate score cells incomplete")
    core: dict[str, Any] = {model: {} for model in MODEL_LABELS}
    for (model, condition), records in grouped.items():
        ids = [(env_idx, candidate_idx) for env_idx, candidate_idx, _, _ in records]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"candidate IDs missing/duplicated: {model}/{condition}")
        values = np.asarray([record[2] for record in records], dtype=np.float64)
        over = np.asarray([record[3] for record in records], dtype=bool)
        if not np.array_equal(over, values > 40.0):
            raise ValueError(f">40 flag mismatch: {model}/{condition}")
        core[model][condition] = {
            "count": len(values),
            "median_E_roll_mm": float(np.median(values)),
            "roll_gt_40mm_rate": float(np.mean(over)),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_total = len(MODEL_LABELS) * len(CONDITIONS) * len(AUDIT_ENVS) * NUM_CANDIDATES
    if (
        summary.get("format_version") != "cube_offpolicy_v2_offline_summary_v1"
        or summary.get("protocol", {}).get("candidate_count") != expected_total
        or summary.get("protocol", {}).get("stored_official_latent_cost_used") is not False
        or summary.get("protocol", {}).get("outcomes_joined_only_after_freeze") is not True
    ):
        raise ValueError("V2 offline summary protocol malformed")
    # Require complete primary/sensitivity x latent/probe reranking tables.
    ranks = summary.get("reranking", {})
    for model in MODEL_LABELS:
        for condition in CONDITIONS:
            cell = ranks.get(model, {}).get(condition, {})
            if set(cell) != {"latent_cost", "probe_cost"}:
                raise ValueError(f"rerank cost tables incomplete: {model}/{condition}")
            for cost in ("latent_cost", "probe_cost"):
                table = cell[cost]
                if len(table.get("per_env", [])) != len(AUDIT_ENVS):
                    raise ValueError(f"rerank per-env table incomplete: {model}/{condition}/{cost}")
                if not {"final_success", "ever_success"}.issubset(table):
                    raise ValueError(f"rerank success semantics incomplete: {model}/{condition}/{cost}")
        for condition in CONDITIONS:
            actual = summary["errors"][model][condition]["all"]
            expected = core[model][condition]
            if (
                int(actual["count"]) != expected["count"]
                or not np.isclose(
                    float(actual["E_roll_mm"]["median"]),
                    expected["median_E_roll_mm"],
                    atol=1e-12,
                    rtol=1e-12,
                )
                or not np.isclose(
                    float(actual["roll_gt_40mm_rate"]),
                    expected["roll_gt_40mm_rate"],
                    atol=1e-15,
                    rtol=0,
                )
            ):
                raise ValueError(f"candidate summary/CSV mismatch: {model}/{condition}")
    return core


def _read_expert_core(csv_path: Path, summary_path: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPERT_FIELDS:
            raise ValueError("expert measurement schema mismatch")
        for record in reader:
            model, depth = record["model"], int(record["depth"])
            segment_id, error = int(record["segment_id"]), float(record["E_roll_mm"])
            if model not in MODEL_LABELS or depth not in EXPERT_DEPTHS:
                raise ValueError("unexpected expert measurement cell")
            if not 0 <= segment_id < EXPERT_COUNT or not np.isfinite(error):
                raise ValueError("invalid expert measurement row")
            if record["action_teacher_forcing"].lower() != "true" or record[
                "latent_teacher_forcing"
            ].lower() != "false":
                raise ValueError("expert teacher-forcing contract changed")
            grouped.setdefault((model, depth), []).append((segment_id, error))
    expected_cells = {(model, depth) for model in MODEL_LABELS for depth in EXPERT_DEPTHS}
    if set(grouped) != expected_cells:
        raise ValueError("expert measurement cells incomplete")
    output: dict[str, Any] = {model: {} for model in MODEL_LABELS}
    expected_ids = set(range(EXPERT_COUNT))
    for (model, depth), records in grouped.items():
        ids = [item[0] for item in records]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"expert segment IDs missing/duplicated: {model}/depth{depth}")
        output[model][str(depth)] = {
            "count": len(records),
            "median_E_roll_mm": float(np.median([item[1] for item in records])),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("format_version") != "cube_offpolicy_v2_expert_measurement_v1"
        or summary.get("protocol", {}).get("num_segments") != EXPERT_COUNT
        or summary.get("protocol", {}).get("formal50_excluded") is not True
        or summary.get("candidate_outcomes_read") is not False
    ):
        raise ValueError("expert summary protocol malformed")
    for model in MODEL_LABELS:
        for depth in EXPERT_DEPTHS:
            actual = summary["by_model_depth"][model][str(depth)]
            expected = output[model][str(depth)]
            if int(actual["num_segments"]) != expected["count"] or not np.isclose(
                float(actual["E_roll_mm"]["median"]),
                expected["median_E_roll_mm"],
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError(f"expert summary/CSV mismatch: {model}/depth{depth}")
    return output


def _validate_gate(
    path: Path,
    checkpoint: Path,
    dataset: Path = DATASET,
    manifest: Path = MANIFEST,
    index: Path = MEMORY_INDEX,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or OFFLINE_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError(f"gate must be below {OFFLINE_ROOT}: {resolved}")
    gate = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        gate.get("format_version") != "cube_offpolicy_v2_aggregate_gate_v1"
        or gate.get("status") != "PASS"
        or gate.get("all_requirements_AND") is not True
        or gate.get("all_colors_required") is not True
    ):
        raise ValueError(f"aggregate V2 gate is not passing: {gate.get('status')}")
    checkpoint = _checkpoint(checkpoint)
    checkpoint_sha = v1._sha256(checkpoint)
    config_contract = _config_contract(checkpoint)
    gate_checkpoint = gate.get("checkpoint", {})
    if (
        gate_checkpoint.get("weights", {}).get("sha256") != checkpoint_sha
        or gate_checkpoint.get("config", {}).get("sha256")
        != config_contract["new_config"]["sha256"]
        or gate_checkpoint.get("full_config_semantically_equal_to_masked_base") is not True
    ):
        raise ValueError("gate checkpoint/config differs from requested V2 model")
    required_artifacts = {
        "frozen_inputs.json",
        "probe_provenance.json",
        "training_provenance.json",
        "expert_measurement.csv",
        "expert_measurement.json",
        "candidate_scores.csv",
        "summary.json",
    }
    artifacts = gate.get("artifacts", {})
    if set(artifacts) != required_artifacts:
        raise ValueError("gate artifact set mismatch")
    for name, identity in artifacts.items():
        artifact = Path(identity["path"]).resolve()
        if artifact.parent != resolved.parent or not artifact.is_file():
            raise ValueError(f"gate artifact missing/outside gate directory: {name}")
        if v1._sha256(artifact) != identity.get("sha256"):
            raise ValueError(f"gate artifact changed after freeze: {name}")
    wrapper = json.loads((resolved.parent / "probe_provenance.json").read_text())
    if (
        wrapper.get("format_version") != "cube_offpolicy_v2_probe_provenance_v1"
        or wrapper.get("created_before_outcome_join") is not True
        or wrapper.get("new_checkpoint", {}).get("sha256") != checkpoint_sha
        or wrapper.get("shared_encoder_projector_sha256") is None
        or float(wrapper.get("probe_test_median_xyz_error_mm", np.inf)) >= 15.0
        or wrapper.get("config_contract", {}).get("canonical_config_sha256")
        != config_contract["canonical_config_sha256"]
    ):
        raise ValueError("V2 probe provenance invalid")
    frozen = json.loads((resolved.parent / "frozen_inputs.json").read_text())
    if (
        frozen.get("format_version") != "cube_offpolicy_v2_offline_inputs_frozen_v1"
        or frozen.get("created_before_outcome_join") is not True
        or frozen.get("new_checkpoint", {}).get("sha256") != checkpoint_sha
        or frozen.get("stored_latent_cost_allowed") is not False
        or int(frozen.get("expert_segment_count", -1)) != EXPERT_COUNT
        or frozen.get("expert_formal50_exclusion_verified_against_current_h5") is not True
        or frozen.get("outcome_join_allowed_after_this_artifact_and_probe_provenance") is not True
    ):
        raise ValueError("V2 frozen inputs invalid")
    training_path = resolved.parent / "training_provenance.json"
    training_frozen = json.loads(training_path.read_text(encoding="utf-8"))
    training_current = _training_contract(checkpoint)
    if training_frozen != training_current:
        raise ValueError("same-run completed/posthoc training provenance changed or is invalid")
    if frozen.get("training_provenance") != v1._identity(training_path):
        raise ValueError("frozen inputs do not bind training_provenance.json")
    expected_training_gate = {
        **training_current["expert_teacher_pred"],
        "pass": True,
    }
    if gate.get("training_expert_stopline") != expected_training_gate:
        raise ValueError("aggregate gate does not bind the reproducible training stopline PASS")
    current_inputs = {
        "dataset": v1._identity(dataset, include_sha256=False),
        "formal_manifest": v1._identity(manifest),
        "memory_index": v1._memory_index_identity(index),
        "expert_segment_manifest": v1._identity(EXPERT_SEGMENTS),
    }
    for key, actual in current_inputs.items():
        if frozen.get(key) != actual:
            raise ValueError(f"gate input identity changed: {key}")
    manifest_payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if frozen.get("formal_rows") != manifest_payload.get("formal_rows"):
        raise ValueError("gate formal rows changed")
    if set(frozen.get("audit_inputs", {})) != set(CONDITIONS):
        raise ValueError("frozen audit input conditions incomplete")
    for condition in CONDITIONS:
        cases = frozen["audit_inputs"][condition]
        if len(cases) != len(AUDIT_ENVS):
            raise ValueError(f"frozen audit cases incomplete: {condition}")
        for expected_env, case in zip(AUDIT_ENVS, cases, strict=True):
            if int(case.get("env_idx", -1)) != expected_env:
                raise ValueError(f"frozen audit env order changed: {condition}")
            for name in ("population.npz", "physical_outcomes.npz", "candidate_outcomes.csv"):
                identity = case.get(name, {})
                if not identity.get("sha256") or not identity.get("path"):
                    raise ValueError(f"frozen audit identity incomplete: {condition}/{name}")
    segments = _load_fixed_expert_segments()
    segment_sha = hashlib.sha256(
        segments["start_row"].astype("<i8", copy=False).tobytes()
    ).hexdigest()
    if frozen.get("expert_segment_start_rows_sha256") != segment_sha:
        raise ValueError("expert segment row list changed")
    declared_dataset = current_inputs["memory_index"].get("dataset_declared", {})
    current_dataset = current_inputs["dataset"]
    if (
        Path(declared_dataset.get("path", "")).resolve()
        != Path(current_dataset["path"]).resolve()
        or int(declared_dataset.get("size_bytes", -1)) != int(current_dataset["size"])
        or int(declared_dataset.get("mtime_ns", -1)) != int(current_dataset["mtime_ns"])
    ):
        raise ValueError("memory index dataset identity differs from formal dataset")

    candidate = _read_candidate_core(
        resolved.parent / "candidate_scores.csv", resolved.parent / "summary.json"
    )
    expert = _read_expert_core(
        resolved.parent / "expert_measurement.csv",
        resolved.parent / "expert_measurement.json",
    )
    if set(gate.get("colors", {})) != set(CONDITIONS):
        raise ValueError("gate does not contain exactly three colors")
    for condition in CONDITIONS:
        base = candidate["masked_base"][condition]
        new = candidate["offpolicy_new"][condition]
        cell = gate["colors"][condition]
        expected_base = EXPECTED_BASE[condition]
        base_reference_pass = _base_reference_matches(
            condition,
            base["median_E_roll_mm"],
            base["roll_gt_40mm_rate"],
        )
        expected = {
            "count": new["count"],
            "base_median_E_roll_mm": base["median_E_roll_mm"],
            "new_median_E_roll_mm": new["median_E_roll_mm"],
            "base_roll_gt_40mm_rate": base["roll_gt_40mm_rate"],
            "new_roll_gt_40mm_rate": new["roll_gt_40mm_rate"],
            "rate_threshold_frozen": RATE_THRESHOLDS[condition],
        }
        for key, value in expected.items():
            actual = cell.get(key)
            equal = int(actual) == int(value) if key == "count" else np.isclose(
                float(actual), float(value), atol=1e-12, rtol=1e-12
            )
            if not equal:
                raise ValueError(f"gate candidate value is not reproducible: {condition}/{key}")
        if (
            cell.get("status") != "PASS"
            or cell.get("base_reference_pass") is not True
            or cell.get("expected_base_reference") != expected_base
            or not np.isclose(
                float(cell.get("base_reference_median_atol_mm", np.nan)),
                BASELINE_MEDIAN_ATOL_MM,
                atol=0.0,
                rtol=0.0,
            )
            or not base_reference_pass
            or cell.get("median_pass") is not True
            or cell.get("rate_pass") is not True
            or not new["median_E_roll_mm"] < 40.0
            or not new["roll_gt_40mm_rate"] <= RATE_THRESHOLDS[condition]
        ):
            raise ValueError(f"candidate color gate is not passing: {condition}")
    expert_cell = gate.get("expert_manifold", {})
    expert_new = expert["offpolicy_new"]["5"]
    expert_base = expert["masked_base"]["5"]
    if (
        expert_cell.get("status") != "PASS"
        or expert_cell.get("pass") is not True
        or int(expert_cell.get("num_segments", -1)) != EXPERT_COUNT
        or not np.isclose(
            float(expert_cell.get("new_depth5_median_E_roll_mm", np.nan)),
            expert_new["median_E_roll_mm"],
            atol=1e-12,
            rtol=1e-12,
        )
        or not np.isclose(
            float(expert_cell.get("base_depth5_median_E_roll_mm", np.nan)),
            expert_base["median_E_roll_mm"],
            atol=1e-12,
            rtol=1e-12,
        )
        or expert_new["median_E_roll_mm"] > EXPERT_DEPTH5_LIMIT_MM
        or float(expert_cell.get("threshold_mm_inclusive", np.nan)) != EXPERT_DEPTH5_LIMIT_MM
    ):
        raise ValueError("expert manifold gate is not passing/reproducible")
    summary = json.loads((resolved.parent / "summary.json").read_text())
    if (
        summary.get("models", {}).get("offpolicy_new", {}).get("checkpoint", {}).get("sha256")
        != checkpoint_sha
        or summary.get("models", {}).get("offpolicy_new", {}).get("config", {}).get("sha256")
        != config_contract["new_config"]["sha256"]
    ):
        raise ValueError("offline summary model identity mismatch")
    return gate


def command_validate_gate(args: argparse.Namespace) -> int:
    checkpoint = _checkpoint(args.checkpoint)
    gate = _validate_gate(args.gate, checkpoint, args.dataset, args.manifest, args.index)
    print(json.dumps({"status": gate["status"], "checkpoint": str(checkpoint)}, sort_keys=True))
    return 0


def command_self_test(_: argparse.Namespace) -> int:
    cost = np.full(NUM_CANDIDATES, 2.0, dtype=np.float64)
    cost[:3], cost[3] = 1.0, 0.0
    rank = v1._stable_rank(cost)
    if rank[:4].tolist() != [2, 3, 4, 1]:
        raise AssertionError("stable rank contract failed")
    synthetic_colors = {
        condition: {
            "median": 39.999,
            "rate": RATE_THRESHOLDS[condition],
            "pass": 39.999 < 40.0 and RATE_THRESHOLDS[condition] <= RATE_THRESHOLDS[condition],
        }
        for condition in CONDITIONS
    }
    if not all(cell["pass"] for cell in synthetic_colors.values()):
        raise AssertionError("inclusive rate/strict median gate failed")
    if 40.0 < 40.0 or not (EXPERT_DEPTH5_LIMIT_MM <= EXPERT_DEPTH5_LIMIT_MM):
        raise AssertionError("boundary semantics failed")
    base_expert = {
        "provenance_sha256": "paired",
        "num_batches": 50,
        "mean": {"teacher_pred_loss": 0.1},
    }
    final_expert = {
        "provenance_sha256": "paired",
        "num_batches": 50,
        "mean": {"teacher_pred_loss": 0.109},
    }
    stopline = {
        "threshold_relative_increase": 0.10,
        "base_teacher_pred_loss": 0.1,
        "final_teacher_pred_loss": 0.109,
        "relative_increase": 0.09,
        "status": "PASS",
        "offline_gate_authorized": True,
    }
    stopline_result = _recompute_expert_stopline(
        base_expert, final_expert, stopline, stopline
    )
    if not np.isclose(stopline_result["relative_increase_recomputed"], 0.09):
        raise AssertionError("positive stopline recomputation failed")
    negative_checks = 0
    for mutation in ("threshold", "authorization", "relative", "increase"):
        bad_stopline = dict(stopline)
        bad_final = {**final_expert, "mean": dict(final_expert["mean"])}
        if mutation == "threshold":
            bad_stopline["threshold_relative_increase"] = 0.11
        elif mutation == "authorization":
            bad_stopline["offline_gate_authorized"] = False
        elif mutation == "relative":
            bad_stopline["relative_increase"] = 0.01
        else:
            bad_final["mean"]["teacher_pred_loss"] = 0.111
        try:
            _recompute_expert_stopline(base_expert, bad_final, bad_stopline, bad_stopline)
        except RuntimeError:
            negative_checks += 1
    if negative_checks != 4:
        raise AssertionError("stopline negative tests did not fail closed")
    if not all(
        _base_reference_matches(
            condition,
            expected["median_E_roll_mm"],
            expected["roll_gt_40mm_rate"],
        )
        and not _base_reference_matches(
            condition,
            expected["median_E_roll_mm"] + 2 * BASELINE_MEDIAN_ATOL_MM,
            expected["roll_gt_40mm_rate"],
        )
        for condition, expected in EXPECTED_BASE.items()
    ):
        raise AssertionError("Masked baseline identity fail-closed test failed")
    print(
        json.dumps(
            {
                "self_test": "PASS",
                "stable_tie_ranks": rank[:4].tolist(),
                "candidate_rate_thresholds": RATE_THRESHOLDS,
                "expert_depth5_limit_mm": EXPERT_DEPTH5_LIMIT_MM,
                "expert_stopline_positive": stopline_result,
                "expert_stopline_negative_checks": negative_checks,
                "masked_baseline_reference_checks": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="score both models and write the V2 all-AND gate")
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--output", type=Path, default=OFFLINE_ROOT)
    run.add_argument("--device", default="cuda")
    run.add_argument("--rollout-batch-size", type=int, default=300)
    run.add_argument("--encoder-batch-size", type=int, default=128)
    run.add_argument("--cost-batch-size", type=int, default=300)
    run.add_argument("--expert-batch-size", type=int, default=64)
    run.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate-gate", help="validate a V2 aggregate PASS artifact")
    validate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    validate.add_argument("--gate", type=Path, default=OFFLINE_ROOT / "gate.json")
    validate.add_argument("--dataset", type=Path, default=DATASET)
    validate.add_argument("--manifest", type=Path, default=MANIFEST)
    validate.add_argument("--index", type=Path, default=MEMORY_INDEX)
    commands.add_parser("self-test", help="run CPU-only synthetic contract checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if min(
            args.rollout_batch_size,
            args.encoder_batch_size,
            args.cost_batch_size,
            args.expert_batch_size,
        ) < 1:
            raise ValueError("batch sizes must be positive")
        return command_run(args)
    if args.command == "validate-gate":
        return command_validate_gate(args)
    return command_self_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
