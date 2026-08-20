#!/usr/bin/env python3
"""Offline evaluation and aggregate gate for Cube off-policy fine-tuning.

The evaluator deliberately reuses the *old unseeded* 12x300 candidate pools
and their cached physical endpoints.  It recomputes both world-model costs for
the MaskedAug base and the new checkpoint; ``stored_latent_cost`` is never
used because it belongs to the older official checkpoint.

The off-policy run freezes encoder+projector.  Their tensor equality is checked
before any outcome file is opened.  This permits a provenance-wrapped reuse of
the MaskedAug XYZ probe while keeping the paired physical readout identical.
Only after checkpoint/probe/input identities have been written are physical
outcomes joined for scoring.
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


DATASET = AILAB_ROOT / "datasets/ogbench/cube_single_expert.h5"
MANIFEST = AILAB_ROOT / "outputs/audit/cube_cem_manifest.json"
BASE_CHECKPOINT = (
    AILAB_ROOT
    / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt"
)
DEFAULT_CHECKPOINT = (
    AILAB_ROOT
    / "checkpoints/lewm-cube-offpolicy_v1/offpolicy_v1_pred_seed3072/weights_final.pt"
)
MASKED_PROBE = AILAB_ROOT / "models/probes/cube_imagination_error_xyz_v1/masked.pt"
MEMORY_INDEX = AILAB_ROOT / "outputs/memory_index/cube_expert_v1"
OUTPUT_ROOT = AILAB_ROOT / "outputs/eval/cube/offpolicy_v1"
OFFLINE_ROOT = OUTPUT_ROOT / "offline"
CHECKPOINT_ROOT = AILAB_ROOT / "checkpoints/lewm-cube-offpolicy_v1"
CONDITIONS = ("red", "blue_v2", "yellow_v2")
AUDIT_ENVS = (0, 1, 2, 6, 7, 11, 12, 23, 26, 37, 38, 49)
AUDIT_ROOTS = {
    "red": AILAB_ROOT / "outputs/audit/cube_cem_300",
    "blue_v2": AILAB_ROOT / "outputs/audit/cube_cem_300_blue_v2",
    "yellow_v2": AILAB_ROOT / "outputs/audit/cube_cem_300_yellow_v2",
}
MODEL_LABELS = ("masked_base", "offpolicy_new")
NUM_CANDIDATES = 300
SUCCESS_K = (1, 3, 5, 10, 30)
PROBE_TEST_LIMIT_MM = 15.0

CSV_FIELDS = (
    "condition",
    "env_idx",
    "dataset_row",
    "candidate_idx",
    "model",
    "E_roll_mm",
    "E_enc_mm",
    "E_imag_mm",
    "Delta_roll_minus_enc_mm",
    "latent_l2",
    "latent_cosine_distance",
    "roll_gt_40mm",
    "final_success",
    "ever_success",
    "min_goal_distance_m",
    "final_goal_distance_m",
    "latent_cost_recomputed",
    "probe_cost_m",
    "latent_rank",
    "probe_rank",
)


def _configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(AILAB_ROOT),
        "HF_HOME": str(AILAB_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(AILAB_ROOT.parent / "tmp"),
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        os.environ[key] = value
    (AILAB_ROOT.parent / "tmp").mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row[field]) for field in CSV_FIELDS})
    os.replace(temporary, path)


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    result = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = _sha256(resolved)
    return result


def _ensure_child(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{label} must be a concrete child of {root}: {resolved}")
    if resolved.is_symlink():
        raise ValueError(f"refusing symlink {label}: {resolved}")
    return resolved


def _checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = CHECKPOINT_ROOT.resolve()
    if root not in resolved.parents or resolved.name != "weights_final.pt":
        raise ValueError(
            f"off-policy checkpoint must be weights_final.pt below {root}: {resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _prepare_output(path: Path, overwrite: bool) -> Path:
    resolved = _ensure_child(path, OUTPUT_ROOT, "offline output")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output exists but is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is nonempty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _state_dict(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"checkpoint has no tensor state_dict: {path}")
    return payload


def _substate_sha(state: Mapping[str, Any], prefixes: tuple[str, ...]) -> str:
    import torch

    digest = hashlib.sha256()
    count = 0
    for name in sorted(key for key in state if key.startswith(prefixes)):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        count += 1
    if not count:
        raise ValueError(f"checkpoint has no tensors for prefixes={prefixes}")
    return digest.hexdigest()


def _config_contract(checkpoint: Path) -> dict[str, Any]:
    """Require an identical LeWM architecture, not merely loadable tensors."""
    base_path = BASE_CHECKPOINT.parent / "config.json"
    new_path = checkpoint.parent / "config.json"
    if not base_path.is_file() or not new_path.is_file():
        raise FileNotFoundError(
            f"checkpoint config missing: base={base_path}, new={new_path}"
        )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    required = {"_target_", "encoder", "predictor", "action_encoder", "projector", "pred_proj"}
    if set(base) != required or set(new) != required:
        raise ValueError(
            "LeWM config schema changed: "
            f"expected={sorted(required)}, base={sorted(base)}, new={sorted(new)}"
        )
    if base != new:
        differing = sorted(key for key in required if base.get(key) != new.get(key))
        raise RuntimeError(
            "off-policy checkpoint architecture differs from MaskedAug base; "
            f"differing sections={differing}"
        )
    return {
        "full_config_semantically_equal": True,
        "required_sections": sorted(required),
        "base_config": _identity(base_path),
        "new_config": _identity(new_path),
        "canonical_config_sha256": hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _memory_index_identity(index: Path = MEMORY_INDEX) -> dict[str, Any]:
    resolved = index.expanduser().resolve()
    metadata_path = resolved / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"memory index metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared = metadata.get("files", {})
    expected_names = {
        "anchor_episodes.npy",
        "anchor_features_z.npy",
        "anchor_rows.npy",
        "stats.npz",
    }
    if set(declared) != expected_names:
        raise ValueError(
            f"memory index file contract mismatch: expected={sorted(expected_names)}, "
            f"actual={sorted(declared)}"
        )
    files = {}
    for name in sorted(expected_names):
        path = resolved / name
        identity = _identity(path)
        expected_sha = declared[name].get("sha256")
        if identity["sha256"] != expected_sha:
            raise ValueError(
                f"memory index hash mismatch: file={path}, "
                f"expected={expected_sha}, actual={identity['sha256']}"
            )
        files[name] = identity
    return {
        "root": str(resolved),
        "metadata": _identity(metadata_path),
        "files": files,
        "format_version": metadata.get("format_version"),
        "num_anchors": metadata.get("num_anchors"),
        "dataset_declared": metadata.get("dataset"),
    }


def _freeze_probe_contract(checkpoint: Path, output: Path) -> dict[str, Any]:
    """Prove the new checkpoint has the same real-frame latent readout."""
    import torch

    config_contract = _config_contract(checkpoint)
    base_state = _state_dict(BASE_CHECKPOINT)
    new_state = _state_dict(checkpoint)
    prefixes = ("encoder.", "projector.")
    base_keys = sorted(key for key in base_state if key.startswith(prefixes))
    new_keys = sorted(key for key in new_state if key.startswith(prefixes))
    if base_keys != new_keys:
        raise RuntimeError(
            "off-policy checkpoint changed encoder/projector key set; the frozen "
            "Masked XYZ probe cannot be reused"
        )
    differences = []
    for key in base_keys:
        left, right = base_state[key], new_state[key]
        if not torch.is_tensor(left) or not torch.is_tensor(right) or not torch.equal(left, right):
            differences.append(key)
            if len(differences) == 10:
                break
    if differences:
        raise RuntimeError(
            "off-policy checkpoint changed encoder/projector tensors; fresh embedding/probe "
            f"training is required. First differences={differences}"
        )
    probe_payload = torch.load(MASKED_PROBE, map_location="cpu", weights_only=False)
    test_median = float(probe_payload["metrics"]["test"]["xyz_error_mm"]["median"])
    if not np.isfinite(test_median) or test_median >= PROBE_TEST_LIMIT_MM:
        raise RuntimeError(
            "Masked XYZ probe quality prerequisite failed: "
            f"expected<15mm, actual={test_median}"
        )
    shared_sha = _substate_sha(base_state, prefixes)
    actual_new_sha = _substate_sha(new_state, prefixes)
    if shared_sha != actual_new_sha:
        raise RuntimeError("encoder/projector substate hash mismatch after tensor equality")
    payload = {
        "format_version": "cube_offpolicy_probe_provenance_v1",
        "created_before_outcome_join": True,
        "reuse_reason": (
            "The off-policy run freezes encoder and projector elementwise; the probe's "
            "input representation is therefore bitwise identical. Reusing one probe also "
            "keeps base/new physical readout paired."
        ),
        "frozen_prefixes": list(prefixes),
        "num_equal_tensors": len(base_keys),
        "shared_encoder_projector_sha256": shared_sha,
        "base_checkpoint": _identity(BASE_CHECKPOINT),
        "new_checkpoint": _identity(checkpoint),
        "config_contract": config_contract,
        "probe_checkpoint": _identity(MASKED_PROBE),
        "probe_original_world_model_state_sha256": probe_payload[
            "world_model_state_sha256"
        ],
        "probe_test_median_xyz_error_mm": test_median,
        "probe_test_limit_mm_strict": PROBE_TEST_LIMIT_MM,
    }
    _write_json(output / "probe_provenance.json", payload)
    return payload


def _audit_case(condition: str, env_idx: int, row: int) -> Path:
    return AUDIT_ROOTS[condition] / f"env_{env_idx:02d}_row_{row}"


def _freeze_inputs(checkpoint: Path, output: Path) -> dict[str, Any]:
    manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = np.asarray(manifest_payload["formal_rows"], dtype=np.int64)
    if rows.shape != (50,) or len(np.unique(rows)) != 50:
        raise ValueError("formal manifest must contain 50 unique rows")
    audit_inputs: dict[str, Any] = {}
    for condition in CONDITIONS:
        cases = []
        for env_idx in AUDIT_ENVS:
            case = _audit_case(condition, env_idx, int(rows[env_idx]))
            population = case / "population.npz"
            physical = case / "physical_outcomes.npz"
            labels = case / "candidate_outcomes.csv"
            for path in (population, physical, labels):
                if not path.is_file():
                    raise FileNotFoundError(path)
            # Population identity is safe to inspect before labels; physical and
            # CSV identities are frozen without opening their contents.
            cases.append(
                {
                    "env_idx": env_idx,
                    "dataset_row": int(rows[env_idx]),
                    "population": _identity(population),
                    "physical_outcomes": _identity(physical),
                    "candidate_outcomes": _identity(labels),
                }
            )
        audit_inputs[condition] = cases
    payload = {
        "format_version": "cube_offpolicy_offline_inputs_frozen_v1",
        "created_before_outcome_join": True,
        "created_unix_seconds": time.time(),
        "dataset": _identity(DATASET, include_sha256=False),
        "dataset_identity_contract": "resolved path + size + mtime_ns; cross-bound to memory metadata",
        "formal_manifest": _identity(MANIFEST),
        "memory_index": _memory_index_identity(),
        "formal_rows": rows,
        "audit_envs": list(AUDIT_ENVS),
        "base_checkpoint": _identity(BASE_CHECKPOINT),
        "new_checkpoint": _identity(checkpoint),
        "checkpoint_config_contract": _config_contract(checkpoint),
        "masked_probe": _identity(MASKED_PROBE),
        "audit_inputs": audit_inputs,
        "stored_latent_cost_allowed": False,
        "outcome_join_allowed_after_this_artifact_and_probe_provenance": True,
    }
    _write_json(output / "frozen_inputs.json", payload)
    return payload


def _load_model(path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(str(path), cache_dir=str(AILAB_ROOT))
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, {
        "checkpoint": _identity(path),
        "config": _identity(path.parent / "config.json"),
        "world_model_state_sha256": probe_common.torch_module_sha256(model),
    }


def _exact_latent_costs(
    model: Any,
    initial_pixels: np.ndarray,
    goal_pixels: np.ndarray,
    candidates: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Call the real JEPA get_cost path, including model-dtype arithmetic."""
    import torch

    model_dtype = next(model.parameters()).dtype
    initial = probe_common.normalized_image_tensor(
        np.asarray(initial_pixels, dtype=np.uint8)[None], device, model_dtype
    )
    goal = probe_common.normalized_image_tensor(
        np.asarray(goal_pixels, dtype=np.uint8)[None], device, model_dtype
    )
    values = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            action = torch.from_numpy(
                np.asarray(candidates[start : start + batch_size], dtype=np.float32)
            ).to(device=device, dtype=model_dtype)
            count = len(action)
            info = {
                "pixels": initial[:, None].expand(1, count, *initial.shape),
                "goal": goal[:, None].expand(1, count, *goal.shape),
                "action": torch.zeros(
                    1, count, 1, 5, device=device, dtype=model_dtype
                ),
            }
            values.append(model.get_cost(info, action[None]).detach().float().cpu().numpy()[0])
    result = np.concatenate(values).astype(np.float64)
    if result.shape != (NUM_CANDIDATES,) or not np.isfinite(result).all():
        raise RuntimeError(f"recomputed latent cost malformed: {result.shape}")
    return result


def _stable_rank(cost: np.ndarray) -> np.ndarray:
    values = np.asarray(cost, dtype=np.float64)
    if values.shape != (NUM_CANDIDATES,) or not np.isfinite(values).all():
        raise ValueError(f"rank cost malformed: {values.shape}")
    order = np.lexsort((np.arange(NUM_CANDIDATES, dtype=np.int64), values))
    rank = np.empty(NUM_CANDIDATES, dtype=np.int64)
    rank[order] = np.arange(1, NUM_CANDIDATES + 1, dtype=np.int64)
    return rank


def _load_truth(case: Path) -> dict[str, np.ndarray]:
    """First actual outcome read; all checkpoint/probe identities are frozen."""
    with np.load(case / "physical_outcomes.npz", allow_pickle=False) as loaded:
        output = {
            "terminal_images": np.asarray(loaded["terminal_images"], dtype=np.uint8),
            "terminal_xyz": np.asarray(loaded["terminal_cube_position"], dtype=np.float64),
            "ever_success": np.asarray(loaded["ever_success"], dtype=bool),
        }
        if "final_success" in loaded.files:
            output["final_success"] = np.asarray(loaded["final_success"], dtype=bool)
        if "min_goal_distance_m" in loaded.files:
            output["min_goal_distance_m"] = np.asarray(
                loaded["min_goal_distance_m"], dtype=np.float64
            )
        if "final_goal_distance_m" in loaded.files:
            output["final_goal_distance_m"] = np.asarray(
                loaded["final_goal_distance_m"], dtype=np.float64
            )
    with (case / "candidate_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != NUM_CANDIDATES or [int(row["candidate_idx"]) for row in records] != list(
        range(NUM_CANDIDATES)
    ):
        raise ValueError(f"candidate outcome IDs malformed: {case}")
    csv_final = np.asarray(
        [row["final_success"].strip().lower() == "true" for row in records], dtype=bool
    )
    csv_ever = np.asarray(
        [row["ever_success"].strip().lower() == "true" for row in records], dtype=bool
    )
    min_distance = np.asarray([float(row["min_goal_distance_m"]) for row in records])
    final_distance = np.asarray([float(row["final_goal_distance_m"]) for row in records])
    if "final_success" in output and not np.array_equal(output["final_success"], csv_final):
        raise ValueError(f"physical/CSV final-success mismatch: {case}")
    if not np.array_equal(output["ever_success"], csv_ever):
        raise ValueError(f"physical/CSV ever-success mismatch: {case}")
    output["final_success"] = csv_final
    for key, csv_value in (
        ("min_goal_distance_m", min_distance),
        ("final_goal_distance_m", final_distance),
    ):
        if key in output and not np.allclose(output[key], csv_value, atol=1e-12, rtol=0):
            raise ValueError(f"physical/CSV {key} mismatch: {case}")
        output[key] = csv_value
    expected_shapes = {
        "terminal_images": (NUM_CANDIDATES, 224, 224, 3),
        "terminal_xyz": (NUM_CANDIDATES, 3),
        "final_success": (NUM_CANDIDATES,),
        "ever_success": (NUM_CANDIDATES,),
        "min_goal_distance_m": (NUM_CANDIDATES,),
        "final_goal_distance_m": (NUM_CANDIDATES,),
    }
    actual = {key: value.shape for key, value in output.items()}
    if actual != expected_shapes:
        raise ValueError(f"physical outcome shape mismatch: expected={expected_shapes}, actual={actual}")
    return output


def _distribution(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "median": None, "mean": None, "p90": None, "p95": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODEL_LABELS:
        output[model] = {}
        model_rows = [row for row in rows if row["model"] == model]
        for condition in CONDITIONS:
            color_rows = [row for row in model_rows if row["condition"] == condition]
            output[model][condition] = {}
            for name, predicate in (
                ("all", lambda row: True),
                ("final_success", lambda row: row["final_success"]),
                ("final_failure", lambda row: not row["final_success"]),
                ("ever_success", lambda row: row["ever_success"]),
                ("ever_failure", lambda row: not row["ever_success"]),
            ):
                subset = [row for row in color_rows if predicate(row)]
                output[model][condition][name] = {
                    "count": len(subset),
                    "E_roll_mm": _distribution([row["E_roll_mm"] for row in subset]),
                    "E_enc_mm": _distribution([row["E_enc_mm"] for row in subset]),
                    "E_imag_mm": _distribution([row["E_imag_mm"] for row in subset]),
                    "roll_gt_40mm_rate": (
                        float(np.mean([row["roll_gt_40mm"] for row in subset]))
                        if subset
                        else None
                    ),
                }
    return output


def _rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODEL_LABELS:
        output[model] = {}
        for condition in CONDITIONS:
            output[model][condition] = {}
            cell = [
                row
                for row in rows
                if row["model"] == model and row["condition"] == condition
            ]
            for cost_name, rank_field in (
                ("latent_cost", "latent_rank"),
                ("probe_cost", "probe_rank"),
            ):
                env_records = []
                for env_idx in AUDIT_ENVS:
                    env = [row for row in cell if row["env_idx"] == env_idx]
                    if len(env) != NUM_CANDIDATES:
                        raise RuntimeError(
                            f"rank summary cell malformed: {model}/{condition}/env{env_idx}"
                        )
                    item: dict[str, Any] = {"env_idx": env_idx}
                    for success_name in ("final_success", "ever_success"):
                        ranks = sorted(
                            int(row[rank_field]) for row in env if row[success_name]
                        )
                        item[success_name] = {
                            "pool_success_count": len(ranks),
                            "first_success_rank": ranks[0] if ranks else None,
                            "reciprocal_first_success_rank": 1.0 / ranks[0] if ranks else 0.0,
                            **{
                                f"success_at_{k}": bool(ranks and ranks[0] <= k)
                                for k in SUCCESS_K
                            },
                        }
                    by_id = {int(row["candidate_idx"]): row for row in env}
                    min_idx = min(
                        by_id, key=lambda idx: (by_id[idx]["min_goal_distance_m"], idx)
                    )
                    final_idx = min(
                        by_id, key=lambda idx: (by_id[idx]["final_goal_distance_m"], idx)
                    )
                    item["physical_min_distance_optimum"] = {
                        "candidate_idx": min_idx,
                        "cost_rank": int(by_id[min_idx][rank_field]),
                        "distance_m": float(by_id[min_idx]["min_goal_distance_m"]),
                    }
                    item["physical_final_distance_optimum"] = {
                        "candidate_idx": final_idx,
                        "cost_rank": int(by_id[final_idx][rank_field]),
                        "distance_m": float(by_id[final_idx]["final_goal_distance_m"]),
                    }
                    env_records.append(item)
                aggregate: dict[str, Any] = {"per_env": env_records}
                for success_name in ("final_success", "ever_success"):
                    records = [item[success_name] for item in env_records]
                    aggregate[success_name] = {
                        "pool_oracle_envs": int(
                            sum(item["pool_success_count"] > 0 for item in records)
                        ),
                        "first_success_rank": _distribution(
                            [
                                item["first_success_rank"]
                                for item in records
                                if item["first_success_rank"] is not None
                            ]
                        ),
                        "mrr_all_12": float(
                            np.mean([item["reciprocal_first_success_rank"] for item in records])
                        ),
                        **{
                            f"success_at_{k}_env_count": int(
                                sum(item[f"success_at_{k}"] for item in records)
                            )
                            for k in SUCCESS_K
                        },
                        "successful_candidate_rank_distribution": _distribution(
                            [
                                row[rank_field]
                                for row in cell
                                if row[success_name]
                            ]
                        ),
                        "failed_candidate_rank_distribution": _distribution(
                            [
                                row[rank_field]
                                for row in cell
                                if not row[success_name]
                            ]
                        ),
                    }
                aggregate["physical_min_distance_optimum_rank"] = _distribution(
                    [item["physical_min_distance_optimum"]["cost_rank"] for item in env_records]
                )
                aggregate["physical_final_distance_optimum_rank"] = _distribution(
                    [item["physical_final_distance_optimum"]["cost_rank"] for item in env_records]
                )
                output[model][condition][cost_name] = aggregate
    return output


def _build_gate(
    checkpoint: Path,
    errors: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    colors = {}
    for condition in CONDITIONS:
        base = errors["masked_base"][condition]["all"]
        new = errors["offpolicy_new"][condition]["all"]
        median = float(new["E_roll_mm"]["median"])
        base_rate = float(base["roll_gt_40mm_rate"])
        new_rate = float(new["roll_gt_40mm_rate"])
        median_pass = median < 40.0
        rate_threshold = 0.5 * base_rate
        rate_pass = new_rate <= rate_threshold
        colors[condition] = {
            "status": "PASS" if median_pass and rate_pass else "FAIL",
            "count": int(new["count"]),
            "base_median_E_roll_mm": float(base["E_roll_mm"]["median"]),
            "new_median_E_roll_mm": median,
            "median_threshold_mm_strict": 40.0,
            "median_pass": median_pass,
            "base_roll_gt_40mm_rate": base_rate,
            "new_roll_gt_40mm_rate": new_rate,
            "rate_threshold_base_half": rate_threshold,
            "rate_pass": rate_pass,
        }
    status = "PASS" if all(item["status"] == "PASS" for item in colors.values()) else "FAIL"
    artifacts = {
        name: _identity(output / name)
        for name in (
            "frozen_inputs.json",
            "probe_provenance.json",
            "candidate_scores.csv",
            "summary.json",
        )
    }
    return {
        "format_version": "cube_offpolicy_aggregate_gate_v1",
        "status": status,
        "authorization": (
            "all three colors may enter T2 formal evaluation"
            if status == "PASS"
            else "fail-stop: no formal T2 evaluation is authorized"
        ),
        "all_colors_required": True,
        "primary_model": "offpolicy_new",
        "baseline_model": "masked_base",
        "pool": "same old unseeded 3 colors x fixed12 x 300",
        "checkpoint": {
            "weights": _identity(checkpoint),
            "config": _identity(checkpoint.parent / "config.json"),
            "full_config_semantically_equal_to_masked_base": True,
        },
        "colors": colors,
        "artifacts": artifacts,
    }


def command_run(args: argparse.Namespace) -> int:
    _configure_storage()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {args.device}")
    checkpoint = _checkpoint(args.checkpoint)
    output = _prepare_output(args.output, args.overwrite)

    # Freeze everything that may influence scoring before the first label read.
    probe_provenance = _freeze_probe_contract(checkpoint, output)
    frozen_inputs = _freeze_inputs(checkpoint, output)
    base_model, base_meta = _load_model(BASE_CHECKPOINT, args.device)
    new_model, new_meta = _load_model(checkpoint, args.device)
    probe = imag.LoadedXYZProbe(MASKED_PROBE, args.device)
    if (
        probe.payload["world_model_state_sha256"]
        != base_meta["world_model_state_sha256"]
    ):
        raise RuntimeError("Masked XYZ probe does not match the loaded base checkpoint")
    models = {
        "masked_base": (base_model, base_meta),
        "offpolicy_new": (new_model, new_meta),
    }
    formal_rows = np.asarray(frozen_inputs["formal_rows"], dtype=np.int64)
    all_rows: list[dict[str, Any]] = []
    started = time.time()
    for condition in CONDITIONS:
        for env_idx in AUDIT_ENVS:
            dataset_row = int(formal_rows[env_idx])
            case = _audit_case(condition, env_idx, dataset_row)
            with np.load(case / "population.npz", allow_pickle=False) as loaded:
                candidates = np.asarray(loaded["candidates_normalized"], dtype=np.float32)
                initial = np.asarray(loaded["initial_pixels"], dtype=np.uint8)
                goal_pixels = np.asarray(loaded["goal_pixels"], dtype=np.uint8)
                goal_position = np.asarray(loaded["goal_position"], dtype=np.float64)
            if candidates.shape != (NUM_CANDIDATES, 5, 25):
                raise ValueError(f"candidate pool malformed: {case}/{candidates.shape}")
            truth = _load_truth(case)
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
                latent_cost = _exact_latent_costs(
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
                latent_rank = _stable_rank(latent_cost)
                probe_rank = _stable_rank(probe_cost)
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
                            "Delta_roll_minus_enc_mm": float(
                                e_roll[candidate_idx] - e_enc[candidate_idx]
                            ),
                            "latent_l2": float(drift["latent_l2"][candidate_idx]),
                            "latent_cosine_distance": float(
                                drift["latent_cosine_distance"][candidate_idx]
                            ),
                            "roll_gt_40mm": bool(e_roll[candidate_idx] > 40.0),
                            "final_success": bool(truth["final_success"][candidate_idx]),
                            "ever_success": bool(truth["ever_success"][candidate_idx]),
                            "min_goal_distance_m": float(
                                truth["min_goal_distance_m"][candidate_idx]
                            ),
                            "final_goal_distance_m": float(
                                truth["final_goal_distance_m"][candidate_idx]
                            ),
                            "latent_cost_recomputed": float(latent_cost[candidate_idx]),
                            "probe_cost_m": float(probe_cost[candidate_idx]),
                            "latent_rank": int(latent_rank[candidate_idx]),
                            "probe_rank": int(probe_rank[candidate_idx]),
                        }
                    )
    expected_rows = len(MODEL_LABELS) * len(CONDITIONS) * len(AUDIT_ENVS) * NUM_CANDIDATES
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"offline row mismatch: expected={expected_rows}, actual={len(all_rows)}")
    scores_path = output / "candidate_scores.csv"
    _write_csv(scores_path, all_rows)
    errors = _error_summary(all_rows)
    ranks = _rank_summary(all_rows)
    summary = {
        "format_version": "cube_offpolicy_offline_summary_v1",
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
        },
        "models": {"masked_base": base_meta, "offpolicy_new": new_meta},
        "probe_provenance": probe_provenance,
        "frozen_inputs": _identity(output / "frozen_inputs.json"),
        "errors": errors,
        "reranking": ranks,
        "elapsed_seconds": time.time() - started,
        "candidate_scores": _identity(scores_path),
    }
    _write_json(output / "summary.json", summary)
    gate = _build_gate(checkpoint, errors, output)
    _write_json(output / "gate.json", gate)
    print(json.dumps({"status": gate["status"], "output": str(output)}, sort_keys=True))
    return 0


def _recompute_gate_core(
    scores_path: Path, summary_path: Path
) -> tuple[dict[str, dict[str, dict[str, float | int]]], dict[str, Any]]:
    """Recompute every gate-driving number from the frozen row-level CSV."""
    with scores_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                f"candidate score schema mismatch: expected={CSV_FIELDS}, actual={reader.fieldnames}"
            )
        grouped: dict[tuple[str, str], list[tuple[int, int, float, bool]]] = {}
        total = 0
        for record in reader:
            model = record["model"]
            condition = record["condition"]
            if model not in MODEL_LABELS or condition not in CONDITIONS:
                raise ValueError(f"unexpected score cell: {model}/{condition}")
            env_idx = int(record["env_idx"])
            candidate_idx = int(record["candidate_idx"])
            error = float(record["E_roll_mm"])
            over = record["roll_gt_40mm"].strip().lower()
            if env_idx not in AUDIT_ENVS or not 0 <= candidate_idx < NUM_CANDIDATES:
                raise ValueError(
                    f"score row ID outside frozen pool: env={env_idx}, candidate={candidate_idx}"
                )
            if not np.isfinite(error) or over not in {"true", "false"}:
                raise ValueError(f"invalid gate-driving score row: {record}")
            grouped.setdefault((model, condition), []).append(
                (env_idx, candidate_idx, error, over == "true")
            )
            total += 1
    expected_total = len(MODEL_LABELS) * len(CONDITIONS) * len(AUDIT_ENVS) * NUM_CANDIDATES
    expected_cells = {(model, condition) for model in MODEL_LABELS for condition in CONDITIONS}
    if total != expected_total or set(grouped) != expected_cells:
        raise ValueError(
            f"candidate score population mismatch: expected_rows={expected_total}, "
            f"actual_rows={total}, cells={sorted(grouped)}"
        )
    core: dict[str, dict[str, dict[str, float | int]]] = {
        model: {} for model in MODEL_LABELS
    }
    expected_ids = {
        (env_idx, candidate_idx)
        for env_idx in AUDIT_ENVS
        for candidate_idx in range(NUM_CANDIDATES)
    }
    for (model, condition), records in grouped.items():
        ids = [(env_idx, candidate_idx) for env_idx, candidate_idx, _, _ in records]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"candidate IDs are missing/duplicated: {model}/{condition}")
        errors = np.asarray([record[2] for record in records], dtype=np.float64)
        over = np.asarray([record[3] for record in records], dtype=bool)
        if not np.array_equal(over, errors > 40.0):
            raise ValueError(f"roll_gt_40 flag disagrees with E_roll: {model}/{condition}")
        core[model][condition] = {
            "count": int(len(errors)),
            "median_E_roll_mm": float(np.median(errors)),
            "roll_gt_40mm_rate": float(np.mean(over)),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("format_version") != "cube_offpolicy_offline_summary_v1"
        or summary.get("protocol", {}).get("candidate_count") != expected_total
        or summary.get("protocol", {}).get("stored_official_latent_cost_used") is not False
        or summary.get("protocol", {}).get("outcomes_joined_only_after_freeze") is not True
    ):
        raise ValueError("offline summary protocol contract is malformed")
    for model in MODEL_LABELS:
        for condition in CONDITIONS:
            actual = summary["errors"][model][condition]["all"]
            expected = core[model][condition]
            comparisons = {
                "count": int(actual["count"]) == expected["count"],
                "median": np.isclose(
                    float(actual["E_roll_mm"]["median"]),
                    expected["median_E_roll_mm"],
                    atol=1e-12,
                    rtol=1e-12,
                ),
                "rate": np.isclose(
                    float(actual["roll_gt_40mm_rate"]),
                    expected["roll_gt_40mm_rate"],
                    atol=1e-15,
                    rtol=0,
                ),
            }
            if not all(comparisons.values()):
                raise ValueError(
                    f"summary/core CSV mismatch: {model}/{condition}/{comparisons}"
                )
    return core, summary


def _validate_gate(
    path: Path,
    checkpoint: Path,
    dataset: Path = DATASET,
    manifest: Path = MANIFEST,
    index: Path = MEMORY_INDEX,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or OFFLINE_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError(f"gate must be a file under {OFFLINE_ROOT}: {resolved}")
    gate = json.loads(resolved.read_text(encoding="utf-8"))
    if gate.get("format_version") != "cube_offpolicy_aggregate_gate_v1":
        raise ValueError(f"unsupported gate format: {gate.get('format_version')}")
    if gate.get("status") != "PASS" or not gate.get("all_colors_required"):
        raise ValueError(f"aggregate gate is not passing: {gate.get('status')}")
    actual_checkpoint_sha = _sha256(checkpoint)
    config_contract = _config_contract(checkpoint)
    gate_checkpoint = gate.get("checkpoint", {})
    if (
        gate_checkpoint.get("weights", {}).get("sha256") != actual_checkpoint_sha
        or gate_checkpoint.get("config", {}).get("sha256")
        != config_contract["new_config"]["sha256"]
        or gate_checkpoint.get("full_config_semantically_equal_to_masked_base") is not True
    ):
        raise ValueError("aggregate gate checkpoint/config does not match requested model")
    if set(gate.get("colors", {})) != set(CONDITIONS):
        raise ValueError("aggregate gate does not contain exactly three colors")
    for condition in CONDITIONS:
        cell = gate["colors"][condition]
        base_rate = float(cell.get("base_roll_gt_40mm_rate", np.nan))
        new_rate = float(cell.get("new_roll_gt_40mm_rate", np.nan))
        threshold = float(cell.get("rate_threshold_base_half", np.nan))
        median = float(cell.get("new_median_E_roll_mm", np.nan))
        if (
            cell.get("status") != "PASS"
            or int(cell.get("count", -1)) != len(AUDIT_ENVS) * NUM_CANDIDATES
            or not cell.get("median_pass")
            or not cell.get("rate_pass")
            or not np.isfinite([base_rate, new_rate, threshold, median]).all()
            or not median < 40.0
            or not np.isclose(threshold, 0.5 * base_rate, atol=0.0, rtol=1e-12)
            or not new_rate <= threshold
        ):
            raise ValueError(f"aggregate gate color is not passing: {condition}/{cell}")
    required_artifacts = {
        "frozen_inputs.json",
        "probe_provenance.json",
        "candidate_scores.csv",
        "summary.json",
    }
    artifacts = gate.get("artifacts", {})
    if set(artifacts) != required_artifacts:
        raise ValueError(
            f"gate artifact set mismatch: expected={sorted(required_artifacts)}, "
            f"actual={sorted(artifacts)}"
        )
    for name, identity in artifacts.items():
        artifact = Path(identity["path"]).resolve()
        if artifact.parent != resolved.parent or not artifact.is_file():
            raise ValueError(f"gate artifact is missing/outside gate directory: {artifact}")
        if _sha256(artifact) != identity["sha256"]:
            raise ValueError(f"gate artifact changed after freeze: {name}")
    wrapper_path = resolved.parent / "probe_provenance.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    if (
        wrapper.get("format_version") != "cube_offpolicy_probe_provenance_v1"
        or not wrapper.get("created_before_outcome_join")
        or float(wrapper.get("probe_test_median_xyz_error_mm", np.inf)) >= 15.0
        or wrapper.get("new_checkpoint", {}).get("sha256") != actual_checkpoint_sha
        or wrapper.get("shared_encoder_projector_sha256") is None
        or wrapper.get("config_contract", {}).get("new_config", {}).get("sha256")
        != config_contract["new_config"]["sha256"]
        or wrapper.get("config_contract", {}).get("canonical_config_sha256")
        != config_contract["canonical_config_sha256"]
        or wrapper.get("config_contract", {}).get("full_config_semantically_equal") is not True
    ):
        raise ValueError(f"probe provenance wrapper is invalid: {wrapper_path}")
    frozen_path = resolved.parent / "frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (
        frozen.get("format_version") != "cube_offpolicy_offline_inputs_frozen_v1"
        or not frozen.get("created_before_outcome_join")
        or frozen.get("new_checkpoint", {}).get("sha256") != actual_checkpoint_sha
        or frozen.get("stored_latent_cost_allowed") is not False
        or frozen.get("checkpoint_config_contract", {}).get("new_config", {}).get("sha256")
        != config_contract["new_config"]["sha256"]
    ):
        raise ValueError(f"frozen input provenance is invalid: {frozen_path}")
    current_dataset = _identity(dataset, include_sha256=False)
    current_manifest = _identity(manifest)
    current_index = _memory_index_identity(index)
    if frozen.get("dataset") != current_dataset:
        raise ValueError(
            f"gate dataset identity changed: expected={frozen.get('dataset')}, "
            f"actual={current_dataset}"
        )
    if frozen.get("formal_manifest") != current_manifest:
        raise ValueError("gate formal manifest path/size/mtime/hash changed")
    if frozen.get("memory_index") != current_index:
        raise ValueError("gate memory index metadata or file identity changed")
    manifest_payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if frozen.get("formal_rows") != manifest_payload.get("formal_rows"):
        raise ValueError("gate formal rows differ from current formal manifest")
    declared_dataset = current_index.get("dataset_declared", {})
    if (
        Path(declared_dataset.get("path", "")).resolve() != Path(current_dataset["path"]).resolve()
        or int(declared_dataset.get("size_bytes", -1)) != int(current_dataset["size"])
        or int(declared_dataset.get("mtime_ns", -1)) != int(current_dataset["mtime_ns"])
    ):
        raise ValueError("memory index dataset identity differs from formal dataset")

    core, summary = _recompute_gate_core(
        resolved.parent / "candidate_scores.csv", resolved.parent / "summary.json"
    )
    if (
        summary.get("models", {}).get("offpolicy_new", {}).get("checkpoint", {}).get("sha256")
        != actual_checkpoint_sha
        or summary.get("models", {}).get("offpolicy_new", {}).get("config", {}).get("sha256")
        != config_contract["new_config"]["sha256"]
    ):
        raise ValueError("offline summary new model/config identity mismatch")
    for condition in CONDITIONS:
        base = core["masked_base"][condition]
        new = core["offpolicy_new"][condition]
        cell = gate["colors"][condition]
        recomputed = {
            "count": new["count"],
            "base_median_E_roll_mm": base["median_E_roll_mm"],
            "new_median_E_roll_mm": new["median_E_roll_mm"],
            "base_roll_gt_40mm_rate": base["roll_gt_40mm_rate"],
            "new_roll_gt_40mm_rate": new["roll_gt_40mm_rate"],
            "rate_threshold_base_half": 0.5 * base["roll_gt_40mm_rate"],
        }
        for key, expected in recomputed.items():
            actual = cell.get(key)
            equal = (
                int(actual) == int(expected)
                if key == "count"
                else np.isclose(float(actual), float(expected), atol=1e-12, rtol=1e-12)
            )
            if not equal:
                raise ValueError(
                    f"gate value not reproduced from candidate_scores.csv: "
                    f"{condition}/{key}, expected={expected}, actual={actual}"
                )
    return gate


def command_validate_gate(args: argparse.Namespace) -> int:
    checkpoint = _checkpoint(args.checkpoint)
    gate = _validate_gate(args.gate, checkpoint, args.dataset, args.manifest, args.index)
    print(json.dumps({"status": gate["status"], "checkpoint": str(checkpoint)}, sort_keys=True))
    return 0


def command_self_test(_: argparse.Namespace) -> int:
    # Pure CPU synthetic checks for stable ties and aggregate rank semantics.
    cost = np.full(NUM_CANDIDATES, 2.0, dtype=np.float64)
    cost[:3] = 1.0
    cost[3] = 0.0
    rank = _stable_rank(cost)
    if rank[:3].tolist() != [2, 3, 4] or int(rank[3]) != 1:
        raise AssertionError(f"stable tie rank failed: {rank[:4].tolist()}")
    values = _distribution([1.0, 3.0, 2.0])
    if values["median"] != 2.0 or values["count"] != 3:
        raise AssertionError(values)
    print(json.dumps({"self_test": "PASS", "stable_tie_ranks": rank[:4].tolist()}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="score both models and write aggregate gate")
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--output", type=Path, default=OFFLINE_ROOT)
    run.add_argument("--device", default="cuda")
    run.add_argument("--rollout-batch-size", type=int, default=300)
    run.add_argument("--encoder-batch-size", type=int, default=128)
    run.add_argument("--cost-batch-size", type=int, default=300)
    run.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate-gate", help="validate aggregate PASS artifact")
    validate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    validate.add_argument("--gate", type=Path, default=OFFLINE_ROOT / "gate.json")
    validate.add_argument("--dataset", type=Path, default=DATASET)
    validate.add_argument("--manifest", type=Path, default=MANIFEST)
    validate.add_argument("--index", type=Path, default=MEMORY_INDEX)
    commands.add_parser("self-test", help="run pure CPU synthetic contract checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if min(args.rollout_batch_size, args.encoder_batch_size, args.cost_batch_size) < 1:
            raise ValueError("batch sizes must be positive")
        return command_run(args)
    if args.command == "validate-gate":
        return command_validate_gate(args)
    return command_self_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
