#!/usr/bin/env python3
"""Fail-stop offline gate for the single Cube play-v1 dynamics arm.

The evaluator consumes no new simulator data.  It binds the unique completed
training run, proves the robust-v1 encoder/projector/pred_proj are bitwise
unchanged, reuses the robust-v1 XYZ probe, and scores the frozen 3 x 12 x 300
candidate pools plus the fixed 2,000 clean expert segments.  A formal online
run is authorized only by an aggregate PASS written by this program.
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
LEWM = HERE.parent
AILAB = LEWM.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cube_imagination_error_common as imag  # noqa: E402
import cube_probe_common as probe_common  # noqa: E402
import evaluate_cube_offpolicy as legacy  # noqa: E402
import run_cube_imagination_error as measurement  # noqa: E402


DATASET = legacy.DATASET
MANIFEST = legacy.MANIFEST
MEMORY_INDEX = legacy.MEMORY_INDEX
ROBUST_CHECKPOINT = (
    AILAB
    / "checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt"
)
ROBUST_CHECKPOINT_SHA256 = (
    "cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed"
)
ROBUST_PROBE = AILAB / "models/probes/cube_robust_v1_xyz/robust_v1.pt"
ROBUST_PROBE_METADATA = AILAB / "outputs/probe/cube_robust_v1/dataset/metadata.json"
PLAY_MANIFEST = AILAB / "datasets/ogbench_play/manifest.json"
EXPERT_SEGMENTS = AILAB / "outputs/eval/cube/imagination_error/measurement1_segments.json"

RUN_ID = "play_v1_dyn_seed3072"
TRAIN_ROOT = AILAB / "outputs/train/play_v1" / RUN_ID
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-play_v1"
DEFAULT_CHECKPOINT = CHECKPOINT_ROOT / RUN_ID / "weights_final.pt"
OUTPUT_ROOT = AILAB / "outputs/eval/cube/play_v1"
OFFLINE_ROOT = OUTPUT_ROOT / "offline"

CONDITIONS = ("red", "blue_v2", "yellow_v2")
AUDIT_ENVS = legacy.AUDIT_ENVS
NUM_CANDIDATES = 300
MODEL_LABELS = ("robust_base", "play_new")
EXPERT_COUNT = 2_000
EXPERT_DEPTHS = tuple(range(1, imag.HORIZON + 1))
EXPERT_DEPTH5_LIMIT_MM = 8.0
STOPLINE_THRESHOLD = 0.10
STOPLINE_INTERVAL = 500
STOPLINE_NUMERIC_ATOL = 1e-12
FROZEN_PREFIXES = ("encoder.", "projector.", "pred_proj.")
FROZEN_MODULES = ("encoder", "projector", "pred_proj")
PLAY_EXCLUSION_ZERO_FIELDS = (
    "exact_episode_hash_overlap_with_expert",
    "exact_episode_hash_overlap_with_formal50",
    "exact_episode_hash_overlap_with_measurement1",
    "quantized_signature_overlap_with_expert",
    "quantized_signature_overlap_with_formal50",
    "quantized_signature_overlap_with_measurement1",
)
PLAY_EXCLUSION_COUNT_CONTRACT = {
    "expert_episode_count": 10_000,
    "formal50_episode_count": 50,
    "measurement1_unique_episode_count": 1_801,
    "play_episode_count": 1_100,
}
PLAY_EXCLUSION_TEXT_FIELDS = (
    "exact_hash_contract",
    "independent_collection_claim",
    "quantized_signature_contract",
)
PLAY_EXCLUSION_KEYS = (
    set(PLAY_EXCLUSION_ZERO_FIELDS)
    | set(PLAY_EXCLUSION_COUNT_CONTRACT)
    | set(PLAY_EXCLUSION_TEXT_FIELDS)
    | {"play_episode_namespace", "quantization"}
)

# Frozen from the Masked baseline on these exact pools; the gate does not infer
# a more convenient threshold from robust-v1 or play-v1 outputs.
MASKED_REFERENCE = {
    "red": {"median_E_roll_mm": 85.72032358672334, "roll_gt_40mm_rate": 0.6244444444444445},
    "blue_v2": {"median_E_roll_mm": 112.47600267594952, "roll_gt_40mm_rate": 0.77},
    "yellow_v2": {"median_E_roll_mm": 123.35626844455668, "roll_gt_40mm_rate": 0.7702777777777777},
}
RATE_THRESHOLDS = {
    condition: value["roll_gt_40mm_rate"] / 2.0
    for condition, value in MASKED_REFERENCE.items()
}

LOSS_CONTRACT: dict[str, Any] = {
    "formula": "0.625*expert_teacher + 0.375*play_teacher + 0.09*shared_target_SIGReg",
    "teacher_forcing_steps": 3,
    "frames": 4,
    "expert_weight": 0.625,
    "play_weight": 0.375,
    "sigreg_weight": 0.09,
    "sigreg_input": "frozen encoder+projector target embeddings, expert+play concatenated",
    "sigreg_gradient_to_trainable_dynamics": False,
    "rollout_enabled": False,
    "rollout_depth": 0,
    "rollout_weight": 0.0,
    "model_rollout_calls": 0,
    "target_encoder_stop_gradient": True,
}

CANDIDATE_FIELDS = (
    "condition", "env_idx", "dataset_row", "candidate_idx", "model",
    "E_roll_mm", "E_enc_mm", "E_imag_mm", "Delta_roll_minus_enc_mm",
    "latent_l2", "latent_cosine_distance", "roll_gt_40mm",
    "final_success", "ever_success", "min_goal_distance_m", "final_goal_distance_m",
)
EXPERT_FIELDS = (
    "model", "segment_id", "episode_idx", "start_row", "target_row", "depth",
    "action_teacher_forcing", "latent_teacher_forcing", "E_roll_mm", "E_enc_mm",
    "E_imag_mm", "Delta_roll_minus_enc_mm", "latent_l2",
    "latent_cosine_distance", "roll_gt_40mm",
)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


LOSS_CONTRACT_SHA256 = _canonical_sha(LOSS_CONTRACT)


def _configure_storage() -> None:
    legacy._configure_storage()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _declares_file(identity: Any, path: Path) -> bool:
    """Accept the two established local identity key spellings."""
    if not isinstance(identity, Mapping):
        return False
    actual = legacy._identity(path)
    declared_size = identity.get("size", identity.get("size_bytes", -1))
    has_mtime = "mtime_ns" in identity
    has_sha256 = "sha256" in identity
    return bool(
        Path(identity.get("path", "")).resolve() == path.resolve()
        and int(declared_size) == actual["size"]
        and (has_mtime or has_sha256)
        and (not has_mtime or int(identity["mtime_ns"]) == actual["mtime_ns"])
        and (not has_sha256 or identity["sha256"] == actual["sha256"])
    )


def _declared_file(identity: Any, label: str) -> dict[str, Any]:
    """Validate a self-contained run-plan identity and return it unchanged."""
    if not isinstance(identity, Mapping) or not identity.get("path"):
        raise RuntimeError(f"{label} is not a file identity")
    path = Path(str(identity["path"])).expanduser().resolve()
    if not _declares_file(identity, path):
        raise RuntimeError(f"{label} changed: {path}")
    return dict(identity)


def _stopline_exceeded(relative: float) -> bool:
    """Implement strict >10% without classifying roundoff at 10% as failure."""
    return bool(
        relative > STOPLINE_THRESHOLD
        and not np.isclose(
            relative,
            STOPLINE_THRESHOLD,
            atol=STOPLINE_NUMERIC_ATOL,
            rtol=0.0,
        )
    )


def _validate_play_exclusion(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("play exclusion contract is not an object")
    if set(value) != PLAY_EXCLUSION_KEYS:
        raise RuntimeError(
            "play exclusion key schema changed: "
            f"expected={sorted(PLAY_EXCLUSION_KEYS)}, actual={sorted(value)}"
        )
    nonzero = {
        name: value.get(name)
        for name in PLAY_EXCLUSION_ZERO_FIELDS
        if value.get(name) != 0
    }
    bad_counts = {
        name: value.get(name)
        for name, expected in PLAY_EXCLUSION_COUNT_CONTRACT.items()
        if value.get(name) != expected
    }
    empty_text = [
        name
        for name in PLAY_EXCLUSION_TEXT_FIELDS
        if not isinstance(value.get(name), str) or not value[name].strip()
    ]
    if (
        nonzero
        or bad_counts
        or empty_text
        or value.get("play_episode_namespace") != "official_play_train_val"
        or not np.isclose(float(value.get("quantization", np.nan)), 0.001, atol=0, rtol=0)
    ):
        raise RuntimeError(
            "play exclusion contract failed: "
            f"nonzero={nonzero}, counts={bad_counts}, empty_text={empty_text}"
        )
    return dict(value)


def _checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != DEFAULT_CHECKPOINT.resolve():
        raise ValueError(
            f"play-v1 has one formal checkpoint: expected={DEFAULT_CHECKPOINT.resolve()}, actual={resolved}"
        )
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
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"non-empty output: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: legacy._jsonable(row[key]) for key in fields})
    os.replace(temporary, path)


def _validate_history(history: Mapping[str, Any], max_steps: int) -> dict[str, Any]:
    expected_keys = {
        "format_version", "run_id", "baseline", "threshold", "interval",
        "comparison", "evaluated_after_optimizer_step", "triggered", "records",
        "resume_events",
    }
    if (
        set(history) != expected_keys
        or history.get("format_version") != "cube_play_v1_stopline_history_v1"
        or history.get("run_id") != RUN_ID
        or float(history.get("threshold", np.nan)) != STOPLINE_THRESHOLD
        or int(history.get("interval", -1)) != STOPLINE_INTERVAL
        or history.get("comparison") != "relative increase strictly > threshold triggers"
        or history.get("evaluated_after_optimizer_step") is not True
        or history.get("triggered") is not False
    ):
        raise RuntimeError("play-v1 stopline top-level PASS contract failed")
    baseline = history.get("baseline", {})
    base_loss = float(baseline.get("teacher_pred_loss", np.nan))
    provenance = baseline.get("provenance")
    expected_provenance = {
        "num_batches": 34,
        "examples": 4_352,
        "batch_size": 128,
        "shuffle": False,
        "drop_last": True,
        "precision": "bf16",
    }
    if (
        set(baseline) != {"step", "teacher_pred_loss", "relative_increase", "status", "provenance"}
        or int(baseline.get("step", -1)) != 0
        or baseline.get("status") != "PASS"
        or float(baseline.get("relative_increase", np.nan)) != 0.0
        or not np.isfinite(base_loss)
        or base_loss <= 0
        or not isinstance(provenance, Mapping)
        or set(provenance) != set(expected_provenance) | {"expert_clip_indices_sha256"}
        or any(provenance.get(key) != value for key, value in expected_provenance.items())
        or len(str(provenance.get("expert_clip_indices_sha256", ""))) != 64
    ):
        raise RuntimeError("play-v1 step-zero expert stopline is malformed")
    records = history.get("records")
    expected_steps = list(range(STOPLINE_INTERVAL, max_steps + 1, STOPLINE_INTERVAL))
    if not isinstance(records, list) or [int(row.get("step", -1)) for row in records] != expected_steps:
        raise RuntimeError("play-v1 stopline history is incomplete")
    normalized = []
    for row, step in zip(records, expected_steps, strict=True):
        current = float(row.get("teacher_pred_loss", np.nan))
        relative = current / base_loss - 1.0
        if (
            set(row) != {"step", "teacher_pred_loss", "relative_increase", "status", "provenance"}
            or int(row.get("step", -1)) != step
            or not np.isfinite(current)
            or current < 0
            or row.get("status") != "PASS"
            or row.get("provenance") != provenance
            or not np.isclose(float(row.get("relative_increase", np.nan)), relative, atol=1e-15, rtol=1e-15)
            or _stopline_exceeded(relative)
        ):
            raise RuntimeError(f"play-v1 stopline record failed at step {step}")
        normalized.append({"step": step, "teacher_pred_loss": current, "relative_increase": relative})
    resume_events = history.get("resume_events")
    if not isinstance(resume_events, list):
        raise RuntimeError("play-v1 resume_events must be a list")
    for index, event in enumerate(resume_events):
        if (
            not isinstance(event, Mapping)
            or set(event) != {"step", "created_at_utc", "checkpoint", "status"}
            or event.get("status") != "CANONICAL_INFRASTRUCTURE_RESUME"
            or int(event.get("step", -1)) not in expected_steps
            or Path(str(event.get("checkpoint", ""))).resolve()
            != (DEFAULT_CHECKPOINT.parent / "lightning/last.ckpt").resolve()
        ):
            raise RuntimeError(f"invalid canonical infrastructure resume event {index}")
    return {
        "baseline_teacher_pred_loss": base_loss,
        "final_teacher_pred_loss": normalized[-1]["teacher_pred_loss"],
        "relative_increase": normalized[-1]["relative_increase"],
        "threshold": STOPLINE_THRESHOLD,
        "record_steps": expected_steps,
        "provenance": provenance,
        "infrastructure_resume_events": resume_events,
        "status": "PASS",
    }


def _training_contract(checkpoint: Path) -> dict[str, Any]:
    checkpoint = _checkpoint(checkpoint)
    paths = {
        name: TRAIN_ROOT / name
        for name in ("run_plan.json", "completed.json", "stopline_history.json", "frozen_integrity.json")
    }
    values = {name: _read_json(path) for name, path in paths.items()}
    plan = values["run_plan.json"]
    completed = values["completed.json"]
    history = values["stopline_history.json"]
    integrity = values["frozen_integrity.json"]
    checkpoint_identity = legacy._identity(checkpoint)
    warm = plan.get("inputs", {}).get("warm_start", {})
    inputs = plan.get("inputs", {})
    optimization = plan.get("optimization", {})
    batch = plan.get("batch", {})
    freeze = plan.get("freeze", {})
    splits = plan.get("splits", {})
    exclusion = plan.get("formal_episode_exclusion", {})
    resume_contract = {
        "infrastructure_only": True,
        "scientific_retry": False,
        "requirements": "canonical run id, byte-identical run_plan, split, data and code",
        "recovery_checkpoint_interval_steps": STOPLINE_INTERVAL,
    }
    play_exclusion = exclusion.get("play_exclusion", {})
    play_exclusion = _validate_play_exclusion(play_exclusion)
    try:
        play_manifest_identity = _declared_file(inputs.get("play_manifest"), "play manifest")
        if Path(play_manifest_identity["path"]).resolve() != PLAY_MANIFEST.resolve():
            raise RuntimeError("play manifest path is noncanonical")
        if set(inputs.get("play_sources", {})) != {"train", "val"}:
            raise RuntimeError("play source split keys changed")
        if set(inputs.get("play_shards", {})) != {"train", "val"}:
            raise RuntimeError("play shard split keys changed")
        if set(inputs.get("play_reports", {})) != {"health", "qc", "validation"}:
            raise RuntimeError("play report keys changed")
        play_sources = {
            key: _declared_file(value, f"play source {key}")
            for key, value in inputs["play_sources"].items()
        }
        play_shards = {
            key: _declared_file(value, f"play shard {key}")
            for key, value in inputs["play_shards"].items()
        }
        play_reports = {
            key: _declared_file(value, f"play report {key}")
            for key, value in inputs["play_reports"].items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("play-v1 run-plan data identities are malformed") from error
    max_steps = int(optimization.get("max_steps", -1))
    if (
        plan.get("format_version") != "cube_play_v1_train_plan_v1"
        or plan.get("run_id") != RUN_ID
        or plan.get("single_arm") is not True
        or plan.get("retry_allowed") is not False
        or plan.get("resume_allowed") != resume_contract
        or plan.get("loss_contract") != LOSS_CONTRACT
        or plan.get("loss_contract_sha256") != LOSS_CONTRACT_SHA256
        or _canonical_sha(plan.get("loss_contract")) != LOSS_CONTRACT_SHA256
        or Path(warm.get("path", "")).resolve() != ROBUST_CHECKPOINT.resolve()
        or warm.get("sha256") != ROBUST_CHECKPOINT_SHA256
        or not _declares_file(inputs.get("expert_dataset"), DATASET)
        or not _declares_file(inputs.get("formal_manifest"), MANIFEST)
        or Path(inputs.get("measurement1_segments", {}).get("path", "")).resolve() != EXPERT_SEGMENTS.resolve()
        or inputs.get("measurement1_segments", {}).get("sha256") != legacy._sha256(EXPERT_SEGMENTS)
        or int(inputs.get("measurement1_segments", {}).get("segments", -1)) != EXPERT_COUNT
        or int(exclusion.get("episode_count", -1)) != 50
        or len(exclusion.get("episode_ids", [])) != 50
        or len(set(map(int, exclusion.get("episode_ids", [])))) != 50
        or exclusion.get("manifest_sha256") != legacy._sha256(MANIFEST)
        or splits.get("unit") != "whole episode"
        or splits.get("expert", {}).get("cross_split_overlap") != 0
        or splits.get("play", {}).get("cross_split_overlap") != 0
        or splits.get("fixed50_and_measurement1_excluded_from_expert") is not True
        or splits.get("play_independent_episode_namespace") is not True
        or batch.get("total") != 128
        or batch.get("expert") != 80
        or batch.get("play") != 48
        or float(batch.get("expert_fraction", np.nan)) != 0.625
        or float(batch.get("play_fraction", np.nan)) != 0.375
        or optimization.get("optimizer") != "AdamW"
        or float(optimization.get("learning_rate", np.nan)) != 1e-5
        or optimization.get("precision") != "bf16-mixed"
        or not 4_000 <= max_steps <= 6_000
        or int(optimization.get("batch_size", -1)) != 128
        or freeze.get("trainable_prefixes") != ["predictor", "action_encoder"]
        or freeze.get("frozen_integrity_modules") != list(FROZEN_MODULES)
    ):
        raise RuntimeError("play-v1 run_plan violates the unique formal arm contract")
    live = _validate_history(history, max_steps)
    expected_source_examples = {"expert": 80 * max_steps, "play": 48 * max_steps}
    if (
        completed.get("format_version") != "cube_play_v1_completed_v1"
        or completed.get("run_id") != RUN_ID
        or completed.get("status") != "PASS"
        or completed.get("offline_gate_authorized") is not True
        or completed.get("retry_allowed") is not False
        or int(completed.get("global_step", -1)) != max_steps
        or not _declares_file(completed.get("final_weights"), checkpoint)
        or completed.get("stopped_weights") is not None
        or completed.get("live_stopline_checkpoint") is not None
        or completed.get("batch") != {"expert": 80, "play": 48, "total": 128}
        or completed.get("source_examples_seen") != expected_source_examples
        or completed.get("source_examples_expected") != expected_source_examples
        or completed.get("source_examples_exact") is not True
        or bool(completed.get("infrastructure_resumed")) != bool(live["infrastructure_resume_events"])
        or completed.get("loss_contract") != LOSS_CONTRACT
        or completed.get("loss_contract_sha256") != LOSS_CONTRACT_SHA256
        or completed.get("stopline", {}).get("triggered") is not False
        or completed.get("stopline", {}).get("event") is not None
        or not np.isclose(float(completed.get("stopline", {}).get("baseline_teacher_pred_loss", np.nan)), live["baseline_teacher_pred_loss"], atol=0, rtol=0)
        or not np.isclose(float(completed.get("stopline", {}).get("final_teacher_pred_loss", np.nan)), live["final_teacher_pred_loss"], atol=0, rtol=0)
        or not np.isclose(float(completed.get("stopline", {}).get("relative_increase", np.nan)), live["relative_increase"], atol=1e-15, rtol=1e-15)
        or not _declares_file(completed.get("stopline", {}).get("history"), paths["stopline_history.json"])
    ):
        raise RuntimeError("completed.json does not authorize the exact play-v1 run")
    before = integrity.get("before")
    after = integrity.get("after")
    if (
        integrity.get("format_version") != "cube_play_v1_frozen_integrity_v1"
        or integrity.get("status") != "PASS"
        or integrity.get("modules") != list(FROZEN_MODULES)
        or integrity.get("exact_match") is not True
        or not isinstance(before, Mapping)
        or set(before) != set(FROZEN_MODULES)
        or before != after
        or completed.get("frozen_integrity") != {"before": before, "after": after, "exact_match": True}
        or freeze.get("frozen_sha256_before") != before
    ):
        raise RuntimeError("encoder/projector/pred_proj frozen-integrity contract failed")
    if (TRAIN_ROOT / "stopline_event.json").exists() or list(checkpoint.parent.glob("weights_stopped_step*.pt")):
        raise RuntimeError("stopline failure artifacts exist beside a claimed PASS")
    return {
        "format_version": "cube_play_v1_training_provenance_v1",
        "run_id": RUN_ID,
        "checkpoint": checkpoint_identity,
        "run_plan": legacy._identity(paths["run_plan.json"]),
        "completed": legacy._identity(paths["completed.json"]),
        "stopline_history": legacy._identity(paths["stopline_history.json"]),
        "frozen_integrity": legacy._identity(paths["frozen_integrity.json"]),
        "warm_start": legacy._identity(ROBUST_CHECKPOINT),
        "loss_contract": {"value": LOSS_CONTRACT, "sha256": LOSS_CONTRACT_SHA256, "rollout_disabled": True},
        "live_stopline": live,
        "resume_contract": resume_contract,
        "play_inputs": {
            "manifest": play_manifest_identity,
            "sources": play_sources,
            "shards": play_shards,
            "reports": play_reports,
            "exclusion": play_exclusion,
        },
        "formal_episode_exclusion": exclusion,
        "max_steps": max_steps,
    }


def _probe_contract(checkpoint: Path, output: Path) -> dict[str, Any]:
    import torch

    base = legacy._state_dict(ROBUST_CHECKPOINT)
    new = legacy._state_dict(checkpoint)
    base_config = _read_json(ROBUST_CHECKPOINT.parent / "config.json")
    new_config = _read_json(checkpoint.parent / "config.json")
    if base_config != new_config:
        raise RuntimeError("play-v1 architecture/config differs from robust-v1")
    keys = sorted(key for key in base if key.startswith(FROZEN_PREFIXES))
    if keys != sorted(key for key in new if key.startswith(FROZEN_PREFIXES)):
        raise RuntimeError("play-v1 changed frozen module key set")
    differences = [key for key in keys if not torch.equal(base[key], new[key])]
    if differences:
        raise RuntimeError(f"play-v1 changed frozen tensors: {differences[:10]}")
    probe_payload = torch.load(ROBUST_PROBE, map_location="cpu", weights_only=False)
    median = float(probe_payload["metrics"]["test"]["xyz_error_mm"]["median"])
    metadata = _read_json(ROBUST_PROBE_METADATA)
    probe_world_state = str(probe_payload.get("world_model_state_sha256", ""))
    metadata_world_state = str(metadata.get("world_model_state_sha256", ""))
    metadata_sha = legacy._sha256(ROBUST_PROBE_METADATA)
    if not np.isfinite(median) or median >= 15.0:
        raise RuntimeError(f"robust XYZ probe failed quality gate: {median}mm")
    if (
        len(probe_world_state) != 64
        or metadata_world_state != probe_world_state
        or probe_payload.get("embedding_dataset_metadata_sha256") != metadata_sha
    ):
        raise RuntimeError("robust probe metadata/world-state provenance mismatch")
    module_hashes = {
        name: legacy._substate_sha(base, (f"{name}.",)) for name in FROZEN_MODULES
    }
    if module_hashes != {
        name: legacy._substate_sha(new, (f"{name}.",)) for name in FROZEN_MODULES
    }:
        raise RuntimeError("frozen substate hashes differ after tensor equality")
    payload = {
        "format_version": "cube_play_v1_probe_provenance_v1",
        "created_before_outcome_join": True,
        "frozen_modules": list(FROZEN_MODULES),
        "equal_tensor_count": len(keys),
        "module_state_sha256": module_hashes,
        "robust_checkpoint": legacy._identity(ROBUST_CHECKPOINT),
        "play_checkpoint": legacy._identity(checkpoint),
        "robust_config": legacy._identity(ROBUST_CHECKPOINT.parent / "config.json"),
        "play_config": legacy._identity(checkpoint.parent / "config.json"),
        "config_semantically_equal": True,
        "probe": legacy._identity(ROBUST_PROBE),
        "probe_dataset_metadata": legacy._identity(ROBUST_PROBE_METADATA),
        "probe_world_model_state_sha256": probe_world_state,
        "metadata_world_model_state_sha256": metadata_world_state,
        "embedding_dataset_metadata_sha256": metadata_sha,
        "probe_test_median_xyz_error_mm": median,
        "probe_test_limit_mm_strict": 15.0,
    }
    legacy._write_json(output / "probe_provenance.json", payload)
    return payload


def _fixed_expert_segments() -> dict[str, np.ndarray]:
    payload = _read_json(EXPERT_SEGMENTS)
    starts = np.asarray(payload.get("start_rows"), dtype=np.int64)
    episodes = np.asarray(payload.get("episode_indices"), dtype=np.int64)
    excluded = np.asarray(payload.get("formal_episodes_excluded"), dtype=np.int64)
    if (
        payload.get("format_version") != "cube_imagination_error_segments_v1"
        or starts.shape != (EXPERT_COUNT,)
        or episodes.shape != (EXPERT_COUNT,)
        or excluded.shape != (50,)
        or len(np.unique(starts)) != EXPERT_COUNT
        or np.intersect1d(episodes, excluded).size
    ):
        raise RuntimeError("fixed Measurement-1 expert segments are malformed/leaky")
    return {"start_row": starts, "episode_idx": episodes, "formal_episodes": excluded}


def _freeze_inputs(checkpoint: Path, output: Path) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401
    import h5py

    training = _training_contract(checkpoint)
    legacy._write_json(output / "training_provenance.json", training)
    manifest = _read_json(MANIFEST)
    formal_rows = np.asarray(manifest.get("formal_rows"), dtype=np.int64)
    if formal_rows.shape != (50,) or len(np.unique(formal_rows)) != 50:
        raise RuntimeError("formal manifest must contain 50 unique rows")
    segments = _fixed_expert_segments()
    with h5py.File(DATASET, "r", swmr=True) as h5:
        formal_episodes = np.asarray(h5["ep_idx"][formal_rows], dtype=np.int64)
        segment_episodes = np.asarray(h5["ep_idx"][segments["start_row"]], dtype=np.int64)
    if not np.array_equal(formal_episodes, segments["formal_episodes"]):
        raise RuntimeError("formal50 exclusion differs from current H5")
    if not np.array_equal(
        formal_episodes,
        np.asarray(training["formal_episode_exclusion"]["episode_ids"], dtype=np.int64),
    ):
        raise RuntimeError("training run-plan formal episode exclusion differs from current H5")
    if not np.array_equal(segment_episodes, segments["episode_idx"]):
        raise RuntimeError("expert segments differ from current H5")
    audits: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        audits[condition] = []
        for env_idx in AUDIT_ENVS:
            case = legacy._audit_case(condition, env_idx, int(formal_rows[env_idx]))
            files = {
                name: case / name
                for name in ("population.npz", "physical_outcomes.npz", "candidate_outcomes.csv")
            }
            audits[condition].append({
                "env_idx": env_idx,
                "dataset_row": int(formal_rows[env_idx]),
                **{name: legacy._identity(path) for name, path in files.items()},
            })
    payload = {
        "format_version": "cube_play_v1_offline_inputs_frozen_v1",
        "created_before_outcome_join": True,
        "created_unix_seconds": time.time(),
        "expert_dataset": legacy._identity(DATASET, include_sha256=False),
        "play_inputs": training["play_inputs"],
        "formal_manifest": legacy._identity(MANIFEST),
        "memory_index": legacy._memory_index_identity(),
        "expert_segments": legacy._identity(EXPERT_SEGMENTS),
        "expert_segment_count": EXPERT_COUNT,
        "formal50_exclusion_verified": True,
        "formal_rows": formal_rows,
        "audit_envs": list(AUDIT_ENVS),
        "audit_inputs": audits,
        "robust_checkpoint": legacy._identity(ROBUST_CHECKPOINT),
        "play_checkpoint": legacy._identity(checkpoint),
        "robust_probe": legacy._identity(ROBUST_PROBE),
        "training_provenance": legacy._identity(output / "training_provenance.json"),
        "outcomes_join_allowed_after_provenance_freeze": True,
    }
    legacy._write_json(output / "frozen_inputs.json", payload)
    return payload


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    return legacy._distribution(values)


def _error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODEL_LABELS:
        output[model] = {}
        for condition in CONDITIONS:
            cell = [row for row in rows if row["model"] == model and row["condition"] == condition]
            output[model][condition] = {}
            for name, predicate in (
                ("all", lambda row: True),
                ("final_success", lambda row: row["final_success"]),
                ("final_failure", lambda row: not row["final_success"]),
                ("ever_success", lambda row: row["ever_success"]),
                ("ever_failure", lambda row: not row["ever_success"]),
            ):
                subset = [row for row in cell if predicate(row)]
                output[model][condition][name] = {
                    "count": len(subset),
                    "E_roll_mm": _distribution([row["E_roll_mm"] for row in subset]),
                    "E_enc_mm": _distribution([row["E_enc_mm"] for row in subset]),
                    "E_imag_mm": _distribution([row["E_imag_mm"] for row in subset]),
                    "roll_gt_40mm_rate": float(np.mean([row["roll_gt_40mm"] for row in subset])) if subset else None,
                }
    return output


def _expert_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODEL_LABELS:
        output[model] = {}
        for depth in EXPERT_DEPTHS:
            cell = [row for row in rows if row["model"] == model and row["depth"] == depth]
            output[model][str(depth)] = {
                "num_segments": len(cell),
                "E_roll_mm": _distribution([row["E_roll_mm"] for row in cell]),
                "E_enc_mm": _distribution([row["E_enc_mm"] for row in cell]),
                "E_imag_mm": _distribution([row["E_imag_mm"] for row in cell]),
                "roll_gt_40mm_rate": float(np.mean([row["roll_gt_40mm"] for row in cell])),
            }
    return output


def _measure_expert(models: Mapping[str, tuple[Any, Mapping[str, Any]]], probe: Any, device: str,
                    segment_batch_size: int, encoder_batch_size: int, output: Path) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401
    import h5py
    import torch

    segments = _fixed_expert_segments()
    action_mean, action_scale, scaler_meta = measurement._load_action_normalizer()
    rows: list[dict[str, Any]] = []
    with h5py.File(DATASET, "r", swmr=True) as h5:
        starts_all = segments["start_row"]
        for depth in EXPERT_DEPTHS:
            if not np.array_equal(np.asarray(h5["ep_idx"][starts_all + depth * imag.ACTION_BLOCK]), segments["episode_idx"]):
                raise RuntimeError(f"expert segment crosses episode at depth {depth}")
        for model_label, (model, _) in models.items():
            for first in range(0, EXPERT_COUNT, segment_batch_size):
                ids = np.arange(first, min(first + segment_batch_size, EXPERT_COUNT))
                starts = starts_all[ids]
                pixels, actions, xyz = measurement._segment_batch(h5, starts, action_mean, action_scale)
                predicted = measurement._rollout(model, pixels[:, 0], actions, device)[:, 1:]
                actual = imag.encode_uint8(model, pixels[:, 1:].reshape(-1, 224, 224, 3), device, encoder_batch_size).reshape(len(starts), imag.HORIZON, imag.LATENT_DIM)
                with torch.inference_mode():
                    roll_xyz = probe(predicted).detach().cpu().numpy()
                    enc_xyz = probe(actual).detach().cpu().numpy()
                e_roll = imag.xyz_error_mm(roll_xyz, xyz[:, 1:])
                e_enc = imag.xyz_error_mm(enc_xyz, xyz[:, 1:])
                e_imag = imag.xyz_error_mm(roll_xyz, enc_xyz)
                drift = imag.latent_metrics(predicted, actual)
                for local, segment_id in enumerate(ids):
                    for depth_index, depth in enumerate(EXPERT_DEPTHS):
                        er = float(e_roll[local, depth_index])
                        ee = float(e_enc[local, depth_index])
                        rows.append({
                            "model": model_label, "segment_id": int(segment_id),
                            "episode_idx": int(segments["episode_idx"][segment_id]),
                            "start_row": int(starts[local]),
                            "target_row": int(starts[local] + depth * imag.ACTION_BLOCK),
                            "depth": depth, "action_teacher_forcing": True,
                            "latent_teacher_forcing": False, "E_roll_mm": er,
                            "E_enc_mm": ee, "E_imag_mm": float(e_imag[local, depth_index]),
                            "Delta_roll_minus_enc_mm": er - ee,
                            "latent_l2": float(drift["latent_l2"][local, depth_index]),
                            "latent_cosine_distance": float(drift["latent_cosine_distance"][local, depth_index]),
                            "roll_gt_40mm": bool(er > 40.0),
                        })
    expected = len(MODEL_LABELS) * EXPERT_COUNT * len(EXPERT_DEPTHS)
    if len(rows) != expected:
        raise RuntimeError(f"expert row count mismatch: {len(rows)} != {expected}")
    csv_path = output / "expert_measurement.csv"
    _write_csv(csv_path, rows, EXPERT_FIELDS)
    payload = {
        "format_version": "cube_play_v1_expert_measurement_v1",
        "protocol": {"segments": EXPERT_COUNT, "depths": list(EXPERT_DEPTHS), "seed": 42,
                     "action_teacher_forcing": True, "latent_teacher_forcing": False,
                     "formal50_excluded": True, "action_normalizer": scaler_meta},
        "by_model_depth": _expert_summary(rows),
        "num_csv_rows": len(rows),
        "candidate_outcomes_read": False,
        "csv": legacy._identity(csv_path),
    }
    legacy._write_json(output / "expert_measurement.json", payload)
    return payload


def _build_gate(checkpoint: Path, errors: Mapping[str, Any], expert: Mapping[str, Any], output: Path) -> dict[str, Any]:
    training = _read_json(output / "training_provenance.json")
    colors: dict[str, Any] = {}
    for condition in CONDITIONS:
        base = errors["robust_base"][condition]["all"]
        new = errors["play_new"][condition]["all"]
        median = float(new["E_roll_mm"]["median"])
        rate = float(new["roll_gt_40mm_rate"])
        median_pass = median < 40.0
        rate_pass = rate <= RATE_THRESHOLDS[condition]
        colors[condition] = {
            "status": "PASS" if median_pass and rate_pass else "FAIL",
            "count": int(new["count"]),
            "masked_reference": MASKED_REFERENCE[condition],
            "robust_base_median_E_roll_mm": float(base["E_roll_mm"]["median"]),
            "robust_base_roll_gt_40mm_rate": float(base["roll_gt_40mm_rate"]),
            "play_new_median_E_roll_mm": median,
            "median_threshold_mm_strict": 40.0,
            "median_pass": median_pass,
            "play_new_roll_gt_40mm_rate": rate,
            "rate_threshold_masked_half_inclusive": RATE_THRESHOLDS[condition],
            "rate_pass": rate_pass,
            "final_success_stratum": errors["play_new"][condition]["final_success"],
            "final_failure_stratum": errors["play_new"][condition]["final_failure"],
        }
    base5 = expert["by_model_depth"]["robust_base"]["5"]
    new5 = expert["by_model_depth"]["play_new"]["5"]
    expert_median = float(new5["E_roll_mm"]["median"])
    expert_gate = {
        "status": "PASS" if expert_median <= EXPERT_DEPTH5_LIMIT_MM else "FAIL",
        "num_segments": int(new5["num_segments"]),
        "robust_base_depth5_median_E_roll_mm": float(base5["E_roll_mm"]["median"]),
        "play_new_depth5_median_E_roll_mm": expert_median,
        "threshold_mm_inclusive": EXPERT_DEPTH5_LIMIT_MM,
        "pass": expert_median <= EXPERT_DEPTH5_LIMIT_MM,
    }
    stopline = training["live_stopline"]
    training_pass = (
        stopline["status"] == "PASS"
        and not _stopline_exceeded(float(stopline["relative_increase"]))
        and training["loss_contract"]["sha256"] == LOSS_CONTRACT_SHA256
        and training["loss_contract"]["rollout_disabled"] is True
    )
    status = "PASS" if training_pass and expert_gate["pass"] and all(cell["status"] == "PASS" for cell in colors.values()) else "FAIL"
    artifact_names = (
        "training_provenance.json", "frozen_inputs.json", "probe_provenance.json",
        "expert_measurement.csv", "expert_measurement.json", "candidate_scores.csv", "summary.json",
    )
    return {
        "format_version": "cube_play_v1_aggregate_gate_v1",
        "status": status,
        "authorization": "formal online evaluation authorized" if status == "PASS" else "fail-stop: no online evaluation authorized",
        "all_requirements_AND": True,
        "all_colors_required": True,
        "checkpoint": legacy._identity(checkpoint),
        "training_stopline": {**stopline, "pass": training_pass},
        "colors": colors,
        "expert_manifold": expert_gate,
        "artifacts": {name: legacy._identity(output / name) for name in artifact_names},
    }


def _gate_exit_code(status: str) -> int:
    if status not in {"PASS", "FAIL"}:
        raise ValueError(status)
    return 0 if status == "PASS" else 3


def command_run(args: argparse.Namespace) -> int:
    _configure_storage()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    checkpoint = _checkpoint(args.checkpoint)
    output = _prepare_output(args.output, args.overwrite)
    probe_provenance = _probe_contract(checkpoint, output)
    frozen = _freeze_inputs(checkpoint, output)
    base_model, base_meta = legacy._load_model(ROBUST_CHECKPOINT, args.device)
    new_model, new_meta = legacy._load_model(checkpoint, args.device)
    probe = imag.LoadedXYZProbe(ROBUST_PROBE, args.device)
    if probe.payload["world_model_state_sha256"] != base_meta["world_model_state_sha256"]:
        raise RuntimeError("robust probe/model state mismatch")
    models = {"robust_base": (base_model, base_meta), "play_new": (new_model, new_meta)}
    expert = _measure_expert(models, probe, args.device, args.expert_batch_size, args.encoder_batch_size, output)
    formal_rows = np.asarray(frozen["formal_rows"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for condition in CONDITIONS:
        for env_idx in AUDIT_ENVS:
            dataset_row = int(formal_rows[env_idx])
            case = legacy._audit_case(condition, env_idx, dataset_row)
            with np.load(case / "population.npz", allow_pickle=False) as loaded:
                candidates = np.asarray(loaded["candidates_normalized"], dtype=np.float32)
                initial = np.asarray(loaded["initial_pixels"], dtype=np.uint8)
            if candidates.shape != (NUM_CANDIDATES, 5, 25):
                raise RuntimeError(f"candidate pool malformed: {case}")
            truth = legacy._load_truth(case)
            for model_label, (model, _) in models.items():
                predicted = probe_common.exact_candidate_terminal_embeddings(model, initial, candidates, args.device, batch_size=args.rollout_batch_size)
                actual = imag.encode_uint8(model, truth["terminal_images"], args.device, args.encoder_batch_size)
                with torch.inference_mode():
                    roll_xyz = probe(predicted).detach().cpu().numpy()
                    enc_xyz = probe(actual).detach().cpu().numpy()
                e_roll = imag.xyz_error_mm(roll_xyz, truth["terminal_xyz"])
                e_enc = imag.xyz_error_mm(enc_xyz, truth["terminal_xyz"])
                e_imag = imag.xyz_error_mm(roll_xyz, enc_xyz)
                drift = imag.latent_metrics(predicted, actual)
                for idx in range(NUM_CANDIDATES):
                    rows.append({
                        "condition": condition, "env_idx": env_idx, "dataset_row": dataset_row,
                        "candidate_idx": idx, "model": model_label,
                        "E_roll_mm": float(e_roll[idx]), "E_enc_mm": float(e_enc[idx]),
                        "E_imag_mm": float(e_imag[idx]),
                        "Delta_roll_minus_enc_mm": float(e_roll[idx] - e_enc[idx]),
                        "latent_l2": float(drift["latent_l2"][idx]),
                        "latent_cosine_distance": float(drift["latent_cosine_distance"][idx]),
                        "roll_gt_40mm": bool(e_roll[idx] > 40.0),
                        "final_success": bool(truth["final_success"][idx]),
                        "ever_success": bool(truth["ever_success"][idx]),
                        "min_goal_distance_m": float(truth["min_goal_distance_m"][idx]),
                        "final_goal_distance_m": float(truth["final_goal_distance_m"][idx]),
                    })
    expected = len(MODEL_LABELS) * len(CONDITIONS) * len(AUDIT_ENVS) * NUM_CANDIDATES
    if len(rows) != expected:
        raise RuntimeError(f"candidate row count mismatch: {len(rows)} != {expected}")
    if legacy._identity(checkpoint) != frozen["play_checkpoint"] or legacy._identity(ROBUST_PROBE) != frozen["robust_probe"]:
        raise RuntimeError("checkpoint/probe changed during offline scoring")
    scores = output / "candidate_scores.csv"
    _write_csv(scores, rows, CANDIDATE_FIELDS)
    errors = _error_summary(rows)
    summary = {
        "format_version": "cube_play_v1_offline_summary_v1",
        "protocol": {"pool": "same cached red/blue_v2/yellow_v2 x 12 x 300", "new_simulation": False,
                     "same_actions_and_physical_outcomes_for_both_models": True,
                     "masked_rate_reference_frozen": MASKED_REFERENCE,
                     "candidate_success_failure_strata_reported": True,
                     "expert_measurement": "fixed Measurement-1 2000 clean segments, depth1..5"},
        "models": {"robust_base": base_meta, "play_new": new_meta},
        "probe_provenance": probe_provenance,
        "errors": errors,
        "expert": expert,
        "elapsed_seconds_candidate_pools": time.time() - started,
        "candidate_scores": legacy._identity(scores),
    }
    legacy._write_json(output / "summary.json", summary)
    gate = _build_gate(checkpoint, errors, expert, output)
    legacy._write_json(output / "gate.json", gate)
    print(json.dumps({"status": gate["status"], "gate": str(output / "gate.json")}, sort_keys=True))
    return _gate_exit_code(gate["status"])


def _csv_core(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANDIDATE_FIELDS:
            raise RuntimeError("candidate CSV schema changed")
        rows = list(reader)
    parsed = []
    keys = set()
    for row in rows:
        if any(row[name].lower() not in {"true", "false"} for name in ("roll_gt_40mm", "final_success", "ever_success")):
            raise RuntimeError("candidate CSV contains a non-boolean token")
        item = {
            "condition": row["condition"], "model": row["model"],
            "env_idx": int(row["env_idx"]), "dataset_row": int(row["dataset_row"]),
            "candidate_idx": int(row["candidate_idx"]),
            "E_roll_mm": float(row["E_roll_mm"]),
            "E_enc_mm": float(row["E_enc_mm"]),
            "E_imag_mm": float(row["E_imag_mm"]),
            "roll_gt_40mm": row["roll_gt_40mm"].lower() == "true",
            "final_success": row["final_success"].lower() == "true",
            "ever_success": row["ever_success"].lower() == "true",
        }
        if not np.isfinite([item["E_roll_mm"], item["E_enc_mm"], item["E_imag_mm"]]).all():
            raise RuntimeError("candidate CSV contains non-finite metrics")
        key = (item["model"], item["condition"], item["env_idx"], item["candidate_idx"])
        if key in keys:
            raise RuntimeError(f"duplicate candidate CSV key: {key}")
        keys.add(key)
        parsed.append(item)
    expected_keys = _candidate_expected_keys()
    _require_exact_coverage("candidate CSV", keys, expected_keys)
    return {"rows": parsed, "errors": _error_summary(parsed)}


def _candidate_expected_keys() -> set[tuple[str, str, int, int]]:
    return {
        (model, condition, env_idx, candidate_idx)
        for model in MODEL_LABELS
        for condition in CONDITIONS
        for env_idx in AUDIT_ENVS
        for candidate_idx in range(NUM_CANDIDATES)
    }


def _expert_expected_keys() -> set[tuple[str, int, int]]:
    return {
        (model, depth, segment_id)
        for model in MODEL_LABELS
        for depth in EXPERT_DEPTHS
        for segment_id in range(EXPERT_COUNT)
    }


def _require_exact_coverage(label: str, actual: set[Any], expected: set[Any]) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{label} coverage changed: expected={len(expected)}, actual={len(actual)}"
        )


def _expert_csv_core(csv_path: Path, json_path: Path) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPERT_FIELDS:
            raise RuntimeError("expert CSV schema changed")
        raw_rows = list(reader)
    parsed = []
    keys = set()
    for row in raw_rows:
        if row["action_teacher_forcing"].lower() != "true" or row["latent_teacher_forcing"].lower() != "false":
            raise RuntimeError("expert teacher-forcing contract changed")
        if row["roll_gt_40mm"].lower() not in {"true", "false"}:
            raise RuntimeError("expert CSV contains a non-boolean token")
        item = {
            "model": row["model"], "segment_id": int(row["segment_id"]),
            "depth": int(row["depth"]), "E_roll_mm": float(row["E_roll_mm"]),
            "E_enc_mm": float(row["E_enc_mm"]), "E_imag_mm": float(row["E_imag_mm"]),
            "roll_gt_40mm": row["roll_gt_40mm"].lower() == "true",
        }
        if not np.isfinite([item["E_roll_mm"], item["E_enc_mm"], item["E_imag_mm"]]).all():
            raise RuntimeError("expert CSV contains non-finite metrics")
        key = (item["model"], item["depth"], item["segment_id"])
        if key in keys:
            raise RuntimeError(f"duplicate expert CSV key: {key}")
        keys.add(key)
        parsed.append(item)
    expected_keys = _expert_expected_keys()
    _require_exact_coverage("expert CSV", keys, expected_keys)
    summary = _expert_summary(parsed)
    declared = _read_json(json_path)
    if (
        declared.get("format_version") != "cube_play_v1_expert_measurement_v1"
        or int(declared.get("num_csv_rows", -1)) != len(expected_keys)
        or declared.get("candidate_outcomes_read") is not False
        or declared.get("csv") != legacy._identity(csv_path)
        or declared.get("by_model_depth") != summary
        or declared.get("protocol", {}).get("segments") != EXPERT_COUNT
        or declared.get("protocol", {}).get("depths") != list(EXPERT_DEPTHS)
        or declared.get("protocol", {}).get("formal50_excluded") is not True
    ):
        raise RuntimeError("expert CSV/JSON summaries are not reproducible")
    return {"rows": parsed, "by_model_depth": summary, "declared": declared}


def _validate_gate(path: Path, checkpoint: Path) -> dict[str, Any]:
    gate_path = path.expanduser().resolve()
    if gate_path != (OFFLINE_ROOT / "gate.json").resolve():
        raise RuntimeError(f"play-v1 gate path is noncanonical: {gate_path}")
    gate = _read_json(gate_path)
    checkpoint = _checkpoint(checkpoint)
    if (
        gate.get("format_version") != "cube_play_v1_aggregate_gate_v1"
        or gate.get("status") != "PASS"
        or gate.get("all_requirements_AND") is not True
        or gate.get("all_colors_required") is not True
        or gate.get("checkpoint") != legacy._identity(checkpoint)
    ):
        raise RuntimeError("aggregate play-v1 gate is not a PASS for this checkpoint")
    current_training = _training_contract(checkpoint)
    artifacts = gate.get("artifacts", {})
    required = {"training_provenance.json", "frozen_inputs.json", "probe_provenance.json",
                "expert_measurement.csv", "expert_measurement.json", "candidate_scores.csv", "summary.json"}
    if set(artifacts) != required:
        raise RuntimeError("aggregate gate artifact set changed")
    for name, identity in artifacts.items():
        target = gate_path.parent / name
        if target.parent != gate_path.parent or identity != legacy._identity(target):
            raise RuntimeError(f"offline artifact changed: {name}")
    if _read_json(gate_path.parent / "training_provenance.json") != current_training:
        raise RuntimeError("training provenance changed after gate")
    frozen = _read_json(gate_path.parent / "frozen_inputs.json")
    formal_rows = np.asarray(frozen.get("formal_rows"), dtype=np.int64)
    if (
        frozen.get("format_version") != "cube_play_v1_offline_inputs_frozen_v1"
        or frozen.get("created_before_outcome_join") is not True
        or frozen.get("formal50_exclusion_verified") is not True
        or frozen.get("outcomes_join_allowed_after_provenance_freeze") is not True
        or frozen.get("audit_envs") != list(AUDIT_ENVS)
        or formal_rows.shape != (50,)
        or frozen.get("expert_dataset") != legacy._identity(DATASET, include_sha256=False)
        or frozen.get("formal_manifest") != legacy._identity(MANIFEST)
        or frozen.get("memory_index") != legacy._memory_index_identity()
        or frozen.get("expert_segments") != legacy._identity(EXPERT_SEGMENTS)
        or frozen.get("play_checkpoint") != legacy._identity(checkpoint)
        or frozen.get("robust_probe") != legacy._identity(ROBUST_PROBE)
        or frozen.get("play_inputs") != current_training["play_inputs"]
    ):
        raise RuntimeError("frozen offline input contract changed")
    if set(frozen.get("audit_inputs", {})) != set(CONDITIONS):
        raise RuntimeError("frozen audit color set changed")
    for condition in CONDITIONS:
        cases = frozen["audit_inputs"][condition]
        if len(cases) != len(AUDIT_ENVS):
            raise RuntimeError(f"frozen audit case count changed: {condition}")
        for env_idx, case in zip(AUDIT_ENVS, cases, strict=True):
            dataset_row = int(formal_rows[env_idx])
            case_root = legacy._audit_case(condition, env_idx, dataset_row)
            if (
                int(case.get("env_idx", -1)) != env_idx
                or int(case.get("dataset_row", -1)) != dataset_row
                or any(
                    case.get(name) != legacy._identity(case_root / name)
                    for name in ("population.npz", "physical_outcomes.npz", "candidate_outcomes.csv")
                )
            ):
                raise RuntimeError(f"frozen audit identity changed: {condition}/env{env_idx}")
    probe_provenance = _read_json(gate_path.parent / "probe_provenance.json")
    if (
        probe_provenance.get("format_version") != "cube_play_v1_probe_provenance_v1"
        or probe_provenance.get("created_before_outcome_join") is not True
        or probe_provenance.get("frozen_modules") != list(FROZEN_MODULES)
        or probe_provenance.get("robust_checkpoint") != legacy._identity(ROBUST_CHECKPOINT)
        or probe_provenance.get("play_checkpoint") != legacy._identity(checkpoint)
        or probe_provenance.get("probe") != legacy._identity(ROBUST_PROBE)
        or probe_provenance.get("probe_dataset_metadata") != legacy._identity(ROBUST_PROBE_METADATA)
        or probe_provenance.get("probe_world_model_state_sha256")
        != probe_provenance.get("metadata_world_model_state_sha256")
        or probe_provenance.get("embedding_dataset_metadata_sha256")
        != legacy._sha256(ROBUST_PROBE_METADATA)
        or probe_provenance.get("config_semantically_equal") is not True
        or float(probe_provenance.get("probe_test_median_xyz_error_mm", np.inf)) >= 15.0
    ):
        raise RuntimeError("offline robust-probe reuse provenance changed")
    core = _csv_core(gate_path.parent / "candidate_scores.csv")
    errors = core["errors"]
    for row in core["rows"]:
        if row["dataset_row"] != int(formal_rows[row["env_idx"]]):
            raise RuntimeError("candidate CSV row/audit-env mapping changed")
    if any(
        errors[model][condition]["all"]["count"] != len(AUDIT_ENVS) * NUM_CANDIDATES
        for model in MODEL_LABELS
        for condition in CONDITIONS
    ):
        raise RuntimeError("candidate model/color cell is not exactly 3,600 rows")
    expert_core = _expert_csv_core(
        gate_path.parent / "expert_measurement.csv",
        gate_path.parent / "expert_measurement.json",
    )
    expert = expert_core["declared"]
    summary = _read_json(gate_path.parent / "summary.json")
    if (
        summary.get("format_version") != "cube_play_v1_offline_summary_v1"
        or summary.get("errors") != errors
        or summary.get("expert") != expert
        or summary.get("candidate_scores")
        != legacy._identity(gate_path.parent / "candidate_scores.csv")
    ):
        raise RuntimeError("candidate/expert CSV and summary.json are inconsistent")
    rebuilt = _build_gate(checkpoint, errors, expert, gate_path.parent)
    # Artifact mtimes can differ only if files were rewritten, which the gate identities
    # above already rejects.  The semantic gate must otherwise reproduce exactly.
    if rebuilt != gate:
        raise RuntimeError("aggregate gate values are not reproducible from frozen artifacts")
    return gate


def command_validate(args: argparse.Namespace) -> int:
    gate = _validate_gate(args.gate, args.checkpoint)
    print(json.dumps({"status": gate["status"], "checkpoint": str(args.checkpoint.resolve())}, sort_keys=True))
    return 0


def command_self_test(_: argparse.Namespace) -> int:
    if _canonical_sha(LOSS_CONTRACT) != LOSS_CONTRACT_SHA256:
        raise AssertionError("loss contract hash mismatch")
    if LOSS_CONTRACT["rollout_enabled"] or LOSS_CONTRACT["rollout_depth"] != 0 or LOSS_CONTRACT["model_rollout_calls"] != 0:
        raise AssertionError("rollout loss is enabled")
    if _gate_exit_code("PASS") != 0 or _gate_exit_code("FAIL") != 3:
        raise AssertionError("gate exit codes changed")
    evaluator_identity = legacy._identity(Path(__file__))
    sha_only_identity = {
        key: value
        for key, value in evaluator_identity.items()
        if key != "mtime_ns"
    }
    mtime_only_identity = {
        key: value
        for key, value in evaluator_identity.items()
        if key != "sha256"
    }
    if not _declares_file(sha_only_identity, Path(__file__)):
        raise AssertionError("SHA-bound identity without mtime was rejected")
    if not _declares_file(mtime_only_identity, Path(__file__)):
        raise AssertionError("mtime-bound identity without SHA was rejected")
    unbound_identity = {
        key: value
        for key, value in evaluator_identity.items()
        if key not in {"mtime_ns", "sha256"}
    }
    tampered_sha_identity = dict(sha_only_identity, sha256="0" * 64)
    if _declares_file(unbound_identity, Path(__file__)) or _declares_file(
        tampered_sha_identity, Path(__file__)
    ):
        raise AssertionError("unbound or tampered file identity was accepted")
    base_loss = 0.01
    provenance = {
        "num_batches": 34, "examples": 4352, "batch_size": 128,
        "shuffle": False, "drop_last": True, "precision": "bf16",
        "expert_clip_indices_sha256": "a" * 64,
    }
    history = {
        "format_version": "cube_play_v1_stopline_history_v1", "run_id": RUN_ID,
        "baseline": {"step": 0, "teacher_pred_loss": base_loss, "relative_increase": 0.0,
                     "status": "PASS", "provenance": provenance},
        "threshold": 0.1, "interval": 500,
        "comparison": "relative increase strictly > threshold triggers",
        "evaluated_after_optimizer_step": True, "triggered": False,
        "records": [{"step": step, "teacher_pred_loss": base_loss * 1.05,
                     "relative_increase": base_loss * 1.05 / base_loss - 1.0,
                     "status": "PASS", "provenance": provenance}
                    for step in range(500, 5001, 500)],
        "resume_events": [],
    }
    positive = _validate_history(history, 5000)
    boundary = json.loads(json.dumps(history))
    boundary_loss = base_loss * 1.1
    boundary_relative = boundary_loss / base_loss - 1.0
    boundary["records"][-1]["teacher_pred_loss"] = boundary_loss
    boundary["records"][-1]["relative_increase"] = boundary_relative
    _validate_history(boundary, 5000)
    if _stopline_exceeded(0.10000000000000009) or not _stopline_exceeded(0.100000000002):
        raise AssertionError("stopline numeric boundary tolerance changed")
    negatives = 0
    for mutation in ("missing", "triggered", "increase", "provenance"):
        bad = json.loads(json.dumps(history))
        if mutation == "missing":
            bad["records"].pop()
        elif mutation == "triggered":
            bad["triggered"] = True
        elif mutation == "increase":
            bad["records"][-1]["teacher_pred_loss"] = base_loss * 1.11
            bad["records"][-1]["relative_increase"] = 0.11
        else:
            bad["records"][-1]["provenance"]["examples"] = 1
        try:
            _validate_history(bad, 5000)
        except RuntimeError:
            negatives += 1
    if negatives != 4:
        raise AssertionError("stopline negative checks did not fail closed")
    candidate_keys = _candidate_expected_keys()
    expert_keys = _expert_expected_keys()
    if len(candidate_keys) != 2 * 3 * 12 * 300 or len(expert_keys) != 2 * 5 * 2000:
        raise AssertionError("formal offline cell cardinalities changed")
    coverage_negatives = 0
    for label, keys in (("candidate", candidate_keys), ("expert", expert_keys)):
        bad_keys = set(keys)
        bad_keys.pop()
        try:
            _require_exact_coverage(label, bad_keys, keys)
        except RuntimeError:
            coverage_negatives += 1
    if coverage_negatives != 2:
        raise AssertionError("coverage negatives did not fail closed")
    if AUDIT_ENVS is not legacy.AUDIT_ENVS or AUDIT_ENVS != (0, 1, 2, 6, 7, 11, 12, 23, 26, 37, 38, 49):
        raise AssertionError("audit env contract is not bound to legacy exact tuple")
    exclusion_fixture = {
        **{name: 0 for name in PLAY_EXCLUSION_ZERO_FIELDS},
        **PLAY_EXCLUSION_COUNT_CONTRACT,
        **{name: f"nonempty {name}" for name in PLAY_EXCLUSION_TEXT_FIELDS},
        "play_episode_namespace": "official_play_train_val",
        "quantization": 0.001,
    }
    _validate_play_exclusion(exclusion_fixture)
    exclusion_negative = 0
    for mutation in ("measurement1_overlap", "unknown", "count"):
        tampered_exclusion = dict(exclusion_fixture)
        if mutation == "measurement1_overlap":
            tampered_exclusion["quantized_signature_overlap_with_measurement1"] = 1
        elif mutation == "unknown":
            tampered_exclusion["unexpected_field"] = 0
        else:
            tampered_exclusion["play_episode_count"] = 1_099
        try:
            _validate_play_exclusion(tampered_exclusion)
        except RuntimeError:
            exclusion_negative += 1
    if exclusion_negative != 3:
        raise AssertionError("play exclusion schema/count mutations were accepted")
    synthetic = []
    for model in MODEL_LABELS:
        for condition in CONDITIONS:
            for idx, success in enumerate((True, False)):
                synthetic.append({"model": model, "condition": condition,
                                  "E_roll_mm": 30.0 + idx, "E_enc_mm": 3.0,
                                  "E_imag_mm": 29.0, "roll_gt_40mm": False,
                                  "final_success": success, "ever_success": success})
    strata = _error_summary(synthetic)
    if any(strata["play_new"][c][s]["count"] != 1 for c in CONDITIONS for s in ("final_success", "final_failure")):
        raise AssertionError("success/failure strata failed")
    print(json.dumps({"self_test": "PASS", "loss_contract_sha256": LOSS_CONTRACT_SHA256,
                      "stopline_positive": positive, "stopline_negative_checks": negatives,
                      "stopline_boundary_relative": boundary_relative,
                      "coverage_negative_checks": coverage_negatives,
                      "candidate_rows": len(candidate_keys), "expert_rows": len(expert_keys),
                      "audit_envs": AUDIT_ENVS,
                      "play_exclusion_zero_fields": PLAY_EXCLUSION_ZERO_FIELDS,
                      "play_exclusion_negative_checks": exclusion_negative,
                      "file_identity_checks": 4,
                      "rate_thresholds": RATE_THRESHOLDS, "strata": "PASS"}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--output", type=Path, default=OFFLINE_ROOT)
    run.add_argument("--device", default="cuda")
    run.add_argument("--rollout-batch-size", type=int, default=300)
    run.add_argument("--encoder-batch-size", type=int, default=128)
    run.add_argument("--expert-batch-size", type=int, default=64)
    run.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate-gate")
    validate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    validate.add_argument("--gate", type=Path, default=OFFLINE_ROOT / "gate.json")
    commands.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if min(args.rollout_batch_size, args.encoder_batch_size, args.expert_batch_size) < 1:
            raise ValueError("batch sizes must be positive")
        return command_run(args)
    if args.command == "validate-gate":
        return command_validate(args)
    return command_self_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
