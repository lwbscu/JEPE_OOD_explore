#!/usr/bin/env python3
"""Frozen contracts shared by the independent Cube Trust-Region tools."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEWM_ROOT = PROJECT_ROOT / "le-wm"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/eval/cube/trust_region"
MEMORY_INDEX = PROJECT_ROOT / "outputs/memory_index/cube_expert_v1"
DATASET = PROJECT_ROOT / "datasets/ogbench/cube_single_expert.h5"
MANIFEST = PROJECT_ROOT / "outputs/audit/cube_cem_manifest.json"
TMP_ROOT = PROJECT_ROOT.parent / "tmp"
MASKED_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt"
)
MASKED_CONFIG = MASKED_CHECKPOINT.parent / "config.json"
MASKED_PROBE = PROJECT_ROOT / "models/probes/cube_imagination_error_xyz_v1/masked.pt"
MASKED_CHECKPOINT_SHA256 = "d64501aa8e7dac1205d3a134c5bd7c160361e16d6da54c79e21e974cdc953117"
MASKED_CONFIG_SHA256 = "86f2ed24c61b48354416c23af51aa51279ae28a33cb36b7ebc3d057eec2b8c0d"

FORMAL_SEED = 42
NUM_SAMPLES = 300
N_STEPS = 10
TOPK = 30
HORIZON = 5
ACTION_BLOCK = 5
ACTION_DIM = 5
MEMORY_SLOTS = 10
NOISY_SLOTS = 20
NOISE_SIGMA = 0.1
AUDIT_ENVS = (0, 1, 2, 6, 7, 11, 12, 23, 26, 37, 38, 49)
CONDITIONS = ("red", "blue_v2", "yellow_v2")
PROTOCOLS = ("t1", "t2")

PROTOCOL_SPECS = {
    "t1": {
        "name": "nearest_seed_initial_mean",
        "initial_mean": "nearest episode-excluded memory action seed",
        "var_scale": 0.2,
        "var_scale_semantics": "legacy CEM multiplies randn by this value; effective initial std=0.2",
        "injected_slots": [],
        "noise_slots": [],
        "noise_sigma": None,
    },
    "t2": {
        "name": "standard_plus_seed_and_local_noise",
        "initial_mean": "standard zeros",
        "var_scale": 1.0,
        "var_scale_semantics": "legacy CEM multiplies randn by this value; effective initial std=1.0",
        "injected_slots": list(range(1, 11)),
        "noise_slots": list(range(11, 31)),
        "noise_sigma": NOISE_SIGMA,
        "noise_clip": [-1.0, 1.0],
        "noise_domain": "raw environment action domain before StandardScaler",
        "noise_shape": [2, 10, 5, 25],
        "noise_reuse": "sample once per planning cycle and reuse for all 10 CEM iterations",
        "noise_parent_mapping": "slots11..20 replica0 seeds0..9; slots21..30 replica1 seeds0..9",
        "noise_rng": "independent CPU torch.Generator; does not touch solver generator",
    },
}


def configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(PROJECT_ROOT),
        "HF_HOME": str(PROJECT_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(PROJECT_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(PROJECT_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        os.environ[key] = value
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: jsonable(row.get(field)) for field in fields})
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, sha256: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    result = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if sha256:
        result["sha256"] = sha256_file(resolved)
    return result


def ensure_data_disk(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    disk = PROJECT_ROOT.parent.resolve()
    if resolved != disk and disk not in resolved.parents:
        raise ValueError(f"{label} must be on /root/autodl-tmp: {resolved}")
    return resolved


def ensure_child(path: Path, root: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise ValueError(f"refusing symlink {label}: {lexical}")
    resolved = ensure_data_disk(lexical, label)
    frozen_root = root.resolve()
    if resolved == frozen_root or frozen_root not in resolved.parents:
        raise ValueError(
            f"{label} must be a concrete child of {frozen_root}: {resolved}"
        )
    return resolved


def prepare_output(path: Path, root: Path, overwrite: bool) -> Path:
    resolved = ensure_child(path, root, "Trust-Region output")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output exists but is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is nonempty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def condition_visual(condition: str) -> tuple[str, str, str]:
    return {
        "red": ("red", "matched", "dataset"),
        "blue_v2": ("blue", "recolor", "blue"),
        "yellow_v2": ("yellow", "recolor", "yellow"),
    }[condition]


def default_eval_output(
    protocol: str,
    condition: str,
    num_eval: int,
) -> Path:
    if protocol not in PROTOCOLS or condition not in CONDITIONS:
        raise ValueError(f"invalid Trust-Region protocol/condition: {protocol}/{condition}")
    if num_eval == 50:
        return OUTPUT_ROOT / protocol.upper() / condition
    return OUTPUT_ROOT / "smoke" / protocol.upper() / condition


def capture_output_root(protocol: str, condition: str) -> Path:
    if protocol not in PROTOCOLS or condition not in CONDITIONS:
        raise ValueError(f"invalid Trust-Region protocol/condition: {protocol}/{condition}")
    return OUTPUT_ROOT / "gate_capture" / protocol.upper() / condition


def physical_cache_root(protocol: str, condition: str) -> Path:
    return OUTPUT_ROOT / "physical_cache" / protocol.upper() / condition


def imagination_output_root(protocol: str, condition: str) -> Path:
    return OUTPUT_ROOT / "imagination_error" / protocol.upper() / condition


def frozen_masked_checkpoint_contract() -> dict[str, Any]:
    """Validate the exact MaskedAug checkpoint used by every Trust-Region arm."""

    for path, expected, label in (
        (MASKED_CHECKPOINT, MASKED_CHECKPOINT_SHA256, "weights"),
        (MASKED_CONFIG, MASKED_CONFIG_SHA256, "config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"frozen MaskedAug {label} missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen MaskedAug {label} hash mismatch: path={path}, "
                f"expected={expected}, actual={actual}"
            )
    return {
        "kind": "frozen_route21_maskedaug",
        "derived_label": "maskedaug",
        "weights": file_identity(MASKED_CHECKPOINT),
        "config": file_identity(MASKED_CONFIG),
    }


def case_name(env_idx: int, dataset_row: int) -> str:
    return f"env_{env_idx:02d}_row_{dataset_row}"


def noise_seed(dataset_row: int, planning_cycle: int) -> tuple[int, np.ndarray]:
    components = np.asarray(
        [FORMAL_SEED, int(dataset_row), int(planning_cycle)], dtype=np.int64
    )
    material = (
        f"cube-trust-region-t2-noise-v1|{FORMAL_SEED}|"
        f"{int(dataset_row)}|{int(planning_cycle)}"
    ).encode("ascii")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    derived &= (1 << 63) - 1
    return derived, components


def noisy_seed_variants(
    seed_actions_raw: np.ndarray,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    dataset_row: int,
    planning_cycle: int,
) -> dict[str, np.ndarray]:
    import torch

    raw = np.asarray(seed_actions_raw, dtype=np.float32)
    mean = np.asarray(action_mean, dtype=np.float64).reshape(ACTION_DIM)
    scale = np.asarray(action_scale, dtype=np.float64).reshape(ACTION_DIM)
    if raw.shape != (MEMORY_SLOTS, HORIZON * ACTION_BLOCK, ACTION_DIM):
        raise ValueError(f"raw memory seed shape mismatch: {raw.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(mean).all() or np.any(scale <= 0):
        raise ValueError("raw seed/scaler contains invalid values")
    seed_value, components = noise_seed(dataset_row, planning_cycle)
    generator = torch.Generator(device="cpu").manual_seed(seed_value)
    noise = (
        torch.randn(
            2,
            MEMORY_SLOTS,
            HORIZON,
            ACTION_BLOCK * ACTION_DIM,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        * NOISE_SIGMA
    ).numpy()
    raw_blocks = raw.reshape(MEMORY_SLOTS, HORIZON, ACTION_BLOCK * ACTION_DIM)
    unclipped = raw_blocks[None] + noise
    clipped = np.clip(unclipped, -1.0, 1.0).astype(np.float32, copy=False)
    # StandardScaler.transform(float32) performs these operations in-place.
    normalized_env_steps = clipped.reshape(2 * MEMORY_SLOTS, HORIZON * ACTION_BLOCK, ACTION_DIM).copy()
    normalized_env_steps -= mean
    normalized_env_steps /= scale
    normalized = normalized_env_steps.reshape(
        2, MEMORY_SLOTS, HORIZON, ACTION_BLOCK * ACTION_DIM
    )
    return {
        "parent_indices": np.tile(np.arange(MEMORY_SLOTS, dtype=np.int64), 2),
        "derived_seed": np.asarray(seed_value, dtype=np.int64),
        "seed_components": components,
        "noise_raw": noise,
        "unclipped_raw": unclipped,
        "clipped_raw": clipped,
        "normalized": normalized,
        "clip_mask": unclipped != clipped,
    }


def distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        if len(array):
            raise ValueError("distribution received nonfinite values")
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
    }
