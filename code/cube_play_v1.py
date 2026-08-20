#!/usr/bin/env python3
"""Data, integrity, and loss contracts for Cube play-v1 dynamics training.

This module intentionally contains no simulator or download path.  Formal
training consumes only the rendered 224px train/val HDF5 shards declared by a
``cube_play_v1_dataset_v1`` manifest.  It never reads OGBench's optional 64px
visual observations.  The loader independently revalidates shard identities,
201-frame episode structure, 4-frame/3-action windows, and every zero-overlap
counter before exposing strict 80-expert/48-play batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import hdf5plugin  # noqa: F401  # Register HDF5 filters before importing h5py.
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


AILAB = Path(__file__).resolve().parent.parent
PREPARED_FORMAT_VERSION = "cube_play_v1_dataset_v1"
NUM_FRAMES = 4
FRAMESKIP = 5
ACTION_DIM = 25
FORMAL_BATCH_SIZE = 128
FORMAL_EXPERT_BATCH = 80
FORMAL_PLAY_BATCH = 48
BUNDLE_EXPERT = 5
BUNDLE_PLAY = 3
LOADER_BATCH = 16
TRAINABLE_PREFIXES = ("predictor", "action_encoder")
FROZEN_MODULES = ("encoder", "projector", "pred_proj")
VALIDATION_FORMAT_VERSION = "cube_play_v1_validation_v1"
OFFICIAL_SOURCE_REPOSITORY = "https://rail.eecs.berkeley.edu/datasets/ogbench/"
TRANSPORT_MIRROR = "https://huggingface.co/datasets/ryanhoangt/ogbench_data"
TRANSPORT_REVISION = "0290b1be6721a8750c77334c316aca998ba4aa8b"
EXPECTED_SOURCE_SHA256 = {
    "train": "80f3b6fd27f4f9d9e9eb6f0d07d6951559012f45b1e15ea4046ef8ecd8d3684e",
    "val": "96d07401bdebdc3f0ea6d56ed1333863e0962f441483adc2c43b83105046eb00",
}
EXPECTED_SOURCE_FILENAMES = {
    "train": "cube-single-play-v0.npz",
    "val": "cube-single-play-v0-val.npz",
}
EXPECTED_EPISODES = {"train": 1000, "val": 100}
RAW_EP_LEN = 1001
STORED_EP_LEN = 201
RAW_STRIDE = 5
WINDOWS_PER_EPISODE = 198
REQUIRED_H5_KEYS = (
    "pixels",
    "action_block",
    "action_block_valid",
    "observation",
    "qpos",
    "qvel",
    "block_pos",
    "ee_pos",
    "ep_idx",
    "step_idx",
    "source_row",
    "ep_offset",
    "ep_len",
)
EXCLUSION_ZERO_FIELDS = (
    "exact_episode_hash_overlap_with_expert",
    "exact_episode_hash_overlap_with_formal50",
    "exact_episode_hash_overlap_with_measurement1",
    "quantized_signature_overlap_with_expert",
    "quantized_signature_overlap_with_formal50",
    "quantized_signature_overlap_with_measurement1",
)
EXCLUSION_EXPLANATION_FIELDS = (
    "exact_hash_contract",
    "independent_collection_claim",
    "quantized_signature_contract",
)
EXCLUSION_FIXED_FIELDS: dict[str, Any] = {
    "expert_episode_count": 10_000,
    "formal50_episode_count": 50,
    "measurement1_unique_episode_count": 1_801,
    "play_episode_count": 1_100,
    "play_episode_namespace": "official_play_train_val",
    "quantization": 0.001,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_identity(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = sha256_file(path)
    return result


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash parameters and mutable buffers (including BatchNorm state)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def freeze_dynamics_stack(model: torch.nn.Module) -> dict[str, Any]:
    """Train exactly predictor+action_encoder and freeze every other tensor."""
    trainable: list[str] = []
    frozen: list[str] = []
    def in_component(parameter_name: str, prefix: str) -> bool:
        return parameter_name == prefix or parameter_name.startswith(prefix + ".")

    for name, parameter in model.named_parameters():
        enabled = any(in_component(name, prefix) for prefix in TRAINABLE_PREFIXES)
        parameter.requires_grad_(enabled)
        (trainable if enabled else frozen).append(name)
    for prefix in TRAINABLE_PREFIXES:
        if not any(in_component(name, prefix) for name in trainable):
            raise RuntimeError(f"trainable model component is missing: {prefix}")
    unexpected = [
        name
        for name in trainable
        if not any(in_component(name, prefix) for prefix in TRAINABLE_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected}")
    for name in FROZEN_MODULES:
        module = getattr(model, name, None)
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError(f"required frozen module is missing: {name}")
        module.eval()
    return {
        "trainable_prefixes": list(TRAINABLE_PREFIXES),
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "frozen_parameter_tensors": len(frozen),
        "frozen_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        ),
        "frozen_integrity_modules": list(FROZEN_MODULES),
        "pred_proj_decision": (
            "frozen: the iron law authorizes only predictor+action_encoder; "
            "pred_proj is a fixed BatchNorm output head"
        ),
    }


def frozen_state_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: module_state_sha256(getattr(model, name)) for name in FROZEN_MODULES
    }


def assert_frozen_hashes(
    model: torch.nn.Module, expected: Mapping[str, str]
) -> dict[str, str]:
    actual = frozen_state_hashes(model)
    if actual != dict(expected):
        raise RuntimeError(
            "frozen module state changed: "
            f"expected={dict(expected)}, actual={actual}, modules={FROZEN_MODULES}"
        )
    return actual


def _resolve_declared_file(
    root: Path,
    declared: Any,
    label: str,
    *,
    verify_sha256: bool,
    require_mtime: bool,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"{label} identity is not an object")
    path = Path(str(declared.get("path", ""))).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    stat = path.stat()
    expected = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        **({"mtime_ns": int(stat.st_mtime_ns)} if require_mtime else {}),
    }
    mismatch = {
        key: {"expected": value, "actual": declared.get(key)}
        for key, value in expected.items()
        if declared.get(key) != value
    }
    digest = str(declared.get("sha256", ""))
    if len(digest) != 64:
        mismatch["sha256"] = {"expected": "64 hex chars", "actual": digest}
    if mismatch:
        raise RuntimeError(f"{label} identity mismatch: {mismatch}")
    if verify_sha256:
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(
                f"{label} SHA256 mismatch: expected={digest}, actual={actual}, path={path}"
            )
    return path, {**dict(declared), "path": str(path)}


def _expected_prepared_schema() -> dict[str, dict[str, Any]]:
    return {
        "pixels": {"dtype": "uint8", "shape": ["N", 224, 224, 3]},
        "action_block": {"dtype": "float32", "shape": ["N", 25]},
        "action_block_valid": {"dtype": "bool", "shape": ["N"]},
        "observation": {"dtype": "float32", "shape": ["N", 28]},
        "qpos": {"dtype": "float32", "shape": ["N", 21]},
        "qvel": {"dtype": "float32", "shape": ["N", 20]},
        "block_pos": {"dtype": "float32", "shape": ["N", 3]},
        "ee_pos": {"dtype": "float32", "shape": ["N", 3]},
        "ep_idx": {"dtype": "int32", "shape": ["N"]},
        "step_idx": {"dtype": "int32", "shape": ["N"]},
        "source_row": {"dtype": "int64", "shape": ["N"]},
        "ep_offset": {"dtype": "int64", "shape": ["E"]},
        "ep_len": {"dtype": "int32", "shape": ["E"]},
    }


def _validate_exclusion_contract(exclusion: Any) -> dict[str, Any]:
    """Require the complete measured exclusion schema, including Measurement-1."""
    expected_keys = (
        set(EXCLUSION_ZERO_FIELDS)
        | set(EXCLUSION_EXPLANATION_FIELDS)
        | set(EXCLUSION_FIXED_FIELDS)
    )
    if not isinstance(exclusion, Mapping) or set(exclusion) != expected_keys:
        actual = sorted(exclusion) if isinstance(exclusion, Mapping) else type(exclusion).__name__
        raise RuntimeError(
            "prepared play exclusion schema changed: "
            f"expected={sorted(expected_keys)}, actual={actual}"
        )
    mismatch: dict[str, Any] = {}
    for name in EXCLUSION_ZERO_FIELDS:
        if type(exclusion[name]) is not int or exclusion[name] != 0:
            mismatch[name] = {"expected": "integer 0", "actual": exclusion[name]}
    for name, expected in EXCLUSION_FIXED_FIELDS.items():
        actual = exclusion[name]
        if isinstance(expected, int):
            valid = type(actual) is int and actual == expected
        elif isinstance(expected, float):
            valid = type(actual) is float and actual == expected
        else:
            valid = type(actual) is str and actual == expected
        if not valid:
            mismatch[name] = {"expected": expected, "actual": actual}
    for name in EXCLUSION_EXPLANATION_FIELDS:
        if type(exclusion[name]) is not str or not exclusion[name].strip():
            mismatch[name] = {"expected": "non-empty explanatory string", "actual": exclusion[name]}
    if mismatch:
        raise RuntimeError(f"prepared play exclusion evidence failed: {mismatch}")
    return dict(exclusion)


def _assert_array_equal(
    actual: np.ndarray, expected: np.ndarray, label: str, position: Any
) -> None:
    if not np.array_equal(actual, expected):
        mismatch = np.argwhere(actual != expected)
        first = mismatch[0].tolist() if len(mismatch) else "shape/dtype"
        raise RuntimeError(
            f"{label} differs from raw source: expected exact equality, "
            f"actual mismatch at {first}, position={position}"
        )


def _validate_raw_alignment(
    split: str,
    raw_path: Path,
    shard_path: Path,
    episodes: int,
) -> dict[str, Any]:
    """Independently reconstruct every stored numeric row from official raw NPZ."""
    raw_rows = episodes * RAW_EP_LEN
    raw_contract = {
        "observations": ((raw_rows, 28), np.dtype("float32")),
        "actions": ((raw_rows, 5), np.dtype("float32")),
        "terminals": ((raw_rows,), np.dtype("bool")),
        "qpos": ((raw_rows, 21), np.dtype("float32")),
        "qvel": ((raw_rows, 20), np.dtype("float32")),
    }
    sampled_steps = np.arange(0, RAW_EP_LEN, RAW_STRIDE, dtype=np.int32)
    local_rows = (
        np.arange(episodes, dtype=np.int64)[:, None] * RAW_EP_LEN
        + sampled_steps.astype(np.int64)[None]
    ).reshape(-1)
    episode_offset = 0 if split == "train" else EXPECTED_EPISODES["train"]
    expected_ep_idx = np.repeat(
        np.arange(episode_offset, episode_offset + episodes, dtype=np.int32),
        STORED_EP_LEN,
    )
    expected_step_idx = np.tile(sampled_steps, episodes)
    expected_offsets = np.arange(episodes, dtype=np.int64) * STORED_EP_LEN
    expected_valid = np.ones((episodes, STORED_EP_LEN), dtype=bool)
    expected_valid[:, -1] = False

    with np.load(raw_path) as raw, h5py.File(shard_path, "r", swmr=True) as h5:
        if set(raw.files) != set(raw_contract):
            raise RuntimeError(f"official {split} NPZ keys changed: {sorted(raw.files)}")
        for key, (shape, dtype) in raw_contract.items():
            if raw[key].shape != shape or raw[key].dtype != dtype:
                raise RuntimeError(
                    f"official {split}.{key} changed: expected={shape}/{dtype}, "
                    f"actual={raw[key].shape}/{raw[key].dtype}"
                )
        terminal_expected = np.arange(RAW_EP_LEN - 1, raw_rows, RAW_EP_LEN)
        terminal_actual = np.flatnonzero(raw["terminals"])
        _assert_array_equal(
            terminal_actual, terminal_expected, f"official {split} terminals", raw_path
        )

        _assert_array_equal(
            np.asarray(h5["ep_offset"][:]), expected_offsets, f"{split}.ep_offset", shard_path
        )
        _assert_array_equal(
            np.asarray(h5["ep_len"][:]),
            np.full(episodes, STORED_EP_LEN, dtype=np.int32),
            f"{split}.ep_len",
            shard_path,
        )
        _assert_array_equal(
            np.asarray(h5["ep_idx"][:]), expected_ep_idx, f"{split}.ep_idx", shard_path
        )
        _assert_array_equal(
            np.asarray(h5["step_idx"][:]), expected_step_idx, f"{split}.step_idx", shard_path
        )
        _assert_array_equal(
            np.asarray(h5["source_row"][:]), local_rows, f"{split}.source_row", shard_path
        )
        valid = np.asarray(h5["action_block_valid"][:], dtype=bool).reshape(
            episodes, STORED_EP_LEN
        )
        _assert_array_equal(valid, expected_valid, f"{split}.action_block_valid", shard_path)

        # Exact time-major contract: block at raw t is actions[t:t+5].reshape(25).
        raw_actions = np.asarray(raw["actions"], dtype=np.float32).reshape(
            episodes, RAW_EP_LEN, 5
        )
        for start_episode in range(0, episodes, 32):
            stop_episode = min(start_episode + 32, episodes)
            row_start = start_episode * STORED_EP_LEN
            row_stop = stop_episode * STORED_EP_LEN
            actual = np.asarray(h5["action_block"][row_start:row_stop], dtype=np.float32).reshape(
                stop_episode - start_episode, STORED_EP_LEN, 25
            )
            expected = np.zeros_like(actual)
            expected[:, :200] = raw_actions[start_episode:stop_episode, :1000].reshape(
                stop_episode - start_episode, 200, 25
            )
            _assert_array_equal(
                actual,
                expected,
                f"{split}.action_block",
                {"episode_start": start_episode, "episode_stop": stop_episode},
            )
        del raw_actions

        for raw_key, h5_key, width in (
            ("observations", "observation", 28),
            ("qpos", "qpos", 21),
            ("qvel", "qvel", 20),
        ):
            source = np.asarray(raw[raw_key])
            for start_episode in range(0, episodes, 32):
                stop_episode = min(start_episode + 32, episodes)
                row_start = start_episode * STORED_EP_LEN
                row_stop = stop_episode * STORED_EP_LEN
                expected = source.reshape(episodes, RAW_EP_LEN, width)[
                    start_episode:stop_episode, ::RAW_STRIDE
                ].reshape(-1, width)
                actual = np.asarray(h5[h5_key][row_start:row_stop])
                _assert_array_equal(
                    actual,
                    expected,
                    f"{split}.{h5_key}",
                    {"episode_start": start_episode, "episode_stop": stop_episode},
                )
            del source

        for start in range(0, episodes * STORED_EP_LEN, 131_072):
            stop = min(start + 131_072, episodes * STORED_EP_LEN)
            observation = np.asarray(h5["observation"][start:stop], dtype=np.float32)
            block = (
                observation[:, 19:22] / np.float32(10.0)
                + np.asarray([0.425, 0.0, 0.0], dtype=np.float32)
            ).astype(np.float32, copy=False)
            ee = (
                observation[:, 12:15] / np.float32(10.0)
                + np.asarray([0.425, 0.0, 0.0], dtype=np.float32)
            ).astype(np.float32, copy=False)
            _assert_array_equal(
                np.asarray(h5["block_pos"][start:stop]), block, f"{split}.block_pos", start
            )
            _assert_array_equal(
                np.asarray(h5["ee_pos"][start:stop]), ee, f"{split}.ee_pos", start
            )
    return {
        "raw_rows": raw_rows,
        "stored_rows": episodes * STORED_EP_LEN,
        "episodes": episodes,
        "action_blocks_exact": episodes * 200,
        "numeric_rows_exact": episodes * STORED_EP_LEN,
    }


def _validate_qc_pixels(
    qc_path: Path, shard_paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Verify report-only cross-EGL diff evidence without gating pixel values."""
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    required_qc_keys = {
        "format_version",
        "status",
        "no_action_rollout",
        "num_frames",
        "tolerance",
        "report_only_reason",
        "runs",
        "geometry_qc",
        "smoke_report",
        "rows",
    }
    if (
        set(qc) != required_qc_keys
        or qc.get("format_version") != "cube_play_v1_pixel_alignment_v1"
        or qc.get("status") != "REPORT_ONLY_EGL_NONDETERMINISM"
        or qc.get("no_action_rollout") is not True
        or qc.get("num_frames") != 20
        or qc.get("tolerance") is not None
        or not isinstance(qc.get("report_only_reason"), str)
        or not qc["report_only_reason"]
        or not isinstance(qc.get("geometry_qc"), Mapping)
        or qc["geometry_qc"].get("status") != "PASS"
    ):
        raise RuntimeError("play pixel alignment report contract changed")
    runs = qc.get("runs")
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or [run.get("run_id") for run in runs if isinstance(run, Mapping)]
        != ["train_report_only_run_1", "full_report_only_run_2"]
    ):
        raise RuntimeError("play pixel report-only run evidence is incomplete")
    first_run, second_run = runs
    if (
        set(first_run)
        != {"run_id", "source", "samples", "byte_exact_samples", "mismatches"}
        or first_run.get("samples") != 10
        or first_run.get("byte_exact_samples") != 7
        or not isinstance(first_run.get("mismatches"), list)
        or len(first_run["mismatches"]) != 3
        or set(second_run)
        != {"run_id", "samples", "byte_exact_samples", "rows_reference"}
        or second_run.get("samples") != 20
        or second_run.get("rows_reference") != "top-level rows"
    ):
        raise RuntimeError("play pixel report-only run schema changed")
    prior_fields = {
        "source_row",
        "expected_sha256",
        "actual_sha256",
        "max_abs_channel_delta",
        "changed_pixels",
        "changed_channels",
        "mean_abs_channel_delta",
    }
    for index, item in enumerate(first_run["mismatches"]):
        if (
            not isinstance(item, Mapping)
            or set(item) != prior_fields
            or len(str(item.get("expected_sha256", ""))) != 64
            or len(str(item.get("actual_sha256", ""))) != 64
            or not all(
                np.isfinite(float(item.get(name, float("nan"))))
                for name in (
                    "max_abs_channel_delta",
                    "changed_pixels",
                    "changed_channels",
                    "mean_abs_channel_delta",
                )
            )
        ):
            raise RuntimeError(f"play prior EGL diff evidence is malformed at {index}")
    smoke_path, _ = _resolve_declared_file(
        qc_path.parent,
        qc["smoke_report"],
        "play render smoke report",
        verify_sha256=True,
        require_mtime=True,
    )
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        smoke.get("format_version") != PREPARED_FORMAT_VERSION
        or smoke.get("status") != "PASS"
        or smoke.get("no_action_rollout") is not True
        or smoke.get("num_frames") != 20
        or not isinstance(smoke.get("rows"), list)
        or len(smoke["rows"]) != 20
    ):
        raise RuntimeError("play render smoke report contract changed")
    geometry = qc["geometry_qc"]
    expected_geometry_keys = {
        "status",
        "samples",
        "max_block_pos_error_m",
        "max_ee_pos_error_m",
        "tolerance",
    }
    max_block_error = max(float(row["block_pos_error_m"]) for row in smoke["rows"])
    max_ee_error = max(float(row["ee_pos_error_m"]) for row in smoke["rows"])
    expected_geometry = {
        "status": "PASS",
        "samples": 20,
        "max_block_pos_error_m": max_block_error,
        "max_ee_pos_error_m": max_ee_error,
        "tolerance": {"block_pos_error_m": 2e-5, "ee_pos_error_m": 2e-3},
    }
    if set(geometry) != expected_geometry_keys or dict(geometry) != expected_geometry:
        raise RuntimeError(
            "play set_state geometry QC differs from bound smoke evidence: "
            f"expected={expected_geometry}, actual={dict(geometry)}"
        )
    smoke_by_key = {
        (str(row.get("split")), int(row.get("episode", -1)), int(row.get("raw_step", -1))): row
        for row in smoke["rows"]
        if isinstance(row, Mapping)
    }
    if len(smoke_by_key) != 20:
        raise RuntimeError("play render smoke anchors are malformed/duplicated")
    rows = qc.get("rows")
    if not isinstance(rows, list) or len(rows) != 20:
        raise RuntimeError(
            f"play pixel QC must contain twenty anchors: actual={type(rows).__name__}/{len(rows) if isinstance(rows, list) else 'n/a'}"
        )
    seen: set[tuple[str, int, int]] = set()
    byte_exact = 0
    max_abs = 0
    max_changed_pixels = 0
    per_split = {"train": 0, "val": 0}
    handles = {
        split: h5py.File(path, "r", swmr=True) for split, path in shard_paths.items()
    }
    try:
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise RuntimeError(f"play pixel QC row {index} is not an object")
            required_row_keys = {
                "split",
                "episode",
                "raw_step",
                "source_row",
                "png",
                "pixel_sha256",
                "expected_sha256",
                "actual_sha256",
                "byte_exact",
                "max_abs_channel_delta",
                "changed_pixels",
                "changed_channels",
                "mean_abs_channel_delta",
                "observation_max_abs_error",
                "block_pos_error_m",
                "ee_pos_error_m",
            }
            if set(row) != required_row_keys:
                raise RuntimeError(
                    f"play pixel QC row {index} evidence schema changed: "
                    f"expected={sorted(required_row_keys)}, actual={sorted(row)}"
                )
            split = str(row.get("split"))
            if split not in handles:
                raise RuntimeError(f"play pixel QC row {index} has invalid split {split!r}")
            episode = int(row.get("episode", -1))
            raw_step = int(row.get("raw_step", -1))
            if (
                episode < 0
                or episode >= EXPECTED_EPISODES[split]
                or raw_step < 0
                or raw_step >= RAW_EP_LEN
                or raw_step % RAW_STRIDE
            ):
                raise RuntimeError(
                    f"play pixel QC row {index} position is invalid: {split}/{episode}/{raw_step}"
                )
            key = (split, episode, raw_step)
            if key in seen:
                raise RuntimeError(f"duplicate play pixel QC anchor: {key}")
            seen.add(key)
            smoke_row = smoke_by_key.get(key)
            smoke_fields = (
                "split",
                "episode",
                "raw_step",
                "source_row",
                "png",
                "pixel_sha256",
                "observation_max_abs_error",
                "block_pos_error_m",
                "ee_pos_error_m",
            )
            if smoke_row is None or any(row.get(name) != smoke_row.get(name) for name in smoke_fields):
                raise RuntimeError(f"play pixel QC row {index} is not bound to render smoke: {key}")
            expected_source_row = episode * RAW_EP_LEN + raw_step
            if int(row.get("source_row", -1)) != expected_source_row:
                raise RuntimeError(
                    f"play pixel QC source_row mismatch at {key}: "
                    f"expected={expected_source_row}, actual={row.get('source_row')}"
                )
            png_name = str(row.get("png", ""))
            png_path = (qc_path.parent / png_name).resolve()
            if png_path.parent != qc_path.parent.resolve() or not png_path.is_file():
                raise RuntimeError(f"play pixel QC PNG is noncanonical/missing: {png_path}")
            with Image.open(png_path) as image:
                archived = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if archived.shape != (224, 224, 3):
                raise RuntimeError(
                    f"play pixel QC PNG shape changed at {key}: {archived.shape}"
                )
            archived_sha = hashlib.sha256(archived.tobytes()).hexdigest()
            if archived_sha != row.get("pixel_sha256"):
                raise RuntimeError(
                    f"play pixel QC PNG SHA mismatch at {key}: "
                    f"expected={row.get('pixel_sha256')}, actual={archived_sha}"
                )
            local_row = episode * STORED_EP_LEN + raw_step // RAW_STRIDE
            actual = np.asarray(handles[split]["pixels"][local_row], dtype=np.uint8)
            delta = np.abs(actual.astype(np.int16) - archived.astype(np.int16))
            frame_max = int(delta.max(initial=0))
            changed = int(np.count_nonzero(np.any(delta != 0, axis=-1)))
            changed_channels = int(np.count_nonzero(delta))
            mean_abs = float(delta.mean())
            actual_sha = hashlib.sha256(actual.tobytes()).hexdigest()
            frame_exact = actual_sha == archived_sha
            declared_metrics = {
                "expected_sha256": archived_sha,
                "actual_sha256": actual_sha,
                "byte_exact": frame_exact,
                "max_abs_channel_delta": frame_max,
                "changed_pixels": changed,
                "changed_channels": changed_channels,
            }
            metric_mismatch = {
                name: {"expected": expected, "actual": row.get(name)}
                for name, expected in declared_metrics.items()
                if row.get(name) != expected
            }
            if metric_mismatch or not math.isclose(
                float(row.get("mean_abs_channel_delta", float("nan"))),
                mean_abs,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError(
                    f"play pixel QC declared metrics differ at {key}: "
                    f"mismatch={metric_mismatch}, expected_mean={mean_abs}, "
                    f"actual_mean={row.get('mean_abs_channel_delta')}"
                )
            byte_exact += int(frame_exact)
            max_abs = max(max_abs, frame_max)
            max_changed_pixels = max(max_changed_pixels, changed)
            per_split[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if per_split != {"train": 10, "val": 10}:
        raise RuntimeError(f"play pixel QC split coverage changed: {per_split}")
    if second_run.get("byte_exact_samples") != byte_exact:
        raise RuntimeError(
            "play second EGL run byte-exact count differs from current evidence: "
            f"expected={byte_exact}, actual={second_run.get('byte_exact_samples')}"
        )
    return {
        "status": "REPORT_ONLY_EGL_NONDETERMINISM",
        "samples": 20,
        "byte_exact_samples": byte_exact,
        "max_abs_channel_delta": max_abs,
        "max_changed_pixels_per_frame": max_changed_pixels,
        "tolerance": None,
        "report_only_reason": qc["report_only_reason"],
        "runs": [run["run_id"] for run in runs],
    }


def load_prepared_play_manifest(
    manifest_path: Path, *, verify_shard_sha256: bool
) -> dict[str, Any]:
    """Fail-closed validation of manifest, audit, shards, and official raw NPZ."""
    manifest_path = manifest_path.expanduser().resolve()
    root = manifest_path.parent
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("format_version") != PREPARED_FORMAT_VERSION:
        raise RuntimeError(
            f"prepared play format changed: expected={PREPARED_FORMAT_VERSION}, "
            f"actual={value.get('format_version')}"
        )
    required_top = {
        "format_version",
        "capture_contract",
        "sources",
        "shards",
        "schema",
        "health_report",
        "validation",
        "exclusion",
        "qc",
    }
    if set(value) != required_top:
        raise RuntimeError(
            f"prepared play manifest keys changed: expected={sorted(required_top)}, "
            f"actual={sorted(value)}"
        )
    expected_capture = {
        "source": "official OGBench state NPZ; deterministic set_state(qpos,qvel) then 224x224 render",
        "simulator_actions_executed": 0,
        "official_train_val_split_preserved": True,
        "play_episode_namespace": "official_play_train_val",
        "global_episode_ids": "train=0..999; val=1000..1099; source_local_episode=global-offset",
        "raw_episode_length": RAW_EP_LEN,
        "raw_temporal_phase": 0,
        "raw_temporal_stride": RAW_STRIDE,
        "stored_episode_length": STORED_EP_LEN,
        "stored_steps": "0,5,...,1000",
        "action_block": "exact raw actions[t:t+5].reshape(25); terminal stored frame is invalid exact-zero placeholder",
        "training_window": "four consecutive stored frames plus first three valid action blocks",
        "warning": "stored rows are 1/5 temporal phase samples and must be loaded with frameskip=1",
    }
    if value["capture_contract"] != expected_capture:
        raise RuntimeError(
            "prepared play capture contract changed: "
            f"expected={expected_capture}, actual={value['capture_contract']}"
        )
    if value["schema"] != _expected_prepared_schema():
        raise RuntimeError("prepared play schema declaration changed")

    required_zero = EXCLUSION_ZERO_FIELDS
    exclusion = _validate_exclusion_contract(value["exclusion"])

    identity_keys = {"path", "sha256", "size_bytes", "mtime_ns"}
    if set(value["health_report"]) != identity_keys:
        raise RuntimeError("play health report identity keys changed")
    if set(value["validation"]) != identity_keys:
        raise RuntimeError("play validation identity keys changed")
    if set(value["qc"]) != identity_keys | {"num_rendered_frames"}:
        raise RuntimeError("play QC identity keys changed")
    if value["qc"].get("num_rendered_frames") != 20:
        raise RuntimeError("formal play QC must bind exactly twenty phase-0 pixels")
    health_path, health_identity = _resolve_declared_file(
        root, value["health_report"], "play health report", verify_sha256=True, require_mtime=True
    )
    qc_path, qc_identity = _resolve_declared_file(
        root, value["qc"], "play QC report", verify_sha256=True, require_mtime=True
    )
    validation_path, validation_identity = _resolve_declared_file(
        root, value["validation"], "play validation report", verify_sha256=True, require_mtime=True
    )
    health = json.loads(health_path.read_text(encoding="utf-8"))
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if health.get("format_version") != PREPARED_FORMAT_VERSION:
        raise RuntimeError("play health report format changed")
    if health.get("exclusion") != exclusion or health.get("sources") != value["sources"]:
        raise RuntimeError("play health report is not bound to manifest sources/exclusion")
    if (
        qc.get("format_version") != "cube_play_v1_pixel_alignment_v1"
        or qc.get("status") != "REPORT_ONLY_EGL_NONDETERMINISM"
        or qc.get("no_action_rollout") is not True
    ):
        raise RuntimeError(
            "play pixel QC report lacks the explicit EGL report-only record"
        )
    required_validation = {
        "format_version",
        "valid",
        "created_unix",
        "sources",
        "shards",
        "qc",
        "checks",
        "exclusion",
    }
    if (
        set(validation) != required_validation
        or validation.get("format_version") != VALIDATION_FORMAT_VERSION
        or validation.get("valid") is not True
        or validation.get("sources") != value["sources"]
        or validation.get("shards") != value["shards"]
        or validation.get("exclusion") != exclusion
        or validation.get("qc") != value["qc"]
    ):
        raise RuntimeError("play validation report header/identity binding failed")
    expected_checks = {
        "full_counts",
        "global_episode_namespace_unique",
        "step_idx_exact_phase0",
        "source_row_exact",
        "action_block_all_exact",
        "terminal_action_block_zero",
        "pixels_qc_exact",
        "qpos_exact",
        "qvel_exact",
        "observation_exact",
        "block_pos_exact_from_observation",
        "ee_pos_exact_from_observation",
    }
    if set(validation["checks"]) != expected_checks:
        raise RuntimeError("play validation check set changed")
    failed_checks = {
        name: check
        for name, check in validation["checks"].items()
        if name != "pixels_qc_exact"
        and (not isinstance(check, Mapping) or check.get("status") != "PASS")
    }
    if failed_checks:
        raise RuntimeError(f"play validation contains non-PASS checks: {failed_checks}")
    pixel_check = validation["checks"]["pixels_qc_exact"]
    if (
        not isinstance(pixel_check, Mapping)
        or pixel_check.get("status") != "REPORT_ONLY_EGL_NONDETERMINISM"
    ):
        raise RuntimeError(
            "play pixel validation must explicitly remain report-only"
        )

    sources = value["sources"]
    shards = value["shards"]
    if not isinstance(sources, Mapping) or set(sources) != {"train", "val"}:
        raise RuntimeError("prepared play sources must be exactly train/val")
    if not isinstance(shards, Mapping) or set(shards) != {"train", "val"}:
        raise RuntimeError("prepared play shards must be exactly train/val")
    resolved_sources: dict[str, dict[str, Any]] = {}
    resolved_shards: dict[str, dict[str, Any]] = {}
    semantic: dict[str, Any] = {}
    all_episode_ids: dict[str, np.ndarray] = {}
    shard_paths: dict[str, Path] = {}
    for split in ("train", "val"):
        source_declared = sources[split]
        expected_source_keys = {
            "path",
            "size_bytes",
            "mtime_ns",
            "sha256",
            "official_split",
            "official_repository",
            "transport_mirror",
            "transport_revision",
            "filename",
        }
        if not isinstance(source_declared, Mapping) or set(source_declared) != expected_source_keys:
            raise RuntimeError(
                f"official play {split} source identity keys changed: "
                f"expected={sorted(expected_source_keys)}, actual={sorted(source_declared)}"
            )
        expected_source_metadata = {
            "official_split": split,
            "official_repository": OFFICIAL_SOURCE_REPOSITORY,
            "transport_mirror": TRANSPORT_MIRROR,
            "transport_revision": TRANSPORT_REVISION,
            "filename": EXPECTED_SOURCE_FILENAMES[split],
            "sha256": EXPECTED_SOURCE_SHA256[split],
        }
        source_mismatch = {
            key: {"expected": expected, "actual": source_declared.get(key)}
            for key, expected in expected_source_metadata.items()
            if source_declared.get(key) != expected
        }
        if source_mismatch:
            raise RuntimeError(f"official play source metadata changed for {split}: {source_mismatch}")
        source_path, source_identity = _resolve_declared_file(
            root,
            source_declared,
            f"official play {split} NPZ",
            verify_sha256=verify_shard_sha256,
            require_mtime=True,
        )
        if source_path.name != EXPECTED_SOURCE_FILENAMES[split] or source_path.parent != root / "source":
            raise RuntimeError(f"official play {split} source path is noncanonical: {source_path}")

        shard_path, shard_identity = _resolve_declared_file(
            root,
            shards[split],
            f"prepared play {split} shard",
            verify_sha256=verify_shard_sha256,
            require_mtime=False,
        )
        expected_shard_keys = {
            "path",
            "sha256",
            "size_bytes",
            "episodes",
            "frames",
            "windows",
            "official_split",
            "global_episode_id_range",
            "source_local_episode_id_offset",
        }
        if set(shards[split]) != expected_shard_keys:
            raise RuntimeError(
                f"prepared play {split} shard identity keys changed: "
                f"expected={sorted(expected_shard_keys)}, actual={sorted(shards[split])}"
            )
        episodes = EXPECTED_EPISODES[split]
        expected_counters = {
            "episodes": episodes,
            "frames": episodes * STORED_EP_LEN,
            "windows": episodes * WINDOWS_PER_EPISODE,
            "official_split": split,
            "global_episode_id_range": [0, 999] if split == "train" else [1000, 1099],
            "source_local_episode_id_offset": 0 if split == "train" else 1000,
        }
        counter_mismatch = {
            key: {"expected": expected, "actual": shards[split].get(key)}
            for key, expected in expected_counters.items()
            if shards[split].get(key) != expected
        }
        if counter_mismatch:
            raise RuntimeError(f"prepared play {split} counters changed: {counter_mismatch}")
        with h5py.File(shard_path, "r", swmr=True) as h5:
            if set(h5) != set(REQUIRED_H5_KEYS):
                raise RuntimeError(
                    f"prepared play {split} H5 keys changed: {sorted(h5.keys())}"
                )
            expected_attrs = {
                "format_version": PREPARED_FORMAT_VERSION,
                "raw_sampling_phase": 0,
                "raw_sampling_stride": RAW_STRIDE,
                "no_action_rollout": True,
            }
            attr_mismatch = {
                key: {"expected": expected, "actual": h5.attrs.get(key)}
                for key, expected in expected_attrs.items()
                if h5.attrs.get(key) != expected
            }
            if attr_mismatch:
                raise RuntimeError(f"prepared play {split} H5 attrs changed: {attr_mismatch}")
            expected_shapes = {
                "pixels": ((episodes * STORED_EP_LEN, 224, 224, 3), np.dtype("uint8")),
                "action_block": ((episodes * STORED_EP_LEN, 25), np.dtype("float32")),
                "action_block_valid": ((episodes * STORED_EP_LEN,), np.dtype("bool")),
                "observation": ((episodes * STORED_EP_LEN, 28), np.dtype("float32")),
                "qpos": ((episodes * STORED_EP_LEN, 21), np.dtype("float32")),
                "qvel": ((episodes * STORED_EP_LEN, 20), np.dtype("float32")),
                "block_pos": ((episodes * STORED_EP_LEN, 3), np.dtype("float32")),
                "ee_pos": ((episodes * STORED_EP_LEN, 3), np.dtype("float32")),
                "ep_idx": ((episodes * STORED_EP_LEN,), np.dtype("int32")),
                "step_idx": ((episodes * STORED_EP_LEN,), np.dtype("int32")),
                "source_row": ((episodes * STORED_EP_LEN,), np.dtype("int64")),
                "ep_offset": ((episodes,), np.dtype("int64")),
                "ep_len": ((episodes,), np.dtype("int32")),
            }
            shape_mismatch = {
                key: {
                    "expected": [list(shape), str(dtype)],
                    "actual": [list(h5[key].shape), str(h5[key].dtype)],
                }
                for key, (shape, dtype) in expected_shapes.items()
                if h5[key].shape != shape or h5[key].dtype != dtype
            }
            if shape_mismatch:
                raise RuntimeError(f"prepared play {split} H5 schema differs: {shape_mismatch}")
            all_episode_ids[split] = np.asarray(
                h5["ep_idx"][np.arange(episodes) * STORED_EP_LEN], dtype=np.int64
            )
        semantic[split] = _validate_raw_alignment(split, source_path, shard_path, episodes)
        shard_paths[split] = shard_path
        resolved_sources[split] = source_identity
        resolved_shards[split] = {**shard_identity, **expected_counters}

    if np.intersect1d(all_episode_ids["train"], all_episode_ids["val"]).size:
        raise RuntimeError("prepared play global train/val episode namespace overlaps")
    pixel_revalidation = _validate_qc_pixels(qc_path, shard_paths)
    if dict(pixel_check) != pixel_revalidation:
        raise RuntimeError(
            "play report-only pixel evidence differs from independent PNG/H5 recheck: "
            f"declared={dict(pixel_check)}, actual={pixel_revalidation}"
        )
    value["manifest_path"] = str(manifest_path)
    value["manifest_sha256"] = sha256_file(manifest_path)
    value["resolved_sources"] = resolved_sources
    value["resolved_shards"] = resolved_shards
    value["resolved_reports"] = {
        "health": health_identity,
        "qc": qc_identity,
        "validation": validation_identity,
    }
    value["semantic_revalidation"] = {
        "status": "PASS",
        "source_visual_64x64_read": False,
        "raw_alignment": semantic,
        "pixel_alignment": pixel_revalidation,
        "exclusion_zero_fields": list(required_zero),
    }
    return value


class PreparedPlayDataset(Dataset):
    """Lazy four-frame/three-action windows from one rendered play H5 shard."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"invalid prepared play split: {split}")
        self.path = Path(str(manifest["resolved_shards"][split]["path"])).resolve()
        self.split = split
        self.transform = transform
        with h5py.File(self.path, "r") as h5:
            self.lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
            self.offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            self.episode_ids = np.asarray(h5["ep_idx"][self.offsets], dtype=np.int64)
        self.window_counts = np.maximum(self.lengths - 3, 0)
        self.window_ends = np.cumsum(self.window_counts)
        if not len(self.window_ends) or int(self.window_ends[-1]) <= 0:
            raise RuntimeError(f"prepared play {split} has no four-frame windows")
        self._pid: int | None = None
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.window_ends[-1])

    def _h5(self) -> h5py.File:
        pid = os.getpid()
        if self._pid != pid or self._handle is None:
            self.close()
            self._pid = pid
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_pid"] = None
        state["_handle"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.window_ends, index)
        previous = 0 if episode == 0 else int(self.window_ends[episode - 1])
        local_start = int(index) - previous
        row = int(self.offsets[episode]) + local_start
        h5 = self._h5()
        pixels = torch.from_numpy(np.asarray(h5["pixels"][row : row + 4], dtype=np.uint8)).permute(
            0, 3, 1, 2
        ).contiguous()
        action = torch.zeros((4, 25), dtype=torch.float32)
        action[:3] = torch.from_numpy(
            np.asarray(h5["action_block"][row : row + 3], dtype=np.float32)
        )
        sample = self.transform({"pixels": pixels, "action": action})
        sample["clip_index"] = torch.tensor(index, dtype=torch.int64)
        sample["source_episode"] = torch.tensor(
            int(self.episode_ids[episode]), dtype=torch.int64
        )
        sample["source_step"] = torch.tensor(local_start, dtype=torch.int64)
        if tuple(sample["pixels"].shape) != (4, 3, 224, 224):
            raise RuntimeError(f"prepared play pixels transformed to {tuple(sample['pixels'].shape)}")
        if tuple(sample["action"].shape) != (4, 25):
            raise RuntimeError(f"prepared play actions transformed to {tuple(sample['action'].shape)}")
        if not torch.isfinite(sample["pixels"]).all() or not torch.isfinite(sample["action"]).all():
            raise RuntimeError("prepared play transformed window is non-finite")
        return sample


def split_clips_by_episode(
    dataset: Any,
    *,
    excluded_episodes: Sequence[int],
    train_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Deterministically split whole episodes and map them to clip indices."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train fraction must be in (0,1)")
    clip_episodes = np.fromiter(
        (int(value[0]) for value in dataset.clip_indices),
        dtype=np.int64,
        count=len(dataset.clip_indices),
    )
    excluded = np.asarray(sorted(set(map(int, excluded_episodes))), dtype=np.int64)
    eligible = np.setdiff1d(np.unique(clip_episodes), excluded)
    if len(eligible) < 2:
        raise RuntimeError("episode split needs at least two eligible episodes")
    shuffled = np.random.default_rng(seed).permutation(eligible)
    train_count = min(max(1, int(np.floor(len(shuffled) * train_fraction))), len(shuffled) - 1)
    train_episodes = np.sort(shuffled[:train_count])
    val_episodes = np.sort(shuffled[train_count:])
    train_ids = np.flatnonzero(np.isin(clip_episodes, train_episodes)).astype(np.int64)
    val_ids = np.flatnonzero(np.isin(clip_episodes, val_episodes)).astype(np.int64)
    if (
        not len(train_ids)
        or not len(val_ids)
        or np.intersect1d(train_episodes, val_episodes).size
        or np.intersect1d(np.unique(clip_episodes[train_ids]), excluded).size
        or np.intersect1d(np.unique(clip_episodes[val_ids]), excluded).size
    ):
        raise RuntimeError("episode split overlap/exclusion contract failed")
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
    }


class SourceClipDataset(Dataset):
    """Apply one source transform while retaining exact clip provenance."""

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int],
        transform: Callable[[dict[str, Any]], dict[str, Any]],
        source: str,
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.source = str(source)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_index = int(self.indices[index])
        sample = self.transform(self.dataset[clip_index])
        sample["clip_index"] = torch.tensor(clip_index, dtype=torch.int64)
        sample["source_episode"] = torch.tensor(
            int(self.dataset.clip_indices[clip_index][0]), dtype=torch.int64
        )
        if tuple(sample["pixels"].shape) != (NUM_FRAMES, 3, 224, 224):
            raise RuntimeError(
                f"{self.source} transformed pixels mismatch: {tuple(sample['pixels'].shape)}"
            )
        if tuple(sample["action"].shape) != (NUM_FRAMES, ACTION_DIM):
            raise RuntimeError(
                f"{self.source} transformed actions mismatch: {tuple(sample['action'].shape)}"
            )
        if not torch.isfinite(sample["pixels"]).all() or not torch.isfinite(
            sample["action"][: NUM_FRAMES - 1]
        ).all():
            raise RuntimeError(f"{self.source} transformed sample is non-finite")
        return sample


def _stack_samples(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples):
        raise RuntimeError("source sample keys differ inside a bundle")
    return {key: torch.stack([sample[key] for sample in samples]) for key in sorted(keys)}


class StrictPlayMixtureDataset(Dataset):
    """Bundle 5 expert and 3 play clips; batch 16 becomes exact 80/48."""

    def __init__(self, expert: Dataset, play: Dataset) -> None:
        if not len(expert) or not len(play):
            raise ValueError("expert/play mixture sources must both be non-empty")
        self.expert = expert
        self.play = play

    def __len__(self) -> int:
        return max(
            (len(self.expert) + BUNDLE_EXPERT - 1) // BUNDLE_EXPERT,
            (len(self.play) + BUNDLE_PLAY - 1) // BUNDLE_PLAY,
        )

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        expert = [
            self.expert[(index * BUNDLE_EXPERT + offset) % len(self.expert)]
            for offset in range(BUNDLE_EXPERT)
        ]
        play = [
            self.play[(index * BUNDLE_PLAY + offset) % len(self.play)]
            for offset in range(BUNDLE_PLAY)
        ]
        return {"expert": _stack_samples(expert), "play": _stack_samples(play)}


def flatten_bundled_source(source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if not torch.is_tensor(value) or value.ndim < 2:
            raise ValueError(f"bundled source {key!r} lacks bundle axes")
        result[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
    return result


def synthetic_contract_smoke() -> dict[str, Any]:
    """CPU proof of exact mixture and frozen-output-head dynamics contract."""

    exclusion_fixture = {
        **{name: 0 for name in EXCLUSION_ZERO_FIELDS},
        **EXCLUSION_FIXED_FIELDS,
        **{name: "measured provenance" for name in EXCLUSION_EXPLANATION_FIELDS},
    }
    _validate_exclusion_contract(exclusion_fixture)
    exclusion_negatives = 0
    for mutation in ("measurement1_overlap", "extra_field"):
        malformed = dict(exclusion_fixture)
        if mutation == "measurement1_overlap":
            malformed["quantized_signature_overlap_with_measurement1"] = 1
        else:
            malformed["unreviewed_field"] = 0
        try:
            _validate_exclusion_contract(malformed)
        except RuntimeError:
            exclusion_negatives += 1
        else:
            raise RuntimeError(f"exclusion mutation was accepted: {mutation}")

    class TinySource(Dataset):
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "pixels": torch.full((4, 3), float(index)),
                "action": torch.full((4, 25), float(index)),
            }

    mixed = StrictPlayMixtureDataset(TinySource(101), TinySource(67))
    batch = torch.utils.data.default_collate([mixed[index] for index in range(LOADER_BATCH)])
    expert = flatten_bundled_source(batch["expert"])
    play = flatten_bundled_source(batch["play"])
    if expert["pixels"].shape[0] != FORMAL_EXPERT_BATCH:
        raise RuntimeError("synthetic expert mixture is not 80")
    if play["pixels"].shape[0] != FORMAL_PLAY_BATCH:
        raise RuntimeError("synthetic play mixture is not 48")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(3, 4)
            self.projector = torch.nn.Linear(4, 4)
            self.predictor = torch.nn.Linear(8, 4)
            self.predictor_shadow = torch.nn.Linear(4, 4)
            self.action_encoder = torch.nn.Linear(2, 4)
            self.pred_proj = torch.nn.Sequential(
                torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)
            )

    model = TinyModel()
    freeze = freeze_dynamics_stack(model)
    if model.predictor_shadow.weight.requires_grad or model.predictor_shadow.bias.requires_grad:
        raise RuntimeError("component-boundary freeze accidentally enabled predictor_shadow")
    before = frozen_state_hashes(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    state = torch.randn(12, 3, 4)
    action = torch.randn(12, 3, 2)
    target = torch.randn(12, 3, 4)
    encoded = model.action_encoder(action)
    predicted = model.predictor(torch.cat((state, encoded), dim=-1))
    model.pred_proj.eval()
    predicted = model.pred_proj(predicted.reshape(-1, 4)).reshape(12, 3, 4)
    loss = (predicted - target).square().mean()
    loss.backward()
    optimizer.step()
    after = assert_frozen_hashes(model, before)
    nonzero_trainable_grad = sum(
        int(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if not nonzero_trainable_grad:
        raise RuntimeError("synthetic dynamics stack has no gradient")
    return {
        "status": "PASS",
        "batch": {"total": 128, "expert": 80, "play": 48},
        "trainable_prefixes": list(TRAINABLE_PREFIXES),
        "pred_proj_trainable": False,
        "frozen_hash_exact": before == after,
        "trainable_nonzero_gradient_elements": nonzero_trainable_grad,
        "exclusion_schema_positive": True,
        "exclusion_mutations_rejected": exclusion_negatives,
        "freeze": freeze,
    }


def parser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description=__doc__)
    sub = parser_.add_subparsers(dest="command", required=True)
    sub.add_parser("synthetic-smoke", help="run CPU-only contract proof")
    return parser_


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "synthetic-smoke":
        print(json.dumps(synthetic_contract_smoke(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
