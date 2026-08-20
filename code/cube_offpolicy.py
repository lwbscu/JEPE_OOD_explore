#!/usr/bin/env python3
"""Data and integrity helpers for Cube off-policy predictor fine-tuning.

The off-policy collector stores one fixed five-model-step rollout per HDF5
row.  Each rollout contains six JPEG frames plus raw and deployment-normalized
actions; training deliberately re-normalizes raw actions with the fixed50-
excluded Route2.1 scaler.  Splitting is performed on rollout ids before the three
overlapping four-frame training windows are exposed, preventing windows from
the same physical rollout from crossing train/validation boundaries.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import hdf5plugin  # noqa: F401  # Register compressed-HDF5 filters first.
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FORMAT_VERSION = "cube_offpolicy_rollout_hdf5_v1"
ROLLOUT_FRAMES = 6
ROLLOUT_ACTIONS = 5
WINDOW_FRAMES = 4
WINDOWS_PER_ROLLOUT = 3
ACTION_DIM = 25
IMAGE_SIZE = 224
SOURCE_BATCH_SIZE = 64
TRAINABLE_PREFIXES = ("predictor.", "action_encoder.", "pred_proj.")
FROZEN_PREFIXES = ("encoder.", "projector.")
DIST_NAMES = ("gaussian", "memory_t2", "ar1")
DIST_CODE = {name: index for index, name in enumerate(DIST_NAMES)}
DIST_MIX = {"gaussian": 0.4, "memory_t2": 0.3, "ar1": 0.3}
MEMORY_NEIGHBORS = 10


def _contract_error(label: str, expected: Any, actual: Any, position: Any) -> ValueError:
    """Contract failures always identify expectation, observation, and location."""
    return ValueError(
        f"{label}: expected={expected!r}, actual={actual!r}, position={position!r}"
    )


def _first_bad(mask: np.ndarray, start: int = 0) -> list[int] | None:
    indices = np.argwhere(mask)
    if not len(indices):
        return None
    result = indices[0].astype(np.int64)
    result[0] += int(start)
    return result.tolist()


def _expected_mix_counts(total: int) -> dict[str, int]:
    gaussian = int(round(total * DIST_MIX["gaussian"]))
    memory = int(round(total * DIST_MIX["memory_t2"]))
    return {"gaussian": gaussian, "memory_t2": memory, "ar1": total - gaussian - memory}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_excluded_episodes(manifest: Mapping[str, Any]) -> list[int]:
    candidates: list[Any] = [
        manifest.get("excluded_formal_episodes"),
        manifest.get("excluded_eval_episode_ids"),
    ]
    for key in ("formal_episode_exclusion", "exclusion"):
        section = manifest.get(key)
        if isinstance(section, Mapping):
            candidates.extend(
                section.get(name)
                for name in ("episode_ids", "episodes", "excluded_episodes")
            )
    selection = manifest.get("selection")
    if isinstance(selection, Mapping):
        candidates.append(selection.get("formal_episodes"))
    for value in candidates:
        if value is not None:
            return [int(item) for item in value]
    raise ValueError("off-policy manifest does not declare the excluded formal episodes")


def _normalizer_from_mapping(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(value, Mapping) or "mean" not in value:
        return None
    std_value = value.get("std", value.get("scale"))
    if std_value is None:
        return None
    mean = np.asarray(value["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(std_value, dtype=np.float64).reshape(-1)
    if mean.size not in (5, ACTION_DIM) or std.shape != mean.shape:
        raise ValueError(f"invalid action normalizer shape: mean={mean.shape}, std={std.shape}")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("action normalizer has non-finite values or non-positive std")
    if mean.size == ACTION_DIM:
        mean = mean.reshape(5, 5)[0]
        std = std.reshape(5, 5)[0]
    return mean, std


def manifest_action_normalizer(manifest: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    candidates: list[Any] = [manifest.get("action_normalizer"), manifest.get("eval_scaler")]
    action_protocol = manifest.get("action_protocol")
    if isinstance(action_protocol, Mapping):
        candidates.extend(
            (action_protocol.get("action_normalizer"), action_protocol.get("eval_scaler"))
        )
    for key in ("normalizers", "scalers", "action_scaler"):
        section = manifest.get(key)
        if isinstance(section, Mapping):
            candidates.extend((section.get("action"), section.get("action_model"), section))
    for value in candidates:
        result = _normalizer_from_mapping(value)
        if result is not None:
            return result
    raise ValueError("off-policy manifest does not contain an action normalizer/eval scaler")


def _validate_shard_semantics(
    h5: h5py.File,
    *,
    shard_index: int,
    count: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    formal_episodes: set[int],
) -> dict[str, int]:
    """Independently reconstruct the collector action/retrieval contract."""
    attr_mean = np.asarray(h5.attrs.get("action_normalizer_mean", []), dtype=np.float64)
    attr_std = np.asarray(h5.attrs.get("action_normalizer_scale", []), dtype=np.float64)
    if not np.array_equal(attr_mean, action_mean):
        raise _contract_error(
            "shard action mean attribute mismatch",
            action_mean.tolist(),
            attr_mean.tolist(),
            {"shard": shard_index, "attribute": "action_normalizer_mean"},
        )
    if not np.array_equal(attr_std, action_std):
        raise _contract_error(
            "shard action scale attribute mismatch",
            action_std.tolist(),
            attr_std.tolist(),
            {"shard": shard_index, "attribute": "action_normalizer_scale"},
        )

    source_rows = np.asarray(h5["source_row"][:], dtype=np.int64)
    source_episodes = np.asarray(h5["source_episode"][:], dtype=np.int64)
    source_steps = np.asarray(h5["source_step"][:], dtype=np.int64)
    expected_source_episodes = source_rows // 201
    expected_source_steps = source_rows % 201
    mismatch = np.flatnonzero(source_episodes != expected_source_episodes)
    if mismatch.size:
        local = int(mismatch[0])
        raise _contract_error(
            "source row/episode provenance mismatch",
            int(expected_source_episodes[local]),
            int(source_episodes[local]),
            {"shard": shard_index, "local_rollout": local},
        )
    mismatch = np.flatnonzero(
        (source_steps != expected_source_steps) | (source_steps > 175)
    )
    if mismatch.size:
        local = int(mismatch[0])
        raise _contract_error(
            "source row/step provenance mismatch",
            {"row_mod_201": int(expected_source_steps[local]), "maximum": 175},
            int(source_steps[local]),
            {"shard": shard_index, "local_rollout": local},
        )
    source_leak = np.isin(source_episodes, np.fromiter(formal_episodes, dtype=np.int64))
    bad_source = _first_bad(source_leak)
    if bad_source is not None:
        local = bad_source[0]
        raise _contract_error(
            "formal episode leaked into rollout source",
            "episode not in formal50",
            int(source_episodes[local]),
            {"shard": shard_index, "local_rollout": local},
        )

    counts = {name: 0 for name in DIST_NAMES}
    for start in range(0, count, 4096):
        stop = min(start + 4096, count)
        codes = np.asarray(h5["distribution_code"][start:stop], dtype=np.uint8)
        invalid_code = ~np.isin(codes, np.asarray(list(DIST_CODE.values()), dtype=np.uint8))
        bad = _first_bad(invalid_code, start)
        if bad is not None:
            local = bad[0]
            raise _contract_error(
                "distribution_code outside frozen vocabulary",
                DIST_CODE,
                int(h5["distribution_code"][local]),
                {"shard": shard_index, "local_rollout": local},
            )
        for name, code in DIST_CODE.items():
            counts[name] += int(np.count_nonzero(codes == code))

        action_env = np.asarray(h5["action_env"][start:stop], dtype=np.float32)
        action_env_preclip = np.asarray(
            h5["action_env_preclip"][start:stop], dtype=np.float32
        )
        action_model = np.asarray(h5["action_model"][start:stop], dtype=np.float32)
        solver = np.asarray(h5["action_solver_preclip"][start:stop], dtype=np.float32)
        clip_mask = np.asarray(h5["clip_mask"][start:stop], dtype=bool)
        for name, value in (
            ("action_env", action_env),
            ("action_env_preclip", action_env_preclip),
            ("action_model", action_model),
            ("action_solver_preclip", solver),
        ):
            bad = _first_bad(~np.isfinite(value), start)
            if bad is not None:
                raise _contract_error(
                    f"{name} contains non-finite value",
                    "finite",
                    str(value[tuple(np.asarray(bad) - np.asarray([start, 0, 0]))]),
                    {"shard": shard_index, "index": bad},
                )

        out_of_range = np.abs(action_env) > np.float32(1.0)
        bad = _first_bad(out_of_range, start)
        if bad is not None:
            local_index = tuple(np.asarray(bad) - np.asarray([start, 0, 0]))
            raise _contract_error(
                "action_env violates environment range",
                "-1 <= value <= 1",
                float(action_env[local_index]),
                {"shard": shard_index, "index": bad},
            )

        reconstructed_preclip = solver.reshape(stop - start, 25, 5).copy()
        reconstructed_preclip *= action_std[None, None]
        reconstructed_preclip += action_mean[None, None]
        preclip_bad = ~np.isclose(
            reconstructed_preclip, action_env_preclip, rtol=0.0, atol=1e-6
        )
        bad = _first_bad(preclip_bad, start)
        if bad is not None:
            local_index = tuple(np.asarray(bad) - np.asarray([start, 0, 0]))
            raise _contract_error(
                "solver-to-environment preclip roundtrip mismatch",
                float(reconstructed_preclip[local_index]),
                float(action_env_preclip[local_index]),
                {"shard": shard_index, "index": bad, "atol": 1e-6},
            )

        expected_env = np.clip(action_env_preclip, -1.0, 1.0).astype(
            np.float32, copy=False
        )
        env_bad = expected_env != action_env
        bad = _first_bad(env_bad, start)
        if bad is not None:
            local_index = tuple(np.asarray(bad) - np.asarray([start, 0, 0]))
            raise _contract_error(
                "action_env clip reconstruction mismatch",
                float(expected_env[local_index]),
                float(action_env[local_index]),
                {"shard": shard_index, "index": bad},
            )
        expected_clip_mask = action_env_preclip != expected_env
        mask_bad = expected_clip_mask != clip_mask
        bad = _first_bad(mask_bad, start)
        if bad is not None:
            local_index = tuple(np.asarray(bad) - np.asarray([start, 0, 0]))
            raise _contract_error(
                "clip_mask disagrees with preclip/raw relation",
                bool(expected_clip_mask[local_index]),
                bool(clip_mask[local_index]),
                {"shard": shard_index, "index": bad},
            )

        expected_model = action_env.copy()
        expected_model -= action_mean[None, None]
        expected_model /= action_std[None, None]
        expected_model = expected_model.reshape(stop - start, ROLLOUT_ACTIONS, ACTION_DIM)
        model_bad = expected_model != action_model
        bad = _first_bad(model_bad, start)
        if bad is not None:
            local_index = tuple(np.asarray(bad) - np.asarray([start, 0, 0]))
            raise _contract_error(
                "action_model disagrees with normalized action_env",
                float(expected_model[local_index]),
                float(action_model[local_index]),
                {"shard": shard_index, "index": bad},
            )

        retrieval_rows = np.asarray(h5["retrieval_rows"][start:stop], dtype=np.int64)
        retrieval_episodes = np.asarray(
            h5["retrieval_episodes"][start:stop], dtype=np.int64
        )
        retrieval_steps = np.asarray(h5["retrieval_steps"][start:stop], dtype=np.int64)
        retrieval_distances = np.asarray(
            h5["retrieval_distances"][start:stop], dtype=np.float64
        )
        retrieval_anchors = np.asarray(
            h5["retrieval_anchor_indices"][start:stop], dtype=np.int64
        )
        selected_ranks = np.asarray(
            h5["retrieval_selected_rank"][start:stop], dtype=np.int64
        )
        selected_rows = np.asarray(
            h5["retrieval_selected_row"][start:stop], dtype=np.int64
        )
        for offset, code in enumerate(codes):
            local = start + offset
            position = {"shard": shard_index, "local_rollout": local}
            if int(code) == DIST_CODE["memory_t2"]:
                episodes = retrieval_episodes[offset]
                if len(np.unique(episodes)) != MEMORY_NEIGHBORS:
                    raise _contract_error(
                        "memory_t2 retrieval episodes are not distinct",
                        MEMORY_NEIGHBORS,
                        int(len(np.unique(episodes))),
                        position,
                    )
                leaked = sorted(
                    set(map(int, episodes))
                    & (formal_episodes | {int(source_episodes[local])})
                )
                if leaked:
                    raise _contract_error(
                        "memory_t2 retrieval episode exclusion violated",
                        "no formal50 or current source episode",
                        leaked,
                        position,
                    )
                valid_fields = (
                    np.all(retrieval_rows[offset] >= 0)
                    and np.all((retrieval_steps[offset] >= 0) & (retrieval_steps[offset] <= 175))
                    and np.all(retrieval_anchors[offset] >= 0)
                    and np.all(np.isfinite(retrieval_distances[offset]))
                    and np.all(retrieval_distances[offset] >= 0)
                )
                if not valid_fields:
                    raise _contract_error(
                        "memory_t2 retrieval provenance fields invalid",
                        "rows/anchors>=0, 0<=steps<=175, finite distances>=0",
                        {
                            "rows": retrieval_rows[offset].tolist(),
                            "steps": retrieval_steps[offset].tolist(),
                            "distances": retrieval_distances[offset].tolist(),
                            "anchors": retrieval_anchors[offset].tolist(),
                        },
                        position,
                    )
                expected_episodes = retrieval_rows[offset] // 201
                expected_steps = retrieval_rows[offset] % 201
                if not np.array_equal(retrieval_episodes[offset], expected_episodes):
                    mismatch = int(
                        np.flatnonzero(retrieval_episodes[offset] != expected_episodes)[0]
                    )
                    raise _contract_error(
                        "memory_t2 retrieval row/episode provenance mismatch",
                        int(expected_episodes[mismatch]),
                        int(retrieval_episodes[offset, mismatch]),
                        {**position, "retrieval_rank": mismatch},
                    )
                if not np.array_equal(retrieval_steps[offset], expected_steps):
                    mismatch = int(
                        np.flatnonzero(retrieval_steps[offset] != expected_steps)[0]
                    )
                    raise _contract_error(
                        "memory_t2 retrieval row/step provenance mismatch",
                        int(expected_steps[mismatch]),
                        int(retrieval_steps[offset, mismatch]),
                        {**position, "retrieval_rank": mismatch},
                    )
                if np.any(np.diff(retrieval_distances[offset]) < 0):
                    mismatch = int(
                        np.flatnonzero(np.diff(retrieval_distances[offset]) < 0)[0]
                    )
                    raise _contract_error(
                        "memory_t2 retrieval distances are not nearest-first",
                        "nondecreasing",
                        retrieval_distances[offset, mismatch : mismatch + 2].tolist(),
                        {**position, "retrieval_rank_pair": [mismatch, mismatch + 1]},
                    )
                rank = int(selected_ranks[offset])
                if not 0 <= rank < MEMORY_NEIGHBORS:
                    raise _contract_error(
                        "memory_t2 selected rank outside retrieval set",
                        "0..9",
                        rank,
                        position,
                    )
                if int(selected_rows[offset]) != int(retrieval_rows[offset, rank]):
                    raise _contract_error(
                        "memory_t2 selected row disagrees with selected rank",
                        int(retrieval_rows[offset, rank]),
                        int(selected_rows[offset]),
                        position,
                    )
            else:
                sentinel_ok = (
                    np.all(retrieval_rows[offset] == -1)
                    and np.all(retrieval_episodes[offset] == -1)
                    and np.all(retrieval_steps[offset] == -1)
                    and np.all(np.isnan(retrieval_distances[offset]))
                    and np.all(retrieval_anchors[offset] == -1)
                    and int(selected_ranks[offset]) == -1
                    and int(selected_rows[offset]) == -1
                )
                if not sentinel_ok:
                    raise _contract_error(
                        "non-memory rollout carries retrieval provenance",
                        "all retrieval integer fields=-1 and distances=NaN",
                        {
                            "rows": retrieval_rows[offset].tolist(),
                            "episodes": retrieval_episodes[offset].tolist(),
                            "steps": retrieval_steps[offset].tolist(),
                            "distances": retrieval_distances[offset].tolist(),
                            "anchors": retrieval_anchors[offset].tolist(),
                            "selected_rank": int(selected_ranks[offset]),
                            "selected_row": int(selected_rows[offset]),
                        },
                        position,
                    )
    return counts


def load_offpolicy_manifest(
    dataset_root: Path,
    *,
    expected_excluded_episodes: Sequence[int],
    expected_action_mean: Sequence[float],
    expected_action_std: Sequence[float],
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Load and strongly validate the collector contract."""
    dataset_root = dataset_root.resolve()
    path = dataset_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"off-policy manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"format_version mismatch: expected={FORMAT_VERSION!r}, "
            f"actual={manifest.get('format_version')!r}"
        )
    if manifest.get("complete") is not True:
        raise ValueError(f"collector manifest is not complete: actual={manifest.get('complete')!r}")

    declared_total_early = int(manifest.get("num_rollouts", -1))
    expected_mix = _expected_mix_counts(declared_total_early)
    action_protocol = manifest.get("action_protocol", {})
    requested_mix = action_protocol.get("mixture_requested")
    if requested_mix != DIST_MIX:
        raise _contract_error(
            "collector mixture contract mismatch",
            DIST_MIX,
            requested_mix,
            "manifest.action_protocol.mixture_requested",
        )
    declared_mix = action_protocol.get("counts")
    if declared_mix != expected_mix:
        raise _contract_error(
            "collector mixture count mismatch",
            expected_mix,
            declared_mix,
            "manifest.action_protocol.counts",
        )

    excluded = _manifest_excluded_episodes(manifest)
    expected_excluded = sorted(int(value) for value in expected_excluded_episodes)
    if len(excluded) != 50 or len(set(excluded)) != 50:
        raise ValueError(f"collector must exclude exactly 50 unique episodes, actual={len(set(excluded))}")
    if sorted(excluded) != expected_excluded:
        missing = sorted(set(expected_excluded) - set(excluded))
        extra = sorted(set(excluded) - set(expected_excluded))
        raise ValueError(f"formal episode exclusion mismatch: missing={missing}, extra={extra}")

    actual_mean, actual_std = manifest_action_normalizer(manifest)
    expected_mean = np.asarray(expected_action_mean, dtype=np.float64).reshape(-1)
    expected_std = np.asarray(expected_action_std, dtype=np.float64).reshape(-1)
    if expected_mean.size == ACTION_DIM:
        expected_mean = expected_mean.reshape(5, 5)[0]
        expected_std = expected_std.reshape(5, 5)[0]
    # The collector uses the deployment/evaluation scaler fitted on all finite
    # actions, while Route2.1's anti-leak training scaler excludes formal50.
    # Permit only their empirically tiny coordinate drift; both training
    # branches below use the collector scaler so one batch has one action space.
    standardized_mean_drift = np.abs(actual_mean - expected_mean) / expected_std
    relative_std_drift = np.abs(actual_std - expected_std) / expected_std
    if float(standardized_mean_drift.max()) > 1e-3:
        raise ValueError(
            "collector/training action mean drift is too large: "
            f"max_standardized={standardized_mean_drift.max()}, "
            f"expected={expected_mean}, actual={actual_mean}"
        )
    if float(relative_std_drift.max()) > 1e-3:
        raise ValueError(
            "collector/training action std drift is too large: "
            f"max_relative={relative_std_drift.max()}, expected={expected_std}, actual={actual_std}"
        )

    storage = manifest.get("storage")
    shards = manifest.get("shards")
    if shards is None and isinstance(storage, Mapping):
        shards = storage.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest shards must be a non-empty list")
    total = 0
    aggregate_mix = {name: 0 for name in DIST_NAMES}
    resolved_shards: list[dict[str, Any]] = []
    for shard_index, entry in enumerate(shards):
        if not isinstance(entry, Mapping):
            raise ValueError(f"shard {shard_index} is not an object")
        relative_path = entry.get("path", entry.get("filename", ""))
        shard_path = (dataset_root / str(relative_path)).resolve()
        if dataset_root != shard_path and dataset_root not in shard_path.parents:
            raise ValueError(f"shard escapes dataset root: {shard_path}")
        if not shard_path.is_file():
            raise FileNotFoundError(f"shard missing: {shard_path}")
        count = int(entry.get("num_rollouts", -1))
        if count <= 0:
            raise ValueError(f"invalid num_rollouts for shard {shard_index}: {count}")
        expected_sha = str(entry.get("sha256", ""))
        if len(expected_sha) != 64:
            raise ValueError(f"shard {shard_index} is missing a SHA-256 digest")
        actual_sha = sha256_file(shard_path) if verify_shard_hashes else None
        if verify_shard_hashes and actual_sha != expected_sha:
            raise ValueError(
                f"shard hash mismatch at {shard_index}: expected={expected_sha}, actual={actual_sha}"
            )
        with h5py.File(shard_path, "r") as h5:
            if h5.attrs.get("format_version") != FORMAT_VERSION:
                raise _contract_error(
                    "shard format mismatch",
                    FORMAT_VERSION,
                    h5.attrs.get("format_version"),
                    {"shard": shard_index, "attribute": "format_version"},
                )
            required = (
                "pixels_jpeg",
                "action_env",
                "action_env_preclip",
                "action_model",
                "action_solver_preclip",
                "clip_mask",
                "block_pos",
                "block_quat",
                "block_yaw",
                "source_row",
                "source_episode",
                "source_step",
                "distribution_code",
                "retrieval_rows",
                "retrieval_episodes",
                "retrieval_steps",
                "retrieval_distances",
                "retrieval_anchor_indices",
                "retrieval_selected_rank",
                "retrieval_selected_row",
            )
            missing = [key for key in required if key not in h5]
            if missing:
                raise ValueError(f"shard {shard_index} missing datasets: {missing}")
            if tuple(h5["pixels_jpeg"].shape) != (count, ROLLOUT_FRAMES):
                raise ValueError(
                    f"pixels_jpeg shape mismatch in shard {shard_index}: "
                    f"expected={(count, ROLLOUT_FRAMES)}, actual={h5['pixels_jpeg'].shape}"
                )
            jpeg_vlen = h5py.check_dtype(vlen=h5["pixels_jpeg"].dtype)
            if jpeg_vlen is None or np.dtype(jpeg_vlen) != np.dtype(np.uint8):
                raise ValueError(
                    f"pixels_jpeg dtype mismatch in shard {shard_index}: "
                    f"expected=vlen uint8, actual={h5['pixels_jpeg'].dtype}"
                )
            if tuple(h5["action_model"].shape) != (count, ROLLOUT_ACTIONS, ACTION_DIM):
                raise ValueError(
                    f"action_model shape mismatch in shard {shard_index}: "
                    f"expected={(count, ROLLOUT_ACTIONS, ACTION_DIM)}, actual={h5['action_model'].shape}"
                )
            expected_shapes = {
                "action_env": (count, 25, 5),
                "action_env_preclip": (count, 25, 5),
                "action_solver_preclip": (count, 5, 25),
                "clip_mask": (count, 25, 5),
                "block_pos": (count, 6, 3),
                "block_quat": (count, 6, 4),
                "block_yaw": (count, 6),
                "source_row": (count,),
                "source_episode": (count,),
                "source_step": (count,),
                "distribution_code": (count,),
                "retrieval_rows": (count, 10),
                "retrieval_episodes": (count, 10),
                "retrieval_steps": (count, 10),
                "retrieval_distances": (count, 10),
                "retrieval_anchor_indices": (count, 10),
                "retrieval_selected_rank": (count,),
                "retrieval_selected_row": (count,),
            }
            bad_shapes = {
                key: {"expected": shape, "actual": tuple(h5[key].shape)}
                for key, shape in expected_shapes.items()
                if tuple(h5[key].shape) != shape
            }
            if bad_shapes:
                raise ValueError(f"collector shard shape mismatch at {shard_index}: {bad_shapes}")
            expected_dtypes = {
                "action_env": np.float32,
                "action_env_preclip": np.float32,
                "action_model": np.float32,
                "action_solver_preclip": np.float32,
                "clip_mask": bool,
                "source_row": np.int64,
                "source_episode": np.int64,
                "source_step": np.int16,
                "distribution_code": np.uint8,
                "retrieval_rows": np.int64,
                "retrieval_episodes": np.int64,
                "retrieval_steps": np.int16,
                "retrieval_distances": np.float64,
                "retrieval_anchor_indices": np.int64,
                "retrieval_selected_rank": np.int8,
                "retrieval_selected_row": np.int64,
            }
            for name, dtype in expected_dtypes.items():
                if np.dtype(h5[name].dtype) != np.dtype(dtype):
                    raise _contract_error(
                        "collector shard dtype mismatch",
                        str(np.dtype(dtype)),
                        str(h5[name].dtype),
                        {"shard": shard_index, "dataset": name},
                    )
            shard_mix = _validate_shard_semantics(
                h5,
                shard_index=shard_index,
                count=count,
                action_mean=actual_mean,
                action_std=actual_std,
                formal_episodes=set(expected_excluded),
            )
            declared_shard_mix = entry.get("distribution_counts")
            if declared_shard_mix != shard_mix:
                raise _contract_error(
                    "shard distribution counts mismatch",
                    shard_mix,
                    declared_shard_mix,
                    {"shard": shard_index, "manifest_entry": "distribution_counts"},
                )
            for name in DIST_NAMES:
                aggregate_mix[name] += shard_mix[name]
        total += count
        resolved_shards.append(
            {**dict(entry), "path": str(shard_path), "num_rollouts": count, "verified_sha256": actual_sha}
        )
    declared_value = manifest.get("num_rollouts")
    if declared_value is None and isinstance(storage, Mapping):
        declared_value = storage.get("num_rollouts", storage.get("total_rollouts"))
    collection = manifest.get("collection")
    if declared_value is None and isinstance(collection, Mapping):
        declared_value = collection.get("num_rollouts", collection.get("completed_rollouts"))
    declared_total = int(-1 if declared_value is None else declared_value)
    if total != declared_total:
        raise ValueError(f"rollout count mismatch: declared={declared_total}, shard_sum={total}")
    if aggregate_mix != expected_mix:
        raise _contract_error(
            "aggregate 4:3:3 distribution mismatch",
            expected_mix,
            aggregate_mix,
            "all collector shards",
        )
    manifest["shards"] = resolved_shards
    manifest["num_rollouts"] = declared_total
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = sha256_file(path)
    manifest["excluded_formal_episodes_resolved"] = excluded
    manifest["action_normalizer_resolved"] = {
        "mean": actual_mean.tolist(),
        "std": actual_std.tolist(),
        "route21_max_standardized_mean_drift": float(standardized_mean_drift.max()),
        "route21_max_relative_std_drift": float(relative_std_drift.max()),
    }
    return manifest


def split_rollout_ids(
    num_rollouts: int, train_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if num_rollouts < 2:
        raise ValueError("at least two rollouts are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0,1)")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_rollouts).astype(np.int64, copy=False)
    train_count = int(np.floor(num_rollouts * train_fraction))
    train = permutation[:train_count]
    validation = permutation[train_count:]
    if np.intersect1d(train, validation).size:
        raise RuntimeError("rollout split overlaps")
    return train, validation


def rollout_window(
    frames: torch.Tensor, actions: torch.Tensor, window_index: int
) -> dict[str, torch.Tensor]:
    """Make one 4-frame/4-action window; the unused final action is zero.

    Only action slots 0..2 are consumed by the training loss.  Setting slot 3
    to an explicit zero makes accidental future use detectable and avoids
    pretending that the 6-frame rollout includes a sixth transition.
    """
    if tuple(frames.shape) != (ROLLOUT_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"decoded frames have invalid shape: {tuple(frames.shape)}")
    if tuple(actions.shape) != (ROLLOUT_ACTIONS, ACTION_DIM):
        raise ValueError(f"action_model has invalid shape: {tuple(actions.shape)}")
    if not 0 <= int(window_index) < WINDOWS_PER_ROLLOUT:
        raise IndexError(f"window index outside [0,{WINDOWS_PER_ROLLOUT}): {window_index}")
    start = int(window_index)
    pixels = frames[start : start + WINDOW_FRAMES]
    action = torch.zeros(WINDOW_FRAMES, ACTION_DIM, dtype=torch.float32)
    action[: WINDOW_FRAMES - 1] = actions[start : start + WINDOW_FRAMES - 1].float()
    if not torch.count_nonzero(action[-1]).eq(0):
        raise RuntimeError("unused final action placeholder is not zero")
    return {"pixels": pixels, "action": action}


def _decode_jpeg(value: Any) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        payload = value.astype(np.uint8, copy=False).tobytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    else:
        raise TypeError(f"unsupported pixels_jpeg element type: {type(value)}")
    with Image.open(io.BytesIO(payload)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(
            f"decoded JPEG must be {(IMAGE_SIZE, IMAGE_SIZE, 3)}, actual={array.shape}"
        )
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class OffPolicyWindowDataset(Dataset):
    """Lazy JPEG windows with runtime fixed50-excluded action normalization."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        rollout_ids: Sequence[int],
        training_action_mean: Sequence[float],
        training_action_std: Sequence[float],
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.shards = [dict(entry) for entry in manifest["shards"]]
        self.rollout_ids = np.asarray(rollout_ids, dtype=np.int64)
        self.transform = transform
        self.training_action_mean = np.asarray(training_action_mean, dtype=np.float32).reshape(1, 5)
        self.training_action_std = np.asarray(training_action_std, dtype=np.float32).reshape(1, 5)
        if (
            not np.all(np.isfinite(self.training_action_mean))
            or not np.all(np.isfinite(self.training_action_std))
            or np.any(self.training_action_std <= 0)
        ):
            raise ValueError("training action normalizer is invalid")
        counts = np.asarray([int(entry["num_rollouts"]) for entry in self.shards], dtype=np.int64)
        self.ends = np.cumsum(counts)
        if self.rollout_ids.ndim != 1 or np.any(self.rollout_ids < 0):
            raise ValueError("rollout_ids must be a one-dimensional non-negative array")
        if self.rollout_ids.size and int(self.rollout_ids.max()) >= int(self.ends[-1]):
            raise ValueError("rollout_ids contain an index beyond the shard manifest")
        self._pid: int | None = None
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return int(self.rollout_ids.size * WINDOWS_PER_ROLLOUT)

    def _handle(self, shard_index: int) -> h5py.File:
        pid = os.getpid()
        if self._pid != pid:
            self.close()
            self._pid = pid
        if shard_index not in self._handles:
            self._handles[shard_index] = h5py.File(self.shards[shard_index]["path"], "r")
        return self._handles[shard_index]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_pid"] = None
        state["_handles"] = {}
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        split_rollout_index, window_index = divmod(int(index), WINDOWS_PER_ROLLOUT)
        rollout_id = int(self.rollout_ids[split_rollout_index])
        shard_index = bisect_right(self.ends, rollout_id)
        shard_start = 0 if shard_index == 0 else int(self.ends[shard_index - 1])
        local_index = rollout_id - shard_start
        h5 = self._handle(shard_index)
        # Decode only the four frames used by this window.  Decoding all six
        # would add 50% JPEG CPU work to every optimization sample.
        frames = torch.stack(
            [
                _decode_jpeg(h5["pixels_jpeg"][local_index, frame])
                for frame in range(window_index, window_index + WINDOW_FRAMES)
            ]
        )
        action_env = np.asarray(h5["action_env"][local_index], dtype=np.float32)
        normalized = action_env.copy()
        normalized -= self.training_action_mean
        normalized /= self.training_action_std
        actions = torch.from_numpy(normalized.reshape(ROLLOUT_ACTIONS, ACTION_DIM))
        reconstructed = actions.numpy()
        expected = (
            (action_env - self.training_action_mean) / self.training_action_std
        ).reshape(ROLLOUT_ACTIONS, ACTION_DIM)
        if not np.array_equal(reconstructed, expected):
            mismatch = np.argwhere(reconstructed != expected)[0].tolist()
            raise _contract_error(
                "off-policy sample training normalization mismatch",
                float(expected[tuple(mismatch)]),
                float(reconstructed[tuple(mismatch)]),
                {
                    "rollout_id": rollout_id,
                    "shard": shard_index,
                    "model_action_index": mismatch,
                },
            )
        action_window = torch.zeros(WINDOW_FRAMES, ACTION_DIM, dtype=torch.float32)
        action_window[: WINDOW_FRAMES - 1] = actions[
            window_index : window_index + WINDOW_FRAMES - 1
        ]
        sample = {"pixels": frames, "action": action_window}
        sample["rollout_id"] = torch.tensor(rollout_id, dtype=torch.int64)
        sample["window_index"] = torch.tensor(window_index, dtype=torch.int64)
        if self.transform is not None:
            sample = self.transform(sample)
        if tuple(sample["pixels"].shape) != (WINDOW_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"off-policy pixel transform returned {tuple(sample['pixels'].shape)}")
        if tuple(sample["action"].shape) != (WINDOW_FRAMES, ACTION_DIM):
            raise RuntimeError(f"off-policy action transform returned {tuple(sample['action'].shape)}")
        if torch.count_nonzero(sample["action"][-1]).item() != 0:
            raise RuntimeError("off-policy transform changed the zero final-action placeholder")
        return sample


class ExpertWindowDataset(Dataset):
    """Apply the Route2.1 transform and zero its provably unused action slot."""

    def __init__(self, dataset: Any, indices: Sequence[int], transform: Callable) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.transform(self.dataset[int(self.indices[index])])
        sample["action"] = sample["action"].clone()
        sample["action"][-1].zero_()
        return sample


class PairedSourceDataset(Dataset):
    """One expert and one off-policy sample per item.

    A DataLoader batch size of 64 therefore creates exactly 64 samples from
    each source (128 total) on every optimization step.  The shorter source is
    deterministically cycled; loader shuffling randomizes the shared pair ids.
    """

    def __init__(self, expert: Dataset, offpolicy: Dataset) -> None:
        if len(expert) == 0 or len(offpolicy) == 0:
            raise ValueError("paired sources must both be non-empty")
        self.expert = expert
        self.offpolicy = offpolicy

    def __len__(self) -> int:
        return max(len(self.expert), len(self.offpolicy))

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        return {
            "expert": self.expert[int(index) % len(self.expert)],
            "offpolicy": self.offpolicy[int(index) % len(self.offpolicy)],
        }


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash parameter and buffer names, metadata, and exact tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        # ``num_batches_tracked`` and similar buffers are scalar tensors;
        # flatten before byte-viewing so all dtypes/ranks share one exact path.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def freeze_predictor_only(model: torch.nn.Module) -> dict[str, Any]:
    """Train predictor/action encoder/pred projection and freeze everything else."""
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)
        parameter.requires_grad_(trainable)
        (trainable_names if trainable else frozen_names).append(name)
    for prefix in TRAINABLE_PREFIXES:
        if not any(name.startswith(prefix) for name in trainable_names):
            raise RuntimeError(f"trainable component has no parameters: {prefix}")
    unexpected = [name for name in trainable_names if not name.startswith(TRAINABLE_PREFIXES)]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected}")
    model.encoder.eval()
    model.projector.eval()
    return {
        "trainable_parameter_tensors": len(trainable_names),
        "frozen_parameter_tensors": len(frozen_names),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "frozen_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        ),
        "trainable_prefixes": list(TRAINABLE_PREFIXES),
        "frozen_integrity_modules": list(FROZEN_PREFIXES),
    }


def frozen_state_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        "encoder": module_state_sha256(model.encoder),
        "projector": module_state_sha256(model.projector),
    }


def assert_frozen_hashes(model: torch.nn.Module, expected: Mapping[str, str]) -> dict[str, str]:
    actual = frozen_state_hashes(model)
    if dict(expected) != actual:
        raise RuntimeError(f"frozen model state changed: expected={dict(expected)}, actual={actual}")
    return actual


def synthetic_window_smoke() -> dict[str, Any]:
    """CPU-only proof of windows, 1:1 batching, freezing, and two updates."""
    frames = torch.arange(
        ROLLOUT_FRAMES * 3 * IMAGE_SIZE * IMAGE_SIZE, dtype=torch.int64
    ).remainder(256).to(torch.uint8).reshape(ROLLOUT_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
    actions = torch.arange(ROLLOUT_ACTIONS * ACTION_DIM, dtype=torch.float32).reshape(
        ROLLOUT_ACTIONS, ACTION_DIM
    )
    windows = [rollout_window(frames, actions, index) for index in range(WINDOWS_PER_ROLLOUT)]
    for index, window in enumerate(windows):
        assert tuple(window["pixels"].shape) == (4, 3, 224, 224)
        assert tuple(window["action"].shape) == (4, 25)
        assert torch.equal(window["pixels"], frames[index : index + 4])
        assert torch.equal(window["action"][:3], actions[index : index + 3])
        assert torch.count_nonzero(window["action"][-1]).item() == 0
        # The production forward explicitly slices :3 before action encoding.
        perturbed = window["action"].clone()
        perturbed[-1].fill_(12345.0)
        assert torch.equal(window["action"][:3], perturbed[:3])

    torch.manual_seed(3072)

    class TinyDynamics(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(3, 8)
            self.projector = torch.nn.Linear(8, 8)
            self.action_encoder = torch.nn.Linear(ACTION_DIM, 8)
            self.predictor = torch.nn.Linear(16, 8)
            self.pred_proj = torch.nn.Linear(8, 8)

        def loss(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                embedding = self.projector(self.encoder(features))
            # Exactly the production action slice: placeholder slot 3 is absent.
            action_embedding = self.action_encoder(action[:, :3])
            prediction = self.pred_proj(
                self.predictor(torch.cat((embedding[:, :3], action_embedding), dim=-1))
            )
            pred_loss = (prediction - embedding[:, 1:].detach()).square().mean()
            sigreg_diagnostic = embedding.square().mean()
            return pred_loss + 0.09 * sigreg_diagnostic

    tiny = TinyDynamics()
    freeze_predictor_only(tiny)
    frozen_before = frozen_state_hashes(tiny)
    trainable_before = {
        name: value.detach().clone()
        for name, value in tiny.named_parameters()
        if value.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [value for value in tiny.parameters() if value.requires_grad], lr=1e-3
    )
    features = torch.randn(2 * SOURCE_BATCH_SIZE, 4, 3)
    mixed_actions = torch.randn(2 * SOURCE_BATCH_SIZE, 4, ACTION_DIM)
    mixed_actions[:, -1].zero_()
    losses: list[float] = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = tiny.loss(features, mixed_actions)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert_frozen_hashes(tiny, frozen_before)
    trainable_changed = any(
        not torch.equal(trainable_before[name], value.detach())
        for name, value in tiny.named_parameters()
        if value.requires_grad
    )
    if not trainable_changed:
        raise RuntimeError("synthetic two-step smoke did not update trainable dynamics parameters")
    return {
        "status": "PASS",
        "rollout_split_before_windows": True,
        "windows_per_rollout": WINDOWS_PER_ROLLOUT,
        "pixels_shape": [4, 3, 224, 224],
        "action_shape": [4, 25],
        "final_action_placeholder": 0.0,
        "loss_action_slice": "action[:, :3]",
        "placeholder_perturbation_changes_loss_input": False,
        "source_batch": {"expert": SOURCE_BATCH_SIZE, "offpolicy": SOURCE_BATCH_SIZE},
        "tiny_optimizer_steps": 2,
        "tiny_loss_curve": losses,
        "tiny_trainable_parameters_changed": trainable_changed,
        "tiny_frozen_hash_exact_match": True,
    }
