#!/usr/bin/env python3
"""Collect fixed-horizon off-policy Cube dynamics data on the data disk.

The collector starts MuJoCo from immutable expert HDF5 states, while globally
excluding the 50 formal-evaluation episodes.  Every rollout executes exactly
25 environment actions and stores the six model-rate frames (t, t+5, ...,
t+25) as deterministic JPEG byte arrays in atomic HDF5 shards.

Three action distributions are frozen at a 4:3:3 ratio:

* Gaussian: iid solver-space N(0, 1), inverse-scaled, then raw clipped.
* T2 local: uniformly choose one of the state-nearest ten distinct-episode
  memory blocks, perturb raw actions by sigma=0.1, then clip.
* AR(1): stationary solver-space Gaussian noise with rho=0.8, inverse-scaled,
  then raw clipped.

``hdf5plugin`` is deliberately imported before ``h5py`` in every HDF5 entry
point.  The script has no dependency beyond packages already used by ailab.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


AILAB_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_cube_memory_index as memory  # noqa: E402


DATASET = AILAB_ROOT / "datasets/ogbench/cube_single_expert.h5"
OUTPUT_ROOT = AILAB_ROOT / "datasets/offpolicy_cube_v1"
SMOKE_ROOT = OUTPUT_ROOT / "smoke_200"
INDEX_ROOT = AILAB_ROOT / "outputs/memory_index/cube_expert_v1"
FORMAL_MANIFEST = AILAB_ROOT / "outputs/audit/cube_cem_manifest.json"
TMP_ROOT = AILAB_ROOT.parent / "tmp"

FORMAT_VERSION = "cube_offpolicy_rollout_hdf5_v1"
SEED = 424_200
FORMAL_COUNT = 30_000
SMOKE_COUNT = 200
SHARD_SIZE = 250
ENV_STEPS = 25
MODEL_STEPS = 5
ACTION_DIM = 5
FRAME_COUNT = 6
MAX_START_STEP = 175
MEMORY_NEIGHBORS = 10
MEMORY_SIGMA = 0.1
AR1_RHO = 0.8
JPEG_QUALITY = 95
MAX_BYTES = 40 * (1 << 30)
MIX = {"gaussian": 0.4, "memory_t2": 0.3, "ar1": 0.3}
DIST_NAMES = ("gaussian", "memory_t2", "ar1")
DIST_CODE = {name: idx for idx, name in enumerate(DIST_NAMES)}


def _configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(AILAB_ROOT),
        "HF_HOME": str(AILAB_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        os.environ[key] = value
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


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


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, include_sha256: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = _sha256(resolved)
    return result


def _assert_output(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise ValueError(f"refusing symlink output: {lexical}")
    resolved = lexical.resolve()
    root = OUTPUT_ROOT.resolve()
    # Formal data live at OUTPUT_ROOT; smoke is its concrete child.
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output must be {root} or a child: {resolved}")
    if AILAB_ROOT.parent.resolve() not in resolved.parents:
        raise ValueError(f"output is not on /root/autodl-tmp: {resolved}")
    return resolved


def _file_system_free(path: Path) -> int:
    probe = path if path.exists() else path.parent
    return int(shutil.disk_usage(probe).free)


def _mix_counts(total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError(f"rollout count must be positive: {total}")
    gaussian = int(round(total * MIX["gaussian"]))
    memory_count = int(round(total * MIX["memory_t2"]))
    ar1 = total - gaussian - memory_count
    counts = {"gaussian": gaussian, "memory_t2": memory_count, "ar1": ar1}
    if min(counts.values()) < 0 or sum(counts.values()) != total:
        raise RuntimeError(f"invalid mixture allocation: total={total}, counts={counts}")
    return counts


def _formal_episodes(h5: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    current = DATASET.resolve().stat()
    frozen_dataset = manifest.get("dataset", {})
    if (
        Path(frozen_dataset.get("path", "")).resolve() != DATASET.resolve()
        or int(frozen_dataset.get("size_bytes", -1)) != int(current.st_size)
        or int(frozen_dataset.get("mtime_ns", -1)) != int(current.st_mtime_ns)
    ):
        raise RuntimeError(
            "formal manifest dataset identity mismatch: "
            f"expected={frozen_dataset}, "
            f"actual={{'path': '{DATASET.resolve()}', 'size_bytes': {current.st_size}, "
            f"'mtime_ns': {current.st_mtime_ns}}}"
        )
    rows = np.asarray(manifest["formal_rows"], dtype=np.int64)
    if rows.shape != (50,) or len(np.unique(rows)) != 50:
        raise RuntimeError(
            "formal manifest mismatch: expected_shape=(50,), "
            f"actual_shape={rows.shape}, unique={len(np.unique(rows))}"
        )
    episodes = np.asarray(h5["ep_idx"][np.sort(rows)], dtype=np.int64)
    # Restore row order after h5py's sorted gather requirement.
    episodes = episodes[np.argsort(np.argsort(rows))]
    if len(np.unique(episodes)) != 50:
        raise RuntimeError(
            "formal rows must map to 50 unique episodes: "
            f"expected=50, actual={len(np.unique(episodes))}"
        )
    return rows, episodes, manifest


def _selection(
    index: memory.CubeMemoryIndex,
    excluded_episodes: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    excluded = np.asarray(excluded_episodes, dtype=np.int64)
    eligible_mask = ~np.isin(np.asarray(index.episodes), excluded)
    eligible_indices = np.flatnonzero(eligible_mask)
    if count > len(eligible_indices):
        raise ValueError(
            "requested rollouts exceed unique eligible anchors: "
            f"expected_at_most={len(eligible_indices)}, actual={count}"
        )
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(eligible_indices, size=count, replace=False)
    rows = np.asarray(index.rows[chosen_indices], dtype=np.int64)
    episodes = np.asarray(index.episodes[chosen_indices], dtype=np.int64)
    if len(np.unique(rows)) != count:
        raise RuntimeError("start-row sampling unexpectedly used replacement")
    if np.intersect1d(episodes, excluded).size:
        raise RuntimeError("formal50 episode leaked into start-row selection")
    counts = _mix_counts(count)
    distribution = np.concatenate(
        [np.full(counts[name], DIST_CODE[name], dtype=np.uint8) for name in DIST_NAMES]
    )
    rng.shuffle(distribution)
    return rows, episodes, distribution


def _query_memory_top10(
    index: memory.CubeMemoryIndex,
    raw_feature: np.ndarray,
    excluded_episodes: set[int],
) -> dict[str, np.ndarray]:
    """Exact distinct-episode top-10 with closed-ball tie completion."""

    raw_query = np.asarray(raw_feature, dtype=np.float64).reshape(1, -1)
    query = index.normalize(raw_query)[0]
    k = 64
    selected: list[tuple[float, int, int, int]] = []
    while True:
        distances, indices = index.tree.query(
            query, k=min(k, len(index.rows)), eps=0.0, workers=1
        )
        candidates = sorted(
            (
                float(distance),
                int(index.rows[int(anchor_idx)]),
                int(index.episodes[int(anchor_idx)]),
                int(anchor_idx),
            )
            for distance, anchor_idx in zip(
                np.atleast_1d(distances), np.atleast_1d(indices), strict=True
            )
            if int(index.episodes[int(anchor_idx)]) not in excluded_episodes
        )
        selected = []
        seen: set[int] = set()
        for item in candidates:
            if item[2] in seen:
                continue
            selected.append(item)
            seen.add(item[2])
            if len(selected) == MEMORY_NEIGHBORS:
                break
        if len(selected) == MEMORY_NEIGHBORS:
            cutoff = selected[-1][0]
            ball = index.tree.query_ball_point(
                query, r=np.nextafter(cutoff, np.inf), eps=0.0, workers=1
            )
            complete = sorted(
                (
                    float(np.linalg.norm(index.features[int(anchor_idx)] - query)),
                    int(index.rows[int(anchor_idx)]),
                    int(index.episodes[int(anchor_idx)]),
                    int(anchor_idx),
                )
                for anchor_idx in ball
                if int(index.episodes[int(anchor_idx)]) not in excluded_episodes
            )
            selected = []
            seen = set()
            for item in complete:
                if item[2] in seen:
                    continue
                selected.append(item)
                seen.add(item[2])
                if len(selected) == MEMORY_NEIGHBORS:
                    break
            if len(selected) == MEMORY_NEIGHBORS:
                break
        if k >= len(index.rows):
            raise RuntimeError(
                "insufficient memory sources: "
                f"expected={MEMORY_NEIGHBORS}, actual={len(selected)}, "
                f"query_position={raw_query[0].tolist()}"
            )
        k = min(2 * k, len(index.rows))
    selected.sort(key=lambda item: (item[0], item[1]))
    return {
        "distances": np.asarray([item[0] for item in selected], dtype=np.float64),
        "rows": np.asarray([item[1] for item in selected], dtype=np.int64),
        "episodes": np.asarray([item[2] for item in selected], dtype=np.int64),
        "steps": np.asarray([item[1] % 201 for item in selected], dtype=np.int16),
        "anchor_indices": np.asarray([item[3] for item in selected], dtype=np.int64),
    }


def _inverse_solver(
    solver: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    flat = np.asarray(solver, dtype=np.float32).reshape(ENV_STEPS, ACTION_DIM).copy()
    flat *= np.asarray(scale, dtype=np.float64)[None]
    flat += np.asarray(mean, dtype=np.float64)[None]
    return flat


def _normalize_raw(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = np.asarray(raw, dtype=np.float32).copy()
    normalized -= np.asarray(mean, dtype=np.float64)[None]
    normalized /= np.asarray(scale, dtype=np.float64)[None]
    return normalized.reshape(MODEL_STEPS, ENV_STEPS)


def _generate_action(
    code: int,
    rng: np.random.Generator,
    index: memory.CubeMemoryIndex,
    h5: Any,
    source_row: int,
    source_episode: int,
    formal_excluded: set[int],
) -> dict[str, Any]:
    mean, scale = index.action_mean, index.action_scale
    provenance = {
        "rows": np.full(MEMORY_NEIGHBORS, -1, dtype=np.int64),
        "episodes": np.full(MEMORY_NEIGHBORS, -1, dtype=np.int64),
        "steps": np.full(MEMORY_NEIGHBORS, -1, dtype=np.int16),
        "distances": np.full(MEMORY_NEIGHBORS, np.nan, dtype=np.float64),
        "anchor_indices": np.full(MEMORY_NEIGHBORS, -1, dtype=np.int64),
        "selected_rank": -1,
        "selected_row": -1,
    }
    if code == DIST_CODE["gaussian"]:
        solver = rng.standard_normal((MODEL_STEPS, ENV_STEPS), dtype=np.float32)
        raw_preclip = _inverse_solver(solver, mean, scale)
    elif code == DIST_CODE["ar1"]:
        innovations = rng.standard_normal((ENV_STEPS, ACTION_DIM), dtype=np.float32)
        chrono = np.empty_like(innovations)
        chrono[0] = innovations[0]
        innovation_scale = np.float32(math.sqrt(1.0 - AR1_RHO**2))
        for step in range(1, ENV_STEPS):
            chrono[step] = np.float32(AR1_RHO) * chrono[step - 1] + innovation_scale * innovations[step]
        solver = chrono.reshape(MODEL_STEPS, ENV_STEPS)
        raw_preclip = _inverse_solver(solver, mean, scale)
    elif code == DIST_CODE["memory_t2"]:
        raw_feature = memory.feature_chunk(h5, source_row, source_row + 1)[0]
        excluded = set(formal_excluded)
        excluded.add(int(source_episode))
        retrieved = _query_memory_top10(index, raw_feature, excluded)
        rank = int(rng.integers(0, MEMORY_NEIGHBORS))
        seed_row = int(retrieved["rows"][rank])
        base = np.asarray(h5["action"][seed_row : seed_row + ENV_STEPS], dtype=np.float32)
        if base.shape != (ENV_STEPS, ACTION_DIM):
            raise RuntimeError(
                f"incomplete memory block: row={seed_row}, actual_shape={base.shape}"
            )
        if not np.isfinite(base).all():
            raise RuntimeError(f"nonfinite memory action block: row={seed_row}")
        noise = rng.standard_normal((ENV_STEPS, ACTION_DIM), dtype=np.float32)
        raw_preclip = base + np.float32(MEMORY_SIGMA) * noise
        solver = _normalize_raw(raw_preclip, mean, scale)
        provenance.update(retrieved)
        provenance["selected_rank"] = rank
        provenance["selected_row"] = seed_row
    else:
        raise ValueError(f"unknown distribution code: {code}")

    raw = np.clip(raw_preclip, -1.0, 1.0).astype(np.float32, copy=False)
    clip_mask = raw_preclip != raw
    model = _normalize_raw(raw, mean, scale).astype(np.float32, copy=False)
    if not (np.isfinite(solver).all() and np.isfinite(raw).all() and np.isfinite(model).all()):
        raise RuntimeError("generated actions contain nonfinite values")
    return {
        "action_solver_preclip": np.asarray(solver, dtype=np.float32),
        "action_env_preclip": np.asarray(raw_preclip, dtype=np.float32),
        "action_env": raw,
        "action_model": model,
        "clip_mask": clip_mask,
        "retrieval": provenance,
    }


def _target_quaternion(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _cube_pose(raw: Any) -> tuple[np.ndarray, np.ndarray, float]:
    quat = np.asarray(raw._data.joint("object_joint_0").qpos[3:7], dtype=np.float64).copy()
    position = np.asarray(raw._data.joint("object_joint_0").qpos[:3], dtype=np.float64).copy()
    yaw = float(math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)))
    return position, quat, yaw


def _jpeg(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    value = np.asarray(image, dtype=np.uint8)
    if value.shape != (224, 224, 3):
        raise RuntimeError(f"render shape mismatch: expected=(224,224,3), actual={value.shape}")
    buffer = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()


def _make_env() -> Any:
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    return gym.make(
        "swm/OGBCube-v0",
        max_episode_steps=1000,
        render_mode="rgb_array",
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
    )


def _setup_snapshot(env: Any, h5: Any, row: int, reset_seed: int) -> Any:
    import cube_cem_audit as audit
    import mujoco

    env.reset(seed=int(reset_seed))
    raw = env.unwrapped
    qpos = np.asarray(h5["qpos"][row], dtype=np.float64)
    qvel = np.asarray(h5["qvel"][row], dtype=np.float64)
    raw.set_state(qpos, qvel)
    target_block = int(h5["privileged_target_block"][row])
    if target_block != 0:
        raise RuntimeError(f"single-cube dataset target id is not zero: row={row}, id={target_block}")
    target_pos = np.asarray(h5["privileged_target_block_pos"][row], dtype=np.float64)
    target_yaw = float(np.asarray(h5["privileged_target_block_yaw"][row]).reshape(-1)[0])
    raw.set_target_pos(target_block, target_pos, _target_quaternion(target_yaw))
    if hasattr(raw, "_prev_qpos"):
        raw._prev_qpos = np.asarray(h5["prev_qpos"][row], dtype=np.float64).copy()
    if hasattr(raw, "_prev_qvel"):
        raw._prev_qvel = np.asarray(h5["prev_qvel"][row], dtype=np.float64).copy()
    mujoco.mj_forward(raw._model, raw._data)
    snapshot = audit._take_snapshot(env)
    # Explicit restore proves the snapshot path used by branch audits is valid.
    audit._restore_snapshot(env, snapshot)
    if not np.array_equal(np.asarray(raw._prev_qpos), np.asarray(h5["prev_qpos"][row])):
        raise RuntimeError(f"previous-qpos restore mismatch at row={row}")
    if not np.array_equal(np.asarray(raw._prev_qvel), np.asarray(h5["prev_qvel"][row])):
        raise RuntimeError(f"previous-qvel restore mismatch at row={row}")
    return snapshot


def _rollout(env: Any, snapshot: Any, actions: np.ndarray) -> dict[str, Any]:
    import cube_cem_audit as audit

    audit._restore_snapshot(env, snapshot)
    raw = env.unwrapped
    frames = [_jpeg(np.asarray(env.render(), dtype=np.uint8))]
    positions, quaternions, yaws = [], [], []
    position, quaternion, yaw = _cube_pose(raw)
    positions.append(position)
    quaternions.append(quaternion)
    yaws.append(yaw)
    terminated_flags = np.zeros(ENV_STEPS, dtype=bool)
    truncated_flags = np.zeros(ENV_STEPS, dtype=bool)
    for step, action in enumerate(np.asarray(actions, dtype=np.float32), start=1):
        _, _, terminated, truncated, _ = env.step(action)
        terminated_flags[step - 1] = bool(terminated)
        truncated_flags[step - 1] = bool(truncated)
        if truncated:
            raise RuntimeError(
                f"off-policy rollout truncated before fixed horizon: step={step}/{ENV_STEPS}"
            )
        # terminate_at_goal=False should make termination impossible.  Failing
        # closed avoids an implicit reset on a subsequent step.
        if terminated:
            raise RuntimeError(
                f"off-policy env terminated despite terminate_at_goal=False: step={step}/{ENV_STEPS}"
            )
        if step % (ENV_STEPS // MODEL_STEPS) == 0:
            frames.append(_jpeg(np.asarray(env.render(), dtype=np.uint8)))
            position, quaternion, yaw = _cube_pose(raw)
            positions.append(position)
            quaternions.append(quaternion)
            yaws.append(yaw)
    if len(frames) != FRAME_COUNT or len(positions) != FRAME_COUNT:
        raise RuntimeError(
            "rollout observation count mismatch: "
            f"expected={FRAME_COUNT}, actual={len(frames)}/{len(positions)}"
        )
    return {
        "pixels_jpeg": frames,
        "block_pos": np.stack(positions),
        "block_quat": np.stack(quaternions),
        "block_yaw": np.asarray(yaws, dtype=np.float64),
        "terminated": terminated_flags,
        "truncated": truncated_flags,
    }


def _empty_batch(count: int) -> dict[str, Any]:
    return {
        "pixels_jpeg": [[None] * FRAME_COUNT for _ in range(count)],
        "action_env": np.empty((count, ENV_STEPS, ACTION_DIM), dtype=np.float32),
        "action_env_preclip": np.empty((count, ENV_STEPS, ACTION_DIM), dtype=np.float32),
        "action_model": np.empty((count, MODEL_STEPS, ENV_STEPS), dtype=np.float32),
        "action_solver_preclip": np.empty((count, MODEL_STEPS, ENV_STEPS), dtype=np.float32),
        "clip_mask": np.empty((count, ENV_STEPS, ACTION_DIM), dtype=bool),
        "block_pos": np.empty((count, FRAME_COUNT, 3), dtype=np.float64),
        "block_quat": np.empty((count, FRAME_COUNT, 4), dtype=np.float64),
        "block_yaw": np.empty((count, FRAME_COUNT), dtype=np.float64),
        "terminated": np.empty((count, ENV_STEPS), dtype=bool),
        "truncated": np.empty((count, ENV_STEPS), dtype=bool),
        "retrieval_rows": np.full((count, MEMORY_NEIGHBORS), -1, dtype=np.int64),
        "retrieval_episodes": np.full((count, MEMORY_NEIGHBORS), -1, dtype=np.int64),
        "retrieval_steps": np.full((count, MEMORY_NEIGHBORS), -1, dtype=np.int16),
        "retrieval_distances": np.full((count, MEMORY_NEIGHBORS), np.nan, dtype=np.float64),
        "retrieval_anchor_indices": np.full((count, MEMORY_NEIGHBORS), -1, dtype=np.int64),
        "retrieval_selected_rank": np.full(count, -1, dtype=np.int8),
        "retrieval_selected_row": np.full(count, -1, dtype=np.int64),
        "initial_qpos": np.empty((count, 21), dtype=np.float64),
        "initial_qvel": np.empty((count, 20), dtype=np.float64),
        "initial_prev_qpos": np.empty((count, 21), dtype=np.float64),
        "initial_prev_qvel": np.empty((count, 20), dtype=np.float64),
        "source_target_pos": np.empty((count, 3), dtype=np.float64),
        "source_target_yaw": np.empty(count, dtype=np.float64),
        "source_h5_pixel_mae": np.empty(count, dtype=np.float32),
    }


def _write_shard(
    path: Path,
    batch: dict[str, Any],
    rollout_ids: np.ndarray,
    rows: np.ndarray,
    episodes: np.ndarray,
    codes: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401
    import h5py

    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if partial.exists():
        partial.unlink()
    n = len(rows)
    compression = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    try:
        with h5py.File(partial, "w", libver="latest") as out:
            out.attrs["format_version"] = FORMAT_VERSION
            out.attrs["jpeg"] = json.dumps(
                {"quality": JPEG_QUALITY, "subsampling": "4:4:4", "mode": "RGB"},
                sort_keys=True,
            )
            out.attrs["distribution_names"] = json.dumps(DIST_NAMES)
            out.attrs["action_normalizer_mean"] = mean
            out.attrs["action_normalizer_scale"] = scale
            out.create_dataset("rollout_id", data=rollout_ids)
            out.create_dataset("source_row", data=rows)
            out.create_dataset("source_episode", data=episodes)
            out.create_dataset("source_step", data=(rows % 201).astype(np.int16))
            out.create_dataset("distribution_code", data=codes)
            jpeg_type = h5py.vlen_dtype(np.dtype("uint8"))
            pixels = out.create_dataset("pixels_jpeg", shape=(n, FRAME_COUNT), dtype=jpeg_type)
            for item in range(n):
                for frame in range(FRAME_COUNT):
                    pixels[item, frame] = batch["pixels_jpeg"][item][frame]
            for name, values in batch.items():
                if name == "pixels_jpeg":
                    continue
                out.create_dataset(name, data=values, **compression)
            out.flush()
        # Make the closed shard durable before the atomic directory entry swap.
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "first_rollout_id": int(rollout_ids[0]),
        "last_rollout_id": int(rollout_ids[-1]),
        "num_rollouts": n,
        "size_bytes": int(stat.st_size),
        "sha256": _sha256(path),
        "distribution_counts": {
            name: int(np.count_nonzero(codes == DIST_CODE[name])) for name in DIST_NAMES
        },
        "memory_selected_rank_counts": {
            str(rank): int(
                np.count_nonzero(batch["retrieval_selected_rank"] == rank)
            )
            for rank in range(MEMORY_NEIGHBORS)
        },
    }


def _load_or_create_selection(
    root: Path,
    index: memory.CubeMemoryIndex,
    excluded: np.ndarray,
    count: int,
    seed: int,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / "selection.npz"
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"resume selection missing: {path}")
        with np.load(path, allow_pickle=False) as data:
            rows = np.asarray(data["source_rows"], dtype=np.int64)
            episodes = np.asarray(data["source_episodes"], dtype=np.int64)
            codes = np.asarray(data["distribution_code"], dtype=np.uint8)
        if rows.shape != (count,) or episodes.shape != (count,) or codes.shape != (count,):
            raise ValueError(
                f"resume selection shape mismatch: expected={(count,)}, "
                f"actual={rows.shape}/{episodes.shape}/{codes.shape}"
            )
        return rows, episodes, codes
    rows, episodes, codes = _selection(index, excluded, count, seed)
    _atomic_npz(
        path,
        source_rows=rows,
        source_episodes=episodes,
        distribution_code=codes,
        excluded_formal_episodes=np.asarray(excluded, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
    )
    return rows, episodes, codes


def _prepare_root(root: Path, overwrite: bool, resume: bool) -> Path:
    root = _assert_output(root)
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    root.mkdir(parents=True, exist_ok=True)
    owned = (root / "shards", root / "manifest.json", root / "progress.json", root / "selection.npz")
    if overwrite:
        if (root / "shards").exists():
            shutil.rmtree(root / "shards")
        for path in owned[1:]:
            if path.exists():
                path.unlink()
    elif not resume and any(path.exists() for path in owned):
        raise FileExistsError(
            f"collection output already exists: {root}; use --resume or --overwrite"
        )
    (root / "shards").mkdir(exist_ok=True)
    return root


_WORKER: dict[str, Any] = {}


def _rollout_seed(seed: int, rollout_id: int) -> int:
    material = f"cube-offpolicy-v1|{int(seed)}|{int(rollout_id)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _worker_close() -> None:
    env = _WORKER.get("env")
    h5 = _WORKER.get("h5")
    if env is not None:
        env.close()
    if h5 is not None:
        h5.close()
    _WORKER.clear()


def _worker_init(formal_episodes: Sequence[int]) -> None:
    """Give every spawned worker its own env, read-only HDF5, and index."""

    import atexit
    import hdf5plugin  # noqa: F401
    import h5py

    _configure_storage()
    _WORKER.update(
        {
            "index": memory.CubeMemoryIndex(INDEX_ROOT, DATASET),
            "h5": h5py.File(DATASET, "r", swmr=True),
            "env": _make_env(),
            "formal": set(map(int, formal_episodes)),
        }
    )
    atexit.register(_worker_close)


def _worker_shard(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        root_text,
        shard_start,
        shard_stop,
        rows,
        episodes,
        codes,
        seed,
        total_count,
    ) = task
    root = Path(root_text)
    index = _WORKER["index"]
    h5 = _WORKER["h5"]
    env = _WORKER["env"]
    formal_set = _WORKER["formal"]
    rows = np.asarray(rows, dtype=np.int64)
    episodes = np.asarray(episodes, dtype=np.int64)
    codes = np.asarray(codes, dtype=np.uint8)
    n = int(shard_stop) - int(shard_start)
    if rows.shape != (n,) or episodes.shape != (n,) or codes.shape != (n,):
        raise ValueError(
            f"worker task shape mismatch: expected={(n,)}, "
            f"actual={rows.shape}/{episodes.shape}/{codes.shape}"
        )
    batch = _empty_batch(n)
    for local, rollout_id in enumerate(range(int(shard_start), int(shard_stop))):
        row = int(rows[local])
        episode = int(episodes[local])
        code = int(codes[local])
        derived_seed = _rollout_seed(int(seed), rollout_id)
        rng = np.random.default_rng(derived_seed)
        action = _generate_action(code, rng, index, h5, row, episode, formal_set)
        snapshot = _setup_snapshot(
            env, h5, row, reset_seed=int(derived_seed & 0xFFFF_FFFF)
        )
        result = _rollout(env, snapshot, action["action_env"])
        batch["pixels_jpeg"][local] = result["pixels_jpeg"]
        for name in (
            "action_env",
            "action_env_preclip",
            "action_model",
            "action_solver_preclip",
            "clip_mask",
        ):
            batch[name][local] = action[name]
        for name in (
            "block_pos",
            "block_quat",
            "block_yaw",
            "terminated",
            "truncated",
        ):
            batch[name][local] = result[name]
        retrieval = action["retrieval"]
        for target, source in (
            ("retrieval_rows", "rows"),
            ("retrieval_episodes", "episodes"),
            ("retrieval_steps", "steps"),
            ("retrieval_distances", "distances"),
            ("retrieval_anchor_indices", "anchor_indices"),
        ):
            batch[target][local] = retrieval[source]
        batch["retrieval_selected_rank"][local] = retrieval["selected_rank"]
        batch["retrieval_selected_row"][local] = retrieval["selected_row"]
        batch["initial_qpos"][local] = h5["qpos"][row]
        batch["initial_qvel"][local] = h5["qvel"][row]
        batch["initial_prev_qpos"][local] = h5["prev_qpos"][row]
        batch["initial_prev_qvel"][local] = h5["prev_qvel"][row]
        batch["source_target_pos"][local] = h5["privileged_target_block_pos"][row]
        batch["source_target_yaw"][local] = np.asarray(
            h5["privileged_target_block_yaw"][row]
        ).reshape(-1)[0]
        from PIL import Image

        rendered = np.asarray(
            Image.open(io.BytesIO(result["pixels_jpeg"][0].tobytes())).convert("RGB"),
            dtype=np.float32,
        )
        reference = np.asarray(h5["pixels"][row], dtype=np.float32)
        batch["source_h5_pixel_mae"][local] = float(
            np.mean(np.abs(rendered - reference))
        )
        if (local + 1) % 25 == 0 or local + 1 == n:
            print(
                f"worker={os.getpid()} shard={shard_start}:{shard_stop} "
                f"local={local + 1}/{n} total_target={total_count}",
                flush=True,
            )
    shard_path = root / "shards" / f"shard_{int(shard_start):06d}_{int(shard_stop):06d}.h5"
    record = _write_shard(
        shard_path,
        batch,
        np.arange(shard_start, shard_stop, dtype=np.int64),
        rows,
        episodes,
        codes,
        index.action_mean,
        index.action_scale,
    )
    record["worker_pid"] = os.getpid()
    return record


def _collect(args: argparse.Namespace, smoke: bool) -> None:
    _configure_storage()
    count = int(args.num_rollouts)
    if count <= 0:
        raise ValueError(f"--num-rollouts must be positive: {count}")
    if not smoke and count != FORMAL_COUNT and not args.allow_nonformal_count:
        raise ValueError(
            f"formal collection is frozen to {FORMAL_COUNT}; actual={count}; "
            "use --allow-nonformal-count only for development"
        )
    if smoke and count != SMOKE_COUNT and not args.allow_nonformal_count:
        raise ValueError(
            f"smoke collection is frozen to {SMOKE_COUNT}; actual={count}; "
            "use --allow-nonformal-count only for development"
        )
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive: {args.workers}")
    effective_shard_size = int(args.shard_size)
    if smoke:
        # A 200-rollout smoke must actually exercise all requested workers so
        # its wall-time extrapolation measures the formal parallel topology.
        effective_shard_size = min(
            effective_shard_size, int(math.ceil(count / args.workers))
        )
    root = _prepare_root(SMOKE_ROOT if smoke else OUTPUT_ROOT, args.overwrite, args.resume)
    free_before = _file_system_free(root)
    if free_before < 2 * (1 << 30):
        raise RuntimeError(
            "insufficient data-disk headroom: "
            f"expected_at_least={2 * (1 << 30)}, actual={free_before}, path={root}"
        )

    import hdf5plugin  # noqa: F401
    import h5py

    index = memory.CubeMemoryIndex(INDEX_ROOT, DATASET)
    progress_path = root / "progress.json"
    progress = None
    if args.resume:
        if not progress_path.is_file():
            raise FileNotFoundError(f"resume progress missing: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(progress["requested_rollouts"]) != count or int(progress["seed"]) != args.seed:
            raise ValueError(
                "resume protocol mismatch: "
                f"expected_count/seed={progress['requested_rollouts']}/{progress['seed']}, "
                f"actual={count}/{args.seed}"
            )
        if int(progress.get("effective_shard_size", -1)) != effective_shard_size:
            raise ValueError(
                "resume shard-size mismatch: "
                f"expected={progress.get('effective_shard_size')}, "
                f"actual={effective_shard_size}"
            )
    with h5py.File(DATASET, "r", swmr=True) as h5:
        formal_rows, formal_episodes, formal_manifest = _formal_episodes(h5)
        rows, episodes, codes = _load_or_create_selection(
            root, index, formal_episodes, count, args.seed, args.resume
        )
    action_mean = index.action_mean.copy()
    action_scale = index.action_scale.copy()
    del index

    shard_records = list(progress.get("shards", [])) if progress else []
    completed_ranges: set[tuple[int, int]] = set()
    for record in shard_records:
        path = root / "shards" / record["filename"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"resume shard identity mismatch: {path}")
        begin = int(record["first_rollout_id"])
        end = int(record["last_rollout_id"]) + 1
        if (begin, end) in completed_ranges:
            raise ValueError(f"duplicate resume shard range: {(begin, end)}")
        completed_ranges.add((begin, end))
    if sum(int(item["num_rollouts"]) for item in shard_records) > count:
        raise ValueError("resume shard count exceeds requested rollout count")
    tasks = []
    for shard_start in range(0, count, effective_shard_size):
        shard_stop = min(shard_start + effective_shard_size, count)
        if (shard_start, shard_stop) in completed_ranges:
            continue
        tasks.append(
            (
                str(root),
                shard_start,
                shard_stop,
                rows[shard_start:shard_stop],
                episodes[shard_start:shard_stop],
                codes[shard_start:shard_stop],
                args.seed,
                count,
            )
        )

    started = time.time()
    prior_elapsed = float(progress.get("elapsed_seconds", 0.0)) if progress else 0.0
    if tasks:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with context.Pool(
            processes=min(args.workers, len(tasks)),
            initializer=_worker_init,
            initargs=(formal_episodes.tolist(),),
        ) as pool:
            for record in pool.imap_unordered(_worker_shard, tasks, chunksize=1):
                shard_records.append(record)
                shard_records.sort(key=lambda item: int(item["first_rollout_id"]))
                completed = sum(int(item["num_rollouts"]) for item in shard_records)
                elapsed = prior_elapsed + time.time() - started
                bytes_written = sum(int(item["size_bytes"]) for item in shard_records)
                projected = int(math.ceil(bytes_written / completed * count))
                if projected > MAX_BYTES:
                    pool.terminate()
                    raise RuntimeError(
                        "projected collection exceeds frozen disk budget: "
                        f"expected_at_most={MAX_BYTES}, actual_projected={projected}, "
                        f"after_rollouts={completed}"
                    )
                free_now = _file_system_free(root)
                remaining_estimate = max(0, projected - bytes_written)
                if free_now < remaining_estimate + (1 << 30):
                    pool.terminate()
                    raise RuntimeError(
                        "insufficient data-disk free space for projected collection: "
                        f"expected_at_least={remaining_estimate + (1 << 30)}, "
                        f"actual={free_now}, after_rollouts={completed}"
                    )
                _atomic_json(
                    progress_path,
                    {
                        "format_version": FORMAT_VERSION,
                        "complete": completed == count,
                        "requested_rollouts": count,
                        "completed_rollouts": completed,
                        "seed": args.seed,
                        "worker_count": args.workers,
                        "effective_shard_size": effective_shard_size,
                        "worker_independent_rollout_seed": True,
                        "elapsed_seconds": elapsed,
                        "bytes_written": bytes_written,
                        "projected_bytes": projected,
                        "shards": shard_records,
                    },
                )
                print(
                    f"completed atomic shards: rollouts={completed}/{count}, "
                    f"bytes={bytes_written}",
                    flush=True,
                )
    if not progress_path.is_file():
        # This occurs only for a zero-task resume of an already complete set.
        raise RuntimeError(f"collection made no progress and has no progress file: {root}")
    progress_final = json.loads(progress_path.read_text(encoding="utf-8"))
    elapsed = float(progress_final["elapsed_seconds"])
    if sum(int(item["num_rollouts"]) for item in shard_records) != count:
        raise RuntimeError(
            "collection returned without all atomic shards: "
            f"expected={count}, actual={sum(int(item['num_rollouts']) for item in shard_records)}"
        )
    total_bytes = sum(int(item["size_bytes"]) for item in shard_records)
    actual_counts = {
        name: int(np.count_nonzero(codes == DIST_CODE[name])) for name in DIST_NAMES
    }
    memory_rank_counts = {
        str(rank): int(
            sum(
                int(item["memory_selected_rank_counts"][str(rank)])
                for item in shard_records
            )
        )
        for rank in range(MEMORY_NEIGHBORS)
    }
    expected_counts = _mix_counts(count)
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"mixture count mismatch: expected={expected_counts}, actual={actual_counts}"
        )
    manifest = {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "scope": "smoke" if smoke else "formal",
        "num_rollouts": count,
        "num_model_transitions": count * MODEL_STEPS,
        "num_environment_steps": count * ENV_STEPS,
        "seed": args.seed,
        "selection": {
            "source": "memory index anchors",
            "without_replacement": True,
            "max_source_step_inclusive": MAX_START_STEP,
            "formal50_globally_excluded": True,
            "formal_rows": formal_rows,
            "formal_episodes": formal_episodes,
            "selection_file": _identity(root / "selection.npz", include_sha256=True),
        },
        "action_protocol": {
            "mixture_requested": MIX,
            "counts": actual_counts,
            "gaussian": "iid solver N(0,1); eval-scaler inverse; raw clip [-1,1]",
            "memory_t2": (
                "current-state exact nearest 10 distinct episodes after global formal50 "
                "and current-episode exclusion; uniform rank; raw sigma=0.1 iid; clip [-1,1]"
            ),
            "memory_selected_rank_counts": memory_rank_counts,
            "ar1": (
                f"stationary solver Gaussian AR(1), rho={AR1_RHO}; eval-scaler inverse; "
                "raw clip [-1,1]"
            ),
            "randomness": (
                "SHA256(master seed, rollout_id) derives one independent NumPy PCG64 "
                "stream per rollout; values and selected memory rank are invariant to worker count"
            ),
            "action_normalizer": {
                "fit_contract": "formal evaluation StandardScaler over 2,000,000 finite H5 actions",
                "mean": action_mean,
                "scale": action_scale,
                "source_index_stats": _identity(INDEX_ROOT / "stats.npz", include_sha256=True),
            },
        },
        "rollout_protocol": {
            "fixed_environment_steps": ENV_STEPS,
            "early_termination": False,
            "model_frame_stride": ENV_STEPS // MODEL_STEPS,
            "frames_per_rollout": FRAME_COUNT,
            "pixels": f"JPEG quality={JPEG_QUALITY}, RGB, 4:4:4",
            "initialization": (
                "env reset; H5 qpos/qvel set_state; H5 target pose; H5 prev_qpos/prev_qvel; "
                "mj_forward; complete integration/Python/RNG snapshot; explicit restore"
            ),
        },
        "storage": {
            "root": str(root.resolve()),
            "budget_bytes": MAX_BYTES,
            "total_shard_bytes": total_bytes,
            "under_budget": total_bytes <= MAX_BYTES,
            "effective_shard_size": effective_shard_size,
            "free_bytes_before": free_before,
            "free_bytes_after": _file_system_free(root),
            "shards": shard_records,
        },
        "timing": {
            "elapsed_seconds": elapsed,
            "workers": args.workers,
            "rollouts_per_second": count / elapsed,
            "seconds_per_rollout": elapsed / count,
            "projected_30000_seconds": elapsed / count * FORMAL_COUNT,
            "projected_30000_hours": elapsed / count * FORMAL_COUNT / 3600.0,
            "bytes_per_rollout": total_bytes / count,
            "projected_30000_bytes": int(math.ceil(total_bytes / count * FORMAL_COUNT)),
            "projection_formula": (
                "projected_30000_seconds = observed_wall_seconds / observed_rollouts * 30000; "
                "observed wall time already includes the configured worker count"
            ),
        },
        "sources": {
            "dataset": _identity(DATASET, include_sha256=False),
            "memory_index_metadata": _identity(INDEX_ROOT / "metadata.json", include_sha256=True),
            "formal_manifest": _identity(FORMAL_MANIFEST, include_sha256=True),
            "formal_manifest_dataset_identity": formal_manifest.get("dataset"),
        },
        "schema": {
            "pixels_jpeg": ["N", FRAME_COUNT, "vlen uint8"],
            "action_env": ["N", ENV_STEPS, ACTION_DIM],
            "action_env_preclip": ["N", ENV_STEPS, ACTION_DIM],
            "action_model": ["N", MODEL_STEPS, ENV_STEPS],
            "action_solver_preclip": ["N", MODEL_STEPS, ENV_STEPS],
            "clip_mask": ["N", ENV_STEPS, ACTION_DIM],
            "block_pos": ["N", FRAME_COUNT, 3],
            "block_quat": ["N", FRAME_COUNT, 4],
            "block_yaw": ["N", FRAME_COUNT],
            "retrieval_provenance": ["N", MEMORY_NEIGHBORS],
        },
    }
    _atomic_json(root / "manifest.json", manifest)
    print(
        f"collection complete: root={root}, rollouts={count}, "
        f"bytes={total_bytes}, elapsed_s={elapsed:.1f}",
        flush=True,
    )


def _decode_jpeg(value: np.ndarray) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(np.asarray(value, dtype=np.uint8).tobytes())).convert("RGB"))


def _validate(args: argparse.Namespace) -> None:
    _configure_storage()
    root = _assert_output(SMOKE_ROOT if args.scope == "smoke" else OUTPUT_ROOT)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION or not manifest.get("complete"):
        raise ValueError(f"incomplete/unsupported manifest: {manifest_path}")
    expected_count = int(manifest["num_rollouts"])
    expected_next = 0
    aggregate = {name: 0 for name in DIST_NAMES}
    rank_aggregate = {str(rank): 0 for rank in range(MEMORY_NEIGHBORS)}
    formal = set(map(int, manifest["selection"]["formal_episodes"]))
    mean = np.asarray(manifest["action_protocol"]["action_normalizer"]["mean"], dtype=np.float64)
    scale = np.asarray(manifest["action_protocol"]["action_normalizer"]["scale"], dtype=np.float64)
    total_size = 0
    all_source_rows: list[np.ndarray] = []
    all_source_episodes: list[np.ndarray] = []
    import hdf5plugin  # noqa: F401
    import h5py

    for shard in manifest["storage"]["shards"]:
        path = root / "shards" / shard["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != shard["sha256"]:
            raise RuntimeError(f"shard hash mismatch: {path}")
        total_size += path.stat().st_size
        with h5py.File(path, "r", swmr=True) as h5:
            if h5.attrs.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"shard format mismatch: {path}")
            ids = np.asarray(h5["rollout_id"][:], dtype=np.int64)
            expected_ids = np.arange(expected_next, expected_next + len(ids), dtype=np.int64)
            if not np.array_equal(ids, expected_ids):
                raise RuntimeError(
                    f"rollout id discontinuity: path={path}, expected={expected_ids[:3].tolist()}, "
                    f"actual={ids[:3].tolist()}"
                )
            expected_next += len(ids)
            rows = np.asarray(h5["source_row"][:], dtype=np.int64)
            episodes = np.asarray(h5["source_episode"][:], dtype=np.int64)
            all_source_rows.append(rows)
            all_source_episodes.append(episodes)
            steps = np.asarray(h5["source_step"][:], dtype=np.int64)
            codes = np.asarray(h5["distribution_code"][:], dtype=np.uint8)
            if np.intersect1d(episodes, np.fromiter(formal, dtype=np.int64)).size:
                raise RuntimeError(f"formal episode leaked into source rows: {path}")
            if np.any(steps > MAX_START_STEP) or not np.array_equal(steps, rows % 201):
                raise RuntimeError(f"invalid source step/row relation: {path}")
            for name in DIST_NAMES:
                aggregate[name] += int(np.count_nonzero(codes == DIST_CODE[name]))
            action_env = np.asarray(h5["action_env"][:], dtype=np.float32)
            action_env_preclip = np.asarray(h5["action_env_preclip"][:], dtype=np.float32)
            action_model = np.asarray(h5["action_model"][:], dtype=np.float32)
            solver = np.asarray(h5["action_solver_preclip"][:], dtype=np.float32)
            clips = np.asarray(h5["clip_mask"][:], dtype=bool)
            if action_env.shape != (len(ids), ENV_STEPS, ACTION_DIM):
                raise RuntimeError(f"action_env shape mismatch: {path} {action_env.shape}")
            if action_model.shape != (len(ids), MODEL_STEPS, ENV_STEPS) or solver.shape != action_model.shape:
                raise RuntimeError(f"model/solver action shape mismatch: {path}")
            if clips.shape != action_env.shape or np.any(np.abs(action_env) > 1.0):
                raise RuntimeError(f"clip shape/range mismatch: {path}")
            raw_preclip = solver.reshape(len(ids), ENV_STEPS, ACTION_DIM).copy()
            raw_preclip *= scale[None, None]
            raw_preclip += mean[None, None]
            if not np.allclose(raw_preclip, action_env_preclip, rtol=0.0, atol=1e-6):
                delta = float(np.max(np.abs(raw_preclip - action_env_preclip)))
                raise RuntimeError(
                    f"solver/raw-preclip roundtrip mismatch: path={path}, max_abs={delta}"
                )
            expected_action_env = np.clip(action_env_preclip, -1.0, 1.0).astype(
                np.float32, copy=False
            )
            expected_clip_mask = action_env_preclip != expected_action_env
            if not np.array_equal(expected_action_env, action_env):
                delta = float(np.max(np.abs(expected_action_env - action_env)))
                raise RuntimeError(
                    f"solver inverse/clip mismatch: path={path}, max_abs={delta}"
                )
            if not np.array_equal(expected_clip_mask, clips):
                bad = int(np.count_nonzero(expected_clip_mask != clips))
                raise RuntimeError(
                    f"clip-mask mismatch: path={path}, differing_elements={bad}"
                )
            reconstructed = action_env.copy()
            reconstructed -= mean[None, None]
            reconstructed /= scale[None, None]
            reconstructed = reconstructed.reshape(len(ids), MODEL_STEPS, ENV_STEPS)
            if not np.array_equal(reconstructed, action_model):
                delta = float(np.max(np.abs(reconstructed - action_model)))
                raise RuntimeError(
                    f"action normalizer roundtrip mismatch: path={path}, max_abs={delta}"
                )
            retrieval_episodes = np.asarray(h5["retrieval_episodes"][:], dtype=np.int64)
            retrieval_rows = np.asarray(h5["retrieval_rows"][:], dtype=np.int64)
            ranks = np.asarray(h5["retrieval_selected_rank"][:], dtype=np.int64)
            for local, code in enumerate(codes):
                if int(code) == DIST_CODE["memory_t2"]:
                    eps = retrieval_episodes[local]
                    if len(np.unique(eps)) != MEMORY_NEIGHBORS:
                        raise RuntimeError(f"memory sources are not 10 unique episodes: {path}:{local}")
                    leaked = set(map(int, eps)) & (formal | {int(episodes[local])})
                    if leaked:
                        raise RuntimeError(
                            f"memory retrieval episode leakage: path={path}, local={local}, leaked={sorted(leaked)}"
                        )
                    rank = int(ranks[local])
                    if not 0 <= rank < MEMORY_NEIGHBORS or int(h5["retrieval_selected_row"][local]) != int(retrieval_rows[local, rank]):
                        raise RuntimeError(f"memory selected-rank provenance mismatch: {path}:{local}")
                elif not (
                    np.all(retrieval_rows[local] == -1)
                    and np.all(retrieval_episodes[local] == -1)
                    and int(ranks[local]) == -1
                ):
                    raise RuntimeError(f"non-memory rollout has retrieval provenance: {path}:{local}")
            for rank in range(MEMORY_NEIGHBORS):
                rank_aggregate[str(rank)] += int(np.count_nonzero(ranks == rank))
            if np.any(np.asarray(h5["terminated"][:])) or np.any(np.asarray(h5["truncated"][:])):
                raise RuntimeError(f"fixed-horizon shard contains terminated/truncated rollout: {path}")
            sample_indices = sorted(set((0, len(ids) - 1)))
            for local in sample_indices:
                for frame in range(FRAME_COUNT):
                    decoded = _decode_jpeg(h5["pixels_jpeg"][local, frame])
                    if decoded.shape != (224, 224, 3) or decoded.dtype != np.uint8:
                        raise RuntimeError(
                            f"JPEG decode mismatch: path={path}, local={local}, frame={frame}, "
                            f"actual={decoded.shape}/{decoded.dtype}"
                        )
            for name, shape in (
                ("block_pos", (len(ids), FRAME_COUNT, 3)),
                ("block_quat", (len(ids), FRAME_COUNT, 4)),
                ("block_yaw", (len(ids), FRAME_COUNT)),
            ):
                value = np.asarray(h5[name][:])
                if value.shape != shape or not np.isfinite(value).all():
                    raise RuntimeError(f"pose dataset mismatch: {path}:{name}:{value.shape}")
    if expected_next != expected_count:
        raise RuntimeError(
            f"validated rollout count mismatch: expected={expected_count}, actual={expected_next}"
        )
    source_rows = np.concatenate(all_source_rows)
    source_episodes = np.concatenate(all_source_episodes)
    if len(np.unique(source_rows)) != expected_count:
        raise RuntimeError(
            "source starts were not sampled without replacement: "
            f"expected_unique={expected_count}, actual={len(np.unique(source_rows))}"
        )
    with h5py.File(DATASET, "r", swmr=True) as source_h5:
        order = np.argsort(source_rows)
        stored_episodes = np.asarray(source_h5["ep_idx"][source_rows[order]], dtype=np.int64)
        stored_steps = np.asarray(source_h5["step_idx"][source_rows[order]], dtype=np.int64)
        if not np.array_equal(stored_episodes, source_episodes[order]):
            raise RuntimeError("source row/episode provenance disagrees with immutable HDF5")
        if np.any(stored_steps > MAX_START_STEP):
            raise RuntimeError(
                "immutable HDF5 source step exceeds frozen maximum: "
                f"expected_at_most={MAX_START_STEP}, actual={int(stored_steps.max())}"
            )
        endpoint_episodes = np.asarray(
            source_h5["ep_idx"][(source_rows + ENV_STEPS)[order]], dtype=np.int64
        )
        if not np.array_equal(endpoint_episodes, stored_episodes):
            raise RuntimeError("one or more source rollouts cross an episode boundary")
    if aggregate != manifest["action_protocol"]["counts"]:
        raise RuntimeError(
            f"validated mixture mismatch: expected={manifest['action_protocol']['counts']}, actual={aggregate}"
        )
    if rank_aggregate != manifest["action_protocol"]["memory_selected_rank_counts"]:
        raise RuntimeError(
            "validated memory-rank histogram mismatch: "
            f"expected={manifest['action_protocol']['memory_selected_rank_counts']}, "
            f"actual={rank_aggregate}"
        )
    if total_size != int(manifest["storage"]["total_shard_bytes"]) or total_size > MAX_BYTES:
        raise RuntimeError(
            "validated storage mismatch: "
            f"expected={manifest['storage']['total_shard_bytes']} <= {MAX_BYTES}, actual={total_size}"
        )
    print(
        json.dumps(
            {
                "valid": True,
                "root": str(root),
                "rollouts": expected_next,
                "mixture": aggregate,
                "bytes": total_size,
            },
            sort_keys=True,
        )
    )


def _add_collect_args(parser: argparse.ArgumentParser, default_count: int) -> None:
    parser.add_argument("--num-rollouts", type=int, default=default_count)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="spawned CPU/MuJoCo workers; per-rollout RNG is worker-count invariant",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-nonformal-count",
        action="store_true",
        help="development only: override frozen 200/30000 rollout count",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="collect the frozen 200-rollout throughput smoke")
    _add_collect_args(smoke, SMOKE_COUNT)
    smoke.set_defaults(func=lambda args: _collect(args, smoke=True))
    collect = sub.add_parser("collect", help="collect the frozen 30,000-rollout formal dataset")
    _add_collect_args(collect, FORMAL_COUNT)
    collect.set_defaults(func=lambda args: _collect(args, smoke=False))
    validate = sub.add_parser("validate", help="validate hashes, schema, mix, and leakage")
    validate.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    validate.set_defaults(func=_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "shard_size") and args.shard_size <= 0:
        raise ValueError(f"--shard-size must be positive: {args.shard_size}")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
