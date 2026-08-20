#!/usr/bin/env python3
"""Dataset and integrity contracts for planner-in-the-loop Cube V2 data.

V2 deliberately exposes an entire six-frame/five-action rollout as one sample.
Unlike V1's synthetic action sources, ``action_model`` is the exact normalized
sequence sampled by the deployed T2 CEM planner.  Splits are made by source
episode before rollouts are exposed, so candidates from one injected simulator
state cannot cross the train/validation boundary.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import hdf5plugin  # noqa: F401  # Register HDF5 compression filters first.
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cube_offpolicy import (
    assert_frozen_hashes,
    freeze_predictor_only,
    frozen_state_hashes,
    module_state_sha256,
    sha256_file,
    write_json,
)


FORMAT_VERSION = "cube_offpolicy_planner_rollout_hdf5_v2"
ROLLOUT_FRAMES = 6
ROLLOUT_ACTIONS = 5
ACTION_DIM = 25
IMAGE_SIZE = 224
FORMAL_BATCH_SIZE = 128
FORMAL_EXPERT_BATCH = 96
FORMAL_V2_BATCH = 32
RETRY_EXPERT_BATCH = 104
RETRY_V2_BATCH = 24


def _contract_error(label: str, expected: Any, actual: Any, position: Any) -> ValueError:
    return ValueError(
        f"{label}: expected={expected!r}, actual={actual!r}, position={position!r}"
    )


def _manifest_shards(manifest: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    entries = manifest.get("shards")
    if entries is None and isinstance(manifest.get("storage"), Mapping):
        entries = manifest["storage"].get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("V2 manifest must contain a non-empty shard list")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"V2 shard entry {index} is not an object")
        relative = raw.get("path", raw.get("filename"))
        if relative is None:
            raise ValueError(f"V2 shard entry {index} has no path/filename")
        path = Path(str(relative))
        if path.is_absolute():
            path = path.resolve()
        else:
            direct = (root / path).resolve()
            nested = (root / "shards" / path).resolve()
            path = direct if direct.is_file() else nested
        if path != root and root not in path.parents:
            raise ValueError(f"V2 shard escapes dataset root: {path}")
        count = int(raw.get("num_rollouts", raw.get("count", -1)))
        digest = str(raw.get("sha256", ""))
        if count <= 0 or len(digest) != 64:
            raise ValueError(f"invalid V2 shard manifest entry {index}: count={count}, sha={digest!r}")
        result.append({**dict(raw), "path": str(path), "num_rollouts": count, "sha256": digest})
    return result


def _selection_episodes(manifest: Mapping[str, Any], name: str) -> list[int]:
    selection = manifest.get("selection", {})
    if not isinstance(selection, Mapping) or name not in selection:
        raise ValueError(f"V2 manifest selection lacks {name!r}")
    return sorted(int(value) for value in selection[name])


def _assert_file_identity(
    declared: Mapping[str, Any], current: Path, label: str, *, require_sha256: bool = True
) -> dict[str, Any]:
    """Bind a collector-declared identity to the exact current file."""
    current = current.expanduser().resolve()
    if not current.is_file():
        raise FileNotFoundError(current)
    stat = current.stat()
    actual = {
        "path": str(current),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if require_sha256:
        actual["sha256"] = sha256_file(current)
    for key, value in actual.items():
        # The shared checkpoint contract records byte length as ``size`` while
        # dataset/manifest identities use ``size_bytes``.  They carry the same
        # value; accept the checkpoint spelling without weakening the check.
        declared_value = declared.get("size") if key == "size_bytes" and key not in declared else declared.get(key)
        if declared_value != value:
            raise _contract_error(
                f"collector {label} identity mismatch", value, declared_value, key
            )
    return actual


def _validate_collector_audit_header(
    root: Path, manifest: Mapping[str, Any], total: int
) -> tuple[dict[str, Any], Path, Path]:
    """Require the collector's final semantic audit and reject stale audits."""
    validation_path = root / "validation.json"
    trace_path = root / "planner_trace.h5"
    capture_path = root / "capture_manifest.json"
    for path in (validation_path, trace_path, capture_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"formal collector validation artifact missing: expected={path}, actual=missing"
            )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    expected = {
        "valid": True,
        "scope": "formal",
        "num_rollouts": total,
        "num_source_episodes": 150,
        "formal_episode_overlap": 0,
        "measurement1_holdout_overlap": 0,
        "memory_global_exclusion_overlap": 0,
        "no_extra_action_clip": True,
        "under_40gb": True,
        "total_shard_bytes": int(manifest.get("storage", {}).get("total_shard_bytes", -1)),
    }
    mismatch = {
        key: {"expected": value, "actual": validation.get(key)}
        for key, value in expected.items()
        if validation.get(key) != value
    }
    if mismatch:
        raise _contract_error(
            "collector validation semantic counters failed", expected, mismatch, validation_path
        )
    if int(validation.get("decoded_jpeg_shards", 0)) < 1:
        raise _contract_error(
            "collector JPEG audit missing", ">=1", validation.get("decoded_jpeg_shards"), validation_path
        )
    trace_identity = _assert_file_identity(
        manifest.get("planner_capture", {}).get("trace", {}), trace_path, "planner trace"
    )
    _assert_file_identity(
        manifest.get("planner_capture", {}).get("capture_manifest", {}),
        capture_path,
        "capture manifest",
    )
    return {
        "path": str(validation_path),
        "sha256": sha256_file(validation_path),
        "mtime_ns": int(validation_path.stat().st_mtime_ns),
        "semantic_counters": expected,
        "decoded_jpeg_shards": int(validation["decoded_jpeg_shards"]),
        "trace": trace_identity,
    }, trace_path, capture_path


def load_planner_manifest(
    dataset_root: Path,
    *,
    expected_eval_episodes: Sequence[int],
    expected_measurement1_episodes: Sequence[int],
    expected_masked_checkpoint: Path,
    expected_formal_manifest: Path,
    expected_measurement1_segments: Path,
) -> dict[str, Any]:
    """Load V2 data and prove schema, provenance exclusions, and identities."""
    root = dataset_root.expanduser().resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise _contract_error(
            "V2 format version mismatch", FORMAT_VERSION, manifest.get("format_version"), path
        )
    if manifest.get("complete") is not True:
        raise _contract_error("V2 collection incomplete", True, manifest.get("complete"), path)

    actual_eval = _selection_episodes(manifest, "excluded_eval_episodes")
    expected_eval = sorted(int(value) for value in expected_eval_episodes)
    if actual_eval != expected_eval:
        raise _contract_error("V2 formal50 exclusion mismatch", expected_eval, actual_eval, path)
    actual_m1 = _selection_episodes(manifest, "measurement1_holdout_episodes")
    expected_m1 = sorted(set(int(value) for value in expected_measurement1_episodes))
    if actual_m1 != expected_m1:
        raise _contract_error(
            "V2 Measurement-1 exclusion mismatch", expected_m1, actual_m1, path
        )

    shards = _manifest_shards(manifest, root)
    declared = int(manifest.get("num_rollouts", manifest.get("storage", {}).get("num_rollouts", -1)))
    if manifest.get("scope") != "formal" or declared != 31_500:
        raise _contract_error(
            "V2 training requires formal collector output",
            {"scope": "formal", "num_rollouts": 31_500},
            {"scope": manifest.get("scope"), "num_rollouts": declared},
            path,
        )
    expected_pool_shape = [150, 10, 300, 5, 25]
    planner_capture = manifest.get("planner_capture", {})
    if planner_capture.get("complete_postinjection_pool_shape") != expected_pool_shape:
        raise _contract_error(
            "V2 planner pool shape mismatch",
            expected_pool_shape,
            planner_capture.get("complete_postinjection_pool_shape"),
            "manifest.planner_capture.complete_postinjection_pool_shape",
        )
    candidate_indices = np.asarray(
        planner_capture.get("physics_candidate_indices", []), dtype=np.int64
    )
    expected_candidates = np.asarray([0, *range(31, 51)], dtype=np.int64)
    if not np.array_equal(candidate_indices, expected_candidates):
        raise _contract_error(
            "V2 replay candidate indices mismatch",
            expected_candidates.tolist(),
            candidate_indices.tolist(),
            "manifest.planner_capture.physics_candidate_indices",
        )
    audit, trace_path, capture_path = _validate_collector_audit_header(root, manifest, declared)
    sources = manifest.get("sources", {})
    bound_inputs = {
        "masked_checkpoint": _assert_file_identity(
            sources.get("masked_checkpoint", {}), expected_masked_checkpoint, "Masked checkpoint"
        ),
        "formal_manifest": _assert_file_identity(
            sources.get("formal_manifest", {}), expected_formal_manifest, "formal manifest"
        ),
        "measurement1_segments": _assert_file_identity(
            sources.get("measurement1_segments", {}),
            expected_measurement1_segments,
            "Measurement-1 segments",
        ),
    }
    capture_payload = json.loads(capture_path.read_text(encoding="utf-8"))
    capture_trace = capture_payload.get("planner_trace", {})
    if capture_trace.get("sha256") != audit["trace"]["sha256"]:
        raise _contract_error(
            "capture manifest trace binding mismatch",
            audit["trace"]["sha256"],
            capture_trace.get("sha256"),
            capture_path,
        )
    capture_weights = capture_payload.get("checkpoint", {}).get("weights", {})
    if capture_weights.get("sha256") != bound_inputs["masked_checkpoint"]["sha256"]:
        raise _contract_error(
            "planner capture checkpoint binding mismatch",
            bound_inputs["masked_checkpoint"]["sha256"],
            capture_weights.get("sha256"),
            capture_path,
        )
    masked_config = expected_masked_checkpoint.parent / "config.json"
    bound_inputs["masked_config"] = _assert_file_identity(
        capture_payload.get("checkpoint", {}).get("config", {}),
        masked_config,
        "Masked config",
    )
    required = {
        "rollout_id": ((None,), np.int64),
        "source_row": ((None,), np.int64),
        "source_episode": ((None,), np.int64),
        "source_step": ((None,), None),
        "goal_row": ((None,), np.int64),
        "cem_iteration": ((None,), None),
        "candidate_index": ((None,), None),
        "source_index": ((None,), None),
        "pixels_jpeg": ((None, ROLLOUT_FRAMES), None),
        "action_env": ((None, 25, 5), np.float32),
        "action_model": ((None, ROLLOUT_ACTIONS, ACTION_DIM), np.float32),
        "block_pos": ((None, ROLLOUT_FRAMES, 3), np.float32),
        "block_quat": ((None, ROLLOUT_FRAMES, 4), np.float32),
        "block_yaw": ((None, ROLLOUT_FRAMES), np.float32),
    }
    eval_set = set(expected_eval)
    m1_set = set(expected_m1)
    source_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    total = 0
    global_excluded = set(
        _selection_episodes(manifest, "globally_excluded_memory_episodes")
    )
    validation_dependencies = [path, trace_path, capture_path]
    with h5py.File(trace_path, "r") as trace:
        if trace.attrs.get("format_version") != "cube_offpolicy_planner_trace_hdf5_v2":
            raise _contract_error(
                "planner trace format mismatch",
                "cube_offpolicy_planner_trace_hdf5_v2",
                trace.attrs.get("format_version"),
                trace_path,
            )
        if tuple(trace["candidates_normalized"].shape) != tuple(expected_pool_shape):
            raise _contract_error(
                "planner trace pool shape mismatch",
                expected_pool_shape,
                list(trace["candidates_normalized"].shape),
                trace_path,
            )
        retrieval = np.asarray(trace["retrieval_episodes"][:], dtype=np.int64)
        if retrieval.shape != (150, 10):
            raise _contract_error(
                "planner retrieval shape mismatch", [150, 10], list(retrieval.shape), trace_path
            )
        for source_index, row in enumerate(retrieval):
            if len(np.unique(row)) != 10:
                raise _contract_error(
                    "planner retrieval episodes not unique", 10, len(np.unique(row)), source_index
                )
        retrieval_leak = sorted(set(map(int, retrieval.reshape(-1))) & global_excluded)
        if retrieval_leak:
            raise _contract_error(
                "planner retrieval exclusion violation", [], retrieval_leak, trace_path
            )
        # Match collector _inverse_action exactly: float32 destination with
        # sequential in-place multiply/add from the float64 scaler values.
        mean = np.asarray(trace.attrs["action_normalizer_mean"], dtype=np.float64)
        scale = np.asarray(trace.attrs["action_normalizer_scale"], dtype=np.float64)
        if mean.shape != (5,) or scale.shape != (5,) or np.any(scale <= 0):
            raise ValueError("planner trace action normalizer is invalid")

        for shard_index, entry in enumerate(shards):
            shard = Path(entry["path"])
            if not shard.is_file():
                raise FileNotFoundError(shard)
            validation_dependencies.append(shard)
            # Formal training always binds every shard; there is deliberately
            # no skip-hash mode for this P1 integrity contract.
            actual_sha = sha256_file(shard)
            if actual_sha != entry["sha256"]:
                raise _contract_error(
                    "V2 shard digest mismatch", entry["sha256"], actual_sha, shard_index
                )
            entry["verified_sha256"] = actual_sha
            count = int(entry["num_rollouts"])
            if count != 210:
                raise _contract_error("V2 source shard size mismatch", 210, count, shard_index)
            with h5py.File(shard, "r") as h5:
                if h5.attrs.get("format_version") != FORMAT_VERSION:
                    raise _contract_error(
                        "V2 shard format mismatch",
                        FORMAT_VERSION,
                        h5.attrs.get("format_version"),
                        shard,
                    )
                missing = sorted(set(required).difference(h5.keys()))
                if missing:
                    raise ValueError(f"V2 shard {shard_index} missing datasets: {missing}")
                for name, (shape, dtype) in required.items():
                    expected_shape = tuple(count if value is None else value for value in shape)
                    if tuple(h5[name].shape) != expected_shape:
                        raise _contract_error(
                            f"V2 {name} shape mismatch",
                            expected_shape,
                            tuple(h5[name].shape),
                            shard,
                        )
                    if dtype is not None and np.dtype(h5[name].dtype) != np.dtype(dtype):
                        raise _contract_error(
                            f"V2 {name} dtype mismatch",
                            str(np.dtype(dtype)),
                            str(h5[name].dtype),
                            shard,
                        )
                jpeg_type = h5py.check_dtype(vlen=h5["pixels_jpeg"].dtype)
                if jpeg_type is None or np.dtype(jpeg_type) != np.dtype(np.uint8):
                    raise _contract_error(
                        "V2 JPEG dtype mismatch",
                        "vlen uint8",
                        str(h5["pixels_jpeg"].dtype),
                        shard,
                    )
                sources = np.asarray(h5["source_episode"][:], dtype=np.int64)
                unique_sources = np.unique(sources)
                trace_source_episode = int(trace["source_episodes"][shard_index])
                if unique_sources.tolist() != [trace_source_episode]:
                    raise _contract_error(
                        "V2 shard source episode not bound to planner trace",
                        [trace_source_episode],
                        unique_sources.tolist(),
                        shard,
                    )
                leaked = sorted(set(map(int, np.unique(sources))) & (eval_set | m1_set))
                if leaked:
                    raise _contract_error(
                        "V2 source episode leakage",
                        "no formal50 or Measurement-1 episode",
                        leaked,
                        shard,
                    )
                iterations = np.asarray(h5["cem_iteration"][:], dtype=np.int64)
                if np.any((iterations < 0) | (iterations >= 10)):
                    bad = int(np.flatnonzero((iterations < 0) | (iterations >= 10))[0])
                    raise _contract_error(
                        "V2 CEM iteration outside planner range",
                        "0..9",
                        int(iterations[bad]),
                        {"shard": shard_index, "local_rollout": bad},
                    )
                candidates = np.asarray(h5["candidate_index"][:], dtype=np.int64)
                expected_iterations = np.repeat(np.arange(10, dtype=np.int64), 21)
                expected_replayed = np.tile(expected_candidates, 10)
                if not np.array_equal(iterations, expected_iterations) or not np.array_equal(
                    candidates, expected_replayed
                ):
                    raise _contract_error(
                        "V2 10x21 replay structure mismatch",
                        "iteration repeat 0..9 x21 and candidates [0,31..50] each round",
                        {"iterations": iterations.tolist(), "candidates": candidates.tolist()},
                        shard,
                    )
                source_indices = np.asarray(h5["source_index"][:], dtype=np.int64)
                if not np.all(source_indices == shard_index):
                    raise _contract_error(
                        "V2 shard/source index mismatch", shard_index,
                        np.unique(source_indices).tolist(), shard,
                    )
                model_actions = np.asarray(h5["action_model"][:], dtype=np.float32)
                raw_actions = np.asarray(h5["action_env"][:], dtype=np.float32)
                inverse = model_actions.reshape(count, 25, 5).copy()
                inverse *= scale
                inverse += mean
                if not np.array_equal(inverse, raw_actions):
                    raise _contract_error(
                        "V2 planner inverse mismatch", "bitwise float32 inverse", "different",
                        {"shard": shard_index, "max_abs": float(np.max(np.abs(inverse - raw_actions)))},
                    )
                source_pool = np.asarray(
                    trace["candidates_normalized"][shard_index], dtype=np.float32
                )
                expected_model = source_pool[iterations, candidates]
                if not np.array_equal(model_actions, expected_model):
                    raise _contract_error(
                        "V2 action_model not bound to planner pool", "exact trace candidates", "different", shard
                    )
                for key in ("action_env", "action_model", "block_pos", "block_quat", "block_yaw"):
                    values = np.asarray(h5[key][:])
                    if not np.all(np.isfinite(values)):
                        bad = np.argwhere(~np.isfinite(values))[0].tolist()
                        raise _contract_error(
                            f"V2 {key} non-finite", "all finite", str(values[tuple(bad)]),
                            {"shard": shard_index, "index": bad},
                        )
                source_parts.append(sources)
                id_parts.append(np.asarray(h5["rollout_id"][:], dtype=np.int64))
            total += count

    if declared != total:
        raise _contract_error("V2 rollout count mismatch", total, declared, path)
    rollout_ids = np.concatenate(id_parts)
    if not np.array_equal(rollout_ids, np.arange(total, dtype=np.int64)):
        mismatch = int(np.flatnonzero(rollout_ids != np.arange(total, dtype=np.int64))[0])
        raise _contract_error(
            "V2 rollout ids are not contiguous", mismatch, int(rollout_ids[mismatch]), mismatch
        )
    source_episodes = np.concatenate(source_parts)
    if len(np.unique(source_episodes)) != 150:
        raise _contract_error(
            "V2 source episodes are not globally distinct",
            150,
            len(np.unique(source_episodes)),
            path,
        )
    newest_dependency = max(item.stat().st_mtime_ns for item in validation_dependencies)
    if audit["mtime_ns"] < newest_dependency:
        raise _contract_error(
            "collector validation is stale",
            f">={newest_dependency}",
            audit["mtime_ns"],
            audit["path"],
        )
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = sha256_file(path)
    manifest["shards"] = shards
    manifest["num_rollouts"] = total
    manifest["source_episode_by_rollout"] = source_episodes
    audit["manifest"] = {
        "path": str(path),
        "sha256": manifest["manifest_sha256"],
        "mtime_ns": int(path.stat().st_mtime_ns),
    }
    audit["bound_shards"] = [
        {"path": entry["path"], "sha256": entry["verified_sha256"]} for entry in shards
    ]
    audit["semantic_revalidation"] = {
        "source_shards": 150,
        "rollouts_per_source": 210,
        "cem_iterations": 10,
        "replayed_candidates_per_iteration": 21,
        "complete_pool_candidates_per_iteration": 300,
        "action_model_exact_planner_pool": True,
        "float32_inverse_exact": True,
        "retrieval_unique_episodes_per_source": 10,
        "retrieval_exclusion_overlap": 0,
    }
    manifest["collector_validation"] = audit
    manifest["bound_inputs"] = bound_inputs
    return manifest


def split_rollouts_by_source_episode(
    source_episode_by_rollout: Sequence[int], train_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Episode-grouped deterministic split; never split candidates of one state."""
    source = np.asarray(source_episode_by_rollout, dtype=np.int64)
    episodes = np.unique(source)
    if len(episodes) < 2 or not 0.0 < train_fraction < 1.0:
        raise ValueError("V2 episode split needs >=2 episodes and fraction in (0,1)")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    count = min(max(1, int(np.floor(len(episodes) * train_fraction))), len(episodes) - 1)
    train_episodes = np.sort(shuffled[:count])
    val_episodes = np.sort(shuffled[count:])
    train_ids = np.flatnonzero(np.isin(source, train_episodes)).astype(np.int64)
    val_ids = np.flatnonzero(np.isin(source, val_episodes)).astype(np.int64)
    if np.intersect1d(source[train_ids], source[val_ids]).size:
        raise RuntimeError("V2 source episode crossed train/validation split")
    return train_ids, val_ids, train_episodes, val_episodes


def _decode_jpeg(value: Any) -> torch.Tensor:
    payload = (
        value.astype(np.uint8, copy=False).tobytes()
        if isinstance(value, np.ndarray)
        else bytes(value)
    )
    with Image.open(io.BytesIO(payload)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"decoded V2 JPEG shape mismatch: {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class PlannerRolloutDataset(Dataset):
    """Lazy six-frame planner rollout dataset using exact captured model actions."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        rollout_ids: Sequence[int],
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.shards = [dict(value) for value in manifest["shards"]]
        self.rollout_ids = np.asarray(rollout_ids, dtype=np.int64)
        self.transform = transform
        counts = np.asarray([int(value["num_rollouts"]) for value in self.shards], dtype=np.int64)
        self.ends = np.cumsum(counts)
        if self.rollout_ids.ndim != 1 or np.any(self.rollout_ids < 0):
            raise ValueError("V2 rollout ids must be a non-negative vector")
        if self.rollout_ids.size and int(self.rollout_ids.max()) >= int(self.ends[-1]):
            raise ValueError("V2 rollout id exceeds manifest")
        self._pid: int | None = None
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return int(len(self.rollout_ids))

    def _handle(self, index: int) -> h5py.File:
        pid = os.getpid()
        if self._pid != pid:
            self.close()
            self._pid = pid
        if index not in self._handles:
            self._handles[index] = h5py.File(self.shards[index]["path"], "r")
        return self._handles[index]

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
        rollout_id = int(self.rollout_ids[index])
        shard_index = bisect_right(self.ends, rollout_id)
        start = 0 if shard_index == 0 else int(self.ends[shard_index - 1])
        local = rollout_id - start
        h5 = self._handle(shard_index)
        pixels = torch.stack([_decode_jpeg(h5["pixels_jpeg"][local, t]) for t in range(6)])
        action = torch.from_numpy(np.asarray(h5["action_model"][local], dtype=np.float32))
        sample: dict[str, Any] = {
            "pixels": pixels,
            "action": action,
            "rollout_id": torch.tensor(rollout_id, dtype=torch.int64),
            "source_episode": torch.tensor(int(h5["source_episode"][local]), dtype=torch.int64),
            "cem_iteration": torch.tensor(int(h5["cem_iteration"][local]), dtype=torch.int64),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        if tuple(sample["pixels"].shape) != (6, 3, 224, 224):
            raise RuntimeError(f"V2 pixel transform returned {tuple(sample['pixels'].shape)}")
        if tuple(sample["action"].shape) != (5, 25) or not torch.isfinite(sample["action"]).all():
            raise RuntimeError("V2 action transform violated [5,25] finite contract")
        return sample


class ExpertRolloutDataset(Dataset):
    """Six expert frames and exactly the five transition actions between them."""

    def __init__(self, dataset: Any, indices: Sequence[int], transform: Callable) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_index = int(self.indices[index])
        sample = self.transform(self.dataset[clip_index])
        sample["action"] = sample["action"][:5].clone()
        sample["expert_clip_index"] = torch.tensor(clip_index, dtype=torch.int64)
        sample["source_episode"] = torch.tensor(
            int(self.dataset.clip_indices[clip_index][0]), dtype=torch.int64
        )
        if tuple(sample["pixels"].shape) != (6, 3, 224, 224):
            raise RuntimeError("expert V2 pixels are not six frames")
        if tuple(sample["action"].shape) != (5, 25):
            raise RuntimeError("expert V2 actions are not five model steps")
        return sample


def _stack_samples(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples):
        raise ValueError("source samples have inconsistent keys")
    return {key: torch.stack([sample[key] for sample in samples]) for key in sorted(keys)}


class StrictMixtureDataset(Dataset):
    """Bundle samples so each optimizer batch has an exact frozen source ratio."""

    def __init__(
        self, expert: Dataset, v2: Dataset, expert_per_bundle: int, v2_per_bundle: int
    ) -> None:
        if not len(expert) or not len(v2) or expert_per_bundle < 1 or v2_per_bundle < 1:
            raise ValueError("strict V2 mixture sources/bundle counts must be positive")
        self.expert = expert
        self.v2 = v2
        self.expert_per_bundle = int(expert_per_bundle)
        self.v2_per_bundle = int(v2_per_bundle)

    def __len__(self) -> int:
        return max(
            (len(self.expert) + self.expert_per_bundle - 1) // self.expert_per_bundle,
            (len(self.v2) + self.v2_per_bundle - 1) // self.v2_per_bundle,
        )

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        expert = [
            self.expert[(index * self.expert_per_bundle + offset) % len(self.expert)]
            for offset in range(self.expert_per_bundle)
        ]
        v2 = [
            self.v2[(index * self.v2_per_bundle + offset) % len(self.v2)]
            for offset in range(self.v2_per_bundle)
        ]
        return {"expert": _stack_samples(expert), "v2": _stack_samples(v2)}


def flatten_bundled_source(source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collapse DataLoader [bundles, per_bundle, ...] to physical samples."""
    result: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if not torch.is_tensor(value) or value.ndim < 2:
            raise ValueError(f"bundled source field {key!r} lacks two bundle axes")
        result[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
    return result


def synthetic_v2_smoke() -> dict[str, Any]:
    """CPU proof for episode split, exact 96/32 bundling, and five-step gradients."""
    source = np.repeat(np.arange(10, dtype=np.int64), 21)
    train, val, train_eps, val_eps = split_rollouts_by_source_episode(source, 0.8, 3072)
    if np.intersect1d(source[train], source[val]).size:
        raise RuntimeError("synthetic V2 episode split overlaps")

    class ToyDataset(Dataset):
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "pixels": torch.full((6, 3), float(index)),
                "action": torch.full((5, 2), float(index)),
            }

    mixed = StrictMixtureDataset(ToyDataset(300), ToyDataset(100), 3, 1)
    batch = torch.utils.data.default_collate([mixed[i] for i in range(32)])
    expert = flatten_bundled_source(batch["expert"])
    v2 = flatten_bundled_source(batch["v2"])
    if expert["pixels"].shape[0] != 96 or v2["pixels"].shape[0] != 32:
        raise RuntimeError("synthetic strict V2 mixture ratio failed")

    torch.manual_seed(3072)
    predictor = torch.nn.Linear(6, 3)
    action_encoder = torch.nn.Linear(2, 3)
    pred_proj = torch.nn.Linear(3, 3)
    state = torch.randn(8, 1, 3)
    actions = torch.randn(8, 5, 2)
    targets = torch.randn(8, 5, 3)
    predictions = []
    current = state
    for depth in range(5):
        act = action_encoder(actions[:, depth : depth + 1])
        predicted = pred_proj(predictor(torch.cat((current[:, -1:], act), dim=-1)))
        predictions.append(predicted)
        current = torch.cat((current, predicted), dim=1)
    loss = (torch.cat(predictions, 1) - targets).square().mean()
    loss.backward()
    if predictor.weight.grad is None or not torch.count_nonzero(predictor.weight.grad):
        raise RuntimeError("synthetic autoregressive rollout has no predictor gradient")
    return {
        "status": "PASS",
        "rollout": {"frames": 6, "actions": 5, "intermediate_detach": False},
        "source_batch": {"expert": 96, "v2": 32},
        "source_episode_split": {
            "train_episodes": len(train_eps),
            "validation_episodes": len(val_eps),
            "overlap": 0,
        },
        "autoregressive_gradient_nonzero": True,
    }


__all__ = [
    "ACTION_DIM",
    "FORMAL_BATCH_SIZE",
    "FORMAL_EXPERT_BATCH",
    "FORMAL_V2_BATCH",
    "RETRY_EXPERT_BATCH",
    "RETRY_V2_BATCH",
    "ExpertRolloutDataset",
    "PlannerRolloutDataset",
    "StrictMixtureDataset",
    "assert_frozen_hashes",
    "flatten_bundled_source",
    "freeze_predictor_only",
    "frozen_state_hashes",
    "load_planner_manifest",
    "module_state_sha256",
    "sha256_file",
    "split_rollouts_by_source_episode",
    "synthetic_v2_smoke",
    "write_json",
]
