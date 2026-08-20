#!/usr/bin/env python3
"""Build and query the frozen OGBench-Cube trajectory-memory state index.

The index intentionally contains states and source-row provenance only.  The
25-step action seed is read from the immutable HDF5 dataset after retrieval,
then transformed with the dataset-wide action StandardScaler.  This avoids a
second ~0.9 GiB copy of action data while retaining the exact frozen contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "datasets/ogbench/cube_single_expert.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/memory_index/cube_expert_v1"
TMP_ROOT = PROJECT_ROOT.parent / "tmp"

FEATURE_NAMES = (
    "block_x",
    "block_y",
    "block_z",
    "sin_block_yaw",
    "cos_block_yaw",
    "ee_x",
    "ee_y",
    "ee_z",
    "gripper_opening",
)
NUM_FRAMES = 2_010_000
NUM_ANCHORS = 1_760_000
MAX_ANCHOR_STEP = 175
ACTION_STEPS = 25
ACTION_DIM = 5


def configure_storage() -> None:
    defaults = {
        "HF_HOME": str(PROJECT_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(PROJECT_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(PROJECT_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def safe_output(path: Path, overwrite: bool) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"refusing symlink output: {raw}")
    resolved = raw.resolve()
    root = (PROJECT_ROOT / "outputs/memory_index").resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a concrete child of {root}: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is nonempty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _slice_1d(dataset: Any, start: int, stop: int) -> np.ndarray:
    value = np.asarray(dataset[start:stop], dtype=np.float64)
    return value.reshape(stop - start, -1)[:, 0]


def feature_chunk(h5: Any, start: int, stop: int) -> np.ndarray:
    """Return raw frozen 9-D features in the specified row interval."""

    block = np.asarray(h5["privileged_block_0_pos"][start:stop], dtype=np.float64)
    yaw = _slice_1d(h5["privileged_block_0_yaw"], start, stop)
    ee = np.asarray(h5["proprio_effector_pos"][start:stop], dtype=np.float64)
    gripper = _slice_1d(h5["proprio_gripper_opening"], start, stop)
    result = np.column_stack(
        (block, np.sin(yaw), np.cos(yaw), ee, gripper)
    ).astype(np.float64, copy=False)
    if result.shape != (stop - start, len(FEATURE_NAMES)):
        raise RuntimeError(f"unexpected feature shape: {result.shape}")
    return result


def feature_from_state(
    block_position: np.ndarray,
    block_yaw: np.ndarray | float,
    effector_position: np.ndarray,
    gripper_opening: np.ndarray | float,
) -> np.ndarray:
    block = np.asarray(block_position, dtype=np.float64).reshape(-1, 3)
    yaw = np.asarray(block_yaw, dtype=np.float64).reshape(-1)
    ee = np.asarray(effector_position, dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(gripper_opening, dtype=np.float64).reshape(-1)
    n = block.shape[0]
    if yaw.size != n or ee.shape[0] != n or grip.size != n:
        raise ValueError("state feature batch dimensions do not agree")
    return np.column_stack((block, np.sin(yaw), np.cos(yaw), ee, grip))


def _chunks(total: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, size):
        yield start, min(start + size, total)


def _streaming_stats(h5: Any, total: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    mean = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    m2 = np.zeros_like(mean)
    for start, stop in _chunks(total, chunk_size):
        values = feature_chunk(h5, start, stop)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite state feature in rows [{start}, {stop})")
        n = len(values)
        chunk_mean = values.mean(axis=0, dtype=np.float64)
        chunk_m2 = np.square(values - chunk_mean).sum(axis=0, dtype=np.float64)
        delta = chunk_mean - mean
        new_count = count + n
        mean += delta * (n / new_count)
        m2 += chunk_m2 + np.square(delta) * count * n / new_count
        count = new_count
    if count != total:
        raise RuntimeError(f"stats count mismatch: {count} != {total}")
    std = np.sqrt(m2 / count)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"invalid population standard deviation: {std}")
    return mean, std


def build_index(dataset: Path, output: Path, chunk_size: int, overwrite: bool) -> None:
    configure_storage()
    dataset = dataset.expanduser().resolve()
    if not dataset.is_file() or PROJECT_ROOT.parent.resolve() not in dataset.parents:
        raise FileNotFoundError(f"dataset must exist on data disk: {dataset}")
    import hdf5plugin  # noqa: F401
    import h5py

    # Validate the immutable source before an intentional --overwrite can
    # remove an older index.  A missing/corrupt HDF5 must never destroy the
    # last good build.
    required = {
        "step_idx": (NUM_FRAMES,),
        "ep_idx": (NUM_FRAMES,),
        "action": (NUM_FRAMES, ACTION_DIM),
        "privileged_block_0_pos": (NUM_FRAMES, 3),
        "privileged_block_0_yaw": (NUM_FRAMES, 1),
        "proprio_effector_pos": (NUM_FRAMES, 3),
        "proprio_gripper_opening": (NUM_FRAMES, 1),
    }
    with h5py.File(dataset, "r", swmr=True) as h5:
        for key, shape in required.items():
            if key not in h5 or h5[key].shape != shape:
                actual = None if key not in h5 else h5[key].shape
                raise RuntimeError(
                    f"dataset field mismatch: key={key}, expected={shape}, actual={actual}"
                )
    output = safe_output(output, overwrite)

    with h5py.File(dataset, "r", swmr=True) as h5:
        total = int(h5["step_idx"].shape[0])
        if total != NUM_FRAMES:
            raise RuntimeError(f"frozen frame count mismatch: {total} != {NUM_FRAMES}")
        mean, std = _streaming_stats(h5, total, chunk_size)
        step = np.asarray(h5["step_idx"][:])
        anchor_rows = np.flatnonzero(step <= MAX_ANCHOR_STEP).astype(np.int64)
        if anchor_rows.shape != (NUM_ANCHORS,):
            raise RuntimeError(
                f"frozen anchor count mismatch: {len(anchor_rows)} != {NUM_ANCHORS}"
            )
        episodes = np.asarray(h5["ep_idx"][anchor_rows], dtype=np.int64)
        if not np.all(np.asarray(h5["ep_idx"][anchor_rows + ACTION_STEPS]) == episodes):
            raise RuntimeError("one or more 25-step memories cross episode boundary")
        np.save(output / "anchor_rows.npy", anchor_rows, allow_pickle=False)
        np.save(output / "anchor_episodes.npy", episodes, allow_pickle=False)
        features = np.lib.format.open_memmap(
            output / "anchor_features_z.npy",
            mode="w+",
            dtype=np.float64,
            shape=(NUM_ANCHORS, len(FEATURE_NAMES)),
        )
        cursor = 0
        # Rows are contiguous 0..175 inside each 201-row episode, but using
        # explicit row gathers makes that frozen property auditable.
        for begin, end in _chunks(NUM_ANCHORS, chunk_size):
            rows = anchor_rows[begin:end]
            contiguous_runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
            pieces = [feature_chunk(h5, int(run[0]), int(run[-1]) + 1) for run in contiguous_runs]
            raw = np.concatenate(pieces, axis=0)
            if raw.shape[0] != end - begin:
                raise RuntimeError("anchor feature gather length mismatch")
            features[begin:end] = (raw - mean) / std
            cursor = end
        features.flush()
        if cursor != NUM_ANCHORS or not np.isfinite(features).all():
            raise RuntimeError("anchor feature materialization failed")
        action = np.asarray(h5["action"][:])
        action = action[np.isfinite(action).all(axis=1)]
        if len(action) != 2_000_000:
            raise RuntimeError(
                f"frozen finite action count mismatch: {len(action)} != 2000000"
            )
        # Match formal evaluation exactly rather than reimplementing its
        # numerics with np.mean/std.
        from sklearn.preprocessing import StandardScaler

        action_scaler = StandardScaler().fit(action)
        action_mean = action_scaler.mean_.copy()
        action_std = action_scaler.scale_.copy()
        dataset_stat = dataset.stat()

    np.savez(
        output / "stats.npz",
        feature_mean=mean,
        feature_std=std,
        action_mean=action_mean,
        action_scale=action_std,
    )
    files = ("anchor_rows.npy", "anchor_episodes.npy", "anchor_features_z.npy", "stats.npz")
    metadata = {
        "format_version": 1,
        "dataset": {
            "path": str(dataset),
            "size_bytes": dataset_stat.st_size,
            "mtime_ns": dataset_stat.st_mtime_ns,
        },
        "feature_names": FEATURE_NAMES,
        "feature_dtype": "float64",
        "stats_population": "all_2_010_000_hdf5_frames_ddof0",
        "num_frames": NUM_FRAMES,
        "anchor_rule": "step_idx <= 175",
        "num_anchors": NUM_ANCHORS,
        "memory_action": "hdf5 action[row:row+25], StandardScaler transform per 5D action, C-order reshape (5,25)",
        "retrieval": "scipy.spatial.cKDTree exact eps=0; stable sort by (distance,row); 10 unique source episodes; exclude only current evaluation episode",
        "files": {name: {"sha256": sha256_file(output / name)} for name in files},
    }
    write_json(output / "metadata.json", metadata)
    print(output)


class CubeMemoryIndex:
    """Read-only exact cKDTree retriever over the frozen anchor array."""

    def __init__(self, root: Path, dataset: Path | None = None) -> None:
        from scipy.spatial import cKDTree

        self.root = root.expanduser().resolve()
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 1:
            raise ValueError("unsupported Cube memory index format")
        expected_files = {
            "anchor_rows.npy",
            "anchor_episodes.npy",
            "anchor_features_z.npy",
            "stats.npz",
        }
        file_metadata = self.metadata.get("files")
        if not isinstance(file_metadata, dict) or set(file_metadata) != expected_files:
            raise RuntimeError(
                "memory index file manifest mismatch: "
                f"expected={sorted(expected_files)}, actual={sorted(file_metadata or {})}"
            )
        for name, provenance in file_metadata.items():
            path = self.root / name
            actual = sha256_file(path)
            expected = provenance.get("sha256")
            if actual != expected:
                raise RuntimeError(
                    f"memory index hash mismatch: file={path}, expected={expected}, actual={actual}"
                )
        self.rows = np.load(self.root / "anchor_rows.npy", mmap_mode="r", allow_pickle=False)
        self.episodes = np.load(self.root / "anchor_episodes.npy", mmap_mode="r", allow_pickle=False)
        self.features = np.load(self.root / "anchor_features_z.npy", mmap_mode="r", allow_pickle=False)
        stats = np.load(self.root / "stats.npz", allow_pickle=False)
        self.feature_mean = np.asarray(stats["feature_mean"], dtype=np.float64)
        self.feature_std = np.asarray(stats["feature_std"], dtype=np.float64)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float64)
        self.action_scale = np.asarray(stats["action_scale"], dtype=np.float64)
        self.dataset = (dataset or Path(self.metadata["dataset"]["path"])).expanduser().resolve()
        dataset_stat = self.dataset.stat()
        identity = self.metadata["dataset"]
        if (
            dataset_stat.st_size != int(identity["size_bytes"])
            or dataset_stat.st_mtime_ns != int(identity["mtime_ns"])
        ):
            raise RuntimeError(
                "memory dataset identity mismatch: "
                f"expected_size/mtime={identity['size_bytes']}/{identity['mtime_ns']}, "
                f"actual={dataset_stat.st_size}/{dataset_stat.st_mtime_ns}"
            )
        expected_shapes = {
            "rows": ((NUM_ANCHORS,), self.rows),
            "episodes": ((NUM_ANCHORS,), self.episodes),
            "features": ((NUM_ANCHORS, len(FEATURE_NAMES)), self.features),
            "feature_mean": ((len(FEATURE_NAMES),), self.feature_mean),
            "feature_std": ((len(FEATURE_NAMES),), self.feature_std),
            "action_mean": ((ACTION_DIM,), self.action_mean),
            "action_scale": ((ACTION_DIM,), self.action_scale),
        }
        for label, (shape, value) in expected_shapes.items():
            if value.shape != shape:
                raise RuntimeError(
                    f"invalid memory index {label} shape: expected={shape}, actual={value.shape}"
                )
            if not np.isfinite(value).all():
                raise RuntimeError(f"non-finite memory index values: {label}")
        expected_dtypes = {
            "rows": np.dtype(np.int64),
            "episodes": np.dtype(np.int64),
            "features": np.dtype(np.float64),
        }
        for label, expected in expected_dtypes.items():
            actual = getattr(self, label).dtype
            if actual != expected:
                raise RuntimeError(
                    f"invalid memory index {label} dtype: expected={expected}, actual={actual}"
                )
        if np.any(self.feature_std <= 0) or np.any(self.action_scale <= 0):
            raise RuntimeError("memory index scale contains non-positive values")
        self.tree = cKDTree(self.features, copy_data=False, balanced_tree=True, compact_nodes=True)

    def normalize(self, raw_features: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_features, dtype=np.float64)
        if values.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
            raise ValueError(f"invalid query feature shape/value: {values.shape}")
        return (values - self.feature_mean) / self.feature_std

    def retrieve(
        self,
        raw_feature: np.ndarray,
        exclude_episode: int,
        count: int = 10,
    ) -> dict[str, np.ndarray]:
        raw_query = np.asarray(raw_feature, dtype=np.float64).reshape(1, -1)
        query = self.normalize(raw_query)[0]
        k = 64
        selected: list[tuple[float, int, int, int]] = []
        while True:
            distances, indices = self.tree.query(query, k=min(k, len(self.rows)), eps=0.0, workers=1)
            candidates = sorted(
                (
                    float(distance),
                    int(self.rows[int(index)]),
                    int(self.episodes[int(index)]),
                    int(index),
                )
                for distance, index in zip(np.atleast_1d(distances), np.atleast_1d(indices))
                if int(self.episodes[int(index)]) != int(exclude_episode)
            )
            selected = []
            seen: set[int] = set()
            for item in candidates:
                if item[2] in seen:
                    continue
                seen.add(item[2])
                selected.append(item)
                if len(selected) == count:
                    break
            if len(selected) == count:
                # cKDTree does not promise row order for equal-distance items.
                # Materialize the complete closed ball at the provisional
                # tenth-source distance, then apply frozen (distance,row)
                # ordering ourselves so a k boundary tie cannot change output.
                cutoff = float(selected[-1][0])
                ball = self.tree.query_ball_point(
                    query, r=np.nextafter(cutoff, np.inf), eps=0.0, workers=1
                )
                complete = sorted(
                    (
                        float(np.linalg.norm(self.features[int(index)] - query)),
                        int(self.rows[int(index)]),
                        int(self.episodes[int(index)]),
                        int(index),
                    )
                    for index in ball
                    if int(self.episodes[int(index)]) != int(exclude_episode)
                )
                selected = []
                seen = set()
                for item in complete:
                    if item[2] in seen:
                        continue
                    seen.add(item[2])
                    selected.append(item)
                    if len(selected) == count:
                        break
                if len(selected) == count:
                    break
            if k >= len(self.rows):
                raise RuntimeError(
                    "insufficient unique source episodes: "
                    f"expected={count}, actual={len(selected)}, "
                    f"query_position={raw_query[0].tolist()}"
                )
            k = min(k * 2, len(self.rows))
        selected.sort(key=lambda item: (item[0], item[1]))
        return {
            "distances": np.asarray([x[0] for x in selected], dtype=np.float64),
            "rows": np.asarray([x[1] for x in selected], dtype=np.int64),
            "episodes": np.asarray([x[2] for x in selected], dtype=np.int64),
            "steps": np.asarray([x[1] % 201 for x in selected], dtype=np.int64),
            "anchor_indices": np.asarray([x[3] for x in selected], dtype=np.int64),
        }

    def action_seed_bundle(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        import hdf5plugin  # noqa: F401
        import h5py

        rows = np.asarray(rows, dtype=np.int64)
        raw_actions = []
        normalized_actions = []
        with h5py.File(self.dataset, "r", swmr=True) as h5:
            for row in rows:
                ep = int(h5["ep_idx"][row])
                block = np.asarray(h5["action"][row : row + ACTION_STEPS], dtype=np.float64)
                if block.shape != (ACTION_STEPS, ACTION_DIM):
                    raise RuntimeError(f"incomplete memory action at row {row}")
                if int(h5["ep_idx"][row + ACTION_STEPS - 1]) != ep or not np.isfinite(block).all():
                    raise RuntimeError(f"invalid memory action at row {row}")
                normalized = (block - self.action_mean) / self.action_scale
                raw_actions.append(block.astype(np.float32))
                normalized_actions.append(
                    normalized.astype(np.float32).reshape(5, 25, order="C")
                )
        return {
            "raw": np.stack(raw_actions),
            "normalized": np.stack(normalized_actions),
        }

    def action_seeds(self, rows: np.ndarray) -> np.ndarray:
        return self.action_seed_bundle(rows)["normalized"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exact Cube trajectory-memory state index")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    build_index(args.dataset, args.output, args.chunk_size, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
