#!/usr/bin/env python3
"""Audit and render the official OGBench Cube play data for play-v1.

This utility never steps the simulator with an action.  Every image is made by
restoring one official qpos/qvel pair and rendering that state.  The conversion
keeps the official train/validation episode split and stores phase-0 frames at
raw steps ``0, 5, ..., 1000``.  Each non-terminal stored frame carries the
*exact* next five raw actions, flattened to 25 values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hdf5plugin  # noqa: F401  # register filters before h5py
import h5py
import numpy as np
from PIL import Image


AILAB = Path(__file__).resolve().parents[2]
DATA_ROOT = AILAB / "datasets/ogbench_play"
SOURCE_ROOT = DATA_ROOT / "source"
TRAIN_NPZ = SOURCE_ROOT / "cube-single-play-v0.npz"
VAL_NPZ = SOURCE_ROOT / "cube-single-play-v0-val.npz"
EXPERT_H5 = AILAB / "datasets/ogbench/cube_single_expert.h5"
FORMAL_MANIFEST = AILAB / "outputs/audit/cube_cem_manifest.json"
M1_SEGMENTS = AILAB / "outputs/eval/cube/imagination_error/measurement1_segments.json"
TMP_ROOT = AILAB.parent / "tmp"

FORMAT_VERSION = "cube_play_v1_dataset_v1"
RAW_EP_LEN = 1001
RAW_ACTION_DIM = 5
SAMPLE_STRIDE = 5
STORED_EP_LEN = 201
NUM_FRAMES = 4
EXPECTED_SPLITS = {"train": 1000, "val": 100}
EXPECTED_SOURCE_SHA256 = {
    "train": "80f3b6fd27f4f9d9e9eb6f0d07d6951559012f45b1e15ea4046ef8ecd8d3684e",
    "val": "96d07401bdebdc3f0ea6d56ed1333863e0962f441483adc2c43b83105046eb00",
}
SOURCE_OFFICIAL_REPOSITORY = "https://rail.eecs.berkeley.edu/datasets/ogbench/"
SOURCE_TRANSPORT_MIRROR = "https://huggingface.co/datasets/ryanhoangt/ogbench_data"
SOURCE_TRANSPORT_REVISION = "0290b1be6721a8750c77334c316aca998ba4aa8b"
XYZ_CENTER = np.asarray([0.425, 0.0, 0.0], dtype=np.float32)
XYZ_SCALE = 10.0
ACTION_BINS = np.linspace(-1.0, 1.0, 81, dtype=np.float64)
HASH_COLUMNS = ("qpos", "qvel", "action")
QUANTIZATION = 1e-3


def configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(AILAB),
        "HF_HOME": str(AILAB.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        os.environ.setdefault(key, value)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, sha256: bool = True) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if sha256:
        result["sha256"] = sha256_file(path)
    return result


def source_path(split: str) -> Path:
    return TRAIN_NPZ if split == "train" else VAL_NPZ


def shard_path(split: str) -> Path:
    return DATA_ROOT / f"cube_single_play_{split}_phase0.h5"


def validate_source(split: str, *, verify_sha256: bool = True) -> dict[str, np.ndarray]:
    path = source_path(split)
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify_sha256:
        actual = sha256_file(path)
        if actual != EXPECTED_SOURCE_SHA256[split]:
            raise RuntimeError(
                f"source SHA256 mismatch for {split}: expected={EXPECTED_SOURCE_SHA256[split]}, actual={actual}"
            )
    # NPZ members are zip-compressed, so NumPy's mmap_mode does not apply.
    # Materialize every member once; repeated ``NpzFile.__getitem__`` calls
    # would otherwise decompress a full 80-110 MiB member for every frame.
    with np.load(path) as archive:
        files = list(archive.files)
        data = {key: archive[key] for key in files}
    expected_rows = EXPECTED_SPLITS[split] * RAW_EP_LEN
    expected = {
        "observations": ((expected_rows, 28), np.dtype("float32")),
        "actions": ((expected_rows, 5), np.dtype("float32")),
        "terminals": ((expected_rows,), np.dtype("bool")),
        "qpos": ((expected_rows, 21), np.dtype("float32")),
        "qvel": ((expected_rows, 20), np.dtype("float32")),
    }
    if set(data) != set(expected):
        raise RuntimeError(f"unexpected {split} NPZ keys: {sorted(data)}")
    for key, (shape, dtype) in expected.items():
        value = data[key]
        if value.shape != shape or value.dtype != dtype:
            raise RuntimeError(
                f"{split}.{key} contract mismatch: expected={shape}/{dtype}, actual={value.shape}/{value.dtype}"
            )
    terminal_rows = np.flatnonzero(data["terminals"])
    wanted = np.arange(RAW_EP_LEN - 1, expected_rows, RAW_EP_LEN)
    if not np.array_equal(terminal_rows, wanted):
        raise RuntimeError(f"{split} terminal layout is not {EXPECTED_SPLITS[split]}x{RAW_EP_LEN}")
    if not all(np.isfinite(data[key]).all() for key in ("observations", "actions", "qpos", "qvel")):
        raise RuntimeError(f"{split} source contains non-finite values")
    return data


def geometry_from_observation(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape[-1] != 28:
        raise ValueError(f"single-Cube observation must end in 28, got {observation.shape}")
    ee = observation[..., 12:15] / XYZ_SCALE + XYZ_CENTER
    block = observation[..., 19:22] / XYZ_SCALE + XYZ_CENTER
    return block.astype(np.float32, copy=False), ee.astype(np.float32, copy=False)


class StreamingMoments:
    def __init__(self, width: int, bins: np.ndarray | None = None) -> None:
        self.width = int(width)
        self.count = np.zeros(width, dtype=np.int64)
        self.total = np.zeros(width, dtype=np.float64)
        self.total_sq = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)
        self.bins = bins
        self.hist = (
            np.zeros((width, len(bins) - 1), dtype=np.int64) if bins is not None else None
        )

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        for dim in range(self.width):
            finite = values[:, dim][np.isfinite(values[:, dim])]
            if not len(finite):
                continue
            self.count[dim] += len(finite)
            self.total[dim] += finite.sum(dtype=np.float64)
            self.total_sq[dim] += np.square(finite).sum(dtype=np.float64)
            self.minimum[dim] = min(self.minimum[dim], float(finite.min()))
            self.maximum[dim] = max(self.maximum[dim], float(finite.max()))
            if self.hist is not None:
                self.hist[dim] += np.histogram(finite, bins=self.bins)[0]

    def summary(self) -> dict[str, Any]:
        denom = np.maximum(self.count, 1)
        mean = self.total / denom
        variance = np.maximum(0.0, self.total_sq / denom - np.square(mean))
        return {
            "finite_count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": np.sqrt(variance),
            "histogram_bins": self.bins,
            "histogram_counts": self.hist,
        }


def histogram_comparison(play: StreamingMoments, expert: StreamingMoments) -> dict[str, Any]:
    if play.hist is None or expert.hist is None or play.bins is None:
        raise ValueError("histograms unavailable")
    overlaps, js = [], []
    for dim in range(play.width):
        p = play.hist[dim].astype(np.float64)
        q = expert.hist[dim].astype(np.float64)
        p /= max(p.sum(), 1.0)
        q /= max(q.sum(), 1.0)
        overlaps.append(float(np.minimum(p, q).sum()))
        midpoint = 0.5 * (p + q)
        p_mask, q_mask = p > 0, q > 0
        p_kl = float(np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask])))
        q_kl = float(np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask])))
        js.append(0.5 * (p_kl + q_kl))
    return {
        "histogram_range": [float(play.bins[0]), float(play.bins[-1])],
        "bins": len(play.bins) - 1,
        "overlap_coefficient_per_action_dim": overlaps,
        "overlap_coefficient_mean": float(np.mean(overlaps)),
        "jensen_shannon_divergence_bits_per_action_dim": js,
        "jensen_shannon_divergence_bits_mean": float(np.mean(js)),
    }


def _canonical_float32(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    # Canonicalize signed zero and all NaNs for source-dtype-independent hashing.
    value = value.copy()
    value[value == 0] = 0
    value[np.isnan(value)] = np.float32(np.nan)
    return np.ascontiguousarray(value)


def exact_episode_hash(columns: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in columns:
        value = _canonical_float32(value)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def quantized_signature(columns: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in columns:
        value = _canonical_float32(value)
        anchors = value[np.asarray([0, len(value) // 2, len(value) - 1])]
        finite = np.nan_to_num(
            anchors,
            nan=1e6,
            posinf=1e6 - 1,
            neginf=-1e6 + 1,
        )
        quantized = np.rint(finite / QUANTIZATION).astype(np.int32)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(quantized.shape, dtype=np.int64).tobytes())
        digest.update(quantized.tobytes(order="C"))
    return digest.hexdigest()


def play_episode_hashes(data: Mapping[str, np.ndarray]) -> tuple[list[str], list[str]]:
    exact, near = [], []
    episodes = len(data["terminals"]) // RAW_EP_LEN
    for episode in range(episodes):
        start = episode * RAW_EP_LEN
        stop = start + RAW_EP_LEN
        columns = [
            ("qpos", data["qpos"][start:stop]),
            ("qvel", data["qvel"][start:stop]),
            ("action", data["actions"][start:stop]),
        ]
        exact.append(exact_episode_hash(columns))
        near.append(quantized_signature(columns))
    return exact, near


def expert_hashes(
    formal_episodes: set[int], measurement1_episodes: set[int]
) -> dict[str, set[str]]:
    pools = {
        "exact_all": set(),
        "near_all": set(),
        "exact_formal50": set(),
        "near_formal50": set(),
        "exact_measurement1": set(),
        "near_measurement1": set(),
    }
    with h5py.File(EXPERT_H5, "r", swmr=True) as h5:
        lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        for episode, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
            start, stop = int(offset), int(offset + length)
            columns = [(name, h5[name][start:stop]) for name in HASH_COLUMNS]
            exact = exact_episode_hash(columns)
            near = quantized_signature(columns)
            pools["exact_all"].add(exact)
            pools["near_all"].add(near)
            if episode in formal_episodes:
                pools["exact_formal50"].add(exact)
                pools["near_formal50"].add(near)
            if episode in measurement1_episodes:
                pools["exact_measurement1"].add(exact)
                pools["near_measurement1"].add(near)
    return pools


def intersection_count(values: Iterable[str], pool: set[str]) -> int:
    return sum(value in pool for value in values)


def summarize_play_sources(data_by_split: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    action = StreamingMoments(5, ACTION_BINS)
    block = StreamingMoments(3)
    ee = StreamingMoments(3)
    rows = 0
    for data in data_by_split.values():
        rows += len(data["actions"])
        action.update(data["actions"])
        block_values, ee_values = geometry_from_observation(data["observations"])
        block.update(block_values)
        ee.update(ee_values)
    return {
        "episodes": sum(EXPECTED_SPLITS.values()),
        "frames": rows,
        "episode_length": RAW_EP_LEN,
        "actions": action.summary(),
        "block_pos_m": block.summary(),
        "ee_pos_m": ee.summary(),
        "success": {
            "status": "N/A",
            "reason": "official state NPZ has no success or reward field",
            "available_fields": sorted(next(iter(data_by_split.values()))),
        },
    }


def summarize_expert() -> tuple[dict[str, Any], StreamingMoments]:
    action = StreamingMoments(5, ACTION_BINS)
    block = StreamingMoments(3)
    ee = StreamingMoments(3)
    rows = 0
    success_count = 0
    with h5py.File(EXPERT_H5, "r", swmr=True) as h5:
        rows = len(h5["action"])
        for start in range(0, rows, 131_072):
            stop = min(start + 131_072, rows)
            action.update(h5["action"][start:stop])
            block.update(h5["privileged_block_0_pos"][start:stop])
            ee.update(h5["proprio_effector_pos"][start:stop])
            success_count += int(np.asarray(h5["success"][start:stop], dtype=bool).sum())
        episodes = len(h5["ep_len"])
    result = {
        "episodes": episodes,
        "frames": rows,
        "actions": action.summary(),
        "block_pos_m": block.summary(),
        "ee_pos_m": ee.summary(),
        "success_frame_fraction": success_count / rows,
    }
    return result, action


def range_overlap(play: Mapping[str, Any], expert: Mapping[str, Any]) -> dict[str, Any]:
    p_min, p_max = np.asarray(play["min"]), np.asarray(play["max"])
    e_min, e_max = np.asarray(expert["min"]), np.asarray(expert["max"])
    intersection = np.maximum(0.0, np.minimum(p_max, e_max) - np.maximum(p_min, e_min))
    union = np.maximum(p_max, e_max) - np.minimum(p_min, e_min)
    return {
        "range_iou_per_dimension": np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        ),
        "play_range_contained_by_expert_fraction_per_dimension": np.divide(
            intersection,
            p_max - p_min,
            out=np.zeros_like(intersection),
            where=(p_max - p_min) > 0,
        ),
    }


def command_audit(args: argparse.Namespace) -> None:
    data = {split: validate_source(split) for split in ("train", "val")}
    play_summary = summarize_play_sources(data)
    expert_summary, expert_actions = summarize_expert()

    formal = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    formal_rows = np.asarray(formal["formal_rows"], dtype=np.int64)
    m1 = json.loads(M1_SEGMENTS.read_text(encoding="utf-8"))
    measurement1_episodes = set(map(int, m1["episode_indices"]))
    with h5py.File(EXPERT_H5, "r", swmr=True) as h5:
        formal_episodes = set(map(int, h5["ep_idx"][formal_rows]))
    pools = expert_hashes(formal_episodes, measurement1_episodes)

    play_exact, play_near = [], []
    for split in ("train", "val"):
        exact, near = play_episode_hashes(data[split])
        play_exact.extend(exact)
        play_near.extend(near)
    exclusion = {
        "play_episode_namespace": "official_play_train_val",
        "exact_hash_contract": "SHA256 canonical-float32 complete qpos+qvel+action per episode including tensor shapes",
        "quantized_signature_contract": "SHA256 qpos+qvel+action at head/middle/tail after 1e-3 quantization",
        "quantization": QUANTIZATION,
        "play_episode_count": len(play_exact),
        "expert_episode_count": len(pools["exact_all"]),
        "formal50_episode_count": len(formal_episodes),
        "measurement1_unique_episode_count": len(measurement1_episodes),
        "exact_episode_hash_overlap_with_expert": intersection_count(play_exact, pools["exact_all"]),
        "exact_episode_hash_overlap_with_formal50": intersection_count(play_exact, pools["exact_formal50"]),
        "exact_episode_hash_overlap_with_measurement1": intersection_count(play_exact, pools["exact_measurement1"]),
        "quantized_signature_overlap_with_expert": intersection_count(play_near, pools["near_all"]),
        "quantized_signature_overlap_with_formal50": intersection_count(play_near, pools["near_formal50"]),
        "quantized_signature_overlap_with_measurement1": intersection_count(play_near, pools["near_measurement1"]),
        "independent_collection_claim": (
            "OGBench labels play and expert as separately collected variants; this provenance statement is report-only, "
            "while the zero exact/quantized intersections above are the measured checks"
        ),
    }
    required_zero = [
        "exact_episode_hash_overlap_with_expert",
        "exact_episode_hash_overlap_with_formal50",
        "exact_episode_hash_overlap_with_measurement1",
        "quantized_signature_overlap_with_expert",
        "quantized_signature_overlap_with_formal50",
    ]
    if any(exclusion[key] != 0 for key in required_zero):
        raise RuntimeError(f"play/expert exclusion failed: {exclusion}")

    play_action_obj = StreamingMoments(5, ACTION_BINS)
    for value in data.values():
        play_action_obj.update(value["actions"])
    overlap = {
        "action": histogram_comparison(play_action_obj, expert_actions),
        "block_pos": range_overlap(play_summary["block_pos_m"], expert_summary["block_pos_m"]),
        "ee_pos": range_overlap(play_summary["ee_pos_m"], expert_summary["ee_pos_m"]),
    }
    output = {
        "format_version": FORMAT_VERSION,
        "created_unix": time.time(),
        "sources": {
            split: {
                **file_identity(source_path(split)),
                "official_split": split,
                "official_repository": SOURCE_OFFICIAL_REPOSITORY,
                "transport_mirror": SOURCE_TRANSPORT_MIRROR,
                "filename": source_path(split).name,
                "transport_revision": SOURCE_TRANSPORT_REVISION,
            }
            for split in ("train", "val")
        },
        "play": play_summary,
        "expert": {**expert_summary, "path": str(EXPERT_H5.resolve())},
        "distribution_overlap": overlap,
        "exclusion": exclusion,
    }
    write_json(DATA_ROOT / "health_report.json", output)
    write_health_markdown(output, DATA_ROOT / "PLAY_DATA_HEALTH.md")
    write_health_plots(output, DATA_ROOT / "qc")
    print(json.dumps({"status": "PASS", "output": str((DATA_ROOT / 'health_report.json').resolve())}, indent=2))


def write_health_plots(report: Mapping[str, Any], output: Path) -> None:
    """Materialize the requested play-vs-expert action histograms and state ranges."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    play = report["play"]
    expert = report["expert"]
    bins = np.asarray(play["actions"]["histogram_bins"], dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    play_hist = np.asarray(play["actions"]["histogram_counts"], dtype=np.float64)
    expert_hist = np.asarray(expert["actions"]["histogram_counts"], dtype=np.float64)
    play_hist /= np.maximum(play_hist.sum(axis=1, keepdims=True), 1.0)
    expert_hist /= np.maximum(expert_hist.sum(axis=1, keepdims=True), 1.0)
    figure, axes = plt.subplots(1, 5, figsize=(18, 3.2), sharey=True)
    for dim, axis in enumerate(axes):
        axis.step(centers, expert_hist[dim], where="mid", label="expert", linewidth=1.2)
        axis.step(centers, play_hist[dim], where="mid", label="play", linewidth=1.2)
        axis.set_title(f"action[{dim}]")
        axis.set_xlabel("value")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("probability / bin")
    axes[-1].legend(frameon=False)
    figure.suptitle("OGBench Cube actions: official play vs expert")
    figure.tight_layout()
    figure.savefig(output / "action_histogram_play_vs_expert.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ("x", "y", "z")
    for axis, key, title in zip(
        axes, ("block_pos_m", "ee_pos_m"), ("Block position range", "End-effector position range"), strict=True
    ):
        for index, source_name in enumerate(("expert", "play")):
            value = report[source_name][key]
            minimum = np.asarray(value["min"])
            maximum = np.asarray(value["max"])
            center = 0.5 * (minimum + maximum)
            error = np.stack((center - minimum, maximum - center))
            axis.errorbar(np.arange(3) + (index - 0.5) * 0.08, center, yerr=error, fmt="o", capsize=4, label=source_name)
        axis.set_xticks(range(3), labels)
        axis.set_ylabel("metres")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "state_coverage_play_vs_expert.png", dpi=160)
    plt.close(figure)


def make_replay_env() -> Any:
    configure_storage()
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


def restore_and_render(env: Any, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = env.unwrapped
    raw.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
    info = raw.compute_ob_info()
    pixel = np.asarray(env.render(), dtype=np.uint8).copy()
    return pixel, {
        "observation": np.asarray(raw.compute_observation(), dtype=np.float32),
        "block_pos": np.asarray(info["privileged/block_0_pos"], dtype=np.float32),
        "ee_pos": np.asarray(info["proprio/effector_pos"], dtype=np.float32),
    }


def pixel_compression() -> dict[str, Any]:
    return hdf5plugin.Blosc(
        cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE
    )


def command_smoke(args: argparse.Namespace) -> None:
    qc = DATA_ROOT / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    for pattern in ("train_ep*_step*.png", "val_ep*_step*.png"):
        for stale in qc.glob(pattern):
            stale.unlink()
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    env = make_replay_env()
    started = time.perf_counter()
    try:
        env.reset(seed=42)
        for split in ("train", "val"):
            data = validate_source(split)
            episode_count = EXPECTED_SPLITS[split]
            for index in range(10):
                episode = int(round(index * (episode_count - 1) / 9))
                # QC rows are members of the formal phase-0 shard so finalize
                # can prove pixel alignment by exact byte hash.
                step = int(round(index * (STORED_EP_LEN - 1) / 9)) * SAMPLE_STRIDE
                row = episode * RAW_EP_LEN + step
                pixel, state = restore_and_render(env, data["qpos"][row], data["qvel"][row])
                block_expected, ee_expected = geometry_from_observation(data["observations"][row])
                observation_error = float(np.max(np.abs(state["observation"] - data["observations"][row])))
                block_error = float(np.linalg.norm(state["block_pos"] - block_expected))
                ee_error = float(np.linalg.norm(state["ee_pos"] - ee_expected))
                # qpos/qvel are distributed as float32.  Recomputing robot
                # forward kinematics in MuJoCo therefore differs slightly from
                # the collector's pre-quantization observation; cube free-joint
                # coordinates remain effectively exact.
                # Contact flags and a few robot-derived state channels are not
                # reproducible from the public float32 qpos/qvel alone, so the
                # full-observation delta is diagnostic only.  The visual scene
                # is guarded by exact cube geometry and a millimetre-level EE
                # continuity check.
                if block_error > 2e-5 or ee_error > 2e-3:
                    raise RuntimeError(
                        f"set_state geometry mismatch {split}/{episode}/{step}: "
                        f"ob={observation_error}, block={block_error}, ee={ee_error}"
                    )
                name = f"{split}_ep{episode:04d}_step{step:04d}.png"
                Image.fromarray(pixel).save(qc / name, optimize=True)
                frames.append(pixel)
                rows.append(
                    {
                        "split": split,
                        "episode": episode,
                        "raw_step": step,
                        "source_row": row,
                        "png": name,
                        "observation_max_abs_error": observation_error,
                        "block_pos_error_m": block_error,
                        "ee_pos_error_m": ee_error,
                        "pixel_sha256": hashlib.sha256(pixel.tobytes()).hexdigest(),
                    }
                )
    finally:
        env.close()
    elapsed = time.perf_counter() - started

    contact = Image.new("RGB", (5 * 224, 4 * 224), color=(0, 0, 0))
    for index, frame in enumerate(frames):
        contact.paste(Image.fromarray(frame), ((index % 5) * 224, (index // 5) * 224))
    contact.save(qc / "render_smoke_contact_sheet.png", optimize=True)

    smoke_h5 = qc / "render_smoke_20.h5"
    with h5py.File(smoke_h5, "w") as h5:
        h5.create_dataset(
            "pixels",
            data=np.stack(frames),
            chunks=(min(20, len(frames)), 224, 224, 3),
            **pixel_compression(),
        )
    nonpixel_bytes = (25 + 28 + 21 + 20 + 3 + 3) * 4 + (4 + 4 + 1)
    projected_frames = sum(EXPECTED_SPLITS.values()) * STORED_EP_LEN
    compressed_per_frame = smoke_h5.stat().st_size / len(frames)
    projected_bytes = int(projected_frames * (compressed_per_frame + nonpixel_bytes))
    report = {
        "format_version": FORMAT_VERSION,
        "status": "PASS",
        "no_action_rollout": True,
        "num_frames": len(frames),
        "elapsed_seconds": elapsed,
        "frames_per_second": len(frames) / elapsed,
        "projected_formal_frames": projected_frames,
        "projected_formal_seconds": elapsed / len(frames) * projected_frames,
        "projected_h5_bytes_from_20_frame_sample": projected_bytes,
        "projected_h5_gib": projected_bytes / (1024**3),
        "projection_note": "20-frame compressed sample; formal ratio may differ with 201-frame episode chunks",
        "compression": "Blosc zstd clevel=5 bitshuffle",
        "geometry_tolerance": {
            "observation_max_abs": "report_only",
            "block_l2_m": 2e-5,
            "ee_l2_m": 2e-3,
            "reason": "official qpos/qvel are float32; contact-derived observation channels are not exactly reconstructable",
        },
        "rows": rows,
        "smoke_h5": file_identity(smoke_h5),
        "contact_sheet": file_identity(qc / "render_smoke_contact_sheet.png"),
    }
    write_json(qc / "smoke_report.json", report)
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))


def create_shard_schema(h5: h5py.File, rows: int, episodes: int) -> None:
    h5.attrs["format_version"] = FORMAT_VERSION
    h5.attrs["raw_sampling_phase"] = 0
    h5.attrs["raw_sampling_stride"] = SAMPLE_STRIDE
    h5.attrs["no_action_rollout"] = True
    h5.create_dataset(
        "pixels", shape=(rows, 224, 224, 3), dtype=np.uint8,
        chunks=(16, 224, 224, 3), **pixel_compression()
    )
    numeric = {
        "action_block": ((rows, 25), np.float32),
        "action_block_valid": ((rows,), np.bool_),
        "observation": ((rows, 28), np.float32),
        "qpos": ((rows, 21), np.float32),
        "qvel": ((rows, 20), np.float32),
        "block_pos": ((rows, 3), np.float32),
        "ee_pos": ((rows, 3), np.float32),
        "ep_idx": ((rows,), np.int32),
        "step_idx": ((rows,), np.int32),
        "source_row": ((rows,), np.int64),
        "ep_offset": ((episodes,), np.int64),
        "ep_len": ((episodes,), np.int32),
    }
    for name, (shape, dtype) in numeric.items():
        chunks = True if rows in shape and rows > 1 else None
        h5.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)


def command_convert(args: argparse.Namespace) -> None:
    split = args.split
    source = validate_source(split)
    output = shard_path(split)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite intentionally")
    if output.is_symlink():
        raise RuntimeError(f"refusing symlink output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    episodes = EXPECTED_SPLITS[split]
    global_episode_offset = 0 if split == "train" else EXPECTED_SPLITS["train"]
    total = episodes * STORED_EP_LEN
    sampled_steps = np.arange(0, RAW_EP_LEN, SAMPLE_STRIDE, dtype=np.int64)
    if len(sampled_steps) != STORED_EP_LEN:
        raise AssertionError(sampled_steps)
    env = make_replay_env()
    start_time = time.perf_counter()
    try:
        env.reset(seed=42)
        with h5py.File(temporary, "w", libver="latest") as h5:
            create_shard_schema(h5, total, episodes)
            for episode in range(episodes):
                source_start = episode * RAW_EP_LEN
                output_start = episode * STORED_EP_LEN
                raw_rows = source_start + sampled_steps
                observations = np.asarray(source["observations"][raw_rows], dtype=np.float32)
                qpos = np.asarray(source["qpos"][raw_rows], dtype=np.float32)
                qvel = np.asarray(source["qvel"][raw_rows], dtype=np.float32)
                block, ee = geometry_from_observation(observations)
                blocks = np.zeros((STORED_EP_LEN, 25), dtype=np.float32)
                for stored, raw_step in enumerate(sampled_steps[:-1]):
                    begin = source_start + int(raw_step)
                    blocks[stored] = np.asarray(source["actions"][begin : begin + 5]).reshape(25)
                valid = np.ones(STORED_EP_LEN, dtype=bool)
                valid[-1] = False
                pixels = np.empty((STORED_EP_LEN, 224, 224, 3), dtype=np.uint8)
                for stored in range(STORED_EP_LEN):
                    pixels[stored], state = restore_and_render(env, qpos[stored], qvel[stored])
                    if (
                        np.linalg.norm(state["block_pos"] - block[stored]) > 2e-5
                        or np.linalg.norm(state["ee_pos"] - ee[stored]) > 2e-3
                    ):
                        raise RuntimeError(f"geometry mismatch during conversion {split}/{episode}/{stored}")
                sl = slice(output_start, output_start + STORED_EP_LEN)
                h5["pixels"][sl] = pixels
                h5["action_block"][sl] = blocks
                h5["action_block_valid"][sl] = valid
                h5["observation"][sl] = observations
                h5["qpos"][sl] = qpos
                h5["qvel"][sl] = qvel
                h5["block_pos"][sl] = block
                h5["ee_pos"][sl] = ee
                h5["ep_idx"][sl] = global_episode_offset + episode
                h5["step_idx"][sl] = sampled_steps
                h5["source_row"][sl] = raw_rows
                h5["ep_offset"][episode] = output_start
                h5["ep_len"][episode] = STORED_EP_LEN
                if (episode + 1) % max(1, args.progress_every) == 0:
                    h5.flush()
                    elapsed = time.perf_counter() - start_time
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "episodes_done": episode + 1,
                                "episodes_total": episodes,
                                "frames_per_second": (episode + 1) * STORED_EP_LEN / elapsed,
                                "temporary_size_gib": temporary.stat().st_size / (1024**3),
                            }
                        ),
                        flush=True,
                    )
            h5.flush()
        os.replace(temporary, output)
    finally:
        env.close()
        if temporary.exists():
            temporary.unlink()
    elapsed = time.perf_counter() - start_time
    result = validate_shard(split, output)
    result.update({"elapsed_seconds": elapsed, "frames_per_second": total / elapsed})
    write_json(DATA_ROOT / f"{split}_conversion.json", result)
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))


def render_only(env: Any, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Fast formal path: one state restore and one render, no action or observation recompute."""
    raw = env.unwrapped
    raw.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
    return np.asarray(env.render(), dtype=np.uint8).copy()


def part_path(split: str, episode_start: int, episode_stop: int, *, benchmark: bool) -> Path:
    root = DATA_ROOT / ("qc/parallel_benchmark" if benchmark else f"work/{split}")
    return root / f"part_{split}_{episode_start:04d}_{episode_stop:04d}.h5"


def validate_part(
    path: Path, split: str, episode_start: int, episode_stop: int
) -> dict[str, Any]:
    count = episode_stop - episode_start
    rows = count * STORED_EP_LEN
    global_offset = 0 if split == "train" else EXPECTED_SPLITS["train"]
    with h5py.File(path, "r", swmr=True) as h5:
        expected_attrs = {
            "format_version": FORMAT_VERSION,
            "split": split,
            "episode_start": episode_start,
            "episode_stop": episode_stop,
            "source_sha256": EXPECTED_SOURCE_SHA256[split],
            "no_action_rollout": True,
        }
        for key, expected in expected_attrs.items():
            actual = h5.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != expected:
                raise RuntimeError(f"part {path} attr {key}: expected={expected!r}, actual={actual!r}")
        if h5["pixels"].shape != (rows, 224, 224, 3):
            raise RuntimeError(f"part pixel shape mismatch: {h5['pixels'].shape}")
        expected_ids = np.repeat(
            np.arange(global_offset + episode_start, global_offset + episode_stop, dtype=np.int32),
            STORED_EP_LEN,
        )
        if not np.array_equal(h5["ep_idx"][:], expected_ids):
            raise RuntimeError(f"part global ep_idx mismatch: {path}")
        terminal = np.arange(STORED_EP_LEN - 1, rows, STORED_EP_LEN)
        if np.any(h5["action_block_valid"][terminal]) or np.any(h5["action_block"][terminal]):
            raise RuntimeError(f"part terminal action placeholder mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "split": split,
        "episode_start": episode_start,
        "episode_stop": episode_stop,
        "episodes": count,
        "frames": rows,
        "size_bytes": path.stat().st_size,
    }


def render_part_worker(task: tuple[str, int, int, bool, bool, int]) -> dict[str, Any]:
    """Spawn-safe worker; owns its source cache, environment, and atomic part."""
    split, episode_start, episode_stop, benchmark, resume, progress_every = task
    configure_storage()
    output = part_path(split, episode_start, episode_stop, benchmark=benchmark)
    output.parent.mkdir(parents=True, exist_ok=True)
    if resume and output.is_file():
        result = validate_part(output, split, episode_start, episode_stop)
        result.update({"resumed": True, "elapsed_seconds": 0.0})
        return result
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    source = validate_source(split)
    env = make_replay_env()
    episodes = episode_stop - episode_start
    total = episodes * STORED_EP_LEN
    global_offset = 0 if split == "train" else EXPECTED_SPLITS["train"]
    sampled_steps = np.arange(0, RAW_EP_LEN, SAMPLE_STRIDE, dtype=np.int64)
    started = time.perf_counter()
    try:
        env.reset(seed=42)
        with h5py.File(temporary, "w", libver="latest") as h5:
            create_shard_schema(h5, total, episodes)
            h5.attrs["split"] = split
            h5.attrs["episode_start"] = episode_start
            h5.attrs["episode_stop"] = episode_stop
            h5.attrs["source_sha256"] = EXPECTED_SOURCE_SHA256[split]
            for local_episode, episode in enumerate(range(episode_start, episode_stop)):
                raw_start = episode * RAW_EP_LEN
                out_start = local_episode * STORED_EP_LEN
                raw_rows = raw_start + sampled_steps
                observations = np.asarray(source["observations"][raw_rows], dtype=np.float32)
                qpos = np.asarray(source["qpos"][raw_rows], dtype=np.float32)
                qvel = np.asarray(source["qvel"][raw_rows], dtype=np.float32)
                block, ee = geometry_from_observation(observations)
                actions = np.zeros((STORED_EP_LEN, 25), dtype=np.float32)
                dense_actions = np.asarray(
                    source["actions"][raw_start : raw_start + RAW_EP_LEN - 1], dtype=np.float32
                )
                actions[:-1] = dense_actions.reshape(STORED_EP_LEN - 1, 25)
                valid = np.ones(STORED_EP_LEN, dtype=bool)
                valid[-1] = False
                pixels = np.empty((STORED_EP_LEN, 224, 224, 3), dtype=np.uint8)
                for stored in range(STORED_EP_LEN):
                    pixels[stored] = render_only(env, qpos[stored], qvel[stored])
                sl = slice(out_start, out_start + STORED_EP_LEN)
                h5["pixels"][sl] = pixels
                h5["action_block"][sl] = actions
                h5["action_block_valid"][sl] = valid
                h5["observation"][sl] = observations
                h5["qpos"][sl] = qpos
                h5["qvel"][sl] = qvel
                h5["block_pos"][sl] = block
                h5["ee_pos"][sl] = ee
                h5["ep_idx"][sl] = global_offset + episode
                h5["step_idx"][sl] = sampled_steps
                h5["source_row"][sl] = raw_rows
                h5["ep_offset"][local_episode] = out_start
                h5["ep_len"][local_episode] = STORED_EP_LEN
                if (local_episode + 1) % max(1, progress_every) == 0:
                    h5.flush()
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "event": "render_progress",
                                "pid": os.getpid(),
                                "split": split,
                                "part": [episode_start, episode_stop],
                                "episodes_done": local_episode + 1,
                                "frames_per_second": (local_episode + 1) * STORED_EP_LEN / elapsed,
                            }
                        ),
                        flush=True,
                    )
            h5.flush()
        os.replace(temporary, output)
    finally:
        env.close()
        if temporary.exists():
            temporary.unlink()
    elapsed = time.perf_counter() - started
    result = validate_part(output, split, episode_start, episode_stop)
    result.update(
        {
            "resumed": False,
            "elapsed_seconds": elapsed,
            "frames_per_second": total / elapsed,
            "sha256": sha256_file(output),
        }
    )
    write_json(output.with_suffix(".json"), result)
    return result


def split_ranges(episodes: int, workers: int) -> list[tuple[int, int]]:
    chunks = np.array_split(np.arange(episodes, dtype=np.int64), workers)
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks if len(chunk)]


def run_parallel_tasks(tasks: list[tuple[str, int, int, bool, bool, int]], workers: int) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(render_part_worker, tasks):
            results.append(result)
            print(json.dumps({"event": "part_complete", **jsonable(result)}), flush=True)
    return sorted(results, key=lambda item: (item["split"], item["episode_start"]))


def command_benchmark(args: argparse.Namespace) -> None:
    before_disk = shutil.disk_usage(AILAB.parent)
    tasks = [
        ("train", episode, episode + args.episodes_per_worker, True, False, 1)
        for episode in range(0, args.workers * args.episodes_per_worker, args.episodes_per_worker)
    ]
    started = time.perf_counter()
    results = run_parallel_tasks(tasks, args.workers)
    elapsed = time.perf_counter() - started
    frames = sum(item["frames"] for item in results)
    fps = frames / elapsed
    projection = sum(EXPECTED_SPLITS.values()) * STORED_EP_LEN / fps
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    after_disk = shutil.disk_usage(AILAB.parent)
    report = {
        "format_version": FORMAT_VERSION,
        "status": "PASS" if projection <= args.max_projected_hours * 3600 else "TOO_SLOW",
        "workers": args.workers,
        "episodes_per_worker": args.episodes_per_worker,
        "episodes": sum(item["episodes"] for item in results),
        "frames": frames,
        "elapsed_seconds": elapsed,
        "aggregate_frames_per_second": fps,
        "projected_formal_seconds": projection,
        "projected_formal_hours": projection / 3600,
        "max_projected_hours": args.max_projected_hours,
        "child_max_rss_kib": int(usage.ru_maxrss),
        "disk_free_before_bytes": before_disk.free,
        "disk_free_after_bytes": after_disk.free,
        "disk_used_by_benchmark_bytes": before_disk.free - after_disk.free,
        "parts": results,
    }
    write_json(DATA_ROOT / "qc/parallel_benchmark/report.json", report)
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    if projection > args.max_projected_hours * 3600:
        raise RuntimeError(
            f"parallel projection {projection / 3600:.2f}h exceeds {args.max_projected_hours:.2f}h"
        )


def merge_parts(split: str, parts: Sequence[Mapping[str, Any]], *, overwrite: bool) -> dict[str, Any]:
    output = shard_path(split)
    if output.exists() and not overwrite:
        return validate_shard(split, output, full_source_check=True)
    temporary = output.with_name(f".{output.name}.merge-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    episodes = EXPECTED_SPLITS[split]
    rows = episodes * STORED_EP_LEN
    ordered = sorted(parts, key=lambda item: int(item["episode_start"]))
    covered = [episode for item in ordered for episode in range(int(item["episode_start"]), int(item["episode_stop"]))]
    if covered != list(range(episodes)):
        raise RuntimeError(f"parts do not exactly cover {split} episodes")
    keys = (
        "pixels", "action_block", "action_block_valid", "observation", "qpos", "qvel",
        "block_pos", "ee_pos", "ep_idx", "step_idx", "source_row",
    )
    try:
        with h5py.File(temporary, "w", libver="latest") as destination:
            create_shard_schema(destination, rows, episodes)
            write_at = 0
            episode_at = 0
            for item in ordered:
                path = Path(str(item["path"]))
                validate_part(path, split, int(item["episode_start"]), int(item["episode_stop"]))
                with h5py.File(path, "r", swmr=True) as source:
                    part_rows = len(source["pixels"])
                    for start in range(0, part_rows, 256):
                        stop = min(start + 256, part_rows)
                        target = slice(write_at + start, write_at + stop)
                        for key in keys:
                            destination[key][target] = source[key][start:stop]
                    count = len(source["ep_len"])
                    destination["ep_len"][episode_at : episode_at + count] = source["ep_len"][:]
                    destination["ep_offset"][episode_at : episode_at + count] = (
                        np.asarray(source["ep_offset"][:], dtype=np.int64) + write_at
                    )
                    write_at += part_rows
                    episode_at += count
                destination.flush()
                print(json.dumps({"event": "merge_progress", "split": split, "episodes_done": episode_at}), flush=True)
            if write_at != rows or episode_at != episodes:
                raise RuntimeError(f"merge row/episode mismatch {write_at}/{episode_at}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return validate_shard(split, output, full_source_check=True)


def command_convert_parallel(args: argparse.Namespace) -> None:
    requested = ("train", "val") if args.split == "both" else (args.split,)
    all_results: dict[str, Any] = {}
    for split in requested:
        ranges = split_ranges(EXPECTED_SPLITS[split], args.workers)
        tasks = [
            (split, start, stop, False, args.resume, args.progress_every)
            for start, stop in ranges
        ]
        started = time.perf_counter()
        parts = run_parallel_tasks(tasks, min(args.workers, len(tasks)))
        render_seconds = time.perf_counter() - started
        merged = merge_parts(split, parts, overwrite=args.overwrite)
        all_results[split] = {
            "render_seconds": render_seconds,
            "aggregate_render_fps": EXPECTED_SPLITS[split] * STORED_EP_LEN / render_seconds,
            "parts": parts,
            "shard": merged,
        }
        write_json(DATA_ROOT / f"{split}_conversion.json", all_results[split])
    write_json(DATA_ROOT / "parallel_conversion.json", all_results)
    print(json.dumps(jsonable(all_results), indent=2, sort_keys=True))


def validate_shard(
    split: str, path: Path, *, full_source_check: bool = False
) -> dict[str, Any]:
    episodes = EXPECTED_SPLITS[split]
    rows = episodes * STORED_EP_LEN
    global_episode_offset = 0 if split == "train" else EXPECTED_SPLITS["train"]
    checks: dict[str, Any] = {}
    source = validate_source(split) if full_source_check else None
    with h5py.File(path, "r", swmr=True) as h5:
        required = {
            "pixels": ((rows, 224, 224, 3), np.dtype("uint8")),
            "action_block": ((rows, 25), np.dtype("float32")),
            "action_block_valid": ((rows,), np.dtype("bool")),
            "observation": ((rows, 28), np.dtype("float32")),
            "qpos": ((rows, 21), np.dtype("float32")),
            "qvel": ((rows, 20), np.dtype("float32")),
            "block_pos": ((rows, 3), np.dtype("float32")),
            "ee_pos": ((rows, 3), np.dtype("float32")),
            "ep_idx": ((rows,), np.dtype("int32")),
            "step_idx": ((rows,), np.dtype("int32")),
            "source_row": ((rows,), np.dtype("int64")),
            "ep_offset": ((episodes,), np.dtype("int64")),
            "ep_len": ((episodes,), np.dtype("int32")),
        }
        for name, (shape, dtype) in required.items():
            if name not in h5 or h5[name].shape != shape or h5[name].dtype != dtype:
                raise RuntimeError(
                    f"{split} shard {name} mismatch: expected={shape}/{dtype}, "
                    f"actual={getattr(h5.get(name), 'shape', None)}/{getattr(h5.get(name), 'dtype', None)}"
                )
        if not np.all(h5["ep_len"][:] == STORED_EP_LEN):
            raise RuntimeError("ep_len contract failed")
        if not np.array_equal(h5["ep_offset"][:], np.arange(episodes) * STORED_EP_LEN):
            raise RuntimeError("ep_offset contract failed")
        expected_ep_idx = np.repeat(
            np.arange(global_episode_offset, global_episode_offset + episodes, dtype=np.int32),
            STORED_EP_LEN,
        )
        if not np.array_equal(h5["ep_idx"][:], expected_ep_idx):
            raise RuntimeError(f"{split} global ep_idx namespace contract failed")
        terminal_rows = np.arange(STORED_EP_LEN - 1, rows, STORED_EP_LEN)
        if np.any(h5["action_block_valid"][terminal_rows]) or np.any(h5["action_block"][terminal_rows]):
            raise RuntimeError("terminal action block placeholder must be invalid and exact zero")
        # All 198 clip starts per episode have three valid transition blocks.
        valid = np.asarray(h5["action_block_valid"][:], dtype=bool).reshape(episodes, STORED_EP_LEN)
        windows = sum(int(np.all(row[start : start + NUM_FRAMES - 1])) for row in valid for start in range(198))
        expected_windows = episodes * 198
        if windows != expected_windows:
            raise RuntimeError(f"valid window count mismatch: {windows} != {expected_windows}")
        checks.update(
            {
                "full_counts": {"status": "PASS", "episodes": episodes, "frames": rows},
                "global_episode_namespace_unique": {
                    "status": "PASS",
                    "unique": episodes,
                    "range": [global_episode_offset, global_episode_offset + episodes - 1],
                },
                "terminal_action_block_zero": {
                    "status": "PASS",
                    "count": episodes,
                },
                "valid_training_windows": {"status": "PASS", "count": expected_windows},
            }
        )
        if full_source_check:
            assert source is not None
            sampled_steps = np.arange(0, RAW_EP_LEN, SAMPLE_STRIDE, dtype=np.int32)
            expected_step = np.tile(sampled_steps, episodes)
            if not np.array_equal(h5["step_idx"][:], expected_step):
                raise RuntimeError(f"{split} step_idx is not exact phase-0 0,5,...,1000")
            local_episode = np.repeat(np.arange(episodes, dtype=np.int64), STORED_EP_LEN)
            expected_source_rows = local_episode * RAW_EP_LEN + np.tile(sampled_steps, episodes)
            if not np.array_equal(h5["source_row"][:], expected_source_rows):
                raise RuntimeError(f"{split} source_row alignment failed")
            comparisons = {
                "qpos_exact": (h5["qpos"][:], source["qpos"][expected_source_rows]),
                "qvel_exact": (h5["qvel"][:], source["qvel"][expected_source_rows]),
                "observation_exact": (
                    h5["observation"][:], source["observations"][expected_source_rows]
                ),
            }
            for name, (actual, expected) in comparisons.items():
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f"{split} {name} source binding failed")
                checks[name] = {"status": "PASS", "rows": rows}
            expected_block, expected_ee = geometry_from_observation(
                source["observations"][expected_source_rows]
            )
            if not np.array_equal(h5["block_pos"][:], expected_block):
                raise RuntimeError(f"{split} block_pos exact observation derivation failed")
            if not np.array_equal(h5["ee_pos"][:], expected_ee):
                raise RuntimeError(f"{split} ee_pos exact observation derivation failed")
            expected_actions = np.zeros((episodes, STORED_EP_LEN, 25), dtype=np.float32)
            expected_actions[:, :-1] = np.asarray(source["actions"], dtype=np.float32).reshape(
                episodes, RAW_EP_LEN, RAW_ACTION_DIM
            )[:, : RAW_EP_LEN - 1].reshape(episodes, STORED_EP_LEN - 1, 25)
            if not np.array_equal(
                h5["action_block"][:].reshape(episodes, STORED_EP_LEN, 25),
                expected_actions,
                equal_nan=True,
            ):
                raise RuntimeError(f"{split} all action blocks differ from official raw actions")
            checks.update(
                {
                    "step_idx_exact_phase0": {"status": "PASS", "rows": rows},
                    "source_row_exact": {"status": "PASS", "rows": rows},
                    "action_block_all_exact": {
                        "status": "PASS",
                        "nonterminal_blocks": episodes * (STORED_EP_LEN - 1),
                    },
                    "block_pos_exact_from_observation": {"status": "PASS", "rows": rows},
                    "ee_pos_exact_from_observation": {"status": "PASS", "rows": rows},
                }
            )

            smoke = json.loads((DATA_ROOT / "qc/smoke_report.json").read_text(encoding="utf-8"))
            qc_rows = [row for row in smoke["rows"] if row["split"] == split]
            pixel_diffs: list[dict[str, Any]] = []
            for row in qc_rows:
                local_row = int(row["episode"]) * STORED_EP_LEN + int(row["raw_step"]) // SAMPLE_STRIDE
                actual_pixels = np.asarray(h5["pixels"][local_row], dtype=np.uint8)
                expected_pixels = np.asarray(
                    Image.open(DATA_ROOT / "qc" / str(row["png"])), dtype=np.uint8
                )
                actual_sha = hashlib.sha256(actual_pixels.tobytes()).hexdigest()
                difference = np.abs(
                    actual_pixels.astype(np.int16) - expected_pixels.astype(np.int16)
                )
                changed_pixels = int(np.any(difference, axis=-1).sum())
                maximum = int(difference.max())
                entry = {
                    "source_row": int(row["source_row"]),
                    "expected_sha256": row["pixel_sha256"],
                    "actual_sha256": actual_sha,
                    "byte_exact": actual_sha == row["pixel_sha256"],
                    "max_abs_channel_delta": maximum,
                    "changed_pixels": changed_pixels,
                    "changed_channels": int(np.count_nonzero(difference)),
                    "mean_abs_channel_delta": float(difference.mean()),
                }
                pixel_diffs.append(entry)
            byte_exact = sum(bool(item["byte_exact"]) for item in pixel_diffs)
            checks["pixels_qc_exact"] = {
                "status": "REPORT_ONLY_EGL_NONDETERMINISM",
                "samples": len(qc_rows),
                "byte_exact_samples": byte_exact,
                "max_abs_channel_delta": max(item["max_abs_channel_delta"] for item in pixel_diffs),
                "max_changed_pixels_per_frame": max(item["changed_pixels"] for item in pixel_diffs),
                "tolerance": None,
                "report_only_reason": (
                    "two independent EGL contexts changed one-LSB antialias rounding at a non-stable pixel count; "
                    "the pixel byte/diff gate was deleted under AGENTS gate discipline"
                ),
                "runs": ["train_report_only_run_1", "full_report_only_run_2"],
                "diffs": pixel_diffs,
            }
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "episodes": episodes,
        "frames": rows,
        "windows": expected_windows,
        "official_split": split,
        "global_episode_id_range": [global_episode_offset, global_episode_offset + episodes - 1],
        "source_local_episode_id_offset": global_episode_offset,
        "checks": checks,
    }


def schema_manifest() -> dict[str, Any]:
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


def command_finalize(args: argparse.Namespace) -> None:
    health_path = DATA_ROOT / "health_report.json"
    smoke_path = DATA_ROOT / "qc/smoke_report.json"
    if not health_path.is_file() or not smoke_path.is_file():
        raise FileNotFoundError("audit and smoke reports must exist before finalize")
    health = json.loads(health_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    validated_shards = {
        split: validate_shard(split, shard_path(split), full_source_check=True)
        for split in ("train", "val")
    }
    shard_checks = {
        split: validated_shards[split].pop("checks") for split in ("train", "val")
    }
    shards = validated_shards
    train_ids = set(range(*[shards["train"]["global_episode_id_range"][0], shards["train"]["global_episode_id_range"][1] + 1]))
    val_ids = set(range(*[shards["val"]["global_episode_id_range"][0], shards["val"]["global_episode_id_range"][1] + 1]))
    if train_ids & val_ids or len(train_ids | val_ids) != sum(EXPECTED_SPLITS.values()):
        raise RuntimeError("train/val global episode namespace overlap")
    exclusion = health["exclusion"]
    five_required = (
        "exact_episode_hash_overlap_with_expert",
        "exact_episode_hash_overlap_with_formal50",
        "exact_episode_hash_overlap_with_measurement1",
        "quantized_signature_overlap_with_expert",
        "quantized_signature_overlap_with_formal50",
    )
    if any(exclusion.get(key) != 0 for key in five_required):
        raise RuntimeError(f"required exclusion values are not all zero: {exclusion}")
    split_diffs = {
        split: shard_checks[split]["pixels_qc_exact"].pop("diffs")
        for split in ("train", "val")
    }
    pixel_checks = [shard_checks[split]["pixels_qc_exact"] for split in ("train", "val")]
    pixel_status = "REPORT_ONLY_EGL_NONDETERMINISM"
    details_by_key = {
        (split, int(item["source_row"])): item
        for split, values in split_diffs.items()
        for item in values
    }
    pixel_rows = []
    for row in smoke["rows"]:
        key = (str(row["split"]), int(row["source_row"]))
        detail = details_by_key[key]
        pixel_rows.append(
            {
                "split": row["split"],
                "episode": row["episode"],
                "raw_step": row["raw_step"],
                "source_row": row["source_row"],
                "png": row["png"],
                "pixel_sha256": row["pixel_sha256"],
                "observation_max_abs_error": row["observation_max_abs_error"],
                "block_pos_error_m": row["block_pos_error_m"],
                "ee_pos_error_m": row["ee_pos_error_m"],
                **detail,
            }
        )
    report_only_reason = (
        "two independent EGL contexts changed one-LSB antialias rounding at a non-stable pixel count; "
        "the pixel byte/diff gate was deleted under AGENTS gate discipline and is not part of dataset validity"
    )
    first_run = {
        "run_id": "train_report_only_run_1",
        "source": str((AILAB / "logs/data/play_v1_parallel_formal.log").resolve()),
        "samples": 10,
        "byte_exact_samples": 7,
        "mismatches": [
            {
                "source_row": 111221,
                "expected_sha256": "023e2782bd96c2a0a3179af32873138f870d11686940b9f256bd863345831c33",
                "actual_sha256": "9fdb40cc90adb39555e7a3473d70d8c891178fc08afd76856c25f6b5251e6cfa",
                "max_abs_channel_delta": 1,
                "changed_pixels": 2,
                "changed_channels": 4,
                "mean_abs_channel_delta": 2.657312925170068e-05,
            },
            {
                "source_row": 667331,
                "expected_sha256": "09b3ce12cfb54ed29287e52fa07f180e4549484dcbe8cce95d37824a2fb920b3",
                "actual_sha256": "77702daa9de1fb43f38974c6e74e40236a3772447b3d0b86deb0777042a915f1",
                "max_abs_channel_delta": 1,
                "changed_pixels": 2,
                "changed_channels": 2,
                "mean_abs_channel_delta": 1.328656462585034e-05,
            },
            {
                "source_row": 778557,
                "expected_sha256": "e45a3766c3bbf3dcd87a4f27cd2486ae93ebaca7d55bda8d1ab5f03e4b277c9b",
                "actual_sha256": "feb990928234ea19ff015e1eb4c27748b948c6583f8bf18081f468b40aa4802a",
                "max_abs_channel_delta": 1,
                "changed_pixels": 1,
                "changed_channels": 1,
                "mean_abs_channel_delta": 6.64328231292517e-06,
            },
        ],
    }
    max_block_error = max(float(row["block_pos_error_m"]) for row in smoke["rows"])
    max_ee_error = max(float(row["ee_pos_error_m"]) for row in smoke["rows"])
    if max_block_error > 2e-5 or max_ee_error > 2e-3:
        raise RuntimeError("set_state geometric QC no longer passes its frozen physical tolerance")
    pixel_qc_details = {
        "format_version": "cube_play_v1_pixel_alignment_v1",
        "status": pixel_status,
        "no_action_rollout": True,
        "num_frames": len(pixel_rows),
        "tolerance": None,
        "report_only_reason": report_only_reason,
        "runs": [
            first_run,
            {
                "run_id": "full_report_only_run_2",
                "samples": len(pixel_rows),
                "byte_exact_samples": sum(bool(row["byte_exact"]) for row in pixel_rows),
                "rows_reference": "top-level rows",
            },
        ],
        "geometry_qc": {
            "status": "PASS",
            "samples": len(smoke["rows"]),
            "max_block_pos_error_m": max_block_error,
            "max_ee_pos_error_m": max_ee_error,
            "tolerance": {"block_pos_error_m": 2e-5, "ee_pos_error_m": 2e-3},
        },
        "smoke_report": file_identity(smoke_path),
        "rows": pixel_rows,
    }
    pixel_qc_path = DATA_ROOT / "qc/pixel_alignment_report.json"
    write_json(pixel_qc_path, pixel_qc_details)
    total_frames = sum(item["frames"] for item in shards.values())
    total_episodes = sum(item["episodes"] for item in shards.values())
    qc_binding = {
        **file_identity(pixel_qc_path),
        "num_rendered_frames": len(pixel_rows),
    }
    validation_checks = {
        "full_counts": {
            "status": "PASS",
            "episodes": total_episodes,
            "frames": total_frames,
            "windows": sum(item["windows"] for item in shards.values()),
        },
        "global_episode_namespace_unique": {
            "status": "PASS",
            "train": shards["train"]["global_episode_id_range"],
            "val": shards["val"]["global_episode_id_range"],
            "intersection": 0,
        },
        "step_idx_exact_phase0": {"status": "PASS", "rows": total_frames},
        "source_row_exact": {"status": "PASS", "rows": total_frames},
        "action_block_all_exact": {
            "status": "PASS",
            "nonterminal_blocks": total_episodes * (STORED_EP_LEN - 1),
        },
        "terminal_action_block_zero": {"status": "PASS", "count": total_episodes},
        "pixels_qc_exact": {
            "status": pixel_status,
            "samples": sum(item["samples"] for item in pixel_checks),
            "byte_exact_samples": sum(item["byte_exact_samples"] for item in pixel_checks),
            "max_abs_channel_delta": max(item["max_abs_channel_delta"] for item in pixel_checks),
            "max_changed_pixels_per_frame": max(item["max_changed_pixels_per_frame"] for item in pixel_checks),
            "tolerance": None,
            "report_only_reason": report_only_reason,
            "runs": ["train_report_only_run_1", "full_report_only_run_2"],
        },
        "qpos_exact": {"status": "PASS", "rows": total_frames},
        "qvel_exact": {"status": "PASS", "rows": total_frames},
        "observation_exact": {"status": "PASS", "rows": total_frames},
        "block_pos_exact_from_observation": {"status": "PASS", "rows": total_frames},
        "ee_pos_exact_from_observation": {"status": "PASS", "rows": total_frames},
    }
    expected_check_keys = {
        "full_counts", "global_episode_namespace_unique", "step_idx_exact_phase0",
        "source_row_exact", "action_block_all_exact", "terminal_action_block_zero",
        "pixels_qc_exact", "qpos_exact", "qvel_exact", "observation_exact",
        "block_pos_exact_from_observation", "ee_pos_exact_from_observation",
    }
    if set(validation_checks) != expected_check_keys:
        raise AssertionError("validation check schema drift")
    validation = {
        "format_version": "cube_play_v1_validation_v1",
        "valid": True,
        "created_unix": time.time(),
        "sources": health["sources"],
        "shards": shards,
        "qc": qc_binding,
        "checks": validation_checks,
        "exclusion": exclusion,
    }
    validation_path = DATA_ROOT / "validation.json"
    write_json(validation_path, validation)
    manifest = {
        "format_version": FORMAT_VERSION,
        "capture_contract": {
            "source": "official OGBench state NPZ; deterministic set_state(qpos,qvel) then 224x224 render",
            "simulator_actions_executed": 0,
            "official_train_val_split_preserved": True,
            "play_episode_namespace": "official_play_train_val",
            "global_episode_ids": "train=0..999; val=1000..1099; source_local_episode=global-offset",
            "raw_episode_length": RAW_EP_LEN,
            "raw_temporal_phase": 0,
            "raw_temporal_stride": SAMPLE_STRIDE,
            "stored_episode_length": STORED_EP_LEN,
            "stored_steps": "0,5,...,1000",
            "action_block": "exact raw actions[t:t+5].reshape(25); terminal stored frame is invalid exact-zero placeholder",
            "training_window": "four consecutive stored frames plus first three valid action blocks",
            "warning": "stored rows are 1/5 temporal phase samples and must be loaded with frameskip=1",
        },
        "sources": health["sources"],
        "shards": shards,
        "schema": schema_manifest(),
        "health_report": file_identity(health_path),
        "exclusion": exclusion,
        "qc": qc_binding,
        "validation": file_identity(validation_path),
    }
    write_json(DATA_ROOT / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "manifest": file_identity(DATA_ROOT / 'manifest.json')}, indent=2))


def _fmt(vector: Sequence[Any], digits: int = 4) -> str:
    return "[" + ", ".join(f"{float(x):.{digits}f}" for x in vector) + "]"


def write_health_markdown(report: Mapping[str, Any], path: Path) -> None:
    play, expert = report["play"], report["expert"]
    exclusion = report["exclusion"]
    action_overlap = report["distribution_overlap"]["action"]
    lines = [
        "# OGBench Cube Play 数据体检",
        "",
        f"格式：`{FORMAT_VERSION}`。源为官方 `cube-single-play-v0` train/val state NPZ；本报告统计完整原始 1.101M 帧，而非降采样后的渲染 H5。",
        f"官方一手源：`{SOURCE_OFFICIAL_REPOSITORY}`。因本机直连该站点时 SSL 失败，字节文件仅经传输镜像 `{SOURCE_TRANSPORT_MIRROR}` 的固定 revision `{SOURCE_TRANSPORT_REVISION}` 获取；文件名与 SHA256 仍作为内容身份。",
        "",
        "## 规模与字段",
        "",
        f"- Play：{play['episodes']} episodes，{play['frames']:,} 帧，每局 {play['episode_length']} 帧。",
        f"- Expert 对照：{expert['episodes']} episodes，{expert['frames']:,} 帧。",
        f"- Play 成功帧占比：**N/A**；原因：{play['success']['reason']}。",
        "",
        "## 动作与状态覆盖",
        "",
        f"- Play action min/max：`{_fmt(play['actions']['min'])}` / `{_fmt(play['actions']['max'])}`。",
        f"- Expert action min/max（逐维过滤终止占位 NaN）：`{_fmt(expert['actions']['min'])}` / `{_fmt(expert['actions']['max'])}`。",
        f"- 动作直方图重叠系数（5 维均值）：**{action_overlap['overlap_coefficient_mean']:.4f}**；JS divergence：**{action_overlap['jensen_shannon_divergence_bits_mean']:.4f} bits**。",
        f"- Play block 范围：`{_fmt(play['block_pos_m']['min'])}` → `{_fmt(play['block_pos_m']['max'])}` m。",
        f"- Expert block 范围：`{_fmt(expert['block_pos_m']['min'])}` → `{_fmt(expert['block_pos_m']['max'])}` m。",
        f"- Play EE 范围：`{_fmt(play['ee_pos_m']['min'])}` → `{_fmt(play['ee_pos_m']['max'])}` m。",
        f"- Expert EE 范围：`{_fmt(expert['ee_pos_m']['min'])}` → `{_fmt(expert['ee_pos_m']['max'])}` m。",
        "",
        "## 轨迹排除核验",
        "",
        f"- 全 expert 完整 episode hash 交集：**{exclusion['exact_episode_hash_overlap_with_expert']}**。",
        f"- 固定 50 episode 完整 hash 交集：**{exclusion['exact_episode_hash_overlap_with_formal50']}**。",
        f"- Measurement-1 留出 episode 完整 hash 交集：**{exclusion['exact_episode_hash_overlap_with_measurement1']}**。",
        f"- 全 expert / 固定 50 的 1e-3 量化头中尾签名交集：**{exclusion['quantized_signature_overlap_with_expert']} / {exclusion['quantized_signature_overlap_with_formal50']}**。",
        "- 独立采集是官方数据变体的来源声明，不作为数值门；门由上述完整与近重复交集为零给出。",
        "",
        "## 转换纪律",
        "",
        "正式 H5 保留官方 1000/100 train/val split；每局只存 phase-0 的 raw step `0,5,…,1000`。这是一套 1/5 时间相位样本，训练必须用 `frameskip=1`。图像只由 `set_state(qpos,qvel)` 后渲染，禁止动作 rollout。",
        "",
        "机器可读全量直方图、逐维矩与交集定义见 `health_report.json`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="full raw health and trajectory-overlap audit")
    sub.add_parser("smoke", help="deterministic 20-frame set_state/render QC and size estimate")
    convert = sub.add_parser("convert", help="render one official split into its formal H5")
    convert.add_argument("--split", choices=("train", "val"), required=True)
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--progress-every", type=int, default=10)
    benchmark = sub.add_parser("benchmark", help="8-worker one-episode-per-worker render benchmark")
    benchmark.add_argument("--workers", type=int, default=8)
    benchmark.add_argument("--episodes-per-worker", type=int, default=1)
    benchmark.add_argument("--max-projected-hours", type=float, default=4.0)
    parallel = sub.add_parser("convert-parallel", help="resumable worker-local rendering and atomic merge")
    parallel.add_argument("--split", choices=("train", "val", "both"), default="both")
    parallel.add_argument("--workers", type=int, default=8)
    parallel.add_argument("--resume", action="store_true")
    parallel.add_argument("--overwrite", action="store_true")
    parallel.add_argument("--progress-every", type=int, default=10)
    sub.add_parser("finalize", help="validate both shards and write manifest")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    configure_storage()
    args = parser().parse_args(argv)
    if args.command == "audit":
        command_audit(args)
    elif args.command == "smoke":
        command_smoke(args)
    elif args.command == "convert":
        command_convert(args)
    elif args.command == "benchmark":
        command_benchmark(args)
    elif args.command == "convert-parallel":
        command_convert_parallel(args)
    elif args.command == "finalize":
        command_finalize(args)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
