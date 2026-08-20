#!/usr/bin/env python3
"""Encode Cube frames and build an episode-disjoint 4D probe dataset.

The 50 frozen evaluation episodes are excluded before sampling or splitting.
Rows are then assigned to train/validation/test strictly through their episode
identifier.  The output consists of ordinary ``.npy`` arrays so later stages
can memory-map the data without duplicating it in RAM.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

import cube_probe_common as common


OUTPUT_PARENT = common.AILAB_ROOT / "outputs/probe"
SPLIT_NAMES = ("train", "val", "test")


def _episode_split(
    episodes: np.ndarray, excluded: np.ndarray, seed: int
) -> tuple[dict[int, int], dict[str, list[int]]]:
    eligible = np.setdiff1d(np.unique(episodes), excluded, assume_unique=False)
    if len(eligible) < 3:
        raise ValueError("fewer than three eligible episodes remain")
    shuffled = eligible.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    n_train = int(np.floor(0.8 * len(shuffled)))
    n_val = int(np.floor(0.1 * len(shuffled)))
    groups = {
        "train": np.sort(shuffled[:n_train]).tolist(),
        "val": np.sort(shuffled[n_train : n_train + n_val]).tolist(),
        "test": np.sort(shuffled[n_train + n_val :]).tolist(),
    }
    mapping = {
        int(episode): split
        for split, name in enumerate(SPLIT_NAMES)
        for episode in groups[name]
    }
    return mapping, groups


def _sample_random_rows(
    episodes: np.ndarray,
    split_by_episode: dict[int, int],
    max_frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    eligible_mask = np.fromiter(
        (int(ep) in split_by_episode for ep in episodes),
        dtype=bool,
        count=len(episodes),
    )
    eligible_rows = np.flatnonzero(eligible_mask)
    if max_frames < 0:
        raise ValueError("--max-frames must be zero (all) or positive")
    if max_frames and max_frames < len(eligible_rows):
        # Sampling is deterministic; sorting permits efficient monotonic HDF5 reads.
        eligible_rows = np.sort(
            np.random.default_rng(seed + 1).choice(
                eligible_rows, size=max_frames, replace=False
            )
        )
    split = np.fromiter(
        (split_by_episode[int(episodes[row])] for row in eligible_rows),
        dtype=np.uint8,
        count=len(eligible_rows),
    )
    if any(np.count_nonzero(split == i) == 0 for i in range(3)):
        raise RuntimeError("sampling produced an empty train/val/test split")
    return eligible_rows.astype(np.int64), split, {
        "sampling_mode": "random_rows",
        "selection_contract": "uniform_without_replacement_across_all_eligible_rows",
        "requested_max_frames": max_frames,
        "effective_frames": len(eligible_rows),
    }


def _split_target_counts(num_frames: int) -> np.ndarray:
    if num_frames < 3:
        raise ValueError("at least three frames are required for nonempty train/val/test")
    counts = np.asarray(
        [int(np.floor(0.8 * num_frames)), int(np.floor(0.1 * num_frames))],
        dtype=np.int64,
    )
    counts = np.append(counts, num_frames - int(counts.sum()))
    if np.any(counts <= 0) or int(counts.sum()) != num_frames:
        raise RuntimeError(f"invalid 80/10/10 frame targets: {counts.tolist()}")
    return counts


def _episode_ranges(episodes: np.ndarray) -> dict[int, tuple[int, int]]:
    episodes = np.asarray(episodes, dtype=np.int64)
    if episodes.ndim != 1 or len(episodes) == 0:
        raise ValueError("episodes must be a nonempty one-dimensional array")
    if np.any(np.diff(episodes) < 0):
        raise ValueError("episode-block sampling requires episode-contiguous HDF5 rows")
    unique, starts, lengths = np.unique(
        episodes, return_index=True, return_counts=True
    )
    return {
        int(episode): (int(start), int(length))
        for episode, start, length in zip(unique, starts, lengths, strict=True)
    }


def _sample_episode_blocks(
    episodes: np.ndarray,
    excluded: np.ndarray,
    max_frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], dict[str, object]]:
    """Select a seeded contiguous episode window, then split by episode.

    For a finite frame cap, every split uses complete episodes plus at most one
    contiguous partial episode.  On the formal Cube dataset (201 frames per
    episode), ``max_frames=400000`` is exactly 320000/40000/40000 rows:
    train=1592 full episodes+8 rows, val=199 full+1 row, test=199 full+1 row.
    """

    if max_frames < 0:
        raise ValueError("--max-frames must be zero (all) or positive")
    ranges = _episode_ranges(episodes)
    eligible = np.setdiff1d(
        np.asarray(sorted(ranges), dtype=np.int64),
        np.asarray(excluded, dtype=np.int64),
        assume_unique=False,
    )
    eligible_frames = int(sum(ranges[int(ep)][1] for ep in eligible))
    if len(eligible) < 3:
        raise ValueError("fewer than three eligible episodes remain")

    if max_frames == 0 or max_frames >= eligible_frames:
        split_by_episode, eligible_groups = _episode_split(
            episodes, excluded, seed
        )
        rows, split, details = _sample_random_rows(
            episodes, split_by_episode, 0, seed
        )
        actual_groups = {
            name: np.unique(episodes[rows[split == split_id]]).astype(int).tolist()
            for split_id, name in enumerate(SPLIT_NAMES)
        }
        details.update(
            {
                "sampling_mode": "episode_blocks",
                "selection_contract": "all_eligible_episodes",
                "eligible_split_episode_counts": {
                    name: len(eligible_groups[name]) for name in SPLIT_NAMES
                },
            }
        )
        return rows, split, actual_groups, details

    target_counts = _split_target_counts(max_frames)
    episode_lengths = np.asarray([ranges[int(ep)][1] for ep in eligible], dtype=np.int64)
    unique_lengths = np.unique(episode_lengths)
    if unique_lengths.size != 1:
        raise ValueError(
            "capped episode-block sampling currently requires uniform episode lengths; "
            f"actual_unique_lengths={unique_lengths.tolist()}"
        )
    frames_per_episode = int(unique_lengths[0])
    needed_episode_counts = np.ceil(target_counts / frames_per_episode).astype(np.int64)
    total_needed = int(needed_episode_counts.sum())
    if total_needed > len(eligible):
        raise ValueError(
            "not enough eligible episodes for requested episode-disjoint cap: "
            f"needed={total_needed}, available={len(eligible)}"
        )

    rng = np.random.default_rng(seed + 1)
    window_start = int(rng.integers(0, len(eligible) - total_needed + 1))
    episode_window = eligible[window_start : window_start + total_needed].copy()
    # The union remains a compact HDF5 window; shuffling only assigns whole
    # episodes to disjoint logical splits and does not randomize row reads.
    assignment = episode_window.copy()
    np.random.default_rng(seed + 2).shuffle(assignment)

    selected_rows: list[np.ndarray] = []
    selected_split: list[np.ndarray] = []
    split_episodes: dict[str, list[int]] = {}
    full_episode_counts: dict[str, int] = {}
    partial_episodes: dict[str, dict[str, int] | None] = {}
    cursor = 0
    for split_id, name in enumerate(SPLIT_NAMES):
        count = int(needed_episode_counts[split_id])
        assigned = assignment[cursor : cursor + count]
        cursor += count
        remaining = int(target_counts[split_id])
        used_episodes: list[int] = []
        full_count = 0
        partial = None
        for episode in assigned:
            episode = int(episode)
            row_start, episode_length = ranges[episode]
            take = min(episode_length, remaining)
            if take <= 0:
                break
            # Only the last assigned episode can be partial.  A deterministic
            # contiguous offset avoids a time-zero-only bias while retaining
            # chunk locality.
            offset = 0
            if take < episode_length:
                offset = int(
                    np.random.default_rng(seed + 100 + split_id).integers(
                        0, episode_length - take + 1
                    )
                )
                partial = {
                    "episode": episode,
                    "episode_length": episode_length,
                    "selected_offset": offset,
                    "selected_frames": take,
                }
            else:
                full_count += 1
            block = np.arange(row_start + offset, row_start + offset + take, dtype=np.int64)
            selected_rows.append(block)
            selected_split.append(np.full(take, split_id, dtype=np.uint8))
            used_episodes.append(episode)
            remaining -= take
        if remaining != 0:
            raise RuntimeError(
                f"episode allocation underfilled {name}: remaining={remaining}"
            )
        split_episodes[name] = sorted(used_episodes)
        full_episode_counts[name] = full_count
        partial_episodes[name] = partial
    if cursor != total_needed:
        raise RuntimeError("episode assignment cursor mismatch")

    rows = np.concatenate(selected_rows)
    split = np.concatenate(selected_split)
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    split = split[order]
    if len(rows) != max_frames or not np.all(np.diff(rows) > 0):
        raise RuntimeError(
            f"episode-block sampler expected {max_frames} unique sorted rows, got {len(rows)}"
        )
    if any(np.count_nonzero(split == split_id) != target_counts[split_id] for split_id in range(3)):
        raise RuntimeError("episode-block sampler did not preserve 80/10/10 frame counts")
    selected_episode_sets = [set(split_episodes[name]) for name in SPLIT_NAMES]
    if any(
        selected_episode_sets[a] & selected_episode_sets[b]
        for a in range(3)
        for b in range(a + 1, 3)
    ):
        raise RuntimeError("episode-block split leaked an episode across splits")
    return rows, split, split_episodes, {
        "sampling_mode": "episode_blocks",
        "selection_contract": (
            "seeded_compact_eligible_episode_window; whole_episodes_plus_at_most_"
            "one_contiguous_partial_episode_per_split"
        ),
        "requested_max_frames": max_frames,
        "effective_frames": len(rows),
        "split_target_frames": dict(
            zip(SPLIT_NAMES, target_counts.astype(int).tolist(), strict=True)
        ),
        "frames_per_episode": frames_per_episode,
        "episode_window_eligible_index_start": window_start,
        "episode_window_count": total_needed,
        "episode_window_min_id": int(np.min(episode_window)),
        "episode_window_max_id": int(np.max(episode_window)),
        "selected_episode_counts": dict(
            zip(SPLIT_NAMES, needed_episode_counts.astype(int).tolist(), strict=True)
        ),
        "full_episode_counts": full_episode_counts,
        "partial_episodes": partial_episodes,
    }


def _prepare_destination(output: Path, overwrite: bool) -> tuple[Path, Path]:
    output = common.ensure_output_child(output, OUTPUT_PARENT, "probe dataset output")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output is not empty: {output}")
    staging = output.parent / f".{output.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    return output, staging


def _commit(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if any(output.iterdir()) and not overwrite:
            raise FileExistsError(f"output became nonempty during build: {output}")
        shutil.rmtree(output)
    os.replace(staging, output)


def run(args: argparse.Namespace) -> int:
    common.configure_storage()
    dataset = common.ensure_data_disk(args.dataset, "dataset")
    manifest = common.ensure_data_disk(args.manifest, "manifest")
    if not dataset.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"missing dataset/manifest: {dataset}, {manifest}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    import hdf5plugin  # noqa: F401 - register dataset compression filters
    import h5py
    import torch
    import stable_worldmodel as swm

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    formal_rows, _ = common.load_formal_rows(manifest)
    with h5py.File(dataset, "r", swmr=True) as h5:
        required = (
            "pixels",
            "ep_idx",
            "privileged_block_0_pos",
            "privileged_block_0_yaw",
        )
        missing = [key for key in required if key not in h5]
        if missing:
            raise KeyError(f"dataset missing fields: {missing}")
        episodes = np.asarray(h5["ep_idx"][:], dtype=np.int64)
        excluded = np.asarray(h5["ep_idx"][formal_rows], dtype=np.int64)
        if len(np.unique(excluded)) != 50:
            raise RuntimeError("frozen formal rows must identify 50 unique episodes")
        if args.sampling_mode == "episode_blocks":
            rows, row_split, split_episodes, sampling_details = (
                _sample_episode_blocks(
                    episodes, excluded, args.max_frames, args.seed
                )
            )
        else:
            split_by_episode, _ = _episode_split(episodes, excluded, args.seed)
            rows, row_split, sampling_details = _sample_random_rows(
                episodes, split_by_episode, args.max_frames, args.seed
            )
            split_episodes = {
                name: np.unique(
                    episodes[rows[row_split == split_id]]
                ).astype(int).tolist()
                for split_id, name in enumerate(SPLIT_NAMES)
            }
        selected_episodes = episodes[rows].astype(np.int32)
        if np.intersect1d(selected_episodes, excluded).size:
            raise RuntimeError("formal evaluation episode leaked into probe rows")
        if not np.all(np.diff(rows) > 0):
            raise RuntimeError("selected HDF5 rows must be unique and strictly sorted")
        episode_sets = [set(split_episodes[name]) for name in SPLIT_NAMES]
        if any(
            episode_sets[a] & episode_sets[b]
            for a in range(3)
            for b in range(a + 1, 3)
        ):
            raise RuntimeError("an episode appears in more than one split")
        pixel_chunk_rows = int(h5["pixels"].chunks[0])
        sampling_details.update(
            {
                "pixel_chunk_rows": pixel_chunk_rows,
                "estimated_pixel_chunks_touched": int(
                    len(np.unique(rows // pixel_chunk_rows))
                ),
                "total_pixel_chunks": int(
                    np.ceil(len(episodes) / pixel_chunk_rows)
                ),
            }
        )

    output, staging = _prepare_destination(args.output, args.overwrite)
    try:
        np.save(staging / "rows.npy", rows, allow_pickle=False)
        np.save(staging / "episodes.npy", selected_episodes, allow_pickle=False)
        np.save(staging / "split.npy", row_split, allow_pickle=False)
        targets = np.lib.format.open_memmap(
            staging / "targets_block4d.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), 4),
        )

        model = swm.wm.utils.load_pretrained(
            args.checkpoint, cache_dir=str(common.AILAB_ROOT)
        )
        model = model.to(args.device).eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        world_model_state_sha256 = common.torch_module_sha256(model)
        embeddings = None
        embedding_dim = None

        with h5py.File(dataset, "r", swmr=True) as h5, torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                stop = min(start + args.batch_size, len(rows))
                batch_rows = rows[start:stop]
                pixels = np.asarray(h5["pixels"][batch_rows], dtype=np.uint8)
                encoded = common.encode_pixels(model, pixels, args.device)
                encoded_np = encoded.detach().cpu().float().numpy()
                if embeddings is None:
                    embedding_dim = int(encoded_np.shape[1])
                    if embedding_dim != common.LEWM_CONTROL_LATENT_DIM:
                        raise RuntimeError(
                            "unexpected LeWM control latent dimension: "
                            f"expected={common.LEWM_CONTROL_LATENT_DIM}, "
                            f"actual={embedding_dim}"
                        )
                    embeddings = np.lib.format.open_memmap(
                        staging / "embeddings.npy",
                        mode="w+",
                        dtype=np.dtype(args.embedding_dtype),
                        shape=(len(rows), embedding_dim),
                    )
                if encoded_np.shape != (stop - start, embedding_dim):
                    raise RuntimeError(f"encoder shape changed at rows {start}:{stop}")
                embeddings[start:stop] = encoded_np.astype(args.embedding_dtype)
                targets[start:stop, :3] = np.asarray(
                    h5["privileged_block_0_pos"][batch_rows], dtype=np.float32
                )
                targets[start:stop, 3] = np.asarray(
                    h5["privileged_block_0_yaw"][batch_rows], dtype=np.float32
                ).reshape(-1)
                if start == 0 or stop == len(rows) or stop % (50 * args.batch_size) == 0:
                    print(f"encoded {stop}/{len(rows)} frames")
        assert embeddings is not None and embedding_dim is not None
        embeddings.flush()
        targets.flush()
        del embeddings, targets

        arrays = {}
        for name in (
            "rows.npy",
            "episodes.npy",
            "split.npy",
            "targets_block4d.npy",
            "embeddings.npy",
        ):
            path = staging / name
            arrays[name] = {
                "size": path.stat().st_size,
                "sha256": common.sha256_file(path),
            }
        metadata = {
            "format_version": "cube_block4d_embedding_dataset_v1",
            "dataset": common.file_identity(dataset, include_sha256=False),
            "manifest": common.file_identity(manifest),
            "checkpoint": args.checkpoint,
            "world_model_state_sha256": world_model_state_sha256,
            "encoder_contract": (
                "192D LeWM encode(pixels) projector-after-CLS control latent; "
                "ImageNet normalization; 224x224"
            ),
            "embedding_dim": embedding_dim,
            "embedding_dtype": args.embedding_dtype,
            "target_names": list(common.TARGET_NAMES),
            "target_units": ["m", "m", "m", "rad"],
            "yaw_representation": "wrapped scalar radians in [-pi,pi]",
            "seed": args.seed,
            "max_frames": args.max_frames,
            "sampling_mode": args.sampling_mode,
            "sampling": sampling_details,
            "num_rows": len(rows),
            "split_id": {name: i for i, name in enumerate(SPLIT_NAMES)},
            "split_row_counts": {
                name: int(np.count_nonzero(row_split == i))
                for i, name in enumerate(SPLIT_NAMES)
            },
            "split_episode_counts": {
                name: len(split_episodes[name]) for name in SPLIT_NAMES
            },
            "split_episodes": split_episodes,
            "excluded_formal_episodes": excluded,
            "excluded_formal_rows": formal_rows,
            "arrays": arrays,
            "builder": common.file_identity(Path(__file__)),
        }
        common.write_json(staging / "metadata.json", metadata)
        _commit(staging, output, args.overwrite)
    except BaseException:
        # Preserve a failed staging directory for diagnosis; never replace a
        # previously complete output with a partial build.
        print(f"incomplete staging retained for inspection: {staging}", file=sys.stderr)
        raise

    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build episode-disjoint LeWM Cube block xyz+yaw embeddings"
    )
    parser.add_argument("--dataset", type=Path, default=common.DATASET_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST_DEFAULT)
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--output", type=Path, default=common.PROBE_DATA_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400_000,
        help="deterministic frame cap; use 0 for all eligible frames",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("episode_blocks", "random_rows"),
        default="episode_blocks",
        help=(
            "episode_blocks (default) reads a compact seeded episode window; "
            "random_rows preserves the older full-dataset random-row behavior"
        ),
    )
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
