#!/usr/bin/env python3
"""Train the strict robust_v1 192D-latent to block-XYZ probe.

The input is an embedding dataset produced by ``build_cube_probe_dataset.py``
from the canonical robust_v1 checkpoint.  Existing episode-disjoint split IDs
are preserved and independently verified; the frozen 50 evaluation episodes
must be absent.  The resulting checkpoint is directly loadable by
``cube_imagination_error_common.LoadedXYZProbe``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cube_imagination_error_common as common  # noqa: E402


PROJECT = HERE.parents[1]
ROBUST_CHECKPOINT = (
    PROJECT
    / "checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt"
)
ROBUST_CHECKPOINT_SHA256 = (
    "cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed"
)
ROBUST_MODEL_CONFIG = (
    PROJECT
    / "checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/config.json"
)
ROBUST_MODEL_CONFIG_SHA256 = (
    "86f2ed24c61b48354416c23af51aa51279ae28a33cb36b7ebc3d057eec2b8c0d"
)
ROBUST_RUN_PLAN = (
    PROJECT / "outputs/train/robust_v1/lewm-cube-robust_v1/run_plan.json"
)
ROBUST_RUN_PLAN_SHA256 = (
    "5830ad4091e13764f4eee765805e247e36c6968b52afd34397ef65745752bbf9"
)
DEFAULT_DATASET = PROJECT / "outputs/probe/cube_robust_v1/dataset"
OUTPUT_PARENT = PROJECT / "models/probes"
DEFAULT_OUTPUT = OUTPUT_PARENT / "cube_robust_v1_xyz"
CHECKPOINT_NAME = "robust_v1.pt"
SPLIT_NAMES = ("train", "val", "test")
QUALITY_LIMIT_MM = 15.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _identity(path: Path) -> dict[str, Any]:
    return common.file_identity(path.resolve())


def _canonical_robust_contract() -> dict[str, Any]:
    checkpoint = _identity(ROBUST_CHECKPOINT)
    config = _identity(ROBUST_MODEL_CONFIG)
    run_plan = _identity(ROBUST_RUN_PLAN)
    if checkpoint["sha256"] != ROBUST_CHECKPOINT_SHA256:
        raise ValueError(
            "canonical robust_v1 weights changed: "
            f"expected={ROBUST_CHECKPOINT_SHA256}, actual={checkpoint['sha256']}, "
            f"path={ROBUST_CHECKPOINT}"
        )
    if config["sha256"] != ROBUST_MODEL_CONFIG_SHA256:
        raise ValueError(
            "canonical robust_v1 model config changed: "
            f"expected={ROBUST_MODEL_CONFIG_SHA256}, actual={config['sha256']}, "
            f"path={ROBUST_MODEL_CONFIG}"
        )
    if run_plan["sha256"] != ROBUST_RUN_PLAN_SHA256:
        raise ValueError(
            "canonical robust_v1 run plan changed: "
            f"expected={ROBUST_RUN_PLAN_SHA256}, actual={run_plan['sha256']}, "
            f"path={ROBUST_RUN_PLAN}"
        )
    return {
        "checkpoint": checkpoint,
        "model_config": config,
        "training_run_plan": run_plan,
        "expected_weights_sha256": ROBUST_CHECKPOINT_SHA256,
        "expected_model_config_sha256": ROBUST_MODEL_CONFIG_SHA256,
        "expected_run_plan_sha256": ROBUST_RUN_PLAN_SHA256,
    }


def _array_identity(root: Path, filename: str, expected: Mapping[str, Any], verify: bool) -> None:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    expected_size = int(expected.get("size", -1))
    if actual_size != expected_size:
        raise ValueError(
            f"embedding array size mismatch: file={filename}, "
            f"expected={expected_size}, actual={actual_size}"
        )
    if verify:
        actual_sha = common.sha256_file(path)
        expected_sha = str(expected.get("sha256"))
        if actual_sha != expected_sha:
            raise ValueError(
                f"embedding array SHA mismatch: file={filename}, "
                f"expected={expected_sha}, actual={actual_sha}"
            )


def _fixed50_contract(dataset_path: Path, manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import hdf5plugin  # noqa: F401
    import h5py

    manifest = _read_json(manifest_path)
    rows = np.asarray(manifest.get("formal_rows"), dtype=np.int64)
    if rows.shape != (50,) or np.any(np.diff(rows) <= 0):
        raise ValueError(f"formal_rows must be 50 increasing rows, got {rows.shape}")
    with h5py.File(dataset_path, "r", swmr=True) as h5:
        episodes = np.asarray(h5["ep_idx"][rows], dtype=np.int64)
    if len(np.unique(episodes)) != 50:
        raise ValueError("formal rows must identify 50 unique episodes")
    return rows, episodes


def _validate_episode_split(
    episodes: np.ndarray,
    split: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if episodes.shape != split.shape or episodes.ndim != 1:
        raise ValueError(f"episodes/split shapes differ: {episodes.shape}/{split.shape}")
    unique_split = np.unique(split)
    if not np.array_equal(unique_split, np.asarray([0, 1, 2], dtype=unique_split.dtype)):
        raise ValueError(f"split IDs must be exactly 0/1/2, got {unique_split.tolist()}")
    sets = {
        name: set(map(int, np.unique(episodes[split == split_id])))
        for split_id, name in enumerate(SPLIT_NAMES)
    }
    for left in range(3):
        for right in range(left + 1, 3):
            overlap = sets[SPLIT_NAMES[left]] & sets[SPLIT_NAMES[right]]
            if overlap:
                raise ValueError(
                    "episode split leakage: "
                    f"splits={SPLIT_NAMES[left]}/{SPLIT_NAMES[right]}, "
                    f"examples={sorted(overlap)[:10]}"
                )
    expected_counts = metadata.get("split_row_counts", {})
    actual_counts = {
        name: int(np.count_nonzero(split == split_id))
        for split_id, name in enumerate(SPLIT_NAMES)
    }
    if {key: int(value) for key, value in expected_counts.items()} != actual_counts:
        raise ValueError(
            f"split row counts disagree with metadata: "
            f"expected={expected_counts}, actual={actual_counts}"
        )
    metadata_episodes = metadata.get("split_episodes", {})
    for name in SPLIT_NAMES:
        expected = set(map(int, metadata_episodes.get(name, [])))
        if expected != sets[name]:
            raise ValueError(
                f"split episode set disagrees with metadata for {name}: "
                f"expected_count={len(expected)}, actual_count={len(sets[name])}"
            )
    return {
        "episode_disjoint": True,
        "split_row_counts": actual_counts,
        "split_episode_counts": {name: len(sets[name]) for name in SPLIT_NAMES},
    }


def _validate_dataset(
    root: Path,
    source_dataset: Path,
    manifest: Path,
    verify_hashes: bool,
) -> dict[str, Any]:
    root = common.ensure_data_disk(root, "robust probe embedding dataset")
    source_dataset = common.ensure_data_disk(source_dataset, "source HDF5 dataset")
    manifest = common.ensure_data_disk(manifest, "fixed50 manifest")
    metadata_path = root / "metadata.json"
    metadata = _read_json(metadata_path)
    if metadata.get("format_version") != "cube_block4d_embedding_dataset_v1":
        raise ValueError(f"unsupported embedding format: {metadata.get('format_version')}")
    if int(metadata.get("embedding_dim", -1)) != common.LATENT_DIM:
        raise ValueError(
            f"embedding dimension must be {common.LATENT_DIM}, "
            f"actual={metadata.get('embedding_dim')}"
        )
    target_names = list(metadata.get("target_names", []))
    if target_names[:3] != ["block_x", "block_y", "block_z"]:
        raise ValueError(f"embedding targets do not begin with strict XYZ: {target_names}")
    metadata_checkpoint = Path(str(metadata.get("checkpoint", ""))).expanduser().resolve()
    if metadata_checkpoint != ROBUST_CHECKPOINT.resolve():
        raise ValueError(
            "embedding dataset was not encoded by canonical robust_v1: "
            f"expected={ROBUST_CHECKPOINT.resolve()}, actual={metadata_checkpoint}"
        )
    canonical = _canonical_robust_contract()
    required_arrays = (
        "embeddings.npy",
        "targets_block4d.npy",
        "split.npy",
        "episodes.npy",
        "rows.npy",
    )
    declared = metadata.get("arrays", {})
    for filename in required_arrays:
        if filename not in declared:
            raise ValueError(f"embedding metadata is missing array identity: {filename}")
        _array_identity(root, filename, declared[filename], verify_hashes)

    embeddings = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(root / "targets_block4d.npy", mmap_mode="r", allow_pickle=False)
    split = np.load(root / "split.npy", mmap_mode="r", allow_pickle=False)
    episodes = np.load(root / "episodes.npy", mmap_mode="r", allow_pickle=False)
    rows = np.load(root / "rows.npy", mmap_mode="r", allow_pickle=False)
    count = int(metadata.get("num_rows", -1))
    expected_shapes = {
        "embeddings": ((count, common.LATENT_DIM), embeddings.shape),
        "targets": ((count, 4), targets.shape),
        "split": ((count,), split.shape),
        "episodes": ((count,), episodes.shape),
        "rows": ((count,), rows.shape),
    }
    malformed = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in expected_shapes.items()
        if expected != actual
    }
    if malformed:
        raise ValueError(f"embedding array shapes malformed: {malformed}")
    if np.any(np.diff(np.asarray(rows, dtype=np.int64)) <= 0):
        raise ValueError("embedding source rows must be unique and strictly increasing")
    split_contract = _validate_episode_split(
        np.asarray(episodes, dtype=np.int64), np.asarray(split, dtype=np.uint8), metadata
    )
    formal_rows, formal_episodes = _fixed50_contract(source_dataset, manifest)
    metadata_rows = np.asarray(metadata.get("excluded_formal_rows"), dtype=np.int64)
    metadata_episodes = np.asarray(metadata.get("excluded_formal_episodes"), dtype=np.int64)
    if not np.array_equal(metadata_rows, formal_rows):
        raise ValueError("embedding metadata excludes different formal rows")
    if set(metadata_episodes.tolist()) != set(formal_episodes.tolist()):
        raise ValueError("embedding metadata excludes different formal episodes")
    leaked = np.intersect1d(np.asarray(episodes, dtype=np.int64), formal_episodes)
    if len(leaked):
        raise ValueError(f"fixed50 episode leakage in probe data: {leaked[:10].tolist()}")
    world_model_state_sha = str(metadata.get("world_model_state_sha256", ""))
    if len(world_model_state_sha) != 64:
        raise ValueError("embedding metadata has malformed world-model state SHA")
    return {
        "root": root,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "metadata_identity": _identity(metadata_path),
        "arrays": {
            "embeddings": embeddings,
            "targets": targets,
            "split": split,
        },
        "fixed50": {
            "manifest": _identity(manifest),
            "source_dataset": common.file_identity(source_dataset, include_sha256=False),
            "formal_rows": formal_rows,
            "formal_episodes": formal_episodes,
            "excluded": True,
        },
        "split_contract": split_contract,
        "canonical_robust_v1": canonical,
        "world_model_state_sha256": world_model_state_sha,
    }


def _moments(array: np.ndarray, indices: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(dimensions, dtype=np.float64)
    squares = np.zeros(dimensions, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 16_384):
        values = np.asarray(array[indices[start : start + 16_384], :dimensions], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite values in normalization block starting {start}")
        total += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
        count += len(values)
    if not count:
        raise ValueError("cannot normalize an empty split")
    mean = total / count
    variance = np.maximum(squares / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class _XYZDataset:
    def __init__(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        normalization: Mapping[str, np.ndarray],
    ) -> None:
        self.embeddings = embeddings
        self.targets = targets
        self.indices = indices
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = int(self.indices[index])
        embedding = np.asarray(self.embeddings[row], dtype=np.float32)
        target = np.asarray(self.targets[row, :3], dtype=np.float32)
        return (
            (embedding - self.normalization["input_mean"])
            / self.normalization["input_scale"],
            (target - self.normalization["target_mean"])
            / self.normalization["target_scale"],
        )


def _predict(
    model: Any,
    embeddings: np.ndarray,
    indices: np.ndarray,
    normalization: Mapping[str, np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            values = np.asarray(embeddings[indices[start : start + batch_size]], dtype=np.float32)
            values = torch.from_numpy(
                (values - normalization["input_mean"])
                / normalization["input_scale"]
            ).to(device)
            decoded = model(values).float().cpu().numpy()
            outputs.append(
                decoded * normalization["target_scale"]
                + normalization["target_mean"]
            )
    return np.concatenate(outputs, axis=0)


def _quality_metric(payload: Mapping[str, Any]) -> float:
    try:
        value = float(payload["metrics"]["test"]["xyz_error_mm"]["median"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("test median XYZ metric is unavailable") from error
    if not np.isfinite(value) or not value < QUALITY_LIMIT_MM:
        raise ValueError(
            "probe quality gate failed: "
            f"expected=median test XYZ < {QUALITY_LIMIT_MM:.1f}mm, "
            f"actual={value:.6f}mm"
        )
    return value


def _train(
    embeddings: np.ndarray,
    targets: np.ndarray,
    split: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader

    indices = {
        name: np.flatnonzero(split == split_id).astype(np.int64)
        for split_id, name in enumerate(SPLIT_NAMES)
    }
    input_mean, input_scale = _moments(
        embeddings, indices["train"], common.LATENT_DIM
    )
    target_mean, target_scale = _moments(targets, indices["train"], common.XYZ_DIM)
    normalization = {
        "input_mean": input_mean,
        "input_scale": input_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = common.make_xyz_probe(common.LATENT_DIM, args.hidden_dim).to(args.device)
    loader = DataLoader(
        _XYZDataset(embeddings, targets, indices["train"], normalization),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_median = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for values, target in loader:
            values = values.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.square(model(values) - target).mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(values)
            count += len(values)
        validation = _predict(
            model,
            embeddings,
            indices["val"],
            normalization,
            args.device,
            args.eval_batch_size,
        )
        validation_metrics = common.xyz_probe_metrics(
            np.asarray(targets[indices["val"], :3], dtype=np.float32), validation
        )
        median = float(validation_metrics["xyz_error_mm"]["median"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_normalized_xyz_mse": loss_sum / count,
                "val_xyz_error_mm": validation_metrics["xyz_error_mm"],
            }
        )
        print(
            f"epoch={epoch + 1} loss={loss_sum / count:.7f} "
            f"val_median_xyz_mm={median:.4f}"
        )
        if median < best_median:
            best_median = median
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no best checkpoint")
    model.load_state_dict(best_state)
    metrics = {}
    for name, selected in indices.items():
        prediction = _predict(
            model,
            embeddings,
            selected,
            normalization,
            args.device,
            args.eval_batch_size,
        )
        metrics[name] = common.xyz_probe_metrics(
            np.asarray(targets[selected, :3], dtype=np.float32), prediction
        )
    return model, normalization, metrics, history


def _prepare_output(output: Path, overwrite: bool) -> tuple[Path, Path]:
    output = common.ensure_data_disk(output, "XYZ probe output")
    parent = OUTPUT_PARENT.resolve()
    if output == parent or parent not in output.parents:
        raise ValueError(f"output must be a child of {parent}: {output}")
    if output.is_symlink():
        raise ValueError(f"refusing symlink output: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"non-empty output: {output}; pass --overwrite")
    staging = output.parent / f".{output.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    return output, staging


def run(args: argparse.Namespace) -> int:
    common.configure_storage()
    if min(args.epochs, args.patience, args.batch_size, args.eval_batch_size) <= 0:
        raise ValueError("epochs, patience, and batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    dataset = _validate_dataset(
        args.dataset,
        args.source_dataset,
        args.manifest,
        args.verify_hashes,
    )
    arrays = dataset["arrays"]
    model, normalization, metrics, history = _train(
        arrays["embeddings"], arrays["targets"], arrays["split"], args
    )
    payload = {
        "format_version": "cube_imagination_error_xyz_probe_v1",
        "model_label": "robust_v1",
        "input_dim": common.LATENT_DIM,
        "hidden_dim": args.hidden_dim,
        "target_names": ["block_x", "block_y", "block_z"],
        "target_units": ["m", "m", "m"],
        **normalization,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "embedding_dataset_metadata": str(dataset["metadata_path"]),
        "embedding_dataset_metadata_sha256": dataset["metadata_identity"]["sha256"],
        "world_model_state_sha256": dataset["world_model_state_sha256"],
        "world_model_checkpoint": str(ROBUST_CHECKPOINT.resolve()),
        "canonical_robust_v1": dataset["canonical_robust_v1"],
        "data_contract": {
            "strict_xyz_only": True,
            "episode_split": dataset["split_contract"],
            "fixed50": dataset["fixed50"],
        },
        "training": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "selection": "lowest validation median XYZ error in millimeters",
            "loss": "normalized XYZ MSE only; no yaw target or loss",
        },
        "metrics": metrics,
    }
    test_median = _quality_metric(payload)
    output, staging = _prepare_output(args.output, args.overwrite)
    try:
        import torch

        checkpoint = staging / CHECKPOINT_NAME
        torch.save(payload, checkpoint)
        common.write_json(staging / "history.json", history)
        checkpoint_identity = _identity(checkpoint)
        checkpoint_identity["path"] = str((output / CHECKPOINT_NAME).resolve())
        training_metadata = {
            "format_version": "cube_robust_v1_xyz_probe_training_v1",
            "checkpoint": checkpoint_identity,
            "embedding_dataset_metadata": dataset["metadata_identity"],
            "canonical_robust_v1": dataset["canonical_robust_v1"],
            "world_model_state_sha256": dataset["world_model_state_sha256"],
            "strict_xyz_only": True,
            "episode_split": dataset["split_contract"],
            "fixed50": dataset["fixed50"],
            "quality_gate": {
                "metric": "test median XYZ error in millimeters",
                "operator": "strictly_less_than",
                "threshold_mm": QUALITY_LIMIT_MM,
                "actual_mm": test_median,
                "passed": True,
            },
            "metrics": metrics,
            "training": payload["training"],
            "trainer": _identity(Path(__file__)),
        }
        common.write_json(staging / "metadata.json", training_metadata)
        common.write_text(
            staging / "REPORT.md",
            "# robust_v1 strict XYZ probe\n\n"
            f"Test median XYZ error: {test_median:.6f} mm "
            f"(required < {QUALITY_LIMIT_MM:.1f} mm).\n\n"
            "The target is block xyz only. Splits are episode-disjoint and "
            "exclude the frozen 50 evaluation episodes.\n",
        )
        # Validate the exact serialized format before replacing any prior run.
        loaded = common.LoadedXYZProbe(checkpoint, "cpu")
        if loaded.payload["world_model_state_sha256"] != dataset["world_model_state_sha256"]:
            raise RuntimeError("serialized checkpoint changed world-model provenance")
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except BaseException:
        print(f"incomplete training staging retained: {staging}", file=sys.stderr)
        raise
    print(output)
    return 0


def self_test() -> int:
    common.configure_storage()
    import torch

    rng = np.random.default_rng(42)
    count = 240
    embeddings = rng.normal(size=(count, common.LATENT_DIM)).astype(np.float32)
    targets = np.zeros((count, 4), dtype=np.float32)
    targets[:, :3] = (
        np.asarray([0.42, 0.0, 0.025], dtype=np.float32)
        + 0.001 * embeddings[:, :3]
    )
    split = np.repeat(np.asarray([0, 1, 2], dtype=np.uint8), [160, 40, 40])
    args = argparse.Namespace(
        seed=42,
        hidden_dim=16,
        device="cpu",
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        epochs=2,
        patience=2,
    )
    model, normalization, metrics, history = _train(
        embeddings, targets, split, args
    )
    payload = {
        "format_version": "cube_imagination_error_xyz_probe_v1",
        "model_label": "synthetic_robust_v1",
        "input_dim": common.LATENT_DIM,
        "hidden_dim": args.hidden_dim,
        "target_names": ["block_x", "block_y", "block_z"],
        "target_units": ["m", "m", "m"],
        **normalization,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "embedding_dataset_metadata": "synthetic",
        "embedding_dataset_metadata_sha256": "0" * 64,
        "world_model_state_sha256": "1" * 64,
        "world_model_checkpoint": "synthetic",
        "training": {"loss": "normalized XYZ MSE only"},
        "metrics": metrics,
    }
    test_median = _quality_metric(payload)
    temporary_parent = PROJECT.parent / "tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
        checkpoint = Path(directory) / "synthetic.pt"
        torch.save(payload, checkpoint)
        loaded = common.LoadedXYZProbe(checkpoint, "cpu")
        decoded = loaded(torch.from_numpy(embeddings[:4]))
        assert tuple(decoded.shape) == (4, 3)
        assert loaded.payload["target_names"] == ["block_x", "block_y", "block_z"]
    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint_loader_compatible": True,
                "strict_xyz_only": True,
                "synthetic_epochs": len(history),
                "synthetic_test_median_xyz_mm": test_median,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-dataset", type=Path, default=common.DATASET)
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-hashes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
