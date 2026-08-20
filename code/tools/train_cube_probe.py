#!/usr/bin/env python3
"""Train linear and small-MLP probes for Cube block xyz+yaw."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import cube_probe_common as common


OUTPUT_PARENT = common.AILAB_ROOT / "models/probes"
SPLIT_NAMES = ("train", "val", "test")


def _validate_dataset(root: Path, verify_hashes: bool) -> dict[str, Any]:
    root = common.ensure_data_disk(root, "probe dataset")
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != "cube_block4d_embedding_dataset_v1":
        raise ValueError(f"unsupported dataset format: {metadata.get('format_version')}")
    if int(metadata.get("embedding_dim", -1)) != common.LEWM_CONTROL_LATENT_DIM:
        raise ValueError(
            "probe dataset must contain the 192D LeWM control latent: "
            f"actual={metadata.get('embedding_dim')}"
        )
    for filename, identity in metadata["arrays"].items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != identity["size"]:
            raise ValueError(f"probe dataset array missing/size mismatch: {path}")
        if verify_hashes:
            digest = common.sha256_file(path)
            if digest != identity["sha256"]:
                raise ValueError(
                    f"probe dataset hash mismatch: expected={identity['sha256']}, "
                    f"actual={digest}, path={path}"
                )
    rows = np.load(root / "rows.npy", mmap_mode="r", allow_pickle=False)
    episodes = np.load(root / "episodes.npy", mmap_mode="r", allow_pickle=False)
    split = np.load(root / "split.npy", mmap_mode="r", allow_pickle=False)
    target = np.load(root / "targets_block4d.npy", mmap_mode="r", allow_pickle=False)
    emb = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    n = int(metadata["num_rows"])
    expected = {
        "rows": (rows.shape, (n,)),
        "episodes": (episodes.shape, (n,)),
        "split": (split.shape, (n,)),
        "target": (target.shape, (n, 4)),
        "embedding": (emb.shape, (n, int(metadata["embedding_dim"]))),
    }
    bad = {key: value for key, value in expected.items() if value[0] != value[1]}
    if bad:
        raise ValueError(f"probe dataset shapes malformed: {bad}")
    if not np.all(np.isin(np.unique(split), [0, 1, 2])):
        raise ValueError("probe split IDs must be 0/1/2")
    excluded = set(map(int, metadata["excluded_formal_episodes"]))
    if any(int(ep) in excluded for ep in np.unique(episodes)):
        raise ValueError("formal evaluation episode leaked into probe dataset")
    episode_splits: dict[int, set[int]] = {}
    for split_id in range(3):
        episode_splits[split_id] = set(map(int, np.unique(episodes[split == split_id])))
    if any(episode_splits[a] & episode_splits[b] for a in range(3) for b in range(a + 1, 3)):
        raise ValueError("episode-disjoint split invariant failed")
    return metadata


def _moments(array: np.ndarray, indices: np.ndarray, chunk: int = 16_384) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(array.shape[1], dtype=np.float64)
    squares = np.zeros(array.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk):
        value = np.asarray(array[indices[start : start + chunk]], dtype=np.float64)
        common.finite_or_raise("normalization batch", value)
        total += value.sum(axis=0)
        squares += np.square(value).sum(axis=0)
        count += len(value)
    mean = total / count
    variance = np.maximum(squares / count - mean**2, 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class _ProbeDataset:
    def __init__(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        input_mean: np.ndarray,
        input_scale: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
    ) -> None:
        self.embeddings = embeddings
        self.targets = targets
        self.indices = indices
        self.input_mean = input_mean
        self.input_scale = input_scale
        self.target_mean = target_mean
        self.target_scale = target_scale

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = int(self.indices[index])
        embedding = np.asarray(self.embeddings[row], dtype=np.float32)
        target = np.asarray(self.targets[row], dtype=np.float32)
        return (
            (embedding - self.input_mean) / self.input_scale,
            (target - self.target_mean) / self.target_scale,
        )


def _loss(prediction: Any, target: Any, target_mean: Any, target_scale: Any) -> Any:
    import torch

    xyz = torch.square(prediction[..., :3] - target[..., :3]).mean()
    prediction_yaw = prediction[..., 3] * target_scale[3] + target_mean[3]
    target_yaw = target[..., 3] * target_scale[3] + target_mean[3]
    yaw = torch.square(
        common.wrap_angle_torch(prediction_yaw - target_yaw) / target_scale[3]
    ).mean()
    return xyz + yaw


def _predict(
    model: Any,
    embeddings: np.ndarray,
    indices: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            x = np.asarray(embeddings[indices[start : start + batch_size]], dtype=np.float32)
            x = torch.from_numpy((x - input_mean) / input_scale).to(device)
            normalized = model(x).float().cpu().numpy()
            physical = normalized * target_scale + target_mean
            physical[:, 3] = common.wrap_angle_np(physical[:, 3])
            outputs.append(physical)
    return np.concatenate(outputs, axis=0)


def _train_one(
    kind: str,
    hidden_dim: int,
    embeddings: np.ndarray,
    targets: np.ndarray,
    split_indices: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], list[dict[str, float]]]:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed + (0 if kind == "linear" else 1))
    model = common.make_probe(kind, embeddings.shape[1], hidden_dim).to(args.device)
    train_data = _ProbeDataset(
        embeddings,
        targets,
        split_indices["train"],
        input_mean,
        input_scale,
        target_mean,
        target_scale,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    target_mean_t = torch.from_numpy(target_mean).to(args.device)
    target_scale_t = torch.from_numpy(target_scale).to(args.device)
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, target in loader:
            x = x.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(x), target, target_mean_t, target_scale_t)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(x)
            count += len(x)
        val_prediction = _predict(
            model,
            embeddings,
            split_indices["val"],
            input_mean,
            input_scale,
            target_mean,
            target_scale,
            args.device,
            args.eval_batch_size,
        )
        val_metrics = common.probe_metrics(
            np.asarray(targets[split_indices["val"]], dtype=np.float32),
            val_prediction,
        )
        val_score = val_metrics["median_xyz_error_mm"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / count,
                "val_median_xyz_error_mm": val_score,
                "val_mean_xyz_error_mm": val_metrics["mean_xyz_error_mm"],
            }
        )
        print(
            f"{kind} epoch={epoch + 1} loss={loss_sum / count:.6f} "
            f"val_median_xyz_mm={val_score:.3f}"
        )
        if val_score < best_val:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{kind} training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics = {}
    for split_name in SPLIT_NAMES:
        idx = split_indices[split_name]
        prediction = _predict(
            model,
            embeddings,
            idx,
            input_mean,
            input_scale,
            target_mean,
            target_scale,
            args.device,
            args.eval_batch_size,
        )
        metrics[split_name] = common.probe_metrics(
            np.asarray(targets[idx], dtype=np.float32), prediction
        )
    return model, metrics, history


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = common.ensure_output_child(path, OUTPUT_PARENT, "probe checkpoint output")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(args: argparse.Namespace) -> int:
    common.configure_storage()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    metadata = _validate_dataset(args.dataset, args.verify_hashes)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = args.dataset.resolve()
    embeddings = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(root / "targets_block4d.npy", mmap_mode="r", allow_pickle=False)
    split = np.load(root / "split.npy", mmap_mode="r", allow_pickle=False)
    split_indices = {
        name: np.flatnonzero(split == split_id).astype(np.int64)
        for split_id, name in enumerate(SPLIT_NAMES)
    }
    input_mean, input_scale = _moments(embeddings, split_indices["train"])
    target_mean, target_scale = _moments(targets, split_indices["train"])
    # Yaw is a wrapped scalar.  A pi scale keeps its optimization weight stable
    # and avoids a data-dependent discontinuity at -pi/pi.
    target_mean[3] = 0.0
    target_scale[3] = np.pi

    output = _prepare_output(args.output, args.overwrite)
    dataset_metadata_path = root / "metadata.json"
    metadata_sha = common.sha256_file(dataset_metadata_path)
    summaries = {}
    for kind in ("linear", "mlp"):
        model, metrics, history = _train_one(
            kind,
            args.hidden_dim,
            embeddings,
            targets,
            split_indices,
            input_mean,
            input_scale,
            target_mean,
            target_scale,
            args,
        )
        checkpoint = {
            "format_version": "cube_block4d_probe_v1",
            "model_kind": kind,
            "input_dim": int(embeddings.shape[1]),
            "hidden_dim": args.hidden_dim,
            "input_mean": input_mean,
            "input_scale": input_scale,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "target_names": list(common.TARGET_NAMES),
            "target_units": ["m", "m", "m", "rad"],
            "yaw_loss": "circular_wrapped_error",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "dataset_metadata": str(dataset_metadata_path),
            "dataset_metadata_sha256": metadata_sha,
            "world_model_state_sha256": metadata["world_model_state_sha256"],
            "training": {
                "seed": args.seed,
                "epochs_requested": args.epochs,
                "epochs_completed": len(history),
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
            },
            "metrics": metrics,
        }
        path = output / f"{kind}.pt"
        torch.save(checkpoint, path)
        common.write_json(output / f"{kind}_history.json", history)
        summaries[kind] = {
            "checkpoint": common.file_identity(path),
            "metrics": metrics,
            "epochs_completed": len(history),
        }

    report = {
        "format_version": "cube_block4d_probe_training_v1",
        "dataset_metadata": common.file_identity(dataset_metadata_path),
        "dataset_format": metadata["format_version"],
        "normalization": {
            "input_mean": input_mean,
            "input_scale": input_scale,
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        "split_row_counts": {name: len(idx) for name, idx in split_indices.items()},
        "models": summaries,
        "trainer": common.file_identity(Path(__file__)),
    }
    common.write_json(output / "metrics.json", report)
    table_rows = []
    for kind in ("linear", "mlp"):
        for split_name in SPLIT_NAMES:
            metric = summaries[kind]["metrics"][split_name]
            table_rows.append(
                [
                    kind,
                    split_name,
                    f"{metric['median_xyz_error_mm']:.3f}",
                    *(f"{metric['r2_per_dimension'][name]:.6f}" for name in common.TARGET_NAMES),
                ]
            )
    (output / "REPORT.md").write_text(
        "# Cube 4D block-state probe\n\n"
        + common.markdown_table(
            ["Probe", "Split", "Median xyz error (mm)", "R² x", "R² y", "R² z", "R² yaw"],
            table_rows,
        )
        + "\n\nYaw errors are circularly wrapped. Online ranking defaults to xyz only; yaw weight is zero.\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train linear and MLP Cube block4d probes")
    parser.add_argument("--dataset", type=Path, default=common.PROBE_DATA_DEFAULT)
    parser.add_argument("--output", type=Path, default=common.PROBE_MODEL_DEFAULT)
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
    parser.add_argument("--verify-hashes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
