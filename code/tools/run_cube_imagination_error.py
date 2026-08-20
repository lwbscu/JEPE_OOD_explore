#!/usr/bin/env python3
"""Run the frozen two-checkpoint Cube imagination-error experiment.

Stages:
1. Reuse Route1's official 400k embedding dataset, build a paired 400k dataset
   for Route2.1 Masked, then train a fresh XYZ-only MLP for each checkpoint.
2. Measure five-depth autoregressive rollout error on 2,000 expert segments.
   Real expert action blocks are teacher-forced; latent states are never fed
   back from the encoder after the initial frame.
3. Measure terminal error on the frozen unseeded red/blue-v2/yellow-v2
   12x300 candidate pools using their cached terminal images and positions.
4. Emit JSON, CSV, and Markdown under the one frozen output root.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import hdf5plugin  # noqa: F401  # Must register filters before importing h5py.
import h5py
import numpy as np

import cube_imagination_error_common as common
import cube_probe_common as route1


PROBE_DATA_BUILDER = Path(__file__).with_name("build_cube_probe_dataset.py")
FORMAL_SEGMENTS = 2_000
OOD_SEGMENTS = 500
SEED = 42
DEPTHS = tuple(range(1, common.HORIZON + 1))
M1_FIELDS = (
    "model",
    "condition",
    "segment_id",
    "episode_idx",
    "start_row",
    "target_row",
    "depth",
    "action_teacher_forcing",
    "latent_teacher_forcing",
    "initial_color_intervention",
    "endpoint_encoder_domain",
    "true_x_m",
    "true_y_m",
    "true_z_m",
    "roll_x_m",
    "roll_y_m",
    "roll_z_m",
    "enc_x_m",
    "enc_y_m",
    "enc_z_m",
    "E_roll_mm",
    "E_enc_mm",
    "Delta_roll_minus_enc_mm",
    "E_imag_mm",
    "latent_l2",
    "latent_cosine_distance",
    "roll_gt_40mm",
    "roll_gt_40_and_enc_le_40",
)
M2_FIELDS = (
    "model",
    "condition",
    "env_idx",
    "dataset_row",
    "episode_idx",
    "candidate_idx",
    "rollout_depth",
    "frozen_action_pool",
    "true_terminal_x_m",
    "true_terminal_y_m",
    "true_terminal_z_m",
    "roll_x_m",
    "roll_y_m",
    "roll_z_m",
    "enc_x_m",
    "enc_y_m",
    "enc_z_m",
    "E_roll_mm",
    "E_enc_mm",
    "Delta_roll_minus_enc_mm",
    "E_imag_mm",
    "latent_l2",
    "latent_cosine_distance",
    "roll_gt_40mm",
    "roll_gt_40_and_enc_le_40",
    "final_success",
    "ever_success",
    "min_goal_distance_m",
    "final_goal_distance_m",
    "stored_latent_cost",
)


def _bool(text: str) -> bool:
    normalized = str(text).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"expected True/False, got {text!r}")
    return normalized == "true"


def _load_embedding_dataset(root: Path) -> dict[str, Any]:
    root = common.ensure_data_disk(root, "embedding dataset")
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != "cube_block4d_embedding_dataset_v1":
        raise ValueError(f"unsupported Route1 embedding dataset: {metadata.get('format_version')}")
    if int(metadata.get("num_rows", -1)) != 400_000:
        raise ValueError("probe embedding protocol requires exactly 400,000 rows")
    if int(metadata.get("embedding_dim", -1)) != common.LATENT_DIM:
        raise ValueError("probe embedding latent dimension mismatch")
    required = {
        "rows.npy": (400_000,),
        "episodes.npy": (400_000,),
        "split.npy": (400_000,),
        "targets_block4d.npy": (400_000, 4),
        "embeddings.npy": (400_000, common.LATENT_DIM),
    }
    arrays = {}
    for filename, shape in required.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape:
            raise ValueError(f"{filename} shape mismatch: expected={shape}, actual={array.shape}")
        arrays[filename] = array
    formal = set(map(int, metadata.get("excluded_formal_episodes", [])))
    if len(formal) != 50:
        raise ValueError("embedding metadata must freeze exactly 50 excluded episodes")
    if any(int(value) in formal for value in np.unique(arrays["episodes.npy"])):
        raise ValueError("formal50 episode leaked into embedding dataset")
    split = arrays["split.npy"]
    if [int(np.count_nonzero(split == value)) for value in range(3)] != [320_000, 40_000, 40_000]:
        raise ValueError("embedding split is not frozen 320k/40k/40k")
    return {"root": root, "metadata": metadata, "arrays": arrays}


def _validate_paired_embeddings() -> dict[str, dict[str, Any]]:
    datasets = {label: _load_embedding_dataset(path) for label, path in common.EMBEDDING_DATASETS.items()}
    official = datasets["official"]["arrays"]
    masked = datasets["masked"]["arrays"]
    for filename in ("rows.npy", "episodes.npy", "split.npy", "targets_block4d.npy"):
        if not np.array_equal(official[filename], masked[filename]):
            raise ValueError(f"masked embedding dataset is not paired on {filename}")
    return datasets


def _moments(array: np.ndarray, indices: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(dimensions, dtype=np.float64)
    squares = np.zeros(dimensions, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 16_384):
        value = np.asarray(array[indices[start : start + 16_384], :dimensions], dtype=np.float64)
        common.finite("normalization values", value)
        total += value.sum(axis=0)
        squares += np.square(value).sum(axis=0)
        count += len(value)
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - np.square(mean), 0.0))
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class _XYZDataset:
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
        target = np.asarray(self.targets[row, :3], dtype=np.float32)
        return (
            (embedding - self.input_mean) / self.input_scale,
            (target - self.target_mean) / self.target_scale,
        )


def _predict_probe(
    model: Any,
    embeddings: np.ndarray,
    indices: np.ndarray,
    normalization: Mapping[str, np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            value = np.asarray(embeddings[indices[start : start + batch_size]], dtype=np.float32)
            value = torch.from_numpy(
                (value - normalization["input_mean"]) / normalization["input_scale"]
            ).to(device)
            decoded = model(value).float().cpu().numpy()
            output.append(
                decoded * normalization["target_scale"] + normalization["target_mean"]
            )
    return np.concatenate(output)


def _train_xyz_probe(
    model_label: str,
    dataset: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader

    arrays = dataset["arrays"]
    embeddings = arrays["embeddings.npy"]
    targets = arrays["targets_block4d.npy"]
    split = arrays["split.npy"]
    split_indices = {
        name: np.flatnonzero(split == split_id).astype(np.int64)
        for split_id, name in enumerate(("train", "val", "test"))
    }
    input_mean, input_scale = _moments(embeddings, split_indices["train"], common.LATENT_DIM)
    target_mean, target_scale = _moments(targets, split_indices["train"], common.XYZ_DIM)
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
    train_data = _XYZDataset(
        embeddings,
        targets,
        split_indices["train"],
        input_mean,
        input_scale,
        target_mean,
        target_scale,
    )
    loader = DataLoader(
        train_data,
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.probe_learning_rate, weight_decay=args.probe_weight_decay
    )
    best_state = None
    best_median = float("inf")
    stale = 0
    history = []
    for epoch in range(args.probe_epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for value, target in loader:
            value = value.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.square(model(value) - target).mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(value)
            count += len(value)
        val_prediction = _predict_probe(
            model,
            embeddings,
            split_indices["val"],
            normalization,
            args.device,
            args.probe_eval_batch_size,
        )
        val_metrics = common.xyz_probe_metrics(
            np.asarray(targets[split_indices["val"], :3]), val_prediction
        )
        median = float(val_metrics["xyz_error_mm"]["median"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_normalized_xyz_mse": loss_sum / count,
                "val_xyz_error_mm": val_metrics["xyz_error_mm"],
            }
        )
        print(
            f"xyz-probe {model_label} epoch={epoch + 1} "
            f"loss={loss_sum / count:.7f} val_median_mm={median:.4f}"
        )
        if median < best_median:
            best_median = median
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.probe_patience:
                break
    if best_state is None:
        raise RuntimeError(f"{model_label} xyz probe produced no best state")
    model.load_state_dict(best_state)
    metrics = {}
    for name, indices in split_indices.items():
        prediction = _predict_probe(
            model,
            embeddings,
            indices,
            normalization,
            args.device,
            args.probe_eval_batch_size,
        )
        metrics[name] = common.xyz_probe_metrics(
            np.asarray(targets[indices, :3]), prediction
        )
    payload = {
        "format_version": "cube_imagination_error_xyz_probe_v1",
        "model_label": model_label,
        "input_dim": common.LATENT_DIM,
        "hidden_dim": args.hidden_dim,
        "target_names": ["block_x", "block_y", "block_z"],
        "target_units": ["m", "m", "m"],
        **normalization,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "embedding_dataset_metadata": str(dataset["root"] / "metadata.json"),
        "embedding_dataset_metadata_sha256": common.sha256_file(
            dataset["root"] / "metadata.json"
        ),
        "world_model_state_sha256": dataset["metadata"]["world_model_state_sha256"],
        "world_model_checkpoint": str(common.CHECKPOINTS[model_label]),
        "training": {
            "seed": args.seed,
            "epochs_requested": args.probe_epochs,
            "epochs_completed": len(history),
            "patience": args.probe_patience,
            "learning_rate": args.probe_learning_rate,
            "weight_decay": args.probe_weight_decay,
            "batch_size": args.probe_batch_size,
            "selection": "lowest validation median XYZ error in millimeters",
            "loss": "normalized XYZ MSE only; no yaw target or loss",
        },
        "metrics": metrics,
    }
    return payload, history


def _validate_probe_links(device: str = "cpu") -> dict[str, common.LoadedXYZProbe]:
    datasets = _validate_paired_embeddings()
    probes = {}
    for label, path in common.PROBE_PATHS.items():
        probe = common.LoadedXYZProbe(path, device)
        metadata_sha = common.sha256_file(datasets[label]["root"] / "metadata.json")
        if probe.payload["embedding_dataset_metadata_sha256"] != metadata_sha:
            raise ValueError(f"{label} probe/embedding metadata hash mismatch")
        if probe.payload["world_model_state_sha256"] != datasets[label]["metadata"][
            "world_model_state_sha256"
        ]:
            raise ValueError(f"{label} probe/world-model metadata mismatch")
        _enforce_probe_quality(label, probe.payload, path)
        probes[label] = probe
    return probes


def _enforce_probe_quality(
    model_label: str, payload: Mapping[str, Any], checkpoint: Path
) -> float:
    try:
        actual = float(payload["metrics"]["test"]["xyz_error_mm"]["median"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"XYZ probe quality metric missing/malformed: model={model_label}, "
            f"checkpoint={checkpoint}, expected=test median < "
            f"{common.PROBE_TEST_MEDIAN_LIMIT_MM:.1f} mm, actual=unavailable"
        ) from error
    if not math.isfinite(actual) or not actual < common.PROBE_TEST_MEDIAN_LIMIT_MM:
        raise ValueError(
            f"XYZ probe quality gate failed: model={model_label}, checkpoint={checkpoint}, "
            f"expected=test median < {common.PROBE_TEST_MEDIAN_LIMIT_MM:.1f} mm, "
            f"actual={actual:.6f} mm"
        )
    return actual


def prepare_probes(args: argparse.Namespace) -> int:
    common.configure_storage()
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for requested probe preparation device")
    official = _load_embedding_dataset(common.OFFICIAL_EMBEDDINGS)
    masked_metadata = common.MASKED_EMBEDDINGS / "metadata.json"
    if not masked_metadata.is_file() or args.overwrite_embeddings:
        command = [
            sys.executable,
            str(PROBE_DATA_BUILDER),
            "--dataset", str(common.DATASET),
            "--manifest", str(common.MANIFEST),
            "--checkpoint", str(common.MASKED_CHECKPOINT),
            "--output", str(common.MASKED_EMBEDDINGS),
            "--device", args.device,
            "--batch-size", str(args.encoder_batch_size),
            "--max-frames", "400000",
            "--sampling-mode", "episode_blocks",
            "--embedding-dtype", "float32",
            "--seed", str(args.seed),
        ]
        if args.overwrite_embeddings:
            command.append("--overwrite")
        print("run:", json.dumps(command))
        subprocess.run(command, cwd=str(PROBE_DATA_BUILDER.parent), check=True, shell=False)
    datasets = _validate_paired_embeddings()
    # The paired metadata arrays are the data-level proof that formal50 are
    # excluded identically for both checkpoints.
    if official["metadata"]["excluded_formal_episodes"] != datasets["masked"][
        "metadata"
    ]["excluded_formal_episodes"]:
        raise ValueError("paired embedding datasets exclude different formal episodes")

    complete = all(path.is_file() for path in common.PROBE_PATHS.values())
    if complete and not args.overwrite_probes:
        _validate_probe_links("cpu")
        print("reusing complete paired XYZ probes")
        return 0
    existing_probe_outputs = [
        path
        for path in (
            *common.PROBE_PATHS.values(),
            *(common.PROBE_ROOT / f"{label}_history.json" for label in common.CHECKPOINTS),
            common.PROBE_ROOT / "training_summary.json",
        )
        if path.exists()
    ]
    if existing_probe_outputs and not args.overwrite_probes:
        raise FileExistsError(
            "incomplete probe outputs exist; inspect them, then use "
            f"--overwrite-probes explicitly: {existing_probe_outputs}"
        )
    common.PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    staging = common.PROBE_ROOT / f".building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        summaries = {}
        for label in common.CHECKPOINTS:
            payload, history = _train_xyz_probe(label, datasets[label], args)
            _enforce_probe_quality(label, payload, staging / f"{label}.pt")
            import torch

            torch.save(payload, staging / f"{label}.pt")
            common.write_json(staging / f"{label}_history.json", history)
            staged_identity = common.file_identity(staging / f"{label}.pt")
            summaries[label] = {
                "checkpoint": {
                    **staged_identity,
                    "path": str(common.PROBE_PATHS[label].resolve()),
                },
                "metrics": payload["metrics"],
                "epochs_completed": len(history),
                "world_model_state_sha256": payload["world_model_state_sha256"],
            }
        common.write_json(
            staging / "training_summary.json",
            {
                "format_version": "cube_imagination_error_xyz_probe_training_v1",
                "strict_xyz_only": True,
                "paired_rows_split_targets": True,
                "models": summaries,
            },
        )
        for path in staging.iterdir():
            os.replace(path, common.PROBE_ROOT / path.name)
        staging.rmdir()
    except BaseException:
        print(f"incomplete probe staging retained: {staging}", file=sys.stderr)
        raise
    _validate_probe_links("cpu")
    return 0


def _load_action_normalizer() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and cross-check the exact StandardScaler used by formal eval/audits."""
    reference_mean = None
    reference_scale = None
    sources = []
    for condition in common.CONDITIONS:
        for env_idx in common.AUDIT_ENVS:
            path = _audit_case(condition, env_idx) / "population.npz"
            with np.load(path, allow_pickle=False) as population:
                mean = np.asarray(population["action_scaler_mean"], dtype=np.float64)
                scale = np.asarray(population["action_scaler_scale"], dtype=np.float64)
            if mean.shape != (common.ACTION_DIM,) or scale.shape != (common.ACTION_DIM,):
                raise ValueError(
                    f"evaluation action scaler shape mismatch: {path}; "
                    f"expected={(common.ACTION_DIM,)}, actual={mean.shape}/{scale.shape}"
                )
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f"evaluation action scaler is malformed: {path}")
            if reference_mean is None:
                reference_mean, reference_scale = mean.copy(), scale.copy()
            elif not np.array_equal(mean, reference_mean) or not np.array_equal(
                scale, reference_scale
            ):
                raise ValueError(
                    f"evaluation action scaler mismatch: {path}; expected "
                    f"mean={reference_mean.tolist()}, scale={reference_scale.tolist()}; "
                    f"actual mean={mean.tolist()}, scale={scale.tolist()}"
                )
            sources.append(common.file_identity(path))
    if reference_mean is None or reference_scale is None:
        raise RuntimeError("no frozen audit action scaler was found")
    finite_rows = 0
    total_rows = 0
    with h5py.File(common.DATASET, "r", swmr=True) as h5:
        action = h5["action"]
        total_rows = int(len(action))
        for start in range(0, total_rows, 131_072):
            block = np.asarray(action[start : start + 131_072])
            finite_rows += int(np.count_nonzero(np.isfinite(block).all(axis=1)))
    if finite_rows != 2_000_000:
        raise ValueError(
            "evaluation action scaler source row count mismatch: "
            f"expected=2000000 finite rows, actual={finite_rows}, dataset={common.DATASET}"
        )
    return reference_mean, reference_scale, {
        "source": "old unseeded audit population.npz action_scaler_mean/action_scaler_scale",
        "fit_contract": "formal eval StandardScaler over all finite H5 action rows",
        "transform_contract": (
            "float32 copy; in-place subtract mean then in-place divide scale, "
            "bitwise-equivalent to sklearn StandardScaler.transform(float32)"
        ),
        "dataset": common.file_identity(common.DATASET, include_sha256=False),
        "dataset_total_rows": total_rows,
        "dataset_finite_action_rows": finite_rows,
        "verified_population_files": len(sources),
        "population_files": sources,
        "mean": reference_mean,
        "scale": reference_scale,
    }


def _select_segments(
    h5: h5py.File,
    count: int,
    ood_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    manifest = json.loads(common.MANIFEST.read_text(encoding="utf-8"))
    formal_rows = np.asarray(manifest["formal_rows"], dtype=np.int64)
    if formal_rows.shape != (50,) or len(np.unique(formal_rows)) != 50:
        raise ValueError("manifest must contain 50 unique formal rows")
    formal_episodes = np.asarray(h5["ep_idx"][formal_rows], dtype=np.int64)
    if len(np.unique(formal_episodes)) != 50:
        raise ValueError("formal rows do not map to 50 unique episodes")
    offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
    excluded = set(map(int, formal_episodes))
    candidates = []
    for episode, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        if episode in excluded or length <= common.HORIZON * common.ACTION_BLOCK:
            continue
        # Inclusive last start has target frame start+25 at episode end and
        # consumes action rows start..start+24, never the terminal NaN action.
        candidates.append(
            np.arange(
                int(offset),
                int(offset + length - common.HORIZON * common.ACTION_BLOCK),
                dtype=np.int64,
            )
        )
    eligible = np.concatenate(candidates)
    if count > len(eligible) or ood_count > count:
        raise ValueError("requested segment count exceeds eligible population")
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(eligible, size=count, replace=False))
    episodes = np.asarray(h5["ep_idx"][starts], dtype=np.int64)
    if np.intersect1d(episodes, formal_episodes).size:
        raise RuntimeError("formal50 episode leaked into measurement-one segments")
    for depth in DEPTHS:
        if not np.array_equal(np.asarray(h5["ep_idx"][starts + depth * common.ACTION_BLOCK]), episodes):
            raise RuntimeError(f"segment crosses episode boundary at depth {depth}")
    extension_ids = np.sort(np.random.default_rng(seed + 1).choice(count, size=ood_count, replace=False))
    return {
        "segment_id": np.arange(count, dtype=np.int64),
        "start_row": starts,
        "episode_idx": episodes,
        "extension_segment_id": extension_ids,
        "formal_rows": formal_rows,
        "formal_episodes": formal_episodes,
        "eligible_start_count": np.asarray(len(eligible), dtype=np.int64),
    }


def _indexed_h5(dataset: Any, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    unique, inverse = np.unique(rows, return_inverse=True)
    values = np.asarray(dataset[unique])
    return values[inverse].reshape(*rows.shape, *values.shape[1:])


def _segment_batch(
    h5: h5py.File,
    starts: np.ndarray,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel_rows = starts[:, None] + np.arange(common.HORIZON + 1) * common.ACTION_BLOCK
    action_rows = starts[:, None] + np.arange(common.HORIZON * common.ACTION_BLOCK)
    pixels = _indexed_h5(h5["pixels"], pixel_rows).astype(np.uint8, copy=False)
    actions = _indexed_h5(h5["action"], action_rows).astype(np.float32, copy=False)
    xyz = _indexed_h5(h5["privileged_block_0_pos"], pixel_rows).astype(np.float32, copy=False)
    if actions.shape != (len(starts), 25, 5) or pixels.shape[1:] != (6, 224, 224, 3):
        raise RuntimeError(f"segment tensor contract mismatch: {pixels.shape}/{actions.shape}")
    # Match sklearn StandardScaler.transform on float32 exactly: its in-place
    # subtract/divide rounds after each operation.
    normalized = actions.copy()
    normalized -= action_mean
    normalized /= action_scale
    normalized = common.finite("normalized expert actions", normalized).reshape(
        len(starts), common.HORIZON, common.ACTION_BLOCK * common.ACTION_DIM
    )
    return pixels, normalized, xyz


def _rollout(model: Any, initial_pixels: np.ndarray, actions: np.ndarray, device: str) -> Any:
    import torch

    model_dtype = next(model.parameters()).dtype
    normalized = common.normalize_uint8_images(initial_pixels, device).to(model_dtype)
    action_tensor = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(
        device=device, dtype=model_dtype
    )
    info = {"pixels": normalized[:, None, None]}
    with torch.inference_mode():
        predicted = model.rollout(info, action_tensor[:, None])["predicted_emb"][:, 0]
    expected = (len(initial_pixels), common.HORIZON + 1, common.LATENT_DIM)
    if tuple(predicted.shape) != expected:
        raise RuntimeError(f"rollout shape mismatch: expected={expected}, actual={tuple(predicted.shape)}")
    return predicted.detach().float()


def _load_bundle(label: str, device: str) -> tuple[Any, common.LoadedXYZProbe, dict[str, Any]]:
    import stable_worldmodel as swm

    probe = common.LoadedXYZProbe(common.PROBE_PATHS[label], device)
    model = swm.wm.utils.load_pretrained(str(common.CHECKPOINTS[label]), cache_dir=str(common.AILAB_ROOT))
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    actual_sha = route1.torch_module_sha256(model)
    if actual_sha != probe.payload["world_model_state_sha256"]:
        raise ValueError(
            f"{label} probe/model mismatch: probe={probe.payload['world_model_state_sha256']}, "
            f"actual={actual_sha}"
        )
    return model, probe, {
        "checkpoint": common.file_identity(common.CHECKPOINTS[label]),
        "world_model_state_sha256": actual_sha,
        "probe": probe.provenance(),
    }


def _summarize_m1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {}
    metrics = (
        "E_roll_mm",
        "E_enc_mm",
        "Delta_roll_minus_enc_mm",
        "E_imag_mm",
        "latent_l2",
        "latent_cosine_distance",
    )
    for model in common.CHECKPOINTS:
        grouped[model] = {}
        for condition in common.CONDITIONS:
            grouped[model][condition] = {}
            for depth in DEPTHS:
                subset = [row for row in rows if row["model"] == model and row["condition"] == condition and row["depth"] == depth]
                grouped[model][condition][str(depth)] = {
                    "num_segments": len(subset),
                    **{
                        metric: common.distribution([float(row[metric]) for row in subset])
                        for metric in metrics
                    },
                    "roll_gt_40mm_rate": float(np.mean([row["roll_gt_40mm"] for row in subset])) if subset else None,
                    "roll_gt_40_and_enc_le_40_rate": float(
                        np.mean([row["roll_gt_40_and_enc_le_40"] for row in subset])
                    ) if subset else None,
                }
    return grouped


def measure_one(args: argparse.Namespace) -> int:
    common.configure_storage()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("measurement one requires requested CUDA device")
    _validate_probe_links("cpu")
    root = common.stage_root(args.smoke)
    csv_path = root / "measurement1.csv"
    json_path = root / "measurement1.json"
    manifest_path = root / "measurement1_segments.json"
    common.protect_outputs((csv_path, json_path, manifest_path), args.overwrite)
    count = args.smoke_segments if args.smoke else FORMAL_SEGMENTS
    ood_count = min(count, args.smoke_segments) if args.smoke else OOD_SEGMENTS
    action_mean, action_scale, scaler_meta = _load_action_normalizer()
    with h5py.File(common.DATASET, "r", swmr=True) as h5:
        segments = _select_segments(h5, count, ood_count, args.seed)
        manifest = {
            "format_version": "cube_imagination_error_segments_v1",
            "seed": args.seed,
            "num_red_segments": count,
            "num_blue_v2_segments": ood_count,
            "num_yellow_v2_segments": ood_count,
            "start_rows": segments["start_row"],
            "episode_indices": segments["episode_idx"],
            "extension_segment_ids": segments["extension_segment_id"],
            "eligible_start_count": int(segments["eligible_start_count"]),
            "depth_target_rows": {
                str(depth): segments["start_row"] + depth * common.ACTION_BLOCK
                for depth in DEPTHS
            },
            "formal_rows_excluded": segments["formal_rows"],
            "formal_episodes_excluded": segments["formal_episodes"],
            "dataset": common.file_identity(common.DATASET, include_sha256=False),
            "formal_manifest": common.file_identity(common.MANIFEST),
            "action_alignment": "25 consecutive H5 actions t..t+24 reshaped to five 5x5 blocks; targets t+5k",
            "latent_protocol": "single encoded initial frame; real expert action teacher forcing; autoregressive latent feedback only",
            "ood_initial_intervention": "only initial t image recolored; all real endpoint encoder baselines remain clean",
            "action_normalizer": scaler_meta,
        }
        rows: list[dict[str, Any]] = []
        model_meta = {}
        empty_masks = {condition: 0 for condition in common.CONDITIONS}
        for label in common.CHECKPOINTS:
            model, probe, model_meta[label] = _load_bundle(label, args.device)
            for condition in common.CONDITIONS:
                ids = (
                    segments["segment_id"]
                    if condition == "red"
                    else segments["extension_segment_id"]
                )
                for batch_start in range(0, len(ids), args.segment_batch_size):
                    batch_ids = ids[batch_start : batch_start + args.segment_batch_size]
                    starts = segments["start_row"][batch_ids]
                    pixels, actions, xyz = _segment_batch(
                        h5, starts, action_mean, action_scale
                    )
                    initial = pixels[:, 0]
                    if condition != "red":
                        initial, empty = common.recolor_red_pixels(
                            initial, common.TARGET_HUES[condition]
                        )
                        if label == "official":
                            empty_masks[condition] += empty
                    actual_latent = common.encode_uint8(
                        model,
                        pixels[:, 1:].reshape(-1, 224, 224, 3),
                        args.device,
                        args.encoder_batch_size,
                    ).reshape(len(starts), common.HORIZON, common.LATENT_DIM)
                    predicted_latent = _rollout(model, initial, actions, args.device)
                    with torch.inference_mode():
                        roll_xyz = probe(predicted_latent[:, 1:]).detach().cpu().numpy()
                        enc_xyz = probe(actual_latent).detach().cpu().numpy()
                    drift = common.latent_metrics(predicted_latent[:, 1:], actual_latent)
                    true_xyz = xyz[:, 1:]
                    e_roll = common.xyz_error_mm(roll_xyz, true_xyz)
                    e_enc = common.xyz_error_mm(enc_xyz, true_xyz)
                    e_imag = common.xyz_error_mm(roll_xyz, enc_xyz)
                    for local, segment_id in enumerate(batch_ids):
                        for depth_index, depth in enumerate(DEPTHS):
                            truth = true_xyz[local, depth_index]
                            roll = roll_xyz[local, depth_index]
                            enc = enc_xyz[local, depth_index]
                            roll_error = float(e_roll[local, depth_index])
                            enc_error = float(e_enc[local, depth_index])
                            rows.append(
                                {
                                    "model": label,
                                    "condition": condition,
                                    "segment_id": int(segment_id),
                                    "episode_idx": int(segments["episode_idx"][segment_id]),
                                    "start_row": int(starts[local]),
                                    "target_row": int(starts[local] + depth * common.ACTION_BLOCK),
                                    "depth": depth,
                                    "action_teacher_forcing": True,
                                    "latent_teacher_forcing": False,
                                    "initial_color_intervention": "none" if condition == "red" else f"red_mask_to_{condition}",
                                    "endpoint_encoder_domain": "clean_h5",
                                    "true_x_m": float(truth[0]),
                                    "true_y_m": float(truth[1]),
                                    "true_z_m": float(truth[2]),
                                    "roll_x_m": float(roll[0]),
                                    "roll_y_m": float(roll[1]),
                                    "roll_z_m": float(roll[2]),
                                    "enc_x_m": float(enc[0]),
                                    "enc_y_m": float(enc[1]),
                                    "enc_z_m": float(enc[2]),
                                    "E_roll_mm": roll_error,
                                    "E_enc_mm": enc_error,
                                    "Delta_roll_minus_enc_mm": roll_error - enc_error,
                                    "E_imag_mm": float(e_imag[local, depth_index]),
                                    "latent_l2": float(drift["latent_l2"][local, depth_index]),
                                    "latent_cosine_distance": float(drift["latent_cosine_distance"][local, depth_index]),
                                    "roll_gt_40mm": roll_error > 40.0,
                                    "roll_gt_40_and_enc_le_40": roll_error > 40.0 and enc_error <= 40.0,
                                }
                            )
            del model, probe
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    summary = {
        "format_version": "cube_imagination_error_measurement1_v1",
        "protocol": manifest,
        "models": model_meta,
        "empty_initial_recolor_masks": empty_masks,
        "by_model_condition_depth": _summarize_m1(rows),
        "num_csv_rows": len(rows),
        "expected_num_csv_rows": 2 * 5 * (count + 2 * ood_count),
        "primary_supervision_and_error": "block XYZ only",
    }
    if len(rows) != summary["expected_num_csv_rows"]:
        raise RuntimeError("measurement-one row count mismatch")
    common.write_csv(csv_path, rows, M1_FIELDS)
    common.write_json(json_path, summary)
    common.write_json(manifest_path, manifest)
    print(json.dumps({"measurement1_rows": len(rows), "output": str(root)}))
    return 0


def _audit_case(condition: str, env_idx: int) -> Path:
    matches = sorted(common.AUDIT_ROOTS[condition].glob(f"env_{env_idx:02d}_row_*"))
    if len(matches) != 1:
        raise ValueError(f"expected one audit case for {condition}/env{env_idx}, got {matches}")
    return matches[0]


def _load_candidate_labels(case: Path) -> dict[str, np.ndarray]:
    with (case / "candidate_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 300 or [int(row["candidate_idx"]) for row in records] != list(range(300)):
        raise ValueError(f"candidate CSV IDs malformed: {case}")
    return {
        "final_success": np.asarray([_bool(row["final_success"]) for row in records], dtype=bool),
        "ever_success": np.asarray([_bool(row["ever_success"]) for row in records], dtype=bool),
        "min_goal_distance_m": np.asarray([float(row["min_goal_distance_m"]) for row in records]),
        "final_goal_distance_m": np.asarray([float(row["final_goal_distance_m"]) for row in records]),
        "terminal_xyz_csv": np.asarray(
            [[float(row[f"terminal_cube_{axis}"]) for axis in "xyz"] for row in records]
        ),
    }


def _cliffs_delta(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Tie-aware Cliff delta: P(first>second)-P(first<second)."""
    first = np.asarray(first, dtype=np.float64)
    second = np.sort(np.asarray(second, dtype=np.float64))
    if not len(first) or not len(second):
        return None
    less = sum(int(np.searchsorted(second, value, side="left")) for value in first)
    greater = sum(int(len(second) - np.searchsorted(second, value, side="right")) for value in first)
    return float((less - greater) / (len(first) * len(second)))


def _exact_sign_flip(differences: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        return {"num_informative_envs": 0, "mean_failure_minus_success_mm": None, "median_failure_minus_success_mm": None, "p_raw_two_sided": None}
    observed = abs(float(np.mean(values)))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))), dtype=np.float64)
    statistics = np.abs(np.mean(signs * values[None], axis=1))
    p = float(np.mean(statistics >= observed - 1e-12))
    return {
        "num_informative_envs": int(len(values)),
        "mean_failure_minus_success_mm": float(np.mean(values)),
        "median_failure_minus_success_mm": float(np.median(values)),
        "p_raw_two_sided": p,
    }


def _holm(records: list[dict[str, Any]], family_size: int = 6) -> None:
    valid = sorted(
        ((index, row["p_raw_two_sided"]) for index, row in enumerate(records) if row["p_raw_two_sided"] is not None),
        key=lambda item: item[1],
    )
    running = 0.0
    for rank, (index, p) in enumerate(valid):
        adjusted = min(1.0, (family_size - rank) * float(p))
        running = max(running, adjusted)
        records[index]["p_holm_six_comparisons"] = running
    for row in records:
        row.setdefault("p_holm_six_comparisons", None)


def _summarize_m2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "E_roll_mm",
        "E_enc_mm",
        "Delta_roll_minus_enc_mm",
        "E_imag_mm",
        "latent_l2",
        "latent_cosine_distance",
    )
    strata = {
        "all": lambda row: True,
        "final_success": lambda row: row["final_success"],
        "final_failure": lambda row: not row["final_success"],
        "ever_success": lambda row: row["ever_success"],
        "ever_failure": lambda row: not row["ever_success"],
    }
    grouped = {}
    primary_tests = []
    sensitivity_tests = []
    for model in common.CHECKPOINTS:
        grouped[model] = {}
        for condition in common.CONDITIONS:
            population = [row for row in rows if row["model"] == model and row["condition"] == condition]
            grouped[model][condition] = {}
            for name, predicate in strata.items():
                subset = [row for row in population if predicate(row)]
                grouped[model][condition][name] = {
                    "num_candidates": len(subset),
                    **{metric: common.distribution([float(row[metric]) for row in subset]) for metric in metrics},
                    "roll_gt_40mm_rate": float(np.mean([row["roll_gt_40mm"] for row in subset])) if subset else None,
                    "roll_gt_40_and_enc_le_40_rate": float(np.mean([row["roll_gt_40_and_enc_le_40"] for row in subset])) if subset else None,
                }
            for success_key, destination in (("final_success", primary_tests), ("ever_success", sensitivity_tests)):
                differences = []
                for env_idx in common.AUDIT_ENVS:
                    env = [row for row in population if row["env_idx"] == env_idx]
                    success = [row["E_roll_mm"] for row in env if row[success_key]]
                    failure = [row["E_roll_mm"] for row in env if not row[success_key]]
                    if success and failure:
                        differences.append(float(np.median(failure) - np.median(success)))
                test = {
                    "model": model,
                    "condition": condition,
                    "success_definition": success_key,
                    **_exact_sign_flip(differences),
                    "cliffs_delta_success_vs_failure": _cliffs_delta(
                        [row["E_roll_mm"] for row in population if row[success_key]],
                        [row["E_roll_mm"] for row in population if not row[success_key]],
                    ),
                    "cliffs_delta_direction": "negative means successful candidates have lower E_roll",
                }
                destination.append(test)
    _holm(primary_tests, family_size=6)
    return {
        "by_model_condition_stratum": grouped,
        "env_equal_sign_flip_primary_final_success": primary_tests,
        "env_equal_sign_flip_sensitivity_ever_success_unadjusted": sensitivity_tests,
        "multiple_comparison": "Holm correction over exactly 2 checkpoints x 3 colors for final_success primary tests",
    }


def measure_two(args: argparse.Namespace) -> int:
    common.configure_storage()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("measurement two requires requested CUDA device")
    _validate_probe_links("cpu")
    root = common.stage_root(args.smoke)
    csv_path = root / "measurement2.csv"
    json_path = root / "measurement2.json"
    common.protect_outputs((csv_path, json_path), args.overwrite)
    envs = common.AUDIT_ENVS[:1] if args.smoke else common.AUDIT_ENVS
    formal_manifest = json.loads(common.MANIFEST.read_text(encoding="utf-8"))
    formal_rows = np.asarray(formal_manifest["formal_rows"], dtype=np.int64)
    if formal_rows.shape != (50,):
        raise ValueError("formal manifest must contain exactly 50 rows")
    rows: list[dict[str, Any]] = []
    model_meta = {}
    audit_meta = {}
    for label in common.CHECKPOINTS:
        model, probe, model_meta[label] = _load_bundle(label, args.device)
        for condition in common.CONDITIONS:
            audit_meta.setdefault(condition, {})
            for env_idx in envs:
                case = _audit_case(condition, env_idx)
                with np.load(case / "population.npz", allow_pickle=False) as population:
                    candidates = np.asarray(population["candidates_normalized"], dtype=np.float32)
                    initial = np.asarray(population["initial_pixels"], dtype=np.uint8)
                    latent_costs = np.asarray(population["latent_costs"], dtype=np.float64)
                with np.load(case / "physical_outcomes.npz", allow_pickle=False) as physical:
                    terminal_images = np.asarray(physical["terminal_images"], dtype=np.uint8)
                    terminal_xyz = np.asarray(physical["terminal_cube_position"], dtype=np.float64)
                    physical_ever_success = np.asarray(physical["ever_success"], dtype=bool)
                labels = _load_candidate_labels(case)
                if candidates.shape != (300, 5, 25) or terminal_images.shape != (300, 224, 224, 3) or terminal_xyz.shape != (300, 3):
                    raise ValueError(f"audit array shape mismatch: {case}")
                if not np.array_equal(labels["ever_success"], physical_ever_success):
                    raise ValueError(f"ever-success cache/CSV mismatch: {case}")
                if not np.allclose(labels["terminal_xyz_csv"], terminal_xyz, atol=1e-10, rtol=0):
                    raise ValueError(f"terminal XYZ cache/CSV mismatch: {case}")
                predicted_latent = route1.exact_candidate_terminal_embeddings(
                    model,
                    initial,
                    candidates,
                    args.device,
                    batch_size=args.rollout_batch_size,
                )
                actual_latent = common.encode_uint8(
                    model, terminal_images, args.device, args.encoder_batch_size
                )
                with torch.inference_mode():
                    roll_xyz = probe(predicted_latent).detach().cpu().numpy()
                    enc_xyz = probe(actual_latent).detach().cpu().numpy()
                drift = common.latent_metrics(predicted_latent, actual_latent)
                e_roll = common.xyz_error_mm(roll_xyz, terminal_xyz)
                e_enc = common.xyz_error_mm(enc_xyz, terminal_xyz)
                e_imag = common.xyz_error_mm(roll_xyz, enc_xyz)
                capture = json.loads((case / "capture_meta.json").read_text(encoding="utf-8"))
                dataset_row = int(capture["dataset_row"])
                episode_idx = int(capture["episode_idx"])
                if int(capture["env_idx"]) != env_idx or dataset_row != int(formal_rows[env_idx]):
                    raise ValueError(
                        f"audit case/formal manifest mapping mismatch: {case}; "
                        f"capture env/row={capture['env_idx']}/{dataset_row}, "
                        f"expected={env_idx}/{formal_rows[env_idx]}"
                    )
                if env_idx not in audit_meta[condition]:
                    audit_meta[condition][env_idx] = {
                        "case": str(case.resolve()),
                        "population": common.file_identity(case / "population.npz"),
                        "physical_outcomes": common.file_identity(case / "physical_outcomes.npz"),
                        "candidate_labels": common.file_identity(case / "candidate_outcomes.csv"),
                    }
                for candidate_idx in range(300):
                    truth = terminal_xyz[candidate_idx]
                    roll = roll_xyz[candidate_idx]
                    enc = enc_xyz[candidate_idx]
                    roll_error = float(e_roll[candidate_idx])
                    enc_error = float(e_enc[candidate_idx])
                    rows.append(
                        {
                            "model": label,
                            "condition": condition,
                            "env_idx": env_idx,
                            "dataset_row": dataset_row,
                            "episode_idx": episode_idx,
                            "candidate_idx": candidate_idx,
                            "rollout_depth": 5,
                            "frozen_action_pool": "unseeded_final300",
                            "true_terminal_x_m": float(truth[0]),
                            "true_terminal_y_m": float(truth[1]),
                            "true_terminal_z_m": float(truth[2]),
                            "roll_x_m": float(roll[0]),
                            "roll_y_m": float(roll[1]),
                            "roll_z_m": float(roll[2]),
                            "enc_x_m": float(enc[0]),
                            "enc_y_m": float(enc[1]),
                            "enc_z_m": float(enc[2]),
                            "E_roll_mm": roll_error,
                            "E_enc_mm": enc_error,
                            "Delta_roll_minus_enc_mm": roll_error - enc_error,
                            "E_imag_mm": float(e_imag[candidate_idx]),
                            "latent_l2": float(drift["latent_l2"][candidate_idx]),
                            "latent_cosine_distance": float(drift["latent_cosine_distance"][candidate_idx]),
                            "roll_gt_40mm": roll_error > 40.0,
                            "roll_gt_40_and_enc_le_40": roll_error > 40.0 and enc_error <= 40.0,
                            "final_success": bool(labels["final_success"][candidate_idx]),
                            "ever_success": bool(labels["ever_success"][candidate_idx]),
                            "min_goal_distance_m": float(labels["min_goal_distance_m"][candidate_idx]),
                            "final_goal_distance_m": float(labels["final_goal_distance_m"][candidate_idx]),
                            "stored_latent_cost": float(latent_costs[candidate_idx]),
                        }
                    )
        del model, probe
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    expected = len(common.CHECKPOINTS) * len(common.CONDITIONS) * len(envs) * 300
    if len(rows) != expected:
        raise RuntimeError(f"measurement-two row mismatch: expected={expected}, actual={len(rows)}")
    summary = {
        "format_version": "cube_imagination_error_measurement2_v1",
        "protocol": {
            "conditions": list(common.CONDITIONS),
            "audit_envs": list(envs),
            "candidates_per_case": 300,
            "pool": "old unseeded cube_cem_300{,_blue_v2,_yellow_v2}",
            "formal_manifest": common.file_identity(common.MANIFEST),
            "both_models_use_identical_frozen_actions_and_physical_labels": True,
            "terminal_embedding": "same-checkpoint encoder(physical_outcomes terminal_images)",
            "terminal_supervision": "cached terminal_cube_position XYZ only",
            "primary_success_stratification": "final_success",
            "sensitivity_success_stratification": "ever_success",
            "stored_latent_cost": "copied from each frozen audit; not recomputed for the masked checkpoint",
        },
        "models": model_meta,
        "audit_inputs": audit_meta,
        **_summarize_m2(rows),
        "num_csv_rows": len(rows),
    }
    common.write_csv(csv_path, rows, M2_FIELDS)
    common.write_json(json_path, summary)
    print(json.dumps({"measurement2_rows": len(rows), "output": str(root)}))
    return 0


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _triplet(distribution: Mapping[str, Any], digits: int = 3) -> str:
    return "/".join(
        _fmt(distribution.get(name), digits) for name in ("median", "p90", "p95")
    )


def _depth5_extension_table(csv_path: Path) -> tuple[list[list[Any]], dict[str, Any]]:
    """Build the paired 500-segment color matrix without altering raw outputs."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if int(row["depth"]) == 5]
    table_rows: list[list[Any]] = []
    details: dict[str, Any] = {}
    for model in common.CHECKPOINTS:
        by_condition = {
            condition: {
                int(row["segment_id"]): float(row["E_roll_mm"])
                for row in records
                if row["model"] == model and row["condition"] == condition
            }
            for condition in common.CONDITIONS
        }
        paired_ids = set(by_condition["blue_v2"])
        if paired_ids != set(by_condition["yellow_v2"]):
            raise ValueError(f"depth5 blue/yellow segment IDs are not paired: model={model}")
        if not paired_ids or not paired_ids.issubset(by_condition["red"]):
            raise ValueError(f"depth5 clean reference is incomplete: model={model}")
        ordered_ids = sorted(paired_ids)
        clean = np.asarray(
            [by_condition["red"][segment_id] for segment_id in ordered_ids],
            dtype=np.float64,
        )
        details[model] = {}
        for display, condition in (
            ("clean", "red"),
            ("blue", "blue_v2"),
            ("yellow", "yellow_v2"),
        ):
            values = np.asarray(
                [by_condition[condition][segment_id] for segment_id in ordered_ids],
                dtype=np.float64,
            )
            distribution = common.distribution(values)
            paired_difference = values - clean
            clean_median = float(np.median(clean))
            ratio = (
                float(distribution["median"]) / clean_median
                if clean_median > 0
                else None
            )
            item = {
                "n": len(values),
                "E_roll_mm": distribution,
                "roll_gt_40mm_rate": float(np.mean(values > 40.0)),
                "paired_clean_difference_median_mm": float(np.median(paired_difference)),
                "ratio_of_medians_vs_paired_clean": ratio,
            }
            details[model][display] = item
            table_rows.append(
                [
                    model,
                    display,
                    item["n"],
                    _triplet(distribution),
                    _fmt(item["roll_gt_40mm_rate"], 4),
                    _fmt(item["paired_clean_difference_median_mm"]),
                    _fmt(ratio, 3),
                ]
            )
    return table_rows, details


def report(args: argparse.Namespace) -> int:
    root = common.stage_root(args.smoke)
    m1_path = root / "measurement1.json"
    m2_path = root / "measurement2.json"
    if not m1_path.is_file() or not m2_path.is_file():
        raise FileNotFoundError(f"measurement outputs incomplete: {m1_path}, {m2_path}")
    output_json = root / "summary.json"
    output_csv = root / "summary.csv"
    output_md = root / "REPORT.md"
    common.protect_outputs((output_json, output_csv, output_md), args.overwrite)
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    m2 = json.loads(m2_path.read_text(encoding="utf-8"))
    extension_table, _ = _depth5_extension_table(root / "measurement1.csv")
    probe_table = []
    for model in common.CHECKPOINTS:
        probe = m1["models"][model]["probe"]
        metrics = probe["test_metrics"]
        median = float(metrics["xyz_error_mm"]["median"])
        probe_table.append(
            [
                model,
                metrics["num_samples"],
                _fmt(median),
                _fmt(metrics["r2"]["block_x"], 6),
                _fmt(metrics["r2"]["block_y"], 6),
                _fmt(metrics["r2"]["block_z"], 6),
                "PASS" if median < common.PROBE_TEST_MEDIAN_LIMIT_MM else "FAIL",
                probe["checkpoint"]["sha256"],
            ]
        )
    summary_rows = []
    m1_table = []
    for model in common.CHECKPOINTS:
        for condition in common.CONDITIONS:
            for depth in DEPTHS:
                group = m1["by_model_condition_depth"][model][condition][str(depth)]
                row = {
                    "stage": "measurement1",
                    "model": model,
                    "condition": condition,
                    "depth_or_stratum": depth,
                    "count": group["num_segments"],
                    "E_roll_median_mm": group["E_roll_mm"]["median"],
                    "E_roll_p90_mm": group["E_roll_mm"]["p90"],
                    "E_roll_p95_mm": group["E_roll_mm"]["p95"],
                    "E_enc_median_mm": group["E_enc_mm"]["median"],
                    "E_enc_p90_mm": group["E_enc_mm"]["p90"],
                    "E_enc_p95_mm": group["E_enc_mm"]["p95"],
                    "Delta_median_mm": group["Delta_roll_minus_enc_mm"]["median"],
                    "Delta_p90_mm": group["Delta_roll_minus_enc_mm"]["p90"],
                    "Delta_p95_mm": group["Delta_roll_minus_enc_mm"]["p95"],
                    "E_imag_median_mm": group["E_imag_mm"]["median"],
                    "E_imag_p90_mm": group["E_imag_mm"]["p90"],
                    "E_imag_p95_mm": group["E_imag_mm"]["p95"],
                    "latent_l2_median": group["latent_l2"]["median"],
                    "latent_l2_p90": group["latent_l2"]["p90"],
                    "latent_l2_p95": group["latent_l2"]["p95"],
                    "latent_cosine_median": group["latent_cosine_distance"]["median"],
                    "latent_cosine_p90": group["latent_cosine_distance"]["p90"],
                    "latent_cosine_p95": group["latent_cosine_distance"]["p95"],
                    "roll_gt_40_rate": group["roll_gt_40mm_rate"],
                    "roll_gt_40_enc_le_40_rate": group["roll_gt_40_and_enc_le_40_rate"],
                    "p_raw": None,
                    "p_holm": None,
                }
                summary_rows.append(row)
                m1_table.append(
                    [model, condition, depth, group["num_segments"], _triplet(group["E_roll_mm"]), _triplet(group["E_enc_mm"]), _triplet(group["Delta_roll_minus_enc_mm"]), _triplet(group["E_imag_mm"]), _triplet(group["latent_l2"]), _triplet(group["latent_cosine_distance"], 4), _fmt(row["roll_gt_40_rate"], 4)]
                )
    m2_table = []
    tests = {(row["model"], row["condition"]): row for row in m2["env_equal_sign_flip_primary_final_success"]}
    for model in common.CHECKPOINTS:
        for condition in common.CONDITIONS:
            for stratum in ("all", "final_success", "final_failure", "ever_success", "ever_failure"):
                group = m2["by_model_condition_stratum"][model][condition][stratum]
                test = tests[(model, condition)] if stratum == "all" else {}
                row = {
                    "stage": "measurement2",
                    "model": model,
                    "condition": condition,
                    "depth_or_stratum": stratum,
                    "count": group["num_candidates"],
                    "E_roll_median_mm": group["E_roll_mm"]["median"],
                    "E_roll_p90_mm": group["E_roll_mm"]["p90"],
                    "E_roll_p95_mm": group["E_roll_mm"]["p95"],
                    "E_enc_median_mm": group["E_enc_mm"]["median"],
                    "E_enc_p90_mm": group["E_enc_mm"]["p90"],
                    "E_enc_p95_mm": group["E_enc_mm"]["p95"],
                    "Delta_median_mm": group["Delta_roll_minus_enc_mm"]["median"],
                    "Delta_p90_mm": group["Delta_roll_minus_enc_mm"]["p90"],
                    "Delta_p95_mm": group["Delta_roll_minus_enc_mm"]["p95"],
                    "E_imag_median_mm": group["E_imag_mm"]["median"],
                    "E_imag_p90_mm": group["E_imag_mm"]["p90"],
                    "E_imag_p95_mm": group["E_imag_mm"]["p95"],
                    "latent_l2_median": group["latent_l2"]["median"],
                    "latent_l2_p90": group["latent_l2"]["p90"],
                    "latent_l2_p95": group["latent_l2"]["p95"],
                    "latent_cosine_median": group["latent_cosine_distance"]["median"],
                    "latent_cosine_p90": group["latent_cosine_distance"]["p90"],
                    "latent_cosine_p95": group["latent_cosine_distance"]["p95"],
                    "roll_gt_40_rate": group["roll_gt_40mm_rate"],
                    "roll_gt_40_enc_le_40_rate": group["roll_gt_40_and_enc_le_40_rate"],
                    "p_raw": test.get("p_raw_two_sided"),
                    "p_holm": test.get("p_holm_six_comparisons"),
                }
                summary_rows.append(row)
                if stratum in {"all", "final_success", "final_failure"}:
                    m2_table.append(
                        [model, condition, stratum, group["num_candidates"], _triplet(group["E_roll_mm"]), _triplet(group["E_enc_mm"]), _triplet(group["Delta_roll_minus_enc_mm"]), _triplet(group["E_imag_mm"]), _triplet(group["latent_l2"]), _triplet(group["latent_cosine_distance"], 4), _fmt(row["roll_gt_40_rate"], 4)]
                    )
    background = None
    route1_metrics = common.AILAB_ROOT / "models/probes/cube_block4d_v1/metrics.json"
    if route1_metrics.is_file():
        old = json.loads(route1_metrics.read_text(encoding="utf-8"))
        background = old["models"]["mlp"]["metrics"]["test"]["median_xyz_error_mm"]
    final = {
        "format_version": "cube_imagination_error_report_v1",
        "primary_target_and_error": "block XYZ only; yaw unused",
        "checkpoints": {label: common.file_identity(path) for label, path in common.CHECKPOINTS.items()},
        "probes": {label: common.file_identity(path) for label, path in common.PROBE_PATHS.items()},
        "measurement1": m1,
        "measurement2": m2,
        "background_not_current_probe": {
            "route1_4d_mlp_test_median_xyz_error_mm": background,
            "warning": "background only; this experiment trains fresh checkpoint-specific XYZ-only probes",
        },
        "interpretation_limits": [
            "Measurement1 blue/yellow recolors only the initial frame; endpoint encoder floors stay clean.",
            "Measurement2 is counterfactual model inference on identical frozen actions and cached physics; it does not rerun the simulator.",
            "Candidate-level rows are clustered within env; primary inference uses env-equal sign flips with six-way Holm correction.",
        ],
    }
    common.write_json(output_json, final)
    fields = tuple(summary_rows[0])
    common.write_csv(output_csv, summary_rows, fields)
    test_rows = [
        [row["model"], row["condition"], row["num_informative_envs"], _fmt(row["mean_failure_minus_success_mm"]), _fmt(row["cliffs_delta_success_vs_failure"], 4), _fmt(row["p_raw_two_sided"], 6), _fmt(row["p_holm_six_comparisons"], 6)]
        for row in m2["env_equal_sign_flip_primary_final_success"]
    ]
    clean_depth5 = {
        model: m1["by_model_condition_depth"][model]["red"]["5"]["E_roll_mm"][
            "median"
        ]
        for model in common.CHECKPOINTS
    }
    candidate_medians = [
        m2["by_model_condition_stratum"][model][condition]["all"]["E_roll_mm"][
            "median"
        ]
        for model in common.CHECKPOINTS
        for condition in common.CONDITIONS
    ]
    official_ood_floor = [
        m2["by_model_condition_stratum"]["official"][condition]["all"]["E_enc_mm"][
            "median"
        ]
        for condition in ("blue_v2", "yellow_v2")
    ]
    masked_ood_floor = [
        m2["by_model_condition_stratum"]["masked"][condition]["all"]["E_enc_mm"][
            "median"
        ]
        for condition in ("blue_v2", "yellow_v2")
    ]
    significant = [
        f"{row['model']}/{row['condition']}"
        for row in m2["env_equal_sign_flip_primary_final_success"]
        if row["p_holm_six_comparisons"] is not None
        and float(row["p_holm_six_comparisons"]) < 0.05
    ]
    verdict = (
        f"On clean expert-action segments, depth-5 median E_roll is "
        f"{_fmt(clean_depth5['official'])} mm (official) and "
        f"{_fmt(clean_depth5['masked'])} mm (masked), whereas the frozen CEM "
        f"candidate pools span {_fmt(min(candidate_medians))}–"
        f"{_fmt(max(candidate_medians))} mm across model/color cells. These are "
        "different trajectory populations, so the candidate-pool result does not "
        "by itself establish that the predictor is universally inaccurate. "
        f"For blue/yellow candidate terminal frames, the official encoder/probe "
        f"floor is {_fmt(min(official_ood_floor))}–{_fmt(max(official_ood_floor))} "
        f"mm, versus {_fmt(min(masked_ood_floor))}–{_fmt(max(masked_ood_floor))} "
        "mm for the masked model; official OOD rollout error therefore cannot be "
        "attributed solely to autoregressive prediction. In the env-level "
        "final-success sign-flip tests with six-way Holm correction, significance "
        f"is limited to {', '.join(significant) if significant else 'no cell'}; "
        "blue/yellow are not significant and should not be described as such."
    )
    common.write_text(
        output_md,
        "# Cube imagination-error report\n\n"
        "Primary supervision and all formal physical errors are strict block XYZ; yaw is unused.\n\n"
        "## Fresh XYZ-only probe quality\n\n"
        + common.markdown_table(
            ["Model", "Test N", "Median XYZ mm", "R² x", "R² y", "R² z", "<15 mm", "Probe SHA-256"],
            probe_table,
        )
        + "\n\nBoth fresh checkpoint-specific probes pass the frozen strict `<15 mm` test-median requirement.\n\n"
        "## Measurement 1 — expert-action autoregressive depth\n\n"
        + common.markdown_table(
            ["Model", "Condition", "Depth", "N", "E_roll mm med/p90/p95", "E_enc mm med/p90/p95", "Delta mm med/p90/p95", "E_imag mm med/p90/p95", "Latent L2 med/p90/p95", "Latent cosine med/p90/p95", "P(E_roll>40)"],
            m1_table,
        )
        + "\n\nOnly real expert **actions** are teacher-forced. Latents are autoregressive from one initial encoded frame. For blue/yellow only that initial frame is recolored; encoder endpoints are clean H5 frames.\n\n"
        "## Depth-5 paired color extension\n\n"
        + common.markdown_table(
            ["Model", "Initial color", "Paired N", "E_roll mm med/p90/p95", "P(E_roll>40)", "Median paired Δ vs clean mm", "Ratio of medians vs clean"],
            extension_table,
        )
        + "\n\nBlue and yellow use the same frozen segment IDs as their clean reference; the delta is the median of per-segment `OOD − clean` differences, while the ratio is the ratio of marginal medians.\n\n"
        "## Measurement 2 — frozen unseeded 12x300 pools\n\n"
        + common.markdown_table(
            ["Model", "Condition", "Stratum", "N", "E_roll mm med/p90/p95", "E_enc mm med/p90/p95", "Delta mm med/p90/p95", "E_imag mm med/p90/p95", "Latent L2 med/p90/p95", "Latent cosine med/p90/p95", "P(E_roll>40)"],
            m2_table,
        )
        + "\n\n## Env-equal primary inference (final success)\n\n"
        + common.markdown_table(
            ["Model", "Condition", "Informative envs", "Failure-success med diff mean mm", "Cliff delta", "p raw", "p Holm (6)"],
            test_rows,
        )
        + "\n\n## Decision\n\n"
        + verdict
        + f"\n\nThe older Route1 4D probe's {_fmt(background)} mm test median is background only and is not reused as either current probe.\n",
    )
    print(root)
    return 0


def plan() -> int:
    payload = {
        "format_version": "cube_imagination_error_plan_v1",
        "output_root": str(common.OUTPUT_ROOT),
        "checkpoints": {label: str(path) for label, path in common.CHECKPOINTS.items()},
        "embedding_datasets": {label: str(path) for label, path in common.EMBEDDING_DATASETS.items()},
        "fresh_xyz_probes": {label: str(path) for label, path in common.PROBE_PATHS.items()},
        "measurement1": {
            "red_segments": FORMAL_SEGMENTS,
            "blue_v2_initial_only_segments": OOD_SEGMENTS,
            "yellow_v2_initial_only_segments": OOD_SEGMENTS,
            "depths": list(DEPTHS),
            "actions": "25 H5 actions -> five 25D blocks; targets t+5k",
            "latent_teacher_forcing": False,
        },
        "measurement2": {
            "conditions": list(common.CONDITIONS),
            "envs": list(common.AUDIT_ENVS),
            "candidates_per_case": 300,
            "pool": "old unseeded audits",
            "checkpoint_pairing": "identical frozen actions and physical labels",
        },
        "commands": [
            f"{sys.executable} {Path(__file__).name} prepare-probes",
            f"{sys.executable} {Path(__file__).name} measure-one",
            f"{sys.executable} {Path(__file__).name} measure-two",
            f"{sys.executable} {Path(__file__).name} report",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    prepare = sub.add_parser("prepare-probes")
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--encoder-batch-size", type=int, default=256)
    prepare.add_argument("--seed", type=int, default=SEED)
    prepare.add_argument("--hidden-dim", type=int, default=256)
    prepare.add_argument("--probe-epochs", type=int, default=50)
    prepare.add_argument("--probe-patience", type=int, default=8)
    prepare.add_argument("--probe-batch-size", type=int, default=2048)
    prepare.add_argument("--probe-eval-batch-size", type=int, default=8192)
    prepare.add_argument("--probe-learning-rate", type=float, default=1e-3)
    prepare.add_argument("--probe-weight-decay", type=float, default=1e-4)
    prepare.add_argument("--num-workers", type=int, default=4)
    prepare.add_argument("--overwrite-embeddings", action="store_true")
    prepare.add_argument("--overwrite-probes", action="store_true")
    measurement_one = sub.add_parser("measure-one")
    measurement_one.add_argument("--device", default="cuda")
    measurement_one.add_argument("--encoder-batch-size", type=int, default=128)
    measurement_one.add_argument("--segment-batch-size", type=int, default=64)
    measurement_one.add_argument("--seed", type=int, default=SEED)
    measurement_one.add_argument("--smoke", action="store_true")
    measurement_one.add_argument("--smoke-segments", type=int, default=2)
    measurement_one.add_argument("--overwrite", action="store_true")
    measurement_two = sub.add_parser("measure-two")
    measurement_two.add_argument("--device", default="cuda")
    measurement_two.add_argument("--encoder-batch-size", type=int, default=128)
    measurement_two.add_argument("--rollout-batch-size", type=int, default=300)
    measurement_two.add_argument("--smoke", action="store_true")
    measurement_two.add_argument("--overwrite", action="store_true")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--smoke", action="store_true")
    report_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    numeric = [
        value
        for key, value in vars(args).items()
        if key.endswith("batch_size") or key in {"probe_epochs", "probe_patience", "smoke_segments"}
    ]
    if any(value <= 0 for value in numeric):
        raise ValueError("batch sizes, epochs, patience, and smoke segments must be positive")
    if args.command == "plan":
        return plan()
    if args.command == "prepare-probes":
        return prepare_probes(args)
    if args.command == "measure-one":
        return measure_one(args)
    if args.command == "measure-two":
        return measure_two(args)
    if args.command == "report":
        return report(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
