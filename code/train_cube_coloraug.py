#!/usr/bin/env python3
"""Route-2 warm-start fine-tuning with online Cube hue rotation.

This standalone entry point intentionally does not modify ``train.py`` or the
official pretrained checkpoint. It excludes every episode represented by the
frozen 50-row evaluation protocol before splitting train/validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

import hdf5plugin  # noqa: F401  # Register compressed-HDF5 filters first.
import h5py
import numpy as np
import torch

from cube_coloraug import (
    IndexedTransformDataset,
    RandomHueRotation,
    allowed_clip_indices,
    split_indices,
    streaming_mean_std,
)


AILAB = Path(__file__).resolve().parent.parent
DATASET = AILAB / "datasets/ogbench/cube_single_expert.h5"
MANIFEST = AILAB / "outputs/audit/cube_cem_manifest.json"
WARM_WEIGHTS = AILAB / "checkpoints/models--quentinll--lewm-cube/weights.pt"
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-coloraug"
OUTPUT_ROOT = AILAB / "outputs/train/route2_coloraug"
TENSORBOARD_ROOT = AILAB / "logs/tensorboard/route2_coloraug"
TMP_ROOT = AILAB.parent / "tmp"
NUM_STEPS = 4
FRAMESKIP = 5
HISTORY_SIZE = 3
NUM_PREDS = 1


def _configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(AILAB),
        "LOCAL_DATASET_DIR": str(AILAB),
        "HF_HOME": str(AILAB.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
    }
    for key, value in values.items():
        os.environ.setdefault(key, value)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) or run_id in {".", ".."}:
        raise ValueError("--run-id must be one safe path component")
    return run_id


def _validate_data_disk(path: Path, label: str, must_exist: bool = True) -> Path:
    path = path.expanduser().resolve()
    data_disk = AILAB.parent.resolve()
    if path != data_disk and data_disk not in path.parents:
        raise ValueError(f"{label} must be on /root/autodl-tmp: {path}")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    return path


def _heldout_protocol(dataset_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = np.asarray(manifest.get("formal_rows"), dtype=np.int64)
    if rows.shape != (50,) or len(np.unique(rows)) != 50:
        raise RuntimeError("frozen manifest must contain 50 unique formal rows")
    stat = dataset_path.stat()
    identity = manifest.get("dataset", {})
    if identity.get("size_bytes") != stat.st_size or identity.get("mtime_ns") != stat.st_mtime_ns:
        raise RuntimeError("dataset identity differs from frozen evaluation manifest")
    with h5py.File(dataset_path, "r") as h5:
        episode_key = "episode_idx" if "episode_idx" in h5 else "ep_idx"
        episodes = np.asarray(h5[episode_key][rows], dtype=np.int64)
        lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    if len(np.unique(episodes)) != 50:
        raise RuntimeError("formal rows do not map to 50 unique episodes")
    if np.any(rows < offsets[episodes]) or np.any(rows >= offsets[episodes] + lengths[episodes]):
        raise RuntimeError("formal row-to-episode mapping is inconsistent")
    span = NUM_STEPS * FRAMESKIP
    clips_by_episode = np.maximum(lengths - span + 1, 0)
    return {
        "formal_rows": rows,
        "episodes": episodes,
        "lengths": lengths,
        "offsets": offsets,
        "total_clips": int(clips_by_episode.sum()),
        "excluded_clips": int(clips_by_episode[episodes].sum()),
        "manifest_sha256": _sha(manifest_path),
        "dataset_size_bytes": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns,
    }


def _row_inclusion(lengths: np.ndarray, offsets: np.ndarray, excluded: set[int]) -> np.ndarray:
    total_rows = int(np.max(offsets + lengths))
    included = np.ones(total_rows, dtype=bool)
    for episode in excluded:
        start = int(offsets[episode])
        included[start : start + int(lengths[episode])] = False
    return included


class ColumnNormalizer:
    """Picklable per-column z-score transform for DataLoader workers."""

    def __init__(self, source: str, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.source = source
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample[self.source] = ((sample[self.source] - self.mean) / self.std).float()
        return sample


def route2_forward(self: Any, batch: dict[str, torch.Tensor], stage: str, cfg: Any) -> dict[str, torch.Tensor]:
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    embedding = output["emb"]
    action_embedding = output["act_emb"]
    prediction = self.model.predict(
        embedding[:, : cfg.history_size],
        action_embedding[:, : cfg.history_size],
    )
    target = embedding[:, cfg.num_preds :]
    output["pred_loss"] = (prediction - target).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(embedding.transpose(0, 1))
    output["loss"] = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
    self.log_dict(
        {f"{stage}/{key}": value.detach() for key, value in output.items() if "loss" in key},
        on_step=True,
        sync_dist=True,
    )
    return output


def configure_manual_gradient_clipping(
    module: Any,
    optimizer_name: str,
    clip_val: float,
    algorithm: str = "norm",
) -> None:
    """Configure spt.Module's explicit manual-optimization clip path.

    ``spt.Module.training_step`` calls ``manual_backward`` first, then (after
    any accumulation averaging) calls ``module.clip_gradients`` with these
    per-optimizer values immediately before ``LightningOptimizer.step``. The
    Route2 protocol is locked to BF16 mixed precision, which does not use a
    GradScaler: clipping therefore occurs after backward (and accumulation
    averaging) and before the optimizer step. Trainer-level automatic clipping
    must remain unset for manual optimization.
    """
    if clip_val <= 0:
        raise ValueError("manual gradient clip value must be positive")
    if algorithm not in {"norm", "value"}:
        raise ValueError(f"unsupported gradient clip algorithm: {algorithm}")
    if not hasattr(module, "_optimizer_gradient_clip_val") or not hasattr(
        module, "_optimizer_gradient_clip_algorithm"
    ):
        raise TypeError("module does not expose stable_pretraining manual clip maps")
    module._optimizer_gradient_clip_val[optimizer_name] = float(clip_val)
    module._optimizer_gradient_clip_algorithm[optimizer_name] = algorithm


def _plan(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    allowed = protocol["total_clips"] - protocol["excluded_clips"]
    train = math.floor(allowed * args.train_split)
    val = allowed - train
    steps_per_epoch = train // args.batch_size
    total_steps = args.max_steps if args.max_steps is not None else steps_per_epoch * args.max_epochs
    warmup_steps = max(1, int(0.01 * total_steps))
    return {
        "route": "route2_online_hsv_hue_rotation",
        "run_id": args.run_id,
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "warm_start": str(args.warm_start),
        "warm_start_sha256": _sha(args.warm_start),
        "formal_episode_exclusion": {
            "episode_count": len(protocol["episodes"]),
            "episode_ids": [int(value) for value in protocol["episodes"]],
            "excluded_clips": protocol["excluded_clips"],
            "manifest_sha256": protocol["manifest_sha256"],
        },
        "clips": {"before_exclusion": protocol["total_clips"], "allowed": allowed, "train": train, "validation": val},
        "augmentation": {
            "space": "HSV",
            "scope": "whole RGB frame; one shared hue delta across all frames in a sequence",
            "hue_delta_turns": [-args.max_hue_delta, args.max_hue_delta],
            "probability": args.hue_probability,
            "saturation_changed": False,
            "value_changed": False,
            "validation_augmented": False,
        },
        "schedule": {
            "type": "LinearWarmupCosineAnnealingLR",
            "interval": "step",
            "estimated_total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "warmup_start_lr": 0.0,
            "eta_min": 0.0,
            "max_epochs": args.max_epochs if args.max_steps is None else None,
            "max_steps": args.max_steps,
            "two_step_smoke_has_nonzero_second_update": total_steps != 2 or warmup_steps == 1,
        },
        "training": {
            "seed": args.seed,
            "train_split": args.train_split,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "precision": args.precision,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "LinearWarmupCosineAnnealingLR adapted from trainer.estimated_stepping_batches",
            "sigreg_weight": args.sigreg_weight,
            "limit_val_batches": args.limit_val_batches,
            "full_model_finetune": True,
            "gradient_clipping": {
                "value": args.gradient_clip_val,
                "algorithm": "norm",
                "owner": "stable_pretraining manual-optimization training_step",
                "position": "after manual_backward and before optimizer.step (BF16 has no GradScaler)",
                "trainer_automatic_clipping": False,
            },
        },
        "paths": {
            "checkpoint": str((CHECKPOINT_ROOT / args.run_id).resolve()),
            "output": str((OUTPUT_ROOT / args.run_id).resolve()),
            "tensorboard": str((TENSORBOARD_ROOT / args.run_id).resolve()),
        },
    }


class SaveRoute2Weights:
    """Factory namespace to defer Lightning/stable-worldmodel imports."""

    @staticmethod
    def create(run_id: str, model_config: Any):
        from lightning.pytorch.callbacks import Callback
        from stable_worldmodel.wm.utils import save_pretrained

        class CallbackImpl(Callback):
            def _save(self, trainer: Any, module: Any, filename: str) -> None:
                if trainer.is_global_zero:
                    save_pretrained(
                        module.model,
                        run_name=f"lewm-cube-coloraug/{run_id}",
                        config=model_config,
                        filename=filename,
                        cache_dir=str(AILAB),
                    )

            def on_train_epoch_end(self, trainer: Any, module: Any) -> None:
                self._save(trainer, module, f"weights_epoch_{trainer.current_epoch + 1}.pt")

            def on_train_end(self, trainer: Any, module: Any) -> None:
                self._save(trainer, module, "weights_final.pt")

        return CallbackImpl()


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    args.run_id = _safe_run_id(args.run_id)
    args.dataset = _validate_data_disk(args.dataset, "dataset")
    args.manifest = _validate_data_disk(args.manifest, "manifest")
    args.warm_start = _validate_data_disk(args.warm_start, "warm-start checkpoint")
    config_path = args.warm_start.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"warm-start config missing: {config_path}")
    protocol = _heldout_protocol(args.dataset, args.manifest)
    plan = _plan(args, protocol)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    output_dir = OUTPUT_ROOT / args.run_id
    checkpoint_dir = CHECKPOINT_ROOT / args.run_id
    tensorboard_dir = TENSORBOARD_ROOT / args.run_id
    existing = [path for path in (output_dir, checkpoint_dir, tensorboard_dir) if path.exists() and any(path.iterdir())]
    if existing and not args.resume:
        raise FileExistsError(f"run outputs already exist; choose a new --run-id: {existing}")
    resume_path = checkpoint_dir / "lightning/last.ckpt"
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(f"--resume requested but checkpoint is missing: {resume_path}")
    for path in (output_dir, checkpoint_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_plan.json", plan)

    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from omegaconf import OmegaConf

    from module import SIGReg
    from utils import get_img_preprocessor

    pl.seed_everything(args.seed, workers=True)
    dataset = swm.data.load_dataset(
        str(args.dataset),
        transform=None,
        num_steps=NUM_STEPS,
        frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action", "observation"],
        keys_to_cache=["action", "observation"],
        keys_to_merge={"proprio": "proprio"},
    )
    excluded = {int(value) for value in protocol["episodes"]}
    allowed = allowed_clip_indices(dataset.clip_indices, excluded)
    if len(allowed) != plan["clips"]["allowed"]:
        raise RuntimeError(f"clip exclusion mismatch: planned={plan['clips']['allowed']}, actual={len(allowed)}")
    train_indices, val_indices = split_indices(allowed, args.train_split, args.seed)
    if len(train_indices) != plan["clips"]["train"] or len(val_indices) != plan["clips"]["validation"]:
        raise RuntimeError("train/validation split differs from frozen plan")
    if any(int(dataset.clip_indices[index][0]) in excluded for index in np.concatenate((train_indices, val_indices))):
        raise RuntimeError("held-out episode leaked into a training/validation clip")

    included_rows = _row_inclusion(protocol["lengths"], protocol["offsets"], excluded)
    normalizers = []
    normalizer_meta = {}
    for column in ("action", "observation"):
        mean, std, count = streaming_mean_std(dataset.get_col_data(column), included_rows)
        transform_mean, transform_std = mean, std
        if column == "action":
            # HDF5Dataset reshapes each frameskip block from [5,5] to [25]
            # before our split-specific wrapper runs.
            transform_mean = mean.repeat(1, FRAMESKIP)
            transform_std = std.repeat(1, FRAMESKIP)
        normalizers.append(ColumnNormalizer(column, transform_mean, transform_std))
        normalizer_meta[column] = {"count": count, "mean": mean.tolist(), "std": std.tolist()}
    _write_json(output_dir / "normalizers.json", normalizer_meta)

    preprocessor = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    train_transform = spt.data.transforms.Compose(
        RandomHueRotation(probability=args.hue_probability, max_delta=args.max_hue_delta),
        preprocessor,
        *normalizers,
    )
    val_transform = spt.data.transforms.Compose(preprocessor, *normalizers)
    train_set = IndexedTransformDataset(dataset, train_indices, train_transform)
    val_set = IndexedTransformDataset(dataset, val_indices, val_transform)
    generator = torch.Generator().manual_seed(args.seed)
    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_common.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    train_loader = torch.utils.data.DataLoader(
        train_set, **loader_common, shuffle=True, drop_last=True, generator=generator
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **loader_common, shuffle=False, drop_last=False
    )

    model = swm.wm.utils.load_pretrained(str(args.warm_start), cache_dir=str(AILAB))
    # stable_worldmodel.save_pretrained serializes via OmegaConf.to_container;
    # keep the model's complete Hydra config as an OmegaConf object so the
    # exported config.json remains directly loadable by load_pretrained.
    model_config = OmegaConf.create(json.loads(config_path.read_text(encoding="utf-8")))
    forward_cfg = OmegaConf.create({
        "history_size": HISTORY_SIZE,
        "num_preds": NUM_PREDS,
        "loss": {"sigreg": {"weight": args.sigreg_weight}},
    })
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {"type": "AdamW", "lr": args.learning_rate, "weight_decay": args.weight_decay},
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "interval": "step",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(knots=17, num_proj=1024),
        forward=partial(route2_forward, cfg=forward_cfg),
        optim=optimizers,
    )
    configure_manual_gradient_clipping(
        module,
        optimizer_name="model_opt",
        clip_val=args.gradient_clip_val,
        algorithm="norm",
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    weight_callback = SaveRoute2Weights.create(args.run_id, model_config)
    checkpoint_kwargs = {
        "dirpath": str(checkpoint_dir / "lightning"),
        "save_last": True,
        "save_top_k": -1,
        "enable_version_counter": False,
    }
    if args.max_steps is None:
        checkpoint_kwargs.update({"every_n_epochs": 1, "filename": "epoch{epoch:02d}-step{step}"})
    else:
        checkpoint_kwargs.update({"every_n_train_steps": min(1000, args.max_steps), "filename": "step{step}"})
    checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
    logger = TensorBoardLogger(
        save_dir=str(TENSORBOARD_ROOT), name="", version=args.run_id,
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        max_epochs=args.max_epochs if args.max_steps is None else -1,
        max_steps=-1 if args.max_steps is None else args.max_steps,
        callbacks=[weight_callback, checkpoint_callback],
        logger=logger,
        default_root_dir=str(output_dir / "lightning"),
        num_sanity_val_steps=1,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
    )
    trainer.fit(module, datamodule=data_module, ckpt_path=str(resume_path) if args.resume else None)
    _write_json(output_dir / "completed.json", {
        "run_id": args.run_id,
        "global_step": trainer.global_step,
        "epochs_completed": trainer.current_epoch,
        "final_weights": str((checkpoint_dir / "weights_final.pt").resolve()),
        "tensorboard": str(tensorboard_dir.resolve()),
    })
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--warm-start", type=Path, default=WARM_WEIGHTS)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--max-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--hue-probability", type=float, default=0.8)
    p.add_argument("--max-hue-delta", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--precision", choices=("bf16-mixed",), default="bf16-mixed")
    p.add_argument("--limit-val-batches", type=int, default=50)
    p.add_argument("--log-every-n-steps", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.max_epochs < 1:
        parser().error("--max-epochs must be >=1")
    if parsed.max_steps is not None and parsed.max_steps < 1:
        parser().error("--max-steps must be >=1")
    if parsed.batch_size < 1 or parsed.num_workers < 0:
        parser().error("batch size/workers are invalid")
    if parsed.gradient_clip_val <= 0:
        parser().error("--gradient-clip-val must be positive")
    raise SystemExit(run(parsed))
