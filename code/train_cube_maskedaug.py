#!/usr/bin/env python3
"""Route 2.1: warm-start Cube LeWM with online red-mask hue augmentation.

This entry point preserves Route2's optimizer, schedule, budget, validation,
and fixed-50 episode exclusion.  Only the image intervention changes: hue is
rotated inside the frozen float64 red-pixel mask, while all other pixels are
kept element-for-element unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cube_coloraug import (
    IndexedTransformDataset,
    allowed_clip_indices,
    split_indices,
    streaming_mean_std,
)
from cube_maskedaug import MASK_METADATA_PREFIX, RandomMaskedHueRotation, save_qc_artifacts
from train_cube_coloraug import (
    ColumnNormalizer,
    _configure_storage,
    _heldout_protocol,
    _row_inclusion,
    _safe_run_id,
    _sha,
    _validate_data_disk,
    _write_json,
    configure_manual_gradient_clipping,
    route2_forward,
)


AILAB = Path(__file__).resolve().parent.parent
DATASET = AILAB / "datasets/ogbench/cube_single_expert.h5"
MANIFEST = AILAB / "outputs/audit/cube_cem_manifest.json"
WARM_WEIGHTS = AILAB / "checkpoints/models--quentinll--lewm-cube/weights.pt"
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-maskedaug"
OUTPUT_ROOT = AILAB / "outputs/train/route21_maskedaug"
TENSORBOARD_ROOT = AILAB / "logs/tensorboard/route21_maskedaug"
NUM_STEPS = 4
FRAMESKIP = 5
HISTORY_SIZE = 3
NUM_PREDS = 1
QC_FRAMES = 10
STAT_NAMES = (
    "empty_frames",
    "total_frames",
    "masked_pixels",
    "applied_clips",
    "seen_clips",
)


def route21_forward(
    self: Any,
    batch: dict[str, torch.Tensor],
    stage: str,
    cfg: Any,
) -> dict[str, torch.Tensor]:
    """Consume augmentation audit metadata, then run Route2's JEPA loss."""
    step_stats: dict[str, torch.Tensor] = {}
    for name in STAT_NAMES:
        key = f"{MASK_METADATA_PREFIX}{name}"
        if key in batch:
            step_stats[name] = batch.pop(key).sum().detach()
    # stable_pretraining.Module.training_step calls forward with stage="fit"
    # (validation uses "validate").  Accept "train" as a compatibility alias,
    # but never infer training from worker-local state.
    if stage in {"fit", "train"} and step_stats:
        totals = getattr(self, "_maskedaug_totals", None)
        if totals is None:
            totals = {name: value.clone() for name, value in step_stats.items()}
            self._maskedaug_totals = totals
        else:
            for name, value in step_stats.items():
                totals[name] = totals[name] + value
        total_frames = step_stats.get("total_frames")
        empty_frames = step_stats.get("empty_frames")
        if total_frames is not None and empty_frames is not None:
            self.log(
                "train/maskedaug_empty_frame_rate",
                empty_frames.float() / total_frames.clamp_min(1).float(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
    return route2_forward(self, batch, stage, cfg)


def _runtime_stats(module: Any) -> dict[str, int | float]:
    totals = getattr(module, "_maskedaug_totals", {})
    result = {
        name: int(totals[name].detach().cpu().item()) if name in totals else 0
        for name in STAT_NAMES
    }
    result["empty_frame_rate"] = (
        result["empty_frames"] / result["total_frames"]
        if result["total_frames"]
        else 0.0
    )
    return result


def _plan(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    allowed = protocol["total_clips"] - protocol["excluded_clips"]
    train = math.floor(allowed * args.train_split)
    validation = allowed - train
    steps_per_epoch = train // args.batch_size
    total_steps = args.max_steps if args.max_steps is not None else steps_per_epoch * args.max_epochs
    warmup_steps = max(1, int(0.01 * total_steps))
    return {
        "route": "route21_online_float64_red_mask_hsv_rotation",
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
        "clips": {
            "before_exclusion": protocol["total_clips"],
            "allowed": allowed,
            "train": train,
            "validation": validation,
        },
        "augmentation": {
            "space_and_precision": "float64 HSV before ImageNet normalization",
            "mask": {
                "hue": ">0.9",
                "saturation": ">0.4",
                "value": ">0.15",
            },
            "scope": "masked pixels only; one shared hue delta across all 4 frames",
            "hue_delta_turns": [-args.max_hue_delta, args.max_hue_delta],
            "shift_probability": args.hue_probability,
            "identity_probability": 1.0 - args.hue_probability,
            "outside_mask_elementwise_unchanged": True,
            "saturation_changed": False,
            "value_changed": False,
            "validation_augmented": False,
            "empty_mask_counted_at_runtime": True,
        },
        "qc": {
            "num_frames": QC_FRAMES,
            "output": str((OUTPUT_ROOT / args.run_id / "qc").resolve()),
            "contact_sheet_grid": "2 columns x 5 rows",
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


class SaveRoute21Weights:
    @staticmethod
    def create(run_id: str, model_config: Any):
        from lightning.pytorch.callbacks import Callback
        from stable_worldmodel.wm.utils import save_pretrained

        class CallbackImpl(Callback):
            def _save(self, trainer: Any, module: Any, filename: str) -> None:
                if trainer.is_global_zero:
                    save_pretrained(
                        module.model,
                        run_name=f"lewm-cube-maskedaug/{run_id}",
                        config=model_config,
                        filename=filename,
                        cache_dir=str(AILAB),
                    )

            def on_train_epoch_end(self, trainer: Any, module: Any) -> None:
                self._save(trainer, module, f"weights_epoch_{trainer.current_epoch + 1}.pt")

            def on_train_end(self, trainer: Any, module: Any) -> None:
                self._save(trainer, module, "weights_final.pt")

        return CallbackImpl()


def _existing_qc(path: Path) -> dict[str, Any]:
    metadata = path / "qc.json"
    pngs = sorted(path.glob("frame_*.png")) if path.is_dir() else []
    if not metadata.is_file() or len(pngs) != QC_FRAMES or not (path / "contact_sheet.png").is_file():
        raise RuntimeError(f"resume requires complete frozen 10-frame QC artifacts: {path}")
    return json.loads(metadata.read_text(encoding="utf-8"))


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

    qc_dir = output_dir / "qc"
    if args.resume:
        qc = _existing_qc(qc_dir)
    else:
        qc = save_qc_artifacts(
            dataset,
            train_indices,
            qc_dir,
            seed=args.seed,
            probability=args.hue_probability,
            max_delta=args.max_hue_delta,
            num_frames=QC_FRAMES,
        )
    if not qc.get("all_outside_equal_float64") or not qc.get("all_outside_equal_float32"):
        raise RuntimeError("QC detected a change outside the frozen HSV mask")

    included_rows = _row_inclusion(protocol["lengths"], protocol["offsets"], excluded)
    normalizers = []
    normalizer_meta = {}
    for column in ("action", "observation"):
        mean, std, count = streaming_mean_std(dataset.get_col_data(column), included_rows)
        transform_mean, transform_std = mean, std
        if column == "action":
            transform_mean = mean.repeat(1, FRAMESKIP)
            transform_std = std.repeat(1, FRAMESKIP)
        normalizers.append(ColumnNormalizer(column, transform_mean, transform_std))
        normalizer_meta[column] = {"count": count, "mean": mean.tolist(), "std": std.tolist()}
    _write_json(output_dir / "normalizers.json", normalizer_meta)

    preprocessor = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    train_transform = spt.data.transforms.Compose(
        RandomMaskedHueRotation(probability=args.hue_probability, max_delta=args.max_hue_delta),
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
        forward=partial(route21_forward, cfg=forward_cfg),
        optim=optimizers,
    )
    configure_manual_gradient_clipping(module, "model_opt", args.gradient_clip_val, "norm")
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    weight_callback = SaveRoute21Weights.create(args.run_id, model_config)
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
    logger = TensorBoardLogger(
        save_dir=str(TENSORBOARD_ROOT), name="", version=args.run_id, default_hp_metric=False
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        max_epochs=args.max_epochs if args.max_steps is None else -1,
        max_steps=-1 if args.max_steps is None else args.max_steps,
        callbacks=[weight_callback, ModelCheckpoint(**checkpoint_kwargs)],
        logger=logger,
        default_root_dir=str(output_dir / "lightning"),
        num_sanity_val_steps=1,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
    )
    trainer.fit(module, datamodule=data_module, ckpt_path=str(resume_path) if args.resume else None)
    _write_json(
        output_dir / "completed.json",
        {
            "run_id": args.run_id,
            "global_step": trainer.global_step,
            "epochs_completed": trainer.current_epoch,
            "final_weights": str((checkpoint_dir / "weights_final.pt").resolve()),
            "tensorboard": str(tensorboard_dir.resolve()),
            "masked_augmentation_runtime": _runtime_stats(module),
            "qc": str((qc_dir / "qc.json").resolve()),
        },
    )
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


def main(argv: Sequence[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.max_epochs < 1:
        p.error("--max-epochs must be >=1")
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max-steps must be >=1")
    if args.batch_size < 1 or args.num_workers < 0:
        p.error("batch size/workers are invalid")
    if args.gradient_clip_val <= 0:
        p.error("--gradient-clip-val must be positive")
    if not 0.0 <= args.hue_probability <= 1.0:
        p.error("--hue-probability must be in [0,1]")
    if not 0.0 < args.max_hue_delta <= 0.5:
        p.error("--max-hue-delta must be in (0,0.5]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
