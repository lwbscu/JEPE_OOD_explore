#!/usr/bin/env python3
"""Train the single Cube off-policy V3 arm with a realtime expert stopline.

All V2 formal data, split, augmentation, optimizer, and runtime choices are
frozen.  The sole training change disables expert autoregressive rollout while
retaining the attached five-step V2 rollout.  A paired clean-expert metric is
measured at step zero and after every 500 completed optimizer steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import hdf5plugin  # noqa: F401
import numpy as np
import torch

from cube_maskedaug import MASK_METADATA_PREFIX, RandomMaskedHueRotation
from cube_offpolicy_v3 import (
    CANONICAL_ARM,
    EXPERT_BASELINE,
    FORMAL_BATCH_SIZE,
    FORMAL_EXPERT_BATCH,
    FORMAL_RUN_ID,
    FORMAL_V2_BATCH,
    LOSS_CONTRACT,
    LOSS_CONTRACT_SHA256,
    NORMALIZERS_SHA256,
    STOPLINE_BATCHES,
    STOPLINE_EXAMPLES,
    STOPLINE_INTERVAL,
    STOPLINE_THRESHOLD,
    V2_FORMAL_PROVENANCE_SHA256,
    V2_FORMAL_SPLIT_SHA256,
    ExpertRolloutDataset,
    PlannerRolloutDataset,
    StrictMixtureDataset,
    assert_frozen_hashes,
    atomic_write_json,
    bind_formal_v2_split,
    file_identity,
    flatten_bundled_source,
    freeze_predictor_only,
    frozen_state_hashes,
    load_planner_manifest,
    sha256_file,
)
from train_cube_coloraug import (
    ColumnNormalizer,
    _configure_storage,
    _heldout_protocol,
    _safe_run_id,
    _validate_data_disk,
    configure_manual_gradient_clipping,
)
from train_cube_offpolicy_v2 import (
    MASK_STAT_NAMES,
    _global_episode_split,
    _load_normalizers,
    _measurement1_holdout,
)


AILAB = Path(__file__).resolve().parent.parent
EXPERT_DATASET = AILAB / "datasets/ogbench/cube_single_expert.h5"
FORMAL_MANIFEST = AILAB / "outputs/audit/cube_cem_manifest.json"
MEASUREMENT1_SEGMENTS = AILAB / "outputs/eval/cube/imagination_error/measurement1_segments.json"
V2_ROOT = AILAB / "datasets/offpolicy_cube_v2"
WARM_WEIGHTS = (
    AILAB / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt"
)
NORMALIZERS = (
    AILAB / "outputs/train/route21_maskedaug/route21_masked_hsv_seed3072/normalizers.json"
)
V2_FORMAL_SPLIT = (
    AILAB / "outputs/train/offpolicy_v2/offpolicy_v2_pred_seed3072/episode_split.npz"
)
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-offpolicy_v3"
OUTPUT_ROOT = AILAB / "outputs/train/offpolicy_v3"
TENSORBOARD_ROOT = AILAB / "logs/tensorboard/offpolicy_v3"
NUM_FRAMES = 6
TEACHER_FRAMES = 4
TEACHER_ACTIONS = 3
HISTORY_SIZE = 3
EXPECTED_TRAINABLE_TENSORS = 93
EXPECTED_TRAINABLE_PARAMETERS = 11_740_484


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _encode_targets(model: Any, pixels: torch.Tensor) -> torch.Tensor:
    """Encode all six frames exactly as V2 did; targets remain stop-gradient."""
    if pixels.ndim != 5 or tuple(pixels.shape[1:]) != (6, 3, 224, 224):
        raise ValueError(f"expected six normalized frames, actual={tuple(pixels.shape)}")
    batch = pixels.shape[0]
    model.encoder.eval()
    model.projector.eval()
    with torch.no_grad():
        output = model.encoder(
            pixels.float().reshape(batch * NUM_FRAMES, 3, 224, 224),
            interpolate_pos_encoding=True,
        )
        embedding = model.projector(output.last_hidden_state[:, 0])
    return embedding.reshape(batch, NUM_FRAMES, -1)


def _teacher_from_targets(
    model: Any, target: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    if tuple(target.shape[:2]) != (action.shape[0], NUM_FRAMES):
        raise ValueError(
            f"teacher target/action batch mismatch: target={tuple(target.shape)}, action={tuple(action.shape)}"
        )
    action_embedding = model.action_encoder(action[:, :TEACHER_ACTIONS])
    prediction = model.predict(target[:, :TEACHER_ACTIONS], action_embedding)
    return (prediction - target[:, 1:TEACHER_FRAMES].detach()).square().mean()


def _v2_rollout_from_targets(
    model: Any,
    pixels: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the production rollout exactly once for the V2 source only."""
    rolled = model.rollout(
        {
            "pixels": pixels[:, None, :1],
            "emb": target[:, None, :1].detach(),
        },
        action[:, None],
        history_size=HISTORY_SIZE,
    )["predicted_emb"][:, 0, 1:]
    if tuple(rolled.shape[:2]) != (pixels.shape[0], 5):
        raise RuntimeError(f"model.rollout returned unexpected shape {tuple(rolled.shape)}")
    squared = (rolled - target[:, 1:].detach()).square()
    depth_losses = squared.mean(dim=(0, 2))
    return depth_losses.mean(), depth_losses


def _expert_source_losses(
    model: Any, source: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expert is teacher-only; zero rollout metrics are exact and detached."""
    pixels = source["pixels"]
    action = source["action"]
    if tuple(action.shape[1:]) != (5, 25) or not torch.isfinite(action).all():
        raise ValueError(f"expected finite expert action [B,5,25], actual={tuple(action.shape)}")
    target = _encode_targets(model, pixels)
    teacher = _teacher_from_targets(model, target, action)
    rollout = teacher.new_zeros(())
    depths = teacher.new_zeros((5,))
    if rollout.requires_grad or depths.requires_grad or torch.count_nonzero(depths).item():
        raise RuntimeError("expert rollout zeros must be exact and gradient-free")
    return teacher, rollout, depths, target


def _v2_source_losses(
    model: Any, source: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pixels = source["pixels"]
    action = source["action"]
    if tuple(action.shape[1:]) != (5, 25) or not torch.isfinite(action).all():
        raise ValueError(f"expected finite V2 action [B,5,25], actual={tuple(action.shape)}")
    target = _encode_targets(model, pixels)
    teacher = _teacher_from_targets(model, target, action)
    rollout, depths = _v2_rollout_from_targets(model, pixels, action, target)
    return teacher, rollout, depths, target


def _sigreg_shared_projection(
    sigreg: Any, expert: torch.Tensor, v2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    combined = torch.cat((expert[:, :4], v2[:, :4]), dim=0).transpose(0, 1)
    expert_tb = expert[:, :4].transpose(0, 1)
    v2_tb = v2[:, :4].transpose(0, 1)
    projection = torch.randn(combined.size(-1), sigreg.num_proj, device=combined.device)
    projection = projection.div_(projection.norm(p=2, dim=0))

    def statistic(value: torch.Tensor) -> torch.Tensor:
        x_t = (value @ projection).unsqueeze(-1) * sigreg.t
        error = (
            (x_t.cos().mean(-3) - sigreg.phi).square()
            + x_t.sin().mean(-3).square()
        )
        return ((error @ sigreg.weights) * value.size(-2)).mean()

    return statistic(combined), statistic(expert_tb), statistic(v2_tb)


def v3_forward(self: Any, batch: dict[str, Any], stage: str, cfg: Any) -> dict[str, torch.Tensor]:
    data_keys = set(batch) - {"batch_idx"}
    if data_keys != {"expert", "v2"}:
        raise ValueError(f"unexpected V3 batch keys: {sorted(batch)}")
    expert = flatten_bundled_source(batch["expert"])
    v2 = flatten_bundled_source(batch["v2"])
    expected = {"expert": FORMAL_EXPERT_BATCH, "v2": FORMAL_V2_BATCH}
    actual = {"expert": int(expert["pixels"].shape[0]), "v2": int(v2["pixels"].shape[0])}
    if actual != expected or sum(actual.values()) != FORMAL_BATCH_SIZE:
        raise RuntimeError(f"strict V3 source batch mismatch: expected={expected}, actual={actual}")

    mask_stats: dict[str, torch.Tensor] = {}
    for name in MASK_STAT_NAMES:
        key = f"{MASK_METADATA_PREFIX}{name}"
        if key in expert:
            mask_stats[name] = expert.pop(key).sum().detach()
    if stage in {"fit", "train"}:
        counts = getattr(self, "_v3_source_examples", {"expert": 0, "v2": 0})
        counts["expert"] += FORMAL_EXPERT_BATCH
        counts["v2"] += FORMAL_V2_BATCH
        self._v3_source_examples = counts
        totals = getattr(self, "_v3_mask_totals", {name: 0 for name in MASK_STAT_NAMES})
        for name, value in mask_stats.items():
            totals[name] = totals[name] + value
        self._v3_mask_totals = totals

    expert_teacher, expert_rollout, expert_depth, expert_embedding = _expert_source_losses(
        self.model, expert
    )
    v2_teacher, v2_rollout, v2_depth, v2_embedding = _v2_source_losses(self.model, v2)
    shared_sigreg, expert_sigreg, v2_sigreg = _sigreg_shared_projection(
        self.sigreg, expert_embedding, v2_embedding
    )
    teacher_loss = 0.75 * expert_teacher + 0.25 * v2_teacher
    rollout_loss = 0.25 * v2_rollout
    loss = teacher_loss + 0.09 * shared_sigreg + 0.5 * rollout_loss
    metrics: dict[str, torch.Tensor] = {
        "loss": loss,
        "pred_loss": teacher_loss,
        "sigreg_loss": shared_sigreg,
        "rollout_loss": rollout_loss,
        "expert_pred_loss": expert_teacher,
        "v2_pred_loss": v2_teacher,
        "expert_sigreg_loss": expert_sigreg,
        "v2_sigreg_loss": v2_sigreg,
        "expert_rollout_loss": expert_rollout,
        "v2_rollout_loss": v2_rollout,
    }
    for depth in range(5):
        metrics[f"expert_rollout_depth{depth + 1}_loss"] = expert_depth[depth]
        metrics[f"v2_rollout_depth{depth + 1}_loss"] = v2_depth[depth]
    if not all(bool(torch.isfinite(value.detach()).item()) for value in metrics.values()):
        raise FloatingPointError(
            "V3 non-finite loss: " + ", ".join(f"{key}={value.detach()}" for key, value in metrics.items())
        )
    if any(float(metrics[f"expert_rollout_depth{depth}_loss"].detach()) != 0.0 for depth in range(1, 6)):
        raise RuntimeError("expert rollout depth logging is not exact zero")
    self.log_dict(
        {f"{stage}/{key}": value.detach() for key, value in metrics.items()},
        on_step=True,
        sync_dist=True,
    )
    self._v3_last_metrics = {
        key: float(value.detach().float().cpu()) for key, value in metrics.items()
    }
    return metrics


def _build_datasets(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    measurement: Mapping[str, Any],
    collector: Mapping[str, Any],
) -> dict[str, Any]:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_img_preprocessor

    dataset = swm.data.load_dataset(
        str(args.dataset), transform=None, num_steps=NUM_FRAMES, frameskip=5,
        keys_to_load=["pixels", "action"], keys_to_cache=["action"],
    )
    excluded = set(map(int, protocol["episodes"])) | set(map(int, measurement["episodes"]))
    split = _global_episode_split(
        dataset,
        v2_source_by_rollout=collector["source_episode_by_rollout"],
        excluded=excluded,
        train_fraction=args.train_fraction,
        seed=args.split_seed,
    )
    normalizers = _load_normalizers(args.normalizers)
    mean = torch.tensor(normalizers["action"]["mean"], dtype=torch.float32).reshape(1, 5).repeat(1, 5)
    std = torch.tensor(normalizers["action"]["std"], dtype=torch.float32).reshape(1, 5).repeat(1, 5)
    preprocessor = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    expert_train_transform = spt.data.transforms.Compose(
        RandomMaskedHueRotation(probability=args.hue_probability, max_delta=args.max_hue_delta),
        preprocessor,
        ColumnNormalizer("action", mean, std),
    )
    clean_transform = spt.data.transforms.Compose(
        preprocessor, ColumnNormalizer("action", mean, std)
    )
    expert_train = ExpertRolloutDataset(dataset, split["expert_train_ids"], expert_train_transform)
    expert_val = ExpertRolloutDataset(dataset, split["expert_val_ids"], clean_transform)
    v2_train = PlannerRolloutDataset(collector, split["v2_train_ids"], preprocessor)
    v2_val = PlannerRolloutDataset(collector, split["v2_val_ids"], preprocessor)
    return {
        "train": StrictMixtureDataset(expert_train, v2_train, 3, 1),
        "val": StrictMixtureDataset(expert_val, v2_val, 3, 1),
        "expert_val": expert_val,
        "v2_val": v2_val,
        **split,
        "bundle": {"expert": 3, "v2": 1, "loader_batch": 32},
    }


def _runtime_mask_stats(module: Any) -> dict[str, int | float]:
    totals = getattr(module, "_v3_mask_totals", {})
    result = {
        name: int(totals[name].detach().cpu()) if torch.is_tensor(totals.get(name)) else int(totals.get(name, 0))
        for name in MASK_STAT_NAMES
    }
    result["empty_frame_rate"] = (
        result["empty_frames"] / result["total_frames"] if result["total_frames"] else 0.0
    )
    return result


def _split_summary(datasets: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expert_train_clips": len(datasets["expert_train_ids"]),
        "expert_validation_clips": len(datasets["expert_val_ids"]),
        "expert_train_episodes": len(datasets["expert_train_episodes"]),
        "expert_validation_episodes": len(datasets["expert_val_episodes"]),
        "v2_train_rollouts": len(datasets["v2_train_ids"]),
        "v2_validation_rollouts": len(datasets["v2_val_ids"]),
        "v2_train_source_episodes": len(datasets["v2_train_episodes"]),
        "v2_validation_source_episodes": len(datasets["v2_val_episodes"]),
        "global_train_episodes": len(datasets["global_train_episodes"]),
        "global_validation_episodes": len(datasets["global_val_episodes"]),
        "assignment": "exact formal V2 global episode split",
        "cross_split_episode_overlaps": dict(datasets["cross_split_episode_overlaps"]),
        "formal_v2_episode_split_sha256": V2_FORMAL_SPLIT_SHA256,
    }


def _plan(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    measurement: Mapping[str, Any],
    collector: Mapping[str, Any],
    datasets: Mapping[str, Any],
    freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": "cube_offpolicy_train_v3",
        "run_id": args.run_id,
        "arm": CANONICAL_ARM,
        "single_arm": True,
        "retry_allowed": False,
        "resume_allowed": False,
        "inputs": {
            "expert_dataset": str(args.dataset),
            "formal_manifest": str(args.manifest),
            "formal_manifest_sha256": protocol["manifest_sha256"],
            "measurement1_segments": {
                "path": measurement["path"], "sha256": measurement["sha256"],
                "num_segments": int(measurement["num_segments"]),
                "num_episodes": int(measurement["num_episodes"]),
            },
            "v2_root": str(args.offpolicy_root),
            "v2_manifest": collector["manifest_path"],
            "v2_manifest_sha256": collector["manifest_sha256"],
            "collector_validation": collector["collector_validation"],
            "collector_bound_inputs": collector["bound_inputs"],
            "warm_start": str(args.warm_start),
            "warm_start_sha256": sha256_file(args.warm_start),
            "normalizers": str(args.normalizers),
            "normalizers_sha256": sha256_file(args.normalizers),
            "formal_v2_episode_split": file_identity(V2_FORMAL_SPLIT),
        },
        "splits": _split_summary(datasets),
        "batch": {
            "total": FORMAL_BATCH_SIZE,
            "expert": FORMAL_EXPERT_BATCH,
            "v2": FORMAL_V2_BATCH,
            "expert_fraction": 0.75,
            "v2_fraction": 0.25,
            "strict_every_step": True,
        },
        "loss_contract": dict(LOSS_CONTRACT),
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "optimization": {
            "optimizer": "AdamW", "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "steps": args.max_steps,
            "precision": args.precision, "scheduler": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": 50, "gradient_clip_norm": args.gradient_clip_val,
            "checkpoints_every_steps": 1000,
        },
        "expert_augmentation": {
            "masked_hue_probability": args.hue_probability,
            "max_hue_delta_turns": args.max_hue_delta,
            "validation": "clean identity",
        },
        "realtime_stopline": {
            "metric": "paired clean expert exact original four-frame teacher-forced pred loss",
            "baseline": EXPERT_BASELINE,
            "threshold_relative_increase": STOPLINE_THRESHOLD,
            "comparison": "current/baseline-1 strictly greater than threshold",
            "interval_completed_optimizer_steps": STOPLINE_INTERVAL,
            "batches": STOPLINE_BATCHES,
            "examples": STOPLINE_EXAMPLES,
            "batch_size": FORMAL_BATCH_SIZE,
            "shuffle": False,
            "precision": "bf16",
            "on_failure": "atomic history+event, live Lightning checkpoint, stopped weights, normal trainer stop",
        },
        "runtime": {
            "seed": args.seed, "split_seed": args.split_seed,
            "train_fraction": args.train_fraction, "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "limit_val_batches": args.limit_val_batches,
        },
        "freeze": dict(freeze or {}),
        "paths": {
            "output": str((OUTPUT_ROOT / args.run_id).resolve()),
            "checkpoint": str((CHECKPOINT_ROOT / args.run_id).resolve()),
            "tensorboard": str((TENSORBOARD_ROOT / args.run_id).resolve()),
        },
        "code": {
            "train": sha256_file(Path(__file__)),
            "dataset": sha256_file(Path(__file__).with_name("cube_offpolicy_v3.py")),
            "v2_dataset": sha256_file(Path(__file__).with_name("cube_offpolicy_v2.py")),
        },
    }


def _write_contract(output: Path, plan: Mapping[str, Any], datasets: Mapping[str, Any]) -> None:
    atomic_write_json(output / "run_plan.json", plan)
    identity = bind_formal_v2_split(V2_FORMAL_SPLIT, output / "episode_split.npz", datasets)
    if identity["sha256"] != V2_FORMAL_SPLIT_SHA256:
        raise RuntimeError("V3 episode split did not retain exact formal V2 bytes")


def _move_source(source: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in source.items()
    }


def _stopline_loader(datasets: Mapping[str, Any], args: argparse.Namespace) -> Any:
    common: dict[str, Any] = {
        "batch_size": FORMAL_BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers:
        common.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    loader = torch.utils.data.DataLoader(datasets["expert_val"], **common)
    if len(loader) < STOPLINE_BATCHES:
        raise RuntimeError(
            f"stopline lacks exact heldout batches: expected>={STOPLINE_BATCHES}, actual={len(loader)}"
        )
    return loader


def _evaluate_current_teacher(model: Any, loader: Any) -> tuple[float, dict[str, Any]]:
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("formal V3 realtime stopline requires CUDA bf16")
    modes = {id(submodule): bool(submodule.training) for submodule in model.modules()}
    model.eval()
    total = 0.0
    batches = 0
    provenance = hashlib.sha256()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if batch_index >= STOPLINE_BATCHES:
                    break
                ids = batch["expert_clip_index"]
                provenance.update(np.asarray(ids, dtype=np.int64).tobytes())
                source = _move_source(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    target = _encode_targets(model, source["pixels"])
                    teacher = _teacher_from_targets(model, target, source["action"])
                value = float(teacher.float().cpu())
                if not np.isfinite(value):
                    raise FloatingPointError(f"stopline teacher loss non-finite at batch {batch_index}")
                total += value
                batches += 1
    finally:
        for submodule in model.modules():
            submodule.training = modes[id(submodule)]
    if batches != STOPLINE_BATCHES:
        raise RuntimeError(
            f"stopline batch count mismatch: expected={STOPLINE_BATCHES}, actual={batches}"
        )
    digest = provenance.hexdigest()
    if digest != V2_FORMAL_PROVENANCE_SHA256:
        raise RuntimeError(
            "stopline heldout provenance changed: "
            f"expected={V2_FORMAL_PROVENANCE_SHA256}, actual={digest}"
        )
    return total / batches, {
        "num_batches": STOPLINE_BATCHES,
        "examples": STOPLINE_EXAMPLES,
        "batch_size": FORMAL_BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "precision": "bf16",
        "expert_clip_indices_sha256": digest,
        "episode_split_sha256": V2_FORMAL_SPLIT_SHA256,
    }


class V3Callbacks:
    @staticmethod
    def create(
        *,
        run_id: str,
        model_config: Any,
        output_dir: Path,
        checkpoint_dir: Path,
        frozen_before: Mapping[str, str],
        expert_loader: Any,
        every: int,
    ) -> tuple[list[Any], Any]:
        from lightning.pytorch.callbacks import Callback
        from stable_worldmodel.wm.utils import save_pretrained

        class StoplineAndExport(Callback):
            def __init__(self) -> None:
                self.history: dict[str, Any] | None = None
                self.triggered = False
                self.event_path: Path | None = None
                self.live_checkpoint: Path | None = None
                self.stopped_weights: Path | None = None

            def _export(self, trainer: Any, module: Any, filename: str) -> Path:
                assert_frozen_hashes(module.model, frozen_before)
                save_pretrained(
                    module.model,
                    run_name=f"lewm-cube-offpolicy_v3/{run_id}",
                    config=model_config,
                    filename=filename,
                    cache_dir=str(AILAB),
                )
                path = checkpoint_dir / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                return path

            def _publish_history(self) -> None:
                if self.history is None:
                    raise RuntimeError("V3 stopline history is not initialized")
                atomic_write_json(output_dir / "stopline_history.json", self.history)

            def on_train_start(self, trainer: Any, module: Any) -> None:
                if not trainer.is_global_zero:
                    return
                current, provenance = _evaluate_current_teacher(module.model, expert_loader)
                if current != EXPERT_BASELINE:
                    raise RuntimeError(
                        "step0 baseline exact mismatch: "
                        f"expected={EXPERT_BASELINE!r}, actual={current!r}, "
                        f"position=paired clean expert 34x128 bf16"
                    )
                self.history = {
                    "format_version": "cube_offpolicy_v3_stopline_history_v1",
                    "run_id": run_id,
                    "arm": CANONICAL_ARM,
                    "baseline": {
                        "expected": EXPERT_BASELINE,
                        "current": current,
                        "step": 0,
                        "relative": current / EXPERT_BASELINE - 1.0,
                        "status": "PASS",
                        "provenance": provenance,
                    },
                    "threshold": STOPLINE_THRESHOLD,
                    "comparison": "relative strictly greater than threshold triggers",
                    "interval": STOPLINE_INTERVAL,
                    "evaluated_after_optimizer_step": True,
                    "triggered": False,
                    "records": [],
                }
                self._publish_history()

            def on_train_batch_end(
                self, trainer: Any, module: Any, outputs: Any, batch: Any, batch_idx: int
            ) -> None:
                step = int(trainer.global_step)
                if (
                    not trainer.is_global_zero
                    or self.triggered
                    or step < STOPLINE_INTERVAL
                    or step % STOPLINE_INTERVAL != 0
                ):
                    return
                assert self.history is not None
                if self.history["records"] and self.history["records"][-1]["step"] == step:
                    return
                current, provenance = _evaluate_current_teacher(module.model, expert_loader)
                if provenance != self.history["baseline"]["provenance"]:
                    raise RuntimeError("realtime stopline is not paired to the step0 heldout batches")
                relative = current / EXPERT_BASELINE - 1.0
                failed = relative > STOPLINE_THRESHOLD
                record = {
                    "step": step,
                    "current": current,
                    "relative": relative,
                    "status": "STOPLINE_FAIL" if failed else "PASS",
                    "provenance": provenance,
                }
                self.history["records"].append(record)
                self.history["triggered"] = failed
                if failed:
                    self.history["trigger"] = dict(record)
                self._publish_history()
                if not failed:
                    return

                self.triggered = True
                self.event_path = output_dir / "stopline_event.json"
                atomic_write_json(self.event_path, {
                    "format_version": "cube_offpolicy_v3_stopline_event_v1",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "arm": CANONICAL_ARM,
                    **record,
                    "threshold": STOPLINE_THRESHOLD,
                    "history_sha256_before_checkpoint": sha256_file(
                        output_dir / "stopline_history.json"
                    ),
                    "action": "save live checkpoint and stopped weights, then set trainer.should_stop",
                })
                self.live_checkpoint = checkpoint_dir / "lightning" / f"stopline_step{step}.ckpt"
                self.live_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(str(self.live_checkpoint))
                self.stopped_weights = self._export(
                    trainer, module, f"weights_stopped_step{step}.pt"
                )
                trainer.should_stop = True

            def on_train_end(self, trainer: Any, module: Any) -> None:
                if not trainer.is_global_zero:
                    return
                self._export(trainer, module, "weights_final.pt")
                after = assert_frozen_hashes(module.model, frozen_before)
                atomic_write_json(output_dir / "frozen_integrity.json", {
                    "status": "PASS", "before": dict(frozen_before), "after": after,
                    "exact_match": after == dict(frozen_before),
                })

        class Curves(Callback):
            def on_train_batch_end(
                self, trainer: Any, module: Any, outputs: Any, batch: Any, batch_idx: int
            ) -> None:
                step = int(trainer.global_step)
                if not trainer.is_global_zero or not (
                    step == 1 or step == trainer.max_steps or step % every == 0
                ):
                    return
                values = dict(getattr(module, "_v3_last_metrics", {}))
                values.update({
                    "step": step, "epoch": int(trainer.current_epoch),
                    "batch_idx": int(batch_idx),
                    "learning_rate": float(trainer.optimizers[0].param_groups[0]["lr"]),
                })
                with (output_dir / "loss_curve.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(values, sort_keys=True) + "\n")

        stopline = StoplineAndExport()
        return [stopline, Curves()], stopline


def _posthoc_payload(
    args: argparse.Namespace,
    callback: Any,
    final_checkpoint: Path,
) -> dict[str, Any]:
    history = callback.history
    _validate_stopline_history(history, args.max_steps)
    assert history is not None
    base = history["baseline"]
    final = history["records"][-1]
    provenance = base["provenance"]
    if final["provenance"] != provenance:
        raise RuntimeError("V3 final stopline record is not paired with baseline")
    passed = not history["triggered"] and final["relative"] <= STOPLINE_THRESHOLD

    def source(value: float) -> dict[str, Any]:
        return {
            "num_batches": STOPLINE_BATCHES,
            "examples": STOPLINE_EXAMPLES,
            "provenance_sha256": provenance["expert_clip_indices_sha256"],
            "mean": {"teacher_pred_loss": value},
        }

    payload = {
        "format_version": "cube_offpolicy_v3_paired_posthoc_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "read_only": True,
        "protocol": {
            "clean_expert": True, "paired_base_final": True,
            "requested_batches": STOPLINE_BATCHES,
            "available_full_batches": {"expert": STOPLINE_BATCHES},
            "actual_batches": STOPLINE_BATCHES,
            "validation_examples_per_source": STOPLINE_EXAMPLES,
            "sampling": "finite expert heldout loader prefix without cycling",
            "metric": "exact original four-frame teacher-forced pred loss",
            "precision": "bf16", "shuffle": False,
            "measurement": "realtime current-model evaluations; no checkpoint reload",
        },
        "base": {
            "label": "masked_base_step0", "checkpoint": str(args.warm_start.resolve()),
            "checkpoint_sha256": sha256_file(args.warm_start),
            "sources": {"expert": source(base["current"])},
        },
        "final": {
            "label": "v3_stopped" if history["triggered"] else "v3_final",
            "checkpoint": str(final_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(final_checkpoint),
            "sources": {"expert": source(final["current"])},
        },
        "expert_stopline": {
            "threshold_relative_increase": STOPLINE_THRESHOLD,
            "base_teacher_pred_loss": base["current"],
            "final_teacher_pred_loss": final["current"],
            "relative_increase": final["relative"],
            "status": "PASS" if passed else "FAIL",
            "offline_gate_authorized": passed,
        },
    }
    atomic_write_json(OUTPUT_ROOT / args.run_id / "posthoc_validation.json", payload)
    return payload


def _validate_stopline_history(history: Any, max_steps: int) -> None:
    """Prove the realtime callback ran at every required completed step."""
    if not isinstance(history, Mapping) or not history.get("records"):
        raise RuntimeError("V3 has no periodic stopline record")
    records = history["records"]
    triggered = bool(history.get("triggered"))
    last_step = int(records[-1].get("step", -1))
    expected_last = last_step if triggered else int(max_steps)
    expected_steps = list(range(STOPLINE_INTERVAL, expected_last + 1, STOPLINE_INTERVAL))
    actual_steps = [int(record.get("step", -1)) for record in records]
    if actual_steps != expected_steps:
        raise RuntimeError(
            "V3 stopline schedule incomplete: "
            f"expected={expected_steps}, actual={actual_steps}, position=history.records.step"
        )
    baseline = history.get("baseline", {})
    provenance = baseline.get("provenance")
    if (
        history.get("interval") != STOPLINE_INTERVAL
        or history.get("threshold") != STOPLINE_THRESHOLD
        or baseline.get("step") != 0
        or baseline.get("expected") != EXPERT_BASELINE
        or baseline.get("current") != EXPERT_BASELINE
        or baseline.get("status") != "PASS"
    ):
        raise RuntimeError("V3 stopline baseline/interval/threshold contract changed")
    for index, record in enumerate(records):
        relative = float(record["current"]) / EXPERT_BASELINE - 1.0
        if (
            record.get("provenance") != provenance
            or not math.isclose(float(record["relative"]), relative, rel_tol=1e-15, abs_tol=1e-15)
        ):
            raise RuntimeError(f"V3 stopline record is unpaired/inconsistent at index {index}")
        is_last = index == len(records) - 1
        expected_status = "STOPLINE_FAIL" if triggered and is_last else "PASS"
        if record.get("status") != expected_status:
            raise RuntimeError(
                f"V3 stopline status mismatch: expected={expected_status}, "
                f"actual={record.get('status')}, position=records[{index}]"
            )
        if (relative > STOPLINE_THRESHOLD) != (triggered and is_last):
            raise RuntimeError(f"V3 stopline comparison mismatch at record {index}")
    if triggered and history.get("trigger") != records[-1]:
        raise RuntimeError("V3 stopline trigger does not bind the final failing record")
    if not triggered and "trigger" in history:
        raise RuntimeError("V3 PASS history unexpectedly contains a trigger")


def _load_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _heldout_protocol(args.dataset, args.manifest)
    measurement = _measurement1_holdout(args.measurement1_segments)
    collector = load_planner_manifest(
        args.offpolicy_root,
        expected_eval_episodes=protocol["episodes"],
        expected_measurement1_episodes=measurement["episodes"],
        expected_masked_checkpoint=args.warm_start,
        expected_formal_manifest=args.manifest,
        expected_measurement1_segments=args.measurement1_segments,
    )
    return protocol, measurement, collector


def synthetic_v3_smoke() -> dict[str, Any]:
    """CPU spy proof: expert never rolls out; V2 rolls once at batch 32 with BPTT."""
    from module import SIGReg

    torch.manual_seed(3072)

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_encoder = torch.nn.Linear(2, 4)
            self.predictor = torch.nn.Linear(8, 4)
            self.pred_proj = torch.nn.Linear(4, 4)
            self.rollout_calls: list[int] = []
            self.nodes: list[torch.Tensor] = []

        def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return self.pred_proj(self.predictor(torch.cat((state, action), dim=-1)))

        def rollout(
            self, observation: Mapping[str, torch.Tensor], action: torch.Tensor, *, history_size: int
        ) -> dict[str, torch.Tensor]:
            if history_size != HISTORY_SIZE:
                raise RuntimeError("synthetic history size changed")
            batch = int(action.shape[0])
            self.rollout_calls.append(batch)
            current = observation["emb"][:, 0]
            predictions = [current]
            self.nodes = []
            for depth in range(5):
                encoded = self.action_encoder(action[:, 0, depth : depth + 1])
                predicted = self.predict(current[:, -1:], encoded)
                predicted.retain_grad()
                self.nodes.append(predicted)
                predictions.append(predicted)
                current = torch.cat((current, predicted), dim=1)
            return {"predicted_emb": torch.stack((torch.cat(predictions, 1),), dim=1)}

    model = ToyModel()
    expert_target = torch.randn(FORMAL_EXPERT_BATCH, 6, 4)
    expert_action = torch.randn(FORMAL_EXPERT_BATCH, 5, 2)
    v2_target = torch.randn(FORMAL_V2_BATCH, 6, 4)
    v2_action = torch.randn(FORMAL_V2_BATCH, 5, 2)
    expert_teacher = _teacher_from_targets(model, expert_target, expert_action)
    expert_rollout = expert_teacher.new_zeros(())
    expert_depth = expert_teacher.new_zeros(5)
    if model.rollout_calls or expert_rollout.requires_grad or expert_depth.requires_grad:
        raise RuntimeError("synthetic expert path called rollout or created rollout gradient")
    v2_teacher = _teacher_from_targets(model, v2_target, v2_action)
    dummy_pixels = torch.empty(FORMAL_V2_BATCH, 6, 3, 1, 1)
    v2_rollout, v2_depth = _v2_rollout_from_targets(
        model, dummy_pixels, v2_action, v2_target
    )
    if model.rollout_calls != [FORMAL_V2_BATCH]:
        raise RuntimeError(
            f"rollout spy mismatch: expected=[{FORMAL_V2_BATCH}], actual={model.rollout_calls}"
        )
    sigreg = SIGReg(knots=5, num_proj=16)
    shared, _, _ = _sigreg_shared_projection(sigreg, expert_target, v2_target)
    loss = 0.75 * expert_teacher + 0.25 * v2_teacher + 0.09 * shared + 0.25 * 0.5 * v2_rollout
    loss.backward()
    first_node_grad = model.nodes[0].grad
    if first_node_grad is None or not torch.count_nonzero(first_node_grad):
        raise RuntimeError("five-step V2 autoregressive graph detached an intermediate prediction")
    return {
        "status": "PASS",
        "arm": CANONICAL_ARM,
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "source_batch": {"expert": FORMAL_EXPERT_BATCH, "v2": FORMAL_V2_BATCH},
        "rollout_spy_calls": model.rollout_calls,
        "expert_rollout_loss": float(expert_rollout),
        "expert_rollout_depth_loss": [float(value) for value in expert_depth],
        "expert_rollout_requires_grad": expert_rollout.requires_grad,
        "v2_depths": len(v2_depth),
        "v2_intermediate_bptt_gradient_nonzero": True,
    }


def _real_model_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """CPU real-model branch proof using one clean allowed expert clip."""
    from unittest.mock import patch

    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_img_preprocessor

    protocol = _heldout_protocol(args.dataset, args.manifest)
    measurement = _measurement1_holdout(args.measurement1_segments)
    dataset = swm.data.load_dataset(
        str(args.dataset), transform=None, num_steps=6, frameskip=5,
        keys_to_load=["pixels", "action"], keys_to_cache=["action"],
    )
    excluded = set(map(int, protocol["episodes"])) | set(map(int, measurement["episodes"]))
    allowed = next(
        index for index, value in enumerate(dataset.clip_indices) if int(value[0]) not in excluded
    )
    normalizers = _load_normalizers(args.normalizers)
    mean = torch.tensor(normalizers["action"]["mean"], dtype=torch.float32).reshape(1, 5).repeat(1, 5)
    std = torch.tensor(normalizers["action"]["std"], dtype=torch.float32).reshape(1, 5).repeat(1, 5)
    transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=224),
        ColumnNormalizer("action", mean, std),
    )
    sample = ExpertRolloutDataset(dataset, [allowed], transform)[0]
    source = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = swm.wm.utils.load_pretrained(str(args.warm_start), cache_dir=str(AILAB)).cpu().eval()
    freeze = freeze_predictor_only(model)
    before = frozen_state_hashes(model)
    target = _encode_targets(model, source["pixels"])
    with patch.object(model, "rollout", wraps=model.rollout) as rollout_spy:
        expert_teacher = _teacher_from_targets(model, target, source["action"])
        expert_rollout = expert_teacher.new_zeros(())
        expert_depth = expert_teacher.new_zeros(5)
        if rollout_spy.call_count != 0:
            raise RuntimeError("real expert branch called model.rollout")
        v2_teacher = _teacher_from_targets(model, target, source["action"])
        v2_rollout, v2_depth = _v2_rollout_from_targets(
            model, source["pixels"], source["action"], target
        )
        if rollout_spy.call_count != 1:
            raise RuntimeError("real V2 branch did not call model.rollout exactly once")
    loss = 0.75 * expert_teacher + 0.25 * v2_teacher + 0.25 * 0.5 * v2_rollout
    loss.backward()
    gradient = sum(
        int(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    assert_frozen_hashes(model, before)
    if gradient == 0 or not torch.isfinite(v2_depth).all():
        raise RuntimeError("real-model V3 smoke produced no trainable gradient/non-finite depth")
    return {
        "status": "PASS", "checkpoint": str(args.warm_start),
        "expert_clip_index": allowed,
        "source_episode": int(dataset.clip_indices[allowed][0]),
        "expert_teacher_loss": float(expert_teacher.detach()),
        "expert_rollout_loss": float(expert_rollout),
        "expert_rollout_depth_loss": [float(value) for value in expert_depth],
        "expert_rollout_calls": 0, "v2_rollout_calls": 1,
        "v2_teacher_loss": float(v2_teacher.detach()),
        "v2_rollout_loss": float(v2_rollout.detach()),
        "v2_rollout_depth_loss": [float(value) for value in v2_depth.detach()],
        "trainable_nonzero_gradient_elements": gradient,
        "trainable_parameter_tensors": freeze["trainable_parameter_tensors"],
        "trainable_parameters": freeze["trainable_parameters"],
        "frozen_hash_exact": True,
    }


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    args.run_id = _safe_run_id(args.run_id)
    args.dataset = _validate_data_disk(args.dataset, "expert dataset")
    args.manifest = _validate_data_disk(args.manifest, "formal manifest")
    args.measurement1_segments = _validate_data_disk(
        args.measurement1_segments, "Measurement-1 segments"
    )
    args.warm_start = _validate_data_disk(args.warm_start, "MaskedAug warm start")
    args.normalizers = _validate_data_disk(args.normalizers, "Route2.1 normalizers")
    normalizers_sha = sha256_file(args.normalizers)
    if normalizers_sha != NORMALIZERS_SHA256:
        raise RuntimeError(
            "V3 normalizers identity changed: "
            f"expected={NORMALIZERS_SHA256}, actual={normalizers_sha}, position={args.normalizers}"
        )
    args.offpolicy_root = args.offpolicy_root.expanduser().resolve()
    if AILAB.parent.resolve() not in args.offpolicy_root.parents:
        raise ValueError("V2 dataset must reside on /root/autodl-tmp")

    if args.synthetic_smoke:
        print(json.dumps(synthetic_v3_smoke(), indent=2, sort_keys=True))
        return 0
    if args.model_smoke:
        print(json.dumps(_real_model_smoke(args), indent=2, sort_keys=True))
        return 0

    protocol, measurement, collector = _load_protocol(args)
    datasets = _build_datasets(args, protocol, measurement, collector)
    if args.dry_run:
        print(json.dumps(_plan(args, protocol, measurement, collector, datasets), indent=2, sort_keys=True))
        return 0

    output_dir = OUTPUT_ROOT / args.run_id
    checkpoint_dir = CHECKPOINT_ROOT / args.run_id
    tensorboard_dir = TENSORBOARD_ROOT / args.run_id
    existing = [
        path for path in (output_dir, checkpoint_dir, tensorboard_dir)
        if path.exists() and any(path.iterdir())
    ]
    if existing:
        raise FileExistsError(
            f"V3 is a non-resumable single arm and output already exists: {existing}"
        )
    for path in (output_dir, checkpoint_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)

    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from module import SIGReg
    from omegaconf import OmegaConf

    pl.seed_everything(args.seed, workers=True)
    loader_common: dict[str, Any] = {
        "batch_size": 32, "num_workers": args.num_workers,
        "pin_memory": True, "drop_last": True,
    }
    if args.num_workers:
        loader_common.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        datasets["train"], shuffle=True, generator=generator, **loader_common
    )
    val_loader = torch.utils.data.DataLoader(datasets["val"], shuffle=False, **loader_common)
    stopline_loader = _stopline_loader(datasets, args)

    model = swm.wm.utils.load_pretrained(str(args.warm_start), cache_dir=str(AILAB))
    freeze = freeze_predictor_only(model)
    if (
        freeze["trainable_parameter_tensors"] != EXPECTED_TRAINABLE_TENSORS
        or freeze["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(
            "V3 trainable parameter contract changed: "
            f"expected=({EXPECTED_TRAINABLE_TENSORS},{EXPECTED_TRAINABLE_PARAMETERS}), "
            f"actual=({freeze['trainable_parameter_tensors']},{freeze['trainable_parameters']})"
        )
    frozen_before = frozen_state_hashes(model)
    freeze["frozen_sha256_before"] = frozen_before
    plan = _plan(args, protocol, measurement, collector, datasets, freeze)
    _write_contract(output_dir, plan, datasets)

    config_path = args.warm_start.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    model_config = OmegaConf.create(_load_json(config_path))
    optimizers = {
        "model_opt": {
            "modules": r"model\.(predictor|action_encoder|pred_proj)(?:\.|$)",
            "optimizer": {"type": "AdamW", "lr": args.learning_rate, "weight_decay": args.weight_decay},
            "scheduler": "LinearWarmupCosineAnnealingLR", "interval": "step",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(knots=17, num_proj=1024),
        forward=partial(v3_forward, cfg=None),
        optim=optimizers,
    )
    configure_manual_gradient_clipping(module, "model_opt", args.gradient_clip_val, "norm")
    callbacks, stopline_callback = V3Callbacks.create(
        run_id=args.run_id, model_config=model_config, output_dir=output_dir,
        checkpoint_dir=checkpoint_dir, frozen_before=frozen_before,
        expert_loader=stopline_loader, every=args.log_every_n_steps,
    )
    callbacks.append(ModelCheckpoint(
        dirpath=str(checkpoint_dir / "lightning"), save_last=True, save_top_k=-1,
        every_n_train_steps=1000, filename="step{step}", enable_version_counter=False,
    ))
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=args.precision,
        max_epochs=-1, max_steps=args.max_steps, callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir=str(TENSORBOARD_ROOT), name="", version=args.run_id,
            default_hp_metric=False,
        ),
        default_root_dir=str(output_dir / "lightning"),
        num_sanity_val_steps=1, limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps, enable_checkpointing=True,
    )
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    trainer.fit(module, datamodule=spt.data.DataModule(train=train_loader, val=val_loader))

    final_checkpoint = checkpoint_dir / "weights_final.pt"
    if not final_checkpoint.is_file():
        raise FileNotFoundError(final_checkpoint)
    posthoc = _posthoc_payload(args, stopline_callback, final_checkpoint)
    history_path = output_dir / "stopline_history.json"
    history = stopline_callback.history
    if history is None:
        raise RuntimeError("V3 stopline history disappeared")
    stopped = bool(history["triggered"])
    final_relative = float(history["records"][-1]["relative"])
    passed = not stopped and int(trainer.global_step) == args.max_steps and final_relative <= STOPLINE_THRESHOLD
    if not stopped and not passed:
        raise RuntimeError(
            "V3 ended without a stopline trigger or a complete PASS: "
            f"step={trainer.global_step}, relative={final_relative}"
        )
    completed = {
        "format_version": "cube_offpolicy_v3_completed_v1",
        "status": "STOPLINE_FAIL" if stopped else "PASS",
        "offline_gate_authorized": passed,
        "run_id": args.run_id,
        "arm": CANONICAL_ARM,
        "global_step": int(trainer.global_step),
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - started,
        "final_weights": str(final_checkpoint.resolve()),
        "final_weights_sha256": sha256_file(final_checkpoint),
        "stopped_weights": (
            file_identity(stopline_callback.stopped_weights)
            if stopline_callback.stopped_weights is not None else None
        ),
        "live_stopline_checkpoint": (
            file_identity(stopline_callback.live_checkpoint)
            if stopline_callback.live_checkpoint is not None else None
        ),
        "source_mix": {"expert": FORMAL_EXPERT_BATCH, "v2": FORMAL_V2_BATCH},
        "source_examples_seen": dict(getattr(module, "_v3_source_examples", {})),
        "cross_split_episode_overlaps": dict(datasets["cross_split_episode_overlaps"]),
        "masked_augmentation_runtime": _runtime_mask_stats(module),
        "frozen_integrity": {
            "before": frozen_before,
            "after": assert_frozen_hashes(model, frozen_before),
            "exact_match": True,
        },
        "loss_contract": dict(LOSS_CONTRACT),
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "stopline_history": file_identity(history_path),
        "stopline_event": (
            file_identity(stopline_callback.event_path)
            if stopline_callback.event_path is not None else None
        ),
        "expert_stopline": posthoc["expert_stopline"],
    }
    atomic_write_json(output_dir / "completed.json", completed)
    print(json.dumps({
        "status": completed["status"],
        "offline_gate_authorized": completed["offline_gate_authorized"],
        "global_step": completed["global_step"],
        "expert_stopline": completed["expert_stopline"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 3


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=FORMAL_RUN_ID)
    parser.add_argument("--dataset", type=Path, default=EXPERT_DATASET)
    parser.add_argument("--manifest", type=Path, default=FORMAL_MANIFEST)
    parser.add_argument("--measurement1-segments", type=Path, default=MEASUREMENT1_SEGMENTS)
    parser.add_argument("--offpolicy-root", type=Path, default=V2_ROOT)
    parser.add_argument("--warm-start", type=Path, default=WARM_WEIGHTS)
    parser.add_argument("--normalizers", type=Path, default=NORMALIZERS)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--hue-probability", type=float, default=0.8)
    parser.add_argument("--max-hue-delta", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16-mixed",), default="bf16-mixed")
    parser.add_argument("--limit-val-batches", type=int, default=25)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--model-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser_ = parser()
    args = parser_.parse_args(argv)
    modes = sum((args.synthetic_smoke, args.model_smoke, args.dry_run))
    if modes > 1:
        parser_.error("synthetic/model/dry-run modes are mutually exclusive")
    if not modes and args.run_id != FORMAL_RUN_ID:
        parser_.error(
            f"formal V3 is one non-retry arm; --run-id must be exactly {FORMAL_RUN_ID!r}"
        )
    frozen = {
        "--batch-size": (args.batch_size, 128),
        "--seed": (args.seed, 3072),
        "--split-seed": (args.split_seed, 3072),
        "--train-fraction": (args.train_fraction, 0.9),
        "--num-workers": (args.num_workers, 6),
        "--prefetch-factor": (args.prefetch_factor, 3),
        "--hue-probability": (args.hue_probability, 0.8),
        "--max-hue-delta": (args.max_hue_delta, 0.5),
        "--learning-rate": (args.learning_rate, 1e-5),
        "--weight-decay": (args.weight_decay, 1e-3),
        "--gradient-clip-val": (args.gradient_clip_val, 1.0),
        "--limit-val-batches": (args.limit_val_batches, 25),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (actual, expected) in frozen.items() if actual != expected
    }
    if mismatches:
        parser_.error(f"V3 frozen V2 contract mismatch: {mismatches}")
    if args.max_steps != 5000 and not (args.synthetic_smoke or args.model_smoke):
        parser_.error("--max-steps is frozen at 5000")
    if not modes and not torch.cuda.is_available():
        parser_.error("formal V3 training requires CUDA bf16")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
