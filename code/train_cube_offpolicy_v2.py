#!/usr/bin/env python3
"""Fine-tune Cube dynamics on real T2 planner rollouts with five-step loss.

The formal arm uses 96 cleanly sourced expert samples and 32 planner-in-the-
loop samples on every 128-example optimizer step.  A single permitted retry
uses 104/24.  Encoder and projector are immutable; the recurrent five-step
loss backpropagates through every imagined state without intermediate detach.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import hdf5plugin  # noqa: F401
import numpy as np
import torch

from cube_maskedaug import MASK_METADATA_PREFIX, RandomMaskedHueRotation
from cube_offpolicy_v2 import (
    FORMAL_BATCH_SIZE,
    FORMAL_EXPERT_BATCH,
    FORMAL_V2_BATCH,
    RETRY_EXPERT_BATCH,
    RETRY_V2_BATCH,
    ExpertRolloutDataset,
    PlannerRolloutDataset,
    StrictMixtureDataset,
    assert_frozen_hashes,
    flatten_bundled_source,
    freeze_predictor_only,
    frozen_state_hashes,
    load_planner_manifest,
    module_state_sha256,
    sha256_file,
    synthetic_v2_smoke,
    write_json,
)
from train_cube_coloraug import (
    ColumnNormalizer,
    _configure_storage,
    _heldout_protocol,
    _safe_run_id,
    _validate_data_disk,
    configure_manual_gradient_clipping,
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
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-offpolicy_v2"
OUTPUT_ROOT = AILAB / "outputs/train/offpolicy_v2"
TENSORBOARD_ROOT = AILAB / "logs/tensorboard/offpolicy_v2"
NUM_FRAMES = 6
NUM_ACTIONS = 5
TEACHER_FRAMES = 4
TEACHER_ACTIONS = 3
HISTORY_SIZE = 3
MASK_STAT_NAMES = ("empty_frames", "total_frames", "masked_pixels", "applied_clips", "seen_clips")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_normalizers(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    for key in ("action", "observation"):
        if key not in value or "mean" not in value[key] or "std" not in value[key]:
            raise ValueError(f"normalizers lack {key} mean/std")
    return value


def _measurement1_holdout(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    episodes = np.asarray(payload.get("episode_indices"), dtype=np.int64)
    starts = np.asarray(payload.get("start_rows"), dtype=np.int64)
    if episodes.shape != (2000,) or starts.shape != (2000,):
        raise ValueError(
            "Measurement-1 frozen segment shape mismatch: "
            f"expected=(2000,), actual={episodes.shape}/{starts.shape}"
        )
    unique = np.unique(episodes)
    formal = set(map(int, payload.get("formal_episodes_excluded", [])))
    if len(unique) != 1801 or len(formal) != 50 or formal & set(map(int, unique)):
        raise ValueError("Measurement-1 holdout identity/exclusion contract changed")
    return {
        "episodes": unique,
        "num_segments": len(episodes),
        "num_episodes": len(unique),
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _encode_targets(model: Any, pixels: torch.Tensor) -> torch.Tensor:
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


def _source_losses(
    model: Any, source: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact original 4-frame teacher loss plus formal model.rollout loss."""
    pixels = source["pixels"]
    action = source["action"]
    if tuple(action.shape[1:]) != (5, 25) or not torch.isfinite(action).all():
        raise ValueError(f"expected finite action [B,5,25], actual={tuple(action.shape)}")
    target = _encode_targets(model, pixels)

    # Preserve the exact original JEPA teacher-forced objective: four frames,
    # three action blocks, prediction targets t+1..t+3.
    teacher_action_embedding = model.action_encoder(action[:, :TEACHER_ACTIONS])
    teacher_prediction = model.predict(
        target[:, :TEACHER_ACTIONS], teacher_action_embedding
    )
    teacher_loss = (teacher_prediction - target[:, 1:TEACHER_FRAMES].detach()).square().mean()

    # Use the production inference method, not an approximate hand-written
    # loop.  It emits the encoded initial state followed by five autoregressive
    # predictions.  No predicted state is detached inside JEPA.rollout.
    rolled = model.rollout(
        {
            "pixels": pixels[:, None, :1],
            # Reuse the exact frozen target encoding.  LeWM.rollout's public
            # cache path is the production path and avoids a redundant ViT
            # pass without changing any imagined predictor state.
            "emb": target[:, None, :1].detach(),
        },
        action[:, None],
        history_size=HISTORY_SIZE,
    )["predicted_emb"][:, 0, 1:]
    if tuple(rolled.shape[:2]) != (pixels.shape[0], 5):
        raise RuntimeError(f"model.rollout returned unexpected shape {tuple(rolled.shape)}")
    squared = (rolled - target[:, 1:].detach()).square()
    depth_losses = squared.mean(dim=(0, 2))
    rollout_loss = depth_losses.mean()
    return teacher_loss, rollout_loss, depth_losses, target


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


def v2_forward(self: Any, batch: dict[str, Any], stage: str, cfg: Any) -> dict[str, torch.Tensor]:
    data_keys = set(batch) - {"batch_idx"}
    if data_keys != {"expert", "v2"}:
        raise ValueError(f"unexpected V2 batch keys: {sorted(batch)}")
    expert = flatten_bundled_source(batch["expert"])
    v2 = flatten_bundled_source(batch["v2"])
    expected = {"expert": int(cfg.expert_batch), "v2": int(cfg.v2_batch)}
    actual = {"expert": int(expert["pixels"].shape[0]), "v2": int(v2["pixels"].shape[0])}
    if actual != expected or sum(actual.values()) != FORMAL_BATCH_SIZE:
        raise RuntimeError(f"strict V2 source batch mismatch: expected={expected}, actual={actual}")

    mask_stats: dict[str, torch.Tensor] = {}
    for name in MASK_STAT_NAMES:
        key = f"{MASK_METADATA_PREFIX}{name}"
        if key in expert:
            mask_stats[name] = expert.pop(key).sum().detach()
    if stage in {"fit", "train"}:
        counts = getattr(self, "_v2_source_examples", {"expert": 0, "v2": 0})
        counts["expert"] += expected["expert"]
        counts["v2"] += expected["v2"]
        self._v2_source_examples = counts
        totals = getattr(self, "_v2_mask_totals", {name: 0 for name in MASK_STAT_NAMES})
        for name, value in mask_stats.items():
            totals[name] = totals[name] + value
        self._v2_mask_totals = totals

    expert_pred, expert_roll, expert_depth, expert_embedding = _source_losses(self.model, expert)
    v2_pred, v2_roll, v2_depth, v2_embedding = _source_losses(self.model, v2)
    expert_weight = expected["expert"] / FORMAL_BATCH_SIZE
    v2_weight = expected["v2"] / FORMAL_BATCH_SIZE
    pred_loss = expert_weight * expert_pred + v2_weight * v2_pred
    rollout_loss = expert_weight * expert_roll + v2_weight * v2_roll
    sigreg_loss, expert_sigreg, v2_sigreg = _sigreg_shared_projection(
        self.sigreg, expert_embedding, v2_embedding
    )
    loss = pred_loss + float(cfg.sigreg_weight) * sigreg_loss + float(cfg.rollout_weight) * rollout_loss
    metrics: dict[str, torch.Tensor] = {
        "loss": loss,
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "rollout_loss": rollout_loss,
        "expert_pred_loss": expert_pred,
        "v2_pred_loss": v2_pred,
        "expert_sigreg_loss": expert_sigreg,
        "v2_sigreg_loss": v2_sigreg,
        "expert_rollout_loss": expert_roll,
        "v2_rollout_loss": v2_roll,
    }
    for depth in range(5):
        metrics[f"expert_rollout_depth{depth + 1}_loss"] = expert_depth[depth]
        metrics[f"v2_rollout_depth{depth + 1}_loss"] = v2_depth[depth]
    if not all(bool(torch.isfinite(value.detach()).item()) for value in metrics.values()):
        raise FloatingPointError(
            "V2 non-finite loss: " + ", ".join(f"{k}={v.detach()}" for k, v in metrics.items())
        )
    self.log_dict(
        {f"{stage}/{key}": value.detach() for key, value in metrics.items()},
        on_step=True,
        sync_dist=True,
    )
    self._v2_last_metrics = {key: float(value.detach().float().cpu()) for key, value in metrics.items()}
    return metrics


def _global_episode_split(
    dataset: Any,
    *,
    v2_source_by_rollout: Sequence[int],
    excluded: set[int],
    train_fraction: float,
    seed: int,
) -> dict[str, np.ndarray | dict[str, int]]:
    """One global episode assignment mapped onto both expert and V2 sources."""
    clip_episodes = np.fromiter(
        (int(value[0]) for value in dataset.clip_indices), dtype=np.int64, count=len(dataset.clip_indices)
    )
    eligible_episodes = np.setdiff1d(np.unique(clip_episodes), np.fromiter(excluded, dtype=np.int64))
    v2_source = np.asarray(v2_source_by_rollout, dtype=np.int64)
    missing_v2 = np.setdiff1d(np.unique(v2_source), eligible_episodes)
    if missing_v2.size:
        raise RuntimeError(f"V2 source episodes are outside global eligible set: {missing_v2.tolist()}")
    shuffled = np.random.default_rng(seed).permutation(eligible_episodes)
    count = min(max(1, int(np.floor(len(shuffled) * train_fraction))), len(shuffled) - 1)
    global_train = np.sort(shuffled[:count])
    global_val = np.sort(shuffled[count:])
    expert_train_ids = np.flatnonzero(np.isin(clip_episodes, global_train)).astype(np.int64)
    expert_val_ids = np.flatnonzero(np.isin(clip_episodes, global_val)).astype(np.int64)
    v2_train_ids = np.flatnonzero(np.isin(v2_source, global_train)).astype(np.int64)
    v2_val_ids = np.flatnonzero(np.isin(v2_source, global_val)).astype(np.int64)
    if not all(len(value) for value in (expert_train_ids, expert_val_ids, v2_train_ids, v2_val_ids)):
        raise RuntimeError("global episode split produced an empty expert/V2 partition")
    expert_train_eps = np.unique(clip_episodes[expert_train_ids])
    expert_val_eps = np.unique(clip_episodes[expert_val_ids])
    v2_train_eps = np.unique(v2_source[v2_train_ids])
    v2_val_eps = np.unique(v2_source[v2_val_ids])
    overlaps = {
        "expert_train_vs_expert_validation": int(
            np.intersect1d(expert_train_eps, expert_val_eps).size
        ),
        "v2_train_vs_v2_validation": int(np.intersect1d(v2_train_eps, v2_val_eps).size),
        "expert_train_vs_v2_validation": int(
            np.intersect1d(expert_train_eps, v2_val_eps).size
        ),
        "expert_validation_vs_v2_train": int(
            np.intersect1d(expert_val_eps, v2_train_eps).size
        ),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"global cross-source split leakage: {overlaps}")
    if (set(map(int, np.concatenate((expert_train_eps, expert_val_eps)))) & excluded):
        raise RuntimeError("excluded episode leaked into expert split")
    return {
        "expert_train_ids": expert_train_ids,
        "expert_val_ids": expert_val_ids,
        "expert_train_episodes": expert_train_eps,
        "expert_val_episodes": expert_val_eps,
        "v2_train_ids": v2_train_ids,
        "v2_val_ids": v2_val_ids,
        "v2_train_episodes": v2_train_eps,
        "v2_val_episodes": v2_val_eps,
        "global_train_episodes": global_train,
        "global_val_episodes": global_val,
        "cross_split_episode_overlaps": overlaps,
    }


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
        str(args.dataset),
        transform=None,
        num_steps=NUM_FRAMES,
        frameskip=5,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
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
    expert_val_transform = spt.data.transforms.Compose(
        preprocessor, ColumnNormalizer("action", mean, std)
    )
    expert_train = ExpertRolloutDataset(
        dataset, split["expert_train_ids"], expert_train_transform
    )
    expert_val = ExpertRolloutDataset(dataset, split["expert_val_ids"], expert_val_transform)
    v2_train = PlannerRolloutDataset(collector, split["v2_train_ids"], preprocessor)
    v2_val = PlannerRolloutDataset(collector, split["v2_val_ids"], preprocessor)
    per_bundle = (13, 3) if args.retry else (3, 1)
    bundle_batch = 8 if args.retry else 32
    return {
        "train": StrictMixtureDataset(expert_train, v2_train, *per_bundle),
        "val": StrictMixtureDataset(expert_val, v2_val, *per_bundle),
        "expert_val": expert_val,
        "v2_val": v2_val,
        **split,
        "bundle": {"expert": per_bundle[0], "v2": per_bundle[1], "loader_batch": bundle_batch},
    }


def _source_batch(args: argparse.Namespace) -> tuple[int, int]:
    return (RETRY_EXPERT_BATCH, RETRY_V2_BATCH) if args.retry else (
        FORMAL_EXPERT_BATCH,
        FORMAL_V2_BATCH,
    )


def _runtime_mask_stats(module: Any) -> dict[str, int | float]:
    totals = getattr(module, "_v2_mask_totals", {})
    result = {
        name: int(totals[name].detach().cpu()) if torch.is_tensor(totals.get(name)) else int(totals.get(name, 0))
        for name in MASK_STAT_NAMES
    }
    result["empty_frame_rate"] = (
        result["empty_frames"] / result["total_frames"] if result["total_frames"] else 0.0
    )
    return result


def _plan(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    measurement: Mapping[str, Any],
    collector: Mapping[str, Any],
    datasets: Mapping[str, Any] | None = None,
    freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expert_batch, v2_batch = _source_batch(args)
    split = {}
    if datasets is not None:
        split = {
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
            "assignment": "one global episode split, then mapped to expert and V2",
            "cross_split_episode_overlaps": dict(
                datasets["cross_split_episode_overlaps"]
            ),
        }
    return {
        "format_version": "cube_offpolicy_train_v2",
        "run_id": args.run_id,
        "arm": "retry_104E_24V2" if args.retry else "formal_96E_32V2",
        "decision_record": {
            "planner_data_only": True,
            "v1_mixed": False,
            "reason": "isolate real ten-iteration T2 distribution and protect the expert manifold",
            "retry_reason": "single allowed expert-protection retry at 81.25% expert" if args.retry else None,
            "train_scope": "predictor + action_encoder + pred_proj; encoder + projector frozen",
            "measurement1_holdout_removed_from_expert_and_collector": True,
        },
        "inputs": {
            "expert_dataset": str(args.dataset),
            "formal_manifest": str(args.manifest),
            "formal_manifest_sha256": protocol["manifest_sha256"],
            "measurement1_segments": {
                "path": measurement["path"],
                "sha256": measurement["sha256"],
                "num_segments": int(measurement["num_segments"]),
                "num_episodes": int(measurement["num_episodes"]),
                "episode_ids": [int(value) for value in measurement["episodes"]],
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
        },
        "formal_episode_exclusion": {
            "count": len(protocol["episodes"]),
            "episode_ids": sorted(map(int, protocol["episodes"])),
        },
        "splits": split,
        "batch": {
            "total": FORMAL_BATCH_SIZE,
            "expert": expert_batch,
            "v2": v2_batch,
            "expert_fraction": expert_batch / FORMAL_BATCH_SIZE,
            "v2_fraction": v2_batch / FORMAL_BATCH_SIZE,
            "strict_every_step": True,
        },
        "loss": {
            "formula": "sample-weighted teacher_pred + 0.09*SIGReg + 0.5*five_step_rollout",
            "teacher_forced": "exact original first-four-frame/three-action JEPA loss",
            "rollout": "JEPA.rollout from frame t through 5 actions; target encoder stop-grad; intermediate predictions attached",
            "sigreg_weight": args.sigreg_weight,
            "rollout_weight": args.rollout_weight,
            "depths": [1, 2, 3, 4, 5],
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "steps": args.max_steps,
            "precision": args.precision,
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": max(1, int(0.01 * args.max_steps)),
            "gradient_clip_norm": args.gradient_clip_val,
            "checkpoints_every_steps": 1000,
        },
        "expert_augmentation": {
            "masked_hue_probability": args.hue_probability,
            "max_hue_delta_turns": args.max_hue_delta,
            "validation": "clean identity",
        },
        "stopline": {
            "metric": "paired clean expert original teacher-forced pred loss",
            "maximum_relative_increase": args.expert_stopline,
            "action_if_failed": "do not enter offline gate; run one --retry arm, otherwise stop",
        },
        "runtime": {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "num_workers": args.num_workers,
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
            "dataset": sha256_file(Path(__file__).with_name("cube_offpolicy_v2.py")),
        },
    }


def _first_difference(expected: Any, actual: Any, path: str = "root") -> tuple[str, Any, Any] | None:
    if type(expected) is not type(actual):
        return path, expected, actual
    if isinstance(expected, Mapping):
        if set(expected) != set(actual):
            return f"{path}.keys", sorted(expected), sorted(actual)
        for key in sorted(expected):
            found = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}.length", len(expected), len(actual)
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            found = _first_difference(left, right, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if expected == actual else (path, expected, actual)


def _write_or_verify_contract(
    output: Path, plan: Mapping[str, Any], datasets: Mapping[str, Any], resume: bool
) -> None:
    plan_path = output / "run_plan.json"
    split_path = output / "episode_split.npz"
    arrays = {
        name: np.asarray(datasets[name], dtype=np.int64)
        for name in (
            "global_train_episodes", "global_val_episodes",
            "expert_train_episodes", "expert_val_episodes", "v2_train_episodes", "v2_val_episodes",
            "v2_train_ids", "v2_val_ids",
        )
    }
    if not resume:
        write_json(plan_path, plan)
        np.savez_compressed(split_path, **arrays)
        return
    if not plan_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("V2 resume contract artifacts are missing")
    difference = _first_difference(_load_json(plan_path), dict(plan))
    if difference:
        position, expected, actual = difference
        raise RuntimeError(
            f"V2 resume plan mismatch: expected={expected!r}, actual={actual!r}, position={position!r}"
        )
    with np.load(split_path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise RuntimeError("V2 resume split keys changed")
        for name, value in arrays.items():
            if not np.array_equal(saved[name], value):
                raise RuntimeError(f"V2 resume split changed at {name}")


class V2Callbacks:
    @staticmethod
    def create(
        run_id: str,
        model_config: Any,
        output_dir: Path,
        frozen_before: Mapping[str, str],
        every: int,
    ) -> list[Any]:
        from lightning.pytorch.callbacks import Callback
        from stable_worldmodel.wm.utils import save_pretrained

        class Export(Callback):
            def _save(self, trainer: Any, module: Any, filename: str) -> None:
                if trainer.is_global_zero:
                    assert_frozen_hashes(module.model, frozen_before)
                    save_pretrained(
                        module.model,
                        run_name=f"lewm-cube-offpolicy_v2/{run_id}",
                        config=model_config,
                        filename=filename,
                        cache_dir=str(AILAB),
                    )

            def on_train_end(self, trainer: Any, module: Any) -> None:
                self._save(trainer, module, "weights_final.pt")
                if trainer.is_global_zero:
                    after = assert_frozen_hashes(module.model, frozen_before)
                    write_json(output_dir / "frozen_integrity.json", {
                        "status": "PASS", "before": dict(frozen_before), "after": after,
                        "exact_match": after == dict(frozen_before),
                    })

        class Curves(Callback):
            def on_train_batch_end(
                self, trainer: Any, module: Any, outputs: Any, batch: Any, batch_idx: int
            ) -> None:
                step = int(trainer.global_step)
                if not trainer.is_global_zero or not (step == 1 or step == trainer.max_steps or step % every == 0):
                    return
                values = dict(getattr(module, "_v2_last_metrics", {}))
                values.update({
                    "step": step,
                    "epoch": int(trainer.current_epoch),
                    "batch_idx": int(batch_idx),
                    "learning_rate": float(trainer.optimizers[0].param_groups[0]["lr"]),
                })
                with (output_dir / "loss_curve.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(values, sort_keys=True) + "\n")

        return [Export(), Curves()]


def _move_source(source: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in source.items()
    }


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    expert_loader: Any,
    v2_loader: Any,
    args: argparse.Namespace,
    label: str,
    actual_batches: int,
) -> dict[str, Any]:
    import stable_worldmodel as swm

    device = torch.device(args.validation_device)
    checkpoint_sha = sha256_file(checkpoint)
    model = swm.wm.utils.load_pretrained(str(checkpoint), cache_dir=str(AILAB)).to(device).eval()
    freeze_predictor_only(model)
    state_before = module_state_sha256(model)
    source_loaders = {"expert": expert_loader, "v2": v2_loader}
    result: dict[str, Any] = {}
    with torch.inference_mode():
        for source_name, loader in source_loaders.items():
            sums = {"teacher_pred_loss": 0.0, "rollout_loss": 0.0}
            sums.update({f"rollout_depth{depth}_loss": 0.0 for depth in range(1, 6)})
            provenance = hashlib.sha256()
            rows = []
            for batch_index, batch in enumerate(loader):
                if batch_index >= actual_batches:
                    break
                ids = batch["expert_clip_index"] if source_name == "expert" else batch["rollout_id"]
                provenance.update(np.asarray(ids, dtype=np.int64).tobytes())
                source = _move_source(batch, device)
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" and args.validation_precision == "bf16"
                    else nullcontext()
                )
                with context:
                    teacher, rollout, depth, _ = _source_losses(model, source)
                values = {
                    "teacher_pred_loss": float(teacher.float().cpu()),
                    "rollout_loss": float(rollout.float().cpu()),
                    **{f"rollout_depth{index + 1}_loss": float(value.float().cpu()) for index, value in enumerate(depth)},
                }
                if not all(np.isfinite(value) for value in values.values()):
                    raise FloatingPointError(f"posthoc {label}/{source_name} non-finite at {batch_index}")
                for name, value in values.items():
                    sums[name] += value
                rows.append({"batch": batch_index, **values})
            if len(rows) != actual_batches:
                raise RuntimeError(
                    f"posthoc batch count mismatch: expected={actual_batches}, actual={len(rows)}, "
                    f"position={label}/{source_name}"
                )
            result[source_name] = {
                "num_batches": len(rows),
                "examples": len(rows) * FORMAL_BATCH_SIZE,
                "provenance_sha256": provenance.hexdigest(),
                "mean": {name: value / len(rows) for name, value in sums.items()},
                "batches": rows,
            }
    state_after = module_state_sha256(model)
    if state_after != state_before or sha256_file(checkpoint) != checkpoint_sha:
        raise RuntimeError(f"read-only posthoc mutated model/checkpoint for {label}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "model_state_sha256": state_before,
        "sources": result,
    }


def _posthoc_loaders(datasets: Mapping[str, Any], args: argparse.Namespace) -> tuple[Any, Any]:
    workers = args.num_workers if args.validation_num_workers is None else args.validation_num_workers
    common: dict[str, Any] = {
        "batch_size": FORMAL_BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "num_workers": workers,
        "pin_memory": args.validation_device == "cuda",
    }
    if workers:
        common.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    return (
        torch.utils.data.DataLoader(datasets["expert_val"], **common),
        torch.utils.data.DataLoader(datasets["v2_val"], **common),
    )


def _actual_posthoc_batches(requested: int, available: Mapping[str, int]) -> int:
    if requested < 1 or set(available) != {"expert", "v2"}:
        raise ValueError(
            f"invalid posthoc batch contract: requested={requested}, available={dict(available)}"
        )
    actual = min(int(requested), *(int(value) for value in available.values()))
    if actual < 1:
        raise RuntimeError(
            "paired posthoc has no full batch: "
            f"requested={requested}, available={dict(available)}, batch_size={FORMAL_BATCH_SIZE}"
        )
    return actual


def _run_posthoc(
    args: argparse.Namespace,
    datasets: Mapping[str, Any],
    final_checkpoint: Path,
    output_path: Path,
) -> dict[str, Any]:
    expert_loader, v2_loader = _posthoc_loaders(datasets, args)
    available = {"expert": len(expert_loader), "v2": len(v2_loader)}
    actual_batches = _actual_posthoc_batches(args.validation_batches, available)
    base = _evaluate_checkpoint(
        args.warm_start,
        expert_loader=expert_loader,
        v2_loader=v2_loader,
        args=args,
        label="masked_base",
        actual_batches=actual_batches,
    )
    final = _evaluate_checkpoint(
        final_checkpoint,
        expert_loader=expert_loader,
        v2_loader=v2_loader,
        args=args,
        label="v2_final",
        actual_batches=actual_batches,
    )
    for source in ("expert", "v2"):
        if base["sources"][source]["provenance_sha256"] != final["sources"][source]["provenance_sha256"]:
            raise RuntimeError(f"base/final posthoc {source} batches differ")
    base_pred = base["sources"]["expert"]["mean"]["teacher_pred_loss"]
    final_pred = final["sources"]["expert"]["mean"]["teacher_pred_loss"]
    increase = final_pred / base_pred - 1.0
    passed = increase <= args.expert_stopline
    payload = {
        "format_version": "cube_offpolicy_v2_paired_posthoc_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "read_only": True,
        "protocol": {
            "clean_expert": True,
            "paired_base_final": True,
            "requested_batches": args.validation_batches,
            "available_full_batches": available,
            "actual_batches": actual_batches,
            "validation_examples_per_source": actual_batches * FORMAL_BATCH_SIZE,
            "sampling": "finite loader prefix without cycling",
            "metric": "exact original four-frame teacher-forced pred loss",
        },
        "base": base,
        "final": final,
        "expert_stopline": {
            "threshold_relative_increase": args.expert_stopline,
            "base_teacher_pred_loss": base_pred,
            "final_teacher_pred_loss": final_pred,
            "relative_increase": increase,
            "status": "PASS" if passed else "FAIL",
            "offline_gate_authorized": passed,
        },
    }
    write_json(output_path, payload)
    return payload


def _safe_posthoc_name(value: str) -> str:
    if not re.fullmatch(r"posthoc_validation(?:[._-][A-Za-z0-9][A-Za-z0-9._-]*)?\.json", value):
        raise ValueError("posthoc output must be a safe posthoc_validation*.json basename")
    return value


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


def _real_model_smoke(args: argparse.Namespace) -> dict[str, Any]:
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
    freeze_predictor_only(model)
    before = frozen_state_hashes(model)
    teacher, rollout, depth, _ = _source_losses(model, source)
    (teacher + args.rollout_weight * rollout).backward()
    gradient = sum(
        int(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    assert_frozen_hashes(model, before)
    if gradient == 0 or not torch.isfinite(depth).all():
        raise RuntimeError("real-model V2 smoke produced no trainable gradient/non-finite depth")
    return {
        "status": "PASS",
        "checkpoint": str(args.warm_start),
        "expert_clip_index": allowed,
        "source_episode": int(dataset.clip_indices[allowed][0]),
        "teacher_pred_loss": float(teacher.detach()),
        "rollout_loss": float(rollout.detach()),
        "rollout_depth_loss": [float(value) for value in depth.detach()],
        "trainable_nonzero_gradient_elements": gradient,
        "frozen_hash_exact": True,
    }


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    args.run_id = _safe_run_id(args.run_id)
    args.dataset = _validate_data_disk(args.dataset, "expert dataset")
    args.manifest = _validate_data_disk(args.manifest, "formal manifest")
    args.measurement1_segments = _validate_data_disk(args.measurement1_segments, "Measurement-1 segments")
    args.warm_start = _validate_data_disk(args.warm_start, "MaskedAug warm start")
    args.normalizers = _validate_data_disk(args.normalizers, "Route2.1 normalizers")
    args.offpolicy_root = args.offpolicy_root.expanduser().resolve()
    if AILAB.parent.resolve() not in args.offpolicy_root.parents:
        raise ValueError("V2 dataset must reside on /root/autodl-tmp")

    if args.synthetic_smoke:
        print(json.dumps(synthetic_v2_smoke(), indent=2, sort_keys=True))
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
    final_checkpoint = checkpoint_dir / "weights_final.pt"
    if args.validate_only:
        if not final_checkpoint.is_file():
            raise FileNotFoundError(final_checkpoint)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = _run_posthoc(
            args, datasets, final_checkpoint, output_dir / _safe_posthoc_name(args.posthoc_name)
        )
        print(json.dumps(payload["expert_stopline"], indent=2, sort_keys=True))
        return 0 if payload["expert_stopline"]["status"] == "PASS" else 3

    existing = [path for path in (output_dir, checkpoint_dir, tensorboard_dir) if path.exists() and any(path.iterdir())]
    if existing and not args.resume:
        raise FileExistsError(f"V2 run output already exists: {existing}")
    resume_path = checkpoint_dir / "lightning/last.ckpt"
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    for path in (output_dir, checkpoint_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)

    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from omegaconf import OmegaConf
    from module import SIGReg

    pl.seed_everything(args.seed, workers=True)
    expert_batch, v2_batch = _source_batch(args)
    loader_common: dict[str, Any] = {
        "batch_size": datasets["bundle"]["loader_batch"],
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers:
        loader_common.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        datasets["train"], shuffle=True, generator=generator, **loader_common
    )
    val_loader = torch.utils.data.DataLoader(datasets["val"], shuffle=False, **loader_common)

    model = swm.wm.utils.load_pretrained(str(args.warm_start), cache_dir=str(AILAB))
    freeze = freeze_predictor_only(model)
    frozen_before = frozen_state_hashes(model)
    freeze["frozen_sha256_before"] = frozen_before
    plan = _plan(args, protocol, measurement, collector, datasets, freeze)
    _write_or_verify_contract(output_dir, plan, datasets, args.resume)

    config_path = args.warm_start.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    model_config = OmegaConf.create(_load_json(config_path))
    forward_cfg = OmegaConf.create({
        "expert_batch": expert_batch,
        "v2_batch": v2_batch,
        "sigreg_weight": args.sigreg_weight,
        "rollout_weight": args.rollout_weight,
    })
    optimizers = {
        "model_opt": {
            "modules": r"model\.(predictor|action_encoder|pred_proj)(?:\.|$)",
            "optimizer": {"type": "AdamW", "lr": args.learning_rate, "weight_decay": args.weight_decay},
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "interval": "step",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(knots=17, num_proj=1024),
        forward=partial(v2_forward, cfg=forward_cfg),
        optim=optimizers,
    )
    configure_manual_gradient_clipping(module, "model_opt", args.gradient_clip_val, "norm")
    callbacks = V2Callbacks.create(
        args.run_id, model_config, output_dir, frozen_before, args.log_every_n_steps
    )
    callbacks.append(ModelCheckpoint(
        dirpath=str(checkpoint_dir / "lightning"), save_last=True, save_top_k=-1,
        every_n_train_steps=1000, filename="step{step}", enable_version_counter=False,
    ))
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        max_epochs=-1,
        max_steps=args.max_steps,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir=str(TENSORBOARD_ROOT), name="", version=args.run_id, default_hp_metric=False
        ),
        default_root_dir=str(output_dir / "lightning"),
        num_sanity_val_steps=1,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
    )
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    trainer.fit(
        module,
        datamodule=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=str(resume_path) if args.resume else None,
    )
    completed = {
        "run_id": args.run_id,
        "arm": plan["arm"],
        "global_step": int(trainer.global_step),
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - started,
        "final_weights": str(final_checkpoint.resolve()),
        "final_weights_sha256": sha256_file(final_checkpoint),
        "source_mix": {"expert": expert_batch, "v2": v2_batch},
        "source_examples_seen": dict(getattr(module, "_v2_source_examples", {})),
        "cross_split_episode_overlaps": dict(
            datasets["cross_split_episode_overlaps"]
        ),
        "masked_augmentation_runtime": _runtime_mask_stats(module),
        "frozen_integrity": {
            "before": frozen_before,
            "after": assert_frozen_hashes(model, frozen_before),
            "exact_match": True,
        },
    }
    write_json(output_dir / "completed.json", completed)
    posthoc = _run_posthoc(
        args, datasets, final_checkpoint, output_dir / "posthoc_validation.json"
    )
    completed["expert_stopline"] = posthoc["expert_stopline"]
    write_json(output_dir / "completed.json", completed)
    print(json.dumps(posthoc["expert_stopline"], indent=2, sort_keys=True))
    return 0 if posthoc["expert_stopline"]["status"] == "PASS" else 3


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="offpolicy_v2_pred_seed3072")
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
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--rollout-weight", type=float, default=0.5)
    parser.add_argument("--expert-stopline", type=float, default=0.10)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16-mixed", "32-true"), default="bf16-mixed")
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--limit-val-batches", type=int, default=25)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--validation-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--validation-precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--validation-num-workers", type=int)
    parser.add_argument("--posthoc-name", default="posthoc_validation.json")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser_ = parser()
    args = parser_.parse_args(argv)
    modes = sum((args.synthetic_smoke, args.model_smoke, args.dry_run, args.validate_only))
    if modes > 1 or args.resume and modes:
        parser_.error("smoke/dry-run/validate-only modes are mutually exclusive with resume and each other")
    if args.retry and args.run_id == "offpolicy_v2_pred_seed3072":
        args.run_id = "offpolicy_v2_retry_seed3072"
    if args.batch_size != 128:
        parser_.error("--batch-size is frozen at 128")
    trainer_smoke = args.max_steps == 2 and "smoke" in args.run_id.lower()
    if not 4000 <= args.max_steps <= 6000 and not (
        args.synthetic_smoke or args.model_smoke or trainer_smoke
    ):
        parser_.error("--max-steps must remain in the frozen 4k-6k budget")
    if not 0.0 < args.train_fraction < 1.0:
        parser_.error("--train-fraction must be in (0,1)")
    if args.rollout_weight <= 0 or args.sigreg_weight < 0:
        parser_.error("loss weights are invalid")
    if args.expert_stopline != 0.10:
        parser_.error("expert stopline is frozen at 10%")
    if args.validation_batches < 1 or args.num_workers < 0:
        parser_.error("validation/worker counts must be positive/non-negative")
    if args.validation_device == "cpu" and args.validation_precision != "float32":
        parser_.error("CPU posthoc requires float32")
    if args.validation_device == "cuda" and (args.validate_only or not modes) and not torch.cuda.is_available():
        parser_.error("CUDA validation requested but unavailable")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
