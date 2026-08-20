#!/usr/bin/env python3
"""Fine-tune Cube LeWM dynamics on a strict 1:1 expert/off-policy mixture.

Only the predictor, action encoder, and predictor projection are trainable.
The MaskedAug visual encoder and projector are frozen and byte-hashed before
and after training.  Expert samples retain Route2.1's 80/20 masked hue
augmentation; collected off-policy frames remain identity images.
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

from cube_coloraug import allowed_clip_indices, split_indices
from cube_maskedaug import MASK_METADATA_PREFIX, RandomMaskedHueRotation
from cube_offpolicy import (
    ACTION_DIM,
    IMAGE_SIZE,
    SOURCE_BATCH_SIZE,
    ExpertWindowDataset,
    OffPolicyWindowDataset,
    PairedSourceDataset,
    assert_frozen_hashes,
    freeze_predictor_only,
    frozen_state_hashes,
    load_offpolicy_manifest,
    module_state_sha256,
    sha256_file,
    split_rollout_ids,
    synthetic_window_smoke,
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
OFFPOLICY_ROOT = AILAB / "datasets/offpolicy_cube_v1"
WARM_WEIGHTS = (
    AILAB
    / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt"
)
NORMALIZERS = (
    AILAB
    / "outputs/train/route21_maskedaug/route21_masked_hsv_seed3072/normalizers.json"
)
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-offpolicy_v1"
OUTPUT_ROOT = AILAB / "outputs/train/offpolicy_v1"
TENSORBOARD_ROOT = AILAB / "logs/tensorboard/offpolicy_v1"
NUM_STEPS = 4
FRAMESKIP = 5
HISTORY_SIZE = 3
NUM_PREDS = 1
MASK_STAT_NAMES = ("empty_frames", "total_frames", "masked_pixels", "applied_clips", "seen_clips")


def _load_normalizers(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen Route2.1 normalizers missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    for column in ("action", "observation"):
        if column not in value or "mean" not in value[column] or "std" not in value[column]:
            raise ValueError(f"normalizers file lacks {column} mean/std")
    return value


def _encode_frozen_pixels(model: Any, pixels: torch.Tensor) -> torch.Tensor:
    """Encode targets without building a graph or updating frozen BN buffers."""
    if pixels.ndim != 5 or tuple(pixels.shape[1:]) != (NUM_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected pixels [B,4,3,224,224], actual={tuple(pixels.shape)}")
    batch = pixels.size(0)
    model.encoder.eval()
    model.projector.eval()
    with torch.no_grad():
        flat = pixels.float().reshape(batch * NUM_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE)
        output = model.encoder(flat, interpolate_pos_encoding=True)
        embedding = model.projector(output.last_hidden_state[:, 0])
    return embedding.reshape(batch, NUM_STEPS, -1)


def _sigreg_shared_projection(
    sigreg: Any, expert: torch.Tensor, offpolicy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One RNG draw, original combined loss, and source diagnostics."""
    combined = torch.cat((expert, offpolicy), dim=0).transpose(0, 1)
    expert_tb = expert.transpose(0, 1)
    offpolicy_tb = offpolicy.transpose(0, 1)
    projection = torch.randn(combined.size(-1), sigreg.num_proj, device=combined.device)
    projection = projection.div_(projection.norm(p=2, dim=0))

    def statistic(value: torch.Tensor) -> torch.Tensor:
        x_t = (value @ projection).unsqueeze(-1) * sigreg.t
        error = (
            (x_t.cos().mean(-3) - sigreg.phi).square()
            + x_t.sin().mean(-3).square()
        )
        return ((error @ sigreg.weights) * value.size(-2)).mean()

    return statistic(combined), statistic(expert_tb), statistic(offpolicy_tb)


def _source_prediction(model: Any, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = batch["pixels"]
    action = torch.nan_to_num(batch["action"], 0.0)
    if action.ndim != 3 or tuple(action.shape[1:]) != (NUM_STEPS, ACTION_DIM):
        raise ValueError(f"expected action [B,4,25], actual={tuple(action.shape)}")
    if torch.count_nonzero(action[:, -1]).item() != 0:
        raise RuntimeError("unused final action must be the explicit zero placeholder")
    embedding = _encode_frozen_pixels(model, pixels)
    # This slice is the executable proof that the placeholder action cannot
    # affect the loss: slot 3 is never passed to action_encoder or predictor.
    action_embedding = model.action_encoder(action[:, :HISTORY_SIZE])
    prediction = model.predict(
        embedding[:, :HISTORY_SIZE],
        action_embedding,
    )
    target = embedding[:, NUM_PREDS:].detach()
    return (prediction - target).pow(2).mean(), embedding


def offpolicy_forward(
    self: Any, batch: dict[str, Any], stage: str, cfg: Any
) -> dict[str, torch.Tensor]:
    if set(batch).difference({"expert", "offpolicy", "batch_idx"}):
        raise ValueError(f"unexpected top-level mixed-batch keys: {sorted(batch)}")
    expert = dict(batch["expert"])
    offpolicy = dict(batch["offpolicy"])
    expected = int(cfg.source_batch_size)
    actual = {"expert": int(expert["pixels"].size(0)), "offpolicy": int(offpolicy["pixels"].size(0))}
    if actual != {"expert": expected, "offpolicy": expected}:
        raise RuntimeError(f"source batch mismatch: expected={expected} each, actual={actual}")
    if stage in {"fit", "train"}:
        counts = getattr(self, "_offpolicy_source_examples", None)
        if counts is None:
            self._offpolicy_source_examples = {"expert": expected, "offpolicy": expected}
        else:
            counts["expert"] += expected
            counts["offpolicy"] += expected

    mask_stats: dict[str, torch.Tensor] = {}
    for name in MASK_STAT_NAMES:
        key = f"{MASK_METADATA_PREFIX}{name}"
        if key in expert:
            mask_stats[name] = expert.pop(key).sum().detach()
    if stage in {"fit", "train"} and mask_stats:
        totals = getattr(self, "_offpolicy_mask_totals", None)
        if totals is None:
            self._offpolicy_mask_totals = {name: value.clone() for name, value in mask_stats.items()}
        else:
            for name, value in mask_stats.items():
                totals[name] = totals[name] + value

    expert_pred, expert_embedding = _source_prediction(self.model, expert)
    offpolicy_pred, offpolicy_embedding = _source_prediction(self.model, offpolicy)
    pred_loss = 0.5 * (expert_pred + offpolicy_pred)
    sigreg_loss, expert_sigreg, offpolicy_sigreg = _sigreg_shared_projection(
        self.sigreg, expert_embedding, offpolicy_embedding
    )
    loss = pred_loss + float(cfg.sigreg_weight) * sigreg_loss
    if not all(
        bool(torch.isfinite(value.detach()).item())
        for value in (loss, pred_loss, sigreg_loss, expert_pred, offpolicy_pred)
    ):
        raise FloatingPointError(
            "non-finite training loss: "
            f"total={loss.detach()}, pred={pred_loss.detach()}, sigreg={sigreg_loss.detach()}, "
            f"expert_pred={expert_pred.detach()}, offpolicy_pred={offpolicy_pred.detach()}"
        )
    output = {
        "loss": loss,
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "expert_pred_loss": expert_pred,
        "offpolicy_pred_loss": offpolicy_pred,
        "expert_sigreg_loss": expert_sigreg,
        "offpolicy_sigreg_loss": offpolicy_sigreg,
    }
    self.log_dict(
        {f"{stage}/{key}": value.detach() for key, value in output.items()},
        on_step=True,
        sync_dist=True,
    )
    self._offpolicy_last_metrics = {
        key: float(value.detach().cpu().item()) for key, value in output.items()
    }
    return output


def _runtime_mask_stats(module: Any) -> dict[str, int | float]:
    totals = getattr(module, "_offpolicy_mask_totals", {})
    result = {
        name: int(totals[name].detach().cpu().item()) if name in totals else 0
        for name in MASK_STAT_NAMES
    }
    result["empty_frame_rate"] = (
        result["empty_frames"] / result["total_frames"] if result["total_frames"] else 0.0
    )
    return result


def _plan(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    collector: Mapping[str, Any],
    expert_train_count: int,
    expert_val_count: int,
    off_train_rollouts: int,
    off_val_rollouts: int,
    freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    steps = int(args.max_steps)
    excluded_action = _load_normalizers(args.normalizers)["action"]
    return {
        "format_version": "cube_offpolicy_train_v1",
        "run_id": args.run_id,
        "decision_record": {
            "offpolicy_fraction": 0.5,
            "reason": "strict 64 expert + 64 off-policy batches isolate the dynamics intervention while retaining masked-hue anti-forgetting data",
            "train_scope": "predictor + action_encoder + pred_proj; encoder + projector frozen",
            "step_budget": steps,
            "reason_step_budget": "requested first-pass 3-6k range; default midpoint is 4000 steps",
        },
        "inputs": {
            "expert_dataset": str(args.dataset),
            "formal_manifest": str(args.manifest),
            "offpolicy_root": str(args.offpolicy_root),
            "offpolicy_manifest": collector["manifest_path"],
            "offpolicy_manifest_sha256": collector["manifest_sha256"],
            "warm_start": str(args.warm_start),
            "warm_start_sha256": sha256_file(args.warm_start),
            "normalizers": str(args.normalizers),
            "normalizers_sha256": sha256_file(args.normalizers),
        },
        "formal_episode_exclusion": {
            "episode_count": len(protocol["episodes"]),
            "episode_ids": [int(value) for value in protocol["episodes"]],
            "manifest_sha256": protocol["manifest_sha256"],
            "collector_exact_match": True,
        },
        "splits": {
            "rule": "90/10 at rollout level before three sliding windows",
            "seed": args.split_seed,
            "expert_train_clips": expert_train_count,
            "expert_validation_clips": expert_val_count,
            "offpolicy_train_rollouts": off_train_rollouts,
            "offpolicy_validation_rollouts": off_val_rollouts,
            "offpolicy_train_windows": off_train_rollouts * 3,
            "offpolicy_validation_windows": off_val_rollouts * 3,
            "cross_split_rollout_overlap": 0,
        },
        "offpolicy_action_distribution": {
            "requested": collector.get("action_protocol", {}).get("mixture_requested"),
            "collected_counts": collector.get("action_protocol", {}).get("counts"),
            "source": "collector manifest; no resampling by distribution during training",
        },
        "batch": {
            "effective_size": 128,
            "expert": SOURCE_BATCH_SIZE,
            "offpolicy": SOURCE_BATCH_SIZE,
            "offpolicy_fraction": 0.5,
            "strict_every_step": True,
        },
        "windows": {
            "pixels": [4, 3, 224, 224],
            "action": [4, 25],
            "collector_stored_action_model": {
                "normalizer": "deployment/evaluation scaler",
                "consumed_for_training": False,
                "audit": "exactly reconstructed from action_env during manifest validation",
                "scaler": collector["action_normalizer_resolved"],
            },
            "training_action_source": "action_env normalized online then flattened 5x5 -> 25",
            "training_action_normalizer": {
                "contract": "Route2.1 fixed50-excluded expert normalizer for both sources",
                "mean": np.asarray(excluded_action["mean"]).reshape(-1).tolist(),
                "std": np.asarray(excluded_action["std"]).reshape(-1).tolist(),
                "count": int(excluded_action["count"]),
            },
            "unused_final_action": 0.0,
            "loss_action_slice": "[:, :3]",
        },
        "augmentation": {
            "expert": "Route2.1 masked red HSV hue rotation, 80% transform / 20% identity",
            "offpolicy": "identity",
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "steps": steps,
            "scheduler": "LinearWarmupCosineAnnealingLR, interval=step",
            "warmup_steps": max(1, int(0.01 * steps)),
            "gradient_clip_norm": args.gradient_clip_val,
            "loss": "pred_loss + 0.09 * SIGReg",
            "sigreg_weight": args.sigreg_weight,
        },
        "runtime_contract": {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "source_batch_size": SOURCE_BATCH_SIZE,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "hue_probability": args.hue_probability,
            "max_hue_delta": args.max_hue_delta,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "sigreg_weight": args.sigreg_weight,
            "gradient_clip_val": args.gradient_clip_val,
            "precision": args.precision,
            "accelerator": args.accelerator,
            "limit_val_batches": args.limit_val_batches,
            "log_every_n_steps": args.log_every_n_steps,
            "skip_shard_hash_verification": args.skip_shard_hash_verification,
            "optimizer_scope": ["predictor", "action_encoder", "pred_proj"],
            "frozen_scope": ["encoder", "projector"],
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "scheduler_interval": "step",
        },
        "freeze": dict(freeze or {"status": "resolved when model is loaded"}),
        "paths": {
            "checkpoint": str((CHECKPOINT_ROOT / args.run_id).resolve()),
            "output": str((OUTPUT_ROOT / args.run_id).resolve()),
            "tensorboard": str((TENSORBOARD_ROOT / args.run_id).resolve()),
        },
    }


def _first_contract_difference(
    expected: Any, actual: Any, path: str = "run_plan"
) -> tuple[str, Any, Any] | None:
    if type(expected) is not type(actual):
        return path, expected, actual
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return f"{path}.keys", sorted(expected_keys), sorted(actual_keys)
        for key in sorted(expected):
            mismatch = _first_contract_difference(expected[key], actual[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}.length", len(expected), len(actual)
        for index, (old, new) in enumerate(zip(expected, actual)):
            mismatch = _first_contract_difference(old, new, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if expected != actual:
        return path, expected, actual
    return None


def _assert_resume_contract(
    output_dir: Path,
    plan: Mapping[str, Any],
    train_rollout_ids: np.ndarray,
    validation_rollout_ids: np.ndarray,
) -> None:
    """Resume is continuation only; any experiment change requires a new id."""
    plan_path = output_dir / "run_plan.json"
    split_path = output_dir / "rollout_split.npz"
    if not plan_path.is_file():
        raise FileNotFoundError(
            f"resume run plan missing: expected={plan_path}, actual=missing, position={output_dir}"
        )
    if not split_path.is_file():
        raise FileNotFoundError(
            f"resume split missing: expected={split_path}, actual=missing, position={output_dir}"
        )
    frozen_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mismatch = _first_contract_difference(frozen_plan, dict(plan))
    if mismatch is not None:
        position, expected, actual = mismatch
        raise ValueError(
            "resume contract mismatch; changed data, steps, LR, scheduler, or freeze scope "
            "requires a new --run-id: "
            f"expected={expected!r}, actual={actual!r}, position={position!r}"
        )
    with np.load(split_path, allow_pickle=False) as split:
        expected_keys = {"train_rollout_ids", "validation_rollout_ids"}
        actual_keys = set(split.files)
        if actual_keys != expected_keys:
            raise ValueError(
                "resume split keys mismatch: "
                f"expected={sorted(expected_keys)!r}, actual={sorted(actual_keys)!r}, "
                f"position={str(split_path)!r}"
            )
        for name, recomputed in (
            ("train_rollout_ids", train_rollout_ids),
            ("validation_rollout_ids", validation_rollout_ids),
        ):
            frozen = np.asarray(split[name], dtype=np.int64)
            recomputed = np.asarray(recomputed, dtype=np.int64)
            if not np.array_equal(frozen, recomputed):
                if frozen.shape != recomputed.shape:
                    position: Any = {"artifact": str(split_path), "array": name, "index": "shape"}
                    expected: Any = tuple(frozen.shape)
                    actual: Any = tuple(recomputed.shape)
                else:
                    index = int(np.flatnonzero(frozen != recomputed)[0])
                    position = {"artifact": str(split_path), "array": name, "index": index}
                    expected = int(frozen[index])
                    actual = int(recomputed[index])
                raise ValueError(
                    "resume rollout split mismatch: "
                    f"expected={expected!r}, actual={actual!r}, position={position!r}"
                )


def _posthoc_loader(val_set: Any, args: argparse.Namespace) -> torch.utils.data.DataLoader:
    workers = args.num_workers if args.validation_num_workers is None else args.validation_num_workers
    common: dict[str, Any] = {
        "batch_size": SOURCE_BATCH_SIZE,
        "num_workers": workers,
        "pin_memory": args.validation_device == "cuda",
        "drop_last": True,
        "shuffle": False,
    }
    if workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
    return torch.utils.data.DataLoader(val_set, **common)


def _safe_posthoc_name(value: str) -> str:
    """Restrict write ownership to the posthoc-validation artifact family."""
    pattern = r"posthoc_validation(?:[._-][A-Za-z0-9][A-Za-z0-9._-]*)?\.json"
    if not re.fullmatch(pattern, value):
        raise ValueError(
            "posthoc filename is outside write ownership: "
            "expected='posthoc_validation*.json safe basename', "
            f"actual={value!r}, position='--posthoc-name'"
        )
    return value


def _move_validation_source(source: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "pixels": source["pixels"].to(device, non_blocking=True),
        "action": source["action"].to(device, non_blocking=True),
    }


def _evaluate_readonly_checkpoint(
    checkpoint: Path,
    *,
    val_loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    import stable_worldmodel as swm

    from module import SIGReg

    checkpoint = checkpoint.resolve()
    checkpoint_file_sha_before = sha256_file(checkpoint)
    model = swm.wm.utils.load_pretrained(str(checkpoint), cache_dir=str(AILAB))
    device = torch.device(args.validation_device)
    model.to(device).eval()
    sigreg = SIGReg(knots=17, num_proj=1024).to(device).eval()
    model_state_before = module_state_sha256(model)
    torch.manual_seed(args.validation_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.validation_seed)

    metric_names = (
        "loss",
        "pred_loss",
        "sigreg_loss",
        "expert_pred_loss",
        "offpolicy_pred_loss",
        "expert_sigreg_loss",
        "offpolicy_sigreg_loss",
        "expert_loss",
        "offpolicy_loss",
    )
    sums = {name: 0.0 for name in metric_names}
    batch_rows: list[dict[str, Any]] = []
    provenance = hashlib.sha256()
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= args.validation_batches:
                break
            if set(batch).difference({"expert", "offpolicy"}):
                raise ValueError(
                    "posthoc validation batch keys mismatch: "
                    f"expected=['expert', 'offpolicy'], actual={sorted(batch)}, "
                    f"position={{'batch': {batch_index}}}"
                )
            sizes = {
                source: int(batch[source]["pixels"].size(0))
                for source in ("expert", "offpolicy")
            }
            if sizes != {"expert": SOURCE_BATCH_SIZE, "offpolicy": SOURCE_BATCH_SIZE}:
                raise RuntimeError(
                    "posthoc validation source batch mismatch: "
                    f"expected={{'expert': 64, 'offpolicy': 64}}, actual={sizes}, "
                    f"position={{'batch': {batch_index}}}"
                )
            rollout_ids = np.asarray(batch["offpolicy"]["rollout_id"], dtype=np.int64)
            window_ids = np.asarray(batch["offpolicy"]["window_index"], dtype=np.int64)
            provenance.update(rollout_ids.tobytes())
            provenance.update(window_ids.tobytes())
            expert = _move_validation_source(batch["expert"], device)
            offpolicy = _move_validation_source(batch["offpolicy"], device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and args.validation_precision == "bf16"
                else nullcontext()
            )
            with amp_context:
                expert_pred, expert_embedding = _source_prediction(model, expert)
                offpolicy_pred, offpolicy_embedding = _source_prediction(model, offpolicy)
                pred_loss = 0.5 * (expert_pred + offpolicy_pred)
                sigreg_loss, expert_sigreg, offpolicy_sigreg = _sigreg_shared_projection(
                    sigreg, expert_embedding, offpolicy_embedding
                )
                loss = pred_loss + args.sigreg_weight * sigreg_loss
            values = {
                "loss": float(loss.float().cpu()),
                "pred_loss": float(pred_loss.float().cpu()),
                "sigreg_loss": float(sigreg_loss.float().cpu()),
                "expert_pred_loss": float(expert_pred.float().cpu()),
                "offpolicy_pred_loss": float(offpolicy_pred.float().cpu()),
                "expert_sigreg_loss": float(expert_sigreg.float().cpu()),
                "offpolicy_sigreg_loss": float(offpolicy_sigreg.float().cpu()),
                "expert_loss": float(
                    (expert_pred + args.sigreg_weight * expert_sigreg).float().cpu()
                ),
                "offpolicy_loss": float(
                    (offpolicy_pred + args.sigreg_weight * offpolicy_sigreg).float().cpu()
                ),
            }
            if not all(np.isfinite(value) for value in values.values()):
                raise FloatingPointError(
                    "posthoc validation produced non-finite metric: "
                    f"expected=finite, actual={values}, position={{'model': {label!r}, "
                    f"'batch': {batch_index}}}"
                )
            for name, value in values.items():
                sums[name] += value
            batch_rows.append(
                {
                    "batch": batch_index,
                    **values,
                    "offpolicy_rollout_first": int(rollout_ids[0]),
                    "offpolicy_rollout_last": int(rollout_ids[-1]),
                    "offpolicy_window_first": int(window_ids[0]),
                    "offpolicy_window_last": int(window_ids[-1]),
                }
            )
    actual_batches = len(batch_rows)
    if actual_batches != args.validation_batches:
        raise RuntimeError(
            "posthoc validation batch count mismatch: "
            f"expected={args.validation_batches}, actual={actual_batches}, "
            f"position={{'model': {label!r}, 'loader': 'validation'}}"
        )
    model_state_after = module_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError(
            "posthoc validation mutated in-memory weights: "
            f"expected={model_state_before!r}, actual={model_state_after!r}, "
            f"position={{'model': {label!r}}}"
        )
    checkpoint_file_sha_after = sha256_file(checkpoint)
    if checkpoint_file_sha_after != checkpoint_file_sha_before:
        raise RuntimeError(
            "posthoc validation mutated checkpoint file: "
            f"expected={checkpoint_file_sha_before!r}, actual={checkpoint_file_sha_after!r}, "
            f"position={str(checkpoint)!r}"
        )
    result = {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_file_sha_before,
        "checkpoint_sha256_after": checkpoint_file_sha_after,
        "checkpoint_unchanged": True,
        "model_state_sha256_before": model_state_before,
        "model_state_sha256_after": model_state_after,
        "model_state_unchanged": True,
        "num_batches": actual_batches,
        "source_examples": {
            "expert": actual_batches * SOURCE_BATCH_SIZE,
            "offpolicy": actual_batches * SOURCE_BATCH_SIZE,
        },
        "provenance_sha256": provenance.hexdigest(),
        "mean": {name: sums[name] / actual_batches for name in metric_names},
        "batches": batch_rows,
        "duration_seconds": time.monotonic() - started,
    }
    del sigreg, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _run_posthoc_validation(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    collector: Mapping[str, Any],
) -> int:
    import stable_worldmodel as swm

    output_dir = OUTPUT_ROOT / args.run_id
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"formal training output missing: expected={output_dir}, actual=missing, position='output'"
        )
    run_plan_path = output_dir / "run_plan.json"
    split_path = output_dir / "rollout_split.npz"
    checkpoint_dir = CHECKPOINT_ROOT / args.run_id
    planned_final = (checkpoint_dir / "weights_final.pt").resolve()
    final_checkpoint = (args.checkpoint or planned_final).expanduser().resolve()
    if final_checkpoint != planned_final:
        raise ValueError(
            "posthoc final checkpoint is outside the frozen run: "
            f"expected={str(planned_final)!r}, actual={str(final_checkpoint)!r}, "
            f"position='--checkpoint'"
        )
    if not final_checkpoint.is_file():
        raise FileNotFoundError(
            f"final checkpoint missing: expected={final_checkpoint}, actual=missing, position='--checkpoint'"
        )
    if not run_plan_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "posthoc validation contract artifacts missing: "
            f"expected={[str(run_plan_path), str(split_path)]!r}, "
            f"actual={{'run_plan': {run_plan_path.is_file()}, 'split': {split_path.is_file()}}}, "
            "position='formal output'"
        )

    train_set, val_set, expert_train_ids, expert_val_ids, off_train_ids, off_val_ids = _build_datasets(
        args, protocol, collector
    )
    del train_set
    base_model = swm.wm.utils.load_pretrained(str(args.warm_start), cache_dir=str(AILAB))
    freeze = freeze_predictor_only(base_model)
    freeze["frozen_sha256_before"] = frozen_state_hashes(base_model)
    plan = _plan(
        args,
        protocol,
        collector,
        len(expert_train_ids),
        len(expert_val_ids),
        len(off_train_ids),
        len(off_val_ids),
        freeze=freeze,
    )
    del base_model
    _assert_resume_contract(output_dir, plan, off_train_ids, off_val_ids)

    posthoc_name = _safe_posthoc_name(args.posthoc_name)
    output_path = output_dir / posthoc_name
    if output_path.is_symlink():
        raise ValueError(
            "posthoc output symlink is forbidden: "
            f"expected='regular file or absent', actual='symlink', position={str(output_path)!r}"
        )
    if output_path.exists() and not args.overwrite_posthoc:
        raise FileExistsError(
            f"posthoc output exists: expected=absent, actual={output_path}, position='--posthoc-name'"
        )
    val_loader = _posthoc_loader(val_set, args)
    base = _evaluate_readonly_checkpoint(
        args.warm_start,
        val_loader=val_loader,
        args=args,
        label="warm_start_base",
    )
    final = _evaluate_readonly_checkpoint(
        final_checkpoint,
        val_loader=val_loader,
        args=args,
        label="offpolicy_final",
    )
    if base["provenance_sha256"] != final["provenance_sha256"]:
        raise RuntimeError(
            "base/final posthoc batches differ: "
            f"expected={base['provenance_sha256']!r}, actual={final['provenance_sha256']!r}, "
            "position='validation provenance'"
        )
    del val_loader
    payload = {
        "format_version": "cube_offpolicy_posthoc_validation_v1",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": {
            "optimizer_constructed": False,
            "gradient_enabled": False,
            "weights_updated": False,
            "completed_json_overwritten": False,
            "run_plan_overwritten": False,
        },
        "protocol": {
            "validation_batches": args.validation_batches,
            "batch": {"expert": SOURCE_BATCH_SIZE, "offpolicy": SOURCE_BATCH_SIZE},
            "validation_seed": args.validation_seed,
            "device": args.validation_device,
            "precision": args.validation_precision,
            "sigreg_weight": args.sigreg_weight,
            "split": str(split_path),
            "split_sha256": sha256_file(split_path),
            "run_plan": str(run_plan_path),
            "run_plan_sha256": sha256_file(run_plan_path),
            "collector_manifest": collector["manifest_path"],
            "collector_manifest_sha256": collector["manifest_sha256"],
        },
        "base": base,
        "final": final,
        "delta_final_minus_base": {
            name: final["mean"][name] - base["mean"][name] for name in base["mean"]
        },
    }
    write_json(output_path, payload)
    print(json.dumps({"posthoc_validation": str(output_path), "delta": payload["delta_final_minus_base"]}, indent=2, sort_keys=True))
    return 0


class OffPolicyCallbacks:
    @staticmethod
    def create(
        run_id: str,
        model_config: Any,
        output_dir: Path,
        expected_frozen_hashes: Mapping[str, str],
        log_every_n_steps: int,
    ) -> list[Any]:
        from lightning.pytorch.callbacks import Callback
        from stable_worldmodel.wm.utils import save_pretrained

        class FrozenExportCallback(Callback):
            def _verify(self, module: Any) -> dict[str, str]:
                return assert_frozen_hashes(module.model, expected_frozen_hashes)

            def _save(self, trainer: Any, module: Any, filename: str) -> None:
                if trainer.is_global_zero:
                    self._verify(module)
                    save_pretrained(
                        module.model,
                        run_name=f"lewm-cube-offpolicy_v1/{run_id}",
                        config=model_config,
                        filename=filename,
                        cache_dir=str(AILAB),
                    )

            def on_train_epoch_end(self, trainer: Any, module: Any) -> None:
                self._save(trainer, module, f"weights_epoch_{trainer.current_epoch + 1}.pt")

            def on_train_end(self, trainer: Any, module: Any) -> None:
                actual = self._verify(module)
                self._save(trainer, module, "weights_final.pt")
                if trainer.is_global_zero:
                    write_json(
                        output_dir / "frozen_integrity.json",
                        {
                            "status": "PASS",
                            "before": dict(expected_frozen_hashes),
                            "after": actual,
                            "exact_match": actual == dict(expected_frozen_hashes),
                        },
                    )

        class LossCurveCallback(Callback):
            def on_train_batch_end(
                self, trainer: Any, module: Any, outputs: Any, batch: Any, batch_idx: int
            ) -> None:
                step = int(trainer.global_step)
                should_log = step == 1 or step == int(trainer.max_steps) or step % log_every_n_steps == 0
                if not trainer.is_global_zero or not should_log:
                    return
                metrics = dict(getattr(module, "_offpolicy_last_metrics", {}))
                optimizers = trainer.optimizers
                metrics.update(
                    {
                        "step": step,
                        "epoch": int(trainer.current_epoch),
                        "batch_idx": int(batch_idx),
                        "learning_rate": float(optimizers[0].param_groups[0]["lr"]),
                    }
                )
                with (output_dir / "loss_curve.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")

        return [FrozenExportCallback(), LossCurveCallback()]


def _build_datasets(args: argparse.Namespace, protocol: Mapping[str, Any], collector: Mapping[str, Any]):
    import stable_pretraining as spt
    import stable_worldmodel as swm

    from utils import get_img_preprocessor

    dataset = swm.data.load_dataset(
        str(args.dataset),
        transform=None,
        num_steps=NUM_STEPS,
        frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    excluded = {int(value) for value in protocol["episodes"]}
    allowed = allowed_clip_indices(dataset.clip_indices, excluded)
    if len(allowed) != protocol["total_clips"] - protocol["excluded_clips"]:
        raise RuntimeError("expert fixed50 clip exclusion differs from the frozen protocol")
    expert_train_ids, expert_val_ids = split_indices(allowed, args.train_fraction, args.split_seed)
    if any(
        int(dataset.clip_indices[index][0]) in excluded
        for index in np.concatenate((expert_train_ids, expert_val_ids))
    ):
        raise RuntimeError("formal evaluation episode leaked into expert data")

    # Both training sources use Route2.1's fixed50-excluded scaler.  The
    # collector's stored action_model is deployment-scaled and is audited by
    # load_offpolicy_manifest, but intentionally never consumed for training.
    normalizers = _load_normalizers(args.normalizers)
    excluded_mean = np.asarray(normalizers["action"]["mean"], dtype=np.float32).reshape(5)
    excluded_std = np.asarray(normalizers["action"]["std"], dtype=np.float32).reshape(5)
    action_mean = torch.from_numpy(excluded_mean.reshape(1, 5)).repeat(1, FRAMESKIP)
    action_std = torch.from_numpy(excluded_std.reshape(1, 5)).repeat(1, FRAMESKIP)
    preprocessor = get_img_preprocessor(source="pixels", target="pixels", img_size=IMAGE_SIZE)
    expert_transform = spt.data.transforms.Compose(
        RandomMaskedHueRotation(probability=args.hue_probability, max_delta=args.max_hue_delta),
        preprocessor,
        ColumnNormalizer("action", action_mean, action_std),
    )
    validation_expert_transform = spt.data.transforms.Compose(
        preprocessor,
        ColumnNormalizer("action", action_mean, action_std),
    )
    offpolicy_transform = preprocessor

    off_train_ids, off_val_ids = split_rollout_ids(
        int(collector["num_rollouts"]), args.train_fraction, args.split_seed
    )
    expert_train = ExpertWindowDataset(dataset, expert_train_ids, expert_transform)
    expert_val = ExpertWindowDataset(dataset, expert_val_ids, validation_expert_transform)
    off_train = OffPolicyWindowDataset(
        collector, off_train_ids, excluded_mean, excluded_std, offpolicy_transform
    )
    off_val = OffPolicyWindowDataset(
        collector, off_val_ids, excluded_mean, excluded_std, offpolicy_transform
    )
    return (
        PairedSourceDataset(expert_train, off_train),
        PairedSourceDataset(expert_val, off_val),
        expert_train_ids,
        expert_val_ids,
        off_train_ids,
        off_val_ids,
    )


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    args.run_id = _safe_run_id(args.run_id)
    args.dataset = _validate_data_disk(args.dataset, "expert dataset")
    args.manifest = _validate_data_disk(args.manifest, "formal manifest")
    args.warm_start = _validate_data_disk(args.warm_start, "warm-start checkpoint")
    args.normalizers = _validate_data_disk(args.normalizers, "normalizers")
    args.offpolicy_root = args.offpolicy_root.expanduser().resolve()
    if AILAB.parent.resolve() not in args.offpolicy_root.parents:
        raise ValueError(f"off-policy root must be on /root/autodl-tmp: {args.offpolicy_root}")

    if args.synthetic_smoke:
        result = synthetic_window_smoke()
        result["requested_training_steps"] = 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    protocol = _heldout_protocol(args.dataset, args.manifest)
    normalizers = _load_normalizers(args.normalizers)
    action_mean = np.asarray(normalizers["action"]["mean"], dtype=np.float64).reshape(-1)
    action_std = np.asarray(normalizers["action"]["std"], dtype=np.float64).reshape(-1)
    collector = load_offpolicy_manifest(
        args.offpolicy_root,
        expected_excluded_episodes=protocol["episodes"],
        expected_action_mean=action_mean,
        expected_action_std=action_std,
        verify_shard_hashes=not args.skip_shard_hash_verification,
    )
    if args.validate_only:
        return _run_posthoc_validation(args, protocol, collector)
    train_rollouts, val_rollouts = split_rollout_ids(
        int(collector["num_rollouts"]), args.train_fraction, args.split_seed
    )
    allowed_expert = protocol["total_clips"] - protocol["excluded_clips"]
    expert_train_count = math.floor(allowed_expert * args.train_fraction)
    expert_val_count = allowed_expert - expert_train_count
    plan = _plan(
        args,
        protocol,
        collector,
        expert_train_count,
        expert_val_count,
        len(train_rollouts),
        len(val_rollouts),
    )
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
        raise FileNotFoundError(f"--resume checkpoint missing: {resume_path}")
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
    train_set, val_set, expert_train_ids, expert_val_ids, off_train_ids, off_val_ids = _build_datasets(
        args, protocol, collector
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_common: dict[str, Any] = {
        "batch_size": SOURCE_BATCH_SIZE,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_common.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    train_loader = torch.utils.data.DataLoader(
        train_set, **loader_common, shuffle=True, generator=generator
    )
    val_loader = torch.utils.data.DataLoader(val_set, **loader_common, shuffle=False)

    model = swm.wm.utils.load_pretrained(str(args.warm_start), cache_dir=str(AILAB))
    freeze = freeze_predictor_only(model)
    frozen_before = frozen_state_hashes(model)
    freeze["frozen_sha256_before"] = frozen_before
    plan = _plan(
        args,
        protocol,
        collector,
        len(expert_train_ids),
        len(expert_val_ids),
        len(off_train_ids),
        len(off_val_ids),
        freeze=freeze,
    )
    if args.resume:
        _assert_resume_contract(output_dir, plan, off_train_ids, off_val_ids)
    else:
        write_json(output_dir / "run_plan.json", plan)
        np.savez_compressed(
            output_dir / "rollout_split.npz",
            train_rollout_ids=off_train_ids,
            validation_rollout_ids=off_val_ids,
        )

    config_path = args.warm_start.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"warm-start config missing: {config_path}")
    model_config = OmegaConf.create(json.loads(config_path.read_text(encoding="utf-8")))
    forward_cfg = OmegaConf.create(
        {"source_batch_size": SOURCE_BATCH_SIZE, "sigreg_weight": args.sigreg_weight}
    )
    optimizers = {
        "model_opt": {
            "modules": r"model\.(predictor|action_encoder|pred_proj)(?:\.|$)",
            "optimizer": {
                "type": "AdamW",
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "interval": "step",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(knots=17, num_proj=1024),
        forward=partial(offpolicy_forward, cfg=forward_cfg),
        optim=optimizers,
    )
    configure_manual_gradient_clipping(module, "model_opt", args.gradient_clip_val, "norm")
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    callbacks = OffPolicyCallbacks.create(
        args.run_id,
        model_config,
        output_dir,
        frozen_before,
        args.log_every_n_steps,
    )
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(checkpoint_dir / "lightning"),
            save_last=True,
            save_top_k=-1,
            every_n_train_steps=min(1000, args.max_steps),
            filename="step{step}",
            enable_version_counter=False,
        )
    )
    logger = TensorBoardLogger(
        save_dir=str(TENSORBOARD_ROOT), name="", version=args.run_id, default_hp_metric=False
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        max_epochs=-1,
        max_steps=args.max_steps,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(output_dir / "lightning"),
        num_sanity_val_steps=1,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
    )
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    trainer.fit(module, datamodule=data_module, ckpt_path=str(resume_path) if args.resume else None)
    duration_seconds = time.monotonic() - started_monotonic
    ended_at = datetime.now(timezone.utc)
    frozen_after = assert_frozen_hashes(model, frozen_before)
    write_json(
        output_dir / "completed.json",
        {
            "run_id": args.run_id,
            "global_step": int(trainer.global_step),
            "epochs_completed": int(trainer.current_epoch),
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "duration_seconds": duration_seconds,
            "final_weights": str((checkpoint_dir / "weights_final.pt").resolve()),
            "final_weights_sha256": sha256_file(checkpoint_dir / "weights_final.pt"),
            "tensorboard": str(tensorboard_dir.resolve()),
            "loss_curve": str((output_dir / "loss_curve.jsonl").resolve()),
            "source_mix": {"expert": SOURCE_BATCH_SIZE, "offpolicy": SOURCE_BATCH_SIZE},
            "source_examples_seen": dict(
                getattr(module, "_offpolicy_source_examples", {"expert": 0, "offpolicy": 0})
            ),
            "frozen_integrity": {
                "status": "PASS",
                "before": frozen_before,
                "after": frozen_after,
                "exact_match": frozen_before == frozen_after,
            },
            "masked_augmentation_runtime": _runtime_mask_stats(module),
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="offpolicy_v1_pred_seed3072")
    parser.add_argument("--dataset", type=Path, default=EXPERT_DATASET)
    parser.add_argument("--manifest", type=Path, default=FORMAL_MANIFEST)
    parser.add_argument("--offpolicy-root", type=Path, default=OFFPOLICY_ROOT)
    parser.add_argument("--warm-start", type=Path, default=WARM_WEIGHTS)
    parser.add_argument("--normalizers", type=Path, default=NORMALIZERS)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--hue-probability", type=float, default=0.8)
    parser.add_argument("--max-hue-delta", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16-mixed", "32-true"), default="bf16-mixed")
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--limit-val-batches", type=int, default=50)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--skip-shard-hash-verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--validation-seed", type=int, default=3072)
    parser.add_argument("--validation-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--validation-precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--validation-num-workers", type=int)
    parser.add_argument("--posthoc-name", default="posthoc_validation.json")
    parser.add_argument("--overwrite-posthoc", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser_ = parser()
    args = parser_.parse_args(argv)
    if args.validate_only and (args.resume or args.dry_run or args.synthetic_smoke):
        parser_.error("--validate-only is mutually exclusive with --resume/--dry-run/--synthetic-smoke")
    if not args.validate_only and (
        args.checkpoint is not None
        or args.validation_batches != 50
        or args.validation_seed != 3072
        or args.validation_device != "cuda"
        or args.validation_precision != "bf16"
        or args.validation_num_workers is not None
        or args.posthoc_name != "posthoc_validation.json"
        or args.overwrite_posthoc
    ):
        parser_.error("posthoc validation options require --validate-only")
    if args.validation_batches < 1:
        parser_.error("--validation-batches must be positive")
    if args.validation_num_workers is not None and args.validation_num_workers < 0:
        parser_.error("--validation-num-workers must be non-negative")
    if args.validate_only and args.validation_device == "cpu" and args.validation_precision != "float32":
        parser_.error("CPU posthoc validation requires --validation-precision=float32")
    if args.validate_only and args.validation_device == "cuda" and not torch.cuda.is_available():
        parser_.error("CUDA posthoc validation requested but CUDA is unavailable")
    if args.batch_size != 128:
        parser_.error("--batch-size is frozen to 128 (64 expert + 64 off-policy)")
    if args.max_steps < 1:
        parser_.error("--max-steps must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        parser_.error("worker settings are invalid")
    if not 0.0 < args.train_fraction < 1.0:
        parser_.error("--train-fraction must be in (0,1)")
    if not 0.0 <= args.hue_probability <= 1.0:
        parser_.error("--hue-probability must be in [0,1]")
    if not 0.0 < args.max_hue_delta <= 0.5:
        parser_.error("--max-hue-delta must be in (0,0.5]")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_val <= 0:
        parser_.error("optimization hyperparameters are invalid")
    if args.sigreg_weight != 0.09:
        parser_.error("--sigreg-weight is frozen to 0.09")
    if args.accelerator == "cpu" and args.precision != "32-true":
        parser_.error("CPU execution requires --precision=32-true")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
