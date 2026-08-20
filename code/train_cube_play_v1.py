#!/usr/bin/env python3
"""Single-arm Cube dynamics fine-tune on expert + official OGBench play data.

The three play-v1 iron laws are executable contracts, not comments:

* exactly ``predictor`` and ``action_encoder`` are optimized;
* the objective is one-step teacher forcing plus the original target SIGReg,
  and the production ``model.rollout`` method is never called;
* the same frozen expert holdout is measured at step zero and every 500
  optimizer steps; a >10% increase saves the scene and terminates with code 3.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import hdf5plugin  # noqa: F401
import numpy as np
import torch

from cube_maskedaug import MASK_METADATA_PREFIX, RandomMaskedHueRotation
from cube_play_v1 import (
    FORMAL_BATCH_SIZE,
    FORMAL_EXPERT_BATCH,
    FORMAL_PLAY_BATCH,
    FROZEN_MODULES,
    LOADER_BATCH,
    NUM_FRAMES,
    PreparedPlayDataset,
    SourceClipDataset,
    StrictPlayMixtureDataset,
    assert_frozen_hashes,
    atomic_write_json,
    canonical_json_sha256,
    file_identity,
    flatten_bundled_source,
    freeze_dynamics_stack,
    frozen_state_hashes,
    load_prepared_play_manifest,
    sha256_file,
    split_clips_by_episode,
    synthetic_contract_smoke,
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
PLAY_MANIFEST = AILAB / "datasets/ogbench_play/manifest.json"
FORMAL_MANIFEST = AILAB / "outputs/audit/cube_cem_manifest.json"
MEASUREMENT1_SEGMENTS = (
    AILAB / "outputs/eval/cube/imagination_error/measurement1_segments.json"
)
WARM_WEIGHTS = (
    AILAB
    / "checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt"
)
NORMALIZERS = AILAB / "outputs/train/robust_v1/lewm-cube-robust_v1/normalizers.json"
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-play_v1"
OUTPUT_ROOT = AILAB / "outputs/train/play_v1"
TENSORBOARD_ROOT = AILAB / "logs/tensorboard/play_v1"
FORMAL_RUN_ID = "play_v1_dyn_seed3072"
ROBUST_WEIGHTS_SHA256 = "cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed"
NORMALIZERS_SHA256 = "32ab9a37631f6de612e413f2067b009a669b56bacc700f85b67f251ebe3188b4"
EXPECTED_TRAINABLE_TENSORS = 87
EXPECTED_TRAINABLE_PARAMETERS = 10_947_716
TEACHER_ACTIONS = 3
STOPLINE_INTERVAL = 500
STOPLINE_THRESHOLD = 0.10
STOPLINE_NUMERIC_ATOL = 1e-12
STOPLINE_BATCHES = 34
STOPLINE_EXAMPLES = STOPLINE_BATCHES * FORMAL_BATCH_SIZE
MASK_STAT_NAMES = (
    "empty_frames",
    "total_frames",
    "masked_pixels",
    "applied_clips",
    "seen_clips",
)
LOSS_CONTRACT: dict[str, Any] = {
    "formula": "0.625*expert_teacher + 0.375*play_teacher + 0.09*shared_target_SIGReg",
    "teacher_forcing_steps": 3,
    "frames": 4,
    "expert_weight": 0.625,
    "play_weight": 0.375,
    "sigreg_weight": 0.09,
    "sigreg_input": "frozen encoder+projector target embeddings, expert+play concatenated",
    "sigreg_gradient_to_trainable_dynamics": False,
    "rollout_enabled": False,
    "rollout_depth": 0,
    "rollout_weight": 0.0,
    "model_rollout_calls": 0,
    "target_encoder_stop_gradient": True,
}
LOSS_CONTRACT_SHA256 = canonical_json_sha256(LOSS_CONTRACT)


def _stopline_exceeded(relative: float) -> bool:
    """Use the evaluator's strict >10% rule without boundary roundoff trips."""
    return bool(
        relative > STOPLINE_THRESHOLD
        and not np.isclose(
            relative,
            STOPLINE_THRESHOLD,
            atol=STOPLINE_NUMERIC_ATOL,
            rtol=0.0,
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _measurement1_episodes(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    episodes = np.asarray(value.get("episode_indices"), dtype=np.int64)
    formal = np.asarray(value.get("formal_episodes_excluded"), dtype=np.int64)
    if episodes.shape != (2000,) or len(np.unique(episodes)) != 1801:
        raise RuntimeError("Measurement-1 episode holdout contract changed")
    if formal.shape != (50,) or np.intersect1d(episodes, formal).size:
        raise RuntimeError("Measurement-1/formal holdouts are malformed")
    return {
        "episodes": np.unique(episodes),
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "segments": 2000,
    }


def _load_normalizers(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    action = value.get("action", {})
    mean = np.asarray(action.get("mean"), dtype=np.float32)
    std = np.asarray(action.get("std"), dtype=np.float32)
    if mean.shape != (1, 5) or std.shape != (1, 5):
        raise RuntimeError(f"expert action normalizer shape changed: {mean.shape}/{std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("expert action normalizer is non-finite/non-positive")
    return value


def _encode_targets(model: Any, pixels: torch.Tensor) -> torch.Tensor:
    if pixels.ndim != 5 or tuple(pixels.shape[1:]) != (4, 3, 224, 224):
        raise ValueError(f"expected normalized four-frame pixels, actual={tuple(pixels.shape)}")
    batch = int(pixels.shape[0])
    # Lightning recursively calls train() on the whole JEPA.  Reassert eval on
    # every immutable module so dropout and BatchNorm buffers cannot drift.
    model.encoder.eval()
    model.projector.eval()
    with torch.no_grad():
        encoded = model.encoder(
            pixels.float().reshape(batch * NUM_FRAMES, 3, 224, 224),
            interpolate_pos_encoding=True,
        )
        target = model.projector(encoded.last_hidden_state[:, 0])
    return target.reshape(batch, NUM_FRAMES, -1)


def _teacher_from_targets(
    model: Any, target: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    if tuple(action.shape[1:]) != (4, 25) or not torch.isfinite(
        action[:, :TEACHER_ACTIONS]
    ).all():
        raise ValueError(
            "expected finite normalized actions in the three consumed teacher blocks "
            f"[B,3,25], actual full shape={tuple(action.shape)}"
        )
    if tuple(target.shape[:2]) != (action.shape[0], 4):
        raise ValueError("teacher target/action batch mismatch")
    model.pred_proj.eval()  # pred_proj BN buffers are part of the frozen hash.
    # The fourth block corresponds to no predicted transition and may contain
    # the expert H5 terminal NaN padding.  It is deliberately never encoded.
    action_embedding = model.action_encoder(action[:, :TEACHER_ACTIONS])
    prediction = model.predict(target[:, :TEACHER_ACTIONS], action_embedding)
    return (prediction - target[:, 1:].detach()).square().mean()


def _shared_target_sigreg(
    sigreg: Any, expert: torch.Tensor, play: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use one random projection for combined and diagnostic source metrics."""
    combined = torch.cat((expert, play), dim=0).transpose(0, 1)
    expert_tb = expert.transpose(0, 1)
    play_tb = play.transpose(0, 1)
    projection = torch.randn(combined.size(-1), sigreg.num_proj, device=combined.device)
    projection = projection.div_(projection.norm(p=2, dim=0).clamp_min(1e-12))

    def statistic(value: torch.Tensor) -> torch.Tensor:
        x_t = (value @ projection).unsqueeze(-1) * sigreg.t
        error = (
            (x_t.cos().mean(-3) - sigreg.phi).square()
            + x_t.sin().mean(-3).square()
        )
        return ((error @ sigreg.weights) * value.size(-2)).mean()

    return statistic(combined), statistic(expert_tb), statistic(play_tb)


def play_v1_forward(
    self: Any, batch: dict[str, Any], stage: str, cfg: Any
) -> dict[str, torch.Tensor]:
    """Strict one-step loss.  Do not add a call to ``model.rollout`` here."""
    data_keys = set(batch) - {"batch_idx"}
    if data_keys != {"expert", "play"}:
        raise ValueError(f"unexpected play-v1 batch keys: {sorted(batch)}")
    expert = flatten_bundled_source(batch["expert"])
    play = flatten_bundled_source(batch["play"])
    actual = {
        "expert": int(expert["pixels"].shape[0]),
        "play": int(play["pixels"].shape[0]),
    }
    expected = {"expert": FORMAL_EXPERT_BATCH, "play": FORMAL_PLAY_BATCH}
    if actual != expected or sum(actual.values()) != FORMAL_BATCH_SIZE:
        raise RuntimeError(f"strict play mixture mismatch: expected={expected}, actual={actual}")

    mask_stats: dict[str, torch.Tensor] = {}
    for name in MASK_STAT_NAMES:
        key = f"{MASK_METADATA_PREFIX}{name}"
        if key in expert:
            mask_stats[name] = expert.pop(key).sum().detach()
    if stage in {"fit", "train"}:
        source_totals = getattr(self, "_play_v1_source_examples", {"expert": 0, "play": 0})
        source_totals["expert"] += FORMAL_EXPERT_BATCH
        source_totals["play"] += FORMAL_PLAY_BATCH
        self._play_v1_source_examples = source_totals
        augmentation = getattr(
            self, "_play_v1_mask_totals", {name: 0 for name in MASK_STAT_NAMES}
        )
        for name, value in mask_stats.items():
            augmentation[name] = augmentation[name] + value
        self._play_v1_mask_totals = augmentation

    expert_target = _encode_targets(self.model, expert["pixels"])
    play_target = _encode_targets(self.model, play["pixels"])
    expert_pred = _teacher_from_targets(self.model, expert_target, expert["action"])
    play_pred = _teacher_from_targets(self.model, play_target, play["action"])
    shared_sigreg, expert_sigreg, play_sigreg = _shared_target_sigreg(
        self.sigreg, expert_target, play_target
    )
    pred_loss = 0.625 * expert_pred + 0.375 * play_pred
    rollout_zero = pred_loss.new_zeros(())
    loss = pred_loss + 0.09 * shared_sigreg
    metrics = {
        "loss": loss,
        "pred_loss": pred_loss,
        "sigreg_loss": shared_sigreg,
        "expert_pred_loss": expert_pred,
        "play_pred_loss": play_pred,
        "expert_sigreg_loss": expert_sigreg,
        "play_sigreg_loss": play_sigreg,
        "rollout_loss": rollout_zero,
        "expert_rollout_loss": rollout_zero,
        "play_rollout_loss": rollout_zero,
    }
    if any(value.requires_grad for key, value in metrics.items() if "rollout" in key):
        raise RuntimeError("rollout audit zeros unexpectedly require gradients")
    if any(float(value.detach()) != 0.0 for key, value in metrics.items() if "rollout" in key):
        raise RuntimeError("rollout audit loss is not exact zero")
    if not all(bool(torch.isfinite(value.detach()).item()) for value in metrics.values()):
        raise FloatingPointError(
            "non-finite play-v1 loss: "
            + ", ".join(f"{key}={value.detach()}" for key, value in metrics.items())
        )
    self.log_dict(
        {f"{stage}/{key}": value.detach() for key, value in metrics.items()},
        on_step=True,
        sync_dist=True,
    )
    self._play_v1_last_metrics = {
        key: float(value.detach().float().cpu()) for key, value in metrics.items()
    }
    return metrics


def _build_datasets(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    measurement: Mapping[str, Any],
    play_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_img_preprocessor

    expert_dataset = swm.data.load_dataset(
        str(args.expert_dataset),
        transform=None,
        num_steps=NUM_FRAMES,
        frameskip=5,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    expert_excluded = sorted(
        set(map(int, protocol["episodes"])) | set(map(int, measurement["episodes"]))
    )
    expert_split = split_clips_by_episode(
        expert_dataset,
        excluded_episodes=expert_excluded,
        train_fraction=args.train_fraction,
        seed=args.split_seed,
    )
    normalizers = _load_normalizers(args.normalizers)
    mean = torch.tensor(normalizers["action"]["mean"], dtype=torch.float32).repeat(1, 5)
    std = torch.tensor(normalizers["action"]["std"], dtype=torch.float32).repeat(1, 5)
    preprocessor = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    action_normalizer = ColumnNormalizer("action", mean, std)
    expert_train_transform = spt.data.transforms.Compose(
        RandomMaskedHueRotation(
            probability=args.hue_probability, max_delta=args.max_hue_delta
        ),
        preprocessor,
        action_normalizer,
    )
    clean_transform = spt.data.transforms.Compose(preprocessor, action_normalizer)
    expert_train = SourceClipDataset(
        expert_dataset, expert_split["train_ids"], expert_train_transform, "expert"
    )
    expert_val = SourceClipDataset(
        expert_dataset, expert_split["val_ids"], clean_transform, "expert"
    )
    play_train = PreparedPlayDataset(play_manifest, "train", clean_transform)
    play_val = PreparedPlayDataset(play_manifest, "val", clean_transform)
    return {
        "train": StrictPlayMixtureDataset(expert_train, play_train),
        "val": StrictPlayMixtureDataset(expert_val, play_val),
        "expert_val": expert_val,
        "expert_train": expert_train,
        "play_train": play_train,
        "play_val": play_val,
        "expert_split": expert_split,
        "play_manifest": play_manifest,
        "expert_excluded_episodes": np.asarray(expert_excluded, dtype=np.int64),
    }


def _data_contract_smoke(
    args: argparse.Namespace, datasets: Mapping[str, Any]
) -> dict[str, Any]:
    """Deterministically scan the full stopline set and a frozen train batch."""
    expert_val = datasets["expert_val"]
    base = expert_val.dataset

    loader_kwargs: dict[str, Any] = {
        "batch_size": FORMAL_BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "num_workers": args.num_workers,
    }
    if args.num_workers:
        loader_kwargs.update(
            {"persistent_workers": True, "prefetch_factor": args.prefetch_factor}
        )
    stopline_loader = torch.utils.data.DataLoader(
        expert_val,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    scanned = 0
    unused_padding_examples = 0
    provenance = hashlib.sha256()
    for batch_index, batch in enumerate(stopline_loader):
        if batch_index >= STOPLINE_BATCHES:
            break
        if (
            tuple(batch["pixels"].shape[1:]) != (4, 3, 224, 224)
            or tuple(batch["action"].shape[1:]) != (4, 25)
            or not torch.isfinite(batch["pixels"]).all()
            or not torch.isfinite(batch["action"][:, :TEACHER_ACTIONS]).all()
        ):
            raise RuntimeError(f"non-finite clean stopline batch {batch_index}")
        unused_padding_examples += int(
            (~torch.isfinite(batch["action"][:, TEACHER_ACTIONS])).any(dim=-1).sum()
        )
        provenance.update(np.asarray(batch["clip_index"], dtype=np.int64).tobytes())
        scanned += int(batch["pixels"].shape[0])
    if scanned != STOPLINE_EXAMPLES:
        raise RuntimeError(
            f"clean stopline scan incomplete: expected={STOPLINE_EXAMPLES}, actual={scanned}"
        )

    mixed = datasets["train"]
    fixed_indices = np.linspace(0, len(mixed) - 1, LOADER_BATCH, dtype=np.int64)
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(args.seed)
        fixed_batch = torch.utils.data.default_collate(
            [mixed[int(index)] for index in fixed_indices]
        )
    finally:
        torch.set_rng_state(rng_state)
    fixed_expert = flatten_bundled_source(fixed_batch["expert"])
    fixed_play = flatten_bundled_source(fixed_batch["play"])
    if (
        fixed_expert["pixels"].shape[0] != FORMAL_EXPERT_BATCH
        or fixed_play["pixels"].shape[0] != FORMAL_PLAY_BATCH
        or not torch.isfinite(fixed_expert["pixels"]).all()
        or not torch.isfinite(
            fixed_expert["action"][:, :TEACHER_ACTIONS]
        ).all()
        or not torch.isfinite(fixed_play["pixels"]).all()
        or not torch.isfinite(fixed_play["action"]).all()
    ):
        raise RuntimeError("frozen training mixture contains non-finite data")

    terminal = next(
        index
        for index, (episode, start) in enumerate(base.clip_indices)
        if int(episode) == 5 and int(start) == 181
    )
    terminal_view = SourceClipDataset(
        base, [terminal], expert_val.transform, "expert_terminal_padding"
    )
    terminal_sample = terminal_view[0]
    if (
        not torch.isfinite(terminal_sample["pixels"]).all()
        or not torch.isfinite(
            terminal_sample["action"][:TEACHER_ACTIONS]
        ).all()
        or torch.isfinite(terminal_sample["action"][TEACHER_ACTIONS]).all()
    ):
        raise RuntimeError("actual expert terminal-padding contract changed")

    class UsedActionNaNDataset(torch.utils.data.Dataset):
        clip_indices = [(0, 0)]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            del index
            action = torch.zeros((4, 25), dtype=torch.float32)
            action[2, 0] = float("nan")
            return {
                "pixels": torch.zeros((4, 3, 224, 224), dtype=torch.float32),
                "action": action,
            }

    negative_view = SourceClipDataset(
        UsedActionNaNDataset(), [0], lambda sample: sample, "expert_used_action_negative"
    )
    try:
        negative_view[0]
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
    else:
        raise RuntimeError("NaN in a consumed action block was silently accepted")

    return {
        "status": "PASS",
        "clean_stopline": {
            "examples_scanned": scanned,
            "batches": STOPLINE_BATCHES,
            "clip_indices_sha256": provenance.hexdigest(),
            "augmentation": "none",
            "consumed_inputs_all_finite": True,
            "unused_fourth_block_nonfinite_examples": unused_padding_examples,
        },
        "fixed_training_mixture": {
            "mixture_indices": fixed_indices.tolist(),
            "expert_examples": FORMAL_EXPERT_BATCH,
            "play_examples": FORMAL_PLAY_BATCH,
            "masked_augmentation_seed": args.seed,
            "consumed_inputs_all_finite": True,
        },
        "terminal_padding": {
            "policy": "retain clip; validate and encode only three consumed action blocks",
            "terminal_clip": {
                "clip_index": int(terminal),
                "episode": 5,
                "start": 181,
                "accepted": True,
                "consumed_blocks_finite": True,
                "unused_fourth_block_nonfinite": True,
            },
            "consumed_action_nan_negative_rejected": True,
            "nan_replacement_used": False,
        },
    }


def _array_sha(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _split_summary(datasets: Mapping[str, Any]) -> dict[str, Any]:
    expert = datasets["expert_split"]
    play = datasets["play_manifest"]["resolved_shards"]
    return {
        "unit": "whole episode",
        "expert": {
            "train_episodes": int(len(expert["train_episodes"])),
            "validation_episodes": int(len(expert["val_episodes"])),
            "train_clips": int(len(expert["train_ids"])),
            "validation_clips": int(len(expert["val_ids"])),
            "terminal_action_padding_policy": (
                "retain every valid 4-frame/3-transition clip; validate only the first "
                "three consumed action blocks and never encode the fourth block"
            ),
            "train_episode_sha256": _array_sha(expert["train_episodes"]),
            "validation_episode_sha256": _array_sha(expert["val_episodes"]),
            "cross_split_overlap": int(
                np.intersect1d(expert["train_episodes"], expert["val_episodes"]).size
            ),
        },
        "play": {
            "assignment": "official play train/val split; never resplit",
            "train_episodes": int(play["train"]["episodes"]),
            "validation_episodes": int(play["val"]["episodes"]),
            "train_clips": int(play["train"]["windows"]),
            "validation_clips": int(play["val"]["windows"]),
            "train_shard_sha256": play["train"]["sha256"],
            "validation_shard_sha256": play["val"]["sha256"],
            "cross_split_overlap": 0,
        },
        "expert_excluded_episodes": int(len(datasets["expert_excluded_episodes"])),
        "fixed50_and_measurement1_excluded_from_expert": True,
        "play_independent_episode_namespace": True,
    }


def _write_episode_split(path: Path, datasets: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    arrays = {
        "expert_train_episodes": datasets["expert_split"]["train_episodes"],
        "expert_val_episodes": datasets["expert_split"]["val_episodes"],
        "expert_excluded_episodes": datasets["expert_excluded_episodes"],
    }
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_identity(path)


def _validate_episode_split(path: Path, datasets: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "expert_train_episodes": datasets["expert_split"]["train_episodes"],
        "expert_val_episodes": datasets["expert_split"]["val_episodes"],
        "expert_excluded_episodes": datasets["expert_excluded_episodes"],
    }
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise RuntimeError(
                f"resume split keys changed: expected={sorted(expected)}, actual={sorted(saved.files)}"
            )
        for name, value in expected.items():
            if not np.array_equal(saved[name], np.asarray(value, dtype=np.int64)):
                raise RuntimeError(f"resume episode split differs at {name}")
    return file_identity(path)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_trainer_checkpoint(trainer: Any, destination: Path) -> None:
    """Keep the previous recovery scene intact until a complete new scene exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        trainer.save_checkpoint(str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _plan(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    measurement: Mapping[str, Any],
    validation: Mapping[str, Any],
    datasets: Mapping[str, Any],
    freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": "cube_play_v1_train_plan_v1",
        "run_id": args.run_id,
        "single_arm": True,
        "retry_allowed": False,
        "resume_allowed": {
            "infrastructure_only": True,
            "scientific_retry": False,
            "requirements": "canonical run id, byte-identical run_plan, split, data and code",
            "recovery_checkpoint_interval_steps": STOPLINE_INTERVAL,
        },
        "decision_log": [
            "Use exact 80/48 batch (62.5% expert, 37.5% play), centered in requested ranges.",
            "Freeze pred_proj because only predictor+action_encoder are authorized; keep its BatchNorm in eval.",
            "Exclude Measurement-1 episodes as well as fixed50 from expert training to keep both offline regressions held out.",
            "Keep original target SIGReg although frozen targets make its gradient to dynamics exactly zero.",
        ],
        "inputs": {
            "expert_dataset": file_identity(args.expert_dataset, include_sha256=False),
            "play_manifest": file_identity(args.play_manifest),
            "play_sources": dict(validation["resolved_sources"]),
            "play_shards": dict(validation["resolved_shards"]),
            "play_reports": dict(validation["resolved_reports"]),
            "formal_manifest": file_identity(args.manifest),
            "measurement1_segments": {
                "path": measurement["path"],
                "sha256": measurement["sha256"],
                "segments": measurement["segments"],
            },
            "warm_start": file_identity(args.warm_start),
            "normalizers": file_identity(args.normalizers),
        },
        "formal_episode_exclusion": {
            "episode_count": len(protocol["episodes"]),
            "episode_ids": list(map(int, protocol["episodes"])),
            "manifest_sha256": protocol["manifest_sha256"],
            "play_exclusion": dict(validation["exclusion"]),
        },
        "splits": _split_summary(datasets),
        "batch": {
            "total": FORMAL_BATCH_SIZE,
            "expert": FORMAL_EXPERT_BATCH,
            "play": FORMAL_PLAY_BATCH,
            "expert_fraction": 0.625,
            "play_fraction": 0.375,
            "bundle": {"expert": 5, "play": 3, "loader_batch": LOADER_BATCH},
            "strict_every_step": True,
        },
        "augmentation": {
            "expert": "MaskedAug red-mask hue rotation",
            "play": "none",
            "hue_probability": args.hue_probability,
            "max_hue_delta_turns": args.max_hue_delta,
            "validation": "clean",
        },
        "loss_contract": dict(LOSS_CONTRACT),
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": 50,
            "gradient_clip_norm": args.gradient_clip_val,
            "recovery_checkpoint_interval": STOPLINE_INTERVAL,
            "archival_checkpoint_interval": 1000,
        },
        "realtime_stopline": {
            "metric": "clean expert heldout exact 4-frame/3-transition teacher pred loss",
            "baseline": "measured from robust_v1 at step zero in the same process",
            "threshold_relative_increase": STOPLINE_THRESHOLD,
            "numeric_boundary_atol": STOPLINE_NUMERIC_ATOL,
            "comparison": "current/baseline-1 strictly greater than threshold",
            "interval_completed_optimizer_steps": STOPLINE_INTERVAL,
            "batches": STOPLINE_BATCHES,
            "examples": STOPLINE_EXAMPLES,
            "shuffle": False,
            "precision": "bf16",
            "on_failure": "atomic event/history + live Lightning checkpoint + stopped weights + exit code 3",
        },
        "freeze": dict(freeze or {}),
        "runtime": {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "limit_val_batches": args.limit_val_batches,
        },
        "paths": {
            "output": str((OUTPUT_ROOT / args.run_id).resolve()),
            "checkpoint": str((CHECKPOINT_ROOT / args.run_id).resolve()),
            "tensorboard": str((TENSORBOARD_ROOT / args.run_id).resolve()),
        },
        "code": {
            "train_sha256": sha256_file(Path(__file__)),
            "contract_sha256": sha256_file(Path(__file__).with_name("cube_play_v1.py")),
        },
    }


def _move_source(source: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in source.items()
    }


def _stopline_loader(datasets: Mapping[str, Any], args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {
        "batch_size": FORMAL_BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers:
        kwargs.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    loader = torch.utils.data.DataLoader(datasets["expert_val"], **kwargs)
    if len(loader) < STOPLINE_BATCHES:
        raise RuntimeError(
            f"expert holdout too small for stopline: expected>={STOPLINE_BATCHES}, actual={len(loader)}"
        )
    return loader


def _evaluate_current_teacher(
    model: Any,
    loader: Any,
    frozen_expected: Mapping[str, str],
) -> tuple[float, dict[str, Any]]:
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("formal play-v1 stopline requires CUDA bf16")
    assert_frozen_hashes(model, frozen_expected)
    original_modes = {id(module): bool(module.training) for module in model.modules()}
    model.eval()
    total = 0.0
    batches = 0
    provenance = hashlib.sha256()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if batch_index >= STOPLINE_BATCHES:
                    break
                provenance.update(
                    np.asarray(batch["clip_index"], dtype=np.int64).tobytes()
                )
                source = _move_source(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    target = _encode_targets(model, source["pixels"])
                    teacher = _teacher_from_targets(model, target, source["action"])
                value = float(teacher.float().cpu())
                if not np.isfinite(value):
                    raise FloatingPointError(f"non-finite expert stopline at batch {batch_index}")
                total += value
                batches += 1
    finally:
        for module in model.modules():
            module.training = original_modes[id(module)]
        for name in FROZEN_MODULES:
            getattr(model, name).eval()
    if batches != STOPLINE_BATCHES:
        raise RuntimeError(
            f"stopline batch count mismatch: expected={STOPLINE_BATCHES}, actual={batches}"
        )
    assert_frozen_hashes(model, frozen_expected)
    return total / batches, {
        "num_batches": STOPLINE_BATCHES,
        "examples": STOPLINE_EXAMPLES,
        "batch_size": FORMAL_BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "precision": "bf16",
        "expert_clip_indices_sha256": provenance.hexdigest(),
    }


class PlayV1Callbacks:
    @staticmethod
    def create(
        *,
        run_id: str,
        model_config: Any,
        output_dir: Path,
        checkpoint_dir: Path,
        frozen_before: Mapping[str, str],
        expert_loader: Any,
        train_generator: torch.Generator,
        curve_every: int,
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
                self.restored = False
                self.last_recovery_step = 0

            @property
            def state_key(self) -> str:
                return f"PlayV1StoplineAndExport:{run_id}"

            def state_dict(self) -> dict[str, Any]:
                return {
                    "format_version": "cube_play_v1_callback_state_v1",
                    "run_id": run_id,
                    "history": self.history,
                    "triggered": self.triggered,
                    "last_recovery_step": self.last_recovery_step,
                    "train_generator_state": train_generator.get_state(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_cpu_rng_state": torch.get_rng_state(),
                    "torch_cuda_rng_states": torch.cuda.get_rng_state_all(),
                }

            def load_state_dict(self, state_dict: dict[str, Any]) -> None:
                if (
                    state_dict.get("format_version") != "cube_play_v1_callback_state_v1"
                    or state_dict.get("run_id") != run_id
                    or not isinstance(state_dict.get("history"), Mapping)
                ):
                    raise RuntimeError("recovery checkpoint callback state is noncanonical")
                self.history = dict(state_dict["history"])
                self.triggered = bool(state_dict.get("triggered"))
                self.last_recovery_step = int(state_dict.get("last_recovery_step", -1))
                generator_state = state_dict.get("train_generator_state")
                cpu_rng = state_dict.get("torch_cpu_rng_state")
                cuda_rng = state_dict.get("torch_cuda_rng_states")
                if (
                    not torch.is_tensor(generator_state)
                    or not torch.is_tensor(cpu_rng)
                    or not isinstance(cuda_rng, list)
                    or not cuda_rng
                    or "python_rng_state" not in state_dict
                    or "numpy_rng_state" not in state_dict
                ):
                    raise RuntimeError("recovery checkpoint lacks complete sampler/global RNG state")
                train_generator.set_state(generator_state.cpu())
                random.setstate(state_dict["python_rng_state"])
                np.random.set_state(state_dict["numpy_rng_state"])
                torch.set_rng_state(cpu_rng.cpu())
                torch.cuda.set_rng_state_all(cuda_rng)
                self.restored = True

            def _export(self, module: Any, filename: str) -> Path:
                assert_frozen_hashes(module.model, frozen_before)
                save_pretrained(
                    module.model,
                    run_name=f"lewm-cube-play_v1/{run_id}",
                    config=model_config,
                    filename=filename,
                    cache_dir=str(AILAB),
                )
                path = checkpoint_dir / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                return path

            def _publish(self) -> None:
                if self.history is None:
                    raise RuntimeError("stopline history is uninitialized")
                atomic_write_json(output_dir / "stopline_history.json", self.history)

            def on_train_start(self, trainer: Any, module: Any) -> None:
                if not trainer.is_global_zero:
                    return
                restored_step = int(trainer.global_step)
                if self.restored:
                    if self.triggered:
                        raise RuntimeError("STOPLINE_FAIL checkpoint cannot be resumed")
                    if (
                        restored_step < STOPLINE_INTERVAL
                        or restored_step % STOPLINE_INTERVAL
                        or self.last_recovery_step != restored_step
                    ):
                        raise RuntimeError(
                            "recovery checkpoint is not a completed 500-step scene: "
                            f"global_step={restored_step}, callback_step={self.last_recovery_step}"
                        )
                    _validate_history(self.history, restored_step, run_id)
                    assert self.history is not None
                    current, provenance = _evaluate_current_teacher(
                        module.model, expert_loader, frozen_before
                    )
                    last = self.history["records"][-1]
                    if provenance != last["provenance"] or not math.isclose(
                        current,
                        float(last["teacher_pred_loss"]),
                        rel_tol=1e-6,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(
                            "resumed checkpoint does not reproduce its paired expert stopline: "
                            f"expected={last['teacher_pred_loss']}, actual={current}"
                        )
                    self.history["resume_events"].append(
                        {
                            "step": restored_step,
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "checkpoint": str(
                                (checkpoint_dir / "lightning" / "last.ckpt").resolve()
                            ),
                            "status": "CANONICAL_INFRASTRUCTURE_RESUME",
                        }
                    )
                    module._play_v1_source_examples = {
                        "expert": FORMAL_EXPERT_BATCH * restored_step,
                        "play": FORMAL_PLAY_BATCH * restored_step,
                    }
                    self._publish()
                    return
                if restored_step != 0:
                    raise RuntimeError(
                        f"fresh play-v1 callback started at nonzero step {restored_step}"
                    )
                baseline, provenance = _evaluate_current_teacher(
                    module.model, expert_loader, frozen_before
                )
                if not np.isfinite(baseline) or baseline <= 0:
                    raise RuntimeError(f"invalid step-zero expert baseline: {baseline}")
                self.history = {
                    "format_version": "cube_play_v1_stopline_history_v1",
                    "run_id": run_id,
                    "baseline": {
                        "step": 0,
                        "teacher_pred_loss": baseline,
                        "relative_increase": 0.0,
                        "status": "PASS",
                        "provenance": provenance,
                    },
                    "threshold": STOPLINE_THRESHOLD,
                    "interval": STOPLINE_INTERVAL,
                    "comparison": "relative increase strictly > threshold triggers",
                    "evaluated_after_optimizer_step": True,
                    "triggered": False,
                    "records": [],
                    "resume_events": [],
                }
                module._play_v1_source_examples = {"expert": 0, "play": 0}
                self._publish()

            def on_train_batch_end(
                self, trainer: Any, module: Any, outputs: Any, batch: Any, batch_idx: int
            ) -> None:
                step = int(trainer.global_step)
                if (
                    not trainer.is_global_zero
                    or self.triggered
                    or step < STOPLINE_INTERVAL
                    or step % STOPLINE_INTERVAL
                ):
                    return
                assert self.history is not None
                if self.history["records"] and self.history["records"][-1]["step"] == step:
                    return
                current, provenance = _evaluate_current_teacher(
                    module.model, expert_loader, frozen_before
                )
                baseline = float(self.history["baseline"]["teacher_pred_loss"])
                if provenance != self.history["baseline"]["provenance"]:
                    raise RuntimeError("expert stopline batches are not paired with step zero")
                relative = current / baseline - 1.0
                failed = _stopline_exceeded(relative)
                record = {
                    "step": step,
                    "teacher_pred_loss": current,
                    "relative_increase": relative,
                    "status": "STOPLINE_FAIL" if failed else "PASS",
                    "provenance": provenance,
                }
                self.history["records"].append(record)
                self.history["triggered"] = failed
                if failed:
                    self.history["trigger"] = dict(record)
                self._publish()
                self.last_recovery_step = step
                if failed:
                    self.triggered = True
                    self.event_path = output_dir / "stopline_event.json"
                    atomic_write_json(
                        self.event_path,
                        {
                            "format_version": "cube_play_v1_stopline_event_v1",
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            **record,
                            "baseline_teacher_pred_loss": baseline,
                            "threshold": STOPLINE_THRESHOLD,
                            "action": "save recovery scene and quarantined stopped weights; terminate; no retry",
                        },
                    )
                    self.stopped_weights = self._export(
                        module, f"weights_stopped_step{step}.pt"
                    )
                recovery = checkpoint_dir / "lightning" / "last.ckpt"
                _atomic_trainer_checkpoint(trainer, recovery)
                if failed:
                    self.live_checkpoint = (
                        checkpoint_dir / "lightning" / f"stopline_step{step}.ckpt"
                    )
                    _atomic_copy(recovery, self.live_checkpoint)
                    trainer.should_stop = True
                elif step % 1000 == 0:
                    # Archive only passing scenes.  A stop-line scene is
                    # quarantined under stopline_stepN.ckpt and must never be
                    # published through the ordinary checkpoint series.
                    _atomic_copy(
                        recovery,
                        checkpoint_dir / "lightning" / f"step{step}.ckpt",
                    )

            def on_train_end(self, trainer: Any, module: Any) -> None:
                if trainer.is_global_zero:
                    if not self.triggered and int(trainer.global_step) == int(trainer.max_steps):
                        self._export(module, "weights_final.pt")
                    after = assert_frozen_hashes(module.model, frozen_before)
                    atomic_write_json(
                        output_dir / "frozen_integrity.json",
                        {
                            "format_version": "cube_play_v1_frozen_integrity_v1",
                            "status": "PASS",
                            "modules": list(FROZEN_MODULES),
                            "before": dict(frozen_before),
                            "after": after,
                            "exact_match": after == dict(frozen_before),
                        },
                    )

        class Curves(Callback):
            def on_train_batch_end(
                self, trainer: Any, module: Any, outputs: Any, batch: Any, batch_idx: int
            ) -> None:
                step = int(trainer.global_step)
                if not trainer.is_global_zero or not (
                    step == 1 or step == trainer.max_steps or step % curve_every == 0
                ):
                    return
                values = dict(getattr(module, "_play_v1_last_metrics", {}))
                values.update(
                    {
                        "step": step,
                        "epoch": int(trainer.current_epoch),
                        "batch_idx": int(batch_idx),
                        "learning_rate": float(
                            trainer.optimizers[0].param_groups[0]["lr"]
                        ),
                    }
                )
                with (output_dir / "loss_curve.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(values, sort_keys=True) + "\n")

        stopline = StoplineAndExport()
        return [stopline, Curves()], stopline


def _validate_history(history: Any, completed_step: int, run_id: str) -> None:
    if not isinstance(history, Mapping) or not history.get("records"):
        raise RuntimeError("no periodic expert stopline record")
    triggered = bool(history.get("triggered"))
    expected_keys = {
        "format_version",
        "run_id",
        "baseline",
        "threshold",
        "interval",
        "comparison",
        "evaluated_after_optimizer_step",
        "triggered",
        "records",
        "resume_events",
    } | ({"trigger"} if triggered else set())
    if set(history) != expected_keys:
        raise RuntimeError(
            f"stopline history keys changed: expected={sorted(expected_keys)}, actual={sorted(history)}"
        )
    expected_header = {
        "format_version": "cube_play_v1_stopline_history_v1",
        "run_id": run_id,
        "threshold": STOPLINE_THRESHOLD,
        "interval": STOPLINE_INTERVAL,
        "comparison": "relative increase strictly > threshold triggers",
        "evaluated_after_optimizer_step": True,
    }
    header_mismatch = {
        key: {"expected": expected, "actual": history.get(key)}
        for key, expected in expected_header.items()
        if history.get(key) != expected
    }
    if header_mismatch:
        raise RuntimeError(f"stopline history header changed: {header_mismatch}")
    baseline_record = history.get("baseline")
    if not isinstance(baseline_record, Mapping) or set(baseline_record) != {
        "step",
        "teacher_pred_loss",
        "relative_increase",
        "status",
        "provenance",
    }:
        raise RuntimeError("stopline baseline schema changed")
    provenance = baseline_record["provenance"]
    expected_provenance = {
        "num_batches": STOPLINE_BATCHES,
        "examples": STOPLINE_EXAMPLES,
        "batch_size": FORMAL_BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "precision": "bf16",
    }
    if (
        baseline_record.get("step") != 0
        or baseline_record.get("relative_increase") != 0.0
        or baseline_record.get("status") != "PASS"
        or not isinstance(provenance, Mapping)
        or set(provenance) != set(expected_provenance) | {"expert_clip_indices_sha256"}
        or any(provenance.get(key) != value for key, value in expected_provenance.items())
        or len(str(provenance.get("expert_clip_indices_sha256", ""))) != 64
    ):
        raise RuntimeError("stopline baseline/provenance contract changed")
    baseline = float(history["baseline"]["teacher_pred_loss"])
    if not np.isfinite(baseline) or baseline <= 0:
        raise RuntimeError(f"stopline baseline is invalid: {baseline}")
    if completed_step < STOPLINE_INTERVAL or completed_step % STOPLINE_INTERVAL:
        raise RuntimeError(f"stopline completed step is not a recovery boundary: {completed_step}")
    expected_steps = list(range(STOPLINE_INTERVAL, completed_step + 1, STOPLINE_INTERVAL))
    actual_steps = [int(record["step"]) for record in history["records"]]
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"stopline schedule incomplete: expected={expected_steps}, actual={actual_steps}"
        )
    for index, record in enumerate(history["records"]):
        if set(record) != {
            "step",
            "teacher_pred_loss",
            "relative_increase",
            "status",
            "provenance",
        }:
            raise RuntimeError(f"stopline record schema changed at index {index}")
        relative = float(record["teacher_pred_loss"]) / baseline - 1.0
        if not math.isclose(
            relative, float(record["relative_increase"]), rel_tol=1e-15, abs_tol=1e-15
        ):
            raise RuntimeError(f"stopline relative value mismatch at record {index}")
        if record["provenance"] != provenance:
            raise RuntimeError(f"stopline provenance mismatch at record {index}")
        expected_fail = triggered and index == len(history["records"]) - 1
        if _stopline_exceeded(relative) != expected_fail:
            raise RuntimeError(f"stopline comparison mismatch at record {index}")
        if record["status"] != ("STOPLINE_FAIL" if expected_fail else "PASS"):
            raise RuntimeError(f"stopline status mismatch at record {index}")
    if triggered:
        if history.get("trigger") != history["records"][-1]:
            raise RuntimeError("stopline trigger is not the final failing record")
    elif "trigger" in history:
        raise RuntimeError("passing stopline history unexpectedly contains a trigger")
    if not isinstance(history["resume_events"], list):
        raise RuntimeError("stopline resume_events is not a list")
    for index, event in enumerate(history["resume_events"]):
        if (
            not isinstance(event, Mapping)
            or set(event) != {"step", "created_at_utc", "checkpoint", "status"}
            or event.get("status") != "CANONICAL_INFRASTRUCTURE_RESUME"
            or int(event.get("step", -1)) not in expected_steps
        ):
            raise RuntimeError(f"invalid infrastructure resume event at index {index}")


def _runtime_mask_stats(module: Any) -> dict[str, int | float]:
    totals = getattr(module, "_play_v1_mask_totals", {})
    result = {
        name: int(totals[name].detach().cpu())
        if torch.is_tensor(totals.get(name))
        else int(totals.get(name, 0))
        for name in MASK_STAT_NAMES
    }
    result["empty_frame_rate"] = (
        result["empty_frames"] / result["total_frames"]
        if result["total_frames"]
        else 0.0
    )
    return result


def _validate_source_examples(module: Any, completed_steps: int) -> dict[str, int]:
    expected = {
        "expert": FORMAL_EXPERT_BATCH * int(completed_steps),
        "play": FORMAL_PLAY_BATCH * int(completed_steps),
    }
    actual = dict(getattr(module, "_play_v1_source_examples", {}))
    if actual != expected:
        raise RuntimeError(
            "strict source example accounting failed: "
            f"expected={expected}, actual={actual}, step={completed_steps}"
        )
    return actual


def synthetic_training_smoke() -> dict[str, Any]:
    base = synthetic_contract_smoke()
    source = inspect.getsource(play_v1_forward)
    if ".rollout(" in source or "rollout_loss =" in source:
        raise RuntimeError("play_v1_forward contains a rollout execution path")
    if LOSS_CONTRACT["model_rollout_calls"] != 0 or LOSS_CONTRACT["rollout_weight"] != 0.0:
        raise RuntimeError("loss contract enables rollout")
    provenance = {
        "num_batches": STOPLINE_BATCHES,
        "examples": STOPLINE_EXAMPLES,
        "batch_size": FORMAL_BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "precision": "bf16",
        "expert_clip_indices_sha256": "a" * 64,
    }
    history = {
        "format_version": "cube_play_v1_stopline_history_v1",
        "run_id": FORMAL_RUN_ID,
        "baseline": {
            "step": 0,
            "teacher_pred_loss": 0.01,
            "relative_increase": 0.0,
            "status": "PASS",
            "provenance": provenance,
        },
        "threshold": STOPLINE_THRESHOLD,
        "interval": STOPLINE_INTERVAL,
        "comparison": "relative increase strictly > threshold triggers",
        "evaluated_after_optimizer_step": True,
        "triggered": False,
        "records": [
            {
                "step": step,
                "teacher_pred_loss": 0.01,
                "relative_increase": 0.0,
                "status": "PASS",
                "provenance": provenance,
            }
            for step in range(STOPLINE_INTERVAL, 5000 + 1, STOPLINE_INTERVAL)
        ],
        "resume_events": [],
    }
    _validate_history(history, 5000, FORMAL_RUN_ID)
    boundary_history = json.loads(json.dumps(history))
    boundary_loss = 0.01 * 1.1
    boundary_relative = boundary_loss / 0.01 - 1.0
    boundary_history["records"][-1].update(
        {
            "teacher_pred_loss": boundary_loss,
            "relative_increase": boundary_relative,
        }
    )
    _validate_history(boundary_history, 5000, FORMAL_RUN_ID)
    above_history = json.loads(json.dumps(history))
    above_relative = 0.100000000002
    above_history["records"][-1].update(
        {
            "teacher_pred_loss": 0.01 * (1.0 + above_relative),
            "relative_increase": above_relative,
            "status": "STOPLINE_FAIL",
        }
    )
    above_history["triggered"] = True
    above_history["trigger"] = dict(above_history["records"][-1])
    _validate_history(above_history, 5000, FORMAL_RUN_ID)
    if _stopline_exceeded(0.10000000000000009) or not _stopline_exceeded(0.100000000002):
        raise RuntimeError("stopline numeric boundary tolerance changed")
    bad_history = json.loads(json.dumps(history))
    bad_history["threshold"] = 0.5
    try:
        _validate_history(bad_history, 5000, FORMAL_RUN_ID)
    except RuntimeError:
        history_negative = True
    else:
        raise RuntimeError("stopline history mutation negative test did not fail")

    class Counter:
        _play_v1_source_examples = {"expert": 80 * 10, "play": 48 * 10}

    _validate_source_examples(Counter(), 10)
    Counter._play_v1_source_examples["play"] -= 1
    try:
        _validate_source_examples(Counter(), 10)
    except RuntimeError:
        source_count_negative = True
    else:
        raise RuntimeError("source count mutation negative test did not fail")
    return {
        **base,
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "forward_model_rollout_calls": 0,
        "rollout_weight": 0.0,
        "single_step_only": True,
        "history_schema_positive": True,
        "history_mutation_rejected": history_negative,
        "stopline_boundary_roundoff_passes": True,
        "stopline_above_tolerance_fails": True,
        "source_count_positive": True,
        "source_count_mutation_rejected": source_count_negative,
    }


def _real_model_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from unittest.mock import patch

    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_img_preprocessor

    protocol = _heldout_protocol(args.expert_dataset, args.manifest)
    measurement = _measurement1_episodes(args.measurement1_segments)
    excluded = set(map(int, protocol["episodes"])) | set(map(int, measurement["episodes"]))
    dataset = swm.data.load_dataset(
        str(args.expert_dataset),
        transform=None,
        num_steps=4,
        frameskip=5,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    allowed = next(
        index
        for index, value in enumerate(dataset.clip_indices)
        if int(value[0]) not in excluded and int(value[1]) == 181
    )
    normalizers = _load_normalizers(args.normalizers)
    mean = torch.tensor(normalizers["action"]["mean"], dtype=torch.float32).repeat(1, 5)
    std = torch.tensor(normalizers["action"]["std"], dtype=torch.float32).repeat(1, 5)
    transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=224),
        ColumnNormalizer("action", mean, std),
    )
    sample = transform(dataset[allowed])
    pixels = sample["pixels"].unsqueeze(0)
    action = sample["action"].unsqueeze(0)
    if (
        not torch.isfinite(action[:, :TEACHER_ACTIONS]).all()
        or torch.isfinite(action[:, TEACHER_ACTIONS]).all()
    ):
        raise RuntimeError("real-model smoke no longer covers terminal action padding")
    model = swm.wm.utils.load_pretrained(
        str(args.warm_start), cache_dir=str(AILAB)
    ).cpu()
    freeze = freeze_dynamics_stack(model)
    if (
        freeze["trainable_parameter_tensors"] != EXPECTED_TRAINABLE_TENSORS
        or freeze["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(f"real-model trainable contract changed: {freeze}")
    before = frozen_state_hashes(model)
    with patch.object(model, "rollout", side_effect=RuntimeError("rollout forbidden")) as spy:
        target = _encode_targets(model, pixels)
        loss = _teacher_from_targets(model, target, action)
        if not torch.isfinite(loss):
            raise RuntimeError("terminal-padded fourth block contaminated teacher loss")
        used_nan = action.clone()
        used_nan[0, 0, 0] = float("nan")
        try:
            _teacher_from_targets(model, target, used_nan)
        except ValueError:
            used_nan_rejected = True
        else:
            raise RuntimeError("NaN in a consumed teacher action block was accepted")
        loss.backward()
        if spy.call_count:
            raise RuntimeError("real-model one-step smoke called rollout")
    gradients = {
        prefix: int(
            sum(
                torch.count_nonzero(parameter.grad).item()
                for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            )
        )
        for prefix in ("predictor", "action_encoder", "pred_proj", "encoder", "projector")
    }
    if gradients["predictor"] <= 0 or gradients["action_encoder"] <= 0:
        raise RuntimeError(f"real-model dynamics gradient missing: {gradients}")
    if any(gradients[name] for name in ("pred_proj", "encoder", "projector")):
        raise RuntimeError(f"real-model frozen gradient exists: {gradients}")
    assert_frozen_hashes(model, before)
    return {
        "status": "PASS",
        "checkpoint": file_identity(args.warm_start),
        "expert_clip_index": allowed,
        "source_episode": int(dataset.clip_indices[allowed][0]),
        "source_start": int(dataset.clip_indices[allowed][1]),
        "unused_fourth_action_block_nonfinite": True,
        "consumed_action_nan_negative_rejected": used_nan_rejected,
        "teacher_pred_loss": float(loss.detach()),
        "model_rollout_calls": 0,
        "gradients_nonzero_elements": gradients,
        "freeze": freeze,
        "frozen_hash_exact": True,
    }


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    args.run_id = _safe_run_id(args.run_id)
    if args.synthetic_smoke:
        print(json.dumps(synthetic_training_smoke(), indent=2, sort_keys=True))
        return 0
    args.expert_dataset = _validate_data_disk(args.expert_dataset, "expert dataset")
    args.manifest = _validate_data_disk(args.manifest, "formal manifest")
    args.measurement1_segments = _validate_data_disk(
        args.measurement1_segments, "Measurement-1 segments"
    )
    args.warm_start = _validate_data_disk(args.warm_start, "robust-v1 warm start")
    args.normalizers = _validate_data_disk(args.normalizers, "expert normalizers")
    if sha256_file(args.warm_start) != ROBUST_WEIGHTS_SHA256:
        raise RuntimeError("robust-v1 warm-start SHA256 changed")
    if sha256_file(args.normalizers) != NORMALIZERS_SHA256:
        raise RuntimeError("robust-v1 normalizer SHA256 changed")

    if args.model_smoke:
        print(json.dumps(_real_model_smoke(args), indent=2, sort_keys=True))
        return 0

    args.play_manifest = _validate_data_disk(args.play_manifest, "prepared play manifest")
    protocol = _heldout_protocol(args.expert_dataset, args.manifest)
    measurement = _measurement1_episodes(args.measurement1_segments)
    validation = load_prepared_play_manifest(
        args.play_manifest, verify_shard_sha256=not args.dry_run
    )
    datasets = _build_datasets(args, protocol, measurement, validation)
    if args.data_smoke:
        print(json.dumps(_data_contract_smoke(args, datasets), indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(
            json.dumps(
                _plan(args, protocol, measurement, validation, datasets),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_dir = OUTPUT_ROOT / args.run_id
    checkpoint_dir = CHECKPOINT_ROOT / args.run_id
    tensorboard_dir = TENSORBOARD_ROOT / args.run_id
    recovery_checkpoint = checkpoint_dir / "lightning" / "last.ckpt"
    existing = [
        path for path in (output_dir, checkpoint_dir, tensorboard_dir)
        if path.exists() and any(path.iterdir())
    ]
    if existing and not args.resume:
        raise FileExistsError(
            f"play-v1 output already exists; only canonical --resume may continue it: {existing}"
        )
    if args.resume:
        required_resume = (
            output_dir / "run_plan.json",
            output_dir / "episode_split.npz",
            recovery_checkpoint,
        )
        missing_resume = [path for path in required_resume if not path.is_file()]
        if missing_resume:
            raise FileNotFoundError(f"canonical recovery artifacts missing: {missing_resume}")
        if (output_dir / "completed.json").exists():
            raise RuntimeError("completed play-v1 arm cannot be resumed or retried")
    elif existing:
        raise AssertionError(existing)
    for path in (output_dir, checkpoint_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)

    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.loggers import TensorBoardLogger
    from module import SIGReg
    from omegaconf import OmegaConf

    pl.seed_everything(args.seed, workers=True)
    loader_kwargs: dict[str, Any] = {
        "batch_size": LOADER_BATCH,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers:
        loader_kwargs.update(
            {"persistent_workers": True, "prefetch_factor": args.prefetch_factor}
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        datasets["train"], shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        datasets["val"], shuffle=False, **loader_kwargs
    )
    expert_stopline_loader = _stopline_loader(datasets, args)

    model = swm.wm.utils.load_pretrained(str(args.warm_start), cache_dir=str(AILAB))
    freeze = freeze_dynamics_stack(model)
    if (
        freeze["trainable_parameter_tensors"] != EXPECTED_TRAINABLE_TENSORS
        or freeze["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(
            "trainable parameter contract changed: "
            f"expected=({EXPECTED_TRAINABLE_TENSORS},{EXPECTED_TRAINABLE_PARAMETERS}), "
            f"actual=({freeze['trainable_parameter_tensors']},{freeze['trainable_parameters']})"
        )
    frozen_before = frozen_state_hashes(model)
    freeze["frozen_sha256_before"] = dict(frozen_before)
    plan = _plan(args, protocol, measurement, validation, datasets, freeze)
    if args.resume:
        existing_plan = _load_json(output_dir / "run_plan.json")
        if existing_plan != plan:
            raise RuntimeError(
                "--resume rejected because canonical run plan/data/code/config changed"
            )
        split_identity = _validate_episode_split(
            output_dir / "episode_split.npz", datasets
        )
    else:
        atomic_write_json(output_dir / "run_plan.json", plan)
        split_identity = _write_episode_split(output_dir / "episode_split.npz", datasets)

    config_path = args.warm_start.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    model_config = OmegaConf.create(_load_json(config_path))
    optimizer = {
        "model_opt": {
            "modules": r"model\.(predictor|action_encoder)(?:\.|$)",
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
        forward=partial(play_v1_forward, cfg=None),
        optim=optimizer,
    )
    configure_manual_gradient_clipping(
        module, "model_opt", args.gradient_clip_val, "norm"
    )
    callbacks, stopline = PlayV1Callbacks.create(
        run_id=args.run_id,
        model_config=model_config,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        frozen_before=frozen_before,
        expert_loader=expert_stopline_loader,
        train_generator=generator,
        curve_every=args.log_every_n_steps,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        max_epochs=-1,
        max_steps=args.max_steps,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir=str(TENSORBOARD_ROOT),
            name="",
            version=args.run_id,
            default_hp_metric=False,
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
        ckpt_path=str(recovery_checkpoint) if args.resume else None,
    )

    history = stopline.history
    completed_steps = int(trainer.global_step)
    _validate_history(history, completed_steps, args.run_id)
    assert history is not None
    triggered = bool(history["triggered"])
    final_record = history["records"][-1]
    passed = (
        not triggered
        and completed_steps == args.max_steps
        and not _stopline_exceeded(float(final_record["relative_increase"]))
    )
    if not triggered and not passed:
        raise RuntimeError(
            f"training ended without stopline failure or complete pass: step={completed_steps}"
        )
    expected_source_examples = {
        "expert": FORMAL_EXPERT_BATCH * completed_steps,
        "play": FORMAL_PLAY_BATCH * completed_steps,
    }
    source_examples = _validate_source_examples(module, completed_steps)
    final_weights = checkpoint_dir / "weights_final.pt"
    if passed and not final_weights.is_file():
        raise FileNotFoundError(final_weights)
    if triggered and final_weights.exists():
        raise RuntimeError("STOPLINE_FAIL must not publish weights_final.pt")
    frozen_after = assert_frozen_hashes(model, frozen_before)
    completed = {
        "format_version": "cube_play_v1_completed_v1",
        "status": "PASS" if passed else "STOPLINE_FAIL",
        "offline_gate_authorized": passed,
        "retry_allowed": False,
        "run_id": args.run_id,
        "global_step": completed_steps,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - started,
        "infrastructure_resumed": bool(args.resume),
        "final_weights": file_identity(final_weights) if passed else None,
        "stopped_weights": (
            file_identity(stopline.stopped_weights)
            if stopline.stopped_weights is not None else None
        ),
        "live_stopline_checkpoint": (
            file_identity(stopline.live_checkpoint)
            if stopline.live_checkpoint is not None else None
        ),
        "batch": {"expert": 80, "play": 48, "total": 128},
        "source_examples_seen": source_examples,
        "source_examples_expected": expected_source_examples,
        "source_examples_exact": source_examples == expected_source_examples,
        "masked_augmentation_runtime": _runtime_mask_stats(module),
        "loss_contract": dict(LOSS_CONTRACT),
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "frozen_integrity": {
            "before": dict(frozen_before),
            "after": frozen_after,
            "exact_match": frozen_after == dict(frozen_before),
        },
        "episode_split": split_identity,
        "stopline": {
            "baseline_teacher_pred_loss": history["baseline"]["teacher_pred_loss"],
            "final_teacher_pred_loss": final_record["teacher_pred_loss"],
            "relative_increase": final_record["relative_increase"],
            "threshold": STOPLINE_THRESHOLD,
            "triggered": triggered,
            "history": file_identity(output_dir / "stopline_history.json"),
            "event": (
                file_identity(stopline.event_path)
                if stopline.event_path is not None else None
            ),
        },
    }
    atomic_write_json(output_dir / "completed.json", completed)
    print(
        json.dumps(
            {
                "status": completed["status"],
                "offline_gate_authorized": passed,
                "global_step": completed_steps,
                "expert_stopline": completed["stopline"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 3


def parser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description=__doc__)
    parser_.add_argument("--run-id", default=FORMAL_RUN_ID)
    parser_.add_argument("--expert-dataset", type=Path, default=EXPERT_DATASET)
    parser_.add_argument("--play-manifest", type=Path, default=PLAY_MANIFEST)
    parser_.add_argument("--manifest", type=Path, default=FORMAL_MANIFEST)
    parser_.add_argument(
        "--measurement1-segments", type=Path, default=MEASUREMENT1_SEGMENTS
    )
    parser_.add_argument("--warm-start", type=Path, default=WARM_WEIGHTS)
    parser_.add_argument("--normalizers", type=Path, default=NORMALIZERS)
    parser_.add_argument("--seed", type=int, default=3072)
    parser_.add_argument("--split-seed", type=int, default=3072)
    parser_.add_argument("--train-fraction", type=float, default=0.9)
    parser_.add_argument("--max-steps", type=int, default=5000)
    parser_.add_argument("--batch-size", type=int, default=128)
    parser_.add_argument("--num-workers", type=int, default=6)
    parser_.add_argument("--prefetch-factor", type=int, default=3)
    parser_.add_argument("--hue-probability", type=float, default=0.8)
    parser_.add_argument("--max-hue-delta", type=float, default=0.5)
    parser_.add_argument("--learning-rate", type=float, default=1e-5)
    parser_.add_argument("--weight-decay", type=float, default=1e-3)
    parser_.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser_.add_argument("--precision", choices=("bf16-mixed",), default="bf16-mixed")
    parser_.add_argument("--limit-val-batches", type=int, default=25)
    parser_.add_argument("--log-every-n-steps", type=int, default=20)
    parser_.add_argument("--dry-run", action="store_true")
    parser_.add_argument("--synthetic-smoke", action="store_true")
    parser_.add_argument("--model-smoke", action="store_true")
    parser_.add_argument(
        "--data-smoke",
        action="store_true",
        help="CPU scan of all 4,352 clean stopline samples and one frozen train batch",
    )
    parser_.add_argument(
        "--resume",
        action="store_true",
        help="resume only the canonical arm from its atomic 500-step infrastructure scene",
    )
    return parser_


def main(argv: Sequence[str] | None = None) -> int:
    parser_ = parser()
    args = parser_.parse_args(argv)
    modes = sum((args.synthetic_smoke, args.model_smoke, args.data_smoke, args.dry_run))
    if modes > 1:
        parser_.error("synthetic/model/data/dry-run modes are mutually exclusive")
    if args.resume and modes:
        parser_.error("--resume is formal infrastructure recovery and cannot combine with smoke/dry-run")
    if not modes and args.run_id != FORMAL_RUN_ID:
        parser_.error(
            f"formal play-v1 is one non-retry arm; run id must be {FORMAL_RUN_ID!r}"
        )
    frozen = {
        "--batch-size": (args.batch_size, 128),
        "--seed": (args.seed, 3072),
        "--split-seed": (args.split_seed, 3072),
        "--train-fraction": (args.train_fraction, 0.9),
        "--max-steps": (args.max_steps, 5000),
        "--num-workers": (args.num_workers, 6),
        "--prefetch-factor": (args.prefetch_factor, 3),
        "--hue-probability": (args.hue_probability, 0.8),
        "--max-hue-delta": (args.max_hue_delta, 0.5),
        "--learning-rate": (args.learning_rate, 1e-5),
        "--weight-decay": (args.weight_decay, 1e-3),
        "--gradient-clip-val": (args.gradient_clip_val, 1.0),
        "--limit-val-batches": (args.limit_val_batches, 25),
    }
    mismatch = {
        name: {"expected": expected, "actual": actual}
        for name, (actual, expected) in frozen.items()
        if actual != expected
    }
    if mismatch:
        parser_.error(f"frozen play-v1 contract mismatch: {mismatch}")
    if not modes and not torch.cuda.is_available():
        parser_.error("formal play-v1 training requires CUDA bf16")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
