#!/usr/bin/env python3
"""Frozen data and artifact helpers for the single Cube off-policy V3 arm.

V3 deliberately reuses the validated V2 collector, datasets, global episode
split, and strict 96-expert/32-V2 mixture.  This module adds only the immutable
V3 arm/loss declarations and the atomic stopline artifact writer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cube_offpolicy_v2 import (  # Re-export the frozen V2 data contract.
    ACTION_DIM,
    FORMAL_BATCH_SIZE,
    FORMAL_EXPERT_BATCH,
    FORMAL_V2_BATCH,
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
    split_rollouts_by_source_episode,
)


CANONICAL_ARM = "v3_96E_32V2_expert_rollout_disabled"
FORMAL_RUN_ID = "offpolicy_v3_pred_seed3072"
EXPERT_BASELINE = 0.0032953384798020124
STOPLINE_THRESHOLD = 0.10
STOPLINE_INTERVAL = 500
STOPLINE_BATCHES = 34
STOPLINE_EXAMPLES = 4_352
V2_FORMAL_SPLIT_SHA256 = "758ca131803327ac6efc42d6bc9e133e96b149fbc15c884cb64bb0b3db082128"
V2_FORMAL_PROVENANCE_SHA256 = (
    "58a316127e9ef5428174c2b5d73a8e7a868cac15976a8a96e395a601fd088e05"
)
NORMALIZERS_SHA256 = "32ab9a37631f6de612e413f2067b009a669b56bacc700f85b67f251ebe3188b4"
SPLIT_ARRAY_NAMES = (
    "global_train_episodes",
    "global_val_episodes",
    "expert_train_episodes",
    "expert_val_episodes",
    "v2_train_episodes",
    "v2_val_episodes",
    "v2_train_ids",
    "v2_val_ids",
)

LOSS_CONTRACT: dict[str, Any] = {
    "formula": (
        "0.75*E_teacher + 0.25*V2_teacher + "
        "0.09*shared_SIGReg(cat(E[:,:4],V2[:,:4])) + 0.25*0.5*V2_AR5"
    ),
    "expert_teacher_weight": 0.75,
    "v2_teacher_weight": 0.25,
    "shared_sigreg_weight": 0.09,
    "shared_sigreg_frames": 4,
    "expert_rollout_enabled": False,
    "expert_model_rollout_calls": 0,
    "expert_rollout_logged_total": 0.0,
    "expert_rollout_logged_depths": [0.0] * 5,
    "v2_rollout_source_weight": 0.25,
    "v2_ar_weight": 0.5,
    "v2_ar_depth": 5,
    "v2_ar_intermediate_detach": False,
    "target_encoder_stop_gradient": True,
}


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON with the exact canonical encoding used by V3 artifacts."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


LOSS_CONTRACT_SHA256 = canonical_json_sha256(LOSS_CONTRACT)


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically publish one durable JSON artifact in its final directory."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def bind_formal_v2_split(
    source: Path, destination: Path, datasets: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy the exact formal V2 split only after array and digest equality."""
    source = source.expanduser().resolve()
    if sha256_file(source) != V2_FORMAL_SPLIT_SHA256:
        raise RuntimeError(
            "formal V2 split identity changed: "
            f"expected={V2_FORMAL_SPLIT_SHA256}, actual={sha256_file(source)}, position={source}"
        )
    with np.load(source, allow_pickle=False) as saved:
        if set(saved.files) != set(SPLIT_ARRAY_NAMES):
            raise RuntimeError(
                "formal V2 split keys changed: "
                f"expected={sorted(SPLIT_ARRAY_NAMES)}, actual={sorted(saved.files)}, position={source}"
            )
        for name in SPLIT_ARRAY_NAMES:
            expected = np.asarray(datasets[name], dtype=np.int64)
            if not np.array_equal(saved[name], expected):
                raise RuntimeError(
                    f"V3 reconstructed split differs from formal V2 at {name}"
                )
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != V2_FORMAL_SPLIT_SHA256:
            raise RuntimeError("copied formal V2 split digest changed")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_identity(destination)


__all__ = [
    "ACTION_DIM",
    "CANONICAL_ARM",
    "EXPERT_BASELINE",
    "FORMAL_BATCH_SIZE",
    "FORMAL_EXPERT_BATCH",
    "FORMAL_RUN_ID",
    "FORMAL_V2_BATCH",
    "LOSS_CONTRACT",
    "LOSS_CONTRACT_SHA256",
    "NORMALIZERS_SHA256",
    "STOPLINE_BATCHES",
    "STOPLINE_EXAMPLES",
    "STOPLINE_INTERVAL",
    "STOPLINE_THRESHOLD",
    "V2_FORMAL_PROVENANCE_SHA256",
    "V2_FORMAL_SPLIT_SHA256",
    "ExpertRolloutDataset",
    "PlannerRolloutDataset",
    "StrictMixtureDataset",
    "assert_frozen_hashes",
    "atomic_write_json",
    "bind_formal_v2_split",
    "canonical_json_sha256",
    "file_identity",
    "flatten_bundled_source",
    "freeze_predictor_only",
    "frozen_state_hashes",
    "load_planner_manifest",
    "module_state_sha256",
    "sha256_file",
    "split_rollouts_by_source_episode",
]
