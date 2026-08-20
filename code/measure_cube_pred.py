#!/usr/bin/env python3
"""Read-only paired Route2 expert pred-loss measurement for robust_v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cube_coloraug import IndexedTransformDataset  # noqa: E402
from train_cube_coloraug import ColumnNormalizer, _heldout_protocol  # noqa: E402


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(args: argparse.Namespace) -> tuple[torch.utils.data.DataLoader, dict[str, object]]:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_img_preprocessor

    dataset = swm.data.load_dataset(
        str(args.dataset),
        transform=None,
        num_steps=6,
        frameskip=5,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    split_path = args.split
    with np.load(split_path, allow_pickle=False) as split:
        if "expert_val_episodes" not in split:
            raise ValueError(f"split lacks expert_val_episodes: {split_path}")
        val_episodes = np.asarray(split["expert_val_episodes"], dtype=np.int64)
    clip_episodes = np.fromiter(
        (int(value[0]) for value in dataset.clip_indices), dtype=np.int64, count=len(dataset.clip_indices)
    )
    indices = np.flatnonzero(np.isin(clip_episodes, val_episodes)).astype(np.int64)
    normalizers = json.loads(args.normalizers.read_text(encoding="utf-8"))
    mean = torch.tensor(normalizers["action"]["mean"], dtype=torch.float32).reshape(1, 5).repeat(1, 5)
    std = torch.tensor(normalizers["action"]["std"], dtype=torch.float32).reshape(1, 5).repeat(1, 5)
    transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=224),
        ColumnNormalizer("action", mean, std),
    )
    view = IndexedTransformDataset(dataset, indices, transform)
    loader = torch.utils.data.DataLoader(
        view, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader, {
        "clip_count": int(len(indices)),
        "episode_count": int(len(np.unique(val_episodes))),
        "episode_sha256": _sha(split_path),
        "normalizers_sha256": _sha(args.normalizers),
    }


def _measure(checkpoint: Path, loader: torch.utils.data.DataLoader, batches: int, device: torch.device) -> float:
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(str(checkpoint), cache_dir=str(PROJECT)).to(device).eval()
    model.requires_grad_(False)
    values: list[float] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= batches:
                break
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                encoded = model.encode(batch)
                predicted = model.predict(encoded["emb"][:, :3], encoded["act_emb"][:, :3])
                # Route2's teacher objective predicts the three future frames
                # available from the model's three-frame context.  Keeping
                # this slice explicit avoids accidentally comparing against
                # the five-step rollout target used by off-policy training.
                target = encoded["emb"][:, 1:4]
                loss = (predicted - target).square().mean()
            values.append(float(loss.float().cpu()))
    if len(values) != batches:
        raise RuntimeError(f"paired measurement has {len(values)} batches, expected {batches}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=PROJECT / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "datasets/ogbench/cube_single_expert.h5")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "outputs/audit/cube_cem_manifest.json")
    parser.add_argument("--split", type=Path, default=PROJECT / "outputs/train/offpolicy_v2/offpolicy_v2_pred_seed3072/episode_split.npz")
    parser.add_argument("--normalizers", type=Path, default=PROJECT / "outputs/train/route21_maskedaug/route21_masked_hsv_seed3072/normalizers.json")
    parser.add_argument("--batches", type=int, default=34)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batches < 1:
        raise ValueError("batches must be positive")
    for path in (args.base, args.checkpoint, args.dataset, args.manifest, args.split, args.normalizers):
        if not path.is_file():
            raise FileNotFoundError(path)
    loader, provenance = _loader(args)
    device = torch.device(args.device)
    base_loss = _measure(args.base, loader, args.batches, device)
    final_loss = _measure(args.checkpoint, loader, args.batches, device)
    relative = final_loss / base_loss - 1.0
    payload = {
        "format_version": "cube_robust_expert_stopline_v1",
        "base_checkpoint": str(args.base.resolve()),
        "base_checkpoint_sha256": _sha(args.base),
        "final_checkpoint": str(args.checkpoint.resolve()),
        "final_checkpoint_sha256": _sha(args.checkpoint),
        "protocol": {"batches": args.batches, "batch_size": args.batch_size, "shuffle": False, "drop_last": True},
        "provenance": provenance,
        "base_pred_loss": base_loss,
        "final_pred_loss": final_loss,
        "relative_increase": relative,
        "threshold_relative_increase": 0.10,
        "status": "PASS" if relative <= 0.10 else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
