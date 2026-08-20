#!/usr/bin/env python3
"""Cube-only masked HSV augmentation for Route 2.1.

The mask intentionally matches the frozen offline goal-recolor protocol:
``hue > 0.9``, ``saturation > 0.4``, and ``value > 0.15``.  Masking and HSV
math are performed in float64; the RGB tensor is converted to float32 only
after pixels outside the mask have been restored element-for-element.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cube_coloraug import hsv_to_rgb, rgb_to_hsv


MASK_HUE_MIN = 0.9
MASK_SATURATION_MIN = 0.4
MASK_VALUE_MIN = 0.15
MASK_METADATA_PREFIX = "_maskedaug_"


def canonical_rgb64(pixels: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return canonical float64 RGB in [0,1] without changing layout."""
    tensor = pixels if isinstance(pixels, torch.Tensor) else torch.as_tensor(pixels)
    if tensor.ndim < 3 or tensor.shape[-3] != 3:
        raise ValueError(f"expected [...,3,H,W], got {tuple(tensor.shape)}")
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float64) / 255.0
    if not tensor.is_floating_point():
        raise TypeError(f"unsupported pixel dtype: {tensor.dtype}")
    rgb = tensor.to(torch.float64)
    if rgb.numel() and (float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0):
        raise ValueError("floating pixels must be in [0,1] before masked hue rotation")
    return rgb


def frozen_red_mask(rgb64: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the exact frozen red-pixel mask and return ``(mask, hsv64)``."""
    if rgb64.dtype != torch.float64:
        raise TypeError("frozen_red_mask requires float64 RGB")
    hsv64 = rgb_to_hsv(rgb64)
    hue, saturation, value = hsv64.unbind(dim=-3)
    mask = (
        (hue > MASK_HUE_MIN)
        & (saturation > MASK_SATURATION_MIN)
        & (value > MASK_VALUE_MIN)
    )
    return mask, hsv64


def _frame_counts(mask: torch.Tensor) -> tuple[int, int]:
    frame_masks = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
    nonempty = frame_masks.flatten(1).any(dim=1)
    return int((~nonempty).sum().item()), int(frame_masks.shape[0])


def apply_masked_hue_rotation(
    pixels: torch.Tensor | np.ndarray,
    *,
    apply_shift: bool,
    hue_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply one clip-consistent hue shift only inside the frozen red mask.

    Returns float32 RGB, the boolean mask, and audit statistics.  The output
    outside the mask is restored from the canonical input before conversion to
    float32, so it is exactly equal to ``canonical_rgb64(pixels).float()``.
    """
    if not -0.5 <= float(hue_delta) <= 0.5:
        raise ValueError("hue_delta must be in [-0.5,0.5]")
    rgb64 = canonical_rgb64(pixels)
    mask, hsv64 = frozen_red_mask(rgb64)
    empty_frames, total_frames = _frame_counts(mask)
    if apply_shift:
        shifted_hue = torch.remainder(hsv64.select(-3, 0) + float(hue_delta), 1.0)
        shifted_hsv = hsv64.clone()
        shifted_hsv.select(-3, 0).copy_(shifted_hue)
        shifted_rgb64 = hsv_to_rgb(shifted_hsv).clamp_(0.0, 1.0)
        expanded_mask = mask.unsqueeze(-3).expand_as(rgb64)
        output64 = torch.where(expanded_mask, shifted_rgb64, rgb64)
    else:
        expanded_mask = mask.unsqueeze(-3).expand_as(rgb64)
        output64 = rgb64.clone()
    outside_equal_float64 = bool(torch.equal(output64[~expanded_mask], rgb64[~expanded_mask]))
    output = output64.to(torch.float32)
    canonical32 = rgb64.to(torch.float32)
    outside_equal_float32 = bool(torch.equal(output[~expanded_mask], canonical32[~expanded_mask]))
    if not outside_equal_float64 or not outside_equal_float32:
        raise RuntimeError("masked augmentation changed at least one pixel outside the mask")
    stats = {
        "applied": bool(apply_shift),
        "hue_delta": float(hue_delta) if apply_shift else 0.0,
        "masked_pixels": int(mask.sum().item()),
        "empty_frames": empty_frames,
        "total_frames": total_frames,
        "outside_equal_float64": outside_equal_float64,
        "outside_equal_float32": outside_equal_float32,
    }
    return output, mask, stats


class RandomMaskedHueRotation:
    """80/20 online masked hue augmentation with one delta per 4-frame clip."""

    def __init__(
        self,
        source: str = "pixels",
        probability: float = 0.8,
        max_delta: float = 0.5,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0,1]")
        if not 0.0 < max_delta <= 0.5:
            raise ValueError("max_delta must be in (0,0.5]")
        self.source = source
        self.probability = float(probability)
        self.max_delta = float(max_delta)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample = dict(sample)
        apply_shift = bool(torch.rand(()).item() < self.probability)
        delta = (
            float(((torch.rand(()) * 2.0 - 1.0) * self.max_delta).item())
            if apply_shift
            else 0.0
        )
        output, _, stats = apply_masked_hue_rotation(
            sample[self.source], apply_shift=apply_shift, hue_delta=delta
        )
        sample[self.source] = output
        sample[f"{MASK_METADATA_PREFIX}empty_frames"] = torch.tensor(
            stats["empty_frames"], dtype=torch.int64
        )
        sample[f"{MASK_METADATA_PREFIX}total_frames"] = torch.tensor(
            stats["total_frames"], dtype=torch.int64
        )
        sample[f"{MASK_METADATA_PREFIX}masked_pixels"] = torch.tensor(
            stats["masked_pixels"], dtype=torch.int64
        )
        sample[f"{MASK_METADATA_PREFIX}applied_clips"] = torch.tensor(
            int(stats["applied"]), dtype=torch.int64
        )
        sample[f"{MASK_METADATA_PREFIX}seen_clips"] = torch.tensor(1, dtype=torch.int64)
        return sample


def _rgb_u8(rgb: torch.Tensor) -> np.ndarray:
    array = (
        rgb.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).round().astype(np.uint8)
    return array


def save_qc_artifacts(
    dataset: Any,
    clip_indices: Sequence[int],
    output_dir: Path,
    *,
    seed: int,
    probability: float = 0.8,
    max_delta: float = 0.5,
    num_frames: int = 10,
) -> dict[str, Any]:
    """Write exactly ten original/mask/augmented PNG triptychs and a sheet."""
    from PIL import Image

    if num_frames != 10:
        raise ValueError("Route2.1 QC is frozen to exactly 10 frames")
    if len(clip_indices) < num_frames:
        raise ValueError("not enough clips for QC")
    output_dir.mkdir(parents=True, exist_ok=False)
    positions = np.linspace(0, len(clip_indices) - 1, num_frames, dtype=np.int64)
    generator = torch.Generator().manual_seed(seed + 21)
    records: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for qc_index, position in enumerate(positions):
        dataset_index = int(clip_indices[int(position)])
        pixels = dataset[dataset_index]["pixels"]
        frame = pixels[0]
        apply_shift = bool(torch.rand((), generator=generator).item() < probability)
        delta = (
            float(((torch.rand((), generator=generator) * 2.0 - 1.0) * max_delta).item())
            if apply_shift
            else 0.0
        )
        augmented, mask, stats = apply_masked_hue_rotation(
            frame, apply_shift=apply_shift, hue_delta=delta
        )
        original = _rgb_u8(canonical_rgb64(frame).float())
        masked = original.copy()
        mask_np = mask.detach().cpu().numpy()
        masked[mask_np] = np.asarray([0, 255, 0], dtype=np.uint8)
        panel_array = np.concatenate((original, masked, _rgb_u8(augmented)), axis=1)
        panel = Image.fromarray(panel_array, mode="RGB")
        filename = f"frame_{qc_index:02d}.png"
        panel.save(output_dir / filename)
        panels.append(panel)
        records.append(
            {
                "qc_index": qc_index,
                "dataset_clip_index": dataset_index,
                "source_frame_in_clip": 0,
                "png": filename,
                **stats,
            }
        )
    width, height = panels[0].size
    sheet = Image.new("RGB", (width * 2, height * 5))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * width, (index // 2) * height))
    sheet.save(output_dir / "contact_sheet.png")
    payload = {
        "protocol": {
            "mask": "float64 HSV: hue>0.9, saturation>0.4, value>0.15",
            "probability": probability,
            "identity_probability": 1.0 - probability,
            "hue_delta_turns": [-max_delta, max_delta],
            "panels": "original | green mask overlay | augmented",
            "contact_sheet_grid": {"columns": 2, "rows": 5},
        },
        "num_png_frames": len(records),
        "empty_mask_frames": sum(int(row["empty_frames"]) for row in records),
        "all_outside_equal_float64": all(row["outside_equal_float64"] for row in records),
        "all_outside_equal_float32": all(row["outside_equal_float32"] for row in records),
        "records": records,
    }
    (output_dir / "qc.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
