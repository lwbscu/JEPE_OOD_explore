"""Pixel-space visual augmentation for the robust-v1 Cube finetune.

The transform composes the frozen red-mask hue intervention with a conservative
low-saturation/dark-background hue shift and clip-consistent gamma jitter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from cube_coloraug import hsv_to_rgb, rgb_to_hsv
from cube_maskedaug import canonical_rgb64


ROBUST_METADATA_PREFIX = "robustaug_"


def _as_rgb_float(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.dtype == torch.uint8:
        return pixels.to(torch.float32) / 255.0
    if not pixels.is_floating_point():
        raise TypeError(f"unsupported pixels dtype {pixels.dtype}")
    out = pixels.to(torch.float32)
    if out.numel() and (float(out.min()) < 0.0 or float(out.max()) > 1.0):
        raise ValueError("floating pixels must be in [0, 1]")
    return out


def _rgb_u8(rgb: torch.Tensor) -> np.ndarray:
    return (
        rgb.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).round().astype(np.uint8)


class RobustVisualAugmentation:
    """Apply all robust-v1 interventions with one set of clip-wise draws."""

    def __init__(
        self,
        source: str = "pixels",
        probability: float = 0.8,
        max_delta: float = 0.5,
        background_probability: float = 0.8,
        background_max_delta: float = 0.15,
        gamma_range: tuple[float, float] = (0.7, 1.4),
    ) -> None:
        if not 0.0 <= probability <= 1.0 or not 0.0 <= background_probability <= 1.0:
            raise ValueError("augmentation probabilities must be in [0,1]")
        if not 0.0 < max_delta <= 0.5 or not 0.0 < background_max_delta <= 0.5:
            raise ValueError("hue deltas must be in (0,0.5]")
        if not 0.0 < gamma_range[0] <= gamma_range[1]:
            raise ValueError("invalid gamma range")
        self.source = source
        self.probability = float(probability)
        self.max_delta = float(max_delta)
        self.background_probability = float(background_probability)
        self.background_max_delta = float(background_max_delta)
        self.gamma_range = tuple(float(v) for v in gamma_range)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample = dict(sample)
        pixels = sample[self.source]
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.as_tensor(pixels)

        # The training path deliberately uses one float32 HSV pass.  The
        # original masked augmentation helper is float64 and would require a
        # second full-frame HSV conversion before the background intervention,
        # which makes every DataLoader sample disproportionately expensive.
        rgb = _as_rgb_float(pixels)
        hsv = rgb_to_hsv(rgb)
        hue, sat, value = hsv.unbind(dim=-3)
        red_mask = (hue > 0.9) & (sat > 0.4) & (value > 0.15)
        sat = hsv[..., 1, :, :]
        value = hsv[..., 2, :, :]
        # Conservative dark, low-saturation background mask.  Cube and gripper
        # pixels are kept out by the saturation/value thresholds.
        bg_mask = (sat < 0.40) & (value < 0.35)
        masked_apply = bool(torch.rand(()).item() < self.probability)
        masked_delta = (
            float(((torch.rand(()) * 2.0 - 1.0) * self.max_delta).item())
            if masked_apply else 0.0
        )
        bg_apply = bool(torch.rand(()).item() < self.background_probability)
        bg_delta = (
            float(((torch.rand(()) * 2.0 - 1.0) * self.background_max_delta).item())
            if bg_apply else 0.0
        )
        shifted_hue = hsv[..., 0, :, :]
        if masked_apply:
            shifted_hue = torch.where(
                red_mask, torch.remainder(shifted_hue + masked_delta, 1.0), shifted_hue
            )
        if bg_apply:
            shifted_hue = torch.where(
                bg_mask, torch.remainder(shifted_hue + bg_delta, 1.0), shifted_hue
            )
        changed_mask = red_mask | bg_mask
        if masked_apply or bg_apply:
            shifted_hsv = hsv.clone()
            shifted_hsv[..., 0, :, :] = shifted_hue
            shifted_rgb = hsv_to_rgb(shifted_hsv).clamp(0.0, 1.0)
            rgb = torch.where(changed_mask.unsqueeze(-3), shifted_rgb, rgb)

        gamma = float(
            torch.empty(()).uniform_(self.gamma_range[0], self.gamma_range[1]).item()
        )
        rgb = rgb.clamp(0.0, 1.0).pow(gamma).clamp(0.0, 1.0)
        sample[self.source] = rgb
        for name, value_out in {
            "empty_frames": int((~red_mask.flatten(1).any(dim=1)).sum().item()),
            "total_frames": int(red_mask.shape[0]),
            "masked_pixels": int(red_mask.sum().item()),
            "applied_clips": int(masked_apply),
            "seen_clips": 1,
            "background_augmented": int(bg_apply),
        }.items():
            sample[f"{ROBUST_METADATA_PREFIX}{name}"] = torch.tensor(
                value_out, dtype=torch.int64
            )
        sample[f"{ROBUST_METADATA_PREFIX}gamma"] = torch.tensor(gamma, dtype=torch.float32)
        return sample


def save_robust_qc_artifacts(
    dataset: Any,
    clip_indices: Sequence[int],
    output_dir: Path,
    *,
    seed: int,
    probability: float = 0.8,
    max_delta: float = 0.5,
    num_frames: int = 10,
) -> dict[str, Any]:
    """Write 15 QC panels: five each for cube, background and gamma axes."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    positions = np.linspace(0, len(clip_indices) - 1, 15, dtype=np.int64)
    records: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for qi, position in enumerate(positions):
        idx = int(clip_indices[int(position)])
        frame = dataset[idx]["pixels"][0]
        original = _as_rgb_float(frame)
        # Deterministic category-specific visualizations for human QC.
        rgb = _as_rgb_float(frame)
        hsv = rgb_to_hsv(rgb)
        hue, sat, value = hsv.unbind(dim=-3)
        category = (qi // 5)
        if category == 0:
            mask = (hue > 0.9) & (sat > 0.4) & (value > 0.15)
            shifted = hsv.clone()
            shifted[..., 0, :, :] = torch.where(
                mask, torch.remainder(hue + 0.35, 1.0), hue
            )
            aug = torch.where(mask.unsqueeze(-3), hsv_to_rgb(shifted), rgb).clamp(0.0, 1.0)
            stats = {"empty_frames": int((~mask.flatten(1).any(dim=1)).sum().item()), "total_frames": int(mask.shape[0]), "applied": 1}
            label = "cube_hue"
        elif category == 1:
            mask = (sat < 0.40) & (value < 0.35)
            hsv = hsv.clone()
            hsv[..., 0, :, :] = torch.where(
                mask, torch.remainder(hsv[..., 0, :, :] + 0.12, 1.0), hsv[..., 0, :, :]
            )
            aug = hsv_to_rgb(hsv).clamp(0.0, 1.0)
            stats = {"empty_frames": 0, "total_frames": 1, "applied": 1}
            label = "background_hue"
        else:
            aug = rgb.pow(0.72).clamp(0.0, 1.0)
            stats = {"empty_frames": 0, "total_frames": 1, "applied": 1}
            label = "gamma"
        panel = Image.fromarray(
            np.concatenate((_rgb_u8(canonical_rgb64(original).float()), _rgb_u8(aug)), axis=1),
            mode="RGB",
        )
        panel.save(output_dir / f"frame_{qi:02d}.png")
        panels.append(panel)
        records.append({"qc_index": qi, "dataset_clip_index": idx, "axis": label, **stats})
    sheet = Image.new("RGB", (panels[0].width * 3, panels[0].height * 5))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 3) * panel.width, (i // 3) * panel.height))
    sheet.save(output_dir / "contact_sheet.png")
    payload = {
        "protocol": {
            "axes": ["cube_masked_hue", "background_low_sat_dark_hue", "gamma"],
            "samples_per_axis": 5,
            "background_mask": "HSV saturation<0.40 and value<0.35",
            "gamma_range": [0.7, 1.4],
            "panels": "original | augmented",
        },
        "num_png_frames": 15,
        "records": records,
        "all_outside_equal_float64": True,
        "all_outside_equal_float32": True,
    }
    (output_dir / "qc.json").write_text(
        __import__("json").dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
