#!/usr/bin/env python3
"""Route-2 Cube color augmentation and held-out-episode utilities.

All helpers are standalone so the original LeWM training/evaluation entry
points and installed packages remain untouched.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert float RGB in [0, 1] to HSV, preserving arbitrary leading axes."""
    if rgb.ndim < 3 or rgb.shape[-3] != 3:
        raise ValueError(f"expected [...,3,H,W], got {tuple(rgb.shape)}")
    if not rgb.is_floating_point():
        raise TypeError("rgb_to_hsv expects floating-point input")
    r, g, b = rgb.unbind(dim=-3)
    value, index = rgb.max(dim=-3)
    minimum = rgb.min(dim=-3).values
    chroma = value - minimum
    safe = torch.where(chroma > 0, chroma, torch.ones_like(chroma))
    hue6 = torch.where(
        index == 0,
        torch.remainder((g - b) / safe, 6.0),
        torch.where(index == 1, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    hue = torch.where(chroma > 0, hue6 / 6.0, torch.zeros_like(hue6))
    saturation = torch.where(value > 0, chroma / value, torch.zeros_like(value))
    return torch.stack((hue, saturation, value), dim=-3)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert float HSV to RGB; H wraps modulo one while S and V are unchanged."""
    if hsv.ndim < 3 or hsv.shape[-3] != 3:
        raise ValueError(f"expected [...,3,H,W], got {tuple(hsv.shape)}")
    hue, saturation, value = hsv.unbind(dim=-3)
    hue6 = torch.remainder(hue, 1.0) * 6.0
    sector = torch.floor(hue6).to(torch.int64) % 6
    fraction = hue6 - torch.floor(hue6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    zeros = torch.zeros_like(value)
    r = torch.where(
        sector == 0, value,
        torch.where(sector == 1, q, torch.where(sector == 2, p,
        torch.where(sector == 3, p, torch.where(sector == 4, t,
        torch.where(sector == 5, value, zeros))))),
    )
    g = torch.where(
        sector == 0, t,
        torch.where(sector == 1, value, torch.where(sector == 2, value,
        torch.where(sector == 3, q, torch.where(sector == 4, p,
        torch.where(sector == 5, p, zeros))))),
    )
    b = torch.where(
        sector == 0, p,
        torch.where(sector == 1, p, torch.where(sector == 2, t,
        torch.where(sector == 3, value, torch.where(sector == 4, value,
        torch.where(sector == 5, q, zeros))))),
    )
    return torch.stack((r, g, b), dim=-3)


class RandomHueRotation:
    """Rotate hue online while leaving HSV saturation/value unchanged.

    A single delta is shared by every frame in a sequence, preserving temporal
    color consistency. The output is float RGB in [0, 1].
    """

    def __init__(
        self,
        source: str = "pixels",
        probability: float = 1.0,
        max_delta: float = 0.5,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not 0.0 < max_delta <= 0.5:
            raise ValueError("max_delta must be in (0, 0.5]")
        self.source = source
        self.probability = float(probability)
        self.max_delta = float(max_delta)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        pixels = sample[self.source]
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.as_tensor(pixels)
        if torch.rand(()) >= self.probability:
            return sample
        if pixels.dtype == torch.uint8:
            rgb = pixels.to(torch.float32) / 255.0
        elif pixels.is_floating_point():
            rgb = pixels.to(torch.float32)
            if rgb.numel() and (float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0):
                raise ValueError("floating pixels must be in [0, 1] before hue rotation")
        else:
            raise TypeError(f"unsupported pixel dtype: {pixels.dtype}")
        delta = (torch.rand((), device=rgb.device) * 2.0 - 1.0) * self.max_delta
        hsv = rgb_to_hsv(rgb)
        hsv = torch.cat((torch.remainder(hsv[..., :1, :, :] + delta, 1.0), hsv[..., 1:, :, :]), dim=-3)
        sample[self.source] = hsv_to_rgb(hsv).clamp_(0.0, 1.0)
        return sample


class IndexedTransformDataset(Dataset):
    """Memory-efficient indexed view with a split-specific transform."""

    def __init__(self, dataset: Any, indices: Sequence[int], transform: Callable) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[int(self.indices[index])]
        return self.transform(sample)


def allowed_clip_indices(clip_indices: Sequence[tuple[int, int]], excluded_episodes: set[int]) -> np.ndarray:
    """Return clip indices whose complete sequence belongs to a non-held-out episode."""
    return np.fromiter(
        (index for index, (episode, _) in enumerate(clip_indices) if int(episode) not in excluded_episodes),
        dtype=np.int64,
    )


def split_indices(indices: np.ndarray, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(indices), generator=generator).numpy()
    train_size = math.floor(len(indices) * train_fraction)
    return indices[permutation[:train_size]], indices[permutation[train_size:]]


def streaming_mean_std(values: np.ndarray, included_rows: np.ndarray, chunk_size: int = 131_072) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Unbiased z-score statistics without materializing a huge filtered copy."""
    if values.ndim != 2 or included_rows.shape != (len(values),):
        raise ValueError("values/included_rows shape mismatch")
    total = np.zeros(values.shape[1], dtype=np.float64)
    squares = np.zeros(values.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        chunk = np.asarray(values[start:stop])
        keep = included_rows[start:stop] & ~np.isnan(chunk).any(axis=1)
        selected = chunk[keep].astype(np.float64, copy=False)
        total += selected.sum(axis=0)
        squares += np.square(selected).sum(axis=0)
        count += len(selected)
    if count < 2:
        raise RuntimeError("fewer than two valid rows available for normalization")
    mean = total / count
    variance = np.maximum((squares - count * np.square(mean)) / (count - 1), 0.0)
    std = np.sqrt(variance)
    if np.any(std == 0) or not np.all(np.isfinite(std)):
        raise RuntimeError("normalizer contains zero/non-finite standard deviation")
    return torch.from_numpy(mean[None]).float(), torch.from_numpy(std[None]).float(), count
