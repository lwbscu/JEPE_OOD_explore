#!/usr/bin/env python3
"""Shared contracts for the Cube imagination-error experiment.

Only block XYZ is supervised and scored.  Yaw is intentionally absent from
this protocol even though the earlier Route1 background probe decoded 4D.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AILAB_ROOT = Path(__file__).resolve().parents[2]
LEWM_ROOT = AILAB_ROOT / "le-wm"
DATASET = AILAB_ROOT / "datasets/ogbench/cube_single_expert.h5"
MANIFEST = AILAB_ROOT / "outputs/audit/cube_cem_manifest.json"
OUTPUT_ROOT = AILAB_ROOT / "outputs/eval/cube/imagination_error"
OFFICIAL_CHECKPOINT = AILAB_ROOT / "checkpoints/models--quentinll--lewm-cube/weights.pt"
MASKED_CHECKPOINT = (
    AILAB_ROOT
    / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt"
)
OFFICIAL_EMBEDDINGS = AILAB_ROOT / "outputs/probe/cube_block4d_v1/dataset"
MASKED_EMBEDDINGS = AILAB_ROOT / "outputs/probe/cube_imagination_error_masked_v1/dataset"
PROBE_ROOT = AILAB_ROOT / "models/probes/cube_imagination_error_xyz_v1"
AUDIT_ENVS = (0, 1, 2, 6, 7, 11, 12, 23, 26, 37, 38, 49)
CONDITIONS = ("red", "blue_v2", "yellow_v2")
AUDIT_ROOTS = {
    "red": AILAB_ROOT / "outputs/audit/cube_cem_300",
    "blue_v2": AILAB_ROOT / "outputs/audit/cube_cem_300_blue_v2",
    "yellow_v2": AILAB_ROOT / "outputs/audit/cube_cem_300_yellow_v2",
}
CHECKPOINTS = {"official": OFFICIAL_CHECKPOINT, "masked": MASKED_CHECKPOINT}
EMBEDDING_DATASETS = {"official": OFFICIAL_EMBEDDINGS, "masked": MASKED_EMBEDDINGS}
PROBE_PATHS = {label: PROBE_ROOT / f"{label}.pt" for label in CHECKPOINTS}
TARGET_HUES = {"blue_v2": 0.58, "yellow_v2": 0.12}
LATENT_DIM = 192
XYZ_DIM = 3
HORIZON = 5
ACTION_BLOCK = 5
ACTION_DIM = 5
PROBE_TEST_MEDIAN_LIMIT_MM = 15.0


def configure_storage() -> None:
    defaults = {
        "STABLEWM_HOME": str(AILAB_ROOT),
        "HF_HOME": str(AILAB_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(AILAB_ROOT.parent / "tmp"),
        "MUJOCO_GL": "egl",
    }
    # This process is self-contained; force its caches and temporary files to
    # the persistent data disk even if the login shell inherited system-disk
    # defaults such as /root/.cache or /tmp.
    for key, value in defaults.items():
        os.environ[key] = value
    (AILAB_ROOT.parent / "tmp").mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: jsonable(row.get(field)) for field in fields})
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    result = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if include_sha256:
        result["sha256"] = sha256_file(resolved)
    return result


def ensure_data_disk(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    disk = AILAB_ROOT.parent.resolve()
    if resolved != disk and disk not in resolved.parents:
        raise ValueError(f"{label} must be on /root/autodl-tmp: {resolved}")
    return resolved


def stage_root(smoke: bool) -> Path:
    return OUTPUT_ROOT / "smoke" if smoke else OUTPUT_ROOT


def protect_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    root = OUTPUT_ROOT.resolve()
    for path in paths:
        lexical = path.expanduser()
        if lexical.is_symlink():
            raise ValueError(f"refusing symlink output: {lexical}")
        resolved = ensure_data_disk(path, "imagination-error output")
        if resolved == root or root not in resolved.parents:
            raise ValueError(f"output must be below frozen root {root}: {resolved}")
        if resolved.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {resolved}")


def finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        index = np.argwhere(~np.isfinite(array))[0].tolist()
        raise ValueError(f"{name} contains nonfinite data at {index}")
    return array


def distribution(value: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "rmse": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
    }


def latent_metrics(predicted: Any, actual: Any) -> dict[str, np.ndarray]:
    """Per-row latent drift without reducing over samples."""
    import torch

    predicted = predicted.float()
    actual = actual.float()
    difference = predicted - actual
    output = {
        "latent_l2": torch.linalg.vector_norm(difference, dim=-1).cpu().numpy(),
        "latent_mse": torch.mean(difference.square(), dim=-1).cpu().numpy(),
        "latent_cosine_distance": (
            1.0 - torch.nn.functional.cosine_similarity(predicted, actual, dim=-1)
        ).cpu().numpy(),
    }
    return {name: finite(name, value) for name, value in output.items()}


def xyz_error_mm(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = finite("xyz prediction", np.asarray(prediction, dtype=np.float64))
    target = finite("xyz target", np.asarray(target, dtype=np.float64))
    if prediction.shape != target.shape or prediction.shape[-1] != XYZ_DIM:
        raise ValueError(f"xyz shape mismatch: {prediction.shape}/{target.shape}")
    return np.linalg.norm(prediction - target, axis=-1) * 1000.0


def make_xyz_probe(input_dim: int = LATENT_DIM, hidden_dim: int = 256) -> Any:
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, XYZ_DIM),
    )


class LoadedXYZProbe:
    """Load a checkpoint-specific 192D latent -> block XYZ decoder."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch

        self.path = ensure_data_disk(checkpoint, "xyz probe checkpoint")
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != "cube_imagination_error_xyz_probe_v1":
            raise ValueError(f"unsupported xyz probe format: {payload.get('format_version')}")
        if payload.get("target_names") != ["block_x", "block_y", "block_z"]:
            raise ValueError("probe target contract is not strict block XYZ")
        if int(payload.get("input_dim", -1)) != LATENT_DIM:
            raise ValueError("probe latent dimension mismatch")
        self.payload = payload
        self.device = torch.device(device)
        self.model = make_xyz_probe(LATENT_DIM, int(payload["hidden_dim"]))
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.input_mean = torch.as_tensor(payload["input_mean"], device=self.device)
        self.input_scale = torch.as_tensor(payload["input_scale"], device=self.device)
        self.target_mean = torch.as_tensor(payload["target_mean"], device=self.device)
        self.target_scale = torch.as_tensor(payload["target_scale"], device=self.device)
        if self.input_mean.shape != (LATENT_DIM,) or self.target_mean.shape != (XYZ_DIM,):
            raise ValueError("probe normalization shape mismatch")
        if torch.any(self.input_scale <= 0) or torch.any(self.target_scale <= 0):
            raise ValueError("probe normalization scale must be positive")

    def __call__(self, embedding: Any) -> Any:
        value = (embedding.float() - self.input_mean) / self.input_scale
        return self.model(value) * self.target_scale + self.target_mean

    def provenance(self) -> dict[str, Any]:
        return {
            "checkpoint": file_identity(self.path),
            "model_label": self.payload["model_label"],
            "world_model_state_sha256": self.payload["world_model_state_sha256"],
            "embedding_dataset_metadata_sha256": self.payload[
                "embedding_dataset_metadata_sha256"
            ],
            "target_names": self.payload["target_names"],
            "test_metrics": self.payload["metrics"]["test"],
        }


def xyz_probe_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = finite("probe targets", np.asarray(target, dtype=np.float64))
    prediction = finite("probe predictions", np.asarray(prediction, dtype=np.float64))
    errors = xyz_error_mm(prediction, target)
    residual = prediction - target
    r2 = []
    for dimension in range(XYZ_DIM):
        truth = target[:, dimension]
        denominator = float(np.sum(np.square(truth - truth.mean())))
        r2.append(
            float(1.0 - np.sum(np.square(residual[:, dimension])) / denominator)
            if denominator
            else None
        )
    return {
        "num_samples": int(len(target)),
        "xyz_error_mm": distribution(errors),
        "mae_m": np.mean(np.abs(residual), axis=0).tolist(),
        "rmse_m": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
        "r2": dict(zip(("block_x", "block_y", "block_z"), r2, strict=True)),
        "primary_metric": "xyz_error_mm",
    }


def normalize_uint8_images(pixels: np.ndarray, device: str) -> Any:
    import torch

    array = np.asarray(pixels)
    if array.ndim != 4 or array.shape[1:] != (224, 224, 3) or array.dtype != np.uint8:
        raise ValueError(f"pixels must be uint8 (N,224,224,3), got {array.shape}/{array.dtype}")
    value = torch.from_numpy(array.copy()).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (value - mean) / std


def encode_uint8(model: Any, pixels: np.ndarray, device: str, batch_size: int) -> Any:
    import torch

    outputs = []
    model_dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            normalized = normalize_uint8_images(pixels[start : start + batch_size], device)
            encoded = model.encode({"pixels": normalized.to(model_dtype)[:, None]})["emb"][:, 0]
            outputs.append(encoded.detach().float())
    return torch.cat(outputs, dim=0)


def recolor_red_pixels(pixels: np.ndarray, target_hue: float, chunk_size: int = 32) -> tuple[np.ndarray, int]:
    """Frozen float64 HSV mask/recolor, preserving bytes outside the mask."""
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    source = np.asarray(pixels)
    if source.ndim != 4 or source.shape[1:] != (224, 224, 3) or source.dtype != np.uint8:
        raise ValueError(f"recolor source must be uint8 (N,224,224,3), got {source.shape}/{source.dtype}")
    if not 0.0 <= target_hue < 1.0:
        raise ValueError("target_hue must be in [0,1)")
    output = np.empty_like(source)
    empty = 0
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        rgb = source[start:stop].astype(np.float64) / 255.0
        hsv = rgb_to_hsv(rgb)
        mask = (hsv[..., 0] > 0.9) & (hsv[..., 1] > 0.4) & (hsv[..., 2] > 0.15)
        empty += int(np.count_nonzero(~mask.reshape(len(mask), -1).any(axis=1)))
        recolored_hsv = hsv.copy()
        recolored_hsv[..., 0][mask] = float(target_hue)
        recolored = np.rint(np.clip(hsv_to_rgb(recolored_hsv), 0.0, 1.0) * 255.0).astype(np.uint8)
        recolored[~mask] = source[start:stop][~mask]
        if not np.array_equal(recolored[~mask], source[start:stop][~mask]):
            raise RuntimeError("recolor changed a byte outside the frozen mask")
        output[start:stop] = recolored
    return output, empty


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(map(clean, headers)) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(lines)
