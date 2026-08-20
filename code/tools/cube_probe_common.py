#!/usr/bin/env python3
"""Shared, provenance-heavy helpers for the Cube block-state probe route.

This module is intentionally independent of the installed ``stable_worldmodel``
package.  The data builder, trainer, offline reranker, and gated online evaluator
use the same checkpoint and metric contracts through this file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


AILAB_ROOT = Path(__file__).resolve().parents[2]
DATASET_DEFAULT = AILAB_ROOT / "datasets/ogbench/cube_single_expert.h5"
MANIFEST_DEFAULT = AILAB_ROOT / "outputs/audit/cube_cem_manifest.json"
PROBE_DATA_DEFAULT = AILAB_ROOT / "outputs/probe/cube_block4d_v1/dataset"
PROBE_MODEL_DEFAULT = AILAB_ROOT / "models/probes/cube_block4d_v1"
OFFLINE_DEFAULT = AILAB_ROOT / "outputs/eval/cube/probe_cost/offline_v1"
ONLINE_ROOT = AILAB_ROOT / "outputs/eval/cube/probe_cost/online"
TMP_ROOT = AILAB_ROOT.parent / "tmp"

TARGET_NAMES = ("block_x", "block_y", "block_z", "block_yaw")
LEWM_CONTROL_LATENT_DIM = 192
AUDIT_ENVS = (0, 1, 2, 6, 7, 11, 12, 23, 26, 37, 38, 49)
CONDITIONS = ("red", "blue_v2", "yellow_v2")
AUDIT_DIRS = {
    "red": AILAB_ROOT / "outputs/audit/cube_cem_300",
    "blue_v2": AILAB_ROOT / "outputs/audit/cube_cem_300_blue_v2",
    "yellow_v2": AILAB_ROOT / "outputs/audit/cube_cem_300_yellow_v2",
}
BASELINE_CEM_MEAN_EVER_SUCCESS = {"red": 9, "blue_v2": 8, "yellow_v2": 7}


def configure_storage() -> None:
    defaults = {
        "STABLEWM_HOME": str(AILAB_ROOT),
        "HF_HOME": str(AILAB_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
        "MUJOCO_GL": "egl",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def torch_module_sha256(module: Any) -> str:
    """Stable fingerprint over tensor names, shapes, dtypes, and bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        # Flatten first because PyTorch cannot reinterpret a zero-dimensional
        # scalar as bytes when element sizes differ (e.g. int64 -> uint8).
        digest.update(
            value.reshape(-1)
            .view(dtype=__import__("torch").uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def file_identity(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = sha256_file(path)
    return result


def ensure_data_disk(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    disk = AILAB_ROOT.parent.resolve()
    resolved = path.resolve()
    if resolved != disk and disk not in resolved.parents:
        raise ValueError(f"{label} must be on /root/autodl-tmp: {resolved}")
    return resolved


def ensure_output_child(path: Path, root: Path, label: str) -> Path:
    path = ensure_data_disk(path, label)
    root = root.resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{label} must be a concrete child of {root}: {path}")
    if path.is_symlink():
        raise ValueError(f"refusing symlink {label}: {path}")
    return path


def load_formal_rows(manifest_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = ensure_data_disk(manifest_path, "manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = np.asarray(payload["formal_rows"], dtype=np.int64)
    if rows.shape != (50,) or np.any(np.diff(rows) <= 0):
        raise ValueError(
            f"formal rows must be 50 strictly increasing indices, got {rows.shape}"
        )
    return rows, payload


def excluded_formal_episodes(dataset: Path, manifest: Path) -> np.ndarray:
    import h5py

    rows, _ = load_formal_rows(manifest)
    with h5py.File(dataset, "r", swmr=True) as h5:
        episodes = np.asarray(h5["ep_idx"][rows], dtype=np.int64)
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError("frozen formal rows unexpectedly reuse an episode")
    return episodes


def wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_torch(angle: Any) -> Any:
    import torch

    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def quaternion_wxyz_to_yaw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"expected wxyz quaternion last dimension 4, got {q.shape}")
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def probe_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 4:
        raise ValueError(
            f"probe metric arrays must both be (N,4), got {target.shape}/{prediction.shape}"
        )
    residual = prediction - target
    residual[:, 3] = wrap_angle_np(residual[:, 3])
    xyz_error = np.linalg.norm(residual[:, :3], axis=1)
    mae = np.mean(np.abs(residual), axis=0)
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    r2 = []
    for dim in range(4):
        truth = target[:, dim]
        pred = target[:, dim] + residual[:, dim]
        denom = float(np.sum((truth - truth.mean()) ** 2))
        r2.append(float(1.0 - np.sum((pred - truth) ** 2) / denom) if denom else float("nan"))
    return {
        "num_samples": int(target.shape[0]),
        "mae_per_dimension": dict(zip(TARGET_NAMES, mae.tolist(), strict=True)),
        "rmse_per_dimension": dict(zip(TARGET_NAMES, rmse.tolist(), strict=True)),
        "r2_per_dimension": dict(zip(TARGET_NAMES, r2, strict=True)),
        "median_xyz_error_mm": float(np.median(xyz_error) * 1000.0),
        "mean_xyz_error_mm": float(np.mean(xyz_error) * 1000.0),
        "yaw_error_is_circular": True,
    }


def make_probe(kind: str, input_dim: int, hidden_dim: int = 256) -> Any:
    from torch import nn

    if kind == "linear":
        return nn.Linear(input_dim, 4)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
    raise ValueError(f"unknown probe kind: {kind}")


class LoadedProbe:
    """Normalize embeddings and decode a four-dimensional physical state."""

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        import torch

        checkpoint = ensure_data_disk(checkpoint, "probe checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = {
            "format_version",
            "model_kind",
            "input_dim",
            "hidden_dim",
            "input_mean",
            "input_scale",
            "target_mean",
            "target_scale",
            "state_dict",
            "dataset_metadata_sha256",
            "world_model_state_sha256",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"probe checkpoint missing fields: {sorted(missing)}")
        if payload["format_version"] != "cube_block4d_probe_v1":
            raise ValueError(f"unsupported probe format: {payload['format_version']}")
        if int(payload["input_dim"]) != LEWM_CONTROL_LATENT_DIM:
            raise ValueError(
                "Cube probe must consume the 192D projector-after-CLS control latent: "
                f"actual={payload['input_dim']}"
            )
        self.path = checkpoint
        self.payload = payload
        self.device = torch.device(device)
        self.model = make_probe(
            payload["model_kind"], int(payload["input_dim"]), int(payload["hidden_dim"])
        )
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.input_mean = torch.as_tensor(
            payload["input_mean"], dtype=torch.float32, device=self.device
        )
        self.input_scale = torch.as_tensor(
            payload["input_scale"], dtype=torch.float32, device=self.device
        )
        self.target_mean = torch.as_tensor(
            payload["target_mean"], dtype=torch.float32, device=self.device
        )
        self.target_scale = torch.as_tensor(
            payload["target_scale"], dtype=torch.float32, device=self.device
        )
        if self.input_mean.shape != (payload["input_dim"],):
            raise ValueError("checkpoint input normalization shape mismatch")
        if self.target_mean.shape != (4,) or torch.any(self.input_scale <= 0) or torch.any(self.target_scale <= 0):
            raise ValueError("checkpoint normalization is malformed")

    def __call__(self, embedding: Any) -> Any:
        normalized = (embedding.float() - self.input_mean) / self.input_scale
        prediction = self.model(normalized)
        physical = prediction * self.target_scale + self.target_mean
        physical[..., 3] = wrap_angle_torch(physical[..., 3])
        return physical

    def provenance(self) -> dict[str, Any]:
        return {
            "checkpoint": file_identity(self.path),
            "model_kind": self.payload["model_kind"],
            "input_dim": int(self.payload["input_dim"]),
            "hidden_dim": int(self.payload["hidden_dim"]),
            "target_names": list(TARGET_NAMES),
            "dataset_metadata_sha256": self.payload["dataset_metadata_sha256"],
            "world_model_state_sha256": self.payload["world_model_state_sha256"],
        }


def normalized_image_tensor(pixels: np.ndarray, device: str, dtype: Any | None = None) -> Any:
    """Match the 224x224 ImageNet transform used by Cube evaluation."""

    import torch

    array = np.asarray(pixels)
    if array.ndim != 4 or array.shape[1:] != (224, 224, 3) or array.dtype != np.uint8:
        raise ValueError(f"pixels must be uint8 (N,224,224,3), got {array.shape}/{array.dtype}")
    value = torch.from_numpy(array.copy()).to(device=device).permute(0, 3, 1, 2)
    value = value.float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    value = (value - mean) / std
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value


def encode_pixels(model: Any, pixels: np.ndarray, device: str) -> Any:
    value = normalized_image_tensor(pixels, device=device)
    encoded = model.encode({"pixels": value[:, None]})["emb"][:, 0]
    return encoded


def exact_candidate_terminal_embeddings(
    model: Any,
    initial_pixels: np.ndarray,
    candidates_normalized: np.ndarray,
    device: str,
    batch_size: int = 300,
) -> Any:
    """Run the real JEPA rollout, preserving the audit's action tensor semantics."""

    import torch

    candidates = np.asarray(candidates_normalized, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[1:] != (5, 25):
        raise ValueError(f"candidates must be (N,5,25), got {candidates.shape}")
    base_pixels = normalized_image_tensor(
        np.asarray(initial_pixels, dtype=np.uint8)[None], device=device
    )[None]  # (B=1,T=1,C,H,W)
    outputs = []
    model_dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            actions = torch.from_numpy(candidates[start : start + batch_size]).to(
                device=device, dtype=model_dtype
            )
            sample_count = actions.shape[0]
            info = {
                "pixels": base_pixels.to(dtype=model_dtype)[:, None].expand(
                    1, sample_count, *base_pixels.shape[1:]
                ),
                # JEPA.rollout replaces this with the candidate prefix.  The key
                # is retained to match get_cost's input contract.
                "action": torch.zeros(
                    1, sample_count, 1, 5, device=device, dtype=model_dtype
                ),
            }
            rolled = model.rollout(info, actions[None])
            outputs.append(rolled["predicted_emb"][0, :, -1].detach().float())
    return torch.cat(outputs, dim=0)


def exact_latent_cost(terminal_embedding: Any, goal_embedding: Any) -> Any:
    return ((terminal_embedding - goal_embedding.float()[None]) ** 2).sum(dim=-1)


def circular_yaw_error_torch(predicted: Any, target: Any) -> Any:
    return wrap_angle_torch(predicted - target)


def probe_physical_cost(
    predicted_block4d: Any,
    goal_xyz: Any,
    goal_yaw: Any | None = None,
    yaw_weight: float = 0.0,
) -> Any:
    xyz_cost = ((predicted_block4d[..., :3] - goal_xyz) ** 2).sum(dim=-1)
    if yaw_weight:
        if goal_yaw is None:
            raise ValueError("nonzero yaw weight requires a goal yaw")
        yaw_error = circular_yaw_error_torch(predicted_block4d[..., 3], goal_yaw)
        xyz_cost = xyz_cost + float(yaw_weight) * yaw_error.square()
    return xyz_cost


def validate_checkpoint_dataset_link(probe: LoadedProbe, metadata_path: Path) -> None:
    metadata_path = ensure_data_disk(metadata_path, "probe dataset metadata")
    actual = sha256_file(metadata_path)
    expected = str(probe.payload["dataset_metadata_sha256"])
    if actual != expected:
        raise ValueError(
            "probe dataset/checkpoint provenance mismatch: "
            f"expected={expected}, actual={actual}, path={metadata_path}"
        )


def rank_1based(cost: np.ndarray) -> np.ndarray:
    values = np.asarray(cost)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(1, len(values) + 1)
    return rank


def validate_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition}")
    return condition


def condition_visual_protocol(condition: str) -> tuple[str, str]:
    validate_condition(condition)
    return ("red", "matched") if condition == "red" else (condition.split("_")[0], "recolor")


def finite_or_raise(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        bad = np.argwhere(~np.isfinite(value))[0].tolist()
        raise ValueError(f"{name} contains nonfinite data at {bad}")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(map(cell, headers)) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines)
