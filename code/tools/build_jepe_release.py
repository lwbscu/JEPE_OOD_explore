#!/usr/bin/env python3
"""Assemble explicit GitHub and Hugging Face JEPE OOD release trees."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = ROOT / "release"
GITHUB_ROOT = RELEASE_ROOT / "github_staging/JEPE_OOD_explore"
HF_ROOT = RELEASE_ROOT / "hf_staging/JEPE_OOD_explore"

EXCLUDED_PARTS = {"__pycache__", ".git", "lightning_logs"}
EXCLUDED_SUFFIXES = {".pyc", ".ckpt"}


def _safe_destination(path: Path, root: Path) -> Path:
    root = root.resolve()
    path = path.resolve(strict=False)
    if path == root or root not in path.parents:
        raise ValueError(f"destination escapes release root: {path}")
    return path


def _copy(source: Path, destination: Path, *, hardlink: bool = False) -> None:
    if source.is_symlink():
        raise ValueError(f"release sources may not be symlinks: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if hardlink:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def _iter_tree_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if EXCLUDED_PARTS & set(path.parts) or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        yield path


def _prepare(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"nonempty staging root: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _github_code(destination: Path) -> None:
    source_root = ROOT / "le-wm"
    allowed = []
    allowed.extend(sorted(source_root.glob("*.py")))
    allowed.extend(sorted((source_root / "tools").glob("*.py")))
    allowed.extend(sorted((source_root / "config").rglob("*.yaml")))
    allowed.extend(sorted((source_root / "config").rglob("*.yml")))
    for source in allowed:
        rel = source.relative_to(source_root)
        _copy(source, _safe_destination(destination / "code" / rel, destination))


def _github_docs(destination: Path) -> None:
    reports = set()
    for pattern in ("*REPORT*.md", "*VERDICT*.md", "*DIAGNOSIS*.md"):
        reports.update((ROOT / "outputs").rglob(pattern))
    summary = ROOT / "outputs/eval/cube/ROUTE12_SUMMARY.md"
    if summary.is_file():
        reports.add(summary)
    for source in sorted(reports):
        rel = source.relative_to(ROOT / "outputs")
        _copy(source, _safe_destination(destination / "docs" / rel, destination))


def _github_evidence(destination: Path) -> None:
    exact_video_roots = (
        ROOT / "outputs/eval/cube/pretrained/evidence/videos",
        ROOT / "outputs/eval/pusht_pretrained/evidence/videos",
    )
    for source_root in exact_video_roots:
        for source in sorted(source_root.glob("*.mp4")):
            rel = source.relative_to(ROOT / "outputs")
            _copy(source, _safe_destination(destination / "evidence" / rel, destination))

    exact_images = (
        ROOT / "outputs/eval/cube/ood/goal_compare_env0/agent_blue_t0.png",
        ROOT / "outputs/eval/cube/ood/goal_compare_env0/goal_blue_synthetic.png",
        ROOT / "outputs/eval/cube/ood/goal_compare_env0/goal_red_real_h5.png",
        ROOT / "outputs/eval/cube/goal_ood_curve/success_vs_ood_distance.png",
    )
    for source in exact_images:
        if not source.is_file():
            raise FileNotFoundError(source)
        rel = source.relative_to(ROOT / "outputs")
        _copy(source, _safe_destination(destination / "evidence" / rel, destination))

    image_roots = (ROOT / "outputs",)
    for source_root in image_roots:
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            lowered = source.name.lower()
            include = any(word in lowered for word in ("contact", "comparison", "curve"))
            include |= "/qc/" in source.as_posix() and any(
                name in source.as_posix()
                for name in (
                    "route21_masked_hsv_seed3072",
                    "lewm-cube-robust_v1",
                )
            )
            if not include:
                continue
            rel = source.relative_to(ROOT / "outputs")
            _copy(source, _safe_destination(destination / "evidence" / rel, destination))


def build_github(overwrite: bool) -> None:
    _prepare(GITHUB_ROOT, overwrite)
    _github_code(GITHUB_ROOT)
    _github_docs(GITHUB_ROOT)
    _github_evidence(GITHUB_ROOT)


def _hardlink_tree(source_root: Path, destination_root: Path) -> None:
    for source in _iter_tree_files(source_root):
        rel = source.relative_to(source_root)
        _copy(source, destination_root / rel, hardlink=True)


def build_hf(overwrite: bool) -> None:
    _prepare(HF_ROOT, overwrite)
    weight_map = {
        "coloraug/weights_final.pt": ROOT / "checkpoints/lewm-cube-coloraug/route2_hsv_seed3072/weights_final.pt",
        "maskedaug/weights_final.pt": ROOT / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt",
        "robust_v1/weights_final.pt": ROOT / "checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt",
        "control_noaugment/weights_step_12732.pt": ROOT / "checkpoints/lewm-cube-control_noaugment/control_noaugment_seed3072/weights_step_12732.pt",
        "control_noaugment/weights_final.pt": ROOT / "checkpoints/lewm-cube-control_noaugment/control_noaugment_seed3072/weights_final.pt",
        "offpolicy_v1/primary_weights_final.pt": ROOT / "checkpoints/lewm-cube-offpolicy_v1/offpolicy_v1_pred_seed3072/weights_final.pt",
        "offpolicy_v1/lr5e6_retry_weights_final.pt": ROOT / "checkpoints/lewm-cube-offpolicy_v1/offpolicy_v1_pred_lr5e6_seed3072/weights_final.pt",
    }
    for rel, source in weight_map.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        _copy(source, HF_ROOT / "weights" / rel, hardlink=True)
        config = source.parent / "config.json"
        if config.is_file():
            _copy(config, (HF_ROOT / "weights" / rel).parent / "config.json")

    probe_root = ROOT / "models/probes/cube_robust_v1_xyz"
    if not probe_root.is_dir():
        raise FileNotFoundError(probe_root)
    _hardlink_tree(probe_root, HF_ROOT / "weights/probes/robust_v1")
    embedding_metadata = ROOT / "outputs/probe/cube_robust_v1/dataset/metadata.json"
    if not embedding_metadata.is_file():
        raise FileNotFoundError(embedding_metadata)
    _copy(
        embedding_metadata,
        HF_ROOT / "weights/probes/robust_v1/embedding_dataset_metadata.json",
    )

    for name in ("offpolicy_cube_v1", "offpolicy_cube_v2"):
        _hardlink_tree(ROOT / "datasets" / name, HF_ROOT / "datasets" / name)
    _hardlink_tree(
        ROOT / "outputs/memory_index/cube_expert_v1",
        HF_ROOT / "memory_index/cube_expert_v1",
    )
    github_evidence = GITHUB_ROOT / "evidence"
    if not github_evidence.is_dir():
        raise FileNotFoundError("build GitHub staging before HF staging")
    _hardlink_tree(github_evidence, HF_ROOT / "evidence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("github", "hf", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.target in {"github", "all"}:
        build_github(args.overwrite)
        print(GITHUB_ROOT)
    if args.target in {"hf", "all"}:
        build_hf(args.overwrite)
        print(HF_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
