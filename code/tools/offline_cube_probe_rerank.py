#!/usr/bin/env python3
"""Exact-JEPA reranking of three frozen Memory-Seed Cube populations.

The primary candidates, stored latent costs, CEM means, scalers, and physical
labels all come from matching Memory-Seed artifacts.  Older unseeded color
audits supply only condition-matched initial/goal pixels because the compact
seeded pool file intentionally does not duplicate images.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import cube_probe_common as common


OUTPUT_PARENT = common.AILAB_ROOT / "outputs/eval/cube/probe_cost"
SEEDED_POOL_FILES = {
    "red": common.AILAB_ROOT
    / "outputs/eval/cube/memory_seed/red_seeded/first_cycle_pool.npz",
    "blue_v2": common.AILAB_ROOT
    / "outputs/eval/cube/memory_seed/blue_v2_seeded/first_cycle_pool.npz",
    "yellow_v2": common.AILAB_ROOT
    / "outputs/eval/cube/memory_seed/yellow_v2_seeded/first_cycle_pool.npz",
}
PHYSICAL_POOL_DIRS = {
    condition: common.AILAB_ROOT
    / f"outputs/audit/cube_memory_seed_pool_{condition}"
    for condition in common.CONDITIONS
}


def _case_dirs(root: Path) -> list[Path]:
    cases = []
    for env in common.AUDIT_ENVS:
        matches = list(root.glob(f"env_{env:02d}_row_*"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one audit case for env={env} under {root}, got {matches}"
            )
        cases.append(matches[0])
    return cases


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_candidate_outcomes(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 300:
        raise ValueError(f"candidate outcome CSV must have 300 rows: {path}")
    candidate_idx = np.asarray([int(row["candidate_idx"]) for row in rows], dtype=np.int64)
    if not np.array_equal(candidate_idx, np.arange(300)):
        raise ValueError(
            f"candidate IDs must be exactly ordered 0..299: {path}"
        )
    result = {
        "candidate_idx": candidate_idx,
        "is_memory_slot_1_to_10": np.asarray(
            [_parse_bool(row["is_memory_slot_1_to_10"]) for row in rows],
            dtype=bool,
        ),
        "latent_cost": np.asarray([float(row["latent_cost"]) for row in rows]),
        "latent_rank_1based": np.asarray(
            [int(row["latent_rank_1based"]) for row in rows], dtype=np.int64
        ),
        "min_goal_distance_m": np.asarray(
            [float(row["min_goal_distance_m"]) for row in rows]
        ),
        "final_goal_distance_m": np.asarray(
            [float(row["final_goal_distance_m"]) for row in rows]
        ),
        "final_success": np.asarray(
            [_parse_bool(row["final_success"]) for row in rows], dtype=bool
        ),
        "ever_success": np.asarray(
            [_parse_bool(row["ever_success"]) for row in rows], dtype=bool
        ),
    }
    for key in ("latent_cost", "min_goal_distance_m", "final_goal_distance_m"):
        common.finite_or_raise(f"{path}:{key}", result[key])
    expected_slots = np.zeros(300, dtype=bool)
    expected_slots[1:11] = True
    if not np.array_equal(result["is_memory_slot_1_to_10"], expected_slots):
        raise ValueError(f"memory slot flags are not exactly candidate IDs 1..10: {path}")
    return result


def _validate_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.yaw_weight < 0:
        raise ValueError("--yaw-weight must be nonnegative")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive")
    if not args.dataset.is_file() or not args.manifest.is_file() or not args.probe.is_file():
        raise FileNotFoundError(
            f"missing dataset/manifest/probe: {args.dataset}, {args.manifest}, {args.probe}"
        )
    formal_rows, _ = common.load_formal_rows(args.manifest)
    result: dict[str, dict[str, Any]] = {}
    for condition in common.CONDITIONS:
        seeded_pool_path = common.ensure_data_disk(
            args.seeded_pool[condition], f"{condition} seeded first-cycle pool"
        )
        physical_root = common.ensure_data_disk(
            args.physical_pool_dir[condition], f"{condition} seeded physical pool"
        )
        visual_root = common.ensure_data_disk(
            args.visual_audit_dir[condition], f"{condition} visual reference audit"
        )
        aggregate_path = physical_root / "aggregate_summary.json"
        if not seeded_pool_path.is_file() or not aggregate_path.is_file():
            raise FileNotFoundError(
                f"missing seeded pool/physical aggregate: {seeded_pool_path}, {aggregate_path}"
            )
        seeded = np.load(seeded_pool_path, mmap_mode="r", allow_pickle=False)
        expected_shapes = {
            "env_indices": (12,),
            "dataset_rows": (12,),
            "eval_episodes": (12,),
            "candidates_normalized": (12, 300, 5, 25),
            "latent_costs": (12, 300),
            "cem_mean_normalized": (12, 5, 25),
            "action_scaler_mean": (5,),
            "action_scaler_scale": (5,),
        }
        if set(seeded.files) != set(expected_shapes):
            raise ValueError(
                f"seeded first-cycle pool fields mismatch for {condition}: {seeded.files}"
            )
        for key, shape in expected_shapes.items():
            if seeded[key].shape != shape:
                raise ValueError(
                    f"seeded pool {key} shape mismatch: expected={shape}, actual={seeded[key].shape}"
                )
        env_indices = np.asarray(seeded["env_indices"], dtype=np.int64)
        expected_envs = np.asarray(common.AUDIT_ENVS, dtype=np.int64)
        if not np.array_equal(env_indices, expected_envs):
            raise ValueError(
                f"seeded pool env order mismatch: expected={expected_envs.tolist()}, "
                f"actual={env_indices.tolist()}"
            )
        if not np.array_equal(
            np.asarray(seeded["dataset_rows"], dtype=np.int64), formal_rows[env_indices]
        ):
            raise ValueError(f"seeded pool rows are not frozen formal rows: {seeded_pool_path}")
        for key in (
            "candidates_normalized",
            "latent_costs",
            "cem_mean_normalized",
            "action_scaler_mean",
            "action_scaler_scale",
        ):
            common.finite_or_raise(f"{condition} seeded pool {key}", np.asarray(seeded[key]))
        if np.any(np.asarray(seeded["action_scaler_scale"]) <= 0):
            raise ValueError(f"seeded action scaler has nonpositive scale: {seeded_pool_path}")

        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if int(aggregate.get("num_cases", -1)) != 12:
            raise ValueError(f"seeded physical aggregate must contain 12 cases: {aggregate_path}")
        protocol = aggregate.get("protocol", {})
        if protocol.get("fixed_env_indices") != list(common.AUDIT_ENVS):
            raise ValueError(f"physical aggregate env protocol mismatch: {aggregate_path}")
        formal_seeded_input = Path(protocol.get("formal_seeded_input", "")).resolve()
        if formal_seeded_input != seeded_pool_path.parent.resolve():
            raise ValueError(
                "physical labels are not paired to the selected formal seeded input: "
                f"expected={seeded_pool_path.parent}, actual={formal_seeded_input}"
            )
        expected_cem = common.BASELINE_CEM_MEAN_EVER_SUCCESS[condition]
        if int(aggregate.get("cem_mean_ever_success_count", -1)) != expected_cem:
            raise ValueError(
                f"seeded CEM mean count mismatch for {condition}: "
                f"expected={expected_cem}, actual={aggregate.get('cem_mean_ever_success_count')}"
            )

        physical_cases = _case_dirs(physical_root)
        visual_cases = _case_dirs(visual_root)
        bundles = []
        for pool_index, (physical_case, visual_case) in enumerate(
            zip(physical_cases, visual_cases, strict=True)
        ):
            physical_summary_path = physical_case / "summary.json"
            visual_population_path = visual_case / "population.npz"
            visual_capture_path = visual_case / "capture_meta.json"
            required = (
                physical_case / "candidate_outcomes.csv",
                physical_summary_path,
                visual_population_path,
                visual_capture_path,
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"seeded/visual case incomplete: {missing}")
            physical_summary = json.loads(
                physical_summary_path.read_text(encoding="utf-8")
            )
            capture = json.loads(visual_capture_path.read_text(encoding="utf-8"))
            env_idx = int(capture.get("env_idx", -1))
            expected_env = int(env_indices[pool_index])
            expected_row = int(seeded["dataset_rows"][pool_index])
            if env_idx != expected_env:
                raise ValueError(
                    f"visual reference env mismatch: expected={expected_env}, actual={env_idx}"
                )
            if int(capture.get("dataset_row", -1)) != int(formal_rows[env_idx]):
                raise ValueError(f"visual reference row is not frozen: {visual_case}")
            if capture.get("checkpoint") != args.checkpoint:
                raise ValueError(
                    f"visual reference checkpoint mismatch in {visual_case}: "
                    f"expected={args.checkpoint}, actual={capture.get('checkpoint')}"
                )
            if (
                int(physical_summary.get("env_idx", -1)) != expected_env
                or int(physical_summary.get("dataset_row", -1)) != expected_row
            ):
                raise ValueError(f"seeded physical summary pairing mismatch: {physical_case}")
            if Path(physical_summary.get("pool_source", "")).resolve() != seeded_pool_path.resolve():
                raise ValueError(f"seeded physical summary pool_source mismatch: {physical_case}")
            if condition != "red":
                visual = capture.get("visual_protocol", {})
                expected_color = condition.split("_")[0]
                if visual.get("cube_color") != expected_color or visual.get("goal_type") != "recolor":
                    raise ValueError(
                        f"visual reference protocol mismatch for {condition}: {visual_case}"
                    )
            labels = _read_candidate_outcomes(
                physical_case / "candidate_outcomes.csv"
            )
            stored_cost = np.asarray(seeded["latent_costs"][pool_index])
            if not np.allclose(labels["latent_cost"], stored_cost, rtol=0.0, atol=1e-5):
                raise ValueError(
                    f"physical CSV latent costs differ from seeded pool: {physical_case}"
                )
            if not np.array_equal(
                labels["latent_rank_1based"], common.rank_1based(stored_cost)
            ):
                raise ValueError(f"physical CSV latent ranks mismatch: {physical_case}")
            visual_population = np.load(
                visual_population_path, mmap_mode="r", allow_pickle=False
            )
            if visual_population["initial_pixels"].shape != (224, 224, 3) or visual_population["goal_pixels"].shape != (224, 224, 3):
                raise ValueError(f"visual reference pixel shape mismatch: {visual_case}")
            if not np.array_equal(
                np.asarray(visual_population["action_scaler_mean"]),
                np.asarray(seeded["action_scaler_mean"]),
            ) or not np.array_equal(
                np.asarray(visual_population["action_scaler_scale"]),
                np.asarray(seeded["action_scaler_scale"]),
            ):
                raise ValueError(
                    f"seeded/visual action scaler mismatch: {physical_case}/{visual_case}"
                )
            bundles.append(
                {
                    "pool_index": pool_index,
                    "env_idx": expected_env,
                    "dataset_row": expected_row,
                    "physical_case": physical_case,
                    "visual_case": visual_case,
                }
            )
        result[condition] = {
            "seeded_pool_path": seeded_pool_path,
            "physical_root": physical_root,
            "visual_root": visual_root,
            "aggregate_path": aggregate_path,
            "bundles": bundles,
        }
    return result


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = common.ensure_output_child(path, OUTPUT_PARENT, "offline rerank output")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _audit_identity(
    seeded_pool_path: Path, physical_case: Path, visual_case: Path
) -> dict[str, Any]:
    return {
        "seeded_first_cycle_pool": common.file_identity(seeded_pool_path),
        "physical_candidate_outcomes": common.file_identity(
            physical_case / "candidate_outcomes.csv"
        ),
        "physical_summary": common.file_identity(physical_case / "summary.json"),
        "visual_reference_population": common.file_identity(
            visual_case / "population.npz"
        ),
        "visual_reference_capture": common.file_identity(
            visual_case / "capture_meta.json"
        ),
    }


def _distribution(values: Sequence[float | int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"distribution requires a nonempty finite vector, got {array}")
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _rank_distribution(values: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = _distribution(values)
    array = np.asarray(values, dtype=np.int64)
    result.update(
        {
            "top1_count": int(np.count_nonzero(array <= 1)),
            "top5_count": int(np.count_nonzero(array <= 5)),
            "top30_count": int(np.count_nonzero(array <= 30)),
        }
    )
    return result


def _stored_recomputed_diagnostics(
    stored_cost: np.ndarray, recomputed_cost: np.ndarray
) -> dict[str, Any]:
    """Summarize numerical/order parity without asserting identical ordering."""

    from scipy.stats import rankdata

    stored = np.asarray(stored_cost, dtype=np.float64)
    recomputed = np.asarray(recomputed_cost, dtype=np.float64)
    if stored.shape != (300,) or recomputed.shape != (300,):
        raise ValueError(
            f"JEPA ordering diagnostics require two (300,) arrays, got "
            f"{stored.shape}/{recomputed.shape}"
        )
    common.finite_or_raise("stored JEPA cost", stored)
    common.finite_or_raise("recomputed JEPA cost", recomputed)
    stored_rank = common.rank_1based(stored)
    recomputed_rank = common.rank_1based(recomputed)
    stored_rank_average = rankdata(stored, method="average")
    recomputed_rank_average = rankdata(recomputed, method="average")
    spearman = float(np.corrcoef(stored_rank_average, recomputed_rank_average)[0, 1])
    stored_top30 = set(np.argsort(stored, kind="stable")[:30].tolist())
    recomputed_top30 = set(np.argsort(recomputed, kind="stable")[:30].tolist())
    overlap = len(stored_top30 & recomputed_top30)
    union = len(stored_top30 | recomputed_top30)
    return {
        "stored_top1_candidate": int(np.argmin(stored)),
        "recomputed_top1_candidate": int(np.argmin(recomputed)),
        "top1_agreement": bool(np.argmin(stored) == np.argmin(recomputed)),
        "spearman_rank_correlation": spearman,
        "top30_overlap_count": overlap,
        "top30_jaccard": float(overlap / union),
        "max_rank_displacement": int(np.max(np.abs(stored_rank - recomputed_rank))),
        "mean_rank_displacement": float(np.mean(np.abs(stored_rank - recomputed_rank))),
        "ordering_claim": "diagnostic_only_not_full_exact_ordering",
    }


def _physical_best_record(
    candidate: int,
    criterion: str,
    distance_m: np.ndarray,
    probe_rank: np.ndarray,
    stored_rank: np.ndarray,
    recomputed_rank: np.ndarray,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "candidate": int(candidate),
        "distance_m": float(distance_m[candidate]),
        "probe_cost_rank_1based": int(probe_rank[candidate]),
        "stored_jepa_cost_rank_1based": int(stored_rank[candidate]),
        "recomputed_jepa_cost_rank_1based": int(recomputed_rank[candidate]),
        "argmin_tie_break": "numpy_argmin_first_occurrence",
    }


def _condition_rank_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    physical = {}
    for best_key in ("min_goal_distance", "final_goal_distance"):
        physical[best_key] = {
            rank_name: _rank_distribution(
                [record["physical_best"][best_key][field] for record in records]
            )
            for rank_name, field in (
                ("probe_cost", "probe_cost_rank_1based"),
                ("stored_jepa_cost", "stored_jepa_cost_rank_1based"),
                ("recomputed_jepa_cost", "recomputed_jepa_cost_rank_1based"),
            )
        }
    ordering = [record["stored_vs_recomputed_jepa"] for record in records]
    return {
        "physical_best_rank_distributions": physical,
        "stored_vs_recomputed_jepa": {
            "top1_agreement_count": int(sum(x["top1_agreement"] for x in ordering)),
            "top1_agreement_rate": float(np.mean([x["top1_agreement"] for x in ordering])),
            "spearman_rank_correlation": _distribution(
                [x["spearman_rank_correlation"] for x in ordering]
            ),
            "top30_overlap_count": _distribution(
                [x["top30_overlap_count"] for x in ordering]
            ),
            "top30_jaccard": _distribution([x["top30_jaccard"] for x in ordering]),
            "max_rank_displacement": _distribution(
                [x["max_rank_displacement"] for x in ordering]
            ),
            "ordering_claim": "diagnostic_only_not_full_exact_ordering",
        },
    }


def _case_rerank(
    condition: str,
    bundle: dict[str, Any],
    seeded_pool: Any,
    seeded_pool_path: Path,
    model: Any,
    probe: common.LoadedProbe,
    h5: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    pool_index = int(bundle["pool_index"])
    physical_case = Path(bundle["physical_case"])
    visual_case = Path(bundle["visual_case"])
    visual_population = np.load(
        visual_case / "population.npz", allow_pickle=False
    )
    labels = _read_candidate_outcomes(
        physical_case / "candidate_outcomes.csv"
    )
    physical_summary = json.loads(
        (physical_case / "summary.json").read_text(encoding="utf-8")
    )
    capture = json.loads(
        (visual_case / "capture_meta.json").read_text(encoding="utf-8")
    )
    candidates = np.asarray(
        seeded_pool["candidates_normalized"][pool_index], dtype=np.float32
    )
    stored_latent = np.asarray(
        seeded_pool["latent_costs"][pool_index], dtype=np.float32
    )
    cem_mean_normalized = np.asarray(
        seeded_pool["cem_mean_normalized"][pool_index], dtype=np.float32
    )
    initial_pixels = np.asarray(
        visual_population["initial_pixels"], dtype=np.uint8
    )
    goal_pixels = np.asarray(visual_population["goal_pixels"], dtype=np.uint8)
    goal_xyz = np.asarray(visual_population["goal_position"], dtype=np.float32)
    goal_row = int(capture["goal_row"])
    h5_goal_xyz = np.asarray(h5["privileged_block_0_pos"][goal_row], dtype=np.float32)
    if not np.array_equal(goal_xyz, h5_goal_xyz):
        raise ValueError(
            f"visual reference goal xyz differs from H5 goal row: {visual_case}"
        )
    goal_yaw = float(np.asarray(h5["privileged_block_0_yaw"][goal_row]).reshape(-1)[0])

    terminal = common.exact_candidate_terminal_embeddings(
        model,
        initial_pixels,
        candidates,
        args.device,
        args.rollout_batch_size,
    )
    if terminal.shape != (300, common.LEWM_CONTROL_LATENT_DIM):
        raise RuntimeError(
            "exact rollout did not return the 192D control latent: "
            f"actual={tuple(terminal.shape)}"
        )
    with torch.inference_mode():
        goal_embedding = common.encode_pixels(
            model, goal_pixels[None], args.device
        )[0].float()
        latent_recomputed = common.exact_latent_cost(terminal, goal_embedding)
        predicted = probe(terminal)
        goal_xyz_t = torch.as_tensor(goal_xyz, device=args.device)
        goal_yaw_t = torch.as_tensor(goal_yaw, device=args.device)
        probe_cost = common.probe_physical_cost(
            predicted,
            goal_xyz_t,
            goal_yaw_t,
            args.yaw_weight,
        )
        predicted_np = predicted.detach().cpu().float().numpy()
        probe_cost_np = probe_cost.detach().cpu().float().numpy()
        latent_np = latent_recomputed.detach().cpu().float().numpy()

    common.finite_or_raise("predicted block state", predicted_np)
    common.finite_or_raise("probe cost", probe_cost_np)
    latent_abs = np.abs(latent_np - stored_latent)
    reference_tolerance = np.maximum(1e-3, 5e-5 * np.abs(stored_latent))
    # Stored costs were produced in a separate GPU process.  Replaying the
    # exact computational graph is not bit-identical across devices/kernels;
    # strict closeness is therefore reported, while only a gross mismatch
    # (consistent with wrong pixels/actions/model) aborts the protocol.
    gross_guard_tolerance = np.maximum(0.5, 5e-3 * np.abs(stored_latent))
    if np.any(latent_abs > gross_guard_tolerance):
        index = int(np.argmax(latent_abs - gross_guard_tolerance))
        raise RuntimeError(
            "gross latent replay mismatch: "
            f"condition={condition}, case={physical_case.name}, candidate={index}, "
            f"stored={stored_latent[index]}, recomputed={latent_np[index]}, "
            f"abs_error={latent_abs[index]}, tolerance={gross_guard_tolerance[index]}"
        )

    ever = np.asarray(labels["ever_success"], dtype=bool)
    min_goal_distance = np.asarray(labels["min_goal_distance_m"], dtype=np.float64)
    final_goal_distance = np.asarray(labels["final_goal_distance_m"], dtype=np.float64)
    common.finite_or_raise("minimum physical goal distance", min_goal_distance)
    common.finite_or_raise("final physical goal distance", final_goal_distance)
    final = np.asarray(labels["final_success"], dtype=bool)
    order = np.argsort(probe_cost_np, kind="stable")
    top1 = int(order[0])
    top5 = order[:5].astype(np.int64)
    latent_top1 = int(np.argmin(stored_latent))
    xyz_error = np.linalg.norm(predicted_np[:, :3] - goal_xyz[None], axis=1)
    yaw_error = np.abs(common.wrap_angle_np(predicted_np[:, 3] - goal_yaw))
    probe_rank = common.rank_1based(probe_cost_np)
    stored_rank = common.rank_1based(stored_latent)
    recomputed_rank = common.rank_1based(latent_np)
    physical_best = {
        "min_goal_distance": _physical_best_record(
            int(np.argmin(min_goal_distance)),
            "primary_argmin_min_goal_distance_m",
            min_goal_distance,
            probe_rank,
            stored_rank,
            recomputed_rank,
        ),
        "final_goal_distance": _physical_best_record(
            int(np.argmin(final_goal_distance)),
            "secondary_argmin_final_goal_distance_m",
            final_goal_distance,
            probe_rank,
            stored_rank,
            recomputed_rank,
        ),
    }
    ordering_diagnostics = _stored_recomputed_diagnostics(
        stored_latent, latent_np
    )
    summary = {
        "condition": condition,
        "population_protocol": "memory_seeded_first_cycle_final300",
        "seeded_pool_index": pool_index,
        "env_idx": int(capture["env_idx"]),
        "dataset_row": int(capture["dataset_row"]),
        "episode_idx": int(capture["episode_idx"]),
        "goal_row": goal_row,
        "probe_top1_candidate": top1,
        "probe_top1_ever_success": bool(ever[top1]),
        "probe_top1_final_success": bool(final[top1]),
        "probe_top1_xyz_error_to_goal_mm": float(xyz_error[top1] * 1000.0),
        "probe_top1_yaw_error_to_goal_rad": float(yaw_error[top1]),
        "probe_top5_candidates": top5,
        "probe_top5_any_ever_success": bool(np.any(ever[top5])),
        "probe_top5_any_final_success": bool(np.any(final[top5])),
        "probe_top5_uniform_expected_ever_success": float(np.mean(ever[top5])),
        "probe_top5_uniform_expected_final_success": float(np.mean(final[top5])),
        "probe_top5_mean_action_status": "not_evaluated_requires_new_simulation",
        "latent_top1_candidate": latent_top1,
        "latent_top1_ever_success": bool(ever[latent_top1]),
        "cem_mean_ever_success": bool(physical_summary["cem_mean_ever_success"]),
        "cem_mean_final_success": bool(physical_summary["cem_mean_final_success"]),
        "pool_any_ever_success": bool(np.any(ever)),
        "exact_latent_max_abs_error": float(np.max(latent_abs)),
        "latent_replay_reference_tolerance_count": int(
            np.count_nonzero(latent_abs <= reference_tolerance)
        ),
        "latent_replay_reference_tolerance_rate": float(
            np.mean(latent_abs <= reference_tolerance)
        ),
        "latent_replay_gross_guard_max_ratio": float(
            np.max(latent_abs / gross_guard_tolerance)
        ),
        "physical_best": physical_best,
        "stored_vs_recomputed_jepa": ordering_diagnostics,
        "audit_inputs": _audit_identity(
            seeded_pool_path, physical_case, visual_case
        ),
    }
    arrays = {
        "predicted_block4d": predicted_np,
        "probe_cost": probe_cost_np,
        "probe_rank_1based": probe_rank,
        "latent_cost_stored": stored_latent,
        "latent_cost_recomputed": latent_np,
        "latent_rank_stored_1based": stored_rank,
        "latent_rank_recomputed_1based": recomputed_rank,
        "min_goal_distance_m": min_goal_distance,
        "final_goal_distance_m": final_goal_distance,
        "ever_success": ever,
        "final_success": final,
        "goal_xyz": goal_xyz,
        "goal_yaw": np.asarray(goal_yaw, dtype=np.float32),
        "probe_top1_candidate": np.asarray(top1, dtype=np.int64),
        "probe_top5_candidates": top5,
        "candidates_normalized": candidates,
        "cem_mean_normalized": cem_mean_normalized,
        "action_scaler_mean": np.asarray(
            seeded_pool["action_scaler_mean"], dtype=np.float64
        ),
        "action_scaler_scale": np.asarray(
            seeded_pool["action_scaler_scale"], dtype=np.float64
        ),
    }
    return summary, arrays


def run(args: argparse.Namespace) -> int:
    common.configure_storage()
    inputs = _validate_inputs(args)

    import hdf5plugin  # noqa: F401 - register dataset compression filters
    import h5py
    import torch
    import stable_worldmodel as swm

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("offline exact rollout requires the requested CUDA device")
    probe = common.LoadedProbe(args.probe, args.device)
    common.validate_checkpoint_dataset_link(probe, args.probe_dataset_metadata)
    model = swm.wm.utils.load_pretrained(
        args.checkpoint, cache_dir=str(common.AILAB_ROOT)
    )
    model = model.to(args.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    actual_model_sha = common.torch_module_sha256(model)
    expected_model_sha = probe.payload["world_model_state_sha256"]
    if actual_model_sha != expected_model_sha:
        raise ValueError(
            "world model differs from probe embedding source: "
            f"expected={expected_model_sha}, actual={actual_model_sha}"
        )
    if int(probe.payload["input_dim"]) != int(model.projector.out_features if hasattr(model.projector, "out_features") else probe.payload["input_dim"]):
        # The first actual call remains the authoritative dimension check for
        # Identity/projector variants lacking ``out_features``.
        raise ValueError("probe/model embedding dimension mismatch")

    output = _prepare_output(args.output, args.overwrite)
    all_cases: dict[str, list[dict[str, Any]]] = {}
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        for condition in common.CONDITIONS:
            condition_output = output / condition
            condition_output.mkdir(parents=True, exist_ok=True)
            summaries = []
            source = inputs[condition]
            seeded_pool_path = Path(source["seeded_pool_path"])
            seeded_pool = np.load(
                seeded_pool_path, mmap_mode="r", allow_pickle=False
            )
            for bundle in source["bundles"]:
                print(
                    f"rerank {condition}/env_{bundle['env_idx']:02d}_"
                    f"row_{bundle['dataset_row']} [memory-seeded]"
                )
                summary, arrays = _case_rerank(
                    condition,
                    bundle,
                    seeded_pool,
                    seeded_pool_path,
                    model,
                    probe,
                    h5,
                    args,
                )
                stem = f"env_{summary['env_idx']:02d}_row_{summary['dataset_row']}"
                np.savez_compressed(condition_output / f"{stem}.npz", **arrays)
                common.write_json(condition_output / f"{stem}.json", summary)
                summaries.append(summary)
            all_cases[condition] = summaries

    by_condition = {}
    gate_passed = True
    for condition in common.CONDITIONS:
        records = all_cases[condition]
        actual_cem = sum(x["cem_mean_ever_success"] for x in records)
        expected_cem = common.BASELINE_CEM_MEAN_EVER_SUCCESS[condition]
        if actual_cem != expected_cem:
            raise RuntimeError(
                f"frozen CEM mean count mismatch for {condition}: "
                f"expected={expected_cem}, actual={actual_cem}"
            )
        probe_top1 = sum(x["probe_top1_ever_success"] for x in records)
        passed = probe_top1 >= expected_cem
        gate_passed &= passed
        rank_diagnostics = _condition_rank_diagnostics(records)
        by_condition[condition] = {
            "num_cases": 12,
            "probe_top1_ever_success_count": probe_top1,
            "probe_top1_final_success_count": sum(x["probe_top1_final_success"] for x in records),
            "latent_top1_ever_success_count": sum(x["latent_top1_ever_success"] for x in records),
            "cem_mean_ever_success_count": actual_cem,
            "pool_any_ever_success_count": sum(x["pool_any_ever_success"] for x in records),
            "probe_top5_any_ever_success_count": sum(x["probe_top5_any_ever_success"] for x in records),
            "probe_top5_any_final_success_count": sum(x["probe_top5_any_final_success"] for x in records),
            "probe_top5_uniform_expected_ever_success_count": float(sum(x["probe_top5_uniform_expected_ever_success"] for x in records)),
            "probe_top5_uniform_expected_final_success_count": float(sum(x["probe_top5_uniform_expected_final_success"] for x in records)),
            "gate_threshold": expected_cem,
            "gate_passed": passed,
            "max_exact_latent_abs_error": max(x["exact_latent_max_abs_error"] for x in records),
            "latent_replay_reference_tolerance_rate": _distribution(
                [x["latent_replay_reference_tolerance_rate"] for x in records]
            ),
            "latent_replay_gross_guard_max_ratio": max(
                x["latent_replay_gross_guard_max_ratio"] for x in records
            ),
            **rank_diagnostics,
        }
    pooled = {
        key: sum(by_condition[c][key] for c in common.CONDITIONS)
        for key in (
            "probe_top1_ever_success_count",
            "probe_top1_final_success_count",
            "latent_top1_ever_success_count",
            "cem_mean_ever_success_count",
            "pool_any_ever_success_count",
            "probe_top5_any_ever_success_count",
            "probe_top5_any_final_success_count",
            "probe_top5_uniform_expected_ever_success_count",
            "probe_top5_uniform_expected_final_success_count",
        )
    }
    pooled["num_cases"] = 36
    pooled.update(
        _condition_rank_diagnostics(
            [record for condition in common.CONDITIONS for record in all_cases[condition]]
        )
    )
    report = {
        "format_version": "cube_probe_offline_rerank_v1",
        "protocol": {
            "conditions": list(common.CONDITIONS),
            "audit_env_indices": list(common.AUDIT_ENVS),
            "num_candidates_per_case": 300,
            "population_protocol": (
                "memory-seeded formal first-cycle final 300; candidates, stored "
                "latent costs, CEM mean, and scaler come from first_cycle_pool.npz"
            ),
            "physical_label_protocol": (
                "matching cube_memory_seed_pool candidate_outcomes.csv and aggregate"
            ),
            "visual_reference_protocol": (
                "condition-matched unseeded audit contributes only initial/goal pixels "
                "and goal pose; it does not contribute candidates, costs, mean, or labels"
            ),
            "rollout": "exact LeWM JEPA rollout of memory-seeded stored final population",
            "stored_vs_recomputed_ordering": (
                "reported with top1 agreement, tie-aware Spearman, top30 overlap/Jaccard, "
                "and max rank displacement; numerical replay does not claim full exact ordering"
            ),
            "latent_replay_numerics": (
                "exact model/rollout API with fresh info dictionaries; strict GPU-process "
                "reference closeness is diagnostic, while only a gross mismatch aborts"
            ),
            "primary_cost": "squared Euclidean predicted block xyz to privileged goal xyz",
            "supervised_probe_target": list(common.TARGET_NAMES),
            "yaw_weight": args.yaw_weight,
            "yaw_role": "auxiliary diagnostic; excluded from primary ranking when weight=0",
            "top5_semantics": {
                "any": "coverage: whether at least one of five stored candidates succeeds",
                "uniform_expected": "mean stored-candidate success under uniform choice among top5",
                "mean_action": "not evaluated; requires a new simulator rollout",
            },
        },
        "probe": probe.provenance(),
        "world_model_checkpoint": args.checkpoint,
        "world_model_state_sha256": actual_model_sha,
        "dataset": common.file_identity(args.dataset, include_sha256=False),
        "manifest": common.file_identity(args.manifest),
        "primary_inputs": {
            condition: {
                "seeded_first_cycle_pool": common.file_identity(
                    Path(inputs[condition]["seeded_pool_path"])
                ),
                "seeded_physical_aggregate": common.file_identity(
                    Path(inputs[condition]["aggregate_path"])
                ),
                "seeded_physical_root": str(inputs[condition]["physical_root"]),
                "visual_reference_root_nonpopulation_fields_only": str(
                    inputs[condition]["visual_root"]
                ),
            }
            for condition in common.CONDITIONS
        },
        "by_condition": by_condition,
        "pooled": pooled,
        "gate": {
            "definition": (
                "probe-cost top1 ever-success on the Memory-Seed final300 >= "
                "matching frozen Memory-Seed CEM-mean ever-success independently "
                "in every color"
            ),
            "thresholds": common.BASELINE_CEM_MEAN_EVER_SUCCESS,
            "passed_all_colors": gate_passed,
            "online_evaluation_authorized": gate_passed,
        },
        "cases": all_cases,
        "script": common.file_identity(Path(__file__)),
    }
    common.write_json(output / "summary.json", report)
    rows = []
    diagnostic_rows = []
    for condition in common.CONDITIONS:
        s = by_condition[condition]
        rows.append(
            [condition, f"{s['probe_top1_ever_success_count']}/12", f"{s['cem_mean_ever_success_count']}/12", f"{s['probe_top5_any_ever_success_count']}/12", f"{s['probe_top5_uniform_expected_ever_success_count']:.3f}/12", "PASS" if s["gate_passed"] else "FAIL"]
        )
        ordering = s["stored_vs_recomputed_jepa"]
        min_best = s["physical_best_rank_distributions"]["min_goal_distance"]
        diagnostic_rows.append(
            [
                condition,
                f"{ordering['top1_agreement_count']}/12",
                f"{ordering['spearman_rank_correlation']['median']:.6f}",
                f"{ordering['top30_overlap_count']['median']:.1f}",
                f"{ordering['top30_jaccard']['median']:.6f}",
                f"{ordering['max_rank_displacement']['max']:.0f}",
                f"{min_best['probe_cost']['median']:.1f}",
                f"{min_best['stored_jepa_cost']['median']:.1f}",
                f"{min_best['recomputed_jepa_cost']['median']:.1f}",
            ]
        )
    (output / "REPORT.md").write_text(
        "# Cube Memory-Seed probe-cost offline reranking\n\n"
        + common.markdown_table(
            ["Condition", "Probe top1 ever", "CEM mean ever", "Top5-any ever", "Top5 uniform expected ever", "Gate"],
            rows,
        )
        + "\n\n## Physical-best and stored/recomputed diagnostics\n\n"
        + common.markdown_table(
            [
                "Condition",
                "JEPA top1 agree",
                "Spearman median",
                "Top30 overlap median",
                "Top30 Jaccard median",
                "Max rank displacement",
                "Physical-min best probe rank median",
                "Physical-min best stored rank median",
                "Physical-min best recomputed rank median",
            ],
            diagnostic_rows,
        )
        + "\n\nStored/recomputed JEPA ordering statistics are diagnostics; they do not claim full exact ordering.\n"
        + "\nA top-five mean action is **not** inferable from stored outcomes and requires a new simulator rollout.\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate"], sort_keys=True))
    print(output)
    return 0 if gate_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline exact-JEPA Cube probe-cost reranking")
    parser.add_argument("--probe", type=Path, default=common.PROBE_MODEL_DEFAULT / "mlp.pt")
    parser.add_argument("--probe-dataset-metadata", type=Path, default=common.PROBE_DATA_DEFAULT / "metadata.json")
    parser.add_argument("--dataset", type=Path, default=common.DATASET_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST_DEFAULT)
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-batch-size", type=int, default=300)
    parser.add_argument("--yaw-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=common.OFFLINE_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    for condition in common.CONDITIONS:
        parser.add_argument(
            f"--seeded-pool-{condition.replace('_', '-')}",
            dest=f"seeded_pool_{condition}",
            type=Path,
            default=SEEDED_POOL_FILES[condition],
            help="primary memory-seeded first_cycle_pool.npz",
        )
        parser.add_argument(
            f"--physical-pool-{condition.replace('_', '-')}",
            dest=f"physical_pool_{condition}",
            type=Path,
            default=PHYSICAL_POOL_DIRS[condition],
            help="matching memory-seeded candidate physical outcomes",
        )
        parser.add_argument(
            f"--visual-audit-{condition.replace('_', '-')}",
            f"--audit-{condition.replace('_', '-')}",
            dest=f"visual_audit_{condition}",
            type=Path,
            default=common.AUDIT_DIRS[condition],
            help="condition-matched pixel/goal reference only; not the ranked population",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.seeded_pool = {
        condition: getattr(args, f"seeded_pool_{condition}")
        for condition in common.CONDITIONS
    }
    args.physical_pool_dir = {
        condition: getattr(args, f"physical_pool_{condition}")
        for condition in common.CONDITIONS
    }
    args.visual_audit_dir = {
        condition: getattr(args, f"visual_audit_{condition}")
        for condition in common.CONDITIONS
    }
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
