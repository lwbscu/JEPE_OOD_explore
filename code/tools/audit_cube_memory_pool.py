#!/usr/bin/env python3
"""Physically replay the seeded formal first-cycle final population for 12 cases.

This is a CPU/MuJoCo audit only.  It consumes the numerical population saved
by ``eval_memory_seed.py`` and never regenerates CEM samples or writes videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
LEWM_ROOT = HERE.parent
PROJECT_ROOT = LEWM_ROOT.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_cube_memory_index as memory  # noqa: E402
import cube_cem_audit as base_audit  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "datasets/ogbench/cube_single_expert.h5"
DEFAULT_INDEX = PROJECT_ROOT / "outputs/memory_index/cube_expert_v1"
EVAL_ROOT = PROJECT_ROOT / "outputs/eval/cube/memory_seed"
AUDIT_ROOT = PROJECT_ROOT / "outputs/audit"
AUDIT_ENVS = np.asarray([0, 1, 2, 6, 7, 11, 12, 23, 26, 37, 38, 49], dtype=np.int64)
SUCCESS_METERS = 0.04
BASELINE_ROOTS = {
    "red": AUDIT_ROOT / "cube_cem_300",
    "blue": AUDIT_ROOT / "cube_cem_300_blue_v2",
    "yellow": AUDIT_ROOT / "cube_cem_300_yellow_v2",
}


def _name(color: str) -> str:
    return "red" if color == "red" else f"{color}_v2"


def _default_input(color: str) -> Path:
    return EVAL_ROOT / f"{_name(color)}_seeded"


def _default_output(color: str) -> Path:
    return AUDIT_ROOT / f"cube_memory_seed_pool_{_name(color)}"


def _safe_output(path: Path, overwrite: bool) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"refusing symlink output: {raw}")
    resolved = raw.resolve()
    if resolved == AUDIT_ROOT.resolve() or AUDIT_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"output must be a concrete child of {AUDIT_ROOT}: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is nonempty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        return item
    path.write_text(json.dumps(convert(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rollout(env: Any, snapshot: Any, actions: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    base_audit._restore_snapshot(env, snapshot)
    raw = env.unwrapped
    distances = []
    first = None
    terminated = truncated = False
    for step, action in enumerate(actions, start=1):
        _, _, terminated, truncated, _ = env.step(action)
        cube = raw._data.joint("object_joint_0").qpos[:3]
        distance = float(np.linalg.norm(cube - target))
        distances.append(distance)
        if first is None and distance <= SUCCESS_METERS:
            first = step
        if truncated:
            break
    return {
        "executed_steps": len(distances),
        "min_goal_distance_m": float(np.min(distances)),
        "final_goal_distance_m": float(distances[-1]),
        "ever_success": first is not None,
        "final_success": bool(distances[-1] <= SUCCESS_METERS),
        "first_success_step": first,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def run(args: argparse.Namespace) -> int:
    base_audit._configure_storage()
    input_root = (args.input or _default_input(args.color)).expanduser().resolve()
    results_path = input_root / "results.json"
    pool_path = input_root / "first_cycle_pool.npz"
    trace_path = input_root / "memory_trace.json"
    for path in (results_path, pool_path, trace_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    formal = json.loads(results_path.read_text(encoding="utf-8"))
    expected_goal = "matched" if args.color == "red" else "recolor"
    protocol = formal["protocol"]
    if (
        protocol.get("color") != args.color
        or protocol.get("goal_type") != expected_goal
        or protocol.get("selector") != "mean"
    ):
        raise ValueError(f"formal input protocol mismatch: {protocol}")
    pool = np.load(pool_path, allow_pickle=False)
    env_indices = np.asarray(pool["env_indices"], dtype=np.int64)
    if not np.array_equal(env_indices, AUDIT_ENVS):
        raise RuntimeError(
            f"fixed audit env mismatch: expected={AUDIT_ENVS.tolist()}, actual={env_indices.tolist()}"
        )
    candidates = np.asarray(pool["candidates_normalized"], dtype=np.float32)
    costs = np.asarray(pool["latent_costs"], dtype=np.float32)
    rows = np.asarray(pool["dataset_rows"], dtype=np.int64)
    expected_rows = {
        int(case["env_idx"]): int(case["dataset_row"])
        for case in base_audit.AUDIT_CASES
    }
    frozen_rows = np.asarray([expected_rows[int(i)] for i in AUDIT_ENVS])
    if not np.array_equal(rows, frozen_rows):
        raise RuntimeError(
            f"fixed audit row mismatch: expected={frozen_rows.tolist()}, actual={rows.tolist()}"
        )
    if candidates.shape != (12, 300, 5, 25) or costs.shape != (12, 300):
        raise RuntimeError(f"invalid saved first-cycle pool shapes: {candidates.shape}/{costs.shape}")
    means = np.asarray(pool["cem_mean_normalized"], dtype=np.float32)
    if means.shape != (12, 5, 25):
        raise RuntimeError(f"invalid CEM mean shape: {means.shape}")
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    formal_scaler_mean = np.asarray(pool["action_scaler_mean"], dtype=np.float64)
    formal_scaler_scale = np.asarray(pool["action_scaler_scale"], dtype=np.float64)
    if not (
        np.array_equal(formal_scaler_mean, index.action_mean)
        and np.array_equal(formal_scaler_scale, index.action_scale)
    ):
        raise RuntimeError(
            "formal/index action scaler mismatch: "
            f"formal_mean/scale={formal_scaler_mean.tolist()}/{formal_scaler_scale.tolist()}, "
            f"index={index.action_mean.tolist()}/{index.action_scale.tolist()}"
        )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))["records"]
    source_by_env = {
        int(item["env_idx"]): item
        for item in trace
        if int(item["planning_cycle"]) == 0 and int(item["cem_iteration"]) == 0
    }
    baseline_path = BASELINE_ROOTS[args.color] / "aggregate_summary.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_any = {
        int(Path(item["case_dir"]).name.split("_", 2)[1]): bool(
            item["sample_success_count"] > 0
        )
        for item in baseline["cases"]
    }
    expected_old = {"red": 8, "blue": 9, "yellow": 8}[args.color]
    if sum(baseline_any.values()) != expected_old:
        raise RuntimeError(
            f"old pool-any evidence mismatch: expected={expected_old}, actual={sum(baseline_any.values())}"
        )
    # Only now may --overwrite replace a prior derived audit.
    output = _safe_output(args.output or _default_output(args.color), args.overwrite)

    import hdf5plugin  # noqa: F401
    import h5py

    env = base_audit._make_replay_env()
    summaries = []
    try:
        with h5py.File(args.dataset, "r", swmr=True) as h5:
            for case_i, env_idx in enumerate(env_indices):
                row = int(rows[case_i])
                goal_row = row + 25
                data = {
                    "initial_qpos": np.asarray(h5["qpos"][row]),
                    "initial_qvel": np.asarray(h5["qvel"][row]),
                    "initial_prev_qpos": np.asarray(h5["prev_qpos"][row]),
                    "initial_prev_qvel": np.asarray(h5["prev_qvel"][row]),
                    "goal_position": np.asarray(h5["privileged_block_0_pos"][goal_row]),
                    "goal_quaternion": np.asarray(h5["privileged_block_0_quat"][goal_row]),
                }
                cube_color = "dataset" if args.color == "red" else args.color
                snapshot = base_audit._setup_case_env(env, data, cube_color)
                case_rows = []
                order = np.lexsort((np.arange(300), costs[case_i]))
                latent_rank = np.empty(300, dtype=np.int64)
                latent_rank[order] = np.arange(1, 301)
                for candidate_idx in range(300):
                    normalized = candidates[case_i, candidate_idx]
                    actions = base_audit._inverse_scale(
                        normalized, formal_scaler_mean, formal_scaler_scale
                    )
                    outcome, _, _ = base_audit._branch_rollout(
                        env,
                        snapshot,
                        actions,
                        data["goal_position"],
                        collect_frames=False,
                        stop_on_success=False,
                    )
                    case_rows.append(
                        {
                            "candidate_idx": candidate_idx,
                            "is_memory_slot_1_to_10": 1 <= candidate_idx <= 10,
                            "latent_cost": float(costs[case_i, candidate_idx]),
                            "latent_rank_1based": int(latent_rank[candidate_idx]),
                            **outcome,
                        }
                    )
                case_dir = output / f"env_{int(env_idx):02d}_row_{row}"
                case_dir.mkdir(parents=True, exist_ok=True)
                _write_csv(case_dir / "candidate_outcomes.csv", case_rows)
                ever = np.asarray([x["ever_success"] for x in case_rows])
                final = np.asarray([x["final_success"] for x in case_rows])
                seed_ever = ever[1:11]
                seed_final = final[1:11]
                best = int(order[0])
                mean_actions = base_audit._inverse_scale(
                    means[case_i], formal_scaler_mean, formal_scaler_scale
                )
                mean_outcome, _, _ = base_audit._branch_rollout(
                    env,
                    snapshot,
                    mean_actions,
                    data["goal_position"],
                    collect_frames=False,
                    stop_on_success=False,
                )
                source = source_by_env[int(env_idx)]
                summary = {
                    "env_idx": int(env_idx),
                    "dataset_row": row,
                    "pool_source": str(pool_path),
                    "candidate_count": 300,
                    "sample_ever_success_count": int(ever.sum()),
                    "sample_final_success_count": int(final.sum()),
                    "population_any_ever_success": bool(ever.any()),
                    "population_any_final_success": bool(final.any()),
                    "latent_top1_candidate": best,
                    "latent_top1_ever_success": bool(ever[best]),
                    "memory_seed_ever_success_count": int(seed_ever.sum()),
                    "memory_seed_any_ever_success": bool(seed_ever.any()),
                    "memory_seed_final_success_count": int(seed_final.sum()),
                    "memory_seed_any_final_success": bool(seed_final.any()),
                    "cem_mean_ever_success": bool(mean_outcome["ever_success"]),
                    "cem_mean_final_success": bool(mean_outcome["final_success"]),
                    "cem_mean_min_goal_distance_m": mean_outcome["min_goal_distance_m"],
                    "cem_mean_final_goal_distance_m": mean_outcome["final_goal_distance_m"],
                    "old_unseeded_pool_any_ever_success": baseline_any[int(env_idx)],
                    "old_unsolved_to_seeded_solved": bool(
                        not baseline_any[int(env_idx)] and ever.any()
                    ),
                    "memory_source_rows": source["source_rows"],
                    "memory_source_episodes": source["source_episodes"],
                    "memory_source_steps": source["source_steps"],
                    "retrieval_distances": source["retrieval_distances"],
                }
                _write_json(case_dir / "summary.json", summary)
                summaries.append(summary)
    finally:
        env.close()
    aggregate = {
        "protocol": {
            "color": args.color,
            "formal_seeded_input": str(input_root),
            "first_planning_cycle": True,
            "final_population": 300,
            "fixed_env_indices": AUDIT_ENVS,
            "videos": False,
            "success_threshold_m": SUCCESS_METERS,
        },
        "num_cases": len(summaries),
        "cases_with_any_population_success": int(
            sum(x["population_any_ever_success"] for x in summaries)
        ),
        "cases_with_any_population_final_success": int(
            sum(x["population_any_final_success"] for x in summaries)
        ),
        "cases_with_any_memory_seed_success": int(
            sum(x["memory_seed_any_ever_success"] for x in summaries)
        ),
        "latent_top1_success_count": int(
            sum(x["latent_top1_ever_success"] for x in summaries)
        ),
        "memory_seed_any_final_success_count": int(
            sum(x["memory_seed_any_final_success"] for x in summaries)
        ),
        "cem_mean_ever_success_count": int(
            sum(x["cem_mean_ever_success"] for x in summaries)
        ),
        "cem_mean_final_success_count": int(
            sum(x["cem_mean_final_success"] for x in summaries)
        ),
        "old_unseeded_pool_any_ever_success_count": expected_old,
        "old_unsolved_to_seeded_solved_envs": [
            x["env_idx"] for x in summaries if x["old_unsolved_to_seeded_solved"]
        ],
        "cases": summaries,
    }
    _write_json(output / "aggregate_summary.json", aggregate)
    if list(output.rglob("*.mp4")):
        raise RuntimeError("no-video pool audit unexpectedly produced video files")
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay fixed12 seeded Cube first-cycle pools")
    parser.add_argument("--color", choices=("red","blue","yellow"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
