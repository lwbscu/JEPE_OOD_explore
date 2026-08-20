#!/usr/bin/env python3
"""Continuity-constrained B2 rule evaluator.

This evaluator is intentionally a thin, isolated repair around the frozen B2
evaluator.  It imports B2 read-only and replaces only the deterministic
STALLED real-frame selector: before the original midpoint/row stable ordering,
the literal HDF5 effector pose must be within 10 cm of the live effector pose.
All other B2 state-machine, T2, sampling, goal-image, and replan evidence
protocols remain owned by :mod:`eval_brain_b2`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_brain_b1 as b1  # noqa: E402
import eval_brain_b2 as b2  # noqa: E402


PROJECT_ROOT = HERE.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/eval/cube/longhorizon"
CONTINUITY_OUTPUT_ROOT = OUTPUT_ROOT / "rule_continuity_offset100"
SMOKE_OUTPUT_ROOT = OUTPUT_ROOT / "smoke" / "rule_continuity_offset100"
STALLED_EE_DISTANCE_MAX_M = 0.10
OFFSET = b2.OFFSET
BUDGET = b2.BUDGET
SEED = b2.SEED
ORIGINAL_B2_RULE_CHOICE = b2._rule_choice


def _file_identity(path: Path) -> dict[str, Any]:
    return b1._file_identity(path)


def _write_json(path: Path, value: Any) -> None:
    b1._write_json(path, value)


def _h5_vector(h5: Any, key: str, row: int) -> np.ndarray:
    value = np.asarray(h5[key][int(row)], dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid HDF5 {key} vector at row={row}: {value}")
    return value


def _inside_id(position: np.ndarray) -> bool:
    value = np.asarray(position, dtype=np.float64).reshape(3)
    return bool(
        np.isfinite(value).all()
        and b1.ID_X[0] <= value[0] <= b1.ID_X[1]
        and b1.ID_Y[0] <= value[1] <= b1.ID_Y[1]
        and b1.ID_Z[0] <= value[2] <= b1.ID_Z[1]
    )


def _continuity_midpoint_choice(
    index: b2.B2MemoryIndex,
    h5: Any,
    midpoint: np.ndarray,
    current_ee: np.ndarray,
    exclude_episode: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select the original nearest-midpoint anchor after the EE cap.

    The KD-tree query and complete tie expansion exactly follow B2's
    ``nearest_block_xyz`` ordering.  The only new predicate is the literal H5
    effector distance to the current live effector, evaluated before sorting.
    """

    raw = np.asarray(midpoint, dtype=np.float64).reshape(3)
    live_ee = np.asarray(current_ee, dtype=np.float64).reshape(3)
    query = (raw - index.feature_mean[:3]) / index.feature_std[:3]
    if not np.isfinite(query).all() or not np.isfinite(live_ee).all():
        raise ValueError("midpoint/current EE must be finite")

    examined = 0
    qualified = 0
    k = 64
    while True:
        distances, indices = index._block_xyz_tree.query(
            query, k=min(k, len(index.rows)), eps=0.0, workers=1
        )
        candidates: list[tuple[float, int, int, float, np.ndarray, np.ndarray]] = []
        for distance, tree_index in zip(
            np.atleast_1d(distances), np.atleast_1d(indices)
        ):
            anchor = int(tree_index)
            examined += 1
            if not index._allowed_anchor(anchor, exclude_episode):
                continue
            row = int(index.rows[anchor])
            block_h5 = _h5_vector(h5, "privileged_block_0_pos", row)
            ee_h5 = _h5_vector(h5, "proprio_effector_pos", row)
            if not _inside_id(block_h5):
                continue
            ee_distance = float(np.linalg.norm(ee_h5 - live_ee))
            if ee_distance > STALLED_EE_DISTANCE_MAX_M:
                continue
            qualified += 1
            candidates.append(
                (
                    float(distance),
                    row,
                    anchor,
                    ee_distance,
                    block_h5,
                    ee_h5,
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        if candidates:
            cutoff = candidates[0][0]
            tied_indices = index._block_xyz_tree.query_ball_point(
                query, np.nextafter(cutoff, np.inf), eps=0.0, workers=1
            )
            tied: list[tuple[float, int, int, float, np.ndarray, np.ndarray]] = []
            for tree_index in tied_indices:
                anchor = int(tree_index)
                if not index._allowed_anchor(anchor, exclude_episode):
                    continue
                row = int(index.rows[anchor])
                block_h5 = _h5_vector(h5, "privileged_block_0_pos", row)
                ee_h5 = _h5_vector(h5, "proprio_effector_pos", row)
                if not _inside_id(block_h5):
                    continue
                ee_distance = float(np.linalg.norm(ee_h5 - live_ee))
                if ee_distance > STALLED_EE_DISTANCE_MAX_M:
                    continue
                tied.append(
                    (
                        float(
                            np.linalg.norm(
                                np.asarray(index.features[anchor, :3]) - query
                            )
                        ),
                        row,
                        anchor,
                        ee_distance,
                        block_h5,
                        ee_h5,
                    )
                )
            tied.sort(key=lambda item: (item[0], item[1], item[2]))
            if tied:
                distance, row, anchor, ee_distance, block_h5, ee_h5 = tied[0]
                retrieval = index._row_record(
                    anchor,
                    "block_xyz_continuity_ee_cap",
                    raw,
                    distance,
                    (distance, row),
                )
                retrieval.update(
                    {
                        "continuity_ee_distance_m": ee_distance,
                        "continuity_ee_distance_cap_m": STALLED_EE_DISTANCE_MAX_M,
                        "continuity_filter_basis": "literal_hdf5_proprio_effector_pos",
                        "h5_block_pos": block_h5,
                        "h5_ee_pos": ee_h5,
                        "continuity_candidates_examined": examined,
                        "continuity_candidates_qualified": qualified,
                    }
                )
                return retrieval, {
                    "qualified": True,
                    "candidate_count_examined": examined,
                    "candidate_count_qualified": qualified,
                    "selected_row": row,
                    "selected_ee_distance_m": ee_distance,
                }
        if k >= len(index.rows):
            return None, {
                "qualified": False,
                "failure": "no_stalled_anchor_with_h5_ee_distance_le_0.10m",
                "candidate_count_examined": examined,
                "candidate_count_qualified": qualified,
                "ee_distance_cap_m": STALLED_EE_DISTANCE_MAX_M,
            }
        k = min(2 * k, len(index.rows))


def _continuity_rule_choice(
    index: b2.B2MemoryIndex,
    h5: Any,
    event: str,
    state: Mapping[str, Any],
    final_goal: b1.ActiveGoal,
    exclude_episode: int,
) -> tuple[b1.ActiveGoal | None, dict[str, Any] | None, str | None, dict[str, Any]]:
    """B2 rule choice with the one STALLED continuity predicate."""

    if event == "DROPPED":
        # Keep the frozen B2 recovery path byte-for-byte in behavior.
        return ORIGINAL_B2_RULE_CHOICE(
            index, h5, event, state, final_goal, exclude_episode
        )
    if event != "STALLED":
        raise ValueError(event)

    current = np.asarray(state["block_pos"], dtype=np.float64)
    midpoint_raw = (current + np.asarray(final_goal.position, dtype=np.float64)) / 2.0
    midpoint = b2._clamp_position(midpoint_raw)
    retrieval, audit = _continuity_midpoint_choice(
        index,
        h5,
        midpoint,
        np.asarray(state["ee_pos"], dtype=np.float64),
        exclude_episode,
    )
    if retrieval is None:
        # Explicitly record a no-op.  In particular, never fall back to the
        # unconstrained B2 midpoint anchor when the continuity pool is empty.
        decision = {
            "decision": "RULE_SUBGOAL_RETRIEVAL_FAILED",
            "rule": "midpoint_real_frame_with_h5_ee_continuity_cap",
            "applied_decision": "CONTINUE",
            "application_failure": audit["failure"],
            "continuity_audit": audit,
            "midpoint_before_clamp": midpoint_raw,
            "midpoint_after_clamp": midpoint,
            "yaw_ignored_for_query": True,
        }
        return None, None, None, decision

    retrieval["midpoint_before_clamp"] = midpoint_raw
    retrieval["midpoint_after_clamp"] = midpoint
    retrieval["yaw_ignored_for_query"] = True
    goal = b1._goal_from_real_frame(h5, retrieval, "block")
    if not np.allclose(goal.position, b2._clamp_position(goal.position)):
        raise RuntimeError(
            "continuity STALLED midpoint resolved to a real block frame outside frozen ID box"
        )
    h5_goal_ee = _h5_vector(h5, "proprio_effector_pos", int(retrieval["row"]))
    if float(np.linalg.norm(h5_goal_ee - np.asarray(state["ee_pos"]))) > STALLED_EE_DISTANCE_MAX_M + 1e-9:
        raise RuntimeError("continuity STALLED H5 landing violated EE distance cap")
    decision = {
        "decision": "RULE_SUBGOAL",
        "rule": "current_to_final_xyz_midpoint_real_frame_with_h5_ee_continuity_cap",
        "continuity_audit": audit,
    }
    return goal, retrieval, "block_distance", decision


def _validate_output(path: Path, mode: str, num_eval: int) -> None:
    if mode != "rule":
        raise ValueError("continuity repair only supports --mode rule")
    resolved = path.expanduser().absolute().resolve()
    expected = (
        CONTINUITY_OUTPUT_ROOT.resolve()
        if num_eval == 50
        else SMOKE_OUTPUT_ROOT.resolve()
    )
    if resolved != expected:
        raise ValueError(f"continuity output is frozen: expected={expected}, actual={resolved}")


def _update_manifest(output: Path) -> None:
    path = output / "run_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"B2 run did not produce manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    original_rule = dict(manifest.get("rule", {}))
    original_helper = dict(manifest.get("helper_provenance", {}))
    original_helper["eval_brain_b2"] = _file_identity(Path(b2.__file__))
    original_helper["this_evaluator"] = _file_identity(Path(__file__))
    original_helper["repair_scope"] = "STALLED real-frame retrieval only"
    manifest["helper_provenance"] = original_helper
    manifest["output_path_interpretation"] = (
        "continuity-repair output; B2 protocol/results reused with only the STALLED "
        "real-frame EE continuity predicate changed"
    )
    original_rule["STALLED"] = (
        "xyz midpoint(current block, final target); first require literal HDF5 "
        "proprio_effector_pos distance <=0.10m from live EE; then B2 stable "
        "midpoint distance,row ordering; use retrieved frame yaw"
    )
    manifest["rule"] = original_rule
    manifest["continuity_repair"] = {
        "version": "subgoal_continuity_v1",
        "changed_event": "STALLED",
        "h5_effector_distance_cap_m": STALLED_EE_DISTANCE_MAX_M,
        "filter_order": [
            "global_fixed50_and_current_episode_exclusion",
            "literal_hdf5_effector_distance_le_0.10m",
            "real_hdf5_block_id_box",
            "original_b2_midpoint_distance_then_row_stable_sort",
            "literal_hdf5_goal_landing_revalidation",
        ],
        "empty_pool": "explicit failure/CONTINUE with audit; no unconstrained fallback",
        "dropped_rule": "frozen B2 recovery selector unchanged",
    }
    _write_json(path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = b2.build_parser()
    parser.description = __doc__
    # B2's parser remains the source of the frozen protocol flags.  This
    # wrapper rejects brain mode in run() and routes output to the continuity
    # namespace.
    return parser


def run(args: argparse.Namespace) -> int:
    if args.mode != "rule":
        raise ValueError("eval_brain_subgoal_fix.py only supports --mode rule")
    if args.goal_offset_steps != OFFSET or args.eval_budget != BUDGET or args.seed != SEED:
        raise ValueError("continuity repair is frozen at offset=100, budget=200, seed=42")
    requested = args.output
    if requested is None:
        requested = CONTINUITY_OUTPUT_ROOT if args.num_eval == 50 else SMOKE_OUTPUT_ROOT
    args.output = Path(requested)

    original_rule = b2._rule_choice
    original_validate = b2._validate_output
    b2._rule_choice = _continuity_rule_choice
    b2._validate_output = _validate_output
    try:
        result = b2.run(args)
    finally:
        b2._rule_choice = original_rule
        b2._validate_output = original_validate
    if result == 0 and not args.self_test:
        _update_manifest(args.output.expanduser().absolute().resolve())
    return int(result)


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
