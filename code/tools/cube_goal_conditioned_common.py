#!/usr/bin/env python3
"""Frozen contracts shared by the Cube goal-conditioned retrieval tools.

Only the retrieval rule differs from the preceding Trust-Region experiment.
All CEM, checkpoint, action-scaling, noise, and evaluation constants are
imported directly from :mod:`cube_trust_region_common`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import cube_trust_region_common as _trust
from cube_trust_region_common import *  # noqa: F401,F403


OUTPUT_ROOT = PROJECT_ROOT / "outputs/eval/cube/goal_conditioned"
PROTOCOLS = ("g1t2", "g1t1")
TOP100 = 100
GOAL_SCORE_HORIZON = 25
ALIGNMENT_THRESHOLD = 0.0

PROTOCOL_SPECS = {
    "g1t1": {
        **_trust.PROTOCOL_SPECS["t1"],
        "name": "goal_aligned_nearest_seed_initial_mean",
        "retrieval": "exact state top-100 then goal-progress filter",
        "goal_score": "norm(block_t-goal)-norm(block_t_plus_25-goal)",
        "goal_score_threshold": ALIGNMENT_THRESHOLD,
        "retrieval_topn": TOP100,
        "global_eval_episode_exclusion": True,
        "fallback": "same pre-filter top-100 only",
    },
    "g1t2": {
        **_trust.PROTOCOL_SPECS["t2"],
        "name": "goal_aligned_seed_and_local_noise",
        "retrieval": "exact state top-100 then goal-progress filter",
        "goal_score": "norm(block_t-goal)-norm(block_t_plus_25-goal)",
        "goal_score_threshold": ALIGNMENT_THRESHOLD,
        "retrieval_topn": TOP100,
        "global_eval_episode_exclusion": True,
        "fallback": "same pre-filter top-100 only",
    },
}


def base_protocol(protocol: str) -> str:
    """Return the frozen Trust-Region distribution underlying a G1 arm."""

    try:
        return {"g1t1": "t1", "g1t2": "t2"}[str(protocol)]
    except KeyError as error:
        raise ValueError(f"invalid goal-conditioned protocol: {protocol}") from error


def default_eval_output(protocol: str, condition: str, num_eval: int) -> Path:
    if str(protocol) not in PROTOCOLS or condition not in CONDITIONS:
        raise ValueError(f"invalid goal-conditioned protocol/condition: {protocol}/{condition}")
    arm = str(protocol).upper()
    if num_eval == 50:
        return OUTPUT_ROOT / arm / condition
    return OUTPUT_ROOT / "smoke" / arm / condition


def capture_output_root(protocol: str, condition: str) -> Path:
    if str(protocol) not in PROTOCOLS or condition not in CONDITIONS:
        raise ValueError(f"invalid goal-conditioned protocol/condition: {protocol}/{condition}")
    return OUTPUT_ROOT / "gate_capture" / str(protocol).upper() / condition


def physical_cache_root(protocol: str, condition: str) -> Path:
    return OUTPUT_ROOT / "physical_cache" / str(protocol).upper() / condition


def imagination_output_root(protocol: str, condition: str) -> Path:
    return OUTPUT_ROOT / "imagination_error" / str(protocol).upper() / condition


def select_goal_aligned(
    distances: np.ndarray,
    rows: np.ndarray,
    episodes: np.ndarray,
    steps: np.ndarray,
    current_xyz: np.ndarray,
    future_xyz: np.ndarray,
    future_valid: np.ndarray,
    goal_xyz: np.ndarray,
    count: int = MEMORY_SLOTS,
) -> dict[str, np.ndarray | int | float]:
    """Filter a frozen state-nearest pool and fill only from that same pool.

    Inputs must already be in stable ``(distance, dataset_row)`` order and
    contain the exact raw-anchor top-100 after the global evaluation-episode
    exclusion.  Positive goal progress is preferred.  Both preferred and
    fallback selections enforce one source per episode.
    """

    distances = np.asarray(distances, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    episodes = np.asarray(episodes, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    current_xyz = np.asarray(current_xyz, dtype=np.float64)
    future_xyz = np.asarray(future_xyz, dtype=np.float64)
    future_valid = np.asarray(future_valid, dtype=bool)
    goal_xyz = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    n = len(rows)
    expected_shapes = {
        "distances": (n,),
        "episodes": (n,),
        "steps": (n,),
        "current_xyz": (n, 3),
        "future_xyz": (n, 3),
        "future_valid": (n,),
    }
    actual_shapes = {
        "distances": distances.shape,
        "episodes": episodes.shape,
        "steps": steps.shape,
        "current_xyz": current_xyz.shape,
        "future_xyz": future_xyz.shape,
        "future_valid": future_valid.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"goal-alignment input shape mismatch: expected={expected_shapes}, "
            f"actual={actual_shapes}"
        )
    if n != TOP100:
        raise ValueError(f"goal alignment requires exact top-{TOP100}: expected={TOP100}, actual={n}")
    if count <= 0:
        raise ValueError(f"selection count must be positive: {count}")
    if not (
        np.isfinite(distances).all()
        and np.isfinite(current_xyz).all()
        and np.isfinite(goal_xyz).all()
        and np.isfinite(future_xyz[future_valid]).all()
    ):
        raise ValueError("goal-alignment inputs contain non-finite values")
    order = np.lexsort((rows, distances))
    if not np.array_equal(order, np.arange(n)):
        raise ValueError("top-100 must be stably sorted by (distance,dataset_row)")

    scores = np.full(n, np.nan, dtype=np.float64)
    scores[future_valid] = (
        np.linalg.norm(current_xyz[future_valid] - goal_xyz, axis=1)
        - np.linalg.norm(future_xyz[future_valid] - goal_xyz, axis=1)
    )
    aligned = future_valid & (scores > ALIGNMENT_THRESHOLD)
    selected_indices: list[int] = []
    seen_episodes: set[int] = set()
    for candidate_idx in np.flatnonzero(aligned):
        episode = int(episodes[candidate_idx])
        if episode in seen_episodes:
            continue
        selected_indices.append(int(candidate_idx))
        seen_episodes.add(episode)
        if len(selected_indices) == count:
            break
    selected_aligned_count = len(selected_indices)
    for candidate_idx in range(n):
        if len(selected_indices) == count:
            break
        episode = int(episodes[candidate_idx])
        if episode in seen_episodes:
            continue
        selected_indices.append(candidate_idx)
        seen_episodes.add(episode)
    if len(selected_indices) != count:
        raise RuntimeError(
            "insufficient distinct source episodes inside fixed top-100: "
            f"expected={count}, actual={len(selected_indices)}, "
            f"top100_distinct={len(set(int(value) for value in episodes))}"
        )
    selected = np.asarray(selected_indices, dtype=np.int64)
    selected_is_aligned = aligned[selected]
    selected_is_fallback = np.arange(count) >= selected_aligned_count
    # A fallback can itself have a positive score only when its episode was
    # already represented by an earlier aligned anchor.  The slot remains a
    # fallback because it was selected during the fill phase.
    return {
        "scores": scores,
        "aligned": aligned,
        "selected_indices": selected,
        "selected_is_aligned": selected_is_aligned,
        "selected_is_fallback": selected_is_fallback,
        "raw_positive_count": int(np.count_nonzero(aligned)),
        "raw_positive_rate": float(np.mean(aligned)),
        "selected_aligned_count": int(selected_aligned_count),
        "selected_aligned_rate": float(selected_aligned_count / count),
        "fallback_count": int(count - selected_aligned_count),
        "fallback_rate": float((count - selected_aligned_count) / count),
    }


def alignment_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval statistics once per planning cycle."""

    cycles = [row for row in records if int(row["cem_iteration"]) == 0]
    raw_positive = np.asarray(
        [row["alignment_raw_positive_count"] for row in cycles], dtype=np.int64
    )
    selected_aligned = np.asarray(
        [row["alignment_selected_count"] for row in cycles], dtype=np.int64
    )
    fallback = np.asarray(
        [row["alignment_fallback_count"] for row in cycles], dtype=np.int64
    )
    if not cycles:
        raise ValueError("cannot summarize an empty goal-conditioned trace")
    return {
        "num_queries": len(cycles),
        "top100_candidates": int(len(cycles) * TOP100),
        "raw_positive_count": int(raw_positive.sum()),
        "raw_positive_rate": float(raw_positive.sum() / (len(cycles) * TOP100)),
        "selected_seed_slots": int(len(cycles) * MEMORY_SLOTS),
        "selected_aligned_count": int(selected_aligned.sum()),
        "selected_aligned_rate": float(
            selected_aligned.sum() / (len(cycles) * MEMORY_SLOTS)
        ),
        "fallback_count": int(fallback.sum()),
        "fallback_rate": float(fallback.sum() / (len(cycles) * MEMORY_SLOTS)),
        "queries_below_30pct_raw_alignment": int(np.count_nonzero(raw_positive < 30)),
        "queries_with_fallback": int(np.count_nonzero(fallback > 0)),
    }
