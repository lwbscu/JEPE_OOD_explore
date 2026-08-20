#!/usr/bin/env python3
"""B2 long-horizon Cube rule and prompt-v2 supervisor evaluation.

B2 preserves the frozen B1 offset-100 held-out rows, MaskedAug model, T2
planner, trigger state machine, and global retrieval exclusions.  It compares a
deterministic real-frame rule controller with a prompt-v2 controller that may
choose among three precomputed real-frame intentions and then re-retrieves a
real HDF5 frame after any numeric adjustment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import eval_brain_b1 as b1  # noqa: E402


PROJECT_ROOT = HERE.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/eval/cube/longhorizon"
PROMPT_REPLAY_ROOT = PROJECT_ROOT / "outputs/eval/cube/brain_b2/offline_prompt_replay"
MODES = ("rule", "brainv2")
OFFSET = 100
BUDGET = 200
SEED = 42
ACTIVE_GOAL_TIMEOUT_STEPS = 25
RECOVERY_EE_BLOCK_MAX_M = 0.03
RECOVERY_CONTACT_MIN = 0.5
RECOVERY_OPENING_MAX = 0.60
NUM_CANDIDATES = 3
EXPECTED_RECOVERY_ANCHORS = 826_131
EXPECTED_RECOVERY_EPISODES = 9_950
SMOKE_FORMAL_ENV_INDICES = (0, 3)


def _file_identity(path: Path) -> dict[str, Any]:
    return b1._file_identity(path)


def _write_json(path: Path, value: Any) -> None:
    b1._write_json(path, value)


def _default_output(mode: str, num_eval: int) -> Path:
    name = f"{mode}_offset{OFFSET}"
    return OUTPUT_ROOT / name if num_eval == 50 else OUTPUT_ROOT / "smoke" / name


def _validate_output(path: Path, mode: str, num_eval: int) -> None:
    resolved = path.expanduser().absolute().resolve()
    expected = _default_output(mode, num_eval).resolve()
    if resolved != expected:
        raise ValueError(
            f"B2 output is frozen: expected={expected}, actual={resolved}"
        )


def _yaw_wrap(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("yaw must be finite")
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


class B2MemoryIndex(b1.HeldoutMemoryIndex):
    """B1 index plus xyz-only progress and contact-frame retrieval."""

    def __init__(self, root: Path, dataset: Path, fixed_episodes: np.ndarray) -> None:
        super().__init__(root, dataset, fixed_episodes)
        from scipy.spatial import cKDTree

        import hdf5plugin  # noqa: F401
        import h5py

        if np.any(np.diff(np.asarray(self.rows, dtype=np.int64)) <= 0):
            raise RuntimeError("B2 requires strictly increasing unique anchor rows")

        block = (
            np.asarray(self.features[:, :3], dtype=np.float64)
            * self.feature_std[:3]
            + self.feature_mean[:3]
        )
        ee = (
            np.asarray(self.features[:, 5:8], dtype=np.float64)
            * self.feature_std[5:8]
            + self.feature_mean[5:8]
        )
        opening = (
            np.asarray(self.features[:, 8], dtype=np.float64)
            * self.feature_std[8]
            + self.feature_mean[8]
        )
        with h5py.File(dataset, "r", swmr=True) as h5:
            contact = np.asarray(
                h5["proprio_gripper_contact"][np.asarray(self.rows, dtype=np.int64)],
                dtype=np.float64,
            ).reshape(-1)
        proximity = np.linalg.norm(ee - block, axis=1) <= RECOVERY_EE_BLOCK_MAX_M
        contact_on = contact >= RECOVERY_CONTACT_MIN
        opening_closed = opening <= RECOVERY_OPENING_MAX
        block_inside = (
            (block[:, 0] >= b1.ID_X[0])
            & (block[:, 0] <= b1.ID_X[1])
            & (block[:, 1] >= b1.ID_Y[0])
            & (block[:, 1] <= b1.ID_Y[1])
            & (block[:, 2] >= b1.ID_Z[0])
            & (block[:, 2] <= b1.ID_Z[1])
        )
        ee_inside = (
            (ee[:, 0] >= b1.ID_X[0])
            & (ee[:, 0] <= b1.ID_X[1])
            & (ee[:, 1] >= b1.ID_Y[0])
            & (ee[:, 1] <= b1.ID_Y[1])
            & (ee[:, 2] >= b1.ID_Z[0])
            & (ee[:, 2] <= b1.ID_Z[1])
        )
        globally_allowed = ~np.isin(
            np.asarray(self.episodes, dtype=np.int64),
            np.fromiter(sorted(self.fixed_episodes), dtype=np.int64),
        )
        recovery = (
            proximity
            & contact_on
            & opening_closed
            & globally_allowed
        )
        self._block_raw = block
        self._ee_raw = ee
        self._block_inside_id = block_inside
        self._ee_inside_id = ee_inside
        self._opening_raw = opening
        self._contact_raw = contact
        self._recovery_indices = np.flatnonzero(recovery).astype(np.int64)
        recovery_episode_count = len(
            np.unique(np.asarray(self.episodes)[self._recovery_indices])
        )
        if (
            len(self._recovery_indices) != EXPECTED_RECOVERY_ANCHORS
            or recovery_episode_count != EXPECTED_RECOVERY_EPISODES
        ):
            raise RuntimeError(
                "frozen recovery pool drift: "
                f"expected={EXPECTED_RECOVERY_ANCHORS}/{EXPECTED_RECOVERY_EPISODES}, "
                f"actual={len(self._recovery_indices)}/{recovery_episode_count}"
            )
        self._block_xyz_tree = cKDTree(
            np.ascontiguousarray(self.features[:, :3], dtype=np.float64),
            copy_data=False,
            balanced_tree=True,
            compact_nodes=True,
        )
        self._recovery_block_tree = cKDTree(
            np.ascontiguousarray(block[self._recovery_indices], dtype=np.float64),
            copy_data=False,
            balanced_tree=True,
            compact_nodes=True,
        )
        self._recovery_ee_tree = cKDTree(
            np.ascontiguousarray(ee[self._recovery_indices], dtype=np.float64),
            copy_data=False,
            balanced_tree=True,
            compact_nodes=True,
        )
        contact_near = proximity & contact_on
        proxy_near = proximity & opening_closed
        true_positive = int(np.count_nonzero(contact_near & proxy_near))
        self.recovery_filter_manifest = {
            "conditions": {
                "ee_block_distance_m_max": RECOVERY_EE_BLOCK_MAX_M,
                "proprio_gripper_contact_min": RECOVERY_CONTACT_MIN,
                "proprio_gripper_opening_max": RECOVERY_OPENING_MAX,
                "opening_semantics": (
                    "dataset-derived closed/contact upper bound; contact-qualified "
                    "frames have median opening near 0.562 and the <=0.60 cap removes outliers"
                ),
                "selection_requires_block_and_ee_inside_frozen_id_box": True,
            },
            "all_anchor_count": int(len(self.rows)),
            "proximity_anchor_count": int(np.count_nonzero(proximity)),
            "contact_and_proximity_count": int(np.count_nonzero(contact_near)),
            "opening_and_proximity_count": int(np.count_nonzero(proxy_near)),
            "three_condition_count_before_episode_exclusion": int(
                np.count_nonzero(proximity & contact_on & opening_closed)
            ),
            "id_box_eligible_count_before_episode_exclusion": int(
                np.count_nonzero(
                    proximity & contact_on & opening_closed & block_inside & ee_inside
                )
            ),
            "three_condition_count_after_global_fixed50_exclusion": int(
                len(self._recovery_indices)
            ),
            "source_episode_count_after_global_fixed50_exclusion": int(
                recovery_episode_count
            ),
            "contact_frame_opening_median": float(np.median(opening[contact_near])),
            "contact_frame_opening_p95": float(
                np.quantile(opening[contact_near], 0.95)
            ),
            "opening_cap_contact_recall_within_3cm": (
                true_positive / int(np.count_nonzero(contact_near))
            ),
        }

    def _allowed_anchor(
        self,
        anchor_index: int,
        exclude_episode: int,
        additional_excluded: Sequence[int] = (),
    ) -> bool:
        episode = int(self.episodes[int(anchor_index)])
        return episode not in self._forbidden(exclude_episode) and episode not in set(
            int(value) for value in additional_excluded
        )

    def _row_record(
        self,
        anchor_index: int,
        kind: str,
        query_raw: np.ndarray,
        distance: float,
        rank_key: Sequence[Any],
    ) -> dict[str, Any]:
        anchor_index = int(anchor_index)
        return {
            "kind": kind,
            "distance": float(distance),
            "row": int(self.rows[anchor_index]),
            "episode": int(self.episodes[anchor_index]),
            "step": int(self.rows[anchor_index]) % b1.EPISODE_LENGTH,
            "anchor_index": anchor_index,
            "query_raw": np.asarray(query_raw, dtype=np.float64),
            "rank_key": list(rank_key),
            "anchor_block_pos": self._block_raw[anchor_index],
            "anchor_ee_pos": self._ee_raw[anchor_index],
            "anchor_gripper_opening": float(self._opening_raw[anchor_index]),
            "anchor_gripper_contact_raw": float(self._contact_raw[anchor_index]),
        }

    def anchor_for_row(self, row: int) -> int:
        location = int(np.searchsorted(self.rows, int(row)))
        if location >= len(self.rows) or int(self.rows[location]) != int(row):
            raise RuntimeError(f"candidate row absent from memory index: {row}")
        return location

    def nearest_block_xyz(
        self,
        position: np.ndarray,
        exclude_episode: int,
        additional_excluded: Sequence[int] = (),
    ) -> dict[str, Any]:
        raw = np.asarray(position, dtype=np.float64).reshape(3)
        query = (raw - self.feature_mean[:3]) / self.feature_std[:3]
        k = 64
        while True:
            distances, indices = self._block_xyz_tree.query(
                query, k=min(k, len(self.rows)), eps=0.0, workers=1
            )
            candidates = sorted(
                (
                    float(distance),
                    int(self.rows[int(index)]),
                    int(index),
                )
                for distance, index in zip(
                    np.atleast_1d(distances), np.atleast_1d(indices)
                )
                if self._allowed_anchor(
                    int(index), exclude_episode, additional_excluded
                )
                and bool(self._block_inside_id[int(index)])
            )
            if candidates:
                cutoff = candidates[0][0]
                tied_indices = self._block_xyz_tree.query_ball_point(
                    query, np.nextafter(cutoff, np.inf), eps=0.0, workers=1
                )
                tied = sorted(
                    (
                        float(
                            np.linalg.norm(
                                np.asarray(self.features[int(index), :3]) - query
                            )
                        ),
                        int(self.rows[int(index)]),
                        int(index),
                    )
                    for index in tied_indices
                    if self._allowed_anchor(
                        int(index), exclude_episode, additional_excluded
                    )
                    and bool(self._block_inside_id[int(index)])
                )
                if tied:
                    distance, row, anchor_index = tied[0]
                    return self._row_record(
                        anchor_index,
                        "block_xyz",
                        raw,
                        distance,
                        (distance, row),
                    )
            if k >= len(self.rows):
                raise RuntimeError(
                    "rule midpoint landing failed: expected=1 allowed real block frame, "
                    "actual=0, location=B2MemoryIndex.nearest_block_xyz after block ID-box "
                    "filter and fixed50/current exclusion"
                )
            k = min(2 * k, len(self.rows))

    def recovery_candidates(
        self,
        block_position: np.ndarray,
        exclude_episode: int,
        count: int = NUM_CANDIDATES,
    ) -> list[dict[str, Any]]:
        query = np.asarray(block_position, dtype=np.float64).reshape(3)
        k = 128
        while True:
            _, local_indices = self._recovery_block_tree.query(
                query, k=min(k, len(self._recovery_indices)), eps=0.0, workers=1
            )
            anchors = self._recovery_indices[np.atleast_1d(local_indices).astype(np.int64)]
            ranked = sorted(
                (
                    float(np.linalg.norm(self._block_raw[index] - query)),
                    float(np.linalg.norm(self._ee_raw[index] - query)),
                    int(self.rows[index]),
                    int(self.episodes[index]),
                    int(index),
                )
                for index in anchors
                if self._allowed_anchor(int(index), exclude_episode)
                and bool(self._block_inside_id[int(index)])
                and bool(self._ee_inside_id[int(index)])
            )
            selected: list[tuple[float, float, int, int, int]] = []
            seen: set[int] = set()
            for item in ranked:
                if item[3] in seen:
                    continue
                seen.add(item[3])
                selected.append(item)
                if len(selected) == count:
                    break
            if len(selected) == count:
                cutoff = float(selected[-1][0])
                local_ball = self._recovery_block_tree.query_ball_point(
                    query,
                    np.nextafter(cutoff, np.inf),
                    eps=0.0,
                    workers=1,
                )
                complete = sorted(
                    (
                        float(np.linalg.norm(self._block_raw[index] - query)),
                        float(np.linalg.norm(self._ee_raw[index] - query)),
                        int(self.rows[index]),
                        int(self.episodes[index]),
                        int(index),
                    )
                    for index in self._recovery_indices[
                        np.asarray(local_ball, dtype=np.int64)
                    ]
                    if self._allowed_anchor(int(index), exclude_episode)
                    and bool(self._block_inside_id[int(index)])
                    and bool(self._ee_inside_id[int(index)])
                )
                selected = []
                seen = set()
                for item in complete:
                    if item[3] in seen:
                        continue
                    seen.add(item[3])
                    selected.append(item)
                    if len(selected) == count:
                        break
                if len(selected) != count:
                    k = min(2 * k, len(self._recovery_indices))
                    continue
                return [
                    self._row_record(
                        item[4],
                        "recovery_contact_frame",
                        query,
                        item[0],
                        item[:4],
                    )
                    for item in selected
                ]
            if k >= len(self._recovery_indices):
                raise RuntimeError(
                    "recovery candidate selection failed: "
                    f"expected={count} unique allowed real frames, actual={len(selected)}, "
                    "location=B2MemoryIndex.recovery_candidates after triple filter, "
                    "block/ee ID-box filter, fixed50/current exclusion"
                )
            k = min(2 * k, len(self._recovery_indices))

    def nearest_recovery_ee(
        self,
        ee_position: np.ndarray,
        current_block_position: np.ndarray,
        exclude_episode: int,
    ) -> dict[str, Any]:
        query = np.asarray(ee_position, dtype=np.float64).reshape(3)
        block = np.asarray(current_block_position, dtype=np.float64).reshape(3)
        k = 128
        while True:
            _, local_indices = self._recovery_ee_tree.query(
                query, k=min(k, len(self._recovery_indices)), eps=0.0, workers=1
            )
            anchors = self._recovery_indices[np.atleast_1d(local_indices).astype(np.int64)]
            ranked = sorted(
                (
                    float(np.linalg.norm(self._ee_raw[index] - query)),
                    float(np.linalg.norm(self._block_raw[index] - block)),
                    int(self.rows[index]),
                    int(index),
                )
                for index in anchors
                if self._allowed_anchor(int(index), exclude_episode)
                and bool(self._block_inside_id[int(index)])
                and bool(self._ee_inside_id[int(index)])
            )
            if ranked:
                cutoff = float(ranked[0][0])
                local_ball = self._recovery_ee_tree.query_ball_point(
                    query,
                    np.nextafter(cutoff, np.inf),
                    eps=0.0,
                    workers=1,
                )
                complete = sorted(
                    (
                        float(np.linalg.norm(self._ee_raw[index] - query)),
                        float(np.linalg.norm(self._block_raw[index] - block)),
                        int(self.rows[index]),
                        int(index),
                    )
                    for index in self._recovery_indices[
                        np.asarray(local_ball, dtype=np.int64)
                    ]
                    if self._allowed_anchor(int(index), exclude_episode)
                    and bool(self._block_inside_id[int(index)])
                    and bool(self._ee_inside_id[int(index)])
                )
                if not complete:
                    k = min(2 * k, len(self._recovery_indices))
                    continue
                item = complete[0]
                return self._row_record(
                    item[3],
                    "recovery_contact_frame_adjusted",
                    query,
                    item[0],
                    item[:3],
                )
            if k >= len(self._recovery_indices):
                raise RuntimeError(
                    "adjusted recovery landing failed: expected=1 allowed real frame, "
                    "actual=0, location=B2MemoryIndex.nearest_recovery_ee after triple "
                    "filter, block/ee ID-box filter, fixed50/current exclusion"
                )
            k = min(2 * k, len(self._recovery_indices))


@dataclass
class B2Monitor(b1.EnvMonitor):
    active_started_step: int | None = None
    active_deadline_step: int | None = None
    active_completion: str | None = None
    active_intervention_index: int | None = None
    interventions: list[dict[str, Any]] = field(default_factory=list)
    active_goal_outcomes: list[dict[str, Any]] = field(default_factory=list)
    seen_policy_replans: int = 0
    episode_done: bool = False
    episode_done_step: int | None = None
    episode_terminated: bool = False
    episode_truncated: bool = False


def _initial_monitors(
    init_state: Mapping[str, np.ndarray],
    goal_state: Mapping[str, np.ndarray],
    rows: np.ndarray,
    goal_rows: np.ndarray,
    episodes: np.ndarray,
) -> list[B2Monitor]:
    base = b1._initial_monitors(init_state, goal_state, rows, goal_rows, episodes)
    monitors = []
    for item in base:
        monitor = B2Monitor(
            env_idx=item.env_idx,
            final_goal=item.final_goal,
            active_goal=item.active_goal,
            previous_phase=item.previous_phase,
            previous_contact_on=item.previous_contact_on,
            previous_contact_raw=item.previous_contact_raw,
        )
        monitor.dist_history.extend(item.dist_history)
        monitors.append(monitor)
    return monitors


def _switch_goal_with_replan_request(
    monitor: B2Monitor,
    goal: b1.ActiveGoal,
    policy: Any,
    step: int,
    reason: str,
    state: Mapping[str, Any],
    retrieval: Mapping[str, Any] | None,
) -> None:
    """Switch a real-frame goal and require a solve on the next policy call.

    The deployed policy executes a 25-physical-step action buffer.  Clearing it
    in the post-step callback is necessary but, on its own, is not an
    observable guarantee that the following action was planned for the new
    goal.  The B2 policy therefore owns a one-shot request which clears the
    executing buffer again at the next ``get_action`` boundary and verifies an
    actual T2 solve against the newly installed goal pixels.
    """

    env_idx = int(monitor.env_idx)
    buffer_length_before = int(len(policy._action_buffer[env_idx]))
    b1._switch_goal(monitor, goal, policy, step, reason, state, retrieval)
    request = policy.request_b2_goal_replan(
        env_idx=env_idx,
        switch_step=int(step),
        goal_kind=str(goal.kind),
        goal_row=int(goal.source_row),
        goal_pixels_sha256=hashlib.sha256(
            np.ascontiguousarray(goal.pixels).tobytes()
        ).hexdigest(),
    )
    monitor.goal_switches[-1].update(
        {
            "action_buffer_length_before_flush": buffer_length_before,
            "action_buffer_length_after_flush": int(
                len(policy._action_buffer[env_idx])
            ),
            "replan_request": request,
        }
    )


def _install_active_goal(
    monitor: B2Monitor,
    goal: b1.ActiveGoal,
    policy: Any,
    step: int,
    reason: str,
    state: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    completion: str,
    intervention_index: int,
) -> None:
    _switch_goal_with_replan_request(
        monitor, goal, policy, step, reason, state, retrieval
    )
    monitor.active_started_step = int(step)
    monitor.active_deadline_step = int(step + ACTIVE_GOAL_TIMEOUT_STEPS)
    monitor.active_completion = completion
    monitor.active_intervention_index = int(intervention_index)
    monitor.goal_switches[-1].update(
        {
            "active_goal_timeout_steps": ACTIVE_GOAL_TIMEOUT_STEPS,
            "active_goal_deadline_step": monitor.active_deadline_step,
            "active_goal_completion": completion,
        }
    )


def _finish_active_goal(
    monitor: B2Monitor,
    policy: Any,
    step: int,
    state: Mapping[str, Any],
    outcome: str,
    *,
    future_action_expected: bool = True,
    no_replan_reason: str | None = None,
) -> None:
    previous = monitor.active_goal
    record = {
        "env_idx": monitor.env_idx,
        "step": int(step),
        "outcome": outcome,
        "active_kind": previous.kind,
        "active_source_row": previous.source_row,
        "started_step": monitor.active_started_step,
        "deadline_step": monitor.active_deadline_step,
        "elapsed_physical_steps": (
            None
            if monitor.active_started_step is None
            else int(step - monitor.active_started_step)
        ),
        "contact_on": bool(state["contact_on"]),
        "future_action_expected": bool(future_action_expected),
        "no_replan_reason": no_replan_reason,
        "block_distance_m": (
            b1._active_distance(state, previous)
            if previous.target_kind == "block"
            else None
        ),
    }
    monitor.active_goal_outcomes.append(record)
    if monitor.active_intervention_index is not None:
        monitor.interventions[monitor.active_intervention_index].update(
            {"status": outcome, "outcome": record}
        )
    if outcome == "achieved" and previous.target_kind == "block":
        monitor.subgoals_achieved += 1
    if future_action_expected:
        if no_replan_reason is not None:
            raise RuntimeError(
                "future B2 action cannot carry a no-replan reason: "
                f"{no_replan_reason}"
            )
        _switch_goal_with_replan_request(
            monitor,
            monitor.final_goal,
            policy,
            step,
            f"active_goal_{outcome}",
            state,
            None,
        )
        record["final_goal_restore"] = "switch_and_require_next_policy_replan"
    else:
        if no_replan_reason not in {
            "terminal_final_success",
            "episode_truncated",
            "evaluation_budget_exhausted",
        }:
            raise RuntimeError(
                "B2 no-replan active closure lacks an allowed terminal reason: "
                f"{no_replan_reason}"
            )
        # There is no future physical action on which a restored goal could
        # operate.  Close the monitor state explicitly without creating a
        # fictitious goal switch or a policy request that can never execute.
        monitor.active_goal = monitor.final_goal
        monitor.dist_history.clear()
        monitor.dist_history.append(b1._final_distance(state, monitor.final_goal))
        monitor.comparable_costs.clear()
        record["final_goal_restore"] = "monitor_only_no_future_action"
    monitor.active_started_step = None
    monitor.active_deadline_step = None
    monitor.active_completion = None
    monitor.active_intervention_index = None


def _abort_active_goal(
    monitor: B2Monitor,
    step: int,
    state: Mapping[str, Any],
    replacement_event: str,
    outcome: str = "aborted_replaced",
) -> None:
    previous = monitor.active_goal
    record = {
        "env_idx": monitor.env_idx,
        "step": int(step),
        "outcome": outcome,
        "replacement_event": replacement_event,
        "active_kind": previous.kind,
        "active_source_row": previous.source_row,
        "started_step": monitor.active_started_step,
        "deadline_step": monitor.active_deadline_step,
        "elapsed_physical_steps": (
            None
            if monitor.active_started_step is None
            else int(step - monitor.active_started_step)
        ),
        "contact_on": bool(state["contact_on"]),
    }
    monitor.active_goal_outcomes.append(record)
    if monitor.active_intervention_index is not None:
        monitor.interventions[monitor.active_intervention_index].update(
            {"status": outcome, "outcome": record}
        )
    monitor.active_started_step = None
    monitor.active_deadline_step = None
    monitor.active_completion = None
    monitor.active_intervention_index = None


def _active_outcome(
    monitor: B2Monitor, state: Mapping[str, Any], step: int
) -> str | None:
    if monitor.active_goal.kind == "final":
        return None
    if monitor.active_started_step is None:
        raise RuntimeError("active B2 goal is missing its installation step")
    # A goal installed in callback S has not executed any physical step yet.
    # Even if a forced DROPPED smoke starts with contact already high, it must
    # survive callback S and give the flushed policy a chance to replan/execute
    # at S+1 before contact or distance can complete the intervention.
    if int(step) <= int(monitor.active_started_step):
        return None
    active_switches = [
        switch
        for switch in monitor.goal_switches
        if int(switch["step"]) == int(monitor.active_started_step)
        and switch.get("to_kind") == monitor.active_goal.kind
        and int(switch.get("to_row", -1)) == int(monitor.active_goal.source_row)
    ]
    if len(active_switches) != 1:
        raise RuntimeError(
            "active intervention switch identity drift: "
            f"env={monitor.env_idx}, started={monitor.active_started_step}, "
            f"matches={active_switches}"
        )
    if active_switches[0].get("replan_observed_at_step") is None:
        if monitor.active_deadline_step is not None and step >= monitor.active_deadline_step:
            raise RuntimeError(
                "active intervention reached its deadline without an observed policy replan: "
                f"env={monitor.env_idx}, started={monitor.active_started_step}, step={step}"
            )
        return None
    achieved = bool(
        (monitor.active_completion == "contact" and state["contact_on"])
        or (
            monitor.active_completion == "block_distance"
            and b1._active_distance(state, monitor.active_goal)
            <= b1.SUBGOAL_TOLERANCE_M
        )
    )
    if achieved:
        return "achieved"
    if monitor.active_deadline_step is not None and step >= monitor.active_deadline_step:
        return "timeout"
    return None


def _clamp_position(value: Sequence[float]) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(raw).all():
        raise ValueError("position must be finite")
    return np.asarray(
        [
            np.clip(raw[0], *b1.ID_X),
            np.clip(raw[1], *b1.ID_Y),
            np.clip(raw[2], *b1.ID_Z),
        ],
        dtype=np.float64,
    )


def _rule_choice(
    index: B2MemoryIndex,
    h5: Any,
    event: str,
    state: Mapping[str, Any],
    final_goal: b1.ActiveGoal,
    exclude_episode: int,
) -> tuple[b1.ActiveGoal, dict[str, Any], str, dict[str, Any]]:
    if event == "DROPPED":
        retrieval = index.recovery_candidates(
            np.asarray(state["block_pos"], dtype=np.float64), exclude_episode
        )[0]
        anchor = int(retrieval["anchor_index"])
        if (
            not np.allclose(
                index._block_raw[anchor], _clamp_position(index._block_raw[anchor])
            )
            or not np.allclose(
                index._ee_raw[anchor], _clamp_position(index._ee_raw[anchor])
            )
        ):
            raise RuntimeError(
                "rule DROPPED selected a real frame outside the frozen block/ee ID box"
            )
        goal = b1._goal_from_real_frame(
            h5, retrieval, "ee", strategy="RULE_CONTACT_REGRASP"
        )
        decision = {
            "decision": "RULE_RECOVER",
            "rule": "closest_contact_regrasp_frame",
        }
        return goal, retrieval, "contact", decision
    if event == "STALLED":
        current = np.asarray(state["block_pos"], dtype=np.float64)
        midpoint_raw = (current + np.asarray(final_goal.position)) / 2.0
        midpoint = _clamp_position(midpoint_raw)
        retrieval = index.nearest_block_xyz(midpoint, exclude_episode)
        retrieval["midpoint_before_clamp"] = midpoint_raw
        retrieval["midpoint_after_clamp"] = midpoint
        retrieval["yaw_ignored_for_query"] = True
        goal = b1._goal_from_real_frame(h5, retrieval, "block")
        if not np.allclose(goal.position, _clamp_position(goal.position)):
            raise RuntimeError(
                "rule STALLED midpoint resolved to a real block frame outside frozen ID box"
            )
        decision = {
            "decision": "RULE_SUBGOAL",
            "rule": "current_to_final_xyz_midpoint_real_frame",
        }
        return goal, retrieval, "block_distance", decision
    raise ValueError(event)


def _decision_position(decision: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    return b1._decision_position(decision, keys)


def _apply_brain_decision(
    index: B2MemoryIndex,
    candidate_builder: Any,
    h5: Any,
    event: str,
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    state_payload: Mapping[str, Any],
    exclude_episode: int,
) -> tuple[
    b1.ActiveGoal | None,
    dict[str, Any] | None,
    str | None,
    dict[str, Any],
]:
    result = dict(decision)
    action = str(result.get("decision", "CONTINUE")).upper()
    if action == "CONTINUE":
        return None, None, None, result
    expected_action = "SUBGOAL" if event == "STALLED" else "RECOVER"
    if action != expected_action:
        result.update(
            {
                "applied_decision": "CONTINUE",
                "application_failure": (
                    f"event {event} requires {expected_action}, received {action}"
                ),
            }
        )
        return None, None, None, result
    candidate_id = result.get("candidate_id")
    if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
        result.update(
            {
                "applied_decision": "CONTINUE",
                "application_failure": "candidate_id must be integer 0,1,2",
            }
        )
        return None, None, None, result
    if candidate_id not in range(NUM_CANDIDATES):
        result.update(
            {
                "applied_decision": "CONTINUE",
                "application_failure": "candidate_id outside 0,1,2",
            }
        )
        return None, None, None, result
    resolver = getattr(candidate_builder, "resolve_landing", None)
    if not callable(resolver):
        raise RuntimeError("accepted B2 candidate builder lacks resolve_landing")
    landing = resolver(
        state_payload,
        candidates,
        result,
        int(exclude_episode),
    )
    if not isinstance(landing, Mapping):
        raise RuntimeError("B2 real-frame landing resolver returned non-mapping")
    landing = dict(landing)
    if landing.get("decision") != action:
        raise RuntimeError(
            f"B2 landing decision mismatch: expected={action}, actual={landing.get('decision')}"
        )
    row = int(landing["anchor_row"])
    source_episode = int(landing["source_episode"])
    if source_episode in index.fixed_episodes or source_episode == int(exclude_episode):
        raise RuntimeError("B2 landing resolved to a forbidden source episode")
    anchor = index.anchor_for_row(row)
    if int(index.episodes[anchor]) != source_episode:
        raise RuntimeError("B2 landing row/source episode identity mismatch")
    if action == "RECOVER" and (
        not np.allclose(index._block_raw[anchor], _clamp_position(index._block_raw[anchor]))
        or not np.allclose(index._ee_raw[anchor], _clamp_position(index._ee_raw[anchor]))
    ):
        raise RuntimeError(
            "B2 RECOVER landing real frame has block or ee outside frozen ID box"
        )
    selected = dict(candidates[candidate_id])
    reretrieved = bool(landing.get("reretrieval_performed", False))
    contract = str(landing.get("landing_contract", ""))
    if reretrieved != (contract == "reretrieve_real_frame"):
        raise RuntimeError("B2 landing reretrieval/contract mismatch")
    if not reretrieved and row != int(selected["anchor_row"]):
        raise RuntimeError("exact B2 candidate landing did not preserve its anchor row")

    applied_delta = landing.get("applied_position_delta_l2_m")
    if applied_delta is None or float(applied_delta) > 0.03 + 1e-9:
        raise RuntimeError(
            f"B2 landing position adjustment exceeds 0.03m: {applied_delta}"
        )
    applied_decision = landing.get("applied_decision")
    if not isinstance(applied_decision, Mapping):
        raise RuntimeError("B2 landing lacks audited applied_decision")
    position_field = "block_pos" if action == "SUBGOAL" else "ee_pos"
    applied_position = np.asarray(
        applied_decision[position_field], dtype=np.float64
    ).reshape(3)
    if not np.allclose(applied_position, _clamp_position(applied_position)):
        raise RuntimeError("B2 landing applied position escaped frozen ID box")
    if action == "SUBGOAL":
        applied_yaw_delta = landing.get("applied_yaw_delta_rad")
        if applied_yaw_delta is None or abs(float(applied_yaw_delta)) > np.pi / 12 + 1e-9:
            raise RuntimeError(
                "B2 landing yaw adjustment exceeds frozen pi/12 bound: "
                f"{applied_yaw_delta}"
            )

    retrieval = index._row_record(
        anchor,
        "brain_v2_resolved_real_frame",
        applied_position,
        float(landing.get("retrieval_distance_z", 0.0)),
        (
            float(landing.get("retrieval_distance_z", 0.0)),
            row,
        ),
    )
    retrieval.update(
        {
            "candidate_id": candidate_id,
            "selected_candidate": selected,
            "landing": landing,
            "retrieval_after_numeric_adjustment": reretrieved,
            "landing_contract": contract,
        }
    )
    if action == "SUBGOAL":
        goal = b1._goal_from_real_frame(h5, retrieval, "block")
        completion = "block_distance"
    else:
        strategy = str(result.get("strategy", "REGRASP"))
        goal = b1._goal_from_real_frame(h5, retrieval, "ee", strategy=strategy)
        completion = "contact"
    result["applied_decision"] = action
    result["selected_candidate_id"] = candidate_id
    result["resolved_landing"] = landing
    return goal, retrieval, completion, result


def _observe_pending_replans(
    monitor: B2Monitor, policy: Any, recorder: Any, step: int
) -> list[float]:
    """Consume new planner cycles before any callback can change active goals.

    A recorder cycle itself is the authoritative evidence that the flushed
    policy replanned.  Cost arrays are normally populated synchronously, but
    replan attribution must not disappear if a cycle is present before its
    diagnostic cost list is populated.
    """

    cycles = recorder.records[monitor.env_idx]
    start = int(monitor.seen_planning_cycles)
    if start < 0 or start > len(cycles):
        raise RuntimeError(
            "planner cycle cursor drift: "
            f"env={monitor.env_idx}, cursor={start}, cycles={len(cycles)}"
        )
    new_cycles = list(cycles[start:])
    updates: list[float] = []
    for cycle in new_cycles:
        costs = cycle.get("costs", [])
        if costs:
            final_costs = np.asarray(costs[-1], dtype=np.float64)
            value = float(np.min(final_costs))
            if not np.isfinite(value):
                raise RuntimeError(
                    f"nonfinite planner best cost: env={monitor.env_idx}, value={value}"
                )
            monitor.comparable_costs.append(value)
            updates.append(value)
        else:
            updates.append(float("nan"))
    monitor.seen_planning_cycles = len(cycles)

    policy_events = policy.b2_replan_events[monitor.env_idx]
    policy_start = int(monitor.seen_policy_replans)
    if policy_start < 0 or policy_start > len(policy_events):
        raise RuntimeError(
            "policy replan cursor drift: "
            f"env={monitor.env_idx}, cursor={policy_start}, events={len(policy_events)}"
        )
    new_policy_events = list(policy_events[policy_start:])
    monitor.seen_policy_replans = len(policy_events)
    for event in new_policy_events:
        observed_step = int(event["callback_step_expected"])
        if observed_step != int(step):
            raise RuntimeError(
                "policy replan observation reached the wrong callback: "
                f"env={monitor.env_idx}, expected={observed_step}, actual={step}"
            )
        matching_cycle = next(
            (
                cycle
                for cycle in new_cycles
                if int(cycle.get("env_step", -1)) == int(event["trust_env_step"])
            ),
            None,
        )
        if matching_cycle is None:
            raise RuntimeError(
                "B2 forced goal replan event lacks its exact recorder cycle: "
                f"env={monitor.env_idx}, event={event}, new_cycles={new_cycles}"
            )
        value = None
        if matching_cycle is not None and matching_cycle.get("costs"):
            value = float(
                np.min(
                    np.asarray(matching_cycle["costs"][-1], dtype=np.float64)
                )
            )
        matches = [
            switch
            for switch in monitor.goal_switches
            if switch["replan_observed_at_step"] is None
            and int(switch["step"]) == int(event["switch_step"])
            and int(switch.get("to_row", -1)) == int(event["goal_row"])
            and str(switch.get("to_kind")) == str(event["goal_kind"])
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "B2 policy replan event does not identify exactly one goal switch: "
                f"env={monitor.env_idx}, event={event}, matches={matches}"
            )
        switch = matches[0]
        switch["replan_observed_at_step"] = observed_step
        switch["replan_best_cost"] = value
        switch["replan_latency_env_steps"] = int(
            observed_step - int(switch["step"])
        )
        switch["replan_cycle_idx"] = (
            None if matching_cycle is None else matching_cycle.get("cycle_idx")
        )
        switch["replan_cycle_env_step"] = event["trust_env_step"]
        switch["replan_cost_available_at_observation"] = value is not None
        switch["replan_observation_source"] = "b2_forced_goal_replan_request"
        switch["replan_action_executed"] = bool(event["action_returned"])
        switch["solver_input_goal_sha256"] = event[
            "solver_input_goal_sha256"
        ]
        switch["solver_input_goal_history_sha256"] = event[
            "solver_input_goal_history_sha256"
        ]
        switch["solver_input_goal_history_slices"] = int(
            event["solver_input_goal_history_slices"]
        )
        switch["solver_input_goal_all_history_match_request"] = bool(
            event["solver_input_goal_all_history_match_request"]
        )
        switch["solver_input_goal_matches_request"] = bool(
            event["solver_input_goal_matches_request"]
        )
        switch["trust_cycle_before"] = int(event["trust_cycle_before"])
        switch["trust_cycle_after"] = int(event["trust_cycle_after"])
        switch["trust_cycle_delta"] = int(event["trust_cycle_delta"])
        switch["recorder_cycle_count_before"] = int(
            event["recorder_cycle_count_before"]
        )
        switch["recorder_cycle_count_after"] = int(
            event["recorder_cycle_count_after"]
        )
        switch["recorder_cycle_delta"] = int(event["recorder_cycle_delta"])
        switch["recorder_cycle_env_step"] = int(
            event["recorder_cycle_env_step"]
        )
        switch["returned_action_sha256"] = event["returned_action_sha256"]
    return [value for value in updates if np.isfinite(value)]


def _make_b2_observed_policy(base: type) -> type:
    """Force and prove a real solve immediately after every B2 goal switch."""

    class B2ObservedPolicy(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.b2_replan_events: dict[int, list[dict[str, Any]]] = {
                env_idx: [] for env_idx in range(len(self.eval_rows))
            }
            self.b2_pending_goal_replans: dict[int, dict[str, Any]] = {}

        def request_b2_goal_replan(
            self,
            *,
            env_idx: int,
            switch_step: int,
            goal_kind: str,
            goal_row: int,
            goal_pixels_sha256: str,
        ) -> dict[str, Any]:
            env_idx = int(env_idx)
            if env_idx in self.b2_pending_goal_replans:
                raise RuntimeError(
                    "B2 goal switch replaced an unconsumed replan request: "
                    f"env={env_idx}, pending={self.b2_pending_goal_replans[env_idx]}"
                )
            if int(self._trust_env_step) != int(switch_step):
                raise RuntimeError(
                    "B2 goal switch callback is not aligned with policy env step: "
                    f"env={env_idx}, switch_step={switch_step}, "
                    f"trust_env_step={self._trust_env_step}"
                )
            before = int(len(self._action_buffer[env_idx]))
            self._action_buffer[env_idx].clear()
            if self._next_init is not None:
                self._next_init[env_idx] = 0
            request = {
                "env_idx": env_idx,
                "switch_step": int(switch_step),
                "expected_trust_env_step": int(switch_step),
                "expected_callback_step": int(switch_step + 1),
                "goal_kind": str(goal_kind),
                "goal_row": int(goal_row),
                "goal_pixels_sha256": str(goal_pixels_sha256),
                "buffer_length_before_request_clear": before,
                "buffer_length_after_request_clear": int(
                    len(self._action_buffer[env_idx])
                ),
            }
            self.b2_pending_goal_replans[env_idx] = request
            return dict(request)

        @staticmethod
        def _goal_pixels_provenance(
            info_dict: Mapping[str, Any], env_idx: int
        ) -> dict[str, Any]:
            if "goal" not in info_dict:
                raise RuntimeError("B2 replan request reached policy without goal input")
            value = info_dict["goal"]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            pixels = np.asarray(value)[env_idx]
            if pixels.ndim < 3 or tuple(pixels.shape[-3:]) != (224, 224, 3):
                raise RuntimeError(
                    "B2 policy goal input has unexpected shape: "
                    f"env={env_idx}, shape={np.asarray(value).shape}"
                )
            history = pixels.reshape(-1, *pixels.shape[-3:])
            hashes = [
                hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
                for frame in history
            ]
            # jepa.py consumes goal[:, 0].  Record that exact slice and also
            # require every history slice to be the same literal HDF5 frame so
            # the proof remains valid if history selection changes later.
            return {
                "consumed_sha256": hashes[0],
                "history_sha256": hashes,
                "history_slices": int(len(hashes)),
                "all_history_equal": len(set(hashes)) == 1,
            }

        def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
            terminated = info_dict.get("terminated")
            dead = (
                np.asarray(terminated, dtype=bool).reshape(self.env.num_envs, -1)[:, 0]
                if terminated is not None
                else np.zeros(self.env.num_envs, dtype=bool)
            )
            trust_env_step = int(self._trust_env_step)
            due_requests: dict[int, dict[str, Any]] = {}
            for env_idx, request in sorted(self.b2_pending_goal_replans.items()):
                if dead[env_idx]:
                    raise RuntimeError(
                        "B2 goal replan request targeted an already-dead policy slot: "
                        f"env={env_idx}, trust_env_step={trust_env_step}, request={request}"
                    )
                expected = int(request["expected_trust_env_step"])
                if trust_env_step != expected:
                    raise RuntimeError(
                        "B2 goal replan request missed the next policy boundary: "
                        f"env={env_idx}, expected={expected}, actual={trust_env_step}, "
                        f"request={request}"
                    )
                goal_provenance = self._goal_pixels_provenance(info_dict, env_idx)
                goal_sha256 = str(goal_provenance["consumed_sha256"])
                all_history_match = bool(
                    goal_provenance["all_history_equal"]
                    and all(
                        item == request["goal_pixels_sha256"]
                        for item in goal_provenance["history_sha256"]
                    )
                )
                if goal_sha256 != request["goal_pixels_sha256"] or not all_history_match:
                    raise RuntimeError(
                        "B2 dynamic goal pixels did not fill every next-policy history slice: "
                        f"env={env_idx}, expected={request['goal_pixels_sha256']}, "
                        f"actual={goal_provenance}"
                    )
                request = dict(request)
                request["solver_input_goal_sha256"] = goal_sha256
                request["solver_input_goal_history_sha256"] = list(
                    goal_provenance["history_sha256"]
                )
                request["solver_input_goal_history_slices"] = int(
                    goal_provenance["history_slices"]
                )
                request["buffer_length_before_boundary_clear"] = int(
                    len(self._action_buffer[env_idx])
                )
                self._action_buffer[env_idx].clear()
                if self._next_init is not None:
                    self._next_init[env_idx] = 0
                request["buffer_length_after_boundary_clear"] = int(
                    len(self._action_buffer[env_idx])
                )
                request["trust_cycle_before"] = int(self._trust_cycles[env_idx])
                request["recorder_cycle_count_before"] = int(
                    len(self.cost_recorder.records[env_idx])
                )
                due_requests[env_idx] = request

            replans = [
                env_idx
                for env_idx in range(self.env.num_envs)
                if not dead[env_idx] and len(self._action_buffer[env_idx]) == 0
            ]
            buffer_lengths_before = {
                env_idx: len(self._action_buffer[env_idx]) for env_idx in replans
            }
            missing = sorted(set(due_requests) - set(replans))
            if missing:
                raise RuntimeError(
                    "B2 forced goal replan did not empty the executing buffers: "
                    f"envs={missing}"
                )
            action = super().get_action(info_dict, **kwargs)
            action_array = np.asarray(action)
            for env_idx, request in due_requests.items():
                trust_cycle_after = int(self._trust_cycles[env_idx])
                recorder_cycle_count_after = int(
                    len(self.cost_recorder.records[env_idx])
                )
                trust_cycle_delta = trust_cycle_after - int(
                    request["trust_cycle_before"]
                )
                recorder_cycle_delta = recorder_cycle_count_after - int(
                    request["recorder_cycle_count_before"]
                )
                returned_action = np.asarray(action_array[env_idx], dtype=np.float32)
                if trust_cycle_delta != 1 or recorder_cycle_delta != 1:
                    raise RuntimeError(
                        "B2 forced goal replan did not produce exactly one real T2 solve: "
                        f"env={env_idx}, trust_delta={trust_cycle_delta}, "
                        f"recorder_delta={recorder_cycle_delta}"
                    )
                new_recorder_cycle = self.cost_recorder.records[env_idx][
                    int(request["recorder_cycle_count_before"])
                ]
                if int(new_recorder_cycle.get("env_step", -1)) != trust_env_step:
                    raise RuntimeError(
                        "B2 forced goal replan recorder cycle has wrong env step: "
                        f"env={env_idx}, expected={trust_env_step}, "
                        f"cycle={new_recorder_cycle}"
                    )
                if not np.isfinite(returned_action).all():
                    raise RuntimeError(
                        f"B2 forced goal replan returned nonfinite action: env={env_idx}"
                    )
                self.b2_replan_events[env_idx].append(
                    {
                        "env_idx": int(env_idx),
                        "trust_env_step": trust_env_step,
                        "callback_step_expected": int(
                            request["expected_callback_step"]
                        ),
                        "switch_step": int(request["switch_step"]),
                        "goal_kind": str(request["goal_kind"]),
                        "goal_row": int(request["goal_row"]),
                        "buffer_length_before": int(buffer_lengths_before[env_idx]),
                        "buffer_length_after_action": int(
                            len(self._action_buffer[env_idx])
                        ),
                        "solver_input_goal_sha256": request[
                            "solver_input_goal_sha256"
                        ],
                        "solver_input_goal_history_sha256": request[
                            "solver_input_goal_history_sha256"
                        ],
                        "solver_input_goal_history_slices": request[
                            "solver_input_goal_history_slices"
                        ],
                        "solver_input_goal_all_history_match_request": True,
                        "solver_input_goal_matches_request": True,
                        "trust_cycle_before": int(request["trust_cycle_before"]),
                        "trust_cycle_after": trust_cycle_after,
                        "trust_cycle_delta": trust_cycle_delta,
                        "recorder_cycle_count_before": int(
                            request["recorder_cycle_count_before"]
                        ),
                        "recorder_cycle_count_after": recorder_cycle_count_after,
                        "recorder_cycle_delta": recorder_cycle_delta,
                        "recorder_cycle_env_step": int(
                            new_recorder_cycle["env_step"]
                        ),
                        "returned_action_sha256": hashlib.sha256(
                            np.ascontiguousarray(returned_action).tobytes()
                        ).hexdigest(),
                        "action_returned": True,
                    }
                )
                del self.b2_pending_goal_replans[env_idx]
            return action

    return B2ObservedPolicy


def _behavior_summary(
    monitors: Sequence[B2Monitor],
    successes: np.ndarray,
    supervisor_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    interventions = [item for monitor in monitors for item in monitor.interventions]
    outcomes = [item for monitor in monitors for item in monitor.active_goal_outcomes]
    triggers = [item for monitor in monitors for item in monitor.triggers]
    episodes_with_intervention = {
        monitor.env_idx for monitor in monitors if monitor.interventions
    }
    success_after_intervention = sum(
        bool(successes[env_idx]) for env_idx in episodes_with_intervention
    )
    decisions = Counter(
        str(item.get("decision", {}).get("decision", "NONE"))
        for item in triggers
        if isinstance(item.get("decision"), Mapping)
    )
    selected = Counter(
        int(item["decision"]["selected_candidate_id"])
        for item in triggers
        if isinstance(item.get("decision"), Mapping)
        and isinstance(item["decision"].get("selected_candidate_id"), int)
    )
    outcome_counts = Counter(
        (
            "aborted"
            if str(item["outcome"]).startswith("aborted")
            else str(item["outcome"])
        )
        for item in outcomes
    )
    retreat_profiles = Counter()
    retreat_fallback_reasons = Counter()
    for trigger in triggers:
        candidates = trigger.get("candidates")
        if trigger.get("event") != "STALLED" or not isinstance(candidates, list):
            continue
        if len(candidates) != NUM_CANDIDATES or not isinstance(candidates[2], Mapping):
            continue
        retreat = candidates[2]
        profile = "strict_reverse" if retreat.get("strict_reverse") else "detour"
        retreat_profiles[profile] += 1
        if retreat.get("fallback_reason"):
            retreat_fallback_reasons[str(retreat["fallback_reason"])] += 1
    payload = {
        "trigger_count": len(triggers),
        "trigger_count_by_event": dict(Counter(item["event"] for item in triggers)),
        "suppression_count_by_reason": dict(
            Counter(
                str(item["suppression_reason"])
                for item in triggers
                if item.get("suppression_reason")
            )
        ),
        "intervention_count": len(interventions),
        "intervention_count_per_episode": [
            len(monitor.interventions) for monitor in monitors
        ],
        "intervention_outcome_counts": dict(outcome_counts),
        "intervention_achievement_rate": (
            outcome_counts["achieved"] / len(outcomes) if outcomes else None
        ),
        "episodes_with_intervention": len(episodes_with_intervention),
        "episodes_with_intervention_and_final_success": success_after_intervention,
        "episode_final_success_rate_after_any_intervention": (
            success_after_intervention / len(episodes_with_intervention)
            if episodes_with_intervention
            else None
        ),
        "decision_counts": dict(decisions),
        "selected_candidate_counts": {str(key): value for key, value in selected.items()},
        "stalled_retreat_candidate_profiles": dict(retreat_profiles),
        "stalled_retreat_fallback_reasons": dict(retreat_fallback_reasons),
        "shared_execution_guard": {
            "active_goal_timeout_physical_steps": ACTIVE_GOAL_TIMEOUT_STEPS,
            "stalled_suppressed_while_active": True,
            "dropped_may_replace_active_subject_to_shared_cooldown_and_budget": True,
        },
    }
    if supervisor_summary is not None:
        budget = supervisor_summary.get("budget", {})
        accounted = int(budget.get("accounted_tokens", 0))
        limit = int(budget.get("max_total_tokens", 1_000_000))
        if accounted > limit:
            raise RuntimeError(
                f"B2 token budget exceeded: accounted={accounted}, limit={limit}"
            )
        payload["llm_cost"] = {
            "total_tokens": accounted,
            "total_tokens_source": "supervisor_budget_accounted_tokens",
            "token_budget": limit,
            "within_token_budget": accounted <= limit,
            "logical_calls": supervisor_summary.get("logical_calls"),
            "http_attempts": supervisor_summary.get("http_attempts"),
            "average_attempt_latency_ms": supervisor_summary.get(
                "mean_attempt_latency_ms"
            ),
        }
        payload["supervisor"] = supervisor_summary
    return payload


def _brain_payload(
    brain_module: Any,
    candidate_builder: Any,
    monitor: B2Monitor,
    state: Mapping[str, Any],
    event: str,
    step: int,
    exclude_episode: int,
    calls_remaining: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    trend = [float(value) for value in list(monitor.dist_history)[-5:]]
    if not trend:
        trend = [b1._final_distance(state, monitor.final_goal)]
    trend = [trend[0]] * (5 - len(trend)) + trend
    costs = [float(value) for value in monitor.comparable_costs[-5:]]
    if not costs:
        costs = [0.0]
    costs = [costs[0]] * (5 - len(costs)) + costs
    values = {
        "event": event,
        "step": int(step),
        "budget": BUDGET,
        "block_pos": np.asarray(state["block_pos"]).round(6).tolist(),
        "block_yaw": round(float(state["block_yaw"]), 6),
        "target_pos": monitor.final_goal.position.round(6).tolist(),
        "target_yaw": round(float(monitor.final_goal.yaw), 6),
        "ee_pos": np.asarray(state["ee_pos"]).round(6).tolist(),
        "gripper_opening": round(float(state["gripper_opening"]), 6),
        "gripper_contact": bool(state["contact_on"]),
        "dist_to_target": round(b1._final_distance(state, monitor.final_goal), 6),
        "dist_trend_5": [round(value, 6) for value in trend],
        "grasp_state": b1._grasp_state(state, event),
        "phase": monitor.previous_phase,
        "planner_cost_trend": [round(value, 6) for value in costs],
        "calls_remaining": int(calls_remaining),
    }
    v1_protocol = getattr(brain_module, "b1", None)
    builder = getattr(v1_protocol, "build_state_payload", None)
    if not callable(builder):
        raise RuntimeError("brain_supervisor_v2 lacks frozen state payload builder")
    state_payload = builder(**values)
    public_candidates, candidate_audit = candidate_builder.build_candidates(
        state_payload, int(exclude_episode)
    )
    if event == "STALLED":
        retreat = public_candidates[2]
        actual_displacement = float(retreat["current_to_candidate_distance"])
        retreat_intent = str(retreat.get("intent", ""))
        strict_reverse = retreat.get("strict_reverse")
        fallback_reason = retreat.get("fallback_reason")
        if (
            actual_displacement < 0.045 - 2e-6
            or not bool(retreat["retreat_then_advance"])
            or strict_reverse not in (True, False)
            or (
                strict_reverse is True
                and (retreat_intent != "retreat" or fallback_reason is not None)
            )
            or (
                strict_reverse is False
                and (
                    retreat_intent != "detour"
                    or not isinstance(fallback_reason, str)
                    or not fallback_reason
                )
            )
        ):
            raise RuntimeError(
                "B2 retreat candidate failed frozen physical guard: "
                f"current_displacement={actual_displacement}, "
                f"retreat_then_advance={retreat['retreat_then_advance']}, "
                f"intent={retreat_intent}, strict_reverse={strict_reverse}, "
                f"fallback_reason={fallback_reason}"
            )
    validator = getattr(brain_module, "validate_candidate_payload", None)
    if not callable(validator):
        raise RuntimeError("brain_supervisor_v2 lacks candidate validator")
    payload = validator(
        {"state": state_payload, "retrieval_candidates": public_candidates}
    )
    encoded = json.dumps(b1._jsonable(payload), separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 2400:
        raise RuntimeError(f"B2 symbolic payload unexpectedly large: {len(encoded)} bytes")
    return payload, list(public_candidates), list(candidate_audit)


def _supervisor_decide(
    supervisor: Any,
    payload: Mapping[str, Any],
    env_idx: int,
    step: int,
) -> dict[str, Any]:
    result = supervisor.decide(
        state=payload["state"],
        candidates=payload["retrieval_candidates"],
        replay_id=f"online-env{env_idx:02d}-step{step:03d}",
        episode_id=str(env_idx),
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("brain-v2 supervisor decision must be a mapping")
    value = dict(result)
    action = str(value.get("decision", "CONTINUE")).upper()
    if action not in {"CONTINUE", "SUBGOAL", "RECOVER"}:
        value.update(
            {
                "decision": "CONTINUE",
                "protocol_failure": True,
                "normalization_failure": f"unknown decision {action}",
            }
        )
    else:
        value["decision"] = action
    return value


def _run_evaluation(
    args: argparse.Namespace,
    output: Path,
    world: Any,
    policy: Any,
    recorder: Any,
    dataset: Any,
    index: B2MemoryIndex,
    selection: Mapping[str, Any],
    rows: np.ndarray,
    source_formal_env_indices: np.ndarray,
    episodes: np.ndarray,
    starts: np.ndarray,
    init_state: Mapping[str, np.ndarray],
    goal_state: Mapping[str, np.ndarray],
    dataset_videos: list[np.ndarray],
    supervisor: Any | None,
    brain_module: Any | None,
    candidate_builder: Any | None,
) -> tuple[dict[str, Any], list[B2Monitor], dict[str, Any]]:
    import hdf5plugin  # noqa: F401
    import h5py
    from stable_worldmodel.plot import save_panel_videos

    goal_rows = np.asarray(selection["goal_rows"], dtype=np.int64)[
        source_formal_env_indices
    ]
    monitors = _initial_monitors(
        init_state, goal_state, rows, goal_rows, episodes
    )
    successes = np.zeros(args.num_eval, dtype=bool)
    frames: dict[int, list[np.ndarray]] = defaultdict(list)
    goal_frames: dict[int, list[np.ndarray]] = defaultdict(list)
    event_log: list[dict[str, Any]] = []
    retrieval_log: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []
    step_counter = 0
    latest_states: list[dict[str, Any] | None] = [None] * args.num_eval

    h5 = h5py.File(args.dataset, "r", swmr=True)
    try:
        def on_step(active_world: Any) -> None:
            nonlocal step_counter
            step_counter += 1
            successes[:] |= np.asarray(active_world.terminateds, dtype=bool)
            for env_idx, monitor in enumerate(monitors):
                # EnvPool returns False terminal flags for already-masked slots
                # while leaving their stacked infos unchanged.  Persist the
                # first physical done event so a dead slot can never re-enter
                # trigger/LLM/retrieval/goal-switch logic on later callbacks.
                if monitor.episode_done:
                    if args.force_smoke_both_events and step_counter == 5:
                        raise RuntimeError(
                            "B2 forced smoke selected an episode that ended before its "
                            f"step-5 event: env={env_idx}, done_step={monitor.episode_done_step}"
                        )
                    continue
                state = b1._live_state(active_world.infos, env_idx)
                state["contact_on"] = b1._contact_with_hysteresis(
                    float(state["gripper_contact_raw"]), monitor.previous_contact_on
                )
                latest_states[env_idx] = state
                # This must remain before terminal handling, active-goal
                # completion/timeout, abort/replacement, and every goal switch.
                # The step-S+1 planning cycle belongs to the goal installed at
                # S even when contact is already true at S+1.
                _observe_pending_replans(monitor, policy, recorder, step_counter)

                terminated_now = bool(active_world.terminateds[env_idx])
                truncated_now = bool(active_world.truncateds[env_idx])
                if terminated_now or truncated_now:
                    monitor.episode_done = True
                    monitor.episode_done_step = int(step_counter)
                    monitor.episode_terminated = terminated_now
                    monitor.episode_truncated = truncated_now
                    if monitor.active_goal.kind != "final":
                        _finish_active_goal(
                            monitor,
                            policy,
                            step_counter,
                            state,
                            (
                                "aborted_final_success"
                                if terminated_now
                                else "aborted_episode_truncated"
                            ),
                            future_action_expected=False,
                            no_replan_reason=(
                                "terminal_final_success"
                                if terminated_now
                                else "episode_truncated"
                            ),
                        )
                    monitor.previous_phase = b1._derive_phase(
                        state, monitor.active_goal.kind
                    )
                    monitor.previous_contact_on = bool(state["contact_on"])
                    monitor.previous_contact_raw = float(state["gripper_contact_raw"])
                    pixel = np.asarray(active_world.infos["pixels"][env_idx])
                    frames[env_idx].append(
                        (pixel[-1] if pixel.ndim > 3 else pixel).copy()
                    )
                    goal_frames[env_idx].append(monitor.active_goal.pixels.copy())
                    continue

                outcome = _active_outcome(monitor, state, step_counter)
                skip_trigger = outcome is not None
                if outcome is not None:
                    _finish_active_goal(
                        monitor,
                        policy,
                        step_counter,
                        state,
                        str(outcome),
                        future_action_expected=step_counter < BUDGET,
                        no_replan_reason=(
                            None
                            if step_counter < BUDGET
                            else "evaluation_budget_exhausted"
                        ),
                    )

                final_distance = b1._final_distance(state, monitor.final_goal)
                if not skip_trigger:
                    monitor.dist_history.append(final_distance)
                forced_smoke_event = None
                if args.force_smoke_both_events and step_counter == 5:
                    forced_smoke_event = "STALLED" if env_idx == 0 else "DROPPED"
                force_smoke = bool(
                    forced_smoke_event is not None
                    or (
                        args.force_smoke_trigger_step is not None
                        and env_idx == 0
                        and step_counter == args.force_smoke_trigger_step
                    )
                )
                event = None
                diagnostics: dict[str, Any] = {}
                if not skip_trigger:
                    if forced_smoke_event is not None:
                        event = forced_smoke_event
                        diagnostics = {
                            "forced_smoke_diagnostic": True,
                            "forced_event": forced_smoke_event,
                            "excluded_from_formal_behavior": True,
                        }
                    else:
                        event, diagnostics = b1._detect_event(
                            monitor, state, force_smoke
                        )

                active = monitor.active_goal.kind != "final"
                active_stalled_suppression = bool(active and event == "STALLED")
                cooldown_ok = (
                    step_counter - monitor.last_call_step
                    >= b1.MIN_CALL_INTERVAL_STEPS
                )
                budget_ok = monitor.logical_calls < b1.MAX_CALLS_PER_EPISODE
                future_action_available = step_counter < BUDGET
                call_allowed = bool(
                    event is not None
                    and not active_stalled_suppression
                    and cooldown_ok
                    and budget_ok
                    and future_action_available
                )
                record = None
                if event is not None:
                    record = {
                        "env_idx": env_idx,
                        "episode": int(episodes[env_idx]),
                        "step": step_counter,
                        "event": event,
                        "phase_before_step": monitor.previous_phase,
                        "state": state,
                        "diagnostics": diagnostics,
                        "active_goal_before_event": monitor.active_goal.kind,
                        "call_allowed": call_allowed,
                        "suppression_reason": None,
                        "decision": None,
                        "candidates": None,
                    }
                    if active_stalled_suppression:
                        record["suppression_reason"] = "suppressed_active_intervention"
                    elif not cooldown_ok:
                        record["suppression_reason"] = "minimum_call_interval"
                    elif not budget_ok:
                        record["suppression_reason"] = "episode_intervention_budget_exhausted"
                    elif not future_action_available:
                        record["suppression_reason"] = "no_future_action_within_eval_budget"

                if call_allowed:
                    assert event is not None and record is not None
                    goal = None
                    retrieval = None
                    completion = None
                    decision: dict[str, Any]
                    candidates: list[dict[str, Any]] | None = None
                    if args.mode == "rule":
                        goal, retrieval, completion, decision = _rule_choice(
                            index,
                            h5,
                            event,
                            state,
                            monitor.final_goal,
                            int(episodes[env_idx]),
                        )
                        monitor.logical_calls += 1
                        monitor.last_call_step = step_counter
                    else:
                        assert supervisor is not None and brain_module is not None
                        assert candidate_builder is not None
                        payload, public_candidates, candidate_audit = _brain_payload(
                            brain_module,
                            candidate_builder,
                            monitor,
                            state,
                            event,
                            step_counter,
                            int(episodes[env_idx]),
                            b1.MAX_CALLS_PER_EPISODE - monitor.logical_calls,
                        )
                        candidates = [dict(item) for item in public_candidates]
                        monitor.logical_calls += 1
                        monitor.last_call_step = step_counter
                        decision = _supervisor_decide(
                            supervisor, payload, env_idx, step_counter
                        )
                        record["payload"] = payload
                        try:
                            goal, retrieval, completion, decision = _apply_brain_decision(
                                index,
                                candidate_builder,
                                h5,
                                event,
                                decision,
                                candidates,
                                payload["state"],
                                int(episodes[env_idx]),
                            )
                        except (KeyError, TypeError, ValueError) as error:
                            decision = dict(decision)
                            decision.update(
                                {
                                    "applied_decision": "CONTINUE",
                                    "application_failure": (
                                        f"{type(error).__name__}: {error}"
                                    ),
                                }
                            )
                            goal = retrieval = completion = None
                        if (
                            goal is None
                            and args.force_smoke_goal_switch
                            and force_smoke
                        ):
                            selected = candidates[0]
                            anchor = index.anchor_for_row(int(selected["anchor_row"]))
                            retrieval = index._row_record(
                                anchor,
                                "brain_v2_forced_smoke_real_frame",
                                np.asarray(selected["block_pos"], dtype=np.float64),
                                float(candidate_audit[0]["retrieval_distance_z"]),
                                (
                                    float(candidate_audit[0]["retrieval_distance_z"]),
                                    int(selected["anchor_row"]),
                                ),
                            )
                            if event == "STALLED":
                                goal = b1._goal_from_real_frame(
                                    h5, retrieval, "block"
                                )
                                completion = "block_distance"
                            else:
                                goal = b1._goal_from_real_frame(
                                    h5,
                                    retrieval,
                                    "ee",
                                    strategy="SMOKE_SYNTHETIC_REGRASP",
                                )
                                completion = "contact"
                            decision = dict(decision)
                            decision["synthetic_smoke_goal_switch"] = {
                                "enabled": True,
                                "candidate_id": 0,
                                "reason": (
                                    "forced smoke wiring diagnostic after no applicable "
                                    "LLM intervention; excluded from formal protocol"
                                ),
                            }
                        record["candidates"] = public_candidates
                        candidate_log.append(
                            {
                                "env_idx": env_idx,
                                "episode": int(episodes[env_idx]),
                                "step": step_counter,
                                "event": event,
                                "candidates": public_candidates,
                                "candidate_audit": candidate_audit,
                                "decision": decision,
                            }
                        )
                    record["decision"] = decision

                    if goal is not None and retrieval is not None and completion is not None:
                        if monitor.active_goal.kind != "final":
                            _abort_active_goal(
                                monitor,
                                step_counter,
                                state,
                                replacement_event=event,
                            )
                        intervention = {
                            "intervention_index": len(monitor.interventions),
                            "env_idx": env_idx,
                            "episode": int(episodes[env_idx]),
                            "step": step_counter,
                            "event": event,
                            "mode": args.mode,
                            "decision": decision,
                            "retrieval": retrieval,
                            "selected_real_frame": {
                                "row": goal.source_row,
                                "episode": goal.source_episode,
                                "position": goal.position,
                                "yaw": goal.yaw,
                                "pixels_sha256": hashlib.sha256(
                                    np.ascontiguousarray(goal.pixels).tobytes()
                                ).hexdigest(),
                            },
                            "status": "active",
                        }
                        monitor.interventions.append(intervention)
                        if goal.target_kind == "block":
                            monitor.subgoals_started += 1
                        else:
                            monitor.recoveries_started += 1
                            monitor.pending_recovery_success_credit = True
                        _install_active_goal(
                            monitor,
                            goal,
                            policy,
                            step_counter,
                            f"{args.mode}_{event.lower()}",
                            state,
                            retrieval,
                            completion,
                            len(monitor.interventions) - 1,
                        )
                        retrieval_log.append(intervention)

                if record is not None:
                    monitor.triggers.append(record)
                    event_log.append(record)

                monitor.previous_phase = b1._derive_phase(
                    state, monitor.active_goal.kind
                )
                monitor.previous_contact_on = bool(state["contact_on"])
                monitor.previous_contact_raw = float(state["gripper_contact_raw"])
                active_world.infos["goal"][env_idx] = np.broadcast_to(
                    monitor.active_goal.pixels,
                    active_world.infos["goal"][env_idx].shape,
                )
                pixel = np.asarray(active_world.infos["pixels"][env_idx])
                frames[env_idx].append(
                    (pixel[-1] if pixel.ndim > 3 else pixel).copy()
                )
                goal_frames[env_idx].append(monitor.active_goal.pixels.copy())

        for env_idx, monitor in enumerate(monitors):
            world.infos["goal"][env_idx] = np.broadcast_to(
                monitor.final_goal.pixels, world.infos["goal"][env_idx].shape
            )
        world._run(max_steps=BUDGET, mode="wait", on_step=on_step)
    finally:
        h5.close()

    if policy.b2_pending_goal_replans:
        raise RuntimeError(
            "B2 evaluation ended with unexplained pending goal replans: "
            f"{policy.b2_pending_goal_replans}"
        )

    if args.force_smoke_both_events:
        expected_events = ("STALLED", "DROPPED")
        for env_idx, expected_event in enumerate(expected_events):
            forced = [
                item
                for item in monitors[env_idx].triggers
                if item.get("step") == 5
                and item.get("diagnostics", {}).get("forced_smoke_diagnostic")
            ]
            if len(forced) != 1 or forced[0].get("event") != expected_event:
                raise RuntimeError(
                    "B2 both-event smoke did not emit its frozen event: "
                    f"env={env_idx}, expected={expected_event}, actual={forced}"
                )
            matching = [
                item
                for item in monitors[env_idx].interventions
                if item.get("step") == 5 and item.get("event") == expected_event
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    "B2 both-event smoke did not install exactly one real-frame goal: "
                    f"env={env_idx}, event={expected_event}, actual={matching}"
                )
            decision = matching[0].get("decision", {})
            if args.mode == "brainv2":
                expected_decision = (
                    "SUBGOAL" if expected_event == "STALLED" else "RECOVER"
                )
                if decision.get("decision") != expected_decision:
                    raise RuntimeError(
                        "B2 brainv2 both-event smoke requires the real API to choose "
                        f"{expected_decision} for {expected_event}; actual={decision}"
                    )
                call_record = decision.get("call_record", {})
                if call_record.get("status") != "ok":
                    raise RuntimeError(
                        "B2 brainv2 forced smoke API call was not clean: "
                        f"{call_record}"
                    )
            switches = [
                item
                for item in monitors[env_idx].goal_switches
                if item.get("step") == 5 and item.get("to_kind") != "final"
            ]
            smoke_switch_ok = bool(
                len(switches) == 1
                and switches[0].get("action_buffer_flushed") is True
                and switches[0].get("replan_observed_at_step") == 6
                and switches[0].get("replan_latency_env_steps") == 1
                and switches[0].get("replan_observation_source")
                == "b2_forced_goal_replan_request"
                and switches[0].get("replan_action_executed") is True
                and switches[0].get("solver_input_goal_matches_request") is True
                and switches[0].get(
                    "solver_input_goal_all_history_match_request"
                )
                is True
                and switches[0].get("trust_cycle_delta") == 1
                and switches[0].get("recorder_cycle_delta") == 1
            )
            if not smoke_switch_ok:
                raise RuntimeError(
                    "B2 both-event smoke did not verify step-5 goal switch, buffer flush, "
                    "and exact step-6 replan: "
                    f"env={env_idx}, switches={switches}"
                )

    for env_idx, monitor in enumerate(monitors):
        if successes[env_idx] and monitor.pending_recovery_success_credit:
            monitor.recoveries_followed_by_final_success += 1
        if monitor.active_goal.kind != "final":
            state = latest_states[env_idx]
            if state is None:
                raise RuntimeError("missing final live state for active B2 intervention")
            _abort_active_goal(
                monitor,
                step_counter,
                state,
                replacement_event="EVAL_BUDGET_EXHAUSTED",
                outcome="aborted_budget_exhausted",
            )

    save_panel_videos(
        output / "videos",
        {"agent": frames, "dataset": dataset_videos, "goal": goal_frames},
    )
    goal_switches = [
        switch for monitor in monitors for switch in monitor.goal_switches
    ]
    outcomes = [
        outcome for monitor in monitors for outcome in monitor.active_goal_outcomes
    ]
    _write_json(output / "events.json", event_log)
    _write_json(output / "subgoal_retrieval.json", retrieval_log)
    _write_json(output / "candidate_sets.json", candidate_log)
    _write_json(output / "goal_switches.json", goal_switches)
    _write_json(output / "intervention_outcomes.json", outcomes)
    metrics = {
        "success_rate": float(successes.sum()) / args.num_eval * 100.0,
        "success_count": int(successes.sum()),
        "num_eval": args.num_eval,
        "episode_successes": successes,
        "seeds": init_state.get("seed"),
    }
    return metrics, monitors, {
        "physical_steps_executed_global": step_counter,
        "episode_done": [monitor.episode_done for monitor in monitors],
        "episode_done_steps": [monitor.episode_done_step for monitor in monitors],
        "episode_terminated": [monitor.episode_terminated for monitor in monitors],
        "episode_truncated": [monitor.episode_truncated for monitor in monitors],
        "events": event_log,
        "retrievals": retrieval_log,
        "candidate_sets": candidate_log,
        "goal_switches": goal_switches,
        "intervention_outcomes": outcomes,
        "successes": successes,
    }


def _load_prompt_acceptance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().absolute().resolve()
    replay_root = PROMPT_REPLAY_ROOT.resolve()
    if resolved.name != "acceptance.json" or replay_root not in resolved.parents:
        raise ValueError(
            f"prompt acceptance must be round_N/acceptance.json under {replay_root}"
        )
    round_root = resolved.parent
    round_manifest_path = round_root / "round_manifest.json"
    input_manifest_path = PROMPT_REPLAY_ROOT / "input_manifest.json"
    frozen_inputs_path = PROMPT_REPLAY_ROOT / "frozen_inputs.json"
    candidate_audit_path = PROMPT_REPLAY_ROOT / "candidate_audit.json"
    round_artifact_files = {
        "api_manifest": round_root / "api_manifest.json",
        "llm_calls": round_root / "llm_calls.json",
        "summary": round_root / "summary.json",
        "system_prompt_text": round_root / "system_prompt.txt",
        "decisions": round_root / "decisions.json",
        "acceptance": resolved,
    }
    for required in (
        resolved,
        round_manifest_path,
        input_manifest_path,
        frozen_inputs_path,
        candidate_audit_path,
        *round_artifact_files.values(),
    ):
        if not required.is_file():
            raise FileNotFoundError(f"incomplete prompt acceptance artifact: {required}")
    acceptance = json.loads(resolved.read_text(encoding="utf-8"))
    round_manifest = json.loads(round_manifest_path.read_text(encoding="utf-8"))
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if round_manifest.get("format_version") != "cube_brain_b2_prompt_round_v1":
        raise RuntimeError("prompt acceptance round manifest format drift")
    actual_round_artifacts = {
        key: _file_identity(value) for key, value in round_artifact_files.items()
    }
    expected_round_artifacts = round_manifest.get("round_artifacts")
    if actual_round_artifacts != expected_round_artifacts:
        raise RuntimeError(
            "prompt round artifact identity mismatch: "
            f"expected={expected_round_artifacts}, actual={actual_round_artifacts}, "
            f"location={round_root}"
        )
    expected_checks = {
        "exactly_70_final_calls": True,
        "legal_json_fraction_1_0": True,
        "non_continue_fraction_at_least_0_70": True,
        "all_intervention_coordinates_in_id_box": True,
    }
    actual_gate = {
        "passed": acceptance.get("passed"),
        "checks": acceptance.get("checks"),
        "num_final_calls": acceptance.get("num_final_calls"),
        "legal_json_fraction": acceptance.get("legal_json_fraction"),
        "non_continue_fraction": acceptance.get("non_continue_fraction"),
        "round_manifest_acceptance": round_manifest.get("acceptance"),
    }
    expected_gate = {
        "passed": True,
        "checks": expected_checks,
        "num_final_calls": 70,
        "legal_json_fraction": 1.0,
        "non_continue_fraction": acceptance.get("non_continue_fraction"),
        "round_manifest_acceptance": acceptance,
    }
    if actual_gate != expected_gate or float(acceptance["non_continue_fraction"]) < 0.70:
        raise RuntimeError(
            f"prompt-v2 acceptance gate failed: expected={expected_gate}, actual={actual_gate}"
        )
    prompt_version = int(round_manifest.get("prompt_version", -1))
    if prompt_version not in (1, 2, 3) or int(round_manifest.get("round", -1)) != prompt_version:
        raise RuntimeError("prompt round/version identity mismatch")
    if round_root.name != f"round_{prompt_version}":
        raise RuntimeError("acceptance directory does not match prompt version")
    return {
        "acceptance": acceptance,
        "acceptance_identity": _file_identity(resolved),
        "round_manifest": round_manifest,
        "round_manifest_identity": _file_identity(round_manifest_path),
        "input_manifest": input_manifest,
        "input_manifest_identity": _file_identity(input_manifest_path),
        "frozen_inputs_identity": _file_identity(frozen_inputs_path),
        "candidate_audit_identity": _file_identity(candidate_audit_path),
        "round_artifact_identities": actual_round_artifacts,
        "prompt_version": prompt_version,
    }


def _validate_prompt_code_binding(
    acceptance: Mapping[str, Any], brain_module: Any, replay_module: Any
) -> None:
    round_manifest = acceptance["round_manifest"]
    recorded_code = round_manifest.get("code", {})
    recorded_inputs = round_manifest.get("frozen_input_artifacts", {})
    actual_supervisor = _file_identity(Path(brain_module.__file__))
    actual_replay = _file_identity(Path(replay_module.__file__))
    expected = {
        "supervisor_sha": recorded_code.get("supervisor_v2", {}).get("sha256"),
        "replay_sha": recorded_code.get("replay_tool", {}).get("sha256"),
        "frozen_inputs_sha": recorded_inputs.get("frozen_inputs", {}).get("sha256"),
        "input_manifest_sha": recorded_inputs.get("input_manifest", {}).get("sha256"),
        "candidate_audit_sha": recorded_inputs.get("candidate_audit", {}).get("sha256"),
    }
    actual = {
        "supervisor_sha": actual_supervisor["sha256"],
        "replay_sha": actual_replay["sha256"],
        "frozen_inputs_sha": acceptance["frozen_inputs_identity"]["sha256"],
        "input_manifest_sha": acceptance["input_manifest_identity"]["sha256"],
        "candidate_audit_sha": acceptance["candidate_audit_identity"]["sha256"],
    }
    if actual != expected:
        raise RuntimeError(
            f"prompt acceptance code/input binding mismatch: expected={expected}, actual={actual}"
        )


def _validate_prompt_input_protocol(
    acceptance: Mapping[str, Any], selection: Mapping[str, Any], index_path: Path
) -> None:
    manifest = acceptance["input_manifest"]
    b1_brain_root = OUTPUT_ROOT / "brain_offset100"
    expected_algorithm = {
        "STALLED": [
            "progress_one_third",
            "progress_two_thirds",
            "retreat_five_cm",
        ],
        "STALLED_retreat": {
            "desired_displacement_m": 0.05,
            "minimum_actual_current_displacement_m": 0.045,
            "preferred_maximum_actual_current_displacement_m": 0.08,
            "desired_reverse_axis": "current-to-final planar xy unit vector",
            "preferred_minimum_reverse_projection_m": 0.04,
            "directional_fallback": (
                "if strict reverse is absent, use nearest candidate that still satisfies "
                "the displacement and farther-from-final qualifications; record fallback"
            ),
            "must_increase_final_target_distance": True,
            "nearest_metric": "physical block xyz meters then anchor row",
            "unqualified_fallback_forbidden": True,
        },
        "DROPPED_filter": {
            "ee_block_max_m": 0.03,
            "contact_min": 0.5,
            "opening_max": 0.60,
            "selection_requires_block_and_ee_inside_frozen_id_box": True,
            "fixed_50_excluded_in_pool": True,
            "qualified_anchor_count": EXPECTED_RECOVERY_ANCHORS,
            "qualified_source_episode_count": EXPECTED_RECOVERY_EPISODES,
        },
        "STALLED_filter": "retrieved block position inside frozen ID box",
        "stable_order": "intent order then exact nearest stable (distance,row)",
        "distinct_source_episode": "required; fail closed if unavailable",
        "exclusions": "all fixed 50 episodes plus current episode",
    }
    actual = {
        "format": manifest.get("format_version"),
        "num_inputs": manifest.get("num_inputs"),
        "fixed_episodes": manifest.get("fixed_episodes"),
        "candidate_algorithm": manifest.get("candidate_algorithm"),
        "rows_distinct": manifest.get("all_candidate_rows_distinct"),
        "episodes_distinct": manifest.get("all_candidate_episodes_distinct"),
        "retreat_directional_fallback_count": manifest.get("fallback_count"),
        "retreat_candidate_breakdown": manifest.get(
            "retreat_candidate_breakdown"
        ),
        "source_calls_sha": manifest.get("source_calls", {}).get("sha256"),
        "source_run_sha": manifest.get("source_run_manifest", {}).get("sha256"),
        "index_shas": {
            name: value.get("sha256")
            for name, value in manifest.get("index", {}).items()
        },
    }
    expected = {
        "format": "cube_brain_b2_frozen_inputs_v1",
        "num_inputs": 70,
        "fixed_episodes": b1._jsonable(selection["episodes"]),
        "candidate_algorithm": expected_algorithm,
        "rows_distinct": True,
        "episodes_distinct": True,
        "retreat_directional_fallback_count": 10,
        "retreat_candidate_breakdown": {
            "detour:no_qualified_detour_within_8cm": 5,
            "detour:no_strict_reverse_anchor": 5,
            "retreat": 60,
        },
        "source_calls_sha": _file_identity(
            b1_brain_root / "llm_calls.json"
        )["sha256"],
        "source_run_sha": _file_identity(
            b1_brain_root / "run_manifest.json"
        )["sha256"],
        "index_shas": {
            name: _file_identity(index_path / name)["sha256"]
            for name in (
                "metadata.json",
                "anchor_rows.npy",
                "anchor_episodes.npy",
                "anchor_features_z.npy",
                "stats.npz",
            )
        },
    }
    if actual != expected:
        raise RuntimeError(
            f"accepted prompt input protocol drift: expected={expected}, actual={actual}"
        )


def _build_supervisor(
    brain_module: Any,
    output: Path,
    prompt_acceptance: Mapping[str, Any],
) -> Any:
    prompt_version = int(prompt_acceptance["prompt_version"])
    config = brain_module.B2Config(prompt_version=prompt_version, initial_tokens=0)
    supervisor = brain_module.BrainSupervisorV2(config, output)
    manifest = supervisor.manifest()
    round_manifest = prompt_acceptance["round_manifest"]
    actual = {
        "protocol": manifest.get("protocol"),
        "prompt_version": manifest.get("prompt_version"),
        "prompt_sha256": manifest.get("prompt_sha256"),
        "provider": manifest.get("provider"),
        "model": manifest.get("requested_model"),
        "endpoint": manifest.get("endpoint"),
        "thinking": manifest.get("thinking"),
        "temperature": manifest.get("temperature"),
        "max_calls": manifest.get("max_logical_calls_per_episode"),
        "max_tokens": manifest.get("max_total_tokens_all_rounds"),
        "id_box": manifest.get("id_box"),
        "real_frame_landing": manifest.get("real_frame_landing"),
    }
    expected = {
        "protocol": "cube_brain_b2_prompt_replay_v1",
        "prompt_version": prompt_version,
        "prompt_sha256": round_manifest.get("prompt_sha256"),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "thinking": {"type": "disabled"},
        "temperature": 0.1,
        "max_calls": 5,
        "max_tokens": 1_000_000,
        "id_box": {
            "x": list(b1.ID_X),
            "y": list(b1.ID_Y),
            "z": list(b1.ID_Z),
        },
        "real_frame_landing": {
            "exact_candidate": "use selected anchor_row",
            "fine_tuned": "reretrieve a real frame; never synthesize goal pixels",
            "max_position_adjustment_l2_m": 0.03,
            "max_yaw_adjustment_circular_rad": float(np.pi / 12.0),
            "guard_order": (
                "candidate-relative clamp, ID-box clamp, real-frame reretrieval"
            ),
        },
    }
    if actual != expected:
        raise RuntimeError(
            f"B2 online supervisor differs from accepted prompt: expected={expected}, actual={actual}"
        )
    return supervisor


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode not in MODES:
        raise ValueError(args.mode)
    if args.goal_offset_steps != OFFSET or args.eval_budget != BUDGET:
        raise ValueError("B2 is frozen at offset=100, budget=200")
    if args.seed != SEED or args.num_eval not in (2, 50):
        raise ValueError("B2 is frozen at seed=42 and num_eval=2 or 50")
    if args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError("formal B2 run requires --authorize-formal")
    if args.num_eval == 50 and args.self_test:
        raise ValueError("B2 self-test is smoke-only; formal output cannot be synthesized")
    if args.force_smoke_trigger_step is not None:
        if args.num_eval != 2 or args.force_smoke_trigger_step != 5:
            raise ValueError("forced B2 smoke trigger is num_eval=2, physical step 5 only")
    if args.force_smoke_goal_switch and (
        args.mode != "brainv2"
        or args.num_eval != 2
        or args.force_smoke_trigger_step is None
    ):
        raise ValueError(
            "--force-smoke-goal-switch requires brainv2 num_eval=2 and forced trigger"
        )
    if args.force_smoke_both_events and args.num_eval != 2:
        raise ValueError("--force-smoke-both-events is available only for num_eval=2 smoke")
    if args.force_smoke_both_events and args.force_smoke_goal_switch:
        raise ValueError(
            "both-event smoke requires real SUBGOAL/RECOVER decisions; synthetic goal switch is forbidden"
        )
    if args.num_eval == 50 and (
        args.force_smoke_trigger_step is not None
        or args.force_smoke_goal_switch
        or args.force_smoke_both_events
    ):
        raise ValueError("formal B2 rejects forced smoke controls")
    if args.mode == "brainv2" and args.prompt_acceptance is None:
        raise ValueError("brainv2 requires --prompt-acceptance")
    if args.mode == "rule" and args.prompt_acceptance is not None:
        raise ValueError("rule mode does not consume a prompt acceptance artifact")
    for path, label in (
        (args.dataset, "dataset"),
        (args.fixed_manifest, "fixed manifest"),
        (args.index / "metadata.json", "memory index"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    b1._configure_storage()
    selection = b1.select_longhorizon_rows(
        args.dataset, args.fixed_manifest, OFFSET, SEED
    )
    source_formal_env_indices = (
        np.arange(50, dtype=np.int64)
        if args.num_eval == 50
        else np.asarray(SMOKE_FORMAL_ENV_INDICES, dtype=np.int64)
    )
    rows = np.asarray(selection["rows"], dtype=np.int64)[source_formal_env_indices]
    fixed_episodes = np.asarray(selection["episodes"], dtype=np.int64)
    checkpoint = b1.trust_common.frozen_masked_checkpoint_contract()
    baseline_pairing = b1._validate_baseline_pairing(
        OFFSET, selection, checkpoint, args.index
    )
    baseline_results = json.loads(
        (Path(baseline_pairing["root"]) / "results.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_successes = np.asarray(
        baseline_results["metrics"]["episode_successes"], dtype=bool
    )
    if baseline_successes.shape != (50,):
        raise RuntimeError(
            "paired baseline success vector shape drift: "
            f"{baseline_successes.shape}"
        )
    selected_baseline_successes = baseline_successes[source_formal_env_indices]
    if args.num_eval == 2 and selected_baseline_successes.any():
        raise RuntimeError(
            "B2 forced smoke rows must be long-lived baseline failures: "
            f"indices={source_formal_env_indices.tolist()}, "
            f"successes={selected_baseline_successes.tolist()}"
        )
    prompt_acceptance = None
    brain_module = replay_module = None
    if args.mode == "brainv2":
        prompt_acceptance = _load_prompt_acceptance(args.prompt_acceptance)
        import brain_supervisor_v2 as brain_module  # type: ignore[no-redef]
        import replay_brain_b2_prompts as replay_module  # type: ignore[no-redef]

        _validate_prompt_code_binding(prompt_acceptance, brain_module, replay_module)
        _validate_prompt_input_protocol(prompt_acceptance, selection, args.index)
    requested_output = args.output or _default_output(args.mode, args.num_eval)
    _validate_output(requested_output, args.mode, args.num_eval)
    output = b1._safe_output(requested_output, args.overwrite)

    if args.self_test:
        _write_json(
            output / "self_test.json",
            {
                "format_version": "cube_brain_b2_self_test_v1",
                "mode": args.mode,
                "selection": selection,
                "evaluated_rows": rows,
                "source_formal_env_indices": source_formal_env_indices,
                "source_baseline_successes": selected_baseline_successes,
                "checkpoint": checkpoint,
                "baseline_pairing": baseline_pairing,
                "prompt_acceptance": prompt_acceptance,
            },
        )
        print(output)
        return 0

    import hdf5plugin  # noqa: F401
    import h5py
    import stable_worldmodel as swm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("B2 online evaluation requires CUDA")
    index = B2MemoryIndex(args.index, args.dataset, fixed_episodes)
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    selected = dataset.get_row_data(rows)
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = np.asarray(selected[ep_key], dtype=np.int64)
    starts = np.asarray(selected["step_idx"], dtype=np.int64)
    if not np.array_equal(episodes, fixed_episodes[source_formal_env_indices]):
        raise RuntimeError("B2 evaluated episodes differ from frozen held-out order")
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        initial_query_features = np.concatenate(
            [
                b1.memory.feature_chunk(h5, int(row), int(row) + 1)
                for row in rows
            ],
            axis=0,
        )

    model = swm.wm.utils.load_pretrained(
        str(b1.trust_common.MASKED_CHECKPOINT), cache_dir=str(PROJECT_ROOT)
    )
    model = model.to(args.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    scaler = b1.legacy._standard_scaler(index)
    recorder = b1.ood.PlanningCostRecorder(args.num_eval)
    proxy = b1.trust.TrustRegionCostProxy(model, "t2")
    solver_cls = b1.trust.make_trust_region_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=proxy,
        batch_size=1,
        num_samples=b1.trust_common.NUM_SAMPLES,
        var_scale=b1.trust_common.PROTOCOL_SPECS["t2"]["var_scale"],
        n_steps=b1.trust_common.N_STEPS,
        topk=b1.trust_common.TOPK,
        device=args.device,
        seed=SEED,
        callbacks=[recorder],
        selector="mean",
        recorder=recorder,
        trust_protocol="t2",
    )
    config = swm.PlanConfig(
        horizon=b1.trust_common.HORIZON,
        receding_horizon=b1.trust_common.HORIZON,
        action_block=b1.trust_common.ACTION_BLOCK,
    )
    policy_cls = _make_b2_observed_policy(
        b1.trust.make_trust_policy(swm.policy.WorldModelPolicy)
    )
    policy = policy_cls(
        solver=solver,
        config=config,
        process={"action": scaler},
        transform={
            "pixels": b1.ood._image_transform(224),
            "goal": b1.ood._image_transform(224),
        },
        memory_index=index,
        cost_proxy=proxy,
        cost_recorder=recorder,
        eval_episodes=episodes,
        eval_rows=rows,
        initial_query_features=initial_query_features,
        protocol="t2",
    )
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=args.num_eval,
        max_episode_steps=2 * BUDGET,
        image_shape=(224, 224),
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
    )
    world.set_policy(policy)
    init_state, goal_state, dataset_videos, episodes, starts = b1._prepare_world_inputs(
        world, dataset, rows, OFFSET
    )

    supervisor = candidate_builder = None
    if args.mode == "brainv2":
        assert (
            prompt_acceptance is not None
            and brain_module is not None
            and replay_module is not None
        )
        supervisor = _build_supervisor(brain_module, output, prompt_acceptance)
        candidate_builder = replay_module.RealFrameCandidateBuilder(
            args.index, args.dataset, fixed_episodes
        )

    run_manifest = {
        "format_version": "cube_brain_b2_run_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "mode": args.mode,
        "goal_offset_steps": OFFSET,
        "eval_budget": BUDGET,
        "num_eval": args.num_eval,
        "seed": SEED,
        "output_path_interpretation": (
            "B2 shorthand is stored under outputs/eval/cube/longhorizon/"
            f"{args.mode}_offset100 for direct pairing with the B1 baseline"
        ),
        "selection": selection,
        "evaluated_rows": rows,
        "source_formal_env_indices": source_formal_env_indices,
        "source_formal_env_indices": source_formal_env_indices,
        "source_baseline_successes": selected_baseline_successes,
        "evaluated_episodes": episodes,
        "evaluated_starts": starts,
        "checkpoint": checkpoint,
        "t2": b1._frozen_t2_manifest(),
        "baseline_pairing": baseline_pairing,
        "retrieval": {
            "index": _file_identity(args.index / "metadata.json"),
            "global_excluded_episodes": fixed_episodes,
            "global_excluded_episodes_sha256_int64": index.fixed_episodes_sha256,
            "also_exclude_current_episode": True,
            "ten_unique_t2_source_episodes": True,
            "recovery_filter": index.recovery_filter_manifest,
        },
        "rule": {
            "DROPPED": (
                "closest stable contact-qualified real frame by block distance, "
                "then ee distance to current block, then row; three source episodes, use first"
            ),
            "STALLED": (
                "xyz midpoint(current block, final target); yaw excluded from query; "
                "use retrieved frame yaw"
            ),
        },
        "shared_execution_guard": {
            "active_timeout_physical_steps": ACTIVE_GOAL_TIMEOUT_STEPS,
            "deadline_semantics": (
                "trigger S installs goal; S+1..S+25 execute; callback S+25 checks "
                "achievement before timeout; switch final then suppress same-step retrigger"
            ),
            "active_STALLED": "logged and suppressed",
            "active_DROPPED": (
                "may replace active goal only when shared five-step cooldown and "
                "five-intervention/call budget allow"
            ),
            "no_future_action_closure": (
                "terminal callbacks and callback step 200 close active intervention "
                "accounting without a fictitious final-goal switch/replan; new step-200 "
                "interventions are suppressed, and evaluation end requires no pending "
                "policy replan requests"
            ),
            "sticky_episode_done": (
                "first terminated or truncated callback is persisted per monitor; "
                "later EnvPool masked callbacks cannot re-enter trigger, intervention, "
                "retrieval, LLM, history, or goal-switch logic"
            ),
            "smoke_source": (
                None
                if args.num_eval == 50
                else {
                    "formal_env_indices": list(SMOKE_FORMAL_ENV_INDICES),
                    "rows": rows,
                    "paired_baseline_successes": selected_baseline_successes,
                    "paired_baseline_results": baseline_pairing["results"],
                    "reason": (
                        "fixed distinct long-lived baseline failures so both forced "
                        "step-5 events exercise live policy slots"
                    ),
                }
            ),
            "replan_observation": (
                "Every dynamic real-frame goal registers a one-shot request on the "
                "executing B2ObservedPolicy. At next get_action the policy clears the "
                "25-action receding buffer at the policy boundary, verifies the exact "
                "jepa-consumed goal history slice 0 and every other history slice against "
                "the requested HDF5 pixels SHA, then requires exactly one new TrustPolicy "
                "cycle, one recorder cycle, and a finite returned action. Callback S+1 "
                "binds that proof to the exact switch before completion/timeout/switch."
            ),
            "force_smoke_trigger_step": args.force_smoke_trigger_step,
            "force_smoke_goal_switch": args.force_smoke_goal_switch,
            "force_smoke_both_events": args.force_smoke_both_events,
        },
        "brainv2": (
            None
            if supervisor is None
            else {
                "api": supervisor.manifest(),
                "prompt_acceptance": prompt_acceptance,
                "candidate_builder": _file_identity(Path(replay_module.__file__)),
                "online_landing": {
                    "resolver": "RealFrameCandidateBuilder.resolve_landing",
                    "exact_candidate_preserves_anchor_row": True,
                    "fine_tuned_position_l2_max_m": 0.03,
                    "fine_tuned_yaw_circular_max_rad": float(np.pi / 12.0),
                    "guard_order": (
                        "candidate-relative clamp, ID-box clamp, real-frame reretrieval"
                    ),
                    "goal_pixels": "literal HDF5 pixels at resolved anchor_row",
                    "global_fixed50_and_current_episode_excluded": True,
                },
            }
        ),
        "helper_provenance": {
            "eval_brain_b1": _file_identity(Path(b1.__file__)),
            "this_evaluator": _file_identity(Path(__file__)),
            **(
                {
                    "brain_supervisor_v2": _file_identity(Path(brain_module.__file__)),
                    "replay_brain_b2_prompts": _file_identity(
                        Path(replay_module.__file__)
                    ),
                }
                if brain_module is not None and replay_module is not None
                else {}
            ),
        },
    }
    _write_json(output / "run_manifest.json", run_manifest)

    started = time.time()
    try:
        metrics, monitors, runtime = _run_evaluation(
            args,
            output,
            world,
            policy,
            recorder,
            dataset,
            index,
            selection,
            rows,
            source_formal_env_indices,
            episodes,
            starts,
            init_state,
            goal_state,
            dataset_videos,
            supervisor,
            brain_module,
            candidate_builder,
        )
    finally:
        world.close()
    elapsed = time.time() - started

    trace = b1.trust._save_trace(output, proxy)
    cost_history = b1.ood._save_cost_history(
        output, recorder, rows, episodes, starts, "mean"
    )
    supervisor_summary = None
    if supervisor is not None:
        supervisor_summary = supervisor.summary()
        _write_json(output / "summary.json", supervisor_summary)
        if not (output / "llm_calls.json").exists():
            _write_json(output / "llm_calls.json", [])
    successes = np.asarray(runtime.pop("successes"), dtype=bool)
    behavior = _behavior_summary(monitors, successes, supervisor_summary)
    artifacts = {
        name: _file_identity(output / filename)
        for name, filename in (
            ("events", "events.json"),
            ("subgoal_retrieval", "subgoal_retrieval.json"),
            ("candidate_sets", "candidate_sets.json"),
            ("goal_switches", "goal_switches.json"),
            ("intervention_outcomes", "intervention_outcomes.json"),
        )
    }
    if supervisor is not None:
        artifacts.update(
            {
                "llm_calls": _file_identity(output / "llm_calls.json"),
                "api_manifest": _file_identity(output / "api_manifest.json"),
                "api_summary": _file_identity(output / "summary.json"),
            }
        )
    results_payload = {
        "format_version": "cube_brain_b2_evaluation_v1",
        "protocol": {
            "mode": args.mode,
            "goal_offset_steps": OFFSET,
            "eval_budget": BUDGET,
            "seed": SEED,
            "checkpoint": checkpoint,
            "t2": run_manifest["t2"],
            "baseline_pairing": baseline_pairing,
            "prompt_acceptance": prompt_acceptance,
        },
        "selection": selection,
        "evaluated_rows": rows,
        "metrics": metrics,
        "behavior": behavior,
        "elapsed_seconds": elapsed,
        "trace": trace,
        "cost_history": cost_history,
        "runtime": runtime,
        "artifacts": artifacts,
    }
    _write_json(output / "results.json", results_payload)
    success_text = ", ".join(
        "True" if value else "False" for value in metrics["episode_successes"]
    )
    (output / "results.txt").write_text(
        f"mode: {args.mode}\n"
        f"goal_offset_steps: {OFFSET}\n"
        f"eval_budget: {BUDGET}\n"
        f"success_rate: {metrics['success_rate']:.6f}\n"
        f"success_count: {metrics['success_count']}/{metrics['num_eval']}\n"
        f"episode_successes: [{success_text}]\n"
        f"elapsed_seconds: {elapsed:.6f}\n",
        encoding="utf-8",
    )
    run_manifest["status"] = "complete"
    run_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["results"] = _file_identity(output / "results.json")
    run_manifest["behavior"] = behavior
    _write_json(output / "run_manifest.json", run_manifest)
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--goal-offset-steps", type=int, default=OFFSET)
    parser.add_argument("--eval-budget", type=int, default=BUDGET)
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset", type=Path, default=b1.DATASET)
    parser.add_argument("--fixed-manifest", type=Path, default=b1.FIXED_MANIFEST)
    parser.add_argument("--index", type=Path, default=b1.MEMORY_INDEX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt-acceptance", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    parser.add_argument("--force-smoke-trigger-step", type=int)
    parser.add_argument("--force-smoke-goal-switch", action="store_true")
    parser.add_argument("--force-smoke-both-events", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
