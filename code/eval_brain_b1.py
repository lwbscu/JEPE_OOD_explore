#!/usr/bin/env python3
"""B1 long-horizon Cube evaluation with an event-driven text supervisor.

This is an independent evaluator.  It imports the frozen T2 implementation but
does not modify the legacy evaluators, site-packages, model, or memory index.
The physical MuJoCo target is always the final HDF5 ``row + offset`` pose.  A
brain decision may only replace the goal image consumed by the world model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_cube_memory_index as memory  # noqa: E402
import cube_trust_region_common as trust_common  # noqa: E402
import eval_memory_seed as legacy  # noqa: E402
import eval_ood_color as ood  # noqa: E402
import eval_trust_region as trust  # noqa: E402


PROJECT_ROOT = HERE.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/eval/cube/longhorizon"
DATASET = PROJECT_ROOT / "datasets/ogbench/cube_single_expert.h5"
FIXED_MANIFEST = PROJECT_ROOT / "outputs/audit/cube_cem_manifest.json"
MEMORY_INDEX = PROJECT_ROOT / "outputs/memory_index/cube_expert_v1"
TMP_ROOT = PROJECT_ROOT.parent / "tmp"

FORMAL_SEED = 42
FORMAL_NUM_EVAL = 50
EPISODE_LENGTH = 201
OFFSETS = (50, 75, 100)
MODES = ("baseline", "brain")
CONTACT_THRESHOLD = 0.5
SUBGOAL_TOLERANCE_M = 0.04
DROP_Z_M = 0.03
STALL_WINDOW_STEPS = 8
STALL_PROGRESS_M = 0.01
COST_NONDECREASE_EPS = 1e-6
MIN_CALL_INTERVAL_STEPS = 5
MAX_CALLS_PER_EPISODE = 5
ID_X = (0.244, 0.611)
ID_Y = (-0.356, 0.355)
ID_Z = (0.0, 0.35)

EXPECTED_SELECTION_SHA256 = {
    50: "cd5edd27d766d737665b85465a2644781e49d9804d34df76eab9875b33c5bf71",
    75: "954ccf1ee9e6d8bee4ac224d1d0930f81ec4f2bf4f2ba9dda0c15883a82c384e",
    100: "0cd9a6fd177d40f62c5d06d5632454f9cad4aeef357158dd393777db505a78ce",
}
EXPECTED_EPISODES_SHA256 = (
    "f99be3d88d6af9f78bccf7f726541b30e65ee6c02f56ab9fdfbf2d11338186a4"
)
EXPECTED_STARTS_SHA256 = {
    50: "6765c93b33147b5aeeb7d5c2ea95d4cca7268a16885cf630cc73596cc3e2e15e",
    75: "187ada63b157df680e3cacf18a285de6eb007f336581141b1443525a19b6db0e",
    100: "4a2bba7bb976fe3d8e112833a6517482b49101efdaebe9ad256ae181b4befaa2",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(resolved),
    }


def _configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(PROJECT_ROOT),
        "HF_HOME": str(PROJECT_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(PROJECT_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(PROJECT_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        os.environ[key] = value
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_output(path: Path, overwrite: bool) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise ValueError(f"refusing symlink output: {lexical}")
    resolved = lexical.resolve()
    root = OUTPUT_ROOT.resolve()
    data_disk = PROJECT_ROOT.parent.resolve()
    if data_disk not in resolved.parents:
        raise ValueError(f"output must be on /root/autodl-tmp: {resolved}")
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a concrete child of {root}: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output exists but is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is nonempty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _default_output(mode: str, offset: int, num_eval: int) -> Path:
    name = f"{mode}_offset{offset}"
    return OUTPUT_ROOT / name if num_eval == 50 else OUTPUT_ROOT / "smoke" / name


def _validate_output_contract(path: Path, mode: str, offset: int, num_eval: int) -> None:
    resolved = path.expanduser().absolute().resolve()
    if num_eval == FORMAL_NUM_EVAL:
        expected = _default_output(mode, offset, num_eval).resolve()
        if resolved != expected:
            raise ValueError(
                f"formal B1 output is frozen: expected={expected}, actual={resolved}"
            )
        return
    smoke_root = (OUTPUT_ROOT / "smoke").resolve()
    if resolved == smoke_root or smoke_root not in resolved.parents:
        raise ValueError(f"smoke output must be a concrete child of {smoke_root}: {resolved}")


def _fixed_heldout_episodes(
    dataset_path: Path, manifest_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import hdf5plugin  # noqa: F401
    import h5py

    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_rows = np.asarray(frozen.get("formal_rows"), dtype=np.int64)
    if old_rows.shape != (FORMAL_NUM_EVAL,):
        raise RuntimeError(f"fixed formal row shape mismatch: {old_rows.shape}")
    with h5py.File(dataset_path, "r", swmr=True) as h5:
        if int(h5["step_idx"].shape[0]) != 2_010_000:
            raise RuntimeError("unexpected frozen Cube frame count")
        episodes = np.asarray(h5["ep_idx"][old_rows], dtype=np.int64)
    if len(np.unique(episodes)) != FORMAL_NUM_EVAL:
        raise RuntimeError("fixed held-out manifest does not contain 50 unique episodes")
    actual_sha = _array_sha256(episodes)
    if actual_sha != EXPECTED_EPISODES_SHA256:
        raise RuntimeError(
            "fixed held-out episode identity changed: "
            f"expected={EXPECTED_EPISODES_SHA256}, actual={actual_sha}"
        )
    return old_rows, episodes, frozen


def select_longhorizon_rows(
    dataset_path: Path,
    manifest_path: Path,
    offset: int,
    seed: int = FORMAL_SEED,
) -> dict[str, Any]:
    """Resample starts inside the original 50 held-out episodes.

    Sampling new episodes from the training population would no longer be a
    held-out evaluation.  We therefore preserve the original held-out episode
    order and independently sample one valid start inside each episode.
    """

    if offset not in OFFSETS or seed != FORMAL_SEED:
        raise ValueError(f"frozen offset/seed mismatch: offset={offset}, seed={seed}")
    old_rows, episodes, frozen = _fixed_heldout_episodes(dataset_path, manifest_path)
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, EPISODE_LENGTH - offset, size=FORMAL_NUM_EVAL, dtype=np.int64
    )
    rows = episodes * EPISODE_LENGTH + starts
    goal_rows = rows + offset
    if len(np.unique(episodes)) != FORMAL_NUM_EVAL:
        raise RuntimeError("long-horizon formal episodes must remain unique")

    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(dataset_path, "r", swmr=True) as h5:
        row_episodes = np.asarray(h5["ep_idx"][rows], dtype=np.int64)
        row_steps = np.asarray(h5["step_idx"][rows], dtype=np.int64)
        goal_episodes = np.asarray(h5["ep_idx"][goal_rows], dtype=np.int64)
        goal_steps = np.asarray(h5["step_idx"][goal_rows], dtype=np.int64)
    if not np.array_equal(row_episodes, episodes):
        raise RuntimeError("row=episode*201+start invariant changed")
    if not np.array_equal(row_steps, starts):
        raise RuntimeError("selected HDF5 step_idx differs from sampled start")
    if not np.array_equal(goal_episodes, episodes):
        raise RuntimeError("one or more long-horizon goals cross an episode boundary")
    if not np.array_equal(goal_steps - starts, np.full(50, offset)):
        raise RuntimeError("one or more goals are not exactly row+offset")

    identities = {
        "rows_sha256_int64": _array_sha256(rows),
        "episodes_sha256_int64": _array_sha256(episodes),
        "starts_sha256_int64": _array_sha256(starts),
        "goal_rows_sha256_int64": _array_sha256(goal_rows),
    }
    expected = {
        "rows_sha256_int64": EXPECTED_SELECTION_SHA256[offset],
        "episodes_sha256_int64": EXPECTED_EPISODES_SHA256,
        "starts_sha256_int64": EXPECTED_STARTS_SHA256[offset],
    }
    for key, expected_sha in expected.items():
        if identities[key] != expected_sha:
            raise RuntimeError(
                f"long-horizon selection drift for {key}: "
                f"expected={expected_sha}, actual={identities[key]}"
            )
    return {
        "format_version": "cube_brain_b1_selection_v2",
        "reason": (
            "preserve the original 50 held-out episodes; sampling episodes from "
            "the training population would invalidate held-out evaluation"
        ),
        "algorithm": (
            "episodes=old cube_cem_manifest formal-row episodes in original order; "
            "starts=default_rng(42).integers(0,201-offset,size=50,dtype=int64); "
            "rows=episodes*201+starts"
        ),
        "seed": seed,
        "offset": offset,
        "episode_length": EPISODE_LENGTH,
        "num_formal": FORMAL_NUM_EVAL,
        "old_formal_rows": old_rows,
        "episodes": episodes,
        "starts": starts,
        "rows": rows,
        "goal_rows": goal_rows,
        "same_as_old_row_envs": np.flatnonzero(rows == old_rows),
        "identities": identities,
        "fixed_manifest": _file_identity(manifest_path),
        "fixed_manifest_selection": frozen.get("selection"),
    }


class HeldoutMemoryIndex(memory.CubeMemoryIndex):
    """Frozen index query with all 50 held-out episodes excluded globally."""

    def __init__(
        self,
        root: Path,
        dataset: Path,
        fixed_episodes: np.ndarray,
    ) -> None:
        super().__init__(root, dataset)
        values = np.asarray(fixed_episodes, dtype=np.int64)
        if values.shape != (50,) or len(np.unique(values)) != 50:
            raise ValueError("global held-out exclusion must contain 50 unique episodes")
        self.fixed_episodes = frozenset(int(value) for value in values)
        self.fixed_episodes_sha256 = _array_sha256(values)
        self._subgoal_trees: dict[str, Any] = {}
        allowed = ~np.isin(np.asarray(self.episodes), values)
        raw_z = (
            np.asarray(self.features[:, 2], dtype=np.float64) * self.feature_std[2]
            + self.feature_mean[2]
        )
        self.id_z_min = float(np.min(raw_z[allowed]))
        if not np.isfinite(self.id_z_min) or self.id_z_min > 0.35:
            raise RuntimeError(f"invalid ID z lower bound: {self.id_z_min}")

    def _forbidden(self, exclude_episode: int) -> set[int]:
        return set(self.fixed_episodes) | {int(exclude_episode)}

    def retrieve(
        self,
        raw_feature: np.ndarray,
        exclude_episode: int,
        count: int = 10,
    ) -> dict[str, np.ndarray]:
        raw_query = np.asarray(raw_feature, dtype=np.float64).reshape(1, -1)
        query = self.normalize(raw_query)[0]
        forbidden = self._forbidden(exclude_episode)
        k = 64
        selected: list[tuple[float, int, int, int]] = []
        while True:
            distances, indices = self.tree.query(
                query, k=min(k, len(self.rows)), eps=0.0, workers=1
            )
            candidates = sorted(
                (
                    float(distance),
                    int(self.rows[int(index)]),
                    int(self.episodes[int(index)]),
                    int(index),
                )
                for distance, index in zip(
                    np.atleast_1d(distances), np.atleast_1d(indices)
                )
                if int(self.episodes[int(index)]) not in forbidden
            )
            selected = []
            seen: set[int] = set()
            for item in candidates:
                if item[2] in seen:
                    continue
                seen.add(item[2])
                selected.append(item)
                if len(selected) == count:
                    break
            if len(selected) == count:
                cutoff = float(selected[-1][0])
                ball = self.tree.query_ball_point(
                    query, r=np.nextafter(cutoff, np.inf), eps=0.0, workers=1
                )
                complete = sorted(
                    (
                        float(np.linalg.norm(self.features[int(index)] - query)),
                        int(self.rows[int(index)]),
                        int(self.episodes[int(index)]),
                        int(index),
                    )
                    for index in ball
                    if int(self.episodes[int(index)]) not in forbidden
                )
                selected = []
                seen = set()
                for item in complete:
                    if item[2] in seen:
                        continue
                    seen.add(item[2])
                    selected.append(item)
                    if len(selected) == count:
                        break
                if len(selected) == count:
                    break
            if k >= len(self.rows):
                raise RuntimeError(
                    f"insufficient allowed unique memory episodes: {len(selected)}/{count}"
                )
            k = min(k * 2, len(self.rows))
        selected.sort(key=lambda item: (item[0], item[1]))
        result = {
            "distances": np.asarray([x[0] for x in selected], dtype=np.float64),
            "rows": np.asarray([x[1] for x in selected], dtype=np.int64),
            "episodes": np.asarray([x[2] for x in selected], dtype=np.int64),
            "steps": np.asarray([x[1] % EPISODE_LENGTH for x in selected], dtype=np.int64),
            "anchor_indices": np.asarray([x[3] for x in selected], dtype=np.int64),
        }
        if len(np.unique(result["episodes"])) != count:
            raise RuntimeError("memory retrieval episode uniqueness contract failed")
        if any(int(ep) in forbidden for ep in result["episodes"]):
            raise RuntimeError("memory retrieval leaked a held-out/current episode")
        return result

    def _subgoal_tree(self, kind: str) -> tuple[Any, np.ndarray]:
        from scipy.spatial import cKDTree

        dimensions = {
            "block": np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
            "ee": np.asarray([5, 6, 7], dtype=np.int64),
        }[kind]
        if kind not in self._subgoal_trees:
            points = np.ascontiguousarray(self.features[:, dimensions], dtype=np.float64)
            self._subgoal_trees[kind] = cKDTree(
                points, copy_data=False, balanced_tree=True, compact_nodes=True
            )
        return self._subgoal_trees[kind], dimensions

    def nearest_goal_frame(
        self,
        kind: str,
        position: np.ndarray,
        yaw: float | None,
        exclude_episode: int,
    ) -> dict[str, Any]:
        tree, dimensions = self._subgoal_tree(kind)
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if kind == "block":
            if yaw is None or not np.isfinite(yaw):
                raise ValueError("block subgoal requires finite yaw")
            raw = np.concatenate((position, [np.sin(yaw), np.cos(yaw)]))
        elif kind == "ee":
            raw = position
        else:
            raise ValueError(kind)
        query = (raw - self.feature_mean[dimensions]) / self.feature_std[dimensions]
        forbidden = self._forbidden(exclude_episode)
        k = 64
        while True:
            distances, indices = tree.query(
                query, k=min(k, len(self.rows)), eps=0.0, workers=1
            )
            candidates = sorted(
                (
                    float(distance),
                    int(self.rows[int(index)]),
                    int(self.episodes[int(index)]),
                    int(index),
                )
                for distance, index in zip(
                    np.atleast_1d(distances), np.atleast_1d(indices)
                )
                if int(self.episodes[int(index)]) not in forbidden
            )
            if candidates:
                cutoff = candidates[0][0]
                ball = tree.query_ball_point(
                    query, r=np.nextafter(cutoff, np.inf), eps=0.0, workers=1
                )
                tied = sorted(
                    (
                        float(np.linalg.norm(self.features[int(index), dimensions] - query)),
                        int(self.rows[int(index)]),
                        int(self.episodes[int(index)]),
                        int(index),
                    )
                    for index in ball
                    if int(self.episodes[int(index)]) not in forbidden
                )
                if tied:
                    best = tied[0]
                    return {
                        "kind": kind,
                        "distance_z": best[0],
                        "row": best[1],
                        "episode": best[2],
                        "step": best[1] % EPISODE_LENGTH,
                        "anchor_index": best[3],
                        "query_raw": raw,
                        "query_z": query,
                        "dimensions": dimensions,
                        "tie_count": len(tied),
                    }
            if k >= len(self.rows):
                raise RuntimeError("no allowed real frame for subgoal")
            k = min(k * 2, len(self.rows))


def _last(info: Mapping[str, Any], env_idx: int, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name not in info:
            continue
        value = np.asarray(info[name][env_idx])
        while value.ndim > 1:
            value = value[-1]
        return np.asarray(value)
    raise KeyError(f"missing live-state keys: {names}")


def _live_state(info: Mapping[str, Any], env_idx: int) -> dict[str, Any]:
    contact_raw = float(
        _last(
            info,
            env_idx,
            ("proprio/gripper_contact", "proprio_gripper_contact"),
        ).reshape(-1)[-1]
    )
    return {
        "block_pos": _last(
            info,
            env_idx,
            ("privileged/block_0_pos", "privileged_block_0_pos"),
        ).astype(np.float64).reshape(3),
        "block_yaw": float(
            _last(
                info,
                env_idx,
                ("privileged/block_0_yaw", "privileged_block_0_yaw"),
            ).reshape(-1)[-1]
        ),
        "ee_pos": _last(
            info,
            env_idx,
            ("proprio/effector_pos", "proprio_effector_pos"),
        ).astype(np.float64).reshape(3),
        "gripper_opening": float(
            _last(
                info,
                env_idx,
                ("proprio/gripper_opening", "proprio_gripper_opening"),
            ).reshape(-1)[-1]
        ),
        "gripper_contact_raw": contact_raw,
    }


def _contact_with_hysteresis(raw: float, previous: bool) -> bool:
    """Freeze contact above 0.5, release below 0.1, retain state between."""

    if raw >= CONTACT_THRESHOLD:
        return True
    if raw <= 0.1:
        return False
    return bool(previous)


def _derive_phase(state: Mapping[str, Any], active_kind: str) -> str:
    if active_kind == "recover":
        return "RECOVERY"
    if bool(state["contact_on"]) and float(state["block_pos"][2]) >= DROP_Z_M:
        return "TRANSPORT"
    if bool(state["contact_on"]):
        return "GRASP"
    return "APPROACH"


def _grasp_state(state: Mapping[str, Any], event: str | None) -> str:
    if event == "DROPPED":
        return "DROPPED"
    if bool(state["contact_on"]) and float(state["block_pos"][2]) >= DROP_Z_M:
        return "HELD"
    if bool(state["contact_on"]):
        return "CONTACT"
    return "FREE"


@dataclass
class ActiveGoal:
    kind: str
    pixels: np.ndarray
    position: np.ndarray
    yaw: float
    source_row: int
    source_episode: int
    target_kind: str = "block"
    strategy: str | None = None


@dataclass
class EnvMonitor:
    env_idx: int
    final_goal: ActiveGoal
    active_goal: ActiveGoal
    previous_phase: str
    previous_contact_on: bool
    previous_contact_raw: float
    dist_history: deque[float] = field(default_factory=lambda: deque(maxlen=9))
    comparable_costs: list[float] = field(default_factory=list)
    seen_planning_cycles: int = 0
    logical_calls: int = 0
    last_call_step: int = -10**9
    triggers: list[dict[str, Any]] = field(default_factory=list)
    goal_switches: list[dict[str, Any]] = field(default_factory=list)
    subgoals_started: int = 0
    subgoals_achieved: int = 0
    recoveries_started: int = 0
    recoveries_followed_by_final_success: int = 0
    pending_recovery_success_credit: bool = False


def _active_distance(state: Mapping[str, Any], goal: ActiveGoal) -> float:
    source = state["block_pos"] if goal.target_kind == "block" else state["ee_pos"]
    return float(np.linalg.norm(np.asarray(source) - goal.position))


def _final_distance(state: Mapping[str, Any], goal: ActiveGoal) -> float:
    return float(np.linalg.norm(np.asarray(state["block_pos"]) - goal.position))


def _planner_cost_updates(monitor: EnvMonitor, recorder: Any) -> list[float]:
    cycles = recorder.records[monitor.env_idx]
    updates = []
    for cycle in cycles[monitor.seen_planning_cycles :]:
        if not cycle.get("costs"):
            continue
        final_costs = np.asarray(cycle["costs"][-1], dtype=np.float64)
        value = float(np.min(final_costs))
        if not np.isfinite(value):
            raise RuntimeError(
                f"nonfinite planner best cost: env={monitor.env_idx}, value={value}"
            )
        monitor.comparable_costs.append(value)
        updates.append(value)
    monitor.seen_planning_cycles = len(cycles)
    return updates


def _detect_event(
    monitor: EnvMonitor,
    state: Mapping[str, Any],
    force_smoke: bool,
) -> tuple[str | None, dict[str, Any]]:
    dropped = bool(
        monitor.previous_phase == "TRANSPORT"
        and monitor.previous_contact_on
        and not bool(state["contact_on"])
        and float(state["block_pos"][2]) < DROP_Z_M
    )
    progress = None
    if len(monitor.dist_history) >= STALL_WINDOW_STEPS + 1:
        progress = float(monitor.dist_history[-9] - monitor.dist_history[-1])
    cost_nondecrease = bool(
        len(monitor.comparable_costs) >= 2
        and monitor.comparable_costs[-1]
        >= monitor.comparable_costs[-2] - COST_NONDECREASE_EPS
    )
    stalled = bool(
        progress is not None
        and progress < STALL_PROGRESS_M
        and cost_nondecrease
    )
    event = "DROPPED" if dropped else "STALLED" if stalled else None
    if force_smoke:
        event = "STALLED"
    return event, {
        "dropped": dropped,
        "stalled": stalled,
        "forced_smoke": force_smoke,
        "distance_progress_over_8_steps_m": progress,
        "comparable_planner_cost_count": len(monitor.comparable_costs),
        "latest_two_planner_best_costs": monitor.comparable_costs[-2:],
        "cost_nondecrease": cost_nondecrease,
    }


def _normalize_decision(value: Any) -> dict[str, Any]:
    if hasattr(value, "decision"):
        decision = getattr(value, "decision")
        if isinstance(decision, Mapping):
            payload = dict(decision)
        else:
            payload = {"decision": decision}
        for key in (
            "status",
            "protocol_failure",
            "logical_call_index",
            "attempts",
            "total_latency_ms",
        ):
            if hasattr(value, key):
                payload[f"call_{key}"] = getattr(value, key)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = {"decision": "CONTINUE", "protocol_failure": True}
    raw = payload.get("decision", payload.get("action", payload.get("type", "CONTINUE")))
    if isinstance(raw, Mapping):
        nested = dict(raw)
        nested.update({key: val for key, val in payload.items() if key not in nested})
        payload = nested
        raw = payload.get("decision", payload.get("action", payload.get("type", "CONTINUE")))
    action = str(raw).upper()
    if action not in {"CONTINUE", "SUBGOAL", "RECOVER"}:
        payload["protocol_failure"] = True
        action = "CONTINUE"
    payload["decision"] = action
    return payload


def _payload(
    brain_module: Any,
    monitor: EnvMonitor,
    state: Mapping[str, Any],
    event: str,
    step: int,
    budget: int,
    calls_remaining: int,
) -> dict[str, Any]:
    trend = [float(value) for value in list(monitor.dist_history)[-5:]]
    if not trend:
        trend = [_final_distance(state, monitor.final_goal)]
    trend = [trend[0]] * (5 - len(trend)) + trend
    cost_trend = [float(value) for value in monitor.comparable_costs[-5:]]
    if not cost_trend:
        cost_trend = [0.0]
    cost_trend = [cost_trend[0]] * (5 - len(cost_trend)) + cost_trend
    values = {
        "event": event,
        "step": int(step),
        "budget": int(budget),
        "block_pos": np.asarray(state["block_pos"]).round(6).tolist(),
        "block_yaw": round(float(state["block_yaw"]), 6),
        "target_pos": monitor.final_goal.position.round(6).tolist(),
        "target_yaw": round(float(monitor.final_goal.yaw), 6),
        "ee_pos": np.asarray(state["ee_pos"]).round(6).tolist(),
        "gripper_opening": round(float(state["gripper_opening"]), 6),
        "gripper_contact": bool(state["contact_on"]),
        "dist_to_target": round(_final_distance(state, monitor.final_goal), 6),
        "dist_trend_5": [round(float(value), 6) for value in trend],
        "grasp_state": _grasp_state(state, event),
        "phase": monitor.previous_phase,
        "planner_cost_trend": [
            round(float(value), 6) for value in cost_trend
        ],
        "calls_remaining": int(calls_remaining),
    }
    builder = getattr(brain_module, "build_state_payload", None)
    payload = builder(**values) if callable(builder) else values
    encoded = json.dumps(_jsonable(payload), separators=(",", ":"), ensure_ascii=False)
    # A conservative chars/4 estimate; the provider logger records exact usage.
    if len(encoded) > 2400:
        raise RuntimeError(f"brain payload exceeds conservative 600-token bound: {len(encoded)} chars")
    return payload


def _flush_env_plan(policy: Any, env_idx: int) -> None:
    policy._action_buffer[env_idx].clear()
    if policy._next_init is not None:
        policy._next_init[env_idx] = 0


def _goal_from_real_frame(
    h5: Any,
    retrieval: Mapping[str, Any],
    target_kind: str,
    strategy: str | None = None,
) -> ActiveGoal:
    row = int(retrieval["row"])
    if target_kind == "block":
        position = np.asarray(h5["privileged_block_0_pos"][row], dtype=np.float64)
        yaw = float(np.asarray(h5["privileged_block_0_yaw"][row]).reshape(-1)[0])
    else:
        position = np.asarray(h5["proprio_effector_pos"][row], dtype=np.float64)
        yaw = float(np.asarray(h5["proprio_effector_yaw"][row]).reshape(-1)[0])
    return ActiveGoal(
        kind="subgoal" if target_kind == "block" else "recover",
        pixels=np.asarray(h5["pixels"][row], dtype=np.uint8).copy(),
        position=position.copy(),
        yaw=yaw,
        source_row=row,
        source_episode=int(h5["ep_idx"][row]),
        target_kind=target_kind,
        strategy=strategy,
    )


def _decision_position(decision: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    for key in keys:
        if key in decision:
            value = np.asarray(decision[key], dtype=np.float64).reshape(-1)
            if value.shape == (3,) and np.isfinite(value).all():
                return value
    raise ValueError(f"decision lacks finite position in keys={list(keys)}")


def _switch_goal(
    monitor: EnvMonitor,
    goal: ActiveGoal,
    policy: Any,
    step: int,
    reason: str,
    state: Mapping[str, Any],
    retrieval: Mapping[str, Any] | None,
) -> None:
    before = monitor.active_goal
    monitor.active_goal = goal
    monitor.dist_history.clear()
    monitor.dist_history.append(_final_distance(state, monitor.final_goal))
    monitor.comparable_costs.clear()
    monitor.seen_planning_cycles = len(policy.cost_recorder.records[monitor.env_idx])
    _flush_env_plan(policy, monitor.env_idx)
    monitor.goal_switches.append(
        {
            "env_idx": monitor.env_idx,
            "step": step,
            "reason": reason,
            "from_kind": before.kind,
            "from_row": before.source_row,
            "to_kind": goal.kind,
            "to_row": goal.source_row,
            "to_episode": goal.source_episode,
            "target_kind": goal.target_kind,
            "target_position": goal.position,
            "target_yaw": goal.yaw,
            "from_pixels_sha256": hashlib.sha256(
                np.ascontiguousarray(before.pixels).tobytes()
            ).hexdigest(),
            "to_pixels_sha256": hashlib.sha256(
                np.ascontiguousarray(goal.pixels).tobytes()
            ).hexdigest(),
            "strategy": goal.strategy,
            "retrieval": retrieval,
            "action_buffer_flushed": True,
            "planning_cycle_count_before_flush": len(
                policy.cost_recorder.records[monitor.env_idx]
            ),
            "replan_observed_at_step": None,
            "physical_target_changed": False,
        }
    )


def _brain_manifest(supervisor: Any) -> dict[str, Any]:
    method = getattr(supervisor, "manifest", None)
    if callable(method):
        return _jsonable(method())
    return {
        "provider": "deepseek",
        "requested_model": "deepseek-v4-flash",
        "thinking": False,
        "temperature": 0.2,
        "interface": type(supervisor).__name__,
    }


def _brain_summary(supervisor: Any) -> dict[str, Any]:
    method = getattr(supervisor, "summary", None)
    return _jsonable(method()) if callable(method) else {}


def _build_supervisor(brain_module: Any, output: Path) -> Any:
    cls = getattr(brain_module, "BrainSupervisor", None)
    if cls is None:
        raise RuntimeError("tools/brain_supervisor.py lacks BrainSupervisor")
    config_cls = getattr(brain_module, "SupervisorConfig", None)
    if config_cls is None or not hasattr(config_cls, "from_value"):
        raise RuntimeError("tools/brain_supervisor.py lacks SupervisorConfig.from_value")
    config = config_cls.from_value(None)
    supervisor = cls(config, output)
    manifest = supervisor.manifest()
    actual = {
        "provider": manifest.get("provider"),
        "model": manifest.get("requested_model"),
        "endpoint": manifest.get("endpoint"),
        "thinking": manifest.get("thinking"),
        "temperature": manifest.get("temperature"),
        "id_box": manifest.get("id_box"),
    }
    expected = {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "thinking": {"type": "disabled"},
        "temperature": 0.1,
        "id_box": {"x": list(ID_X), "y": list(ID_Y), "z": list(ID_Z)},
    }
    if actual != expected:
        raise RuntimeError(
            f"B1 supervisor frozen config mismatch: expected={expected}, actual={actual}"
        )
    return supervisor


def _supervisor_decide(
    supervisor: Any,
    payload: dict[str, Any],
    event: str,
    env_idx: int,
    step: int,
) -> dict[str, Any]:
    method = getattr(supervisor, "decide", None)
    if not callable(method):
        raise RuntimeError("BrainSupervisor lacks decide")
    result = method(
        payload=payload,
        event=event,
        env_idx=env_idx,
        step=step,
    )
    return _normalize_decision(result)


def _frozen_t2_manifest() -> dict[str, Any]:
    return {
        **trust_common.PROTOCOL_SPECS["t2"],
        "num_samples": 300,
        "n_steps": 10,
        "topk": 30,
        "horizon": 5,
        "receding_horizon": 5,
        "action_block": 5,
        "selector": "legacy_updated_elite_mean",
        "solver_seed": 42,
    }


def _validate_baseline_pairing(
    offset: int,
    selection: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    index_path: Path,
) -> dict[str, Any]:
    baseline_root = OUTPUT_ROOT / f"baseline_offset{offset}"
    manifest_path = baseline_root / "run_manifest.json"
    results_path = baseline_root / "results.json"
    if not manifest_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            f"formal brain requires completed paired baseline: {baseline_root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results_identity = _file_identity(results_path)
    recorded_results_identity = manifest.get("results")
    if recorded_results_identity != results_identity:
        raise RuntimeError(
            "paired baseline results identity mismatch: "
            f"recorded={recorded_results_identity}, actual={results_identity}"
        )
    current_evaluator = _file_identity(Path(__file__))
    index_identity = _file_identity(index_path / "metadata.json")
    actual = {
        "status": manifest.get("status"),
        "mode": manifest.get("mode"),
        "offset": manifest.get("goal_offset_steps"),
        "budget": manifest.get("eval_budget"),
        "num_eval": manifest.get("num_eval"),
        "seed": manifest.get("seed"),
        "rows_sha": manifest.get("selection", {}).get("identities", {}).get(
            "rows_sha256_int64"
        ),
        "success_num_eval": results.get("metrics", {}).get("num_eval"),
        "checkpoint_weights_sha": manifest.get("checkpoint", {})
        .get("weights", {})
        .get("sha256"),
        "checkpoint_config_sha": manifest.get("checkpoint", {})
        .get("config", {})
        .get("sha256"),
        "t2": manifest.get("t2"),
        "global_exclusions_sha": manifest.get("retrieval", {}).get(
            "global_excluded_episodes_sha256_int64"
        ),
        "index_sha": manifest.get("retrieval", {})
        .get("index", {})
        .get("sha256"),
        "evaluator_sha": manifest.get("helper_provenance", {})
        .get("this_evaluator", {})
        .get("sha256"),
    }
    expected = {
        "status": "complete",
        "mode": "baseline",
        "offset": offset,
        "budget": 2 * offset,
        "num_eval": 50,
        "seed": 42,
        "rows_sha": selection["identities"]["rows_sha256_int64"],
        "success_num_eval": 50,
        "checkpoint_weights_sha": checkpoint["weights"]["sha256"],
        "checkpoint_config_sha": checkpoint["config"]["sha256"],
        "t2": _frozen_t2_manifest(),
        "global_exclusions_sha": selection["identities"][
            "episodes_sha256_int64"
        ],
        "index_sha": index_identity["sha256"],
        "evaluator_sha": current_evaluator["sha256"],
    }
    if actual != expected:
        raise RuntimeError(
            f"paired baseline contract mismatch: expected={expected}, actual={actual}"
        )
    if manifest.get("selection", {}).get("rows") != _jsonable(selection["rows"]):
        raise RuntimeError("paired baseline formal rows differ elementwise")
    return {
        "root": str(baseline_root.resolve()),
        "manifest": _file_identity(manifest_path),
        "results": _file_identity(results_path),
    }


def _prepare_world_inputs(
    world: Any,
    dataset: Any,
    rows: np.ndarray,
    offset: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    selected = dataset.get_row_data(rows)
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = np.asarray(selected[ep_key], dtype=np.int64)
    starts = np.asarray(selected["step_idx"], dtype=np.int64)
    init_state, goal_state, dataset_videos = ood._extract_init_goal(
        dataset, episodes, starts, offset
    )
    world.reset(seed=init_state.get("seed"))
    callables = [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
    ]
    merged = {**init_state, **goal_state}
    for env_idx, wrapped in enumerate(world.envs.envs):
        ood._apply_callables(
            wrapped.unwrapped, callables, {key: value[env_idx] for key, value in merged.items()}
        )

    # Red B1 uses the literal HDF5 current and row+offset future frames.
    init_state["pixels"] = np.asarray(init_state["pixels"], dtype=np.uint8).copy()
    goal_state["goal"] = np.asarray(goal_state["goal"], dtype=np.uint8).copy()
    shape_prefix = world.infos["pixels"].shape[:2]
    for state in (init_state, goal_state):
        for key, value in state.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()
    return init_state, goal_state, dataset_videos, episodes, starts


def _initial_monitors(
    init_state: Mapping[str, np.ndarray],
    goal_state: Mapping[str, np.ndarray],
    rows: np.ndarray,
    goal_rows: np.ndarray,
    episodes: np.ndarray,
) -> list[EnvMonitor]:
    monitors = []
    for env_idx in range(len(rows)):
        state = {
            "block_pos": np.asarray(init_state["privileged_block_0_pos"][env_idx]),
            "block_yaw": float(
                np.asarray(init_state["privileged_block_0_yaw"][env_idx]).reshape(-1)[0]
            ),
            "ee_pos": np.asarray(init_state["proprio_effector_pos"][env_idx]),
            "gripper_opening": float(
                np.asarray(init_state["proprio_gripper_opening"][env_idx]).reshape(-1)[0]
            ),
            "gripper_contact_raw": float(
                np.asarray(init_state["proprio_gripper_contact"][env_idx]).reshape(-1)[0]
            ),
        }
        state["contact_on"] = _contact_with_hysteresis(
            float(state["gripper_contact_raw"]), False
        )
        final = ActiveGoal(
            kind="final",
            pixels=np.asarray(goal_state["goal"][env_idx], dtype=np.uint8).copy(),
            position=np.asarray(
                goal_state["goal_privileged_block_0_pos"][env_idx], dtype=np.float64
            ).reshape(3),
            yaw=float(
                np.asarray(goal_state["goal_privileged_block_0_yaw"][env_idx]).reshape(-1)[0]
            ),
            source_row=int(goal_rows[env_idx]),
            source_episode=int(episodes[env_idx]),
            target_kind="block",
        )
        phase = _derive_phase(state, "final")
        monitor = EnvMonitor(
            env_idx=env_idx,
            final_goal=final,
            active_goal=final,
            previous_phase=phase,
            previous_contact_on=bool(state["contact_on"]),
            previous_contact_raw=float(state["gripper_contact_raw"]),
        )
        monitor.dist_history.append(_final_distance(state, final))
        monitors.append(monitor)
    return monitors


def _run_evaluation(
    args: argparse.Namespace,
    output: Path,
    world: Any,
    policy: Any,
    recorder: Any,
    proxy: Any,
    dataset: Any,
    index: HeldoutMemoryIndex,
    selection: Mapping[str, Any],
    rows: np.ndarray,
    episodes: np.ndarray,
    starts: np.ndarray,
    init_state: Mapping[str, np.ndarray],
    goal_state: Mapping[str, np.ndarray],
    dataset_videos: list[np.ndarray],
    supervisor: Any | None,
    brain_module: Any | None,
) -> tuple[dict[str, Any], list[EnvMonitor], dict[str, Any]]:
    import hdf5plugin  # noqa: F401
    import h5py
    from stable_worldmodel.plot import save_panel_videos

    goal_rows = np.asarray(selection["goal_rows"], dtype=np.int64)[: args.num_eval]
    monitors = _initial_monitors(
        init_state, goal_state, rows, goal_rows, episodes
    )
    successes = np.zeros(args.num_eval, dtype=bool)
    frames: dict[int, list[np.ndarray]] = defaultdict(list)
    goal_frames: dict[int, list[np.ndarray]] = defaultdict(list)
    step_counter = 0
    event_log: list[dict[str, Any]] = []
    retrieval_log: list[dict[str, Any]] = []

    h5 = h5py.File(args.dataset, "r", swmr=True)
    try:
        def on_step(active_world: Any) -> None:
            nonlocal step_counter
            step_counter += 1
            successes[:] |= np.asarray(active_world.terminateds, dtype=bool)
            for env_idx, monitor in enumerate(monitors):
                state = _live_state(active_world.infos, env_idx)
                state["contact_on"] = _contact_with_hysteresis(
                    float(state["gripper_contact_raw"]), monitor.previous_contact_on
                )
                planner_updates = _planner_cost_updates(monitor, recorder)
                if planner_updates:
                    for switch in reversed(monitor.goal_switches):
                        if switch["replan_observed_at_step"] is None:
                            switch["replan_observed_at_step"] = step_counter
                            switch["replan_best_cost"] = planner_updates[0]
                            switch["replan_latency_env_steps"] = (
                                step_counter - int(switch["step"])
                            )
                            break

                # The raw environment target is always the final goal.  Once it
                # terminates, record the terminal frame but never spend an LLM
                # call or install another goal for that environment.
                if bool(active_world.terminateds[env_idx]):
                    monitor.previous_phase = _derive_phase(
                        state, monitor.active_goal.kind
                    )
                    monitor.previous_contact_on = bool(state["contact_on"])
                    monitor.previous_contact_raw = float(
                        state["gripper_contact_raw"]
                    )
                    pixel = np.asarray(active_world.infos["pixels"][env_idx])
                    frames[env_idx].append(
                        (pixel[-1] if pixel.ndim > 3 else pixel).copy()
                    )
                    goal_frames[env_idx].append(
                        monitor.active_goal.pixels.copy()
                    )
                    continue

                goal_switched_on_achievement = False
                if (
                    monitor.active_goal.kind != "final"
                    and _active_distance(state, monitor.active_goal) <= SUBGOAL_TOLERANCE_M
                ):
                    monitor.subgoals_achieved += int(monitor.active_goal.kind == "subgoal")
                    _switch_goal(
                        monitor,
                        monitor.final_goal,
                        policy,
                        step_counter,
                        "active_goal_achieved",
                        state,
                        None,
                    )
                    goal_switched_on_achievement = True

                final_distance = _final_distance(state, monitor.final_goal)
                if not goal_switched_on_achievement:
                    monitor.dist_history.append(final_distance)
                force_smoke = bool(
                    args.force_smoke_trigger_step is not None
                    and env_idx == 0
                    and step_counter == args.force_smoke_trigger_step
                )
                event, diagnostics = _detect_event(monitor, state, force_smoke)
                call_allowed = bool(
                    args.mode == "brain"
                    and event is not None
                    and monitor.logical_calls < MAX_CALLS_PER_EPISODE
                    and step_counter - monitor.last_call_step >= MIN_CALL_INTERVAL_STEPS
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
                        "call_allowed": call_allowed,
                        "suppression_reason": None,
                        "decision": None,
                    }
                    if args.mode != "brain":
                        record["suppression_reason"] = "baseline_no_llm"
                    elif monitor.logical_calls >= MAX_CALLS_PER_EPISODE:
                        record["suppression_reason"] = "episode_call_budget_exhausted"
                    elif step_counter - monitor.last_call_step < MIN_CALL_INTERVAL_STEPS:
                        record["suppression_reason"] = "minimum_call_interval"

                if call_allowed:
                    assert supervisor is not None and brain_module is not None
                    payload = _payload(
                        brain_module,
                        monitor,
                        state,
                        str(event),
                        step_counter,
                        args.eval_budget,
                        supervisor.budget.calls_remaining(str(env_idx)),
                    )
                    monitor.logical_calls += 1
                    monitor.last_call_step = step_counter
                    decision = _supervisor_decide(
                        supervisor, payload, str(event), env_idx, step_counter
                    )
                    assert record is not None
                    record["payload"] = payload
                    record["decision"] = decision
                    action = decision["decision"]
                    try:
                        retrieval = None
                        goal = None
                        synthetic_smoke_switch = bool(
                            args.force_smoke_goal_switch
                            and force_smoke
                            and action == "CONTINUE"
                        )
                        applied_action = (
                            "SUBGOAL" if synthetic_smoke_switch else action
                        )
                        if synthetic_smoke_switch:
                            record["synthetic_goal_switch"] = {
                                "enabled": True,
                                "reason": (
                                    "forced smoke diagnostic after a real LLM "
                                    "CONTINUE; excluded from LLM decision statistics"
                                ),
                            }
                        if applied_action == "SUBGOAL":
                            requested = (
                                np.asarray(state["block_pos"], dtype=np.float64)
                                + np.asarray([0.05, 0.0, 0.0], dtype=np.float64)
                                if synthetic_smoke_switch
                                else _decision_position(
                                    decision,
                                    ("block_pos", "position", "target_pos", "pos"),
                                )
                            )
                            clamped = np.asarray(
                                [
                                    np.clip(requested[0], *ID_X),
                                    np.clip(requested[1], *ID_Y),
                                    np.clip(requested[2], *ID_Z),
                                ],
                                dtype=np.float64,
                            )
                            yaw = float(
                                state["block_yaw"]
                                if synthetic_smoke_switch
                                else decision.get("yaw", state["block_yaw"])
                            )
                            if not np.isfinite(yaw):
                                raise ValueError("nonfinite SUBGOAL yaw")
                            retrieval = index.nearest_goal_frame(
                                "block", clamped, yaw, int(episodes[env_idx])
                            )
                            retrieval["requested_position"] = requested
                            retrieval["clamped_position"] = clamped
                            retrieval["clamped"] = bool(not np.array_equal(requested, clamped))
                            retrieval["requested_yaw"] = yaw
                            retrieval["synthetic_smoke_switch"] = synthetic_smoke_switch
                            goal = _goal_from_real_frame(h5, retrieval, "block")
                            monitor.subgoals_started += 1
                        elif applied_action == "RECOVER":
                            requested = _decision_position(
                                decision, ("ee_pos", "position", "target_pos", "pos")
                            )
                            clamped = np.asarray(
                                [
                                    np.clip(requested[0], *ID_X),
                                    np.clip(requested[1], *ID_Y),
                                    np.clip(requested[2], *ID_Z),
                                ],
                                dtype=np.float64,
                            )
                            strategy = str(decision.get("strategy", "unspecified"))
                            retrieval = index.nearest_goal_frame(
                                "ee", clamped, None, int(episodes[env_idx])
                            )
                            retrieval["requested_position"] = requested
                            retrieval["clamped_position"] = clamped
                            retrieval["clamped"] = bool(
                                not np.array_equal(requested, clamped)
                            )
                            retrieval["strategy"] = strategy
                            goal = _goal_from_real_frame(
                                h5, retrieval, "ee", strategy=strategy
                            )
                            monitor.recoveries_started += 1
                            monitor.pending_recovery_success_credit = True
                        if goal is not None and retrieval is not None:
                            _switch_goal(
                                monitor,
                                goal,
                                policy,
                                step_counter,
                                (
                                    "synthetic_smoke_subgoal"
                                    if synthetic_smoke_switch
                                    else f"llm_{applied_action.lower()}"
                                ),
                                state,
                                retrieval,
                            )
                            retrieval_log.append(
                                {
                                    "env_idx": env_idx,
                                    "episode": int(episodes[env_idx]),
                                    "step": step_counter,
                                    "decision": (
                                        "SMOKE_SYNTHETIC_SUBGOAL"
                                        if synthetic_smoke_switch
                                        else applied_action
                                    ),
                                    "retrieval": retrieval,
                                    "selected_real_frame": {
                                        "row": goal.source_row,
                                        "episode": goal.source_episode,
                                        "position": goal.position,
                                        "yaw": goal.yaw,
                                        "pixels_shape": list(goal.pixels.shape),
                                    },
                                }
                            )
                    except (KeyError, TypeError, ValueError) as error:
                        decision["decision"] = "CONTINUE"
                        decision["application_failure"] = f"{type(error).__name__}: {error}"
                        decision["protocol_failure"] = True

                if record is not None:
                    monitor.triggers.append(record)
                    event_log.append(record)

                current_phase = _derive_phase(state, monitor.active_goal.kind)
                monitor.previous_phase = current_phase
                monitor.previous_contact_on = bool(state["contact_on"])
                monitor.previous_contact_raw = float(state["gripper_contact_raw"])
                # Mutable planner goal only; raw MuJoCo target remains final.
                active_world.infos["goal"][env_idx] = np.broadcast_to(
                    monitor.active_goal.pixels,
                    active_world.infos["goal"][env_idx].shape,
                )
                pixel = np.asarray(active_world.infos["pixels"][env_idx])
                frames[env_idx].append((pixel[-1] if pixel.ndim > 3 else pixel).copy())
                goal_frames[env_idx].append(monitor.active_goal.pixels.copy())

        # The first policy call must see the final real HDF5 goal.
        for env_idx, monitor in enumerate(monitors):
            world.infos["goal"][env_idx] = np.broadcast_to(
                monitor.active_goal.pixels, world.infos["goal"][env_idx].shape
            )
        world._run(max_steps=args.eval_budget, mode="wait", on_step=on_step)
    finally:
        h5.close()

    for env_idx, monitor in enumerate(monitors):
        if successes[env_idx] and monitor.pending_recovery_success_credit:
            monitor.recoveries_followed_by_final_success += 1

    save_panel_videos(
        output / "videos",
        {
            "agent": frames,
            "dataset": dataset_videos,
            "goal": goal_frames,
        },
    )
    _write_json(output / "events.json", event_log)
    _write_json(output / "subgoal_retrieval.json", retrieval_log)
    metrics = {
        "success_rate": float(successes.sum()) / args.num_eval * 100.0,
        "success_count": int(successes.sum()),
        "num_eval": args.num_eval,
        "episode_successes": successes,
        "seeds": init_state.get("seed"),
    }
    runtime = {
        "physical_steps_executed_global": step_counter,
        "events": event_log,
        "retrievals": retrieval_log,
        "goal_switches": [
            switch for monitor in monitors for switch in monitor.goal_switches
        ],
    }
    return metrics, monitors, runtime


def _behavior_summary(monitors: Sequence[EnvMonitor], supervisor_summary: Any) -> dict[str, Any]:
    triggers = [record for monitor in monitors for record in monitor.triggers]
    decisions = [
        record["decision"]["decision"]
        for record in triggers
        if isinstance(record.get("decision"), Mapping)
    ]
    call_records = [
        record["decision"].get("call_record", {})
        for record in triggers
        if isinstance(record.get("decision"), Mapping)
        and isinstance(record["decision"].get("call_record"), Mapping)
    ]
    logical_latencies = [
        float(record["total_latency_ms"])
        for record in call_records
        if isinstance(record.get("total_latency_ms"), (int, float))
        and not isinstance(record.get("total_latency_ms"), bool)
    ]
    reported_tokens = sum(
        int(record.get("usage", {}).get("total_tokens", 0))
        for record in call_records
        if isinstance(record.get("usage"), Mapping)
        and isinstance(record.get("usage", {}).get("total_tokens", 0), int)
    )
    by_event = {
        name: sum(record["event"] == name for record in triggers)
        for name in ("DROPPED", "STALLED")
    }
    by_decision = {name: decisions.count(name) for name in ("CONTINUE", "SUBGOAL", "RECOVER")}
    calls = [monitor.logical_calls for monitor in monitors]
    triggers_per_episode = [len(monitor.triggers) for monitor in monitors]
    started = sum(monitor.subgoals_started for monitor in monitors)
    achieved = sum(monitor.subgoals_achieved for monitor in monitors)
    recoveries = sum(monitor.recoveries_started for monitor in monitors)
    recovery_episodes = sum(monitor.recoveries_started > 0 for monitor in monitors)
    recovery_success_episodes = sum(
        monitor.recoveries_followed_by_final_success for monitor in monitors
    )
    budget_summary = (
        supervisor_summary.get("budget", {})
        if isinstance(supervisor_summary, Mapping)
        else {}
    )
    if supervisor_summary and not {
        "accounted_tokens",
        "max_total_tokens",
    }.issubset(budget_summary):
        raise RuntimeError("brain supervisor summary lacks accounted token budget")
    accounted_tokens = int(budget_summary.get("accounted_tokens", 0))
    token_budget = int(budget_summary.get("max_total_tokens", 1_000_000))
    if accounted_tokens > token_budget:
        raise RuntimeError(
            f"brain token budget exceeded: accounted={accounted_tokens}, limit={token_budget}"
        )
    return {
        "trigger_count": len(triggers),
        "trigger_count_by_event": by_event,
        "trigger_count_distribution_per_episode": triggers_per_episode,
        "logical_llm_calls": int(sum(calls)),
        "logical_call_count_distribution_per_episode": calls,
        "mean_calls_per_episode": float(np.mean(calls)) if calls else 0.0,
        "max_calls_in_episode": max(calls, default=0),
        "decision_counts": by_decision,
        "decision_fractions": {
            key: (value / len(decisions) if decisions else 0.0)
            for key, value in by_decision.items()
        },
        "subgoals_started": started,
        "subgoals_achieved": achieved,
        "subgoal_achievement_rate": achieved / started if started else None,
        "recoveries_started": recoveries,
        "recovery_decision_count": recoveries,
        "episodes_with_recovery": recovery_episodes,
        "episodes_with_recovery_and_final_success": recovery_success_episodes,
        "recover_episode_final_success_rate": (
            recovery_success_episodes / recovery_episodes
            if recovery_episodes
            else None
        ),
        "llm_cost": {
            "logical_calls": len(call_records),
            "total_tokens": accounted_tokens,
            "total_tokens_source": "supervisor_budget_accounted_tokens",
            "token_budget": token_budget,
            "within_token_budget": accounted_tokens <= token_budget,
            "reported_total_tokens_from_final_attempts": reported_tokens,
            "average_logical_call_latency_ms": (
                float(np.mean(logical_latencies)) if logical_latencies else None
            ),
            "total_logical_call_latency_ms": float(sum(logical_latencies)),
            "mean_calls_per_episode": float(np.mean(calls)) if calls else 0.0,
        },
        "supervisor": supervisor_summary,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode not in MODES or args.goal_offset_steps not in OFFSETS:
        raise ValueError("unsupported frozen B1 mode/offset")
    if args.seed != FORMAL_SEED:
        raise ValueError("B1 seed is frozen at 42")
    if args.num_eval not in (2, 50):
        raise ValueError("B1 num_eval is frozen at 2 or 50")
    expected_budget = 2 * args.goal_offset_steps
    if args.eval_budget != expected_budget:
        raise ValueError(
            f"B1 budget must equal 2*offset: expected={expected_budget}, actual={args.eval_budget}"
        )
    if args.force_smoke_trigger_step is not None:
        if args.mode != "brain" or args.num_eval != 2:
            raise ValueError("--force-smoke-trigger-step is brain num_eval=2 only")
        if not 1 <= args.force_smoke_trigger_step < args.eval_budget:
            raise ValueError("forced smoke trigger must fall inside the eval budget")
        if args.force_smoke_trigger_step != 5:
            raise ValueError("B1 forced smoke trigger is frozen at physical step 5")
    if args.force_smoke_goal_switch:
        if (
            args.mode != "brain"
            or args.num_eval != 2
            or args.force_smoke_trigger_step is None
        ):
            raise ValueError(
                "--force-smoke-goal-switch requires brain num_eval=2 and "
                "--force-smoke-trigger-step"
            )
    if args.num_eval == 50 and (
        args.force_smoke_trigger_step is not None or args.force_smoke_goal_switch
    ):
        raise ValueError("formal runs reject all forced smoke controls")
    if args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError("formal 50-env B1 run requires --authorize-formal")
    for path, label in (
        (args.dataset, "dataset"),
        (args.fixed_manifest, "fixed manifest"),
        (args.index / "metadata.json", "memory index"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    _configure_storage()
    selection = select_longhorizon_rows(
        args.dataset, args.fixed_manifest, args.goal_offset_steps, args.seed
    )
    rows = np.asarray(selection["rows"], dtype=np.int64)[: args.num_eval]
    fixed_episodes = np.asarray(selection["episodes"], dtype=np.int64)
    checkpoint = trust_common.frozen_masked_checkpoint_contract()
    baseline_pairing = None
    if args.mode == "brain" and args.num_eval == 50:
        baseline_pairing = _validate_baseline_pairing(
            args.goal_offset_steps, selection, checkpoint, args.index
        )

    requested_output = args.output or _default_output(
        args.mode, args.goal_offset_steps, args.num_eval
    )
    _validate_output_contract(
        requested_output, args.mode, args.goal_offset_steps, args.num_eval
    )
    output = _safe_output(requested_output, args.overwrite)

    if args.self_test:
        payload = {
            "format_version": "cube_brain_b1_self_test_v1",
            "selection": selection,
            "evaluated_rows": rows,
            "checkpoint": checkpoint,
        }
        _write_json(output / "self_test.json", payload)
        print(output)
        return 0

    import hdf5plugin  # noqa: F401
    import h5py
    import stable_worldmodel as swm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("B1 online evaluation requires CUDA")
    index = HeldoutMemoryIndex(args.index, args.dataset, fixed_episodes)
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    selected = dataset.get_row_data(rows)
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = np.asarray(selected[ep_key], dtype=np.int64)
    starts = np.asarray(selected["step_idx"], dtype=np.int64)
    if not np.array_equal(episodes, fixed_episodes[: args.num_eval]):
        raise RuntimeError("evaluated episodes differ from held-out selection")
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        initial_query_features = np.concatenate(
            [memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows], axis=0
        )

    model = swm.wm.utils.load_pretrained(
        str(trust_common.MASKED_CHECKPOINT), cache_dir=str(PROJECT_ROOT)
    )
    model = model.to(args.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    scaler = legacy._standard_scaler(index)
    recorder = ood.PlanningCostRecorder(args.num_eval)
    proxy = trust.TrustRegionCostProxy(model, "t2")
    solver_cls = trust.make_trust_region_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=proxy,
        batch_size=1,
        num_samples=trust_common.NUM_SAMPLES,
        var_scale=trust_common.PROTOCOL_SPECS["t2"]["var_scale"],
        n_steps=trust_common.N_STEPS,
        topk=trust_common.TOPK,
        device=args.device,
        seed=FORMAL_SEED,
        callbacks=[recorder],
        selector="mean",
        recorder=recorder,
        trust_protocol="t2",
    )
    config = swm.PlanConfig(
        horizon=trust_common.HORIZON,
        receding_horizon=trust_common.HORIZON,
        action_block=trust_common.ACTION_BLOCK,
    )
    policy_cls = trust.make_trust_policy(swm.policy.WorldModelPolicy)
    policy = policy_cls(
        solver=solver,
        config=config,
        process={"action": scaler},
        transform={"pixels": ood._image_transform(224), "goal": ood._image_transform(224)},
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
        max_episode_steps=2 * args.eval_budget,
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
    init_state, goal_state, dataset_videos, episodes, starts = _prepare_world_inputs(
        world, dataset, rows, args.goal_offset_steps
    )

    brain_module = supervisor = None
    if args.mode == "brain":
        import brain_supervisor as brain_module  # type: ignore[no-redef]

        supervisor = _build_supervisor(brain_module, output)

    run_manifest = {
        "format_version": "cube_brain_b1_run_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "mode": args.mode,
        "goal_offset_steps": args.goal_offset_steps,
        "eval_budget": args.eval_budget,
        "num_eval": args.num_eval,
        "seed": args.seed,
        "selection": selection,
        "evaluated_rows": rows,
        "evaluated_episodes": episodes,
        "evaluated_starts": starts,
        "checkpoint": checkpoint,
        "t2": _frozen_t2_manifest(),
        "retrieval": {
            "index": _file_identity(args.index / "metadata.json"),
            "global_excluded_episodes": fixed_episodes,
            "global_excluded_episodes_sha256_int64": index.fixed_episodes_sha256,
            "also_exclude_current_episode": True,
            "ten_unique_source_episodes": True,
            "stable_order": "distance,row",
        },
        "goal": {
            "final": "literal HDF5 row+offset pixels and privileged pose",
            "physical_target": "always final; never changed by brain",
            "subgoal": "nearest allowed real HDF5 anchor frame",
            "subgoal_tolerance_m": SUBGOAL_TOLERANCE_M,
            "id_box": {
                "x": ID_X,
                "y": ID_Y,
                "z": ID_Z,
                "allowed_anchor_z_min_diagnostic": index.id_z_min,
            },
        },
        "trigger": {
            "contact_threshold": CONTACT_THRESHOLD,
            "contact_release_threshold": 0.1,
            "contact_hysteresis_between_thresholds": True,
            "drop_z_m": DROP_Z_M,
            "stall_window_steps": STALL_WINDOW_STEPS,
            "stall_progress_m": STALL_PROGRESS_M,
            "cost_nondecrease_epsilon": COST_NONDECREASE_EPS,
            "priority": ["DROPPED", "STALLED"],
            "max_calls_per_episode": MAX_CALLS_PER_EPISODE,
            "minimum_call_interval_steps": MIN_CALL_INTERVAL_STEPS,
            "force_smoke_trigger_step": args.force_smoke_trigger_step,
            "force_smoke_goal_switch": args.force_smoke_goal_switch,
        },
        "brain": None if supervisor is None else _brain_manifest(supervisor),
        "baseline_pairing": baseline_pairing,
        "helper_provenance": {
            "eval_trust_region": _file_identity(Path(trust.__file__)),
            "eval_ood_color": _file_identity(Path(ood.__file__)),
            "eval_memory_seed": _file_identity(Path(legacy.__file__)),
            "memory_index_code": _file_identity(Path(memory.__file__)),
            "this_evaluator": _file_identity(Path(__file__)),
            **(
                {"brain_supervisor": _file_identity(Path(brain_module.__file__))}
                if brain_module is not None
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
            proxy,
            dataset,
            index,
            selection,
            rows,
            episodes,
            starts,
            init_state,
            goal_state,
            dataset_videos,
            supervisor,
            brain_module,
        )
    finally:
        world.close()
    elapsed = time.time() - started

    trace = trust._save_trace(output, proxy)
    cost_history = ood._save_cost_history(
        output, recorder, rows, episodes, starts, "mean"
    )
    goal_switches = [
        switch for monitor in monitors for switch in monitor.goal_switches
    ]
    _write_json(output / "goal_switches.json", goal_switches)
    supervisor_summary = {} if supervisor is None else _brain_summary(supervisor)
    if supervisor is not None:
        _write_json(output / "brain_api_summary.json", supervisor_summary)
        llm_path = output / "llm_calls.json"
        if not llm_path.exists():
            _write_json(llm_path, [])
    behavior = _behavior_summary(monitors, supervisor_summary)
    artifacts = {
        "events": _file_identity(output / "events.json"),
        "subgoal_retrieval": _file_identity(output / "subgoal_retrieval.json"),
        "goal_switches": _file_identity(output / "goal_switches.json"),
    }
    if supervisor is not None:
        artifacts.update(
            {
                "llm_calls": _file_identity(output / "llm_calls.json"),
                "brain_api_manifest": _file_identity(
                    output / "brain_api_manifest.json"
                ),
                "brain_api_summary": _file_identity(output / "brain_api_summary.json"),
            }
        )
    payload = {
        "format_version": "cube_brain_b1_evaluation_v1",
        "protocol": {
            "mode": args.mode,
            "goal_offset_steps": args.goal_offset_steps,
            "eval_budget": args.eval_budget,
            "seed": args.seed,
            "checkpoint": checkpoint,
            "t2": run_manifest["t2"],
            "retrieval": run_manifest["retrieval"],
            "brain": run_manifest["brain"],
            "baseline_pairing": baseline_pairing,
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
    _write_json(output / "results.json", payload)
    success_text = ", ".join(
        "True" if value else "False" for value in metrics["episode_successes"]
    )
    (output / "results.txt").write_text(
        f"mode: {args.mode}\n"
        f"goal_offset_steps: {args.goal_offset_steps}\n"
        f"eval_budget: {args.eval_budget}\n"
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
    parser.add_argument("--goal-offset-steps", type=int, choices=OFFSETS, default=75)
    parser.add_argument("--eval-budget", type=int)
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--fixed-manifest", type=Path, default=FIXED_MANIFEST)
    parser.add_argument("--index", type=Path, default=MEMORY_INDEX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    parser.add_argument("--force-smoke-trigger-step", type=int)
    parser.add_argument("--force-smoke-goal-switch", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval_budget is None:
        args.eval_budget = 2 * args.goal_offset_steps
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
