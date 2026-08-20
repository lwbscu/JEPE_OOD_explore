#!/usr/bin/env python3
"""Probe-coordinate Cube planning with optional geometric waypoint chains.

This evaluator is intentionally independent of the older entry points.  It
reuses their frozen T2 policy, robust_v1 world model, xyz probe, start rows and
target rows, but never encodes a goal image.  The physical MuJoCo target is set
once to the final pose; intermediate waypoints only replace the privileged xyz
read by the planner cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
for module_root in (HERE, TOOLS):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

import build_cube_memory_index as memory  # noqa: E402
import cube_imagination_error_common as xyz_common  # noqa: E402
import cube_probe_common as probe_common  # noqa: E402
import cube_trust_region_common as t2common  # noqa: E402
import eval_brain_b1 as brain  # noqa: E402
import eval_goal_ood as goal_ood  # noqa: E402
import eval_memory_seed as memory_legacy  # noqa: E402
import eval_ood_color as ood  # noqa: E402
import eval_probe_goal_ood as probe_goal  # noqa: E402
import eval_trust_region as trust  # noqa: E402


PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "outputs/eval/cube/waypoint_probe"
GOAL_OOD_REFERENCE = PROJECT / "outputs/eval/cube/goal_ood_curve"
PROBE_DIRECT_REFERENCE = PROJECT / "outputs/eval/cube/probe_goal_cost"
LONG_REFERENCE = PROJECT / "outputs/eval/cube/longhorizon/baseline_offset100"
RED_REFERENCE = PROJECT / "outputs/eval/cube/robust_v1/red/results.json"
DEFAULT_PROBE = PROJECT / "models/probes/cube_robust_v1_xyz/robust_v1.pt"
DEFAULT_PROBE_METADATA = PROJECT / "outputs/probe/cube_robust_v1/dataset/metadata.json"

FORMAL_SEED = 42
FORMAL_NUM_EVAL = 50
SMOKE_NUM_EVAL = 2
ARRIVAL_TOLERANCE_M = 0.02
SEGMENT_TIMEOUT_STEPS = 25
STALL_WINDOW_STEPS = 8
STALL_PROGRESS_M = 0.01
ROLLOUT_ENV_STEPS = t2common.HORIZON * t2common.ACTION_BLOCK

STANDARD_ROWS_SHA256 = "b75741cc514bbc3711e04232e8462f16a3181ed5f0bd754ebecd05b9ba9b0f71"
OOD_TARGET_SHA256 = {
    "in_box": "a66153139f71c0f3fb1355d99491f4b6d056adf80ad84df1da6760abd579ecb3",
    "plus_05cm": "48e0a717c00154649273469ab975f679142a4f0606ebd7a9c67631189f223fdc",
    "fallback_max": "6d0a0bffe0eb8224d52e074951f11267e00439a81d55760ca2ab88ed1dcb8c4f",
}
LONG_ROWS_SHA256 = "0cd9a6fd177d40f62c5d06d5632454f9cad4aeef357158dd393777db505a78ce"
LONG_GOALS_SHA256 = "c1002a0c8295ca200302bf430755b02f267a34059ad861cf47f392b10395ceaa"


@dataclass(frozen=True)
class ArmSpec:
    name: str
    scenario: str
    mode: str
    budget: int
    spacing_m: float | None = None
    tier: str | None = None


ARMS: dict[str, ArmSpec] = {}
for _tier in ("in_box", "plus_05cm", "fallback_max"):
    # The paired direct probe arm already exists and is referenced byte-for-byte;
    # do not spend simulator budget rerunning or overwrite it.
    _name = f"ood_{_tier}_waypoint_4cm"
    ARMS[_name] = ArmSpec(_name, "ood", "waypoint_4cm", 50, 0.04, _tier)
for _mode in ("direct", "waypoint_4cm"):
    _spacing = None if _mode == "direct" else 0.04
    _name = f"long_offset100_{_mode}"
    ARMS[_name] = ArmSpec(_name, "long", _mode, 200, _spacing)
    _name = f"red_offset25_{_mode}"
    ARMS[_name] = ArmSpec(_name, "red", _mode, 50, _spacing)
for _label, _spacing in (("2p5cm", 0.025), ("6cm", 0.06)):
    _name = f"ood_in_box_waypoint_{_label}"
    ARMS[_name] = ArmSpec(_name, "ood", f"waypoint_{_label}", 50, _spacing, "in_box")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    return probe_common.file_identity(path.resolve())


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _require_sha(label: str, value: Any, expected: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    actual = _array_sha256(array)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: expected={expected}, actual={actual}")
    return array


def _paired(current: Any, baseline: Any) -> dict[str, Any]:
    left = np.asarray(current, dtype=bool)
    right = np.asarray(baseline, dtype=bool)
    if left.ndim != 1 or left.shape != right.shape or left.size not in (2, 50):
        raise ValueError(f"invalid paired vectors: {left.shape}/{right.shape}")
    return {
        "delta_pp": float((left.mean() - right.mean()) * 100.0),
        "baseline_failure_to_current_success": np.flatnonzero(~right & left),
        "baseline_success_to_current_failure": np.flatnonzero(right & ~left),
    }


def _make_waypoints(start: Any, target: Any, spacing_m: float) -> np.ndarray:
    begin = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(target, dtype=np.float64).reshape(3)
    if not np.isfinite(begin).all() or not np.isfinite(end).all():
        raise ValueError("waypoint endpoints must be finite xyz")
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError(f"invalid waypoint spacing: {spacing_m}")
    delta = end - begin
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return end.reshape(1, 3).copy()
    count = max(1, int(np.ceil(distance / spacing_m)))
    fractions = np.minimum(np.arange(1, count + 1) * spacing_m / distance, 1.0)
    result = begin[None, :] + fractions[:, None] * delta[None, :]
    result[-1] = end
    return result


LIVE_XYZ_ALIASES: dict[str, tuple[str, ...]] = {
    "privileged_block_0_pos": (
        "privileged/block_0_pos",
        "privileged_block_0_pos",
    ),
    "goal_privileged_block_0_pos": (
        "goal_privileged/block_0_pos",
        "goal_privileged_block_0_pos",
    ),
}


def _latest_xyz(info: Mapping[str, Any], key: str, env_idx: int) -> np.ndarray:
    aliases = LIVE_XYZ_ALIASES.get(key, (key,))
    present = [alias for alias in aliases if alias in info]
    if not present:
        raise KeyError(f"missing live xyz keys: canonical={key}, aliases={aliases}")
    values = [np.asarray(info[alias][env_idx]) for alias in present]
    reference = values[0]
    for alias, value in zip(present[1:], values[1:], strict=True):
        if reference.shape != value.shape or not np.array_equal(reference, value):
            raise ValueError(
                "conflicting live xyz aliases: "
                f"canonical={key}, first={present[0]} shape={reference.shape}, "
                f"other={alias} shape={value.shape}"
            )
    value = reference
    while value.ndim > 1:
        value = value[-1]
    result = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite {key} for env {env_idx}")
    return result


def _assign_xyz(info: dict[str, Any], env_idx: int, xyz: Any) -> None:
    canonical = "goal_privileged_block_0_pos"
    if canonical not in info:
        current_aliases = LIVE_XYZ_ALIASES["privileged_block_0_pos"]
        present = [alias for alias in current_aliases if alias in info]
        if not present:
            raise KeyError(
                "cannot create planner goal xyz without live block position: "
                f"aliases={current_aliases}"
            )
        current = np.asarray(info[present[0]])
        for alias in present[1:]:
            other = np.asarray(info[alias])
            if current.shape != other.shape or not np.array_equal(current, other):
                raise ValueError(
                    "conflicting live block-position aliases while creating planner goal"
                )
        if current.ndim < 2 or current.shape[-1] != 3:
            raise ValueError(f"invalid live block-position slot shape: {current.shape}")
        info[canonical] = np.zeros_like(current)
    slot = np.asarray(info[canonical])
    if slot.ndim < 2 or slot.shape[-1] != 3 or not 0 <= env_idx < slot.shape[0]:
        raise ValueError(
            f"invalid planner goal xyz slot: env={env_idx}, shape={slot.shape}"
        )
    value = np.asarray(xyz, dtype=slot.dtype).reshape(3)
    if not np.isfinite(value).all():
        raise ValueError("planner goal xyz must be finite")
    info[canonical][env_idx] = np.broadcast_to(value, slot[env_idx].shape)


def _ensure_zero_goal(info: dict[str, Any]) -> None:
    """Recreate the transient goal slot and enforce zero pixels every step."""

    if "pixels" not in info:
        raise KeyError("cannot create zero goal without live pixels")
    pixels = info["pixels"]
    if "goal" not in info:
        try:
            import torch

            info["goal"] = torch.zeros_like(pixels) if torch.is_tensor(pixels) else np.zeros_like(pixels)
        except ImportError:
            info["goal"] = np.zeros_like(pixels)
    goal = info["goal"]
    if tuple(goal.shape) != tuple(pixels.shape):
        raise ValueError(
            f"zero-goal/pixels shape mismatch: goal={goal.shape}, pixels={pixels.shape}"
        )
    if hasattr(goal, "zero_"):
        goal.zero_()
    elif hasattr(goal, "fill"):
        goal.fill(0)
    else:
        raise TypeError(f"unsupported goal slot type: {type(goal).__name__}")


def _flush_plan(policy: Any, env_idx: int) -> None:
    if policy._action_buffer is None:
        raise RuntimeError("policy action buffer is not configured")
    policy._action_buffer[env_idx].clear()
    if policy._next_init is not None:
        policy._next_init[env_idx] = 0


class ProbeCoordinateModel:
    """XYZ planner cost plus an exact returned-plan prediction hook."""

    def __init__(self, base: Any, probe: xyz_common.LoadedXYZProbe) -> None:
        import torch

        class Module(torch.nn.Module):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.base = base
                inner_self.probe_module = probe.model

            @torch.inference_mode()
            def get_cost(inner_self, info_dict: dict[str, Any], actions: Any) -> Any:
                predicted = inner_self.predict_terminal_xyz(info_dict, actions)
                goal_xyz = probe_goal._goal_value(
                    info_dict["goal_privileged_block_0_pos"], predicted, 3
                )
                return probe_common.probe_physical_cost(predicted, goal_xyz)

            @torch.inference_mode()
            def predict_terminal_xyz(inner_self, info_dict: dict[str, Any], actions: Any) -> Any:
                device = next(inner_self.base.parameters()).device
                rollout_info = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in info_dict.items()
                    if key != "goal"
                }
                rolled = inner_self.base.rollout(rollout_info, actions.to(device))
                terminal = rolled["predicted_emb"][..., -1, :]
                return probe(terminal)

        self.module = Module()


class AuditedTrustProxy(trust.TrustRegionCostProxy):
    """Keep solve contexts until the exact returned-plan rollout is recorded."""

    def __init__(self, base: Any, coordinate_model: ProbeCoordinateModel) -> None:
        super().__init__(base, "t2")
        self.coordinate_model = coordinate_model
        self.audit_contexts: list[dict[str, Any]] | None = None
        self.rollout_audits: list[dict[str, Any]] = []

    def begin_solve(self, contexts: list[dict[str, Any]]) -> None:
        if self.audit_contexts is not None:
            raise RuntimeError("previous exact-plan audit was not consumed")
        self.audit_contexts = [dict(item) for item in contexts]
        super().begin_solve(contexts)

    def record_returned_plan_predictions(
        self, info_dict: dict[str, Any], returned_actions: Any
    ) -> None:
        import torch

        if self.audit_contexts is None:
            raise RuntimeError("solver returned actions without audit contexts")
        if len(self.audit_contexts) != int(returned_actions.shape[0]):
            raise RuntimeError("returned action count differs from audit contexts")
        device = next(self.base.parameters()).device
        actions = returned_actions.to(device).unsqueeze(1)
        # The policy passes prepared but unexpanded (B,T,...) infos to the
        # solver.  Reproduce the solver's sample dimension exactly for the
        # single returned plan per environment: (B,1,T,...).
        expanded_info: dict[str, Any] = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                expanded_info[key] = value.to(device).unsqueeze(1)
            elif isinstance(value, np.ndarray):
                expanded_info[key] = value[:, None, ...]
            else:
                expanded_info[key] = value
        with torch.inference_mode():
            predicted = self.coordinate_model.module.predict_terminal_xyz(
                expanded_info, actions
            )
        values = predicted[:, 0].detach().cpu().float().numpy()
        for context, xyz in zip(self.audit_contexts, values, strict=True):
            self.rollout_audits.append(
                {
                    "env_idx": int(context["env_idx"]),
                    "planning_cycle": int(context["planning_cycle"]),
                    "env_step": int(context["env_step"]),
                    "predicted_terminal_xyz": np.asarray(xyz, dtype=np.float64),
                    "executed_plan_horizon_env_steps": ROLLOUT_ENV_STEPS,
                    "alignment": None,
                }
            )
        self.audit_contexts = None


def _make_audited_solver(base_solver: type, proxy: AuditedTrustProxy) -> type:
    class AuditedSolver(base_solver):
        def solve(self, info_dict: dict[str, Any], init_action: Any = None) -> dict[str, Any]:
            outputs = super().solve(info_dict, init_action=init_action)
            proxy.record_returned_plan_predictions(info_dict, outputs["actions"])
            return outputs

    AuditedSolver.__name__ = "WaypointAuditedTrustRegionCEMSolver"
    return AuditedSolver


def _frozen_scenario(
    spec: ArmSpec,
    dataset: Any,
    formal_rows: np.ndarray,
) -> dict[str, Any]:
    if spec.scenario == "ood":
        assert spec.tier is not None
        frozen = probe_goal._load_frozen_targets(GOAL_OOD_REFERENCE, spec.tier)
        starts = np.asarray(frozen["start_rows"], dtype=np.int64)
        targets = _require_sha(
            f"{spec.tier} target rows", frozen["target_rows"], OOD_TARGET_SHA256[spec.tier]
        )
        if not np.array_equal(starts, formal_rows):
            raise ValueError(f"{spec.tier} OOD starts differ from frozen standard rows")
        direct_path = PROBE_DIRECT_REFERENCE / spec.tier / "probe" / "results.json"
        direct = _read_json(direct_path)
        if not np.array_equal(np.asarray(direct["evaluated_rows"], dtype=np.int64), starts):
            raise ValueError(f"{spec.tier} direct reference start rows changed")
        if not np.array_equal(np.asarray(direct["target_rows"], dtype=np.int64), targets):
            raise ValueError(f"{spec.tier} direct reference target rows changed")
        return {
            "rows": starts,
            "target_rows": targets,
            "budget": 50,
            "reference": _identity(direct_path),
            "reference_successes": np.asarray(direct["metrics"]["episode_successes"], dtype=bool),
            "target_selection": frozen["selection_identity"],
        }

    if spec.scenario == "long":
        manifest_path = LONG_REFERENCE / "run_manifest.json"
        results_path = LONG_REFERENCE / "results.json"
        manifest = _read_json(manifest_path)
        results = _read_json(results_path)
        selection = manifest["selection"]
        starts = _require_sha("offset100 rows", selection["rows"], LONG_ROWS_SHA256)
        targets = _require_sha("offset100 goal rows", selection["goal_rows"], LONG_GOALS_SHA256)
        if not np.array_equal(targets - starts, np.full(50, 100, dtype=np.int64)):
            raise ValueError("offset100 target-row delta is not exactly 100")
        if not np.array_equal(np.asarray(results["evaluated_rows"], dtype=np.int64), starts):
            raise ValueError("offset100 results rows differ from selection")
        return {
            "rows": starts,
            "target_rows": targets,
            "budget": 200,
            "reference": _identity(results_path),
            "reference_successes": np.asarray(results["metrics"]["episode_successes"], dtype=bool),
            "selection_manifest": _identity(manifest_path),
        }

    if spec.scenario == "red":
        reference = _read_json(RED_REFERENCE)
        starts = _require_sha("robust red rows", reference["evaluated_rows"], STANDARD_ROWS_SHA256)
        if not np.array_equal(starts, formal_rows):
            raise ValueError("robust red rows differ from frozen standard rows")
        targets = starts + 25
        rows_state = goal_ood._row_states(dataset, starts)
        target_state = goal_ood._row_states(dataset, targets)
        if not np.array_equal(rows_state["ep_idx"], target_state["ep_idx"]):
            raise ValueError("one or more red +25 targets cross episodes")
        return {
            "rows": starts,
            "target_rows": targets,
            "budget": 50,
            "reference": _identity(RED_REFERENCE),
            "reference_successes": np.asarray(reference["metrics"]["episode_successes"], dtype=bool),
        }
    raise ValueError(spec.scenario)


def _prepare_output(path: Path, overwrite: bool) -> Path:
    resolved = probe_common.ensure_output_child(path, DEFAULT_OUTPUT, "waypoint arm output")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"non-empty output: {resolved}; pass --overwrite")
        if resolved.is_symlink():
            raise ValueError(f"refusing symlink output: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _arm_output(root: Path, arm: str, formal: bool) -> Path:
    return root / arm if formal else root / "smoke" / arm


def _start_segment(
    controller: dict[str, Any], step: int, position: np.ndarray, index: int
) -> dict[str, Any]:
    target = np.asarray(controller["waypoints"][index], dtype=np.float64)
    previous = (
        np.asarray(controller["initial_position"], dtype=np.float64)
        if index == 0
        else np.asarray(controller["waypoints"][index - 1], dtype=np.float64)
    )
    record = {
        "env_idx": controller["env_idx"],
        "segment_index": index,
        "start_step": int(step),
        "start_position": np.asarray(position, dtype=np.float64),
        "target_xyz": target,
        "nominal_segment_length_m": float(np.linalg.norm(target - previous)),
        "initial_physical_distance_m": float(np.linalg.norm(target - position)),
        "end_step": None,
        "steps": None,
        "end_position": None,
        "end_distance_m": None,
        "status": "active",
        "arrived": False,
        "timed_out": False,
        "stalled_events": 0,
        "stalled_satisfied_steps": 0,
        "planner_cost_trend": None,
    }
    controller["segments"].append(record)
    controller["active_segment"] = record
    controller["active_index"] = index
    controller["distance_window"] = deque([(int(step), record["initial_physical_distance_m"])], maxlen=9)
    controller["stalled_latched"] = False
    return record


def _finish_segment(
    controller: dict[str, Any], step: int, position: np.ndarray, status: str
) -> None:
    segment = controller.get("active_segment")
    if segment is None or segment["status"] != "active":
        return
    distance = float(np.linalg.norm(np.asarray(segment["target_xyz"]) - position))
    segment.update(
        {
            "end_step": int(step),
            "steps": int(step - segment["start_step"]),
            "end_position": np.asarray(position, dtype=np.float64),
            "end_distance_m": distance,
            "status": status,
            "arrived": bool(status in {
                "arrived",
                "episode_final_success_at_active_target",
                "forced_smoke_advance",
            }),
            "timed_out": bool(status in {"timeout", "forced_smoke_timeout"}),
        }
    )
    controller["active_segment"] = None


def _record_stall(
    controller: dict[str, Any], step: int, distance: float
) -> None:
    window: deque[tuple[int, float]] = controller["distance_window"]
    window.append((int(step), float(distance)))
    if len(window) < 9 or window[-1][0] - window[0][0] != STALL_WINDOW_STEPS:
        return
    progress = float(window[0][1] - window[-1][1])
    stalled = progress < STALL_PROGRESS_M
    if stalled:
        controller["stalled_satisfied_steps"] += 1
        segment = controller.get("active_segment")
        if segment is not None:
            segment["stalled_satisfied_steps"] += 1
        if not controller["stalled_latched"]:
            controller["stalled_events"] += 1
            if segment is not None:
                segment["stalled_events"] += 1
            controller["stall_events"].append(
                {
                    "env_idx": controller["env_idx"],
                    "step": int(step),
                    "segment_index": controller.get("active_index"),
                    "progress_over_8_steps_m": progress,
                    "threshold_m_strict": STALL_PROGRESS_M,
                }
            )
    controller["stalled_latched"] = stalled


def _switch_target(
    *,
    controller: dict[str, Any],
    policy: Any,
    recorder: Any,
    step: int,
    target: np.ndarray,
    reason: str,
) -> None:
    env_idx = int(controller["env_idx"])
    before = len(recorder.records[env_idx])
    _flush_plan(policy, env_idx)
    event = {
        "env_idx": env_idx,
        "step": int(step),
        "reason": reason,
        "target_xyz": np.asarray(target, dtype=np.float64),
        "action_buffer_flushed": True,
        "next_init_cleared_if_present": True,
        "trust_cycles_before_flush": before,
        "cost_recorder_cycles_before_flush": before,
        "expected_replan_env_step": int(step),
        "replan_observed_at_step": None,
        "replan_trust_cycle_delta": None,
        "replan_cost_recorder_cycle_delta": None,
        "replan_context_env_step": None,
    }
    controller["switches"].append(event)


def _observe_replans(
    controllers: list[dict[str, Any]], recorder: Any, proxy: AuditedTrustProxy, step: int
) -> None:
    contexts_by_env = defaultdict(list)
    for audit in proxy.rollout_audits:
        contexts_by_env[int(audit["env_idx"])].append(audit)
    for controller in controllers:
        env_idx = int(controller["env_idx"])
        current = len(recorder.records[env_idx])
        for event in controller["switches"]:
            if event["replan_observed_at_step"] is not None:
                continue
            before = int(event["cost_recorder_cycles_before_flush"])
            if current <= before:
                continue
            candidates = [
                item for item in contexts_by_env[env_idx]
                if int(item["planning_cycle"]) >= before
            ]
            event["replan_observed_at_step"] = int(step)
            event["replan_cost_recorder_cycle_delta"] = int(current - before)
            event["replan_trust_cycle_delta"] = int(len(candidates))
            event["replan_context_env_step"] = (
                int(candidates[0]["env_step"]) if candidates else None
            )


def _align_imagination(
    audits: list[dict[str, Any]],
    positions: list[dict[int, np.ndarray]],
    interruptions: list[list[dict[str, Any]]],
    terminations: list[int | None],
) -> dict[str, Any]:
    errors = []
    for audit in audits:
        env_idx = int(audit["env_idx"])
        start = int(audit["env_step"])
        end = start + ROLLOUT_ENV_STEPS
        reason = None
        matching_interruptions = [
            item for item in interruptions[env_idx]
            if start < int(item["step"]) < end
        ]
        if matching_interruptions:
            reason = "waypoint_switch_or_fallback_before_25_actions"
        elif terminations[env_idx] is not None and int(terminations[env_idx]) < end:
            reason = "environment_terminated_before_25_actions"
        elif end not in positions[env_idx]:
            reason = "budget_ended_before_25_actions"
        if reason is not None:
            audit["alignment"] = {"status": "censored", "reason": reason}
            continue
        physical = np.asarray(positions[env_idx][end], dtype=np.float64)
        predicted = np.asarray(audit["predicted_terminal_xyz"], dtype=np.float64)
        error_mm = float(np.linalg.norm(predicted - physical) * 1000.0)
        errors.append(error_mm)
        audit["alignment"] = {
            "status": "aligned",
            "physical_env_step": end,
            "physical_xyz": physical,
            "xyz_error_mm": error_mm,
        }
    array = np.asarray(errors, dtype=np.float64)
    return {
        "definition": (
            "probe xyz of exact solver-returned normalized 5x25 plan versus physical xyz "
            "after exactly 25 uninterrupted env actions"
        ),
        "total_solves": len(audits),
        "aligned_solves": int(len(array)),
        "censored_solves": int(len(audits) - len(array)),
        "xyz_error_mm": None if not len(array) else {
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "p90": float(np.percentile(array, 90)),
        },
    }


def _attach_cost_trends(controllers: list[dict[str, Any]], recorder: Any) -> None:
    for controller in controllers:
        cycles = recorder.records[int(controller["env_idx"])]
        for segment in controller["segments"]:
            end = segment["end_step"] if segment["end_step"] is not None else np.inf
            selected = [
                float(cycle["final_top1_cost"])
                for cycle in cycles
                if int(segment["start_step"]) <= int(cycle["env_step"]) < end
            ]
            segment["planner_cost_trend"] = (
                None
                if not selected
                else {
                    "count": len(selected),
                    "first": selected[0],
                    "last": selected[-1],
                    "delta": selected[-1] - selected[0],
                    "fractional_change": (
                        None if abs(selected[0]) <= 1e-12 else selected[-1] / selected[0] - 1.0
                    ),
                }
            )


def _controller_summary(controllers: list[dict[str, Any]]) -> dict[str, Any]:
    segments = [segment for item in controllers for segment in item["segments"]]
    waypoint_segments = [segment for segment in segments if segment["kind"] == "waypoint"]
    arrived = [segment for segment in waypoint_segments if segment["arrived"]]
    timed_out = [segment for segment in waypoint_segments if segment["timed_out"]]
    arrival_steps = [int(segment["steps"]) for segment in arrived]
    fallback_envs = [int(item["env_idx"]) for item in controllers if item["fallback"]]
    reached_by_ordinal: dict[str, Any] = {}
    for ordinal in sorted({int(item["segment_index"]) for item in waypoint_segments}):
        group = [item for item in waypoint_segments if int(item["segment_index"]) == ordinal]
        reached_by_ordinal[str(ordinal)] = {
            "attempts": len(group),
            "arrived": sum(bool(item["arrived"]) for item in group),
            "timeouts": sum(bool(item["timed_out"]) for item in group),
        }
    failures = []
    for item in controllers:
        if item["success"]:
            continue
        planned = [segment for segment in item["segments"] if segment["kind"] == "waypoint"]
        timeout = next((segment for segment in planned if segment["timed_out"]), None)
        not_arrived = [segment for segment in planned if not segment["arrived"]]
        if timeout is not None:
            break_segment = timeout
            break_status = "timeout"
        elif not_arrived:
            break_segment = not_arrived[-1]
            break_status = str(break_segment["status"])
        elif planned:
            break_segment = planned[-1]
            break_status = "final_hold_not_reached"
        else:
            # Direct controls have no waypoint breakpoint.
            break_segment = None
            break_status = "direct_final_not_reached"
        failures.append(
            {
                "env_idx": item["env_idx"],
                "break_segment_index": (
                    None if break_segment is None else break_segment["segment_index"]
                ),
                "break_status": break_status,
                "fallback": item["fallback"],
            }
        )
    return {
        "waypoint_segments_attempted": len(waypoint_segments),
        "waypoint_segments_arrived": len(arrived),
        "waypoint_segment_arrival_rate": (
            None if not waypoint_segments else len(arrived) / len(waypoint_segments)
        ),
        "waypoint_segment_arrival_steps": arrival_steps,
        "waypoint_segment_arrival_steps_median": (
            None if not arrival_steps else float(np.median(arrival_steps))
        ),
        "waypoint_segments_timed_out": len(timed_out),
        "by_segment_ordinal": reached_by_ordinal,
        "fallback_episode_count": len(fallback_envs),
        "fallback_episode_rate": len(fallback_envs) / len(controllers),
        "fallback_envs": fallback_envs,
        "stalled_event_count": sum(int(item["stalled_events"]) for item in controllers),
        "stalled_satisfied_step_count": sum(
            int(item["stalled_satisfied_steps"]) for item in controllers
        ),
        "failed_episode_breakpoints": failures,
    }


CONTROLLER_RUNTIME_FIELDS = frozenset(
    {"active_segment", "distance_window", "stalled_latched"}
)


def _controller_artifacts(
    controllers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy controllers without mutable runtime-only state.

    ``distance_window`` is a deque used only by the online STALLED detector;
    ``active_segment`` aliases an entry already retained under ``segments``;
    ``stalled_latched`` is transient edge-detection state.  Dropping these
    fields leaves every statistic and event intact while making the artifact
    contract independent of Python runtime container types.
    """

    artifacts = []
    for controller in controllers:
        artifact = {
            key: value
            for key, value in controller.items()
            if key not in CONTROLLER_RUNTIME_FIELDS
        }
        artifact["runtime_fields_omitted"] = sorted(CONTROLLER_RUNTIME_FIELDS)
        artifacts.append(artifact)
    return artifacts


def _validate_exact_json(label: str, value: Any) -> None:
    try:
        json.dumps(
            probe_common.jsonable(value),
            allow_nan=False,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} contains a non-JSON artifact value") from error


def _run_arm(
    *,
    args: argparse.Namespace,
    spec: ArmSpec,
    output: Path,
    dataset: Any,
    formal_rows: np.ndarray,
    index: Any,
    long_index: Any,
    scaler: Any,
    base_model: Any,
    loaded_probe: xyz_common.LoadedXYZProbe,
    probe_contract: dict[str, Any],
    robust_contract: dict[str, Any],
) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401
    import h5py
    import stable_worldmodel as swm

    frozen = _frozen_scenario(spec, dataset, formal_rows)
    rows = np.asarray(frozen["rows"], dtype=np.int64)[: args.num_eval]
    target_rows = np.asarray(frozen["target_rows"], dtype=np.int64)[: args.num_eval]
    reference_successes = np.asarray(frozen["reference_successes"], dtype=bool)[: args.num_eval]
    init_state = goal_ood._row_states(dataset, rows)
    target_raw = goal_ood._row_states(dataset, target_rows)
    goal_state = goal_ood._make_goal_state(target_raw)
    episodes = np.asarray(init_state["ep_idx"], dtype=np.int64)
    starts = np.asarray(init_state["step_idx"], dtype=np.int64)
    final_xyz = np.asarray(goal_state["goal_privileged_block_0_pos"], dtype=np.float64)
    initial_xyz = np.asarray(init_state["privileged_block_0_pos"], dtype=np.float64)
    if rows.shape != target_rows.shape or final_xyz.shape != (args.num_eval, 3):
        raise ValueError("malformed frozen scenario rows/targets")

    with h5py.File(args.dataset, "r", swmr=True) as h5:
        initial_query_features = np.concatenate(
            [memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows], axis=0
        )

    recorder = ood.PlanningCostRecorder(args.num_eval)
    coordinate = ProbeCoordinateModel(base_model, loaded_probe)
    planner_model = coordinate.module.to(args.device).eval().requires_grad_(False)
    proxy = AuditedTrustProxy(planner_model, coordinate)
    base_solver = trust.make_trust_region_solver(swm.solver.CEMSolver)
    solver_cls = _make_audited_solver(base_solver, proxy)
    solver = solver_cls(
        model=proxy,
        batch_size=1,
        num_samples=t2common.NUM_SAMPLES,
        var_scale=t2common.PROTOCOL_SPECS["t2"]["var_scale"],
        n_steps=t2common.N_STEPS,
        topk=t2common.TOPK,
        device=args.device,
        seed=FORMAL_SEED,
        callbacks=[recorder],
        selector="mean",
        recorder=recorder,
        trust_protocol="t2",
    )
    planner_index = long_index if spec.scenario == "long" else index
    policy = trust.make_trust_policy(swm.policy.WorldModelPolicy)(
        solver=solver,
        config=swm.PlanConfig(
            horizon=t2common.HORIZON,
            receding_horizon=t2common.HORIZON,
            action_block=t2common.ACTION_BLOCK,
        ),
        process={"action": scaler},
        transform={"pixels": ood._image_transform(224), "goal": ood._image_transform(224)},
        memory_index=planner_index,
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
        max_episode_steps=2 * spec.budget,
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
    world.reset(seed=init_state.get("seed"), options=None)
    merged = {**init_state, **goal_state}
    callables = [
        {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
        {"method": "set_target_pos", "args": {
            "cube_id": {"value": 0, "in_dataset": False},
            "target_pos": {"value": "goal_privileged_block_0_pos"},
            "target_quat": {"value": "goal_privileged_block_0_quat"},
        }},
    ]
    for env_idx, wrapped in enumerate(world.envs.envs):
        ood._apply_callables(
            wrapped.unwrapped, callables, {key: value[env_idx] for key, value in merged.items()}
        )
    init_state["pixels"] = np.asarray(init_state["pixels"], dtype=np.uint8).copy()
    zero_goal = np.zeros_like(np.asarray(goal_state["goal"], dtype=np.uint8))
    goal_state["goal"] = zero_goal
    shape_prefix = world.infos["pixels"].shape[:2]
    for state in (init_state, goal_state):
        for key, value in state.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()

    controllers: list[dict[str, Any]] = []
    for env_idx in range(args.num_eval):
        waypoint_mode = spec.spacing_m is not None
        waypoints = (
            _make_waypoints(initial_xyz[env_idx], final_xyz[env_idx], float(spec.spacing_m))
            if waypoint_mode
            else final_xyz[env_idx].reshape(1, 3).copy()
        )
        controller: dict[str, Any] = {
            "env_idx": env_idx,
            "mode": spec.mode,
            "waypoint_mode": waypoint_mode,
            "spacing_m": spec.spacing_m,
            "initial_position": initial_xyz[env_idx],
            "final_xyz": final_xyz[env_idx],
            "waypoints": waypoints,
            "active_index": 0,
            "active_target": waypoints[0].copy(),
            "active_segment": None,
            "final_hold": False,
            "fallback": False,
            "fallback_step": None,
            "fallback_from_segment": None,
            "segments": [],
            "switches": [],
            "stall_events": [],
            "stalled_events": 0,
            "stalled_satisfied_steps": 0,
            "distance_window": deque(maxlen=9),
            "stalled_latched": False,
            "success": False,
        }
        segment = _start_segment(controller, 0, initial_xyz[env_idx], 0)
        segment["kind"] = "waypoint" if waypoint_mode else "direct"
        controllers.append(controller)
        _assign_xyz(world.infos, env_idx, controller["active_target"])
    _ensure_zero_goal(world.infos)

    successes = np.zeros(args.num_eval, dtype=bool)
    positions: list[dict[int, np.ndarray]] = [
        {0: initial_xyz[i].copy()} for i in range(args.num_eval)
    ]
    terminations: list[int | None] = [None] * args.num_eval
    interruptions: list[list[dict[str, Any]]] = [[] for _ in range(args.num_eval)]
    frames: dict[int, list[np.ndarray]] | None = (
        defaultdict(list) if args.video else None
    )
    step_counter = 0

    def on_step(active_world: Any) -> None:
        nonlocal step_counter
        step_counter += 1
        # World rebuilds live infos from the environment every step, so the
        # planner-only goal fields must be reinstalled before the next policy
        # call.  The image slot is always a zero tensor and the cost wrapper
        # removes it again before world-model rollout.
        _ensure_zero_goal(active_world.infos)
        terminated_now = np.asarray(active_world.terminateds, dtype=bool)
        successes[:] |= terminated_now
        _observe_replans(controllers, recorder, proxy, step_counter)
        for env_idx, controller in enumerate(controllers):
            position = _latest_xyz(active_world.infos, "privileged_block_0_pos", env_idx)
            positions[env_idx][step_counter] = position.copy()
            if frames is not None:
                pixels = np.asarray(active_world.infos["pixels"][env_idx])
                frames[env_idx].append((pixels[-1] if pixels.ndim > 3 else pixels).copy())
            if terminated_now[env_idx]:
                if terminations[env_idx] is None:
                    terminations[env_idx] = step_counter
                    controller["success"] = True
                    active = controller.get("active_segment")
                    active_distance = (
                        np.inf
                        if active is None
                        else float(np.linalg.norm(position - np.asarray(active["target_xyz"])))
                    )
                    status = (
                        "episode_final_success_at_active_target"
                        if active_distance <= ARRIVAL_TOLERANCE_M
                        else "episode_final_success_before_active_waypoint"
                    )
                    _finish_segment(controller, step_counter, position, status)
                continue

            target = np.asarray(controller["active_target"], dtype=np.float64)
            distance = float(np.linalg.norm(position - target))
            _record_stall(controller, step_counter, distance)
            if controller["waypoint_mode"] and controller["active_segment"] is not None:
                segment = controller["active_segment"]
                forced_advance = bool(
                    args.exercise_smoke_branches and env_idx == 0 and step_counter == 2
                )
                forced_timeout = bool(
                    args.exercise_smoke_branches and env_idx == 1 and step_counter == 2
                )
                arrived = distance <= ARRIVAL_TOLERANCE_M or forced_advance
                timed_out = (
                    step_counter - int(segment["start_step"]) >= SEGMENT_TIMEOUT_STEPS
                    or forced_timeout
                )
                if arrived:
                    status = "forced_smoke_advance" if forced_advance else "arrived"
                    _finish_segment(controller, step_counter, position, status)
                    current_index = int(controller["active_index"])
                    if current_index + 1 < len(controller["waypoints"]):
                        next_index = current_index + 1
                        next_segment = _start_segment(controller, step_counter, position, next_index)
                        next_segment["kind"] = "waypoint"
                        controller["active_target"] = np.asarray(
                            controller["waypoints"][next_index], dtype=np.float64
                        ).copy()
                        reason = "forced_smoke_waypoint_advance" if forced_advance else "waypoint_arrived"
                    else:
                        controller["active_index"] = None
                        controller["active_target"] = np.asarray(controller["final_xyz"]).copy()
                        controller["final_hold"] = True
                        controller["distance_window"] = deque(
                            [(step_counter, float(np.linalg.norm(position - controller["final_xyz"])))],
                            maxlen=9,
                        )
                        controller["stalled_latched"] = False
                        reason = "all_waypoints_arrived_final_hold"
                    _switch_target(
                        controller=controller,
                        policy=policy,
                        recorder=recorder,
                        step=step_counter,
                        target=np.asarray(controller["active_target"]),
                        reason=reason,
                    )
                    interruptions[env_idx].append({"step": step_counter, "reason": reason})
                elif timed_out:
                    status = "forced_smoke_timeout" if forced_timeout else "timeout"
                    failed_index = int(controller["active_index"])
                    _finish_segment(controller, step_counter, position, status)
                    controller["fallback"] = True
                    controller["fallback_step"] = step_counter
                    controller["fallback_from_segment"] = failed_index
                    controller["active_index"] = None
                    controller["active_target"] = np.asarray(controller["final_xyz"]).copy()
                    controller["distance_window"] = deque(
                        [(step_counter, float(np.linalg.norm(position - controller["final_xyz"])))],
                        maxlen=9,
                    )
                    controller["stalled_latched"] = False
                    reason = "forced_smoke_timeout_to_direct_final" if forced_timeout else "segment_timeout_to_direct_final"
                    _switch_target(
                        controller=controller,
                        policy=policy,
                        recorder=recorder,
                        step=step_counter,
                        target=np.asarray(controller["active_target"]),
                        reason=reason,
                    )
                    interruptions[env_idx].append({"step": step_counter, "reason": reason})
            _assign_xyz(active_world.infos, env_idx, controller["active_target"])

    started = time.time()
    try:
        world._run(max_steps=spec.budget, mode="wait", on_step=on_step)
    finally:
        world.close()
    elapsed = time.time() - started
    _observe_replans(controllers, recorder, proxy, step_counter)
    for env_idx, controller in enumerate(controllers):
        controller["success"] = bool(successes[env_idx])
        if controller["active_segment"] is not None:
            final_position = positions[env_idx][max(positions[env_idx])]
            status = "terminated" if terminations[env_idx] is not None else "budget_exhausted"
            _finish_segment(controller, step_counter, final_position, status)

    _attach_cost_trends(controllers, recorder)
    imagination = _align_imagination(
        proxy.rollout_audits, positions, interruptions, terminations
    )
    trace = trust._save_trace(output, proxy)
    cost_history = ood._save_cost_history(
        output, recorder, rows, episodes, starts, "mean"
    )
    if frames is not None:
        from stable_worldmodel.plot import save_panel_videos

        save_panel_videos(
            output / "videos", {"agent": frames, "goal_input_zero": zero_goal}
        )

    controller_summary = _controller_summary(controllers)
    controller_artifacts = _controller_artifacts(controllers)
    metrics = {
        "success_rate": float(successes.mean() * 100.0),
        "success_count": int(successes.sum()),
        "num_eval": args.num_eval,
        "episode_successes": successes,
    }
    smoke_checks = None
    if args.num_eval == 2:
        switches = [event for item in controllers for event in item["switches"]]
        smoke_checks = {
            "waypoint_switch_observed": any("advance" in event["reason"] or "arrived" in event["reason"] for event in switches),
            "fallback_observed": any(item["fallback"] for item in controllers),
            "all_switches_replanned": all(
                event["replan_observed_at_step"] is not None
                and int(event["replan_trust_cycle_delta"] or 0) >= 1
                and int(event["replan_cost_recorder_cycle_delta"] or 0) >= 1
                and int(event["replan_context_env_step"]) == int(event["expected_replan_env_step"])
                for event in switches
            ),
            "segment_logs_complete": all(
                segment["end_step"] is not None and segment["steps"] is not None
                for item in controllers for segment in item["segments"]
            ),
            "goal_input_all_zero": True,
            "exercise_smoke_branches": args.exercise_smoke_branches,
        }
        if args.exercise_smoke_branches and spec.spacing_m is not None and not all(
            smoke_checks[key]
            for key in ("waypoint_switch_observed", "fallback_observed", "all_switches_replanned", "segment_logs_complete")
        ):
            raise RuntimeError(f"exercised smoke branch check failed: {smoke_checks}")

    payload = {
        "format_version": "cube_waypoint_probe_eval_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": spec.name,
        "scenario": spec.scenario,
        "mode": spec.mode,
        "waypoint_spacing_m": spec.spacing_m,
        "protocol": {
            "id": "t2",
            "seed": FORMAL_SEED,
            "budget": spec.budget,
            "num_samples": t2common.NUM_SAMPLES,
            "iterations": t2common.N_STEPS,
            "topk": t2common.TOPK,
            "horizon_model_steps": t2common.HORIZON,
            "action_block_env_steps": t2common.ACTION_BLOCK,
            "arrival_tolerance_m": ARRIVAL_TOLERANCE_M,
            "segment_timeout_env_steps": SEGMENT_TIMEOUT_STEPS,
            "timeout_behavior": "immediate direct-final fallback; no later waypoint",
            "stalled_definition": (
                "physical distance to active coordinate target; consecutive 9 frames/8 steps "
                "with total progress strictly <1cm; false-to-true events and all satisfying steps recorded"
            ),
            "cost": "squared xyz distance: probe(predicted terminal embedding) to active coordinate",
            "goal_image_contract": "zero tensor in policy info and removed before every world-model rollout",
            "physical_target_contract": "set once to final pose; never changed by waypoint tracker",
            "planner_memory_exclusion": (
                "B1 offset100 contract: globally exclude all frozen 50 evaluation episodes"
                if spec.scenario == "long"
                else "existing T2 OOD/red contract: exclude only the current evaluation episode"
            ),
            "probe_training_exclusion": "probe dataset excludes all frozen 50 evaluation episodes",
            "checkpoint": robust_contract,
            "probe": probe_contract,
        },
        "evaluated_rows": rows,
        "target_rows": target_rows,
        "episodes": episodes,
        "starts": starts,
        "final_target_xyz": final_xyz,
        "initial_block_xyz": initial_xyz,
        "frozen_reference": frozen,
        "metrics": metrics,
        "versus_reference": _paired(successes, reference_successes),
        "waypoint_statistics": controller_summary,
        "controllers": controller_artifacts,
        "exact_plan_imagination": {
            "summary": imagination,
            "solves": proxy.rollout_audits,
        },
        "smoke_checks": smoke_checks,
        "elapsed_seconds": elapsed,
        "trace": trace,
        "cost_history": cost_history,
        "trace_cost_field_note": "trust_trace latent_costs contains xyz probe costs",
        "script": _identity(Path(__file__)),
    }
    segments_payload = {"controllers": controller_artifacts}
    imagination_payload = {"summary": imagination, "solves": proxy.rollout_audits}
    _validate_exact_json("segments.json", segments_payload)
    _validate_exact_json("imagination_error.json", imagination_payload)
    _validate_exact_json("results.json", payload)
    probe_common.write_json(output / "segments.json", segments_payload)
    probe_common.write_json(
        output / "imagination_error.json",
        imagination_payload,
    )
    probe_common.write_json(output / "results.json", payload)
    (output / "results.txt").write_text(
        f"arm: {spec.name}\n"
        f"success_count: {metrics['success_count']}/{args.num_eval}\n"
        f"success_rate: {metrics['success_rate']:.2f}\n"
        f"waypoint_arrival_rate: {controller_summary['waypoint_segment_arrival_rate']}\n"
        f"fallback_episodes: {controller_summary['fallback_episode_count']}/{args.num_eval}\n",
        encoding="utf-8",
    )
    print(f"{spec.name}: {metrics['success_count']}/{args.num_eval} ({metrics['success_rate']:.2f}%)")
    return payload


def _write_summary(root: Path, formal: bool) -> None:
    base = root if formal else root / "smoke"
    arms: dict[str, Any] = {}
    for name in ARMS:
        result_path = base / name / "results.json"
        if result_path.is_file():
            value = _read_json(result_path)
            arms[name] = {
                "metrics": value["metrics"],
                "waypoint_statistics": value["waypoint_statistics"],
                "versus_reference": value["versus_reference"],
                "results": _identity(result_path),
            }
    probe_common.write_json(
        base / "summary.json",
        {"format_version": "cube_waypoint_probe_summary_v1", "formal": formal, "arms": arms},
    )


def self_test(args: argparse.Namespace) -> int:
    slash_live = {
        "privileged/block_0_pos": np.asarray([[[0.1, 0.2, 0.3]]], dtype=np.float64)
    }
    underscore_live = {
        "privileged_block_0_pos": np.asarray([[[0.1, 0.2, 0.3]]], dtype=np.float64)
    }
    expected_live = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    if not np.array_equal(_latest_xyz(slash_live, "privileged_block_0_pos", 0), expected_live):
        raise AssertionError("slash live xyz alias failed")
    if not np.array_equal(
        _latest_xyz(underscore_live, "privileged_block_0_pos", 0), expected_live
    ):
        raise AssertionError("underscore live xyz alias failed")
    try:
        _latest_xyz({}, "privileged_block_0_pos", 0)
    except KeyError:
        pass
    else:
        raise AssertionError("missing live xyz aliases must fail closed")
    conflicting_live = {**slash_live, **underscore_live}
    conflicting_live["privileged_block_0_pos"] = np.asarray(
        [[[0.1, 0.2, 0.4]]], dtype=np.float64
    )
    try:
        _latest_xyz(conflicting_live, "privileged_block_0_pos", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting live xyz aliases must fail closed")
    # Real World callbacks can contain only slash-style current state and
    # pixels.  Reinstall both transient planner inputs without relying on a
    # goal field surviving from the prior step.
    transient_step: dict[str, Any] = {
        "privileged/block_0_pos": np.zeros((2, 1, 3), dtype=np.float32),
        "pixels": np.ones((2, 1, 4, 4, 3), dtype=np.uint8),
    }
    _assign_xyz(transient_step, 0, [0.3, -0.1, 0.02])
    _assign_xyz(transient_step, 1, [0.5, 0.2, 0.03])
    injected = transient_step["goal_privileged_block_0_pos"]
    if injected.shape != (2, 1, 3) or injected.dtype != np.float32:
        raise AssertionError("dynamic planner-goal slot did not preserve live shape/dtype")
    if not np.allclose(injected[:, 0], [[0.3, -0.1, 0.02], [0.5, 0.2, 0.03]]):
        raise AssertionError("dynamic planner-goal values were not re-injected")
    _ensure_zero_goal(transient_step)
    if transient_step["goal"].shape != transient_step["pixels"].shape:
        raise AssertionError("zero goal was not recreated from live pixels")
    if np.count_nonzero(transient_step["goal"]):
        raise AssertionError("recreated goal image is not exactly zero")
    transient_step["goal"].fill(255)
    _ensure_zero_goal(transient_step)
    if np.count_nonzero(transient_step["goal"]):
        raise AssertionError("existing goal image was not zeroed")
    direct = _make_waypoints([0, 0, 0], [0.1, 0, 0], 0.04)
    if direct.shape != (3, 3) or not np.allclose(direct[:, 0], [0.04, 0.08, 0.1]):
        raise AssertionError(direct)
    short = _make_waypoints([0, 0, 0], [0.01, 0, 0], 0.04)
    if short.shape != (1, 3) or not np.allclose(short[-1], [0.01, 0, 0]):
        raise AssertionError(short)
    expected_arms = 9
    if len(ARMS) != expected_arms:
        raise AssertionError(f"expected {expected_arms} arms, got {len(ARMS)}")
    runtime_controller = {
        "env_idx": 0,
        "distance_window": deque([(0, 0.1), (1, 0.09)], maxlen=9),
        "active_segment": {"segment_index": 0},
        "stalled_latched": True,
        "segments": [
            {
                "segment_index": np.int64(0),
                "target_xyz": np.asarray([0.3, 0.0, 0.02], dtype=np.float64),
                "status": "budget_exhausted",
            }
        ],
        "switches": [],
        "success": np.bool_(False),
    }
    artifact_controllers = _controller_artifacts([runtime_controller])
    if any(
        field in artifact_controllers[0] for field in CONTROLLER_RUNTIME_FIELDS
    ):
        raise AssertionError("controller runtime fields leaked into artifacts")
    exact_serialization_cases = {
        "controller": artifact_controllers[0],
        "segments": {"controllers": artifact_controllers},
        "results": {
            "controllers": artifact_controllers,
            "waypoint_statistics": {"fallback_episode_count": np.int64(0)},
            "exact_plan_imagination": {
                "solves": [{"predicted_terminal_xyz": np.zeros(3)}]
            },
        },
    }
    for label, value in exact_serialization_cases.items():
        try:
            json.dumps(
                probe_common.jsonable(value),
                allow_nan=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise AssertionError(f"{label} artifact is not exact-JSON serializable") from error
    import torch

    class DummyBase(torch.nn.Module):
        def __init__(inner_self) -> None:
            super().__init__()
            inner_self.anchor = torch.nn.Parameter(torch.zeros(()))
            inner_self.goal_seen = None

        def rollout(inner_self, info: dict[str, Any], actions: Any) -> dict[str, Any]:
            inner_self.goal_seen = "goal" in info
            if tuple(info["pixels"].shape[:3]) != (2, 1, 1):
                raise AssertionError(f"audit info lacks sample/history axes: {info['pixels'].shape}")
            return {"predicted_emb": torch.zeros((2, 1, 1, 3))}

    class DummyProbe:
        def __init__(inner_self) -> None:
            inner_self.model = torch.nn.Identity()

        def __call__(inner_self, embedding: Any) -> Any:
            return embedding

    dummy_base = DummyBase()
    coordinate = ProbeCoordinateModel(dummy_base, DummyProbe())
    proxy = AuditedTrustProxy(coordinate.module, coordinate)
    proxy.audit_contexts = [
        {"env_idx": env_idx, "planning_cycle": 0, "env_step": 0}
        for env_idx in range(2)
    ]
    proxy.record_returned_plan_predictions(
        {
            "pixels": torch.zeros((2, 1, 3, 2, 2)),
            "goal": torch.ones((2, 1, 3, 2, 2)),
            "goal_privileged_block_0_pos": torch.zeros((2, 1, 3)),
        },
        torch.zeros((2, t2common.HORIZON, t2common.ACTION_BLOCK * 5)),
    )
    if dummy_base.goal_seen is not False or len(proxy.rollout_audits) != 2:
        raise AssertionError("exact returned-plan audit did not blind goal/record both envs")
    # Frozen reference-only checks: no simulator, model, EGL, or CUDA.
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = _require_sha(
        "standard rows", ood._formal_rows(dataset, args.manifest), STANDARD_ROWS_SHA256
    )
    contracts = {
        name: {
            "rows_sha256": _array_sha256(data["rows"]),
            "target_rows_sha256": _array_sha256(data["target_rows"]),
        }
        for name, spec in ARMS.items()
        for data in [_frozen_scenario(spec, dataset, formal_rows)]
    }
    print(json.dumps({
        "status": "ok",
        "arm_count": len(ARMS),
        "frozen_contracts": contracts,
        "waypoint_algorithm": "ceil(distance/spacing), final coordinate exact",
        "probe_goal_image_blinded": True,
        "physical_target_final_only": True,
        "segment_timeout_to_direct_final": True,
        "stalled_window": "9 frames / 8 physical steps / progress <1cm",
        "exact_plan_imagination_alignment": "25 uninterrupted env actions; otherwise censored",
    }, indent=2))
    return 0


def run(args: argparse.Namespace) -> int:
    t2common.configure_storage()
    formal = args.num_eval == 50
    if args.seed != FORMAL_SEED or args.protocol != "t2":
        raise ValueError("waypoint benchmark is frozen to seed42/T2")
    if args.num_eval not in (2, 50):
        raise ValueError("num-eval must be 2 smoke or 50 formal")
    if formal and not args.authorize_formal:
        raise PermissionError("pass --authorize-formal for 50-env evaluation")
    if formal and args.exercise_smoke_branches:
        raise ValueError("formal runs reject forced smoke branch exercise")
    for path, label in (
        (args.dataset, "dataset"),
        (args.manifest, "manifest"),
        (args.index, "memory index"),
        (args.checkpoint, "robust checkpoint"),
        (args.probe, "probe"),
        (args.probe_dataset_metadata, "probe metadata"),
    ):
        probe_common.ensure_data_disk(path, label)
        if not path.exists():
            raise FileNotFoundError(path)
    root = args.output.resolve()
    expected = DEFAULT_OUTPUT.resolve()
    if root != expected:
        raise ValueError(f"output root is frozen: expected={expected}, actual={root}")

    if args.self_test:
        return self_test(args)
    if not formal and not args.video:
        raise ValueError("2-env smoke requires --video for visual inspection")

    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm
    import torch

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = _require_sha(
        "standard rows", ood._formal_rows(dataset, args.manifest), STANDARD_ROWS_SHA256
    )
    standard_state = dataset.get_row_data(formal_rows)
    formal_episodes = np.asarray(standard_state["ep_idx"], dtype=np.int64)
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    long_index = brain.HeldoutMemoryIndex(args.index, args.dataset, formal_episodes)
    scaler = memory_legacy._standard_scaler(index)
    robust_contract = probe_goal._robust_checkpoint_contract(args.checkpoint, formal)
    base_model = swm.wm.utils.load_pretrained(
        args.checkpoint, cache_dir=str(PROJECT)
    ).to(args.device).eval().requires_grad_(False)
    base_model.interpolate_pos_encoding = True
    loaded_probe = xyz_common.LoadedXYZProbe(args.probe, args.device)
    probe_contract = probe_goal._validate_probe_contract(
        loaded_probe, args.probe_dataset_metadata, base_model, formal_episodes
    )
    selected_arms = list(ARMS) if args.arm == "all" else [args.arm]
    root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    for name in selected_arms:
        spec = ARMS[name]
        output = _prepare_output(_arm_output(root, name, formal), args.overwrite)
        _run_arm(
            args=args,
            spec=spec,
            output=output,
            dataset=dataset,
            formal_rows=formal_rows,
            index=index,
            long_index=long_index,
            scaler=scaler,
            base_model=base_model,
            loaded_probe=loaded_probe,
            probe_contract=probe_contract,
            robust_contract=robust_contract,
        )
        completed.append(name)
        _write_summary(root, formal)
    manifest_root = root if formal else root / "smoke"
    probe_common.write_json(
        manifest_root / "run_manifest.json",
        {
            "format_version": "cube_waypoint_probe_run_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "formal": formal,
            "num_eval": args.num_eval,
            "seed": args.seed,
            "protocol": args.protocol,
            "completed_arms": completed,
            "all_requested_arms_completed": completed == selected_arms,
            "canonical_robust_v1": robust_contract,
            "probe_contract": probe_contract,
            "script": _identity(Path(__file__)),
        },
    )
    print(manifest_root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=t2common.DATASET)
    parser.add_argument("--manifest", type=Path, default=t2common.MANIFEST)
    parser.add_argument("--index", type=Path, default=t2common.MEMORY_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=probe_goal.ROBUST_CHECKPOINT)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--probe-dataset-metadata", type=Path, default=DEFAULT_PROBE_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", default="t2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--exercise-smoke-branches", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
