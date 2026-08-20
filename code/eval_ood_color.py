#!/usr/bin/env python3
"""Evaluate the pretrained single-Cube LeWM under controlled color shifts.

The script deliberately mirrors ``eval.py``'s seed-42, 50-row Cube protocol,
while making the visual intervention explicit:

* ``red/matched`` is the original HDF5-pixel protocol.  It does not recolor
  or re-render either the current or goal image and is only accepted with
  ``rs1`` (the existing CEM10 baseline should be reused instead).
* ``blue`` and ``yellow`` use pure RGB cube colors in the live simulator.
* A ``matched`` goal keeps the current robot state, teleports only the cube
  to the dataset target pose, renders it in the live-cube color, then restores
  the simulator state.
* A ``mismatched`` goal is the unchanged red HDF5 goal image.
* A ``recolor`` goal is the frozen HDF5 future frame with only cube pixels
  recolored offline, indexed by the same verified formal environment index.

Each CEM planning cycle stores the complete per-iteration cost population.
No project file other than this standalone entry point is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np


AILAB_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = AILAB_ROOT / "datasets/ogbench/cube_single_expert.h5"
DEFAULT_MANIFEST = AILAB_ROOT / "outputs/audit/cube_cem_manifest.json"
OUTPUT_ROOT = AILAB_ROOT / "outputs/eval/cube/ood"
SELECT_OUTPUT_ROOT = AILAB_ROOT / "outputs/eval/cube/ood_select"
COLORAUG_OUTPUT_ROOT = AILAB_ROOT / "outputs/eval/cube/coloraug"
COLORAUG_CHECKPOINT_ROOT = AILAB_ROOT / "checkpoints/lewm-cube-coloraug"
MASKEDAUG_OUTPUT_ROOT = AILAB_ROOT / "outputs/eval/cube/maskedaug"
MASKEDAUG_CHECKPOINT_ROOT = AILAB_ROOT / "checkpoints/lewm-cube-maskedaug"
RECOLOR_GOAL_ROOT = OUTPUT_ROOT / "goal_recolor"
TMP_ROOT = AILAB_ROOT.parent / "tmp"

FORMAL_SEED = 42
FORMAL_NUM_EVAL = 50
GOAL_OFFSET = 25
EVAL_BUDGET = 50
IMAGE_SIZE = 224
HORIZON = 5
RECEDING_HORIZON = 5
ACTION_BLOCK = 5
NUM_SAMPLES = 300
TOPK = 30

# ``red`` is a protocol label for the original dataset/environment appearance.
# The actual training red is [0.96, 0.26, 0.33], not pure [1, 0, 0].
PURE_RGB = {
    "blue": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    "yellow": np.asarray([1.0, 1.0, 0.0], dtype=np.float64),
}


def _configure_storage() -> None:
    defaults = {
        "STABLEWM_HOME": str(AILAB_ROOT),
        "HF_HOME": str(AILAB_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
        "MUJOCO_GL": "egl",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_output(
    path: Path,
    overwrite: bool,
    selector: str,
    derived_coloraug: bool = False,
    derived_maskedaug: bool = False,
) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"refusing symlink output directory: {raw}")
    resolved = raw.resolve()
    root = (
        MASKEDAUG_OUTPUT_ROOT.resolve()
        if derived_maskedaug
        else COLORAUG_OUTPUT_ROOT.resolve()
        if derived_coloraug
        else SELECT_OUTPUT_ROOT.resolve()
        if selector == "top1"
        else OUTPUT_ROOT.resolve()
    )
    if resolved == root or root not in resolved.parents:
        raise ValueError(
            f"{selector} output must be a concrete child of {root}: {resolved}"
        )
    recolor_source = RECOLOR_GOAL_ROOT.resolve()
    if resolved == recolor_source or recolor_source in resolved.parents:
        raise ValueError(
            f"output must not overlap frozen recolor inputs: {resolved}"
        )
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output is not empty: {resolved}; pass --overwrite intentionally"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _planner_steps(planner: str) -> int:
    return {"cem10": 10, "rs1": 1}[planner]


def _default_output(args: argparse.Namespace) -> Path:
    if getattr(args, "derived_maskedaug", False):
        protocol = (
            "red"
            if args.color == "red"
            else f"{args.color}_v2"
            if args.goal_type == "recolor"
            else f"{args.color}_{args.goal_type}"
        )
        return (
            MASKEDAUG_OUTPUT_ROOT / protocol
            if args.num_eval == FORMAL_NUM_EVAL
            else MASKEDAUG_OUTPUT_ROOT / "smoke" / protocol
        )
    if getattr(args, "derived_coloraug", False):
        protocol = (
            "red"
            if args.color == "red"
            else f"{args.color}_v2"
            if args.goal_type == "recolor"
            else f"{args.color}_{args.goal_type}"
        )
        return (
            COLORAUG_OUTPUT_ROOT / protocol
            if args.num_eval == FORMAL_NUM_EVAL
            else COLORAUG_OUTPUT_ROOT / "smoke" / protocol
        )
    selector = getattr(args, "selector", "mean")
    if selector == "top1":
        if args.color == "red":
            protocol = "red"
        elif args.goal_type == "recolor":
            protocol = f"{args.color}_v2"
        else:
            protocol = f"{args.color}_{args.goal_type}"
        suffix = "top1" if args.planner == "cem10" else f"top1_{args.planner}"
        name = f"{protocol}_{suffix}"
        return (
            SELECT_OUTPUT_ROOT / name
            if args.num_eval == 50
            else SELECT_OUTPUT_ROOT / "smoke" / f"{name}_n2"
        )
    if args.goal_type == "recolor":
        name = f"{args.color}_v2_{args.planner}"
        return (
            OUTPUT_ROOT / name
            if args.num_eval == 50
            else OUTPUT_ROOT / "smoke" / name
        )
    name = f"{args.color}_{args.goal_type}_{args.planner}"
    return (
        OUTPUT_ROOT / name
        if args.num_eval == 50
        else OUTPUT_ROOT / "smoke" / f"{name}_n2"
    )


def _recolor_goal_path(color: str) -> Path:
    return RECOLOR_GOAL_ROOT / f"{color}_goal.npy"


def _load_recolor_goals(color: str, num_eval: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the frozen env-indexed, recolored HDF5 future-frame goals."""

    path = _recolor_goal_path(color)
    if not path.is_file():
        raise FileNotFoundError(f"recolor goal array missing: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = (FORMAL_NUM_EVAL, IMAGE_SIZE, IMAGE_SIZE, 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"recolor goal shape mismatch: expected={expected_shape}, actual={array.shape}, "
            f"path={path}"
        )
    if array.dtype != np.uint8:
        raise TypeError(
            f"recolor goal dtype mismatch: expected=uint8, actual={array.dtype}, path={path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    goals = np.asarray(array[:num_eval]).copy()
    return goals, {
        "source": str(path.resolve()),
        "sha256": digest,
        "full_shape": list(expected_shape),
        "evaluated_slice": [0, num_eval],
        "dtype": "uint8",
        "index_contract": "axis0_env_idx_matches_verified_formal_rows",
    }


def _validate_protocol(args: argparse.Namespace) -> None:
    selector = getattr(args, "selector", "mean")
    if getattr(args, "derived_coloraug", False) and getattr(
        args, "derived_maskedaug", False
    ):
        raise ValueError("--derived-coloraug and --derived-maskedaug are mutually exclusive")
    if args.num_eval not in (2, 50):
        raise ValueError("--num-eval is frozen to 2 (smoke) or 50 (formal)")
    if args.seed != FORMAL_SEED:
        raise ValueError(f"--seed is frozen to {FORMAL_SEED}")
    if args.goal_offset != GOAL_OFFSET or args.eval_budget != EVAL_BUDGET:
        raise ValueError("Cube OOD protocol is frozen at goal_offset=25, budget=50")
    if args.color == "red" and args.goal_type != "matched":
        raise ValueError(
            "red is the unchanged HDF5 baseline and is only valid with "
            "--goal-type matched"
        )
    if args.goal_type == "recolor" and args.color not in PURE_RGB:
        raise ValueError("--goal-type recolor is only valid for blue or yellow")
    if (
        args.color == "red"
        and args.planner == "cem10"
        and args.num_eval != 2
        and selector != "top1"
        and not getattr(args, "derived_coloraug", False)
        and not getattr(args, "derived_maskedaug", False)
    ):
        raise ValueError(
            "red/matched/cem10 is permitted only as the required 2-env "
            "mean-selector consistency smoke; reuse the completed 50-env baseline. "
            "The top1 selector is allowed for the formal red control."
        )
    if getattr(args, "derived_coloraug", False):
        if args.planner != "cem10" or selector != "mean":
            raise ValueError(
                "--derived-coloraug is frozen to legacy --planner cem10 "
                "--selector mean"
            )
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        checkpoint_root = COLORAUG_CHECKPOINT_ROOT.resolve()
        if (
            not checkpoint.is_file()
            or checkpoint.suffix != ".pt"
            or checkpoint_root not in checkpoint.parents
        ):
            raise ValueError(
                "--derived-coloraug requires a local .pt checkpoint under "
                f"{checkpoint_root}: {checkpoint}"
            )
    if getattr(args, "derived_maskedaug", False):
        if args.planner != "cem10" or selector != "mean":
            raise ValueError(
                "--derived-maskedaug is frozen to legacy --planner cem10 "
                "--selector mean"
            )
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        checkpoint_root = MASKEDAUG_CHECKPOINT_ROOT.resolve()
        if (
            not checkpoint.is_file()
            or checkpoint.suffix != ".pt"
            or checkpoint_root not in checkpoint.parents
        ):
            raise ValueError(
                "--derived-maskedaug requires a local .pt checkpoint under "
                f"{checkpoint_root}: {checkpoint}"
            )
    if args.input_mode == "dataset" and not (
        args.color == "red" and args.goal_type == "matched"
    ):
        raise ValueError(
            "--input-mode dataset is the byte-for-byte HDF5 control and is only "
            "valid for --color red --goal-type matched"
        )
    if args.goal_type == "recolor":
        # Validate the complete frozen 50-row array before any CUDA/EGL work.
        _load_recolor_goals(args.color, args.num_eval)
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    for path, label in ((args.dataset, "dataset"), (args.manifest, "manifest")):
        if AILAB_ROOT.parent.resolve() not in path.resolve().parents:
            raise ValueError(f"{label} must be stored on the data disk: {path}")


def _formal_rows(dataset: Any, manifest_path: Path) -> np.ndarray:
    """Reproduce eval.py exactly, then compare all 50 rows to the frozen manifest."""

    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_col = dataset.get_col_data(col)
    step_col = dataset.get_col_data("step_idx")
    episode_ids, inverse = np.unique(episode_col, return_inverse=True)
    max_steps = np.full(len(episode_ids), -1, dtype=np.int64)
    np.maximum.at(max_steps, inverse, step_col)
    lengths = max_steps + 1
    max_start = lengths - GOAL_OFFSET - 1
    max_by_episode = {ep: max_start[i] for i, ep in enumerate(episode_ids)}
    max_per_row = np.asarray([max_by_episode[ep] for ep in episode_col])
    valid_indices = np.nonzero(step_col <= max_per_row)[0]

    rng = np.random.default_rng(FORMAL_SEED)
    sampled_positions = rng.choice(
        len(valid_indices) - 1, size=FORMAL_NUM_EVAL, replace=False
    )
    actual = np.sort(valid_indices[sampled_positions])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = np.asarray(manifest["formal_rows"], dtype=np.int64)
    if expected.shape != (FORMAL_NUM_EVAL,) or not np.array_equal(actual, expected):
        mismatch = np.nonzero(actual != expected)[0] if actual.shape == expected.shape else []
        where = int(mismatch[0]) if len(mismatch) else "shape"
        raise RuntimeError(
            "formal row selection mismatch: "
            f"position={where}, expected_shape={expected.shape}, actual_shape={actual.shape}"
        )

    rows = dataset.get_row_data(actual)
    episodes = np.asarray(rows[col])
    starts = np.asarray(rows["step_idx"])
    goals = dataset.get_row_data(actual + GOAL_OFFSET)
    goal_episodes = np.asarray(goals[col])
    goal_steps = np.asarray(goals["step_idx"])
    if not np.array_equal(episodes, goal_episodes):
        raise RuntimeError("one or more +25 goal rows cross an episode boundary")
    if not np.array_equal(goal_steps - starts, np.full(50, GOAL_OFFSET)):
        raise RuntimeError("one or more goal rows are not exactly +25 steps")
    return actual


def _image_transform(image_size: int) -> Any:
    import stable_pretraining as spt
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=image_size),
        ]
    )


def _extract_init_goal(
    dataset: Any,
    episodes_idx: Sequence[int],
    start_steps: Sequence[int],
    goal_offset: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[np.ndarray]]:
    """Local copy of the installed evaluator's dataset extraction contract."""

    import torch

    episodes = np.asarray(episodes_idx)
    starts = np.asarray(start_steps)
    chunks = dataset.load_chunk(episodes, starts, starts + goal_offset + 1)
    init_lists: dict[str, list[np.ndarray]] = {}
    goal_lists: dict[str, list[np.ndarray]] = {}
    dataset_videos: list[np.ndarray] = []
    for episode in chunks:
        for col in dataset.column_names:
            if col.startswith("goal") or col not in episode:
                continue
            value = episode[col]
            if col.startswith("pixels"):
                value = value.permute(0, 2, 3, 1)
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                continue
            array = value.numpy() if isinstance(value, torch.Tensor) else value
            init_lists.setdefault(col, []).append(array[0])
            goal_lists.setdefault(col, []).append(array[-1])
            if col == "pixels":
                dataset_videos.append(array)
    init = {key: np.stack(values) for key, values in init_lists.items()}
    goal = {
        ("goal" if key == "pixels" else f"goal_{key}"): np.stack(values)
        for key, values in goal_lists.items()
    }
    return init, goal, dataset_videos


def _apply_callables(env: Any, specs: Sequence[dict], state: dict[str, Any]) -> None:
    for spec in specs:
        if not hasattr(env, spec["method"]):
            raise AttributeError(
                f"Cube environment lacks required method {spec['method']!r}"
            )
        kwargs = {}
        for name, item in spec.get("args", {}).items():
            if item.get("in_dataset", True):
                key = item["value"]
                if key not in state:
                    raise KeyError(f"callable input missing dataset key: {key}")
                kwargs[name] = deepcopy(state[key])
            else:
                kwargs[name] = deepcopy(item.get("value"))
        getattr(env, spec["method"])(**kwargs)


def _set_cube_rgb(raw_env: Any, rgb: np.ndarray) -> None:
    for geom_id in raw_env._cube_geom_ids_list[0]:
        raw_env._model.geom(geom_id).rgba[:3] = rgb
        raw_env._model.geom(geom_id).rgba[3] = 1.0
    for geom_id in raw_env._cube_target_geom_ids_list[0]:
        raw_env._model.geom(geom_id).rgba[:3] = rgb


def _assert_cube_rgb(raw_env: Any, expected: np.ndarray) -> None:
    actual = np.stack(
        [raw_env._model.geom(i).rgba[:3] for i in raw_env._cube_geom_ids_list[0]]
    )
    if not np.allclose(actual, expected[None, :], rtol=0.0, atol=1e-7):
        raise RuntimeError(
            f"cube color injection failed: expected={expected.tolist()}, "
            f"actual={actual.tolist()}"
        )


def _render_color_views(
    world: Any,
    init_state: dict[str, np.ndarray],
    goal_state: dict[str, np.ndarray],
    color: str,
    goal_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Render exact initial/goal simulator states without stepping physics."""

    current = np.asarray(init_state["pixels"]).copy()
    goals = np.asarray(goal_state["goal"]).copy()
    if color == "red":
        # Critical baseline invariant: byte-for-byte HDF5 inputs.
        return current, goals

    rgb = PURE_RGB[color]
    rendered_current: list[np.ndarray] = []
    rendered_goal: list[np.ndarray] = []
    for i, wrapped in enumerate(world.envs.envs):
        import mujoco

        raw = wrapped.unwrapped
        _assert_cube_rgb(raw, rgb)
        raw.set_state(np.asarray(init_state["qpos"][i]), np.asarray(init_state["qvel"][i]))
        rendered_current.append(np.asarray(wrapped.render(), dtype=np.uint8).copy())

        if goal_type == "matched":
            # Protocol requirement: only teleport the cube to the target pose.
            # The arm stays at the current-row state; using the full future qpos
            # would leak the expert's future robot configuration into the goal.
            cube_qpos = raw._data.joint("object_joint_0").qpos
            cube_qpos[:3] = np.asarray(
                goal_state["goal_privileged_block_0_pos"][i]
            )
            cube_qpos[3:] = np.asarray(
                goal_state["goal_privileged_block_0_quat"][i]
            )
            mujoco.mj_forward(raw._model, raw._data)
            rendered_goal.append(np.asarray(wrapped.render(), dtype=np.uint8).copy())

        # Restore the actual evaluation state and live-cube color exactly.
        raw.set_state(np.asarray(init_state["qpos"][i]), np.asarray(init_state["qvel"][i]))
        _assert_cube_rgb(raw, rgb)

    current = np.stack(rendered_current)
    if goal_type == "matched":
        goals = np.stack(rendered_goal)
    # For mismatched goals, retain the original red HDF5 pixels unchanged.
    return current, goals


class PlanningCostRecorder:
    """CEM callback retaining full costs for every env and planning cycle."""

    output_key = "ood_color_cost_history"

    def __init__(self, num_envs: int) -> None:
        self.records: dict[int, list[dict[str, Any]]] = {
            i: [] for i in range(num_envs)
        }
        self.history: list[dict[str, Any]] = []
        self._pending: list[tuple[int, int]] | None = None
        self._batch_cursor = 0
        self._active: dict[str, Any] | None = None
        self.last_solve_cycles: list[dict[str, Any]] = []

    def begin_solve(self, env_steps: Sequence[tuple[int, int]]) -> None:
        if self._pending is not None:
            raise RuntimeError("cost recorder received overlapping solver calls")
        self._pending = [(int(env), int(step)) for env, step in env_steps]

    def reset(self) -> None:
        if self._pending is None:
            raise RuntimeError("solver started without recorder env context")
        self._batch_cursor = 0
        self._active = None
        self.history = []
        self.last_solve_cycles = []

    def start_batch(self) -> None:
        assert self._pending is not None
        if self._batch_cursor >= len(self._pending):
            raise RuntimeError("solver produced more batches than expected")
        env_idx, env_step = self._pending[self._batch_cursor]
        self._active = {
            "cycle_idx": len(self.records[env_idx]),
            "env_step": env_step,
            "costs": [],
            "topk_costs": [],
            "mean": [],
            "variance": [],
        }
        self.records[env_idx].append(self._active)
        self.last_solve_cycles.append(self._active)
        self._batch_cursor += 1

    def __call__(self, **state: Any) -> None:
        if self._active is None:
            raise RuntimeError("cost callback invoked outside a solver batch")
        self._active["costs"].append(
            state["costs"].detach().cpu().float().numpy()[0]
        )
        self._active["topk_costs"].append(
            state["topk_vals"].detach().cpu().float().numpy()[0]
        )
        self._active["mean"].append(
            state["mean"].detach().cpu().float().numpy()[0]
        )
        self._active["variance"].append(
            state["var"].detach().cpu().float().numpy()[0]
        )
        # Overwrite on every CEM iteration so these fields provenance the
        # final iteration without retaining every iteration's action tensor.
        cost_tensor = state["costs"].detach()
        costs = cost_tensor.cpu().float().numpy()[0]
        candidates = state["candidates"].detach().cpu().float().numpy()[0]
        top1_index = int(cost_tensor[0].argmin().cpu().item())
        self._active["final_top1_index"] = top1_index
        self._active["final_top1_cost"] = float(costs[top1_index])
        self._active["final_candidate0_cost"] = float(costs[0])
        self._active["final_candidate0_minus_top1_cost"] = float(
            costs[0] - costs[top1_index]
        )
        self._active["final_top1_actions_normalized"] = candidates[
            top1_index
        ].copy()
        self._active["final_candidate0_actions_normalized"] = candidates[0].copy()
        self._active["final_candidates_normalized"] = candidates.copy()

    def end_solve(self) -> None:
        assert self._pending is not None
        if self._batch_cursor != len(self._pending):
            raise RuntimeError(
                f"expected {len(self._pending)} solver batches, got {self._batch_cursor}"
            )
        self.history = [
            {"env_idx": env, "env_step": step} for env, step in self._pending
        ]
        self._pending = None
        self._active = None

    def selected_top1_actions(self) -> np.ndarray:
        if not self.last_solve_cycles:
            raise RuntimeError("no completed CEM batches are available for top1")
        actions = []
        for cycle in self.last_solve_cycles:
            if "final_top1_actions_normalized" not in cycle:
                raise RuntimeError("CEM callback did not capture final candidates")
            actions.append(cycle["final_top1_actions_normalized"])
        return np.stack(actions)

    def record_solver_returned_actions(self, actions: np.ndarray) -> None:
        if len(actions) != len(self.last_solve_cycles):
            raise RuntimeError(
                "solver returned action count differs from callback batches: "
                f"actions={len(actions)}, batches={len(self.last_solve_cycles)}"
            )
        for cycle, action in zip(self.last_solve_cycles, actions, strict=True):
            cycle["solver_returned_actions_normalized"] = np.asarray(action).copy()


def _make_selecting_solver(base: type) -> type:
    """Return a CEM solver that changes only the post-solve action selector."""

    class SelectingSolver(base):
        def __init__(
            self,
            *args: Any,
            selector: str = "mean",
            recorder: PlanningCostRecorder,
            **kwargs: Any,
        ) -> None:
            if selector not in {"mean", "top1"}:
                raise ValueError(f"unknown action selector: {selector}")
            self._action_selector = selector
            self._selection_recorder = recorder
            super().__init__(*args, **kwargs)

        def solve(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            outputs = super().solve(*args, **kwargs)
            if self._action_selector == "top1":
                import torch

                top1 = torch.from_numpy(
                    self._selection_recorder.selected_top1_actions()
                )
                reference = outputs["actions"]
                if top1.shape != reference.shape:
                    raise RuntimeError(
                        "top1 action shape differs from CEM mean output: "
                        f"top1={tuple(top1.shape)}, mean={tuple(reference.shape)}"
                    )
                outputs["actions"] = top1.to(dtype=reference.dtype)
                outputs["selector"] = "final_population_cost_argmin"
            else:
                outputs["selector"] = "updated_elite_mean"
            self._selection_recorder.record_solver_returned_actions(
                outputs["actions"].detach().cpu().float().numpy()
            )
            return outputs

    return SelectingSolver


def _make_recording_policy(base: type) -> type:
    class RecordingPolicy(base):
        def __init__(self, *args: Any, recorder: PlanningCostRecorder, **kwargs: Any):
            self._cost_recorder = recorder
            self._ood_env_step = 0
            super().__init__(*args, **kwargs)

        def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
            terminated = info_dict.get("terminated")
            if terminated is None:
                dead = np.zeros(self.env.num_envs, dtype=bool)
            else:
                dead = np.asarray(terminated, dtype=bool).reshape(self.env.num_envs, -1)[
                    :, 0
                ]
            replan = [
                i
                for i in range(self.env.num_envs)
                if not dead[i] and len(self._action_buffer[i]) == 0
            ]
            if replan:
                self._cost_recorder.begin_solve(
                    [(i, self._ood_env_step) for i in replan]
                )
            action = super().get_action(info_dict, **kwargs)
            self._ood_env_step += 1
            return action

    return RecordingPolicy


def _save_cost_history(
    output: Path,
    recorder: PlanningCostRecorder,
    rows: np.ndarray,
    episodes: np.ndarray,
    starts: np.ndarray,
    selector: str = "mean",
) -> list[dict[str, Any]]:
    if selector not in {"mean", "top1"}:
        raise ValueError(f"unknown action selector: {selector}")
    root = output / "cost_history"
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for env_idx in range(len(rows)):
        arrays: dict[str, np.ndarray] = {}
        cycles_summary = []
        cycles = recorder.records[env_idx]
        for cycle_idx, cycle in enumerate(cycles):
            prefix = f"cycle_{cycle_idx:02d}"
            for key in ("costs", "topk_costs", "mean", "variance"):
                arrays[f"{prefix}_{key}"] = np.stack(cycle[key])
            for key in (
                "final_top1_actions_normalized",
                "final_candidate0_actions_normalized",
                "final_candidates_normalized",
                "solver_returned_actions_normalized",
            ):
                arrays[f"{prefix}_{key}"] = np.asarray(cycle[key])
            arrays[f"{prefix}_final_top1_index"] = np.asarray(
                cycle["final_top1_index"], dtype=np.int64
            )
            arrays[f"{prefix}_final_top1_cost"] = np.asarray(
                cycle["final_top1_cost"], dtype=np.float32
            )
            arrays[f"{prefix}_final_candidate0_cost"] = np.asarray(
                cycle["final_candidate0_cost"], dtype=np.float32
            )
            arrays[f"{prefix}_final_candidate0_minus_top1_cost"] = np.asarray(
                cycle["final_candidate0_minus_top1_cost"], dtype=np.float32
            )
            costs = arrays[f"{prefix}_costs"]
            topk = arrays[f"{prefix}_topk_costs"]
            if selector == "top1":
                selected_action_key = f"{prefix}_final_top1_actions_normalized"
                selected_action_index = None
                selected_action_source = "final_population_cost_argmin"
            else:
                selected_action_key = f"{prefix}_mean"
                selected_action_index = -1
                selected_action_source = "final_updated_elite_mean"
            cycles_summary.append(
                {
                    "cycle_idx": cycle_idx,
                    "env_step": cycle["env_step"],
                    "num_iterations": int(costs.shape[0]),
                    "num_samples": int(costs.shape[1]),
                    "best_cost_by_iteration": np.min(costs, axis=1),
                    "mean_cost_by_iteration": np.mean(costs, axis=1),
                    "elite_mean_cost_by_iteration": np.mean(topk, axis=1),
                    "final_top1_index": cycle["final_top1_index"],
                    "final_top1_cost": cycle["final_top1_cost"],
                    "final_candidate0_cost": cycle["final_candidate0_cost"],
                    "final_candidate0_minus_top1_cost": cycle[
                        "final_candidate0_minus_top1_cost"
                    ],
                    "top1_action_npz_key": (
                        f"{prefix}_final_top1_actions_normalized"
                    ),
                    "candidate0_action_npz_key": (
                        f"{prefix}_final_candidate0_actions_normalized"
                    ),
                    "final_population_npz_key": (
                        f"{prefix}_final_candidates_normalized"
                    ),
                    "solver_returned_action_npz_key": (
                        f"{prefix}_solver_returned_actions_normalized"
                    ),
                    "selection_population": "final_cem_iteration_300_candidates",
                    "action_selector": selector,
                    "selected_action_source": selected_action_source,
                    "selected_action_npz_key": selected_action_key,
                    "selected_action_npz_index": selected_action_index,
                }
            )
        stem = f"episode_{env_idx:02d}_row_{int(rows[env_idx])}"
        if arrays:
            np.savez_compressed(root / f"{stem}.npz", **arrays)
        summary = {
            "env_idx": env_idx,
            "dataset_row": int(rows[env_idx]),
            "episode_idx": int(episodes[env_idx]),
            "start_step": int(starts[env_idx]),
            "action_selector": selector,
            "planning_cycles": cycles_summary,
            "npz": str((root / f"{stem}.npz").resolve()),
        }
        _write_json(root / f"{stem}.json", summary)
        summaries.append(summary)
    return summaries


def _evaluate(
    world: Any,
    dataset: Any,
    rows: np.ndarray,
    goal_type: str,
    color: str,
    eval_budget: int,
    video_dir: Path | None,
    recolor_goals: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from stable_worldmodel.plot import save_panel_videos

    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    selected = dataset.get_row_data(rows)
    episodes = np.asarray(selected[col])
    starts = np.asarray(selected["step_idx"])
    init_state, goal_state, dataset_videos = _extract_init_goal(
        dataset, episodes, starts, GOAL_OFFSET
    )

    reset_options = None
    if color != "red":
        rgb = PURE_RGB[color]
        reset_options = [
            {
                "variation": [],
                "variation_values": {"cube.color": rgb[None, :].copy()},
            }
            for _ in range(len(rows))
        ]
    world.reset(seed=init_state.get("seed"), options=reset_options)
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
    for i, wrapped in enumerate(world.envs.envs):
        _apply_callables(wrapped.unwrapped, callables, {k: v[i] for k, v in merged.items()})

    current_pixels, goal_pixels = _render_color_views(
        world, init_state, goal_state, color, goal_type
    )
    if goal_type == "recolor":
        if recolor_goals is None:
            raise RuntimeError("recolor goal protocol requires a frozen goal array")
        expected_shape = (len(rows), IMAGE_SIZE, IMAGE_SIZE, 3)
        if recolor_goals.shape != expected_shape or recolor_goals.dtype != np.uint8:
            raise RuntimeError(
                "validated recolor goal array changed before evaluation: "
                f"expected shape/dtype={expected_shape}/uint8, "
                f"actual={recolor_goals.shape}/{recolor_goals.dtype}"
            )
        goal_pixels = recolor_goals.copy()
    init_state["pixels"] = current_pixels
    goal_state["goal"] = goal_pixels

    shape_prefix = world.infos["pixels"].shape[:2]
    for state in (init_state, goal_state):
        for key, value in state.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()
    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}

    successes = np.zeros(len(rows), dtype=bool)
    frames: dict[int, list[np.ndarray]] | None = (
        defaultdict(list) if video_dir is not None else None
    )

    def on_step(active_world: Any) -> None:
        active_world.infos.update(deepcopy(goal_snapshot))
        successes[:] |= active_world.terminateds
        if frames is not None:
            for i in range(active_world.num_envs):
                frame = active_world.infos["pixels"][i]
                frame = frame[-1] if frame.ndim > 3 else frame
                frames[i].append(np.asarray(frame).copy())

    world._run(max_steps=eval_budget, mode="wait", on_step=on_step)
    if frames is not None:
        save_panel_videos(
            video_dir,
            {"agent": frames, "dataset": dataset_videos, "goal": goal_pixels},
        )
    metrics = {
        "success_rate": float(successes.sum()) / len(rows) * 100.0,
        "success_count": int(successes.sum()),
        "num_eval": len(rows),
        "episode_successes": successes,
        "seeds": init_state.get("seed"),
    }
    selected_meta = {"episodes": episodes, "starts": starts}
    return metrics, selected_meta


def run(args: argparse.Namespace) -> int:
    _validate_protocol(args)
    _configure_storage()
    output = args.output or _default_output(args)
    output = _safe_output(
        output,
        args.overwrite,
        args.selector,
        getattr(args, "derived_coloraug", False),
        getattr(args, "derived_maskedaug", False),
    )

    recolor_goals = None
    recolor_metadata = None
    if args.goal_type == "recolor":
        recolor_goals, recolor_metadata = _load_recolor_goals(
            args.color, args.num_eval
        )

    import stable_worldmodel as swm
    import torch
    from sklearn.preprocessing import StandardScaler

    if not torch.cuda.is_available():
        raise RuntimeError("Cube color evaluation requires CUDA")

    print(
        "protocol:",
        json.dumps(
            {
                "color": args.color,
                "goal_type": args.goal_type,
                "planner": args.planner,
                "selector": args.selector,
                "num_eval": args.num_eval,
                "seed": args.seed,
                "goal_offset": args.goal_offset,
                "eval_budget": args.eval_budget,
            },
            sort_keys=True,
        ),
    )

    dataset = swm.data.HDF5Dataset(
        path=args.dataset,
        keys_to_cache=["action"],
    )
    all_rows = _formal_rows(dataset, args.manifest)
    rows = all_rows[: args.num_eval]
    print(f"formal row check: exact 50/50 match; evaluating rows={rows.tolist()}")

    action = dataset.get_col_data("action")
    action = action[~np.isnan(action).any(axis=1)]
    scaler = StandardScaler().fit(action)
    del action

    model = swm.wm.utils.load_pretrained(
        args.checkpoint, cache_dir=str(AILAB_ROOT)
    )
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    recorder = PlanningCostRecorder(args.num_eval)
    solver_cls = _make_selecting_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=model,
        batch_size=1,
        num_samples=NUM_SAMPLES,
        var_scale=1.0,
        n_steps=_planner_steps(args.planner),
        topk=TOPK,
        device=args.device,
        seed=args.seed,
        callbacks=[recorder],
        selector=args.selector,
        recorder=recorder,
    )
    plan_config = swm.PlanConfig(
        horizon=HORIZON,
        receding_horizon=RECEDING_HORIZON,
        action_block=ACTION_BLOCK,
    )
    policy_cls = _make_recording_policy(swm.policy.WorldModelPolicy)
    policy = policy_cls(
        solver=solver,
        config=plan_config,
        process={"action": scaler},
        transform={
            "pixels": _image_transform(IMAGE_SIZE),
            "goal": _image_transform(IMAGE_SIZE),
        },
        recorder=recorder,
    )
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=args.num_eval,
        max_episode_steps=2 * args.eval_budget,
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        env_type="single",
        ob_type="states",
        multiview=False,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        visualize_info=False,
        terminate_at_goal=True,
    )
    world.set_policy(policy)

    video_dir = output / "videos" if args.video else None
    started = time.time()
    try:
        metrics, selected = _evaluate(
            world,
            dataset,
            rows,
            args.goal_type,
            args.color,
            args.eval_budget,
            video_dir,
            recolor_goals,
        )
    finally:
        world.close()
    elapsed = time.time() - started

    cost_summary = _save_cost_history(
        output,
        recorder,
        rows,
        selected["episodes"],
        selected["starts"],
        args.selector,
    )
    payload = {
        "protocol": {
            "color": args.color,
            "current_rgb": (
                "unchanged_hdf5_training_red"
                if args.color == "red"
                else PURE_RGB[args.color]
            ),
            "goal_type": args.goal_type,
            "goal_rgb": (
                PURE_RGB[args.color]
                if args.goal_type in {"matched", "recolor"} and args.color != "red"
                else "unchanged_hdf5_training_red"
            ),
            "goal_protocol": (
                {
                    "kind": "offline_recolor",
                    "source_semantics": (
                        "original_hdf5_future_frame_with_only_cube_pixels_recolored"
                    ),
                    **recolor_metadata,
                }
                if recolor_metadata is not None
                else {"kind": args.goal_type}
            ),
            "planner": args.planner,
            "planner_algorithm": "cem_top30_distribution_update",
            "selector": args.selector,
            "selected_action_source": (
                "updated_top30_elite_mean"
                if args.selector == "mean"
                else "argmin_cost_candidate_from_final_300_population"
            ),
            "selected_mean_cost_recomputed": False,
            "selected_action_cost_status": (
                "not_recomputed_after_final_elite_mean_update"
                if args.selector == "mean"
                else "direct_world_model_cost_from_final_population"
            ),
            "selector_does_not_directly_change_rng_configuration_or_iteration_count": True,
            "pairing_scope": (
                "cycle0 is strictly paired for the same visual protocol and seed; "
                "later cycles are trajectory-dependent after action execution diverges"
            ),
            "selector_provenance": {
                "scope": "each_planning_cycle",
                "population": "final_cem_iteration_300_candidates",
                "argmin_tie_break": "torch_argmin_first_occurrence",
                "action_tensor": "complete_normalized_horizon_action_sequence",
                "top1_cost": "direct_world_model_cost_from_final_population",
                "candidate0": "final_iteration_pre_update_mean_candidate",
                "difference": "candidate0_cost_minus_top1_cost",
                "top1_and_candidate0_actions_saved_in_episode_npz": True,
            },
            "cem_iterations": _planner_steps(args.planner),
            "num_samples": NUM_SAMPLES,
            "var_scale": 1.0,
            "topk": TOPK,
            "solver_batch_size": 1,
            "horizon": HORIZON,
            "receding_horizon": RECEDING_HORIZON,
            "action_block": ACTION_BLOCK,
            "image_size": IMAGE_SIZE,
            "seed": args.seed,
            "goal_offset": args.goal_offset,
            "eval_budget": args.eval_budget,
            "checkpoint": args.checkpoint,
            "input_mode": args.input_mode,
            "device": args.device,
            "world": {
                "env_name": "swm/OGBCube-v0",
                "env_type": "single",
                "ob_type": "states",
                "multiview": False,
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
                "terminate_at_goal": True,
                "max_episode_steps": 2 * args.eval_budget,
            },
        },
        "formal_rows_verified": all_rows,
        "evaluated_rows": rows,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "cost_history": cost_summary,
    }
    if getattr(args, "derived_coloraug", False):
        payload["protocol"]["derived_coloraug"] = {
            "enabled": True,
            "explicit_opt_in_flag": "--derived-coloraug",
            "checkpoint_root": str(COLORAUG_CHECKPOINT_ROOT.resolve()),
            "output_root": str(COLORAUG_OUTPUT_ROOT.resolve()),
            "legacy_action_selector_required": True,
        }
    if getattr(args, "derived_maskedaug", False):
        maskedaug_protocol = (
            "red"
            if args.color == "red"
            else f"{args.color}_v2"
            if args.goal_type == "recolor"
            else f"{args.color}_{args.goal_type}"
        )
        payload["protocol"]["derived_maskedaug"] = {
            "enabled": True,
            "explicit_opt_in_flag": "--derived-maskedaug",
            "checkpoint_root": str(MASKEDAUG_CHECKPOINT_ROOT.resolve()),
            "output_root": str(MASKEDAUG_OUTPUT_ROOT.resolve()),
            "legacy_action_selector_required": True,
            "training_intervention": "float64 HSV red-mask-only hue rotation",
            "protocol_name": maskedaug_protocol,
        }
    _write_json(output / "results.json", payload)
    successes_text = ", ".join(
        "True" if value else "False" for value in metrics["episode_successes"]
    )
    (output / "results.txt").write_text(
        "==== CONFIG ====\n"
        + json.dumps(_jsonable(payload["protocol"]), indent=2, sort_keys=True)
        + "\n\n==== RESULTS ====\n"
        + f"success_rate: {metrics['success_rate']:.6f}\n"
        + f"success_count: {metrics['success_count']}/{metrics['num_eval']}\n"
        + f"episode_successes: [{successes_text}]\n"
        + f"evaluation_time: {elapsed:.6f} seconds\n",
        encoding="utf-8",
    )
    print(
        f"result: {metrics['success_count']}/{metrics['num_eval']} "
        f"({metrics['success_rate']:.2f}%), elapsed={elapsed:.2f}s"
    )
    print(f"artifacts: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-row LeWM Cube color-OOD evaluation"
    )
    parser.add_argument("--color", required=True, choices=("red", "blue", "yellow"))
    parser.add_argument(
        "--goal-type",
        required=True,
        choices=("matched", "mismatched", "recolor"),
    )
    parser.add_argument("--planner", required=True, choices=("cem10", "rs1"))
    parser.add_argument(
        "--selector",
        choices=("mean", "top1"),
        default="mean",
        help=(
            "mean executes the updated top-30 elite mean (legacy default); top1 "
            "executes the final 300-candidate population cost argmin"
        ),
    )
    parser.add_argument("--num-eval", type=int, default=2, choices=(2, 50))
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--goal-offset", type=int, default=GOAL_OFFSET)
    parser.add_argument("--eval-budget", type=int, default=EVAL_BUDGET)
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument(
        "--input-mode",
        choices=("auto", "dataset"),
        default="auto",
        help=(
            "auto applies the requested color protocol; dataset is the exact HDF5 "
            "current+goal control and is only valid for red/matched"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--derived-coloraug",
        action="store_true",
        help=(
            "explicitly evaluate a derived checkpoint under "
            "checkpoints/lewm-cube-coloraug using legacy CEM mean and the "
            "isolated outputs/eval/cube/coloraug root; defaults remain unchanged"
        ),
    )
    parser.add_argument(
        "--derived-maskedaug",
        action="store_true",
        help=(
            "explicitly evaluate a derived checkpoint under "
            "checkpoints/lewm-cube-maskedaug using legacy CEM mean and the "
            "isolated outputs/eval/cube/maskedaug root; defaults remain unchanged"
        ),
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
