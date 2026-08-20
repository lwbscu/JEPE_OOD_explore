#!/usr/bin/env python3
"""Evaluate the robust Cube checkpoint on visual-axis conditions.

This entry point is deliberately independent of the legacy evaluators.  It
reuses their frozen row selection, CEM recorder, action scaler, and policy
construction, while adding only reset-time OGBench visual variations.  The
visual condition is applied to both the initial render and the simulator goal
render; the red control remains the byte-for-byte HDF5 protocol.

Examples::

    python le-wm/eval_cube_robust.py --condition red --num-eval 2
    python le-wm/eval_cube_robust.py --condition floor_red --num-eval 50 \
      --checkpoint checkpoints/lewm-cube-robust_v1/.../weights_final.pt \
      --authorize-formal
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
TOOLS = HERE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import eval_ood_color as legacy  # noqa: E402
import eval_memory_seed as memory_legacy  # noqa: E402
import build_cube_memory_index as memory  # noqa: E402
import cube_trust_region_common as t2common  # noqa: E402
import eval_trust_region as trust  # noqa: E402


CHECKPOINT_ROOT = PROJECT / "checkpoints/lewm-cube-robust_v1"
OUTPUT_ROOT = PROJECT / "outputs/eval/cube/robust_v1"
DATASET = PROJECT / "datasets/ogbench/cube_single_expert.h5"
MANIFEST = PROJECT / "outputs/audit/cube_cem_manifest.json"
FORMAL_SEED = 42
GOAL_OFFSET = 25
EVAL_BUDGET = 50
IMAGE_SIZE = 224


def _floor(rgb1: Sequence[float], rgb2: Sequence[float]) -> np.ndarray:
    value = np.asarray([rgb1, rgb2], dtype=np.float64)
    if value.shape != (2, 3) or np.any(value < 0) or np.any(value > 1):
        raise ValueError(f"invalid floor color: {value}")
    return value


# Values are explicit and are recorded in each result manifest.  They stay
# inside the OGBench variation spaces defined by vendor/stable-worldmodel.
CONDITIONS: dict[str, dict[str, Any]] = {
    "red": {
        "axis": "regression",
        "color": "red",
        "goal_type": "matched",
        "variation": None,
        "label": "unchanged HDF5 red control",
    },
    "blue_v2": {
        "axis": "regression",
        "color": "blue",
        "goal_type": "recolor",
        "variation": None,
        "label": "blue recolor goal protocol",
    },
    "yellow_v2": {
        "axis": "regression",
        "color": "yellow",
        "goal_type": "recolor",
        "variation": None,
        "label": "yellow recolor goal protocol",
    },
    "floor_red": {
        "axis": "floor",
        "color": "red",
        "goal_type": "matched",
        "variation": {"floor.color": _floor((0.62, 0.08, 0.08), (0.86, 0.18, 0.12))},
        "label": "red floor checker",
    },
    "floor_green": {
        "axis": "floor",
        "color": "red",
        "goal_type": "matched",
        "variation": {"floor.color": _floor((0.06, 0.35, 0.12), (0.10, 0.62, 0.20))},
        "label": "green floor checker",
    },
    "light_low": {
        "axis": "light",
        "color": "red",
        "goal_type": "matched",
        "variation": {"light.intensity": np.asarray([0.30], dtype=np.float64)},
        "label": "low global light intensity",
    },
    "light_high": {
        "axis": "light",
        "color": "red",
        "goal_type": "matched",
        "variation": {"light.intensity": np.asarray([1.0], dtype=np.float64)},
        "label": "high global light intensity",
    },
    "camera_minus": {
        "axis": "camera",
        "color": "red",
        "goal_type": "matched",
        "variation": {"camera.angle_delta": np.asarray([[-8.0, -8.0]], dtype=np.float64)},
        "label": "camera angle negative delta",
    },
    "camera_plus": {
        "axis": "camera",
        "color": "red",
        "goal_type": "matched",
        "variation": {"camera.angle_delta": np.asarray([[8.0, 8.0]], dtype=np.float64)},
        "label": "camera angle positive delta",
    },
}


def _configure_storage() -> None:
    os.environ.setdefault("STABLEWM_HOME", str(PROJECT))
    os.environ.setdefault("HF_HOME", str(PROJECT.parent / ".cache/huggingface"))
    os.environ.setdefault("TORCH_HOME", str(PROJECT.parent / ".cache/torch"))
    os.environ.setdefault("TMPDIR", str(PROJECT.parent / "tmp"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    (PROJECT.parent / "tmp").mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = CHECKPOINT_ROOT.resolve()
    if not resolved.is_file() or resolved.suffix != ".pt" or root not in resolved.parents:
        raise ValueError(f"checkpoint must be a .pt file under {root}: {resolved}")
    if not (resolved.parent / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config.json missing: {resolved.parent}")
    return resolved


def _output(path: Path | None, condition: str, num_eval: int, overwrite: bool) -> Path:
    root = OUTPUT_ROOT.resolve()
    target = (path or (OUTPUT_ROOT / condition if num_eval == 50 else OUTPUT_ROOT / "smoke" / condition)).expanduser().resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"output must be a child of {root}: {target}")
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"non-empty output: {target}; pass --overwrite")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _variation_options(spec: dict[str, Any], num_eval: int) -> list[dict[str, Any]] | None:
    variation = spec.get("variation")
    if variation is None:
        # No explicit variation means the environment samples its frozen
        # default.  Passing None preserves the original HDF5 red protocol.
        return None
    keys = list(variation)
    return [
        {
            "variation": keys,
            "variation_values": {key: np.asarray(value).copy() for key, value in variation.items()},
        }
        for _ in range(num_eval)
    ]


def _set_cube_rgb(raw_env: Any, rgb: np.ndarray) -> None:
    for geom_id in raw_env._cube_geom_ids_list[0]:
        raw_env._model.geom(geom_id).rgba[:3] = rgb
        raw_env._model.geom(geom_id).rgba[3] = 1.0
    for geom_id in raw_env._cube_target_geom_ids_list[0]:
        raw_env._model.geom(geom_id).rgba[:3] = rgb


def _render_condition_views(
    world: Any,
    init_state: dict[str, np.ndarray],
    goal_state: dict[str, np.ndarray],
    spec: dict[str, Any],
    recolor_goals: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render current/goal under the same variation, without future arm leakage."""

    color = spec["color"]
    if color == "red" and spec["variation"] is None:
        return np.asarray(init_state["pixels"]).copy(), np.asarray(goal_state["goal"]).copy()

    if color != "red":
        rgb = legacy.PURE_RGB[color]
    else:
        rgb = None
    current: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    for i, wrapped in enumerate(world.envs.envs):
        import mujoco

        raw = wrapped.unwrapped
        if rgb is not None:
            _set_cube_rgb(raw, rgb)
        raw.set_state(np.asarray(init_state["qpos"][i]), np.asarray(init_state["qvel"][i]))
        current.append(np.asarray(wrapped.render(), dtype=np.uint8).copy())
        if spec["goal_type"] == "recolor" and recolor_goals is not None:
            # Keep the frozen v2 goal frame contract for blue/yellow.
            goals.append(np.asarray(recolor_goals[i]).copy())
        else:
            cube_qpos = raw._data.joint("object_joint_0").qpos
            cube_qpos[:3] = np.asarray(goal_state["goal_privileged_block_0_pos"][i])
            cube_qpos[3:] = np.asarray(goal_state["goal_privileged_block_0_quat"][i])
            mujoco.mj_forward(raw._model, raw._data)
            goals.append(np.asarray(wrapped.render(), dtype=np.uint8).copy())
        raw.set_state(np.asarray(init_state["qpos"][i]), np.asarray(init_state["qvel"][i]))
        if rgb is not None:
            _set_cube_rgb(raw, rgb)
    return np.stack(current), np.stack(goals)


def _evaluate_condition(
    *, world: Any, dataset: Any, rows: np.ndarray, spec: dict[str, Any], budget: int,
    output: Path, video: bool, recolor_goals: np.ndarray | None,
) -> tuple[dict[str, Any], float]:
    from stable_worldmodel.plot import save_panel_videos

    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    selected = dataset.get_row_data(rows)
    episodes = np.asarray(selected[col])
    starts = np.asarray(selected["step_idx"])
    init_state, goal_state, dataset_videos = legacy._extract_init_goal(
        dataset, episodes, starts, GOAL_OFFSET
    )
    options = _variation_options(spec, len(rows))
    world.reset(seed=init_state.get("seed"), options=options)
    callables = [
        {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
        {"method": "set_target_pos", "args": {
            "cube_id": {"value": 0, "in_dataset": False},
            "target_pos": {"value": "goal_privileged_block_0_pos"},
            "target_quat": {"value": "goal_privileged_block_0_quat"},
        }},
    ]
    merged = {**init_state, **goal_state}
    for i, wrapped in enumerate(world.envs.envs):
        legacy._apply_callables(wrapped.unwrapped, callables, {key: value[i] for key, value in merged.items()})
    current_pixels, goal_pixels = _render_condition_views(world, init_state, goal_state, spec, recolor_goals)
    init_state["pixels"] = current_pixels
    goal_state["goal"] = goal_pixels
    shape_prefix = world.infos["pixels"].shape[:2]
    for state in (init_state, goal_state):
        for key, value in state.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(value[:, None, ...], shape_prefix + value.shape[1:]).copy()
    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}
    successes = np.zeros(len(rows), dtype=bool)
    frames: dict[int, list[np.ndarray]] | None = {i: [] for i in range(len(rows))} if video else None

    def on_step(active_world: Any) -> None:
        active_world.infos.update(deepcopy(goal_snapshot))
        successes[:] |= active_world.terminateds
        if frames is not None:
            for i in range(active_world.num_envs):
                frame = active_world.infos["pixels"][i]
                frames[i].append(np.asarray(frame[-1] if frame.ndim > 3 else frame).copy())

    started = time.time()
    try:
        world._run(max_steps=budget, mode="wait", on_step=on_step)
    finally:
        world.close()
    elapsed = time.time() - started
    if frames is not None:
        save_panel_videos(output / "videos", {"agent": frames, "dataset": dataset_videos, "goal": goal_pixels})
    metrics = {
        "success_rate": float(successes.mean() * 100.0),
        "success_count": int(successes.sum()),
        "num_eval": int(len(rows)),
        "episode_successes": successes,
    }
    return {"metrics": metrics, "episodes": episodes, "starts": starts}, elapsed


def _build_world_and_policy(args: argparse.Namespace, dataset: Any, rows: np.ndarray) -> tuple[Any, Any]:
    import stable_worldmodel as swm
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    selected = dataset.get_row_data(rows)
    episode_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    eval_episodes = np.asarray(selected[episode_key], dtype=np.int64)
    import hdf5plugin  # noqa: F401
    import h5py
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        query_features = np.concatenate(
            [memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows], axis=0
        )
    scaler = memory_legacy._standard_scaler(index)
    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=str(PROJECT)).to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    recorder = legacy.PlanningCostRecorder(len(rows))
    proxy = trust.TrustRegionCostProxy(model, "t2")
    solver_cls = trust.make_trust_region_solver(swm.solver.CEMSolver)
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
    policy = trust.make_trust_policy(swm.policy.WorldModelPolicy)(
        solver=solver,
        config=swm.PlanConfig(horizon=legacy.HORIZON, receding_horizon=legacy.RECEDING_HORIZON, action_block=legacy.ACTION_BLOCK),
        process={"action": scaler},
        transform={"pixels": legacy._image_transform(IMAGE_SIZE), "goal": legacy._image_transform(IMAGE_SIZE)},
        recorder=recorder,
        memory_index=index,
        cost_proxy=proxy,
        cost_recorder=recorder,
        eval_episodes=eval_episodes,
        eval_rows=rows,
        initial_query_features=query_features,
        protocol="t2",
    )
    world = swm.World(env_name="swm/OGBCube-v0", num_envs=len(rows), max_episode_steps=2 * EVAL_BUDGET,
                      image_shape=(IMAGE_SIZE, IMAGE_SIZE), env_type="single", ob_type="states",
                      multiview=False, width=IMAGE_SIZE, height=IMAGE_SIZE, visualize_info=False,
                      terminate_at_goal=True)
    world.set_policy(policy)
    return world, recorder


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    if args.condition == "all":
        conditions = list(CONDITIONS)
    else:
        conditions = [args.condition]
    if args.seed != FORMAL_SEED or args.goal_offset != GOAL_OFFSET or args.eval_budget != EVAL_BUDGET:
        raise ValueError("robust protocol is frozen to seed42/goal_offset25/budget50")
    if args.num_eval not in (2, 50):
        raise ValueError("num-eval is frozen to 2 smoke or 50 formal")
    if args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError("pass --authorize-formal for formal evaluation")
    if not args.dataset.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("dataset or manifest missing")
    if args.self_test:
        for name, spec in CONDITIONS.items():
            opts = _variation_options(spec, 2)
            if opts is not None and len(opts) != 2:
                raise AssertionError(name)
        print(json.dumps({"conditions": list(CONDITIONS), "seed": FORMAL_SEED, "goal_offset": GOAL_OFFSET, "eval_budget": EVAL_BUDGET}, indent=2))
        return 0
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --self-test is used")
    checkpoint = _checkpoint(args.checkpoint)
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Cube robustness evaluation requires CUDA")
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = legacy._formal_rows(dataset, args.manifest)
    rows = formal_rows[: args.num_eval]
    recolor = {}
    if args.num_eval <= 50:
        for color in ("blue", "yellow"):
            recolor[color], _ = legacy._load_recolor_goals(color, args.num_eval)
    summary: dict[str, Any] = {"format_version": "cube_robust_eval_v1", "conditions": {}, "protocol": {
        "seed": FORMAL_SEED, "goal_offset": GOAL_OFFSET, "eval_budget": EVAL_BUDGET,
        "num_eval": args.num_eval, "checkpoint": str(checkpoint), "formal_rows_verified": formal_rows,
        "visual_axes": {name: {"axis": spec["axis"], "variation": spec["variation"], "label": spec["label"]} for name, spec in CONDITIONS.items()},
    }}
    for condition in conditions:
        spec = CONDITIONS[condition]
        output = _output(args.output if len(conditions) == 1 else None, condition, args.num_eval, args.overwrite)
        world, recorder = _build_world_and_policy(args, dataset, rows)
        try:
            result, elapsed = _evaluate_condition(world=world, dataset=dataset, rows=rows, spec=spec,
                                                  budget=args.eval_budget, output=output, video=args.video,
                                                  recolor_goals=recolor.get(spec["color"]))
        finally:
            # _evaluate_condition closes the world after its run.  This close
            # is idempotent in the supported World implementation.
            try:
                world.close()
            except Exception:
                pass
        cost = legacy._save_cost_history(output, recorder, rows, result["episodes"], result["starts"], "mean")
        payload = {"format_version": "cube_robust_condition_v1", "condition": condition, "label": spec["label"],
                   "axis": spec["axis"], "variation": spec["variation"], "protocol": summary["protocol"],
                   "evaluated_rows": rows, "metrics": result["metrics"], "elapsed_seconds": elapsed,
                   "cost_history": cost}
        _write_json(output / "results.json", payload)
        (output / "results.txt").write_text(
            f"condition: {condition}\naxis: {spec['axis']}\nsuccess_rate: {result['metrics']['success_rate']:.6f}\n"
            f"success_count: {result['metrics']['success_count']}/{args.num_eval}\n"
            f"evaluation_time: {elapsed:.6f} seconds\n", encoding="utf-8")
        summary["conditions"][condition] = {"success_rate": result["metrics"]["success_rate"],
                                              "success_count": result["metrics"]["success_count"],
                                              "results": str((output / "results.json").resolve())}
    root = OUTPUT_ROOT.resolve()
    _write_json(root / ("evaluation_summary.json" if args.num_eval == 50 else "smoke/evaluation_summary.json"), summary)
    print(json.dumps(_jsonable(summary), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="red")
    parser.add_argument("--checkpoint", type=Path, help="robust_v1 .pt checkpoint (required except --self-test)")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--index", type=Path, default=PROJECT / "outputs/memory_index/cube_expert_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--goal-offset", type=int, default=GOAL_OFFSET)
    parser.add_argument("--eval-budget", type=int, default=EVAL_BUDGET)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
