#!/usr/bin/env python3
"""Target-space OOD curve for the frozen Cube MaskedAug + T2 evaluator.

The start rows are the verified seed-42 formal rows (all 50 evaluation
episodes remain excluded from target-frame retrieval).  Each tier selects
real HDF5 frames whose block position is nearest to the requested planar
distance outside the training target box.  The requested distance is allowed
to be unattainable by the dataset; the selected distance and shortfall are
recorded in ``target_selection.json`` rather than being silently synthesized.

This entry point reuses the frozen Trust-Region CEM, memory index, action
scaler, and world-model policy.  It changes only the goal pixel/pose supplied
to the environment.  Formal execution is intentionally explicit via
``--authorize-formal`` and is normally launched by the Leader after the
robustness checkpoint is ready.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_cube_memory_index as memory  # noqa: E402
import cube_trust_region_common as common  # noqa: E402
import eval_memory_seed as legacy  # noqa: E402
import eval_ood_color as ood  # noqa: E402
import eval_trust_region as trust  # noqa: E402


TRAINING_BOX = np.asarray([[0.30, 0.55], [-0.30, 0.30]], dtype=np.float64)
TIERS: dict[str, float] = {
    "in_box": 0.0,
    "plus_05cm": 0.05,
    "plus_10cm": 0.10,
    "plus_20cm": 0.20,
}


def _configure_storage() -> None:
    root = common.PROJECT_ROOT
    os.environ.setdefault("STABLEWM_HOME", str(root))
    os.environ.setdefault("HF_HOME", str(root.parent / ".cache/huggingface"))
    os.environ.setdefault("TORCH_HOME", str(root.parent / ".cache/torch"))
    os.environ.setdefault("TMPDIR", str(root.parent / "tmp"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    (root.parent / "tmp").mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_curve_artifacts(root: Path, results: dict[str, dict[str, Any]]) -> None:
    """Persist a machine-readable CSV and a small report-ready PNG."""

    import csv

    rows = [
        {
            "tier": tier,
            "requested_distance_m": TIERS[tier],
            "selected_distance_median": values["target_distance_median"],
            "selected_shortfall_median_m": values["target_shortfall_median"],
            "success_rate_percent": values["success_rate"],
            "success_count": values["success_count"],
        }
        for tier, values in results.items()
    ]
    with (root / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        requested = [row["requested_distance_m"] * 100.0 for row in rows]
        observed = [row["selected_distance_median"] * 100.0 for row in rows]
        success = [row["success_rate_percent"] for row in rows]
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
        ax.plot(observed, success, marker="o", label="observed frame distance")
        ax.set_xticks(observed, [f"{value:.0f}" for value in requested])
        ax.set_xlabel("Requested OOD distance (cm)")
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(root / "success_vs_ood_distance.png")
        plt.close(fig)
    except Exception as error:  # report artifact should not invalidate results
        (root / "curve_plot_warning.txt").write_text(str(error) + "\n", encoding="utf-8")


def _distance_to_box(xy: np.ndarray) -> np.ndarray:
    """Euclidean distance in XY to the closed training target rectangle."""

    values = np.asarray(xy, dtype=np.float64)
    dx = np.maximum(np.maximum(TRAINING_BOX[0, 0] - values[:, 0], 0.0), values[:, 0] - TRAINING_BOX[0, 1])
    dy = np.maximum(np.maximum(TRAINING_BOX[1, 0] - values[:, 1], 0.0), values[:, 1] - TRAINING_BOX[1, 1])
    return np.hypot(dx, dy)


def _select_target_rows(
    dataset_path: Path,
    start_rows: np.ndarray,
    tier: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick one real, non-formal episode frame per evaluation environment."""

    import hdf5plugin  # noqa: F401
    import h5py

    requested = float(TIERS[tier])
    with h5py.File(dataset_path, "r", swmr=True) as h5:
        all_episodes = np.asarray(h5["ep_idx"], dtype=np.int64)
        starts_episodes = all_episodes[np.asarray(start_rows, dtype=np.int64)]
        excluded = frozenset(int(value) for value in starts_episodes)
        positions = np.asarray(h5["privileged_block_0_pos"], dtype=np.float64)

    distances = _distance_to_box(positions[:, :2])
    in_box = distances <= 1e-12
    eligible = ~np.isin(all_episodes, np.asarray(sorted(excluded), dtype=np.int64))
    if tier == "in_box":
        eligible &= in_box
    else:
        eligible &= ~in_box
    rows = np.flatnonzero(eligible).astype(np.int64)
    if len(rows) < len(start_rows):
        raise RuntimeError(f"insufficient real target rows for {tier}: {len(rows)} < {len(start_rows)}")

    # Stable nearest-to-request ranking, then deterministic seeded tie rotation.
    error = np.abs(distances[rows] - requested)
    rng = np.random.default_rng(int(seed))
    tie = rng.random(len(rows))
    order = np.lexsort((rows, tie, error))
    chosen: list[int] = []
    chosen_episodes: set[int] = set()
    for index in order:
        row = int(rows[int(index)])
        episode = int(all_episodes[row])
        if episode in chosen_episodes:
            continue
        chosen.append(row)
        chosen_episodes.add(episode)
        if len(chosen) == len(start_rows):
            break
    if len(chosen) != len(start_rows):
        raise RuntimeError(
            f"insufficient distinct target episodes for {tier}: "
            f"expected={len(start_rows)}, actual={len(chosen)}"
        )
    selected = np.asarray(chosen, dtype=np.int64)
    selected_distances = distances[selected]
    available_distances = distances[rows]
    requested_reachable = bool(
        requested >= float(available_distances.min()) - 1e-9
        and requested <= float(available_distances.max()) + 1e-9
    )
    metadata = {
        "tier": tier,
        "requested_distance_m": requested,
        "distance_definition": "planar Euclidean distance to closed target box",
        "training_target_box_xy_m": TRAINING_BOX.tolist(),
        "selected_rows": selected,
        "selected_episodes": all_episodes[selected],
        "selected_distances_m": selected_distances,
        "selected_distance_min_m": float(selected_distances.min()),
        "selected_distance_median_m": float(np.median(selected_distances)),
        "selected_distance_max_m": float(selected_distances.max()),
        "absolute_shortfall_median": float(np.median(np.abs(selected_distances - requested))),
        "available_distance_min_m": float(available_distances.min()),
        "available_distance_max_m": float(available_distances.max()),
        "requested_within_observed_range": requested_reachable,
        "eligible_rows": int(len(rows)),
        "eligible_distinct_episodes": int(len(np.unique(all_episodes[rows]))),
        "excluded_eval_episodes": sorted(excluded),
        "selection_seed": int(seed),
        "real_hdf5_frames": True,
        "fallback_used": not requested_reachable,
    }
    return selected, metadata


def _row_states(dataset: Any, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Get HDF5 rows while preserving the evaluator's HWC image contract."""

    requested = np.asarray(rows, dtype=np.int64)
    order = np.argsort(requested, kind="stable")
    sorted_rows = requested[order]
    values = dataset.get_row_data(sorted_rows)
    inverse = np.argsort(order, kind="stable")
    return {key: np.asarray(value)[inverse].copy() for key, value in values.items()}


def _make_goal_state(target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        ("goal" if key == "pixels" else f"goal_{key}"): value
        for key, value in target.items()
        if key in {"pixels", "privileged_block_0_pos", "privileged_block_0_quat"}
    }


def _evaluate_tier(
    *,
    dataset: Any,
    rows: np.ndarray,
    target_rows: np.ndarray,
    eval_episodes: np.ndarray,
    initial_query_features: np.ndarray,
    index: memory.CubeMemoryIndex,
    model: Any,
    scaler: Any,
    output: Path,
    budget: int,
    protocol: str,
    video: bool,
) -> dict[str, Any]:
    import stable_worldmodel as swm
    import torch

    init_state = _row_states(dataset, rows)
    goal_state = _make_goal_state(_row_states(dataset, target_rows))
    num_eval = len(rows)
    recorder = ood.PlanningCostRecorder(num_eval)
    proxy = trust.TrustRegionCostProxy(model, protocol)
    solver_cls = trust.make_trust_region_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=proxy,
        batch_size=1,
        num_samples=common.NUM_SAMPLES,
        var_scale=common.PROTOCOL_SPECS[protocol]["var_scale"],
        n_steps=common.N_STEPS,
        topk=common.TOPK,
        device="cuda",
        seed=common.FORMAL_SEED,
        callbacks=[recorder],
        selector="mean",
        recorder=recorder,
        trust_protocol=protocol,
    )
    config = swm.PlanConfig(horizon=common.HORIZON, receding_horizon=common.HORIZON, action_block=common.ACTION_BLOCK)
    policy = trust.make_trust_policy(swm.policy.WorldModelPolicy)(
        solver=solver,
        config=config,
        process={"action": scaler},
        transform={"pixels": ood._image_transform(224), "goal": ood._image_transform(224)},
        memory_index=index,
        cost_proxy=proxy,
        cost_recorder=recorder,
        eval_episodes=eval_episodes,
        eval_rows=rows,
        initial_query_features=initial_query_features,
        protocol=protocol,
    )
    world = swm.World(
        env_name="swm/OGBCube-v0", num_envs=num_eval, max_episode_steps=2 * budget,
        image_shape=(224, 224), env_type="single", ob_type="states", multiview=False,
        width=224, height=224, visualize_info=False, terminate_at_goal=True,
    )
    world.set_policy(policy)
    world.reset(seed=init_state.get("seed"), options=None)
    merged = {**init_state, **goal_state}
    for i, wrapped in enumerate(world.envs.envs):
        raw = wrapped.unwrapped
        raw.set_state(init_state["qpos"][i], init_state["qvel"][i])
        raw.set_target_pos(
            cube_id=0,
            target_pos=goal_state["goal_privileged_block_0_pos"][i],
            target_quat=goal_state["goal_privileged_block_0_quat"][i],
        )
    shape_prefix = world.infos["pixels"].shape[:2]
    for state in (init_state, goal_state):
        for key, value in state.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(value[:, None, ...], shape_prefix + value.shape[1:]).copy()
    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}
    successes = np.zeros(num_eval, dtype=bool)
    frames = {i: [] for i in range(num_eval)} if video else None

    def on_step(active_world: Any) -> None:
        active_world.infos.update(deepcopy(goal_snapshot))
        successes[:] |= active_world.terminateds
        if frames is not None:
            for i in range(num_eval):
                frame = active_world.infos["pixels"][i]
                frames[i].append(np.asarray(frame[-1] if frame.ndim > 3 else frame).copy())

    started = time.time()
    try:
        world._run(max_steps=budget, mode="wait", on_step=on_step)
    finally:
        world.close()
    elapsed = time.time() - started
    trace = trust._save_trace(output, proxy)
    cost_history = ood._save_cost_history(output, recorder, rows, eval_episodes, init_state["step_idx"], "mean")
    if frames is not None:
        from stable_worldmodel.plot import save_panel_videos
        save_panel_videos(output / "videos", {"agent": frames, "goal": goal_state["goal"]})
    metrics = {
        "success_rate": float(successes.mean() * 100.0),
        "success_count": int(successes.sum()),
        "num_eval": num_eval,
        "episode_successes": successes,
    }
    return {"metrics": metrics, "elapsed_seconds": elapsed, "trace": trace, "cost_history": cost_history}


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    if args.seed != common.FORMAL_SEED:
        raise ValueError("seed is frozen at 42")
    if args.num_eval not in (2, 50):
        raise ValueError("num-eval is frozen to 2 smoke or 50 formal")
    if args.protocol != "t2":
        raise ValueError("this benchmark is frozen to T2")
    if args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError("pass --authorize-formal for the 50-env evaluation")
    if not args.dataset.is_file() or not args.manifest.is_file() or not args.index.is_dir():
        raise FileNotFoundError("dataset, manifest, or memory index missing")
    checkpoint = args.checkpoint.expanduser().resolve()
    frozen_checkpoint = common.MASKED_CHECKPOINT.resolve()
    if checkpoint != frozen_checkpoint:
        raise ValueError(
            "target-space OOD benchmark is frozen to MaskedAug weights: "
            f"expected={frozen_checkpoint}, actual={checkpoint}"
        )
    checkpoint_contract = common.frozen_masked_checkpoint_contract()
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal OOD evaluation requires CUDA")
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = ood._formal_rows(dataset, args.manifest)
    rows = formal_rows[: args.num_eval]
    selected = dataset.get_row_data(rows)
    eval_episodes = np.asarray(selected["ep_idx"], dtype=np.int64)
    import h5py
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        initial_query_features = np.concatenate([memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows], axis=0)
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    model = swm.wm.utils.load_pretrained(checkpoint, cache_dir=str(common.PROJECT_ROOT)).to("cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    scaler = legacy._standard_scaler(index)
    root = args.output or (common.PROJECT_ROOT / "outputs/eval/cube/goal_ood_curve")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for tier_index, tier in enumerate(TIERS):
        target_rows, selection = _select_target_rows(args.dataset, rows, tier, args.seed + tier_index)
        tier_output = root / tier
        if tier_output.exists() and any(tier_output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"non-empty output: {tier_output}; pass --overwrite")
        tier_output.mkdir(parents=True, exist_ok=True)
        _write_json(tier_output / "target_selection.json", selection)
        result = _evaluate_tier(
            dataset=dataset, rows=rows, target_rows=target_rows, eval_episodes=eval_episodes,
            initial_query_features=initial_query_features, index=index, model=model, scaler=scaler,
            output=tier_output, budget=50, protocol=args.protocol, video=args.video,
        )
        payload = {
            "format_version": "cube_goal_ood_t2_v1",
            "protocol": {"id": "t2", "seed": args.seed, "budget": 50, "checkpoint": checkpoint_contract, "fixed50_exclusion": True, "target_source": "real_hdf5_frame"},
            "tier": tier,
            "formal_rows_verified": formal_rows,
            "evaluated_rows": rows,
            "target_rows": target_rows,
            **result,
        }
        _write_json(tier_output / "results.json", payload)
        (tier_output / "results.txt").write_text(f"tier: {tier}\nsuccess_rate: {result['metrics']['success_rate']:.6f}\nsuccess_count: {result['metrics']['success_count']}/{args.num_eval}\n", encoding="utf-8")
        all_results[tier] = {"success_rate": result["metrics"]["success_rate"], "success_count": result["metrics"]["success_count"], "target_distance_median": selection["selected_distance_median_m"], "target_shortfall_median": selection["absolute_shortfall_median"]}
    _write_json(root / "curve.json", {"format_version": "cube_goal_ood_curve_v1", "tiers": all_results, "seed": args.seed, "formal_rows": formal_rows})
    _write_curve_artifacts(root, all_results)
    print(root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=common.DATASET)
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST)
    parser.add_argument("--index", type=Path, default=common.MEMORY_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=common.MASKED_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", default="t2")
    parser.add_argument("--num-eval", type=int, default=2, choices=(2, 50))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test(args: argparse.Namespace) -> int:
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = ood._formal_rows(dataset, args.manifest)
    rows = formal_rows[:2]
    seen = set(int(x) for x in dataset.get_row_data(rows)["ep_idx"])
    for index, tier in enumerate(TIERS):
        selected, meta = _select_target_rows(args.dataset, rows, tier, 42 + index)
        episodes = set(int(x) for x in meta["selected_episodes"])
        assert len(selected) == 2 and not (seen & episodes)
        assert meta["real_hdf5_frames"] and np.isfinite(meta["selected_distances_m"]).all()
        target = _row_states(dataset, selected)
        assert target["pixels"].shape == (2, 224, 224, 3)
        assert np.isfinite(target["privileged_block_0_pos"]).all()
    print(json.dumps({"formal_rows": formal_rows.tolist(), "tiers": list(TIERS), "status": "ok"}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
