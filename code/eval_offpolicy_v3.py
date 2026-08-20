#!/usr/bin/env python3
"""Evaluate frozen T2 with an all-AND-gated off-policy V3 checkpoint.

The planner, retrieval, injection, RNG streams, selector, success criterion,
formal row order, and artifact writers are imported unchanged from the proven
T2 evaluation.  Relative to the existing T2 baseline, the world-model
checkpoint is the only experimental factor.  A 50-environment run requires
the reproducible aggregate PASS gate from ``evaluate_cube_offpolicy_v3.py``;
two-environment smoke outputs are isolated below ``offpolicy_v3/smoke``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
AILAB_ROOT = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_cube_memory_index as memory  # noqa: E402
import cube_cem_audit as audit  # noqa: E402
import cube_trust_region_common as trust_common  # noqa: E402
import eval_memory_seed as legacy  # noqa: E402
import eval_ood_color as ood  # noqa: E402
import eval_trust_region as trust  # noqa: E402
import evaluate_cube_offpolicy_v3 as offline  # noqa: E402


OUTPUT_ROOT = AILAB_ROOT / "outputs/eval/cube/offpolicy_v3"
DEFAULT_CHECKPOINT = offline.DEFAULT_CHECKPOINT
DEFAULT_GATE = OUTPUT_ROOT / "offline/gate.json"
CONDITIONS = offline.CONDITIONS
PROTOCOL = "t2"


def _configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(AILAB_ROOT),
        "HF_HOME": str(AILAB_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(AILAB_ROOT.parent / "tmp"),
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        os.environ[key] = value
    (AILAB_ROOT.parent / "tmp").mkdir(parents=True, exist_ok=True)


def _default_output(condition: str, num_eval: int) -> Path:
    return OUTPUT_ROOT / condition if num_eval == 50 else OUTPUT_ROOT / "smoke" / condition


def _prepare_output(path: Path, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a concrete child of {root}: {resolved}")
    if resolved.is_symlink():
        raise ValueError(f"refusing symlink output: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output exists but is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is nonempty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _protocol_output(path: Path, condition: str, num_eval: int) -> Path:
    """Keep smoke and formal artifacts in non-overlapping frozen namespaces."""
    resolved = path.expanduser().resolve()
    if num_eval == 50:
        expected = (OUTPUT_ROOT / condition).resolve()
        if resolved != expected:
            raise ValueError(f"formal output is frozen: expected={expected}, actual={resolved}")
    else:
        smoke_root = (OUTPUT_ROOT / "smoke").resolve()
        if resolved == smoke_root or smoke_root not in resolved.parents:
            raise ValueError(f"2-env smoke output must be below {smoke_root}: {resolved}")
    return resolved


def _checkpoint_provenance(checkpoint: Path) -> dict[str, Any]:
    resolved = offline._checkpoint(checkpoint)
    config = resolved.parent / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"V3 checkpoint config missing: {config}")
    return {
        "kind": "offpolicy_v3_single_arm_expert_rollout_disabled_finetune",
        "run_id": resolved.parent.name,
        "weights": offline.v1._identity(resolved),
        "config": offline.v1._identity(config),
    }


def _validate_formal_gate(
    gate_path: Path,
    checkpoint: Path,
    condition: str,
    dataset: Path,
    manifest: Path,
    index: Path,
) -> dict[str, Any]:
    gate = offline._validate_gate(
        gate_path,
        checkpoint,
        dataset=dataset,
        manifest=manifest,
        index=index,
    )
    if gate["colors"][condition]["status"] != "PASS":
        raise ValueError(f"aggregate PASS lacks requested condition: {condition}")
    if gate["expert_manifold"]["status"] != "PASS":
        raise ValueError("aggregate PASS lacks expert-manifold PASS")
    return {
        "artifact": offline.v1._identity(gate_path),
        "format_version": gate["format_version"],
        "status": gate["status"],
        "all_requirements_AND": gate["all_requirements_AND"],
        "all_colors_required": gate["all_colors_required"],
        "colors": gate["colors"],
        "training_expert_stopline": gate["training_expert_stopline"],
        "expert_manifold": gate["expert_manifold"],
        "training_provenance": gate["artifacts"]["training_provenance.json"],
    }


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    if args.seed != trust_common.FORMAL_SEED:
        raise ValueError(
            f"T2 seed is frozen: expected={trust_common.FORMAL_SEED}, actual={args.seed}"
        )
    if args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError("formal 50-env evaluation requires --authorize-formal")
    if not args.dataset.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("dataset/manifest input missing")
    checkpoint = offline._checkpoint(args.checkpoint)
    checkpoint_provenance = _checkpoint_provenance(checkpoint)
    gate = (
        _validate_formal_gate(
            args.gate,
            checkpoint,
            args.condition,
            args.dataset,
            args.manifest,
            args.index,
        )
        if args.num_eval == 50
        else None
    )

    import hdf5plugin  # noqa: F401
    import h5py
    import stable_worldmodel as swm
    import torch
    from gymnasium.spaces import Box

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("T2 online evaluation requires a CUDA device")
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    all_rows = ood._formal_rows(dataset, args.manifest)
    rows = all_rows[: args.num_eval]
    selected = dataset.get_row_data(rows)
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    eval_episodes = np.asarray(selected[ep_key], dtype=np.int64)
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        initial_query_features = np.concatenate(
            [memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows], axis=0
        )

    color, goal_type, audit_color = trust_common.condition_visual(args.condition)
    recolor_goals = recolor_meta = None
    if goal_type == "recolor":
        recolor_goals, recolor_meta = ood._load_recolor_goals(color, args.num_eval)
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        raw_inputs = audit._load_capture_inputs(h5, rows, audit_color, recolor_goals)

    model = swm.wm.utils.load_pretrained(str(checkpoint), cache_dir=str(AILAB_ROOT))
    model = model.to(args.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    scaler = legacy._standard_scaler(index)
    requested_output = args.output or _default_output(args.condition, args.num_eval)
    output = _prepare_output(
        _protocol_output(requested_output, args.condition, args.num_eval), args.overwrite
    )

    # These are exact existing T2 implementation objects.  This file adds no
    # local planner, retrieval, injection, selector, or RNG behavior.
    proxy = trust.TrustRegionCostProxy(model, PROTOCOL)
    recorder = ood.PlanningCostRecorder(args.num_eval)
    solver_cls = trust.make_trust_region_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=proxy,
        batch_size=1,
        num_samples=trust_common.NUM_SAMPLES,
        var_scale=trust_common.PROTOCOL_SPECS[PROTOCOL]["var_scale"],
        n_steps=trust_common.N_STEPS,
        topk=trust_common.TOPK,
        device=args.device,
        seed=trust_common.FORMAL_SEED,
        callbacks=[recorder],
        selector="mean",
        recorder=recorder,
        trust_protocol=PROTOCOL,
    )
    config = swm.PlanConfig(
        horizon=trust_common.HORIZON,
        receding_horizon=trust_common.HORIZON,
        action_block=trust_common.ACTION_BLOCK,
    )
    action_space = Box(
        low=np.broadcast_to(-np.inf, (args.num_eval, trust_common.ACTION_DIM)),
        high=np.broadcast_to(np.inf, (args.num_eval, trust_common.ACTION_DIM)),
        dtype=np.float32,
    )
    solver.configure(action_space=action_space, n_envs=args.num_eval, config=config)
    policy_cls = trust.make_trust_policy(swm.policy.WorldModelPolicy)
    policy = policy_cls(
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
        protocol=PROTOCOL,
    )
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=args.num_eval,
        max_episode_steps=100,
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
    started = time.time()
    try:
        metrics, chosen = ood._evaluate(
            world,
            dataset,
            rows,
            goal_type,
            color,
            50,
            output / "videos",
            recolor_goals,
        )
    finally:
        world.close()
    elapsed = time.time() - started
    trace = trust._save_trace(output, proxy)
    cost_history = ood._save_cost_history(
        output, recorder, rows, chosen["episodes"], chosen["starts"], "mean"
    )
    pools = trust._save_first_cycle_pools(
        output,
        proxy,
        recorder,
        rows,
        eval_episodes,
        raw_inputs,
        args.dataset,
        scaler,
        PROTOCOL,
        args.condition,
    )
    payload = {
        "format_version": "cube_offpolicy_v3_t2_evaluation_v1",
        "protocol": {
            "id": PROTOCOL,
            **trust_common.PROTOCOL_SPECS[PROTOCOL],
            "condition": args.condition,
            "color": color,
            "goal_type": goal_type,
            "goal_recolor": recolor_meta,
            "selector": "legacy_updated_elite_mean",
            "num_samples": trust_common.NUM_SAMPLES,
            "n_steps": trust_common.N_STEPS,
            "topk": trust_common.TOPK,
            "seed": trust_common.FORMAL_SEED,
            "formal_full_50_order": args.num_eval == 50,
            "checkpoint": checkpoint_provenance,
            "aggregate_offline_gate": gate,
            "memory_index": trust_common.file_identity(Path(args.index) / "metadata.json"),
            "torch_rng": "unchanged legacy CEM stream; no extra torch draws",
            "noise_rng": (
                "unchanged T2 independent CPU torch.Generator; SHA256-derived from "
                "(base42,dataset_row,planning_cycle), sampled once/cycle and reused"
            ),
            "only_experimental_factor_vs_existing_T2": "world_model_checkpoint",
            "helper_provenance": {
                "eval_trust_region": trust_common.file_identity(Path(trust.__file__)),
                "eval_memory_seed": trust_common.file_identity(Path(legacy.__file__)),
                "eval_ood_color": trust_common.file_identity(Path(ood.__file__)),
                "cube_cem_audit": trust_common.file_identity(Path(audit.__file__)),
                "build_cube_memory_index": trust_common.file_identity(Path(memory.__file__)),
                "offline_gate_evaluator": trust_common.file_identity(Path(offline.__file__)),
                "this_evaluator": trust_common.file_identity(Path(__file__)),
            },
        },
        "formal_rows_verified": all_rows,
        "evaluated_rows": rows,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "trace": trace,
        "cost_history": cost_history,
        "first_cycle_pools": pools,
    }
    trust_common.write_json(output / "results.json", payload)
    successes = ", ".join(
        "True" if value else "False" for value in metrics["episode_successes"]
    )
    (output / "results.txt").write_text(
        f"protocol: {PROTOCOL}\ncondition: {args.condition}\n"
        f"success_rate: {metrics['success_rate']:.6f}\n"
        f"success_count: {metrics['success_count']}/{metrics['num_eval']}\n"
        f"episode_successes: [{successes}]\n"
        f"elapsed_seconds: {elapsed:.6f}\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--seed", type=int, default=trust_common.FORMAL_SEED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--dataset", type=Path, default=trust_common.DATASET)
    parser.add_argument("--manifest", type=Path, default=trust_common.MANIFEST)
    parser.add_argument("--index", type=Path, default=trust_common.MEMORY_INDEX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
