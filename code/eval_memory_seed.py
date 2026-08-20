#!/usr/bin/env python3
"""Evaluate Cube LeWM with exact trajectory-memory seeds in CEM slots 1..10.

The base CEM implementation and its mean selector remain unchanged.  A cost
proxy replaces candidates 1..10 immediately at the existing ``get_cost``
boundary, after CEM has consumed its normal ``randn`` stream and forced
candidate 0 to the current mean.  Thus no extra torch randomness is consumed.
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
PROJECT_ROOT = HERE.parent
TOOLS = HERE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_cube_memory_index as memory  # noqa: E402
import eval_ood_color as ood  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs/eval/cube/memory_seed"
REPORT_PATH = OUTPUT_ROOT / "MEMORY_SEED_REPORT.md"
DEFAULT_INDEX = PROJECT_ROOT / "outputs/memory_index/cube_expert_v1"
DEFAULT_CHECKPOINT = "quentinll/lewm-cube"
DERIVED_LABELS = ("coloraug", "maskedaug")
DERIVED_CHECKPOINT_ROOTS = {
    label: PROJECT_ROOT / f"checkpoints/lewm-cube-{label}"
    for label in DERIVED_LABELS
}
COMBO_OUTPUT_ROOTS = {
    label: PROJECT_ROOT / f"outputs/eval/cube/combo_seedX{label}"
    for label in DERIVED_LABELS
}

FORMAL_SEED = 42
NUM_SAMPLES = 300
N_STEPS = 10
TOPK = 30
MEMORY_SLOTS = 10
AUDIT_ENVS = (0, 1, 2, 6, 7, 11, 12, 23, 26, 37, 38, 49)


def _safe_output(path: Path, overwrite: bool, root: Path = OUTPUT_ROOT) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"refusing symlink output: {raw}")
    resolved = raw.resolve()
    root = root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a concrete child of {root}: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is nonempty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _condition(condition: str) -> tuple[str, str]:
    return {
        "red": ("red", "matched"),
        "blue_v2": ("blue", "recolor"),
        "yellow_v2": ("yellow", "recolor"),
    }[condition]


def _checkpoint_contract(
    checkpoint_arg: str | None, derived_label: str | None
) -> tuple[str, dict[str, Any]]:
    """Resolve a base identifier or a strictly labelled local derived weight."""

    if derived_label is None:
        checkpoint = checkpoint_arg or DEFAULT_CHECKPOINT
        candidate = Path(checkpoint).expanduser()
        if candidate.exists():
            raise ValueError(
                "local derived checkpoints require explicit --derived-label "
                f"{DERIVED_LABELS}: {candidate.resolve()}"
            )
        if checkpoint != DEFAULT_CHECKPOINT:
            raise ValueError(
                "unlabelled checkpoint is frozen to the official base model; "
                f"expected={DEFAULT_CHECKPOINT}, actual={checkpoint}"
            )
        return checkpoint, {
            "kind": "official_pretrained_identifier",
            "identifier": checkpoint,
            "derived_label": None,
            "absolute_path": None,
            "sha256": None,
        }

    if derived_label not in DERIVED_LABELS:
        raise ValueError(
            f"derived label must be one of {DERIVED_LABELS}: {derived_label}"
        )
    if checkpoint_arg is None:
        raise ValueError("--derived-label requires an explicit --checkpoint .pt file")
    raw = Path(checkpoint_arg).expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"refusing symlink derived checkpoint: {raw}")
    checkpoint = raw.resolve()
    root = DERIVED_CHECKPOINT_ROOTS[derived_label].resolve()
    if (
        not checkpoint.is_file()
        or checkpoint.suffix != ".pt"
        or root not in checkpoint.parents
    ):
        raise ValueError(
            f"{derived_label} checkpoint must be a .pt file under {root}: {checkpoint}"
        )
    config = checkpoint.parent / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"derived checkpoint config.json missing: {config}")
    # Parse now so malformed training metadata fails before CUDA/output setup.
    json.loads(config.read_text(encoding="utf-8"))
    checkpoint_stat = checkpoint.stat()
    config_stat = config.stat()
    return str(checkpoint), {
        "kind": "derived_local_weights",
        "derived_label": derived_label,
        "absolute_path": str(checkpoint),
        "sha256": memory.sha256_file(checkpoint),
        "size": checkpoint_stat.st_size,
        "mtime_ns": checkpoint_stat.st_mtime_ns,
        "checkpoint_root": str(root),
        "config": {
            "absolute_path": str(config.resolve()),
            "sha256": memory.sha256_file(config.resolve()),
            "size": config_stat.st_size,
            "mtime_ns": config_stat.st_mtime_ns,
        },
        "explicit_opt_in": "--derived-label",
    }


def _output_root(derived_label: str | None) -> Path:
    return (
        OUTPUT_ROOT
        if derived_label is None
        else COMBO_OUTPUT_ROOTS[derived_label]
    )


def _default_output(
    condition: str,
    memory_mode: str,
    num_eval: int,
    derived_label: str | None = None,
) -> Path:
    if derived_label is not None:
        root = COMBO_OUTPUT_ROOTS[derived_label]
        if memory_mode == "slots10" and num_eval == 50:
            return root / condition
        scope = "formal" if num_eval == 50 else "smoke"
        return root / scope / memory_mode / f"{condition}_n{num_eval}"
    if memory_mode == "slots10" and num_eval == 50:
        return OUTPUT_ROOT / f"{condition}_seeded"
    scope = "formal" if num_eval == 50 else "smoke"
    return OUTPUT_ROOT / scope / memory_mode / f"{condition}_n{num_eval}"


def _last_env_value(info: dict[str, Any], env: int, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name not in info:
            continue
        value = np.asarray(info[name][env])
        # World history adds a leading time dimension.  Always use the live
        # last state, retaining the final feature dimension.
        while value.ndim > 1:
            value = value[-1]
        return np.asarray(value)
    raise KeyError(f"none of the live-state keys exists: {names}")


def raw_feature_from_info(info: dict[str, Any], env: int) -> np.ndarray:
    block = _last_env_value(
        info, env, ("privileged/block_0_pos", "privileged_block_0_pos")
    ).reshape(3)
    yaw = float(
        _last_env_value(
            info, env, ("privileged/block_0_yaw", "privileged_block_0_yaw")
        ).reshape(-1)[-1]
    )
    ee = _last_env_value(
        info, env, ("proprio/effector_pos", "proprio_effector_pos")
    ).reshape(3)
    opening = float(
        _last_env_value(
            info, env, ("proprio/gripper_opening", "proprio_gripper_opening")
        ).reshape(-1)[-1]
    )
    return memory.feature_from_state(block, yaw, ee, opening)[0]


class MemoryCostProxy:
    """Delegate model cost after deterministic in-place memory injection."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.pending: list[dict[str, Any]] | None = None
        self.call_index = 0
        self.trace: list[dict[str, Any]] = []
        self.first_cycle_pool: dict[int, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def begin_solve(self, contexts: list[dict[str, Any]]) -> None:
        if self.pending is not None:
            raise RuntimeError("overlapping memory-seeded CEM solves")
        self.pending = contexts
        self.call_index = 0

    def get_cost(self, info_dict: dict, candidates: Any) -> Any:
        if self.pending is None:
            raise RuntimeError("memory cost proxy called without retrieval context")
        env_batch = self.call_index // N_STEPS
        iteration = self.call_index % N_STEPS
        if env_batch >= len(self.pending):
            raise RuntimeError("CEM made more model-cost calls than expected")
        context = self.pending[env_batch]
        seeds = context["seeds"]
        if candidates.shape[0] != 1 or tuple(candidates.shape[1:]) != (300, 5, 25):
            raise RuntimeError(f"unexpected CEM candidate shape: {tuple(candidates.shape)}")
        import torch

        seed_tensor = torch.as_tensor(
            seeds, device=candidates.device, dtype=candidates.dtype
        )
        # Candidate 0 stays the CEM mean.  randn was already consumed by base
        # CEM before this call, so overwriting slots 1..10 is RNG-neutral.
        candidates[:, 1 : 1 + MEMORY_SLOTS].copy_(seed_tensor.unsqueeze(0))
        costs = self.base.get_cost(info_dict, candidates)
        cost_np = costs.detach().cpu().float().numpy()[0]
        order = np.lexsort((np.arange(NUM_SAMPLES), cost_np))
        ranks = np.empty(NUM_SAMPLES, dtype=np.int64)
        ranks[order] = np.arange(1, NUM_SAMPLES + 1)
        elite = set(order[:TOPK].tolist())
        record = {
            "env_idx": context["env_idx"],
            "planning_cycle": context["planning_cycle"],
            "env_step": context["env_step"],
            "cem_iteration": iteration,
            "query_feature_raw": context["query_feature_raw"],
            "query_source": context["query_source"],
            "excluded_eval_episode": context["excluded_eval_episode"],
            "source_rows": context["source_rows"],
            "source_episodes": context["source_episodes"],
            "source_steps": context["source_steps"],
            "retrieval_distances": context["retrieval_distances"],
            "seed_actions_raw": context["seed_actions_raw"],
            "seed_actions_normalized": context["seeds"],
            "candidate_slots": np.arange(1, 11, dtype=np.int64),
            "seed_costs": cost_np[1:11].copy(),
            "seed_latent_ranks_1based": ranks[1:11].copy(),
            "seed_is_top30": np.asarray([i in elite for i in range(1, 11)]),
        }
        self.trace.append(record)
        if (
            iteration == N_STEPS - 1
            and context["planning_cycle"] == 0
            and context["env_idx"] in AUDIT_ENVS
        ):
            self.first_cycle_pool[int(context["env_idx"])] = {
                "dataset_row": int(context["dataset_row"]),
                "eval_episode": int(context["excluded_eval_episode"]),
                "candidates_normalized": candidates.detach().cpu().float().numpy()[0].copy(),
                "latent_costs": cost_np.copy(),
            }
        self.call_index += 1
        if self.call_index == len(self.pending) * N_STEPS:
            self.pending = None
        return costs


def make_memory_policy(base: type) -> type:
    class MemoryPolicy(base):
        def __init__(
            self,
            *args: Any,
            memory_index: memory.CubeMemoryIndex,
            cost_proxy: MemoryCostProxy,
            cost_recorder: Any,
            eval_episodes: np.ndarray,
            eval_rows: np.ndarray,
            initial_query_features: np.ndarray,
            memory_mode: str,
            **kwargs: Any,
        ) -> None:
            self.memory_index = memory_index
            self.cost_proxy = cost_proxy
            self.cost_recorder = cost_recorder
            self.memory_mode = memory_mode
            self.eval_episodes = np.asarray(eval_episodes, dtype=np.int64)
            self.eval_rows = np.asarray(eval_rows, dtype=np.int64)
            self.initial_query_features = np.asarray(
                initial_query_features, dtype=np.float64
            )
            if self.initial_query_features.shape != (len(self.eval_rows), 9):
                raise ValueError(
                    f"invalid initial query feature shape: {self.initial_query_features.shape}"
                )
            self._memory_env_step = 0
            self._memory_cycles = np.zeros(len(self.eval_episodes), dtype=np.int64)
            super().__init__(*args, **kwargs)

        def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
            terminated = info_dict.get("terminated")
            dead = (
                np.asarray(terminated, dtype=bool).reshape(self.env.num_envs, -1)[:, 0]
                if terminated is not None
                else np.zeros(self.env.num_envs, dtype=bool)
            )
            replans = [
                i
                for i in range(self.env.num_envs)
                if not dead[i] and len(self._action_buffer[i]) == 0
            ]
            if replans:
                self.cost_recorder.begin_solve(
                    [(i, self._memory_env_step) for i in replans]
                )
                contexts = []
                for env_idx in replans if self.memory_mode == "slots10" else ():
                    if self._memory_cycles[env_idx] == 0:
                        query = self.initial_query_features[env_idx].copy()
                        query_source = "hdf5_formal_row"
                    else:
                        query = raw_feature_from_info(info_dict, env_idx)
                        query_source = "live_env_info"
                    found = self.memory_index.retrieve(
                        query, exclude_episode=int(self.eval_episodes[env_idx]), count=10
                    )
                    bundle = self.memory_index.action_seed_bundle(found["rows"])
                    contexts.append(
                        {
                            "env_idx": env_idx,
                            "planning_cycle": int(self._memory_cycles[env_idx]),
                            "env_step": self._memory_env_step,
                            "query_feature_raw": query,
                            "query_source": query_source,
                            "excluded_eval_episode": int(self.eval_episodes[env_idx]),
                            "dataset_row": int(self.eval_rows[env_idx]),
                            "source_rows": found["rows"],
                            "source_episodes": found["episodes"],
                            "source_steps": found["steps"],
                            "retrieval_distances": found["distances"],
                            "seed_actions_raw": bundle["raw"],
                            "seeds": bundle["normalized"],
                        }
                    )
                    self._memory_cycles[env_idx] += 1
                if self.memory_mode == "slots10":
                    self.cost_proxy.begin_solve(contexts)
            result = super().get_action(info_dict, **kwargs)
            self._memory_env_step += 1
            return result

    return MemoryPolicy


def _standard_scaler(index: memory.CubeMemoryIndex) -> Any:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = index.action_mean.copy()
    scaler.scale_ = index.action_scale.copy()
    scaler.var_ = np.square(index.action_scale)
    scaler.n_features_in_ = 5
    scaler.n_samples_seen_ = 2_000_000
    return scaler


def _fit_action_scaler(dataset: Any) -> Any:
    from sklearn.preprocessing import StandardScaler

    values = dataset.get_col_data("action")
    values = values[np.isfinite(values).all(axis=1)]
    return StandardScaler().fit(values)


def _save_trace(
    output: Path,
    proxy: MemoryCostProxy,
    recorder: Any,
    evaluated_rows: np.ndarray,
    eval_episodes: np.ndarray,
    scaler: Any,
) -> dict[str, Any]:
    rows = proxy.trace
    payload = {
        "contract": {
            "retrieve_once_per_planning_cycle": True,
            "slots": list(range(1, 11)),
            "candidate0_preserved": True,
            "same_ten_seeds_all_ten_cem_iterations": True,
            "no_extra_torch_randomness": True,
            "cost_rank": "stable_1based_rank_with_candidate_index_tie_break",
        },
        "records": rows,
    }
    ood._write_json(output / "memory_trace.json", payload)
    if rows:
        np.savez_compressed(
            output / "memory_trace.npz",
            env_idx=np.asarray([x["env_idx"] for x in rows]),
            planning_cycle=np.asarray([x["planning_cycle"] for x in rows]),
            env_step=np.asarray([x["env_step"] for x in rows]),
            cem_iteration=np.asarray([x["cem_iteration"] for x in rows]),
            query_source=np.asarray([x["query_source"] for x in rows]),
            query_feature_raw=np.stack([x["query_feature_raw"] for x in rows]),
            source_rows=np.stack([x["source_rows"] for x in rows]),
            source_episodes=np.stack([x["source_episodes"] for x in rows]),
            source_steps=np.stack([x["source_steps"] for x in rows]),
            retrieval_distances=np.stack([x["retrieval_distances"] for x in rows]),
            seed_costs=np.stack([x["seed_costs"] for x in rows]),
            seed_latent_ranks_1based=np.stack(
                [x["seed_latent_ranks_1based"] for x in rows]
            ),
            seed_is_top30=np.stack([x["seed_is_top30"] for x in rows]),
            seed_actions_raw=np.stack([x["seed_actions_raw"] for x in rows]),
            seed_actions_normalized=np.stack(
                [x["seed_actions_normalized"] for x in rows]
            ),
        )
    envs = np.asarray(
        [i for i in AUDIT_ENVS if i < len(evaluated_rows)], dtype=np.int64
    )
    if len(envs):
        np.savez_compressed(
            output / "first_cycle_pool.npz",
            env_indices=envs,
            dataset_rows=np.asarray(evaluated_rows)[envs],
            eval_episodes=np.asarray(eval_episodes)[envs],
            candidates_normalized=np.stack(
                [recorder.records[int(i)][0]["final_candidates_normalized"] for i in envs]
            ),
            latent_costs=np.stack(
                [recorder.records[int(i)][0]["costs"][-1] for i in envs]
            ),
            cem_mean_normalized=np.stack(
                [recorder.records[int(i)][0]["solver_returned_actions_normalized"] for i in envs]
            ),
            action_scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            action_scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        )
    return {"num_iteration_records": len(rows), "num_planning_cycles": len(rows) // 10}


def _update_report() -> None:
    lines = ["# Cube Memory-Seed Evaluation", "", "Mean-selector CEM10; memory slots 1..10.", ""]
    for name in ("red_seeded", "blue_v2_seeded", "yellow_v2_seeded"):
        path = OUTPUT_ROOT / name / "results.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics = data["metrics"]
            lines.append(f"- `{name}`: {metrics['success_count']}/{metrics['num_eval']} ({metrics['success_rate']:.2f}%)")
        else:
            lines.append(f"- `{name}`: pending")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    ood._configure_storage()
    if args.seed != 42 or args.num_eval not in (2, 50):
        raise ValueError("protocol is frozen at seed=42 and num_eval=2/50")
    color, goal_type = _condition(args.condition)
    checkpoint_ref, checkpoint_provenance = _checkpoint_contract(
        args.checkpoint, args.derived_label
    )
    output_root = _output_root(args.derived_label)

    import stable_worldmodel as swm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal memory-seed evaluation requires CUDA")
    if not args.dataset.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("dataset/manifest input missing")
    index = (
        memory.CubeMemoryIndex(args.index, args.dataset)
        if args.memory_mode == "slots10"
        else None
    )
    recolor_goals = recolor_meta = None
    if goal_type == "recolor":
        recolor_goals, recolor_meta = ood._load_recolor_goals(color, args.num_eval)
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    all_rows = ood._formal_rows(dataset, args.manifest)
    rows = all_rows[: args.num_eval]
    selected = dataset.get_row_data(rows)
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    eval_episodes = np.asarray(selected[ep_key], dtype=np.int64)
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(args.dataset, "r", swmr=True) as h5:
        initial_query_features = np.concatenate(
            [memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows],
            axis=0,
        )
    if initial_query_features.shape != (args.num_eval, 9):
        raise RuntimeError(
            f"formal initial query feature shape mismatch: {initial_query_features.shape}"
        )
    # All external inputs and formal row invariants are validated before an
    # intentional overwrite can replace an older derived result directory.
    output = _safe_output(
        args.output
        or _default_output(
            args.condition,
            args.memory_mode,
            args.num_eval,
            args.derived_label,
        ),
        args.overwrite,
        output_root,
    )

    model = swm.wm.utils.load_pretrained(
        checkpoint_ref, cache_dir=str(PROJECT_ROOT)
    )
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    proxy = MemoryCostProxy(model) if args.memory_mode == "slots10" else None
    recorder = ood.PlanningCostRecorder(args.num_eval)
    scaler = _standard_scaler(index) if index is not None else _fit_action_scaler(dataset)
    solver_cls = ood._make_selecting_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=proxy if proxy is not None else model,
        batch_size=1, num_samples=300, var_scale=1.0,
        n_steps=10, topk=30, device=args.device, seed=42,
        callbacks=[recorder], selector="mean", recorder=recorder,
    )
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    policy_cls = make_memory_policy(swm.policy.WorldModelPolicy)
    if proxy is None:
        # In off mode use the existing recorder wrapper only; there is no
        # memory retrieval, model proxy, or candidate mutation in the path.
        policy_cls = ood._make_recording_policy(swm.policy.WorldModelPolicy)
    policy = policy_cls(
        solver=solver,
        config=config,
        process={"action": scaler},
        transform={"pixels": ood._image_transform(224), "goal": ood._image_transform(224)},
        **(
            {
                "memory_index": index,
                "cost_proxy": proxy,
                "cost_recorder": recorder,
                "eval_episodes": eval_episodes,
                "eval_rows": rows,
                "initial_query_features": initial_query_features,
                "memory_mode": args.memory_mode,
            }
            if proxy is not None
            else {"recorder": recorder}
        ),
    )
    world = swm.World(
        env_name="swm/OGBCube-v0", num_envs=args.num_eval,
        max_episode_steps=100, image_shape=(224,224), env_type="single",
        ob_type="states", multiview=False, width=224, height=224,
        visualize_info=False, terminate_at_goal=True,
    )
    world.set_policy(policy)
    started = time.time()
    try:
        metrics, chosen = ood._evaluate(
            world, dataset, rows, goal_type, color, 50,
            output / "videos", recolor_goals,
        )
    finally:
        world.close()
    elapsed = time.time() - started
    trace = (
        _save_trace(output, proxy, recorder, rows, eval_episodes, scaler)
        if proxy is not None
        else {"num_iteration_records": 0, "num_planning_cycles": 0}
    )
    cost_summary = ood._save_cost_history(
        output, recorder, rows, chosen["episodes"], chosen["starts"], "mean"
    )
    payload = {
        "protocol": {
            "condition": args.condition,
            "color": color,
            "goal_type": goal_type,
            "goal_recolor": recolor_meta,
            "selector": "mean",
            "planner": "cem10",
            "num_samples": 300,
            "memory_mode": args.memory_mode,
            "memory_slots": list(range(1,11)) if args.memory_mode == "slots10" else [],
            "candidate0_preserved": True,
            "memory_index": (
                str(Path(args.index).resolve())
                if args.memory_mode == "slots10"
                else None
            ),
            "seed": 42,
            "goal_offset": 25,
            "eval_budget": 50,
            "checkpoint": checkpoint_provenance,
            "combination": {
                "enabled": args.derived_label is not None,
                "derived_label": args.derived_label,
                "output_root": str(output_root.resolve()),
                "memory_seed_protocol_unchanged": True,
                "retrieval_protocol_unchanged": True,
                "cem_rng_stream_unchanged": True,
                "action_selector": "mean",
            },
            "helper_provenance": {
                "eval_ood_color": {
                    "path": str(Path(ood.__file__).resolve()),
                    "sha256": memory.sha256_file(Path(ood.__file__).resolve()),
                },
                "build_cube_memory_index": {
                    "path": str(Path(memory.__file__).resolve()),
                    "sha256": memory.sha256_file(Path(memory.__file__).resolve()),
                },
                "index_metadata_sha256": (
                    memory.sha256_file(Path(args.index).resolve() / "metadata.json")
                    if index is not None
                    else None
                ),
            },
            "console_capture_expected": "launcher should pipe stdout/stderr through tee into this output directory",
        },
        "formal_rows_verified": all_rows,
        "evaluated_rows": rows,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "memory_trace": trace,
        "cost_history": cost_summary,
    }
    ood._write_json(output / "results.json", payload)
    success_text = ", ".join("True" if x else "False" for x in metrics["episode_successes"])
    (output / "results.txt").write_text(
        "==== CONFIG ====\n" + json.dumps(ood._jsonable(payload["protocol"]), indent=2, sort_keys=True)
        + "\n\n==== RESULTS ====\n"
        + f"success_rate: {metrics['success_rate']:.6f}\n"
        + f"success_count: {metrics['success_count']}/{metrics['num_eval']}\n"
        + f"episode_successes: [{success_text}]\n"
        + f"evaluation_time: {elapsed:.6f} seconds\n",
        encoding="utf-8",
    )
    if args.num_eval == 50 and args.derived_label is None:
        _update_report()
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cube formal memory-seeded LeWM evaluation")
    parser.add_argument("--condition", choices=("red","blue_v2","yellow_v2"), required=True)
    parser.add_argument("--memory-mode", choices=("off","slots10"), required=True)
    parser.add_argument("--num-eval", type=int, choices=(2,50), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, default=ood.DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=ood.DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--checkpoint",
        help=(
            f"base defaults to {DEFAULT_CHECKPOINT}; a derived combination "
            "requires an explicit local .pt file"
        ),
    )
    parser.add_argument(
        "--derived-label",
        choices=DERIVED_LABELS,
        help=(
            "strictly labels a Memory Seed x derived-checkpoint combination and "
            "selects its checkpoint/output roots"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
