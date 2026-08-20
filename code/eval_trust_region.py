#!/usr/bin/env python3
"""Independent Cube Trust-Region evaluation for frozen T1/T2 CEM protocols.

T1 starts every planning cycle at the nearest episode-excluded trajectory
memory seed with ``var_scale=0.2``.  T2 retains the standard zero/unit CEM
distribution, overwrites slots 1..10 with the ten memory seeds, and overwrites
slots 11..30 with two deterministic sigma-0.1 clipped variants per seed.

The legacy CEM mean remains the executed action selector.  Formal evaluation
always solves all 50 environments in their original order; only the frozen 12
audit environments are exported as first-cycle candidate pools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_cube_memory_index as memory  # noqa: E402
import cube_cem_audit as audit  # noqa: E402
import cube_trust_region_common as common  # noqa: E402
import eval_memory_seed as legacy  # noqa: E402
import eval_ood_color as ood  # noqa: E402


def make_trust_region_solver(base: type) -> type:
    """Bind the frozen T1/T2 distribution entry contract to a local solver."""

    selecting = ood._make_selecting_solver(base)

    class TrustRegionCEMSolver(selecting):
        def __init__(self, *args: Any, trust_protocol: str, **kwargs: Any) -> None:
            if trust_protocol not in common.PROTOCOLS:
                raise ValueError(trust_protocol)
            expected = common.PROTOCOL_SPECS[trust_protocol]["var_scale"]
            actual = float(kwargs.get("var_scale", np.nan))
            if actual != expected:
                raise ValueError(
                    f"{trust_protocol.upper()} initial std mismatch: "
                    f"expected={expected}, actual={actual}"
                )
            self.trust_protocol = trust_protocol
            super().__init__(*args, **kwargs)

        def solve(self, info_dict: dict[str, Any], init_action: Any = None) -> dict[str, Any]:
            if self.trust_protocol == "t1" and init_action is None:
                raise RuntimeError("T1 requires the nearest normalized seed as initial mean")
            if self.trust_protocol == "t2" and init_action is not None:
                raise RuntimeError("T2 requires the standard zero initial mean")
            return super().solve(info_dict, init_action=init_action)

    TrustRegionCEMSolver.__name__ = "TrustRegionCEMSolver"
    return TrustRegionCEMSolver


class TrustRegionCostProxy:
    """Apply T2 injection after torch sampling and retain complete candidates."""

    def __init__(self, base: Any, protocol: str) -> None:
        if protocol not in common.PROTOCOLS:
            raise ValueError(protocol)
        self.base = base
        self.protocol = protocol
        self.pending: list[dict[str, Any]] | None = None
        self.call_index = 0
        self.trace: list[dict[str, Any]] = []
        self.first_cycle_final: dict[int, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def begin_solve(self, contexts: list[dict[str, Any]]) -> None:
        if self.pending is not None:
            raise RuntimeError("overlapping Trust-Region CEM solves")
        self.pending = contexts
        self.call_index = 0

    def get_cost(self, info_dict: dict[str, Any], candidates: Any) -> Any:
        if self.pending is None:
            raise RuntimeError("Trust-Region cost proxy called without context")
        env_batch = self.call_index // common.N_STEPS
        iteration = self.call_index % common.N_STEPS
        if env_batch >= len(self.pending):
            raise RuntimeError("CEM made more model-cost calls than expected")
        context = self.pending[env_batch]
        expected = (1, common.NUM_SAMPLES, common.HORIZON, common.ACTION_BLOCK * common.ACTION_DIM)
        if tuple(candidates.shape) != expected:
            raise RuntimeError(
                f"unexpected CEM candidate shape: expected={expected}, actual={tuple(candidates.shape)}"
            )
        before_slots = candidates[0, 1:31].detach().cpu().float().numpy().copy()
        candidate0_before = candidates[0, 0].detach().cpu().float().numpy().copy()
        seeds = np.asarray(context["seed_actions_normalized"], dtype=np.float32)
        noise = {
            "parent_indices": np.full(common.NOISY_SLOTS, -1, dtype=np.int64),
            "derived_seed": np.asarray(-1, dtype=np.int64),
            "seed_components": np.full(3, -1, dtype=np.int64),
            "noise_raw": np.zeros((2, common.MEMORY_SLOTS, common.HORIZON, 25), dtype=np.float32),
            "unclipped_raw": np.zeros((2, common.MEMORY_SLOTS, common.HORIZON, 25), dtype=np.float32),
            "clipped_raw": np.zeros((2, common.MEMORY_SLOTS, common.HORIZON, 25), dtype=np.float32),
            "normalized": np.zeros((2, common.MEMORY_SLOTS, common.HORIZON, 25), dtype=np.float32),
            "clip_mask": np.zeros((2, common.MEMORY_SLOTS, common.HORIZON, 25), dtype=bool),
        }
        if self.protocol == "t1":
            if iteration == 0 and not np.array_equal(candidate0_before, seeds[0]):
                raise RuntimeError(
                    "T1 initial candidate0 is not the nearest memory seed: "
                    f"env={context['env_idx']}, cycle={context['planning_cycle']}"
                )
        else:
            if iteration == 0 and not np.array_equal(
                candidate0_before, np.zeros_like(candidate0_before)
            ):
                raise RuntimeError(
                    "T2 first-iteration candidate0 must retain the standard zero mean"
                )
            # Generated exactly once by the policy for this planning cycle;
            # the same 20 local variants are reused across all 10 iterations.
            noise = context["noise_bundle"]
            import torch

            exact_tensor = torch.as_tensor(
                seeds, device=candidates.device, dtype=candidates.dtype
            )
            noisy_tensor = torch.as_tensor(
                noise["normalized"].reshape(common.NOISY_SLOTS, common.HORIZON, 25),
                device=candidates.device,
                dtype=candidates.dtype,
            )
            candidates[:, 1:11].copy_(exact_tensor.unsqueeze(0))
            candidates[:, 11:31].copy_(noisy_tensor.unsqueeze(0))
            if not np.array_equal(
                candidates[0, 1:11].detach().cpu().float().numpy(), seeds
            ):
                raise RuntimeError("T2 exact seed injection changed values")
        candidate0_after = candidates[0, 0].detach().cpu().float().numpy()
        if not np.array_equal(candidate0_before, candidate0_after):
            raise RuntimeError("Trust-Region injection modified candidate0")
        costs = self.base.get_cost(info_dict, candidates)
        candidate_pool = candidates[0].detach().cpu().float().numpy().copy()
        cost_values = costs[0].detach().cpu().float().numpy().copy()
        record = {
            "env_idx": int(context["env_idx"]),
            "planning_cycle": int(context["planning_cycle"]),
            "env_step": int(context["env_step"]),
            "cem_iteration": iteration,
            "query_feature_raw": context["query_feature_raw"],
            "query_source": context["query_source"],
            "dataset_row": int(context["dataset_row"]),
            "excluded_eval_episode": int(context["excluded_eval_episode"]),
            "source_rows": context["source_rows"],
            "source_episodes": context["source_episodes"],
            "source_steps": context["source_steps"],
            "retrieval_distances": context["retrieval_distances"],
            "seed_actions_raw": context["seed_actions_raw"],
            "seed_actions_normalized": seeds,
            "initial_mean_normalized": seeds[0] if self.protocol == "t1" else np.zeros_like(seeds[0]),
            "preinjection_slots_1_30": before_slots,
            "noise_parent_seed_indices": noise["parent_indices"],
            "noise_derived_seed": noise["derived_seed"],
            "noise_seed_components": noise["seed_components"],
            "noise_values_raw": noise["noise_raw"],
            "noise_unclipped_actions_raw": noise["unclipped_raw"],
            "noise_clipped_actions_raw": noise["clipped_raw"],
            "noise_candidates_normalized": noise["normalized"],
            "noise_clip_mask": noise["clip_mask"],
            "postinjection_slots_1_30": candidate_pool[1:31].copy(),
            "candidates_normalized": candidate_pool,
            "latent_costs": cost_values,
        }
        self.trace.append(record)
        if (
            context["planning_cycle"] == 0
            and iteration == common.N_STEPS - 1
            and context["env_idx"] in common.AUDIT_ENVS
        ):
            self.first_cycle_final[int(context["env_idx"])] = record
        self.call_index += 1
        if self.call_index == len(self.pending) * common.N_STEPS:
            self.pending = None
        return costs


def make_trust_policy(base: type) -> type:
    class TrustPolicy(base):
        def __init__(
            self,
            *args: Any,
            memory_index: memory.CubeMemoryIndex,
            cost_proxy: TrustRegionCostProxy,
            cost_recorder: Any,
            eval_episodes: np.ndarray,
            eval_rows: np.ndarray,
            initial_query_features: np.ndarray,
            protocol: str,
            **kwargs: Any,
        ) -> None:
            self.memory_index = memory_index
            self.cost_proxy = cost_proxy
            self.cost_recorder = cost_recorder
            self.eval_episodes = np.asarray(eval_episodes, dtype=np.int64)
            self.eval_rows = np.asarray(eval_rows, dtype=np.int64)
            self.initial_query_features = np.asarray(initial_query_features, dtype=np.float64)
            self.protocol = protocol
            self._trust_env_step = 0
            self._trust_cycles = np.zeros(len(self.eval_rows), dtype=np.int64)
            if self.initial_query_features.shape != (len(self.eval_rows), 9):
                raise ValueError(
                    f"invalid initial query shape: {self.initial_query_features.shape}"
                )
            super().__init__(*args, **kwargs)

        def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
            terminated = info_dict.get("terminated")
            dead = (
                np.asarray(terminated, dtype=bool).reshape(self.env.num_envs, -1)[:, 0]
                if terminated is not None
                else np.zeros(self.env.num_envs, dtype=bool)
            )
            replans = [
                env_idx
                for env_idx in range(self.env.num_envs)
                if not dead[env_idx] and len(self._action_buffer[env_idx]) == 0
            ]
            if replans:
                self.cost_recorder.begin_solve(
                    [(env_idx, self._trust_env_step) for env_idx in replans]
                )
                contexts = []
                for env_idx in replans:
                    cycle = int(self._trust_cycles[env_idx])
                    if cycle == 0:
                        query = self.initial_query_features[env_idx].copy()
                        query_source = "hdf5_formal_row"
                    else:
                        query = legacy.raw_feature_from_info(info_dict, env_idx)
                        query_source = "live_env_info"
                    retrieved = self.memory_index.retrieve(
                        query,
                        exclude_episode=int(self.eval_episodes[env_idx]),
                        count=common.MEMORY_SLOTS,
                    )
                    bundle = self.memory_index.action_seed_bundle(retrieved["rows"])
                    context = {
                        "env_idx": env_idx,
                        "planning_cycle": cycle,
                        "env_step": self._trust_env_step,
                        "query_feature_raw": query,
                        "query_source": query_source,
                        "dataset_row": int(self.eval_rows[env_idx]),
                        "excluded_eval_episode": int(self.eval_episodes[env_idx]),
                        "source_rows": retrieved["rows"],
                        "source_episodes": retrieved["episodes"],
                        "source_steps": retrieved["steps"],
                        "retrieval_distances": retrieved["distances"],
                        "seed_actions_raw": bundle["raw"],
                        "seed_actions_normalized": bundle["normalized"],
                    }
                    if self.protocol == "t2":
                        context["noise_bundle"] = common.noisy_seed_variants(
                            bundle["raw"],
                            self.memory_index.action_mean,
                            self.memory_index.action_scale,
                            int(self.eval_rows[env_idx]),
                            cycle,
                        )
                    contexts.append(context)
                    self._trust_cycles[env_idx] += 1
                if self.protocol == "t1":
                    import torch

                    if self._next_init is None or tuple(self._next_init.shape) != (
                        self.env.num_envs,
                        common.HORIZON,
                        common.ACTION_BLOCK * common.ACTION_DIM,
                    ):
                        self._next_init = torch.zeros(
                            self.env.num_envs,
                            common.HORIZON,
                            common.ACTION_BLOCK * common.ACTION_DIM,
                            dtype=torch.float32,
                        )
                    for context in contexts:
                        self._next_init[context["env_idx"]].copy_(
                            torch.from_numpy(context["seed_actions_normalized"][0])
                        )
                else:
                    # Horizon equals receding horizon, so legacy warm-start has
                    # no remainder.  Clear explicitly to freeze mean=0 anyway.
                    self._next_init = None
                self.cost_proxy.begin_solve(contexts)
            result = super().get_action(info_dict, **kwargs)
            self._trust_env_step += 1
            return result

    return TrustPolicy


def _save_trace(output: Path, proxy: TrustRegionCostProxy) -> dict[str, Any]:
    records = proxy.trace
    if not records:
        raise RuntimeError("Trust-Region evaluation produced no CEM trace")
    array_fields = (
        "query_feature_raw",
        "source_rows",
        "source_episodes",
        "source_steps",
        "retrieval_distances",
        "seed_actions_raw",
        "seed_actions_normalized",
        "initial_mean_normalized",
        "preinjection_slots_1_30",
        "noise_parent_seed_indices",
        "noise_derived_seed",
        "noise_seed_components",
        "noise_values_raw",
        "noise_unclipped_actions_raw",
        "noise_clipped_actions_raw",
        "noise_candidates_normalized",
        "noise_clip_mask",
        "postinjection_slots_1_30",
        "candidates_normalized",
        "latent_costs",
    )
    arrays = {
        "env_idx": np.asarray([row["env_idx"] for row in records], dtype=np.int64),
        "planning_cycle": np.asarray([row["planning_cycle"] for row in records], dtype=np.int64),
        "env_step": np.asarray([row["env_step"] for row in records], dtype=np.int64),
        "cem_iteration": np.asarray([row["cem_iteration"] for row in records], dtype=np.int64),
        "dataset_row": np.asarray([row["dataset_row"] for row in records], dtype=np.int64),
        "excluded_eval_episode": np.asarray(
            [row["excluded_eval_episode"] for row in records], dtype=np.int64
        ),
        "query_source": np.asarray([row["query_source"] for row in records]),
    }
    for field in array_fields:
        arrays[field] = np.stack([row[field] for row in records])
    trace_path = output / "trust_trace.npz"
    np.savez_compressed(trace_path, **arrays)
    metadata_records = [
        {
            "record_idx": index,
            "env_idx": row["env_idx"],
            "planning_cycle": row["planning_cycle"],
            "env_step": row["env_step"],
            "cem_iteration": row["cem_iteration"],
            "query_source": row["query_source"],
            "dataset_row": row["dataset_row"],
            "excluded_eval_episode": row["excluded_eval_episode"],
            "source_rows": row["source_rows"],
            "source_episodes": row["source_episodes"],
            "source_steps": row["source_steps"],
            "retrieval_distances": row["retrieval_distances"],
            "noise_seed_components": row["noise_seed_components"],
            "noise_clipped_value_count": int(np.count_nonzero(row["noise_clip_mask"])),
            "npz_record_index": index,
        }
        for index, row in enumerate(records)
    ]
    common.write_json(
        output / "trust_trace.json",
        {
            "format_version": "cube_trust_region_trace_v1",
            "protocol": proxy.protocol,
            "torch_rng": "legacy CEM torch.Generator seed=42; proxy consumes no torch randomness",
            "noise_rng": (
                "none"
                if proxy.protocol == "t1"
                else "independent CPU torch.Generator; SHA256-derived seed from [42,dataset_row,planning_cycle]"
            ),
            "arrays": common.file_identity(trace_path),
            "records": metadata_records,
        },
    )
    return {
        "num_iteration_records": len(records),
        "num_planning_cycles": len(records) // common.N_STEPS,
        "npz": common.file_identity(trace_path),
    }


def _save_first_cycle_pools(
    output: Path,
    proxy: TrustRegionCostProxy,
    recorder: Any,
    rows: np.ndarray,
    eval_episodes: np.ndarray,
    raw_inputs: dict[str, np.ndarray],
    dataset_path: Path,
    scaler: Any,
    protocol: str,
    condition: str,
) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401
    import h5py

    root = output / "first_cycle_pools"
    root.mkdir(parents=True, exist_ok=True)
    saved = []
    with h5py.File(dataset_path, "r", swmr=True) as h5:
        for env_idx in common.AUDIT_ENVS:
            if env_idx >= len(rows):
                continue
            if env_idx not in proxy.first_cycle_final:
                raise RuntimeError(f"missing first-cycle final proxy pool: env={env_idx}")
            trace = proxy.first_cycle_final[env_idx]
            cycle = recorder.records[env_idx][0]
            candidates = np.asarray(cycle["final_candidates_normalized"], dtype=np.float32)
            costs = np.asarray(cycle["costs"][-1], dtype=np.float32)
            if not np.array_equal(candidates, trace["candidates_normalized"]):
                raise RuntimeError(f"recorder/proxy final pool mismatch: env={env_idx}")
            if not np.array_equal(costs, trace["latent_costs"]):
                raise RuntimeError(f"recorder/proxy final costs mismatch: env={env_idx}")
            order = np.lexsort((np.arange(common.NUM_SAMPLES), costs))
            topk = order[: common.TOPK]
            row = int(rows[env_idx])
            goal_row = row + 25
            case = root / common.case_name(env_idx, row)
            case.mkdir()
            population_path = case / "population.npz"
            np.savez_compressed(
                population_path,
                candidates_normalized=candidates,
                latent_costs=costs,
                topk_indices=topk.astype(np.int64),
                topk_costs=costs[topk],
                final_mean_normalized=np.asarray(
                    cycle["solver_returned_actions_normalized"], dtype=np.float32
                ),
                final_variance=np.asarray(cycle["variance"][-1], dtype=np.float32),
                previous_mean=np.asarray(
                    cycle["mean"][-2] if len(cycle["mean"]) > 1 else trace["initial_mean_normalized"],
                    dtype=np.float32,
                ),
                previous_variance=np.asarray(
                    cycle["variance"][-2]
                    if len(cycle["variance"]) > 1
                    else np.full_like(trace["initial_mean_normalized"], common.PROTOCOL_SPECS[protocol]["var_scale"]),
                    dtype=np.float32,
                ),
                action_scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
                action_scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
                initial_qpos=np.asarray(h5["qpos"][row]),
                initial_qvel=np.asarray(h5["qvel"][row]),
                initial_prev_qpos=np.asarray(h5["prev_qpos"][row]),
                initial_prev_qvel=np.asarray(h5["prev_qvel"][row]),
                goal_position=np.asarray(h5["privileged_block_0_pos"][goal_row]),
                goal_quaternion=np.asarray(h5["privileged_block_0_quat"][goal_row]),
                initial_pixels=np.asarray(raw_inputs["pixels"][env_idx, 0], dtype=np.uint8),
                goal_pixels=np.asarray(raw_inputs["goal"][env_idx, 0], dtype=np.uint8),
            )
            injection_path = case / "injection_provenance.npz"
            np.savez_compressed(
                injection_path,
                query_feature_raw=trace["query_feature_raw"],
                source_rows=trace["source_rows"],
                source_episodes=trace["source_episodes"],
                source_steps=trace["source_steps"],
                retrieval_distances=trace["retrieval_distances"],
                seed_actions_raw=trace["seed_actions_raw"],
                seed_actions_normalized=trace["seed_actions_normalized"],
                initial_mean_normalized=trace["initial_mean_normalized"],
                preinjection_slots_1_30=trace["preinjection_slots_1_30"],
                noise_parent_seed_indices=trace["noise_parent_seed_indices"],
                noise_derived_seed=trace["noise_derived_seed"],
                noise_seed_components=trace["noise_seed_components"],
                noise_values_raw=trace["noise_values_raw"],
                noise_unclipped_actions_raw=trace["noise_unclipped_actions_raw"],
                noise_clipped_actions_raw=trace["noise_clipped_actions_raw"],
                noise_candidates_normalized=trace["noise_candidates_normalized"],
                noise_clip_mask=trace["noise_clip_mask"],
                postinjection_slots_1_30=trace["postinjection_slots_1_30"],
            )
            meta = {
                "format_version": "cube_trust_region_first_cycle_pool_v1",
                "protocol": protocol,
                "condition": condition,
                "env_idx": env_idx,
                "dataset_row": row,
                "episode_idx": int(eval_episodes[env_idx]),
                "planning_cycle": 0,
                "cem_iteration": common.N_STEPS - 1,
                "full_50_env_solve_order": len(rows) == 50,
                "pool_has_physical_endpoint_truth": False,
                "physical_truth_warning": (
                    "candidate indices are new; old audit outcomes must never be joined. "
                    "Run the independent authorized replay before imagination scoring."
                ),
                "population": common.file_identity(population_path),
                "injection_provenance": common.file_identity(injection_path),
            }
            common.write_json(case / "capture_meta.json", meta)
            saved.append(meta)
    expected = len([env for env in common.AUDIT_ENVS if env < len(rows)])
    if len(saved) != expected:
        raise RuntimeError(f"first-cycle pool count mismatch: expected={expected}, actual={len(saved)}")
    manifest = {
        "format_version": "cube_trust_region_first_cycle_manifest_v1",
        "protocol": protocol,
        "condition": condition,
        "source_evaluation": str(output.resolve()),
        "saved_envs": [item["env_idx"] for item in saved],
        "full_formal_order_then_select_12": len(rows) == 50,
        "cases": saved,
    }
    common.write_json(root / "manifest.json", manifest)
    return {"root": str(root.resolve()), "num_cases": len(saved)}


def _initial_contexts(
    index: memory.CubeMemoryIndex,
    rows: np.ndarray,
    eval_episodes: np.ndarray,
    initial_query_features: np.ndarray,
    protocol: str,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    contexts = []
    init = []
    for env_idx, (row, episode, query) in enumerate(
        zip(rows, eval_episodes, initial_query_features, strict=True)
    ):
        retrieved = index.retrieve(
            query,
            exclude_episode=int(episode),
            count=common.MEMORY_SLOTS,
        )
        bundle = index.action_seed_bundle(retrieved["rows"])
        context = {
            "env_idx": env_idx,
            "planning_cycle": 0,
            "env_step": 0,
            "query_feature_raw": query,
            "query_source": "hdf5_formal_row",
            "dataset_row": int(row),
            "excluded_eval_episode": int(episode),
            "source_rows": retrieved["rows"],
            "source_episodes": retrieved["episodes"],
            "source_steps": retrieved["steps"],
            "retrieval_distances": retrieved["distances"],
            "seed_actions_raw": bundle["raw"],
            "seed_actions_normalized": bundle["normalized"],
        }
        if protocol == "t2":
            context["noise_bundle"] = common.noisy_seed_variants(
                bundle["raw"],
                index.action_mean,
                index.action_scale,
                int(row),
                0,
            )
        else:
            init.append(bundle["normalized"][0])
        contexts.append(context)
    return contexts, None if protocol == "t2" else np.stack(init).astype(np.float32)


def _capture_population_hashes(capture_root: Path) -> dict[str, str]:
    manifest_path = capture_root / "first_cycle_pools/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {}
    for case in manifest["cases"]:
        env_idx = int(case["env_idx"])
        row = int(case["dataset_row"])
        path = capture_root / "first_cycle_pools" / common.case_name(env_idx, row) / "population.npz"
        actual = common.sha256_file(path)
        expected = case["population"]["sha256"]
        if actual != expected:
            raise ValueError(
                f"capture population hash mismatch: env={env_idx}, "
                f"expected={expected}, actual={actual}"
            )
        hashes[str(env_idx)] = actual
    return hashes


def _validate_gate_artifact(
    gate_path: Path,
    protocol: str,
    condition: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    path = common.ensure_child(gate_path, common.OUTPUT_ROOT, "Trust-Region gate")
    if not path.is_file():
        raise FileNotFoundError(f"gate artifact missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "status": gate.get("status"),
        "protocol": gate.get("protocol"),
        "condition": gate.get("condition"),
        "model": gate.get("primary_model"),
        "metric": gate.get("criterion", {}).get("metric"),
        "count": gate.get("criterion", {}).get("count"),
        "threshold_mm": gate.get("criterion", {}).get("threshold_mm"),
    }
    expected = {
        "status": "PASS",
        "protocol": protocol,
        "condition": condition,
        "model": "masked",
        "metric": "E_roll_mm_median_fixed12x300",
        "count": 3600,
        "threshold_mm": 40.0,
    }
    if actual != expected:
        raise ValueError(f"Trust-Region gate contract mismatch: expected={expected}, actual={actual}")
    observed = float(gate["criterion"]["observed_median_mm"])
    if not np.isfinite(observed) or observed > 40.0:
        raise ValueError(
            f"Trust-Region gate is not numerically passing: expected<=40.0mm, actual={observed}"
        )
    for label in ("weights", "config"):
        expected_sha = checkpoint[label]["sha256"]
        actual_sha = gate["checkpoint"][label]["sha256"]
        if actual_sha != expected_sha:
            raise ValueError(
                f"gate MaskedAug {label} mismatch: expected={expected_sha}, actual={actual_sha}"
            )
    probe_identity = gate.get("probe", {}).get("checkpoint", {})
    actual_probe_sha = common.sha256_file(common.MASKED_PROBE)
    if (
        Path(probe_identity.get("path", "")).resolve() != common.MASKED_PROBE.resolve()
        or probe_identity.get("sha256") != actual_probe_sha
    ):
        raise ValueError(
            "gate Masked probe changed or points to another file: "
            f"expected_path={common.MASKED_PROBE.resolve()}, "
            f"actual_path={probe_identity.get('path')}, "
            f"expected_sha={actual_probe_sha}, actual_sha={probe_identity.get('sha256')}"
        )
    capture_root = common.ensure_child(
        Path(gate["capture"]["root"]), common.OUTPUT_ROOT, "gate capture root"
    )
    results_path = capture_root / "results.json"
    manifest_path = capture_root / "first_cycle_pools/manifest.json"
    for label, source, identity in (
        ("capture results", results_path, gate["capture"]["results"]),
        ("capture manifest", manifest_path, gate["capture"]["pool_manifest"]),
    ):
        actual_sha = common.sha256_file(source)
        if actual_sha != identity["sha256"]:
            raise ValueError(
                f"{label} changed after gate: expected={identity['sha256']}, actual={actual_sha}"
            )
    capture_results = json.loads(results_path.read_text(encoding="utf-8"))
    capture_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture_contract = {
        "mode": capture_results.get("mode"),
        "protocol": capture_results.get("protocol", {}).get("id"),
        "condition": capture_results.get("protocol", {}).get("condition"),
        "num_eval": capture_results.get("metrics", {}).get("num_eval"),
        "formal_full_50_order": capture_results.get("protocol", {}).get(
            "formal_full_50_order"
        ),
        "manifest_protocol": capture_manifest.get("protocol"),
        "manifest_condition": capture_manifest.get("condition"),
        "full_formal_order_then_select_12": capture_manifest.get(
            "full_formal_order_then_select_12"
        ),
        "saved_envs": capture_manifest.get("saved_envs"),
    }
    expected_capture_contract = {
        "mode": "capture_only_no_env_step_no_video",
        "protocol": protocol,
        "condition": condition,
        "num_eval": 50,
        "formal_full_50_order": True,
        "manifest_protocol": protocol,
        "manifest_condition": condition,
        "full_formal_order_then_select_12": True,
        "saved_envs": list(common.AUDIT_ENVS),
    }
    if capture_contract != expected_capture_contract:
        raise ValueError(
            "gate capture protocol mismatch: "
            f"expected={expected_capture_contract}, actual={capture_contract}"
        )
    capture_checkpoint = capture_results.get("protocol", {}).get("checkpoint", {})
    for label in ("weights", "config"):
        expected_identity = checkpoint[label]
        actual_identity = capture_checkpoint.get(label, {})
        if (
            Path(actual_identity.get("path", "")).resolve()
            != Path(expected_identity["path"]).resolve()
            or actual_identity.get("sha256") != expected_identity["sha256"]
        ):
            raise ValueError(
                f"capture MaskedAug {label} contract mismatch: "
                f"expected_path/sha={expected_identity['path']}/{expected_identity['sha256']}, "
                f"actual_path/sha={actual_identity.get('path')}/{actual_identity.get('sha256')}"
            )
    physical_identity = gate.get("physical_cache_manifest", {})
    physical_manifest = common.ensure_child(
        Path(physical_identity.get("path", "")),
        common.OUTPUT_ROOT,
        "gate physical-cache manifest",
    )
    if not physical_manifest.is_file():
        raise FileNotFoundError(f"gate physical-cache manifest missing: {physical_manifest}")
    actual_physical_sha = common.sha256_file(physical_manifest)
    if actual_physical_sha != physical_identity.get("sha256"):
        raise ValueError(
            "physical-cache manifest changed after gate: "
            f"expected={physical_identity.get('sha256')}, actual={actual_physical_sha}"
        )
    hashes = _capture_population_hashes(capture_root)
    if hashes != gate["capture"]["population_sha256_by_env"]:
        raise ValueError(
            "gate population hashes no longer match capture: "
            f"expected={gate['capture']['population_sha256_by_env']}, actual={hashes}"
        )
    return {"path": common.file_identity(path), "payload": gate, "population_hashes": hashes}


def _diagnose_online_pools_against_gate(
    output: Path,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Compare formal pools with the gate capture without gating GPU roundoff.

    The PASS gate and all of its checkpoint, probe, capture, and physical-cache
    provenance are validated before the formal run.  A fresh CUDA process can
    nevertheless differ from the gate capture by sub-float32-ULP arithmetic,
    which changes the compressed NPZ byte hash despite preserving candidate
    rankings.  Consequently this *post-run* comparison is diagnostic only for
    numeric values.  Shape, finiteness, and per-environment provenance remain
    hard consistency checks.
    """

    actual_hashes = _capture_population_hashes(output)
    expected_hashes = gate["population_hashes"]
    capture_root = common.ensure_child(
        Path(gate["payload"]["capture"]["root"]),
        common.OUTPUT_ROOT,
        "gate capture root",
    )
    expected_manifest_path = capture_root / "first_cycle_pools/manifest.json"
    actual_manifest_path = output / "first_cycle_pools/manifest.json"
    expected_manifest = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
    actual_manifest = json.loads(actual_manifest_path.read_text(encoding="utf-8"))

    def provenance(manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocol": manifest.get("protocol"),
            "condition": manifest.get("condition"),
            "saved_envs": manifest.get("saved_envs"),
            "full_formal_order_then_select_12": manifest.get(
                "full_formal_order_then_select_12"
            ),
            "cases": [
                {
                    "env_idx": case.get("env_idx"),
                    "dataset_row": case.get("dataset_row"),
                    "episode_idx": case.get("episode_idx"),
                    "planning_cycle": case.get("planning_cycle"),
                    "cem_iteration": case.get("cem_iteration"),
                }
                for case in manifest.get("cases", [])
            ],
        }

    expected_provenance = provenance(expected_manifest)
    actual_provenance = provenance(actual_manifest)
    if actual_provenance != expected_provenance:
        raise RuntimeError(
            "formal/gate first-cycle pool provenance mismatch: "
            f"expected={expected_provenance}, actual={actual_provenance}"
        )
    if set(actual_hashes) != set(expected_hashes):
        raise RuntimeError(
            "formal/gate first-cycle environment set mismatch: "
            f"expected={sorted(expected_hashes)}, actual={sorted(actual_hashes)}"
        )

    per_env: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for case in expected_manifest["cases"]:
        env_idx = int(case["env_idx"])
        env_key = str(env_idx)
        row = int(case["dataset_row"])
        case_name = common.case_name(env_idx, row)
        expected_path = capture_root / "first_cycle_pools" / case_name / "population.npz"
        actual_path = output / "first_cycle_pools" / case_name / "population.npz"
        with np.load(expected_path, allow_pickle=False) as expected_npz, np.load(
            actual_path, allow_pickle=False
        ) as actual_npz:
            required = ("candidates_normalized", "latent_costs")
            for label, archive in (("gate", expected_npz), ("formal", actual_npz)):
                missing = [key for key in required if key not in archive.files]
                if missing:
                    raise RuntimeError(
                        f"{label} population missing arrays: env={env_idx}, missing={missing}"
                    )
            expected_candidates = np.asarray(expected_npz["candidates_normalized"])
            actual_candidates = np.asarray(actual_npz["candidates_normalized"])
            expected_costs = np.asarray(expected_npz["latent_costs"])
            actual_costs = np.asarray(actual_npz["latent_costs"])

        expected_candidate_shape = tuple(int(value) for value in expected_candidates.shape)
        actual_candidate_shape = tuple(int(value) for value in actual_candidates.shape)
        expected_cost_shape = tuple(int(value) for value in expected_costs.shape)
        actual_cost_shape = tuple(int(value) for value in actual_costs.shape)
        required_candidate_shape = (
            common.NUM_SAMPLES,
            common.HORIZON,
            common.ACTION_BLOCK * common.ACTION_DIM,
        )
        required_cost_shape = (common.NUM_SAMPLES,)
        if (
            expected_candidate_shape != required_candidate_shape
            or actual_candidate_shape != required_candidate_shape
            or expected_cost_shape != required_cost_shape
            or actual_cost_shape != required_cost_shape
        ):
            raise RuntimeError(
                "formal/gate population shape mismatch: "
                f"env={env_idx}, expected_candidates={expected_candidate_shape}, "
                f"actual_candidates={actual_candidate_shape}, "
                f"expected_costs={expected_cost_shape}, actual_costs={actual_cost_shape}, "
                f"required_candidates={required_candidate_shape}, required_costs={required_cost_shape}"
            )
        finite = {
            "gate_candidates": bool(np.isfinite(expected_candidates).all()),
            "formal_candidates": bool(np.isfinite(actual_candidates).all()),
            "gate_costs": bool(np.isfinite(expected_costs).all()),
            "formal_costs": bool(np.isfinite(actual_costs).all()),
        }
        if not all(finite.values()):
            raise RuntimeError(
                f"formal/gate population contains non-finite values: env={env_idx}, finite={finite}"
            )

        candidate_max_abs = float(
            np.max(
                np.abs(
                    actual_candidates.astype(np.float64)
                    - expected_candidates.astype(np.float64)
                )
            )
        )
        cost_max_abs = float(
            np.max(
                np.abs(
                    actual_costs.astype(np.float64) - expected_costs.astype(np.float64)
                )
            )
        )
        candidate_exact = bool(np.array_equal(actual_candidates, expected_candidates))
        cost_exact = bool(np.array_equal(actual_costs, expected_costs))
        stable_ids = np.arange(common.NUM_SAMPLES)
        expected_order = np.lexsort((stable_ids, expected_costs))
        actual_order = np.lexsort((stable_ids, actual_costs))
        expected_top1 = int(expected_order[0])
        actual_top1 = int(actual_order[0])
        expected_top30 = {int(value) for value in expected_order[: common.TOPK]}
        actual_top30 = {int(value) for value in actual_order[: common.TOPK]}
        top30_overlap = len(expected_top30 & actual_top30)
        exact_hash = expected_hashes[env_key] == actual_hashes[env_key]
        per_env[env_key] = {
            "dataset_row": row,
            "gate_population_sha256": expected_hashes[env_key],
            "formal_population_sha256": actual_hashes[env_key],
            "exact_population_hash_match": exact_hash,
            "candidate_shape": list(actual_candidate_shape),
            "cost_shape": list(actual_cost_shape),
            "all_finite": finite,
            "candidates_elementwise_exact": candidate_exact,
            "candidate_max_abs_diff": candidate_max_abs,
            "costs_elementwise_exact": cost_exact,
            "cost_max_abs_diff": cost_max_abs,
            "gate_top1_candidate_idx": expected_top1,
            "formal_top1_candidate_idx": actual_top1,
            "top1_agreement": expected_top1 == actual_top1,
            "top30_overlap_count": top30_overlap,
            "top30_size": common.TOPK,
            "top30_overlap_fraction": float(top30_overlap / common.TOPK),
        }

    hash_mismatch_envs = [
        int(env) for env, item in per_env.items() if not item["exact_population_hash_match"]
    ]
    candidate_drift_envs = [
        int(env) for env, item in per_env.items() if not item["candidates_elementwise_exact"]
    ]
    cost_drift_envs = [
        int(env) for env, item in per_env.items() if not item["costs_elementwise_exact"]
    ]
    top1_disagreement_envs = [
        int(env) for env, item in per_env.items() if not item["top1_agreement"]
    ]
    top30_incomplete_envs = [
        int(env)
        for env, item in per_env.items()
        if item["top30_overlap_count"] != common.TOPK
    ]
    if hash_mismatch_envs:
        warnings.append(
            "report-only: compressed population NPZ hashes differ across CUDA processes "
            f"for envs={hash_mismatch_envs}"
        )
    if candidate_drift_envs:
        warnings.append(
            "report-only: candidate values have floating-point drift for "
            f"envs={candidate_drift_envs}"
        )
    if cost_drift_envs:
        warnings.append(
            "report-only: latent costs have floating-point drift for "
            f"envs={cost_drift_envs}"
        )
    if top1_disagreement_envs:
        warnings.append(
            f"report-only: latent top-1 differs for envs={top1_disagreement_envs}"
        )
    if top30_incomplete_envs:
        warnings.append(
            "report-only: latent top-30 set is not identical for envs="
            f"{top30_incomplete_envs}"
        )

    summary = {
        "comparison_policy": (
            "post-formal exact hashes and numeric differences are report-only; "
            "shape, finiteness, and provenance are fail-closed"
        ),
        "num_envs": len(per_env),
        "all_exact_population_hashes_match": not hash_mismatch_envs,
        "hash_mismatch_envs": hash_mismatch_envs,
        "candidate_drift_envs": candidate_drift_envs,
        "cost_drift_envs": cost_drift_envs,
        "max_candidate_abs_diff": max(
            (item["candidate_max_abs_diff"] for item in per_env.values()), default=0.0
        ),
        "max_cost_abs_diff": max(
            (item["cost_max_abs_diff"] for item in per_env.values()), default=0.0
        ),
        "all_top1_agree": not top1_disagreement_envs,
        "top1_disagreement_envs": top1_disagreement_envs,
        "minimum_top30_overlap_count": min(
            (item["top30_overlap_count"] for item in per_env.values()),
            default=common.TOPK,
        ),
        "top30_incomplete_envs": top30_incomplete_envs,
        "warnings": warnings,
    }
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return {
        "format_version": "cube_trust_region_gate_population_diagnostic_v1",
        "gate_capture_root": str(capture_root.resolve()),
        "formal_output_root": str(output.resolve()),
        "hard_checks_passed": True,
        "summary": summary,
        "per_env": per_env,
    }


def _run_capture_only(
    args: argparse.Namespace,
    model: Any,
    index: memory.CubeMemoryIndex,
    dataset: Any,
    rows: np.ndarray,
    eval_episodes: np.ndarray,
    initial_query_features: np.ndarray,
    raw_inputs: dict[str, np.ndarray],
    scaler: Any,
    checkpoint: dict[str, Any],
    recolor_meta: dict[str, Any] | None,
) -> int:
    import torch
    import stable_worldmodel as swm
    from gymnasium.spaces import Box

    output = common.prepare_output(
        args.output or common.capture_output_root(args.protocol, args.condition),
        common.OUTPUT_ROOT,
        args.overwrite,
    )
    proxy = TrustRegionCostProxy(model, args.protocol)
    recorder = ood.PlanningCostRecorder(50)
    solver_cls = make_trust_region_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=proxy,
        batch_size=1,
        num_samples=common.NUM_SAMPLES,
        var_scale=common.PROTOCOL_SPECS[args.protocol]["var_scale"],
        n_steps=common.N_STEPS,
        topk=common.TOPK,
        device=args.device,
        seed=common.FORMAL_SEED,
        callbacks=[recorder],
        selector="mean",
        recorder=recorder,
        trust_protocol=args.protocol,
    )
    config = swm.PlanConfig(
        horizon=common.HORIZON,
        receding_horizon=common.HORIZON,
        action_block=common.ACTION_BLOCK,
    )
    action_space = Box(
        low=np.broadcast_to(-np.inf, (50, common.ACTION_DIM)),
        high=np.broadcast_to(np.inf, (50, common.ACTION_DIM)),
        dtype=np.float32,
    )
    solver.configure(action_space=action_space, n_envs=50, config=config)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process={"action": scaler},
        transform={"pixels": ood._image_transform(224), "goal": ood._image_transform(224)},
    )
    prepared = policy._prepare_info(raw_inputs)
    contexts, init = _initial_contexts(
        index, rows, eval_episodes, initial_query_features, args.protocol
    )
    proxy.begin_solve(contexts)
    recorder.begin_solve([(env_idx, 0) for env_idx in range(50)])
    init_tensor = None if init is None else torch.from_numpy(init)
    started = time.time()
    with torch.inference_mode():
        solver(prepared, init_action=init_tensor)
    elapsed = time.time() - started
    trace = _save_trace(output, proxy)
    pools = _save_first_cycle_pools(
        output,
        proxy,
        recorder,
        rows,
        eval_episodes,
        raw_inputs,
        args.dataset,
        scaler,
        args.protocol,
        args.condition,
    )
    payload = {
        "format_version": "cube_trust_region_gate_capture_v1",
        "mode": "capture_only_no_env_step_no_video",
        "protocol": {
            "id": args.protocol,
            **common.PROTOCOL_SPECS[args.protocol],
            "condition": args.condition,
            "selector": "legacy_updated_elite_mean",
            "formal_full_50_order": True,
            "checkpoint": checkpoint,
            "goal_recolor": recolor_meta,
            "solver_class": "local TrustRegionCEMSolver with proxy candidate hook",
        },
        "metrics": {"num_eval": 50, "success_not_measured": True},
        "evaluated_rows": rows,
        "elapsed_seconds": elapsed,
        "trace": trace,
        "first_cycle_pools": pools,
        "population_sha256_by_env": _capture_population_hashes(output),
    }
    common.write_json(output / "results.json", payload)
    print(output)
    return 0


def run(args: argparse.Namespace) -> int:
    common.configure_storage()
    if args.protocol not in common.PROTOCOLS:
        raise ValueError(args.protocol)
    if args.seed != common.FORMAL_SEED or args.num_eval not in (2, 50):
        raise ValueError("protocol is frozen at seed=42 and num_eval=2/50")
    if args.mode == "capture" and args.num_eval != 50:
        raise ValueError("capture-only mode is frozen to all 50 formal rows")
    if args.mode == "capture" and not args.authorize_capture:
        raise PermissionError(
            "capture-only 50-env model solve is not authorized; "
            "re-run after Leader approval with --authorize-capture"
        )
    if args.mode == "evaluate" and args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError(
            "formal 50-env Trust-Region evaluation is not authorized; "
            "run only after Leader approval with --authorize-formal"
        )
    color, goal_type, audit_color = common.condition_visual(args.condition)
    checkpoint_provenance = common.frozen_masked_checkpoint_contract()
    checkpoint_ref = str(common.MASKED_CHECKPOINT)
    if not args.dataset.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("dataset/manifest input missing")
    gate = None
    if args.mode == "evaluate" and args.num_eval == 50:
        if args.gate is None:
            raise ValueError("formal evaluation requires --gate from the same T1/T2 color cell")
        gate = _validate_gate_artifact(
            args.gate, args.protocol, args.condition, checkpoint_provenance
        )

    import hdf5plugin  # noqa: F401
    import h5py
    import stable_worldmodel as swm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Trust-Region evaluation requires CUDA")
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    all_rows = ood._formal_rows(dataset, args.manifest)
    rows = all_rows if args.mode == "capture" else all_rows[: args.num_eval]
    selected = dataset.get_row_data(rows)
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    eval_episodes = np.asarray(selected[ep_key], dtype=np.int64)
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        initial_query_features = np.concatenate(
            [memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows], axis=0
        )
    recolor_goals = recolor_meta = None
    if goal_type == "recolor":
        recolor_goals, recolor_meta = ood._load_recolor_goals(color, args.num_eval)
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        raw_inputs = audit._load_capture_inputs(
            h5,
            rows,
            audit_color,
            recolor_goals,
        )
    model = swm.wm.utils.load_pretrained(checkpoint_ref, cache_dir=str(common.PROJECT_ROOT))
    model = model.to(args.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    scaler = legacy._standard_scaler(index)
    if args.mode == "capture":
        return _run_capture_only(
            args,
            model,
            index,
            dataset,
            rows,
            eval_episodes,
            initial_query_features,
            raw_inputs,
            scaler,
            checkpoint_provenance,
            recolor_meta,
        )
    output = common.prepare_output(
        args.output
        or common.default_eval_output(args.protocol, args.condition, args.num_eval),
        common.OUTPUT_ROOT,
        args.overwrite,
    )
    proxy = TrustRegionCostProxy(model, args.protocol)
    recorder = ood.PlanningCostRecorder(args.num_eval)
    solver_cls = make_trust_region_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=proxy,
        batch_size=1,
        num_samples=common.NUM_SAMPLES,
        var_scale=common.PROTOCOL_SPECS[args.protocol]["var_scale"],
        n_steps=common.N_STEPS,
        topk=common.TOPK,
        device=args.device,
        seed=common.FORMAL_SEED,
        callbacks=[recorder],
        selector="mean",
        recorder=recorder,
        trust_protocol=args.protocol,
    )
    config = swm.PlanConfig(
        horizon=common.HORIZON,
        receding_horizon=common.HORIZON,
        action_block=common.ACTION_BLOCK,
    )
    policy_cls = make_trust_policy(swm.policy.WorldModelPolicy)
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
        protocol=args.protocol,
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
    trace = _save_trace(output, proxy)
    cost_history = ood._save_cost_history(
        output, recorder, rows, chosen["episodes"], chosen["starts"], "mean"
    )
    pools = _save_first_cycle_pools(
        output,
        proxy,
        recorder,
        rows,
        eval_episodes,
        raw_inputs,
        args.dataset,
        scaler,
        args.protocol,
        args.condition,
    )
    gate_population_diagnostic = None
    if gate is not None:
        gate_population_diagnostic = _diagnose_online_pools_against_gate(output, gate)
        gate_population_diagnostic_path = output / "gate_population_diagnostic.json"
        common.write_json(gate_population_diagnostic_path, gate_population_diagnostic)
    payload = {
        "format_version": "cube_trust_region_evaluation_v1",
        "protocol": {
            "id": args.protocol,
            **common.PROTOCOL_SPECS[args.protocol],
            "condition": args.condition,
            "color": color,
            "goal_type": goal_type,
            "goal_recolor": recolor_meta,
            "selector": "legacy_updated_elite_mean",
            "num_samples": common.NUM_SAMPLES,
            "n_steps": common.N_STEPS,
            "topk": common.TOPK,
            "seed": common.FORMAL_SEED,
            "formal_full_50_order": args.num_eval == 50,
            "checkpoint": checkpoint_provenance,
            "gate": None if gate is None else gate["path"],
            "memory_index": common.file_identity(Path(args.index) / "metadata.json"),
            "torch_rng": "unchanged legacy CEM stream; no extra torch draws",
            "noise_rng": (
                None
                if args.protocol == "t1"
                else (
                    "independent CPU torch.Generator; SHA256-derived seed from "
                    "(base42,dataset_row,planning_cycle); sampled once/cycle and reused"
                )
            ),
            "helper_provenance": {
                "eval_memory_seed": common.file_identity(Path(legacy.__file__)),
                "eval_ood_color": common.file_identity(Path(ood.__file__)),
                "cube_cem_audit": common.file_identity(Path(audit.__file__)),
                "build_cube_memory_index": common.file_identity(Path(memory.__file__)),
                "this_evaluator": common.file_identity(Path(__file__)),
            },
        },
        "formal_rows_verified": all_rows,
        "evaluated_rows": rows,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "trace": trace,
        "cost_history": cost_history,
        "first_cycle_pools": pools,
        "gate_population_diagnostic": (
            None
            if gate_population_diagnostic is None
            else {
                "artifact": common.file_identity(gate_population_diagnostic_path),
                "summary": gate_population_diagnostic["summary"],
            }
        ),
    }
    common.write_json(output / "results.json", payload)
    successes = ", ".join(
        "True" if value else "False" for value in metrics["episode_successes"]
    )
    (output / "results.txt").write_text(
        f"protocol: {args.protocol}\ncondition: {args.condition}\n"
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
    parser.add_argument("--protocol", choices=common.PROTOCOLS, required=True)
    parser.add_argument("--condition", choices=common.CONDITIONS, required=True)
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--mode", choices=("capture", "evaluate"), default="evaluate")
    parser.add_argument("--seed", type=int, default=common.FORMAL_SEED)
    parser.add_argument("--dataset", type=Path, default=common.DATASET)
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST)
    parser.add_argument("--index", type=Path, default=common.MEMORY_INDEX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-capture", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
