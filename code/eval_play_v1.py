#!/usr/bin/env python3
"""Gate-bound online evaluation for the unique Cube play-v1 checkpoint.

Arms are the frozen T2 red/blue_v2/yellow_v2 protocols and the existing
probe-coordinate direct red offset25 protocol.  No arm can run without a
reproducible aggregate offline PASS.
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
AILAB = HERE.parent
for module_root in (HERE, TOOLS):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

import build_cube_memory_index as memory  # noqa: E402
import cube_cem_audit as audit  # noqa: E402
import cube_imagination_error_common as xyz_common  # noqa: E402
import cube_probe_common as probe_common  # noqa: E402
import cube_trust_region_common as trust_common  # noqa: E402
import eval_brain_b1 as brain  # noqa: E402
import eval_memory_seed as legacy  # noqa: E402
import eval_ood_color as ood  # noqa: E402
import eval_probe_goal_ood as probe_goal  # noqa: E402
import eval_trust_region as trust  # noqa: E402
import eval_waypoint_probe as waypoint  # noqa: E402
import evaluate_cube_play_v1 as offline  # noqa: E402


OUTPUT_ROOT = AILAB / "outputs/eval/cube/play_v1"
DEFAULT_CHECKPOINT = offline.DEFAULT_CHECKPOINT
DEFAULT_GATE = offline.OFFLINE_ROOT / "gate.json"
ROBUST_RESULTS = AILAB / "outputs/eval/cube/robust_v1"
ROBUST_PROBE_DIRECT = (
    AILAB / "outputs/eval/cube/waypoint_probe/red_offset25_direct/results.json"
)
FORMAL_SEED = 42
PROTOCOL = "t2"
ARMS = ("red", "blue_v2", "yellow_v2", "probe_red_offset25")
FORMAL_ROWS_SHA256 = "b75741cc514bbc3711e04232e8462f16a3181ed5f0bd754ebecd05b9ba9b0f71"
PROBE_TARGET_ROWS_SHA256 = "b184c6192d503323e262965dd89895b8b357fc5418e2fcbca34ba34c53557a3a"
REFERENCE_CONTRACTS = {
    "red": {
        "path": ROBUST_RESULTS / "red/results.json",
        "sha256": "fc1bf7377adcc1947b4f6653dea25701271e1eb795b6b46a9f7339a6fdcf68f2",
        "format_version": "cube_robust_condition_v1", "success_count": 46,
        "success_bits": "11111111111011011011111101111111111111111111111111",
    },
    "blue_v2": {
        "path": ROBUST_RESULTS / "blue_v2/results.json",
        "sha256": "6bcd094b6b92f8c156a883fa0abfde71c0572949697f823b7d3f7aa50bff9292",
        "format_version": "cube_robust_condition_v1", "success_count": 46,
        "success_bits": "11111111111011011011111101111111111111111111111111",
    },
    "yellow_v2": {
        "path": ROBUST_RESULTS / "yellow_v2/results.json",
        "sha256": "f72da59a3a327f74c5e4bfc915b31a82f88a31574dfb898e8521871fa6ef1cc0",
        "format_version": "cube_robust_condition_v1", "success_count": 43,
        "success_bits": "11111111111011011011110101101111111111011111111111",
    },
    "probe_red_offset25": {
        "path": ROBUST_PROBE_DIRECT,
        "sha256": "ecbc27d82ca7f1977a58032d39fccafecf0555557e5bbf79c52b8831145317aa",
        "format_version": "cube_waypoint_probe_eval_v1", "success_count": 47,
        "success_bits": "11111111111011111111111111001111111111111111111111",
    },
}


def _configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(AILAB),
        "HF_HOME": str(AILAB.parent / ".cache/huggingface"),
        "TORCH_HOME": str(AILAB.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(AILAB.parent / ".cache/pip"),
        "TMPDIR": str(AILAB.parent / "tmp"),
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        os.environ[key] = value
    (AILAB.parent / "tmp").mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _paired(current: Any, reference: Any) -> dict[str, Any]:
    new = np.asarray(current, dtype=bool)
    old = np.asarray(reference, dtype=bool)
    if new.ndim != 1 or new.shape != old.shape or len(new) not in (2, 50):
        raise ValueError(f"paired vectors malformed: {new.shape}/{old.shape}")
    return {
        "delta_pp": float((new.mean() - old.mean()) * 100.0),
        "reference_failure_to_play_success": np.flatnonzero(~old & new).tolist(),
        "reference_success_to_play_failure": np.flatnonzero(old & ~new).tolist(),
    }


def _array_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


def _validate_reference_payload(label: str, payload: dict[str, Any]) -> np.ndarray:
    contract = REFERENCE_CONTRACTS[label]
    metrics = payload.get("metrics", {})
    successes = np.asarray(metrics.get("episode_successes"), dtype=bool)
    bits = "".join("1" if value else "0" for value in successes)
    if (
        payload.get("format_version") != contract["format_version"]
        or successes.shape != (50,)
        or bits != contract["success_bits"]
        or int(metrics.get("success_count", -1)) != contract["success_count"]
        or int(metrics.get("num_eval", -1)) != 50
        or int(successes.sum()) != contract["success_count"]
        or not np.isclose(float(metrics.get("success_rate", np.nan)), contract["success_count"] * 2.0, atol=0, rtol=0)
        or _array_sha256(payload.get("evaluated_rows")) != FORMAL_ROWS_SHA256
    ):
        raise RuntimeError(f"frozen online reference payload changed: {label}")
    protocol = payload.get("protocol", {})
    if label in offline.CONDITIONS:
        if (
            payload.get("axis") != "regression"
            or payload.get("condition") != label
            or payload.get("variation") is not None
            or protocol.get("seed") != 42
            or protocol.get("eval_budget") != 50
            or protocol.get("goal_offset") != 25
            or protocol.get("num_eval") != 50
            or Path(str(protocol.get("checkpoint", ""))).resolve()
            != offline.ROBUST_CHECKPOINT.resolve()
        ):
            raise RuntimeError(f"frozen robust T2 protocol changed: {label}")
    else:
        if (
            payload.get("arm") != "red_offset25_direct"
            or payload.get("scenario") != "red"
            or payload.get("mode") != "direct"
            or payload.get("waypoint_spacing_m") is not None
            or protocol.get("id") != "t2"
            or protocol.get("seed") != 42
            or protocol.get("budget") != 50
            or protocol.get("num_samples") != 300
            or protocol.get("iterations") != 10
            or protocol.get("topk") != 30
            or protocol.get("horizon_model_steps") != 5
            or protocol.get("action_block_env_steps") != 5
            or _array_sha256(payload.get("target_rows")) != PROBE_TARGET_ROWS_SHA256
        ):
            raise RuntimeError("frozen robust probe-direct protocol changed")
    return successes


def _reference(label: str) -> tuple[dict[str, Any], np.ndarray]:
    contract = REFERENCE_CONTRACTS[label]
    path = Path(contract["path"]).resolve()
    if offline.legacy._sha256(path) != contract["sha256"]:
        raise RuntimeError(f"frozen online reference bytes changed: {label}")
    payload = _read_json(path)
    return payload, _validate_reference_payload(label, payload)


def _output(arm: str, num_eval: int) -> Path:
    return OUTPUT_ROOT / arm if num_eval == 50 else OUTPUT_ROOT / "smoke" / arm


def _prepare_output(path: Path, arm: str, num_eval: int, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    expected = _output(arm, num_eval).resolve()
    if resolved != expected:
        raise ValueError(f"output is frozen: expected={expected}, actual={resolved}")
    if resolved.is_symlink():
        raise ValueError(f"refusing symlink output: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"non-empty output: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _gate_contract(
    path: Path,
    checkpoint: Path,
    dataset: Path,
    manifest: Path,
    index: Path,
) -> dict[str, Any]:
    gate = offline._validate_gate(path, checkpoint)
    if gate["status"] != "PASS" or gate["expert_manifold"]["status"] != "PASS":
        raise RuntimeError("play-v1 online requires aggregate offline PASS")
    gate_path = path.resolve()
    frozen = _read_json(gate_path.parent / "frozen_inputs.json")
    if (
        frozen.get("expert_dataset") != offline.legacy._identity(dataset, include_sha256=False)
        or frozen.get("formal_manifest") != offline.legacy._identity(manifest)
        or frozen.get("memory_index") != offline.legacy._memory_index_identity(index)
        or _array_sha256(frozen.get("formal_rows")) != FORMAL_ROWS_SHA256
    ):
        raise RuntimeError("online dataset/manifest/index differs from aggregate gate inputs")
    return {
        "artifact": offline.legacy._identity(path),
        "format_version": gate["format_version"],
        "status": gate["status"],
        "all_requirements_AND": gate["all_requirements_AND"],
        "colors": gate["colors"],
        "expert_manifold": gate["expert_manifold"],
        "training_stopline": gate["training_stopline"],
        "canonical_inputs": {
            "expert_dataset": frozen["expert_dataset"],
            "formal_manifest": frozen["formal_manifest"],
            "memory_index": frozen["memory_index"],
            "formal_rows_sha256": FORMAL_ROWS_SHA256,
        },
        "probe_provenance": gate["artifacts"]["probe_provenance.json"],
    }


def _checkpoint_contract(checkpoint: Path) -> dict[str, Any]:
    checkpoint = offline._checkpoint(checkpoint)
    config = checkpoint.parent / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    return {
        "kind": "play_v1_unique_predictor_action_encoder_finetune",
        "run_id": offline.RUN_ID,
        "weights": offline.legacy._identity(checkpoint),
        "config": offline.legacy._identity(config),
        "warm_start": offline.legacy._identity(offline.ROBUST_CHECKPOINT),
    }


def _run_t2(args: argparse.Namespace, checkpoint: Path, gate: dict[str, Any], output: Path) -> int:
    import hdf5plugin  # noqa: F401
    import h5py
    import stable_worldmodel as swm
    import torch
    from gymnasium.spaces import Box

    if args.arm not in offline.CONDITIONS:
        raise ValueError(args.arm)
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("T2 online evaluation requires CUDA")
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = ood._formal_rows(dataset, args.manifest)
    rows = formal_rows[: args.num_eval]
    selected = dataset.get_row_data(rows)
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = np.asarray(selected[ep_key], dtype=np.int64)
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        query = np.concatenate(
            [memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows], axis=0
        )
    color, goal_type, audit_color = trust_common.condition_visual(args.arm)
    recolor_goals = recolor_meta = None
    if goal_type == "recolor":
        recolor_goals, recolor_meta = ood._load_recolor_goals(color, args.num_eval)
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        raw_inputs = audit._load_capture_inputs(h5, rows, audit_color, recolor_goals)
    model = swm.wm.utils.load_pretrained(str(checkpoint), cache_dir=str(AILAB)).to(args.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    scaler = legacy._standard_scaler(index)
    proxy = trust.TrustRegionCostProxy(model, PROTOCOL)
    recorder = ood.PlanningCostRecorder(args.num_eval)
    solver_cls = trust.make_trust_region_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=proxy, batch_size=1, num_samples=trust_common.NUM_SAMPLES,
        var_scale=trust_common.PROTOCOL_SPECS[PROTOCOL]["var_scale"],
        n_steps=trust_common.N_STEPS, topk=trust_common.TOPK, device=args.device,
        seed=FORMAL_SEED, callbacks=[recorder], selector="mean", recorder=recorder,
        trust_protocol=PROTOCOL,
    )
    config = swm.PlanConfig(horizon=trust_common.HORIZON, receding_horizon=trust_common.HORIZON,
                            action_block=trust_common.ACTION_BLOCK)
    action_space = Box(
        low=np.broadcast_to(-np.inf, (args.num_eval, trust_common.ACTION_DIM)),
        high=np.broadcast_to(np.inf, (args.num_eval, trust_common.ACTION_DIM)), dtype=np.float32,
    )
    solver.configure(action_space=action_space, n_envs=args.num_eval, config=config)
    policy = trust.make_trust_policy(swm.policy.WorldModelPolicy)(
        solver=solver, config=config, process={"action": scaler},
        transform={"pixels": ood._image_transform(224), "goal": ood._image_transform(224)},
        memory_index=index, cost_proxy=proxy, cost_recorder=recorder,
        eval_episodes=episodes, eval_rows=rows, initial_query_features=query, protocol=PROTOCOL,
    )
    world = swm.World(
        env_name="swm/OGBCube-v0", num_envs=args.num_eval, max_episode_steps=100,
        image_shape=(224, 224), env_type="single", ob_type="states", multiview=False,
        width=224, height=224, visualize_info=False, terminate_at_goal=True,
    )
    world.set_policy(policy)
    started = time.time()
    try:
        metrics, chosen = ood._evaluate(world, dataset, rows, goal_type, color, 50,
                                        output / "videos", recolor_goals)
    finally:
        world.close()
    elapsed = time.time() - started
    trace = trust._save_trace(output, proxy)
    history = ood._save_cost_history(output, recorder, rows, chosen["episodes"], chosen["starts"], "mean")
    pools = trust._save_first_cycle_pools(output, proxy, recorder, rows, episodes, raw_inputs,
                                          args.dataset, scaler, PROTOCOL, args.arm)
    reference_path = Path(REFERENCE_CONTRACTS[args.arm]["path"])
    reference, reference_success_all = _reference(args.arm)
    reference_success = reference_success_all[: args.num_eval]
    if not np.array_equal(np.asarray(reference["evaluated_rows"], dtype=np.int64)[: args.num_eval], rows):
        raise RuntimeError("robust-v1 T2 reference rows changed")
    payload = {
        "format_version": "cube_play_v1_t2_evaluation_v1",
        "protocol": {"id": PROTOCOL, **trust_common.PROTOCOL_SPECS[PROTOCOL],
                     "condition": args.arm, "color": color, "goal_type": goal_type,
                     "goal_recolor": recolor_meta, "selector": "legacy_updated_elite_mean",
                     "num_samples": trust_common.NUM_SAMPLES, "n_steps": trust_common.N_STEPS,
                     "topk": trust_common.TOPK, "seed": FORMAL_SEED,
                     "formal_full_50_order": args.num_eval == 50,
                     "checkpoint": _checkpoint_contract(checkpoint),
                     "aggregate_offline_gate": gate,
                     "only_experimental_factor_vs_robust_v1": "world_model dynamics checkpoint"},
        "formal_rows_verified": formal_rows, "evaluated_rows": rows,
        "metrics": metrics,
        "robust_v1_reference": offline.legacy._identity(reference_path),
        "versus_robust_v1": _paired(metrics["episode_successes"], reference_success),
        "elapsed_seconds": elapsed, "trace": trace, "cost_history": history,
        "first_cycle_pools": pools,
    }
    trust_common.write_json(output / "results.json", payload)
    successes = ", ".join("True" if value else "False" for value in metrics["episode_successes"])
    (output / "results.txt").write_text(
        f"protocol: t2\ncondition: {args.arm}\nsuccess_rate: {metrics['success_rate']:.6f}\n"
        f"success_count: {metrics['success_count']}/{metrics['num_eval']}\n"
        f"episode_successes: [{successes}]\nelapsed_seconds: {elapsed:.6f}\n", encoding="utf-8")
    print(output)
    return 0


def _probe_contract_for_frozen_play(probe: Any, metadata_path: Path,
                                    formal_episodes: np.ndarray,
                                    gate: dict[str, Any]) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    expected_sha = str(probe.payload["embedding_dataset_metadata_sha256"])
    actual_sha = probe_common.sha256_file(metadata_path)
    excluded = np.asarray(metadata.get("excluded_formal_episodes"), dtype=np.int64)
    metric = float(probe.payload["metrics"]["test"]["xyz_error_mm"]["median"])
    if (
        metadata.get("format_version") != "cube_block4d_embedding_dataset_v1"
        or expected_sha != actual_sha
        or excluded.shape != (50,)
        or set(excluded.tolist()) != set(formal_episodes.tolist())
        or not np.isfinite(metric)
        or metric >= 15.0
        or gate["status"] != "PASS"
    ):
        raise RuntimeError("robust probe reuse contract failed")
    gate_path = Path(gate["artifact"]["path"]).resolve()
    provenance_path = gate_path.parent / "probe_provenance.json"
    provenance = _read_json(provenance_path)
    probe_identity = offline.legacy._identity(offline.ROBUST_PROBE)
    metadata_identity = offline.legacy._identity(metadata_path)
    probe_world_state = str(probe.payload.get("world_model_state_sha256", ""))
    metadata_world_state = str(metadata.get("world_model_state_sha256", ""))
    if (
        gate.get("probe_provenance") != offline.legacy._identity(provenance_path)
        or provenance.get("frozen_modules") != list(offline.FROZEN_MODULES)
        or provenance.get("probe") != probe_identity
        or provenance.get("probe_dataset_metadata") != metadata_identity
        or provenance.get("probe_world_model_state_sha256") != probe_world_state
        or provenance.get("metadata_world_model_state_sha256") != metadata_world_state
        or probe_world_state != metadata_world_state
        or provenance.get("embedding_dataset_metadata_sha256") != actual_sha
        or Path(str(probe.payload.get("world_model_checkpoint", ""))).resolve()
        != offline.ROBUST_CHECKPOINT.resolve()
        or probe.payload.get("canonical_robust_v1", {}).get("checkpoint", {}).get("sha256")
        != offline.ROBUST_CHECKPOINT_SHA256
    ):
        raise RuntimeError("offline gate/probe identity, metadata, or world-state differs")
    return {
        "reuse_reason": "play-v1 encoder/projector are elementwise equal to robust-v1",
        "probe_test_median_xyz_error_mm": metric,
        "probe_gate_mm_strict": 15.0,
        "probe_dataset_metadata": waypoint._identity(metadata_path),
        "probe": probe.provenance(),
        "offline_frozen_representation": provenance,
    }


def _run_probe_red(args: argparse.Namespace, checkpoint: Path, gate: dict[str, Any], output: Path) -> int:
    import stable_worldmodel as swm
    import torch

    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("probe online evaluation requires CUDA")
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = waypoint._require_sha(
        "standard rows", ood._formal_rows(dataset, args.manifest), waypoint.STANDARD_ROWS_SHA256
    )
    formal_state = dataset.get_row_data(formal_rows)
    formal_episodes = np.asarray(formal_state["ep_idx"], dtype=np.int64)
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    long_index = brain.HeldoutMemoryIndex(args.index, args.dataset, formal_episodes)
    scaler = legacy._standard_scaler(index)
    model = swm.wm.utils.load_pretrained(str(checkpoint), cache_dir=str(AILAB)).to(args.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    loaded_probe = xyz_common.LoadedXYZProbe(offline.ROBUST_PROBE, args.device)
    probe_contract = _probe_contract_for_frozen_play(
        loaded_probe, offline.ROBUST_PROBE_METADATA, formal_episodes, gate
    )
    spec = waypoint.ArmSpec(
        name="probe_red_offset25", scenario="red", mode="direct", budget=50, spacing_m=None
    )
    helper_args = argparse.Namespace(
        num_eval=args.num_eval, dataset=args.dataset, device=args.device,
        video=args.video, exercise_smoke_branches=False,
    )
    payload = waypoint._run_arm(
        args=helper_args, spec=spec, output=output, dataset=dataset,
        formal_rows=formal_rows, index=index, long_index=long_index, scaler=scaler,
        base_model=model, loaded_probe=loaded_probe, probe_contract=probe_contract,
        robust_contract=_checkpoint_contract(checkpoint),
    )
    reference, reference_success_all = _reference("probe_red_offset25")
    rows = np.asarray(payload["evaluated_rows"], dtype=np.int64)
    targets = np.asarray(payload["target_rows"], dtype=np.int64)
    if (
        not np.array_equal(np.asarray(reference["evaluated_rows"], dtype=np.int64)[: args.num_eval], rows)
        or not np.array_equal(np.asarray(reference["target_rows"], dtype=np.int64)[: args.num_eval], targets)
    ):
        raise RuntimeError("robust probe-direct reference pairing changed")
    payload["format_version"] = "cube_play_v1_probe_red_offset25_v1"
    payload["aggregate_offline_gate"] = gate
    payload["robust_probe_direct_reference"] = offline.legacy._identity(ROBUST_PROBE_DIRECT)
    payload["versus_robust_probe_direct"] = _paired(
        payload["metrics"]["episode_successes"],
        reference_success_all[: args.num_eval],
    )
    probe_common.write_json(output / "results.json", payload)
    print(output)
    return 0


def self_test(_: argparse.Namespace) -> int:
    pair = _paired([True, False], [False, True])
    if pair["delta_pp"] != 0 or pair["reference_failure_to_play_success"] != [0] or pair["reference_success_to_play_failure"] != [1]:
        raise AssertionError("paired flip contract failed")
    if set(ARMS) != {"red", "blue_v2", "yellow_v2", "probe_red_offset25"}:
        raise AssertionError("online arm matrix changed")
    negative = 0
    for current, old in (([True], [True, False]), ([[True]], [[True]])):
        try:
            _paired(current, old)
        except ValueError:
            negative += 1
    if negative != 2:
        raise AssertionError("paired-vector negatives did not fail closed")
    reference_negatives = 0
    reference_positive = {}
    for label in ARMS:
        payload, successes = _reference(label)
        reference_positive[label] = int(successes.sum())
        for mutation in ("count", "vector", "protocol"):
            bad = json.loads(json.dumps(payload))
            if mutation == "count":
                bad["metrics"]["success_count"] -= 1
            elif mutation == "vector":
                bad["metrics"]["episode_successes"][0] = not bad["metrics"]["episode_successes"][0]
            elif label in offline.CONDITIONS:
                bad["protocol"]["seed"] = 43
            else:
                bad["protocol"]["topk"] = 29
            try:
                _validate_reference_payload(label, bad)
            except RuntimeError:
                reference_negatives += 1
    if reference_positive != {"red": 46, "blue_v2": 46, "yellow_v2": 43, "probe_red_offset25": 47}:
        raise AssertionError("frozen reference success counts changed")
    if reference_negatives != len(ARMS) * 3:
        raise AssertionError("reference contract negatives did not fail closed")
    print(json.dumps({"self_test": "PASS", "arms": ARMS,
                      "aggregate_pass_required_for_smoke_and_formal": True,
                      "paired_negative_checks": negative,
                      "reference_success_counts": reference_positive,
                      "reference_negative_checks": reference_negatives}, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    _configure_storage()
    if args.self_test:
        return self_test(args)
    if args.seed != FORMAL_SEED or args.num_eval not in (2, 50):
        raise ValueError("play-v1 online is frozen to seed42 and 2/50 env")
    if args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError("formal evaluation requires --authorize-formal")
    if args.num_eval == 2 and not args.video:
        raise ValueError("2-env smoke requires --video")
    for path in (args.dataset, args.manifest, args.index, args.checkpoint, args.gate):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint = offline._checkpoint(args.checkpoint)
    gate = _gate_contract(
        args.gate, checkpoint, args.dataset, args.manifest, args.index
    )
    _reference(args.arm)
    output = _prepare_output(args.output or _output(args.arm, args.num_eval), args.arm,
                             args.num_eval, args.overwrite)
    if args.arm == "probe_red_offset25":
        return _run_probe_red(args, checkpoint, gate, output)
    return _run_t2(args, checkpoint, gate, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="red")
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--dataset", type=Path, default=trust_common.DATASET)
    parser.add_argument("--manifest", type=Path, default=trust_common.MANIFEST)
    parser.add_argument("--index", type=Path, default=trust_common.MEMORY_INDEX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
