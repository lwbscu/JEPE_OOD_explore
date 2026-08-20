#!/usr/bin/env python3
"""Paired latent/probe-cost evaluation on the frozen Cube goal-OOD rows.

The evaluator deliberately reuses the exact start and target rows persisted by
``eval_goal_ood.py``.  The probe arm predicts terminal block xyz from the
world-model rollout and compares it with privileged goal xyz; the goal image is
removed before rollout and therefore cannot contribute to planner cost.  The
real HDF5 goal frame and pose are still installed in the environment so the
physical task and success criterion remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_cube_memory_index as memory  # noqa: E402
import cube_imagination_error_common as xyz_common  # noqa: E402
import cube_probe_common as probe_common  # noqa: E402
import cube_trust_region_common as t2common  # noqa: E402
import eval_goal_ood as goal_ood  # noqa: E402
import eval_memory_seed as memory_legacy  # noqa: E402
import eval_ood_color as ood  # noqa: E402


PROJECT = HERE.parent
DEFAULT_REFERENCE = PROJECT / "outputs/eval/cube/goal_ood_curve"
DEFAULT_OUTPUT = PROJECT / "outputs/eval/cube/probe_goal_cost"
ROBUST_CHECKPOINT = (
    PROJECT
    / "checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt"
)
ROBUST_CHECKPOINT_SHA256 = (
    "cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed"
)
ROBUST_MODEL_CONFIG = (
    PROJECT
    / "checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/config.json"
)
ROBUST_MODEL_CONFIG_SHA256 = (
    "86f2ed24c61b48354416c23af51aa51279ae28a33cb36b7ebc3d057eec2b8c0d"
)
ROBUST_RUN_PLAN = (
    PROJECT / "outputs/train/robust_v1/lewm-cube-robust_v1/run_plan.json"
)
ROBUST_RUN_PLAN_SHA256 = (
    "5830ad4091e13764f4eee765805e247e36c6968b52afd34397ef65745752bbf9"
)
TIERS: Mapping[str, str] = {
    "in_box": "in_box",
    "plus_05cm": "plus_05cm",
    # plus_10cm and plus_20cm selected the same 50 real frames.  Evaluate the
    # physical support point once and name it by its observed median distance.
    "fallback_max": "plus_10cm",
}
MODES = ("latent", "probe")
PROBE_XYZ_GATE_MM = 15.0


def _jsonable(value: Any) -> Any:
    return probe_common.jsonable(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _identity(path: Path) -> dict[str, Any]:
    return probe_common.file_identity(path.resolve())


def _same_int_vector(label: str, left: Any, right: Any, count: int = 50) -> np.ndarray:
    a = np.asarray(left, dtype=np.int64)
    b = np.asarray(right, dtype=np.int64)
    if a.shape != (count,) or b.shape != (count,) or not np.array_equal(a, b):
        mismatch = np.flatnonzero(a != b).tolist() if a.shape == b.shape else "shape"
        raise ValueError(
            f"{label} mismatch: expected_shape={(count,)}, actual={a.shape}/{b.shape}, "
            f"mismatch_indices={mismatch[:10] if isinstance(mismatch, list) else mismatch}"
        )
    return a


def _load_frozen_targets(reference_root: Path, tier: str) -> dict[str, Any]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")
    source_tier = TIERS[tier]
    selection_path = reference_root / source_tier / "target_selection.json"
    result_path = reference_root / source_tier / "results.json"
    selection = _read_json(selection_path)
    result = _read_json(result_path)
    if result.get("format_version") != "cube_goal_ood_t2_v1":
        raise ValueError(f"unsupported latent reference format: {result.get('format_version')}")
    target_rows = _same_int_vector(
        f"{tier} selection/results target rows",
        selection.get("selected_rows"),
        result.get("target_rows"),
    )
    start_rows = _same_int_vector(
        f"{tier} evaluated/formal rows",
        result.get("evaluated_rows"),
        result.get("formal_rows_verified"),
    )
    if not selection.get("real_hdf5_frames"):
        raise ValueError(f"{selection_path} is not marked as real HDF5 frames")
    if tier == "fallback_max":
        other_selection_path = reference_root / "plus_20cm" / "target_selection.json"
        other_result_path = reference_root / "plus_20cm" / "results.json"
        other_selection = _read_json(other_selection_path)
        other_result = _read_json(other_result_path)
        _same_int_vector(
            "deduplicated +10/+20cm selection rows",
            target_rows,
            other_selection.get("selected_rows"),
        )
        _same_int_vector(
            "deduplicated +10/+20cm result rows",
            target_rows,
            other_result.get("target_rows"),
        )
        if not selection.get("fallback_used") or not other_selection.get("fallback_used"):
            raise ValueError("fallback_max requires both source tiers to be marked fallback")
    episode_successes = np.asarray(
        result.get("metrics", {}).get("episode_successes"), dtype=bool
    )
    if episode_successes.shape != (50,):
        raise ValueError(f"malformed reference success vector: {result_path}")
    return {
        "tier": tier,
        "source_tier": source_tier,
        "start_rows": start_rows,
        "target_rows": target_rows,
        "selection": selection,
        "selection_identity": _identity(selection_path),
        "latent_reference": result,
        "latent_reference_identity": _identity(result_path),
        "latent_reference_successes": episode_successes,
    }


def _goal_value(value: Any, predicted: Any, last_dim: int) -> Any:
    """Reduce World info shapes to (batch, samples, dimension)."""

    import torch

    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=predicted.device)
    value = value.to(device=predicted.device, dtype=predicted.dtype)
    if value.shape[-1] != last_dim:
        raise ValueError(f"goal last dimension must be {last_dim}, got {value.shape}")
    if value.ndim == 4:
        value = value[..., -1, :]
    elif value.ndim == 2:
        value = value[:, None, :]
    if value.ndim != 3:
        raise ValueError(f"goal value must reduce to (B,S,D), got {value.shape}")
    if value.shape[:2] != predicted.shape[:2]:
        value = value.expand(predicted.shape[0], predicted.shape[1], last_dim)
    return value


def make_probe_goal_cost_model(base: Any, probe: xyz_common.LoadedXYZProbe) -> Any:
    """Wrap a world model with an xyz-only goal cost and blind its goal image."""

    import torch

    class ProbeGoalCostModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.probe_module = probe.model

        @torch.inference_mode()
        def get_cost(self, info_dict: dict[str, Any], action_candidates: Any) -> Any:
            if "goal_privileged_block_0_pos" not in info_dict:
                raise KeyError("probe-goal cost requires privileged goal block xyz")
            device = next(self.base.parameters()).device
            # Do not mutate the policy-owned dictionary.  Most importantly,
            # remove the goal pixels before entering the world-model rollout.
            rollout_info = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in info_dict.items()
                if key != "goal"
            }
            rolled = self.base.rollout(rollout_info, action_candidates)
            terminal = rolled["predicted_emb"][..., -1, :]
            predicted = probe(terminal)
            goal_xyz = _goal_value(
                info_dict["goal_privileged_block_0_pos"], predicted, 3
            )
            cost = probe_common.probe_physical_cost(predicted, goal_xyz)
            if tuple(cost.shape) != tuple(action_candidates.shape[:2]):
                raise RuntimeError(
                    "probe-goal cost shape mismatch: "
                    f"expected={tuple(action_candidates.shape[:2])}, actual={tuple(cost.shape)}"
                )
            return cost

    return ProbeGoalCostModel()


def _validate_probe_contract(
    probe: xyz_common.LoadedXYZProbe,
    metadata_path: Path,
    base_model: Any,
    formal_episodes: np.ndarray,
) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    if metadata.get("format_version") != "cube_block4d_embedding_dataset_v1":
        raise ValueError(f"unsupported probe dataset format: {metadata.get('format_version')}")
    actual_metadata_sha = probe_common.sha256_file(metadata_path)
    expected_metadata_sha = str(
        probe.payload["embedding_dataset_metadata_sha256"]
    )
    if actual_metadata_sha != expected_metadata_sha:
        raise ValueError(
            "probe dataset/checkpoint provenance mismatch: "
            f"expected={expected_metadata_sha}, actual={actual_metadata_sha}, "
            f"path={metadata_path}"
        )
    actual_model_sha = probe_common.torch_module_sha256(base_model)
    expected_model_sha = str(probe.payload["world_model_state_sha256"])
    metadata_model_sha = str(metadata.get("world_model_state_sha256"))
    if actual_model_sha != expected_model_sha or metadata_model_sha != expected_model_sha:
        raise ValueError(
            "probe/world-model provenance mismatch: "
            f"expected={expected_model_sha}, actual_model={actual_model_sha}, "
            f"dataset={metadata_model_sha}"
        )
    excluded = np.asarray(metadata.get("excluded_formal_episodes"), dtype=np.int64)
    if excluded.shape != (50,) or set(excluded.tolist()) != set(formal_episodes.tolist()):
        raise ValueError(
            "probe dataset does not exclude exactly the frozen 50 evaluation episodes"
        )
    test_metric = float(
        probe.payload.get("metrics", {})
        .get("test", {})
        .get("xyz_error_mm", {})
        .get("median", np.inf)
    )
    if not np.isfinite(test_metric) or test_metric >= PROBE_XYZ_GATE_MM:
        raise ValueError(
            "probe quality gate failed: "
            f"expected <{PROBE_XYZ_GATE_MM:.1f}mm, actual={test_metric:.6f}mm"
        )
    return {
        "world_model_state_sha256": actual_model_sha,
        "probe_test_median_xyz_error_mm": test_metric,
        "probe_gate_mm_strict": PROBE_XYZ_GATE_MM,
        "probe_dataset_metadata": _identity(metadata_path),
        "probe": probe.provenance(),
    }


def _robust_checkpoint_contract(checkpoint: Path, formal: bool) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    checkpoint_identity = _identity(checkpoint)
    config_identity = _identity(ROBUST_MODEL_CONFIG)
    run_plan_identity = _identity(ROBUST_RUN_PLAN)
    matches_path = checkpoint == ROBUST_CHECKPOINT.resolve()
    matches_weights_sha = checkpoint_identity["sha256"] == ROBUST_CHECKPOINT_SHA256
    matches_config_sha = (
        config_identity["sha256"] == ROBUST_MODEL_CONFIG_SHA256
    )
    matches_run_plan_sha = (
        run_plan_identity["sha256"] == ROBUST_RUN_PLAN_SHA256
    )
    if formal and not (
        matches_path
        and matches_weights_sha
        and matches_config_sha
        and matches_run_plan_sha
    ):
        raise ValueError(
            "formal probe-goal evaluation is frozen to canonical robust_v1: "
            f"expected_path={ROBUST_CHECKPOINT.resolve()}, actual_path={checkpoint}, "
            f"expected_weights_sha={ROBUST_CHECKPOINT_SHA256}, "
            f"actual_weights_sha={checkpoint_identity['sha256']}, "
            f"expected_config_sha={ROBUST_MODEL_CONFIG_SHA256}, "
            f"actual_config_sha={config_identity['sha256']}, "
            f"expected_run_plan_sha={ROBUST_RUN_PLAN_SHA256}, "
            f"actual_run_plan_sha={run_plan_identity['sha256']}"
        )
    return {
        "checkpoint": checkpoint_identity,
        "model_config": config_identity,
        "training_run_plan": run_plan_identity,
        "expected_checkpoint_path": str(ROBUST_CHECKPOINT.resolve()),
        "expected_weights_sha256": ROBUST_CHECKPOINT_SHA256,
        "expected_model_config_sha256": ROBUST_MODEL_CONFIG_SHA256,
        "expected_run_plan_sha256": ROBUST_RUN_PLAN_SHA256,
        "matches_canonical_path": matches_path,
        "matches_canonical_weights_sha256": matches_weights_sha,
        "matches_canonical_model_config_sha256": matches_config_sha,
        "matches_canonical_run_plan_sha256": matches_run_plan_sha,
        "formal_frozen": bool(formal),
    }


def _prepare_output(path: Path, overwrite: bool) -> Path:
    resolved = probe_common.ensure_output_child(
        path, DEFAULT_OUTPUT, "probe-goal evaluation output"
    )
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"non-empty output: {resolved}; pass --overwrite")
        if resolved.is_symlink():
            raise ValueError(f"refusing symlink output: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _paired_delta(current: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    current = np.asarray(current, dtype=bool)
    baseline = np.asarray(baseline, dtype=bool)
    if current.shape != baseline.shape or current.ndim != 1 or current.size not in (2, 50):
        raise ValueError(
            "paired vectors must have the same frozen smoke/formal length (2 or 50): "
            f"current={current.shape}, baseline={baseline.shape}"
        )
    return {
        "success_rate_delta_pp": float((current.mean() - baseline.mean()) * 100.0),
        "baseline_failure_to_current_success": np.flatnonzero(~baseline & current),
        "baseline_success_to_current_failure": np.flatnonzero(baseline & ~current),
    }


def _existing_arm(root: Path, tier: str, mode: str) -> dict[str, Any] | None:
    path = root / tier / mode / "results.json"
    return _read_json(path) if path.is_file() else None


def _write_summary(root: Path) -> None:
    tiers: dict[str, Any] = {}
    for tier in TIERS:
        arms: dict[str, Any] = {}
        for mode in MODES:
            payload = _existing_arm(root, tier, mode)
            if payload is None:
                continue
            metrics = payload["metrics"]
            arms[mode] = {
                "success_count": metrics["success_count"],
                "success_rate": metrics["success_rate"],
                "episode_successes": metrics["episode_successes"],
                "results": _identity(root / tier / mode / "results.json"),
            }
        if "latent" in arms and "probe" in arms:
            arms["probe_vs_latent"] = _paired_delta(
                np.asarray(arms["probe"]["episode_successes"], dtype=bool),
                np.asarray(arms["latent"]["episode_successes"], dtype=bool),
            )
        if arms:
            tiers[tier] = arms
    probe_common.write_json(
        root / "summary.json",
        {
            "format_version": "cube_probe_goal_ood_summary_v1",
            "tiers": tiers,
            "tier_sources": dict(TIERS),
            "cost_arms": list(MODES),
        },
    )


def run(args: argparse.Namespace) -> int:
    probe_common.configure_storage()
    formal = args.num_eval == 50
    if args.seed != t2common.FORMAL_SEED or args.protocol != "t2":
        raise ValueError("protocol is frozen to T2/seed42")
    if args.num_eval not in (2, 50):
        raise ValueError("num-eval is frozen to 2 smoke or 50 formal")
    if args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError("pass --authorize-formal for 50-env evaluation")
    if args.device != "cuda":
        raise ValueError("the frozen goal-OOD evaluator requires --device cuda")
    for path, label in (
        (args.dataset, "dataset"),
        (args.manifest, "manifest"),
        (args.index, "memory index"),
        (args.checkpoint, "robust checkpoint"),
        (args.reference_root, "goal-OOD reference root"),
    ):
        probe_common.ensure_data_disk(path, label)
        if not path.exists():
            raise FileNotFoundError(path)
    root = args.output.resolve()
    output_parent = DEFAULT_OUTPUT.resolve()
    smoke_parent = output_parent / "smoke"
    if formal and root != output_parent:
        raise ValueError(
            "formal output root is frozen: "
            f"expected={output_parent}, actual={root}"
        )
    if not formal and root != smoke_parent and smoke_parent not in root.parents:
        raise ValueError(
            "2-env smoke output must be explicitly isolated below the smoke root: "
            f"expected_parent={smoke_parent}, actual={root}"
        )
    robust_contract = _robust_checkpoint_contract(args.checkpoint, formal)
    requested_modes = list(MODES) if args.mode == "both" else [args.mode]
    if "probe" in requested_modes:
        if args.probe is None or args.probe_dataset_metadata is None:
            raise ValueError("probe mode requires --probe and --probe-dataset-metadata")
        for path, label in (
            (args.probe, "probe checkpoint"),
            (args.probe_dataset_metadata, "probe dataset metadata"),
        ):
            probe_common.ensure_data_disk(path, label)
            if not path.is_file():
                raise FileNotFoundError(path)

    import hdf5plugin  # noqa: F401
    import h5py
    import stable_worldmodel as swm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = ood._formal_rows(dataset, args.manifest)
    if np.asarray(formal_rows).shape != (50,):
        raise ValueError("manifest did not produce the frozen 50 formal rows")
    rows = np.asarray(formal_rows[: args.num_eval], dtype=np.int64)
    selected = dataset.get_row_data(formal_rows)
    eval_episodes_all = np.asarray(selected["ep_idx"], dtype=np.int64)
    eval_episodes = eval_episodes_all[: args.num_eval]
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        initial_query_features = np.concatenate(
            [memory.feature_chunk(h5, int(row), int(row) + 1) for row in rows], axis=0
        )
    index = memory.CubeMemoryIndex(args.index, args.dataset)
    scaler = memory_legacy._standard_scaler(index)
    base_model = swm.wm.utils.load_pretrained(
        args.checkpoint, cache_dir=str(PROJECT)
    ).to(args.device).eval().requires_grad_(False)
    base_model.interpolate_pos_encoding = True
    actual_model_sha = probe_common.torch_module_sha256(base_model)

    probe = None
    probe_contract = None
    if "probe" in requested_modes:
        probe = xyz_common.LoadedXYZProbe(args.probe, args.device)
        probe_contract = _validate_probe_contract(
            probe,
            args.probe_dataset_metadata,
            base_model,
            eval_episodes_all,
        )

    selected_tiers = list(TIERS) if args.tier == "all" else [args.tier]
    for tier in selected_tiers:
        frozen = _load_frozen_targets(args.reference_root.resolve(), tier)
        if not np.array_equal(frozen["start_rows"][: args.num_eval], rows):
            raise ValueError(f"{tier} start rows do not match seed42 formal rows")
        target_rows = frozen["target_rows"][: args.num_eval]
        for mode in requested_modes:
            output = _prepare_output(root / tier / mode, args.overwrite)
            planner_model = (
                base_model
                if mode == "latent"
                else make_probe_goal_cost_model(base_model, probe).to(args.device).eval()
            )
            result = goal_ood._evaluate_tier(
                dataset=dataset,
                rows=rows,
                target_rows=target_rows,
                eval_episodes=eval_episodes,
                initial_query_features=initial_query_features,
                index=index,
                model=planner_model,
                scaler=scaler,
                output=output,
                budget=50,
                protocol="t2",
                video=args.video,
            )
            current_successes = np.asarray(
                result["metrics"]["episode_successes"], dtype=bool
            )
            reference_successes = frozen["latent_reference_successes"][: args.num_eval]
            payload = {
                "format_version": "cube_probe_goal_ood_eval_v1",
                "tier": tier,
                "source_tier": frozen["source_tier"],
                "cost_mode": mode,
                "protocol": {
                    "id": "t2",
                    "seed": args.seed,
                    "budget": 50,
                    "num_samples": t2common.NUM_SAMPLES,
                    "iterations": t2common.N_STEPS,
                    "topk": t2common.TOPK,
                    "memory_seed_slots": "1..10",
                    "sigma_0.1_seed_slots": "11..30",
                    "checkpoint": _identity(args.checkpoint),
                    "world_model_state_sha256": actual_model_sha,
                    "canonical_robust_v1": robust_contract,
                    "goal_cost_contract": (
                        "latent distance to encoded real goal frame"
                        if mode == "latent"
                        else (
                            "terminal predicted block xyz to privileged goal xyz; "
                            "goal pixels removed before rollout"
                        )
                    ),
                    "environment_goal_contract": "real HDF5 frame pose and pixels",
                    "fixed50_exclusion": True,
                },
                "probe_contract": probe_contract if mode == "probe" else None,
                "formal_rows_verified": formal_rows,
                "evaluated_rows": rows,
                "target_rows": target_rows,
                "frozen_target_selection": {
                    "selection": frozen["selection_identity"],
                    "latent_reference": frozen["latent_reference_identity"],
                    "selected_distance_median_m": frozen["selection"].get(
                        "selected_distance_median_m"
                    ),
                    "fallback_used": frozen["selection"].get("fallback_used"),
                },
                "versus_existing_masked_latent": (
                    _paired_delta(current_successes, reference_successes)
                    if args.num_eval == 50
                    else None
                ),
                **result,
                "trace_cost_field_note": (
                    "trust_trace.npz legacy field latent_costs contains xyz probe costs"
                    if mode == "probe"
                    else "trust_trace.npz latent_costs contains latent goal costs"
                ),
                "script": _identity(Path(__file__)),
            }
            probe_common.write_json(output / "results.json", payload)
            successes = ", ".join(
                "True" if value else "False" for value in current_successes
            )
            (output / "results.txt").write_text(
                f"tier: {tier}\n"
                f"cost_mode: {mode}\n"
                f"success_rate: {result['metrics']['success_rate']:.6f}\n"
                f"success_count: {result['metrics']['success_count']}/{args.num_eval}\n"
                f"episode_successes: [{successes}]\n",
                encoding="utf-8",
            )
            print(
                f"{tier}/{mode}: {result['metrics']['success_count']}/{args.num_eval} "
                f"({result['metrics']['success_rate']:.2f}%)"
            )
    run_manifest = {
        "format_version": "cube_probe_goal_ood_run_manifest_v1",
        "formal": formal,
        "num_eval": args.num_eval,
        "seed": args.seed,
        "protocol": "t2",
        "tiers": selected_tiers,
        "cost_modes": requested_modes,
        "output_root": str(root),
        "canonical_robust_v1": robust_contract,
        "world_model_state_sha256": actual_model_sha,
        "probe_contract": probe_contract,
        "reference_root": str(args.reference_root.resolve()),
        "fixed50_manifest": _identity(args.manifest),
        "script": _identity(Path(__file__)),
    }
    probe_common.write_json(root / "run_manifest.json", run_manifest)
    _write_summary(root)
    print(root)
    return 0


def self_test(args: argparse.Namespace) -> int:
    import torch

    contracts = {
        tier: _load_frozen_targets(args.reference_root.resolve(), tier)
        for tier in TIERS
    }
    smoke_delta = _paired_delta(
        np.asarray([False, True]), np.asarray([False, False])
    )
    assert smoke_delta["success_rate_delta_pp"] == 50.0
    common_rows = contracts["in_box"]["start_rows"]
    for tier, contract in contracts.items():
        _same_int_vector(f"self-test common starts/{tier}", common_rows, contract["start_rows"])

    class DummyBase(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.saw_goal = None

        def rollout(self, info: dict[str, Any], actions: Any) -> dict[str, Any]:
            self.saw_goal = "goal" in info
            shape = (*actions.shape[:2], 1, 3)
            return {"predicted_emb": torch.zeros(shape, dtype=actions.dtype)}

    class DummyProbe:
        def __init__(self) -> None:
            self.model = torch.nn.Identity()

        def __call__(self, embedding: Any) -> Any:
            return embedding

    base = DummyBase()
    wrapper = make_probe_goal_cost_model(base, DummyProbe())
    actions = torch.zeros((1, 2, 5, 25))
    costs = wrapper.get_cost(
        {
            "pixels": torch.zeros((1, 2, 1, 3, 2, 2)),
            "goal": torch.ones((1, 2, 1, 3, 2, 2)),
            "goal_privileged_block_0_pos": torch.tensor([[[[1.0, 2.0, 2.0]]]]),
        },
        actions,
    )
    assert tuple(costs.shape) == (1, 2)
    assert torch.allclose(costs, torch.full((1, 2), 9.0))
    assert base.saw_goal is False
    print(
        json.dumps(
            {
                "status": "ok",
                "tiers": dict(TIERS),
                "formal_row_count": int(len(common_rows)),
                "fallback_rows_deduplicated": True,
                "probe_goal_image_blinded": True,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=t2common.DATASET)
    parser.add_argument("--manifest", type=Path, default=t2common.MANIFEST)
    parser.add_argument("--index", type=Path, default=t2common.MEMORY_INDEX)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--probe-dataset-metadata", type=Path)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tier", choices=("all", *TIERS), default="all")
    parser.add_argument("--mode", choices=("both", *MODES), default="both")
    parser.add_argument("--protocol", default="t2")
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluation")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
