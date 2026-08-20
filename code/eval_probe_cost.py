#!/usr/bin/env python3
"""Gated online Cube evaluation using a physical probe as the CEM cost.

This standalone entry point reuses the frozen color-OOD evaluator's row,
rendering, video, and cost-history contracts without modifying it.  Online
execution is refused unless the matching offline three-color report passes its
predeclared per-color gate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import cube_probe_common as common  # noqa: E402
import eval_ood_color as ood  # noqa: E402


def _safe_output(path: Path, overwrite: bool) -> Path:
    path = common.ensure_output_child(path, common.ONLINE_ROOT, "probe online output")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_output(condition: str, num_eval: int) -> Path:
    return (
        common.ONLINE_ROOT / f"{condition}_probe_cost"
        if num_eval == 50
        else common.ONLINE_ROOT / "smoke" / f"{condition}_probe_cost_n2"
    )


def _load_gate(path: Path, probe_path: Path, checkpoint: str, yaw_weight: float) -> dict[str, Any]:
    path = common.ensure_data_disk(path, "offline gate report")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != "cube_probe_offline_rerank_v1":
        raise ValueError(f"unsupported gate report: {payload.get('format_version')}")
    gate = payload.get("gate", {})
    if not gate.get("passed_all_colors") or not gate.get("online_evaluation_authorized"):
        raise PermissionError(
            "offline probe-cost gate has not passed independently for all colors"
        )
    for condition, threshold in common.BASELINE_CEM_MEAN_EVER_SUCCESS.items():
        result = payload.get("by_condition", {}).get(condition, {})
        actual_threshold = int(result.get("cem_mean_ever_success_count", -1))
        probe_success = int(result.get("probe_top1_ever_success_count", -1))
        if (
            actual_threshold != threshold
            or probe_success < threshold
            or not result.get("gate_passed")
        ):
            raise ValueError(
                "offline gate report is internally inconsistent: "
                f"condition={condition}, expected_cem={threshold}, "
                f"actual_cem={actual_threshold}, probe_top1={probe_success}"
            )
    actual_probe_sha = common.sha256_file(probe_path)
    expected_probe_sha = payload["probe"]["checkpoint"]["sha256"]
    if actual_probe_sha != expected_probe_sha:
        raise ValueError(
            "online probe differs from gated probe: "
            f"expected={expected_probe_sha}, actual={actual_probe_sha}"
        )
    if payload["world_model_checkpoint"] != checkpoint:
        raise ValueError(
            "online world-model checkpoint differs from gate: "
            f"expected={payload['world_model_checkpoint']}, actual={checkpoint}"
        )
    offline_yaw = float(payload["protocol"]["yaw_weight"])
    if offline_yaw != float(yaw_weight):
        raise ValueError(
            f"yaw weight differs from gate: expected={offline_yaw}, actual={yaw_weight}"
        )
    return {
        "path": str(path),
        "sha256": common.sha256_file(path),
        "gate": gate,
        "probe_sha256": actual_probe_sha,
    }


def _goal_value(value: Any, predicted: Any, last_dim: int) -> Any:
    import torch

    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=predicted.device)
    value = value.to(device=predicted.device, dtype=predicted.dtype)
    if value.shape[-1] != last_dim:
        raise ValueError(f"goal value last dimension must be {last_dim}, got {value.shape}")
    # Solver-expanded inputs are normally (B,S,T,D); tolerate (B,S,D) and
    # (B,D) to keep the cost wrapper independently testable.
    if value.ndim == 4:
        value = value[..., -1, :]
    elif value.ndim == 2:
        value = value[:, None, :]
    if value.ndim != 3:
        raise ValueError(f"goal value must reduce to (B,S,D), got {value.shape}")
    if value.shape[:2] != predicted.shape[:2]:
        value = value.expand(predicted.shape[0], predicted.shape[1], last_dim)
    return value


def make_probe_cost_model(base: Any, probe: common.LoadedProbe, yaw_weight: float) -> Any:
    import torch

    class ProbeCostModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            # Register the trained module as part of this wrapper; normalization
            # tensors remain owned by LoadedProbe on the same device.
            self.probe_module = probe.model
            self.yaw_weight = float(yaw_weight)

        @torch.inference_mode()
        def get_cost(self, info_dict: dict[str, Any], action_candidates: Any) -> Any:
            if "goal_privileged_block_0_pos" not in info_dict:
                raise KeyError("online probe cost requires privileged goal block xyz")
            device = next(self.base.parameters()).device
            for key in list(info_dict):
                if torch.is_tensor(info_dict[key]):
                    info_dict[key] = info_dict[key].to(device)
            rolled = self.base.rollout(info_dict, action_candidates)
            terminal = rolled["predicted_emb"][..., -1, :]
            predicted = probe(terminal)
            goal_xyz = _goal_value(
                info_dict["goal_privileged_block_0_pos"], predicted, 3
            )
            goal_yaw = None
            if self.yaw_weight:
                if "goal_privileged_block_0_yaw" not in info_dict:
                    raise KeyError("nonzero yaw weight requires privileged goal yaw")
                goal_yaw_value = info_dict["goal_privileged_block_0_yaw"]
                goal_yaw = _goal_value(goal_yaw_value, predicted, 1)[..., 0]
            cost = common.probe_physical_cost(
                predicted, goal_xyz, goal_yaw, self.yaw_weight
            )
            if cost.shape != action_candidates.shape[:2]:
                raise RuntimeError(
                    f"probe cost shape mismatch: expected={action_candidates.shape[:2]}, actual={cost.shape}"
                )
            return cost

    return ProbeCostModel()


def _validate(args: argparse.Namespace) -> tuple[str, str, Path, dict[str, Any]]:
    common.validate_condition(args.condition)
    if args.num_eval not in (2, 50):
        raise ValueError("--num-eval is frozen to 2 or 50")
    if args.seed != 42 or args.goal_offset != 25 or args.eval_budget != 50:
        raise ValueError("online protocol is frozen to seed42/goal_offset25/budget50")
    if args.yaw_weight < 0:
        raise ValueError("--yaw-weight must be nonnegative")
    for path, label in (
        (args.dataset, "dataset"),
        (args.manifest, "manifest"),
        (args.probe, "probe"),
        (args.probe_dataset_metadata, "probe dataset metadata"),
    ):
        common.ensure_data_disk(path, label)
        if not path.is_file():
            raise FileNotFoundError(path)
    color, goal_type = common.condition_visual_protocol(args.condition)
    output = args.output or _default_output(args.condition, args.num_eval)
    # Validate every input and the offline decision before creating output or
    # importing CUDA/EGL state.
    gate = _load_gate(args.gate_report, args.probe, args.checkpoint, args.yaw_weight)
    return color, goal_type, output, gate


def run(args: argparse.Namespace) -> int:
    common.configure_storage()
    color, goal_type, raw_output, gate = _validate(args)

    import stable_worldmodel as swm
    import torch
    from sklearn.preprocessing import StandardScaler

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    recolor_goals = None
    recolor_metadata = None
    if goal_type == "recolor":
        recolor_goals, recolor_metadata = ood._load_recolor_goals(color, args.num_eval)

    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    all_rows = ood._formal_rows(dataset, args.manifest)
    rows = all_rows[: args.num_eval]
    action = dataset.get_col_data("action")
    action = action[~np.isnan(action).any(axis=1)]
    scaler = StandardScaler().fit(action)
    del action

    base_model = swm.wm.utils.load_pretrained(
        args.checkpoint, cache_dir=str(common.AILAB_ROOT)
    )
    base_model = base_model.to(args.device).eval().requires_grad_(False)
    base_model.interpolate_pos_encoding = True
    probe = common.LoadedProbe(args.probe, args.device)
    common.validate_checkpoint_dataset_link(probe, args.probe_dataset_metadata)
    actual_model_sha = common.torch_module_sha256(base_model)
    if actual_model_sha != probe.payload["world_model_state_sha256"]:
        raise ValueError(
            "online world model differs from probe embedding source: "
            f"expected={probe.payload['world_model_state_sha256']}, "
            f"actual={actual_model_sha}"
        )
    model = make_probe_cost_model(base_model, probe, args.yaw_weight).to(args.device).eval()

    output = _safe_output(raw_output, args.overwrite)
    recorder = ood.PlanningCostRecorder(args.num_eval)
    solver_cls = ood._make_selecting_solver(swm.solver.CEMSolver)
    solver = solver_cls(
        model=model,
        batch_size=1,
        num_samples=300,
        var_scale=1.0,
        n_steps=10,
        topk=30,
        device=args.device,
        seed=args.seed,
        callbacks=[recorder],
        selector="mean",
        recorder=recorder,
    )
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    policy_cls = ood._make_recording_policy(swm.policy.WorldModelPolicy)
    policy = policy_cls(
        solver=solver,
        config=config,
        process={"action": scaler},
        transform={"pixels": ood._image_transform(224), "goal": ood._image_transform(224)},
        recorder=recorder,
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
        metrics, selected = ood._evaluate(
            world,
            dataset,
            rows,
            goal_type,
            color,
            args.eval_budget,
            output / "videos" if args.video else None,
            recolor_goals,
        )
    finally:
        world.close()
    elapsed = time.time() - started
    histories = ood._save_cost_history(
        output,
        recorder,
        rows,
        selected["episodes"],
        selected["starts"],
        "mean",
    )
    protocol = {
        "condition": args.condition,
        "color": color,
        "goal_type": goal_type,
        "goal_recolor": recolor_metadata,
        "planner": "cem10",
        "selector": "updated_top30_elite_mean",
        "num_samples": 300,
        "cem_iterations": 10,
        "topk": 30,
        "horizon": 5,
        "action_block": 5,
        "seed": args.seed,
        "goal_offset": args.goal_offset,
        "eval_budget": args.eval_budget,
        "checkpoint": args.checkpoint,
        "world_model_state_sha256": actual_model_sha,
        "probe": probe.provenance(),
        "probe_supervision": list(common.TARGET_NAMES),
        "primary_cost": "squared Euclidean predicted block xyz to privileged goal xyz",
        "yaw_weight": args.yaw_weight,
        "offline_gate": gate,
        "offline_gate_top1_vs_online_selector_note": (
            "offline gate ranks stored candidates by probe-cost top1; online CEM "
            "uses the same probe cost to update elites and executes the updated mean"
        ),
    }
    payload = {
        "format_version": "cube_probe_online_eval_v1",
        "protocol": protocol,
        "formal_rows_verified": all_rows,
        "evaluated_rows": rows,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "cost_history": histories,
        "script": common.file_identity(Path(__file__)),
    }
    common.write_json(output / "results.json", payload)
    successes = ", ".join("True" if x else "False" for x in metrics["episode_successes"])
    (output / "results.txt").write_text(
        "==== CONFIG ====\n"
        + json.dumps(common.jsonable(protocol), indent=2, sort_keys=True)
        + "\n\n==== RESULTS ====\n"
        + f"success_rate: {metrics['success_rate']:.6f}\n"
        + f"success_count: {metrics['success_count']}/{metrics['num_eval']}\n"
        + f"episode_successes: [{successes}]\n"
        + f"evaluation_time: {elapsed:.6f} seconds\n",
        encoding="utf-8",
    )
    print(
        f"result: {metrics['success_count']}/{metrics['num_eval']} "
        f"({metrics['success_rate']:.2f}%), elapsed={elapsed:.2f}s"
    )
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gated online Cube probe-cost evaluation")
    parser.add_argument("--condition", required=True, choices=common.CONDITIONS)
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--probe", type=Path, default=common.PROBE_MODEL_DEFAULT / "mlp.pt")
    parser.add_argument("--probe-dataset-metadata", type=Path, default=common.PROBE_DATA_DEFAULT / "metadata.json")
    parser.add_argument("--gate-report", type=Path, default=common.OFFLINE_DEFAULT / "summary.json")
    parser.add_argument("--dataset", type=Path, default=common.DATASET_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST_DEFAULT)
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yaw-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
