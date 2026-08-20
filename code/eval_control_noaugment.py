#!/usr/bin/env python3
"""Evaluate the frozen no-augmentation control under the Cube T2 protocol.

Only the two formal control exports are accepted.  The evaluator uses the
same frozen rows, seed-42 T2 seed/noise injection, image goal contracts, and
50-env ordering as ``eval_cube_robust.py``; it intentionally owns no CEM
implementation of its own.

Examples::

    python le-wm/eval_control_noaugment.py --self-test
    python le-wm/eval_control_noaugment.py --checkpoint \
      checkpoints/lewm-cube-control_noaugment/control_noaugment_seed3072/weights_step_12732.pt \
      --condition red --num-eval 2
    python le-wm/eval_control_noaugment.py --checkpoint \
      checkpoints/lewm-cube-control_noaugment/control_noaugment_seed3072/weights_final.pt \
      --condition all --num-eval 50 --authorize-formal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TOOLS = HERE / "tools"
for _path in (HERE, TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_cube_robust as robust  # noqa: E402
import eval_memory_seed as memory_legacy  # noqa: E402
import eval_ood_color as legacy  # noqa: E402
import cube_trust_region_common as t2common  # noqa: E402


RUN_ID = "control_noaugment_seed3072"
CHECKPOINT_DIR = PROJECT / "checkpoints/lewm-cube-control_noaugment" / RUN_ID
TRAIN_DIR = PROJECT / "outputs/train/control_noaugment" / RUN_ID
OUTPUT_ROOT = PROJECT / "outputs/eval/cube/control_noaugment"
DATASET = PROJECT / "datasets/ogbench/cube_single_expert.h5"
MANIFEST = PROJECT / "outputs/audit/cube_cem_manifest.json"
FORMAL_SEED = 42
GOAL_OFFSET = 25
EVAL_BUDGET = 50

CHECKPOINT_SPECS = {
    "weights_step_12732.pt": {
        "label": "step_12732",
        "phase": "phase_a",
        "phase_steps": 12_732,
        "cumulative_step_end": 12_732,
    },
    "weights_final.pt": {
        "label": "step_16732",
        "phase": "phase_b",
        "phase_steps": 4_000,
        "cumulative_step_end": 16_732,
    },
}
CONDITIONS = ("red", "blue_v2", "yellow_v2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _checkpoint_contract(path: Path) -> tuple[Path, dict[str, Any]]:
    """Require the exact formal export and its adjacent training identity."""

    resolved = path.expanduser().resolve()
    expected_dir = CHECKPOINT_DIR.resolve()
    if resolved.parent != expected_dir or resolved.name not in CHECKPOINT_SPECS:
        raise ValueError(
            "checkpoint must be one canonical control export: "
            + ", ".join(str(expected_dir / name) for name in CHECKPOINT_SPECS)
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"control checkpoint missing: {resolved}")
    config = expected_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"adjacent control config missing: {config}")

    spec = CHECKPOINT_SPECS[resolved.name]
    master = _load_json(TRAIN_DIR / "run_plan.json", "control master run plan")
    if master.get("experiment") != "control_noaugment" or master.get("run_id") != RUN_ID:
        raise ValueError(
            "control master run identity mismatch: "
            f"expected experiment=control_noaugment/run_id={RUN_ID}, "
            f"actual experiment={master.get('experiment')}/run_id={master.get('run_id')}"
        )
    phases = {item.get("name"): item for item in master.get("phases", []) if isinstance(item, dict)}
    phase_plan = phases.get(spec["phase"])
    if phase_plan is None:
        raise ValueError(f"control master plan lacks expected {spec['phase']} entry")
    expected_export = str(resolved)
    if (
        int(phase_plan.get("steps", -1)) != spec["phase_steps"]
        or int(phase_plan.get("cumulative_step_end", -1)) != spec["cumulative_step_end"]
        or str(Path(phase_plan.get("export", "")).resolve()) != expected_export
    ):
        raise ValueError(
            f"control master {spec['phase']} contract mismatch: "
            f"expected steps={spec['phase_steps']}, cumulative={spec['cumulative_step_end']}, export={expected_export}; "
            f"actual={phase_plan}"
        )
    completed = _load_json(TRAIN_DIR / spec["phase"] / "completed.json", f"{spec['phase']} completion")
    actual_export = Path(completed.get("exported_weights", "")).resolve()
    actual_sha = str(completed.get("exported_weights_sha256", ""))
    observed_sha = _sha256(resolved)
    expected_completion = {
        "phase": spec["phase"],
        "phase_steps": spec["phase_steps"],
        "cumulative_step_end": spec["cumulative_step_end"],
        "export": str(resolved),
        "sha256": observed_sha,
    }
    actual_completion = {
        "phase": completed.get("phase"),
        "phase_steps": completed.get("phase_steps"),
        "cumulative_step_end": completed.get("cumulative_step_end"),
        "export": str(actual_export),
        "sha256": actual_sha,
    }
    if actual_completion != expected_completion:
        raise ValueError(
            "control completion contract mismatch: "
            f"expected={expected_completion}, actual={actual_completion}"
        )
    if int(completed.get("global_step", -1)) != spec["phase_steps"]:
        raise ValueError(
            f"control completion global step mismatch: expected={spec['phase_steps']}, "
            f"actual={completed.get('global_step')}"
        )
    overall = _load_json(TRAIN_DIR / "completed.json", "control overall completion")
    expected_exports = [
        str((CHECKPOINT_DIR / "weights_step_12732.pt").resolve()),
        str((CHECKPOINT_DIR / "weights_final.pt").resolve()),
    ]
    actual_overall = {
        "run_id": overall.get("run_id"),
        "complete": overall.get("complete"),
        "formal_cumulative_steps": overall.get("formal_cumulative_steps"),
        "exports": overall.get("exports"),
    }
    expected_overall = {
        "run_id": RUN_ID,
        "complete": True,
        "formal_cumulative_steps": 16_732,
        "exports": expected_exports,
    }
    if actual_overall != expected_overall:
        raise ValueError(
            "control overall completion contract mismatch: "
            f"expected={expected_overall}, actual={actual_overall}"
        )
    if spec["phase"] == "phase_b":
        phase_a = _load_json(TRAIN_DIR / "phase_a" / "completed.json", "phase_a completion")
        phase_a_path = CHECKPOINT_DIR / "weights_step_12732.pt"
        if (
            Path(phase_a.get("exported_weights", "")).resolve() != phase_a_path.resolve()
            or str(phase_a.get("exported_weights_sha256", "")) != _sha256(phase_a_path)
        ):
            raise ValueError("phase_b identity does not bind the verified Phase A control export")
    return resolved, {
        "label": spec["label"],
        "phase": spec["phase"],
        "weights": {"path": str(resolved), "sha256": observed_sha, "size": resolved.stat().st_size},
        "config": {"path": str(config.resolve()), "sha256": _sha256(config), "size": config.stat().st_size},
        "master_run_plan": {"path": str((TRAIN_DIR / "run_plan.json").resolve()), "sha256": _sha256(TRAIN_DIR / "run_plan.json")},
        "phase_completion": {"path": str((TRAIN_DIR / spec["phase"] / "completed.json").resolve()), "sha256": _sha256(TRAIN_DIR / spec["phase"] / "completed.json")},
    }


def _prepare_output(path: Path | None, label: str, condition: str, num_eval: int, overwrite: bool) -> Path:
    default = OUTPUT_ROOT / (
        Path(label) / condition
        if num_eval == 50
        else Path("smoke") / label / condition
    )
    target = (path or default).expanduser().resolve()
    root = OUTPUT_ROOT.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"output must be a concrete child of {root}: {target}")
    if num_eval == 50 and target != default.resolve():
        raise ValueError(
            f"formal output is frozen: expected={default.resolve()}, actual={target}"
        )
    if target.exists() and target.is_symlink():
        raise ValueError(f"refusing symlink output: {target}")
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"non-empty output: {target}; pass --overwrite")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _build_world_and_policy(args: argparse.Namespace, dataset: Any, rows: np.ndarray) -> tuple[Any, Any]:
    """Reuse the robust evaluator's local T2 implementation unchanged."""

    return robust._build_world_and_policy(args, dataset, rows)


def run(args: argparse.Namespace) -> int:
    robust._configure_storage()
    if args.self_test:
        assert tuple(CONDITIONS) == ("red", "blue_v2", "yellow_v2")
        assert {item["label"] for item in CHECKPOINT_SPECS.values()} == {"step_12732", "step_16732"}
        assert t2common.NUM_SAMPLES == 300 and t2common.N_STEPS == 10
        print(json.dumps({"status": "PASS", "protocol": "T2", "seed": FORMAL_SEED, "checkpoints": CHECKPOINT_SPECS}, indent=2))
        return 0
    if args.seed != FORMAL_SEED or args.goal_offset != GOAL_OFFSET or args.eval_budget != EVAL_BUDGET:
        raise ValueError("control protocol is frozen to seed42/goal_offset25/budget50")
    if args.num_eval not in (2, 50):
        raise ValueError("num-eval is frozen to 2 smoke or 50 formal")
    if args.num_eval == 50 and not args.authorize_formal:
        raise PermissionError("pass --authorize-formal for 50-env control evaluation")
    if not args.dataset.is_file() or not args.manifest.is_file() or not args.index.is_dir():
        raise FileNotFoundError("control dataset, audit manifest, or memory index missing")
    checkpoint, provenance = _checkpoint_contract(args.checkpoint)
    if args.condition == "all":
        conditions = list(CONDITIONS)
    else:
        conditions = [args.condition]

    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("control T2 evaluation requires CUDA")
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    formal_rows = legacy._formal_rows(dataset, args.manifest)
    rows = formal_rows[: args.num_eval]
    recolor = {}
    for color in ("blue", "yellow"):
        recolor[color], _ = legacy._load_recolor_goals(color, args.num_eval)
    summary: dict[str, Any] = {
        "format_version": "cube_control_noaugment_eval_v1",
        "checkpoint": provenance,
        "protocol": {
            "id": "T2",
            "seed": FORMAL_SEED,
            "goal_offset": GOAL_OFFSET,
            "eval_budget": EVAL_BUDGET,
            "num_eval": args.num_eval,
            "num_samples": t2common.NUM_SAMPLES,
            "n_steps": t2common.N_STEPS,
            "memory_slots": t2common.MEMORY_SLOTS,
            "noisy_slots": t2common.NOISY_SLOTS,
            "free_slots": t2common.NUM_SAMPLES - t2common.MEMORY_SLOTS - t2common.NOISY_SLOTS,
            "formal_rows": formal_rows,
            "fixed_50_paired": True,
        },
        "conditions": {},
    }
    for condition in conditions:
        spec = robust.CONDITIONS[condition]
        output = _prepare_output(args.output if len(conditions) == 1 else None, provenance["label"], condition, args.num_eval, args.overwrite)
        world, recorder = _build_world_and_policy(args, dataset, rows)
        try:
            result, elapsed = robust._evaluate_condition(
                world=world,
                dataset=dataset,
                rows=rows,
                spec=spec,
                budget=args.eval_budget,
                output=output,
                video=args.video,
                recolor_goals=recolor.get(spec["color"]),
            )
        finally:
            try:
                world.close()
            except Exception:
                pass
        cost = legacy._save_cost_history(output, recorder, rows, result["episodes"], result["starts"], "mean")
        payload = {
            "format_version": "cube_control_noaugment_condition_v1",
            "condition": condition,
            "checkpoint": provenance,
            "protocol": summary["protocol"],
            "evaluated_rows": rows,
            "metrics": result["metrics"],
            "elapsed_seconds": elapsed,
            "cost_history": cost,
        }
        _write_json(output / "results.json", payload)
        (output / "results.txt").write_text(
            f"condition: {condition}\ncheckpoint: {provenance['label']}\n"
            f"success_rate: {result['metrics']['success_rate']:.6f}\n"
            f"success_count: {result['metrics']['success_count']}/{args.num_eval}\n"
            f"evaluation_time: {elapsed:.6f} seconds\n",
            encoding="utf-8",
        )
        summary["conditions"][condition] = {
            "success_rate": result["metrics"]["success_rate"],
            "success_count": result["metrics"]["success_count"],
            "results": str((output / "results.json").resolve()),
        }
    summary_path = OUTPUT_ROOT / (f"evaluation_summary_{provenance['label']}.json" if args.num_eval == 50 else f"smoke/evaluation_summary_{provenance['label']}.json")
    _write_json(summary_path, summary)
    print(json.dumps(_jsonable(summary), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="one canonical formal control checkpoint")
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="red")
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
