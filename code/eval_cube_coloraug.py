#!/usr/bin/env python3
"""Evaluate a derived Cube color-aug checkpoint with legacy CEM mean.

Red is always executed first. Its formal 50-row result is a quality gate at
69%; blue-v2 and yellow-v2 still run if the gate fails, but the emitted summary
forbids promotion into a later combination experiment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


AILAB = Path(__file__).resolve().parent.parent
EVALUATOR = Path(__file__).resolve().parent / "eval_ood_color.py"
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-coloraug"
OUTPUT_ROOT = AILAB / "outputs/eval/cube/coloraug"
QUALITY_GATE_PERCENT = 69.0
PROTOCOLS = (
    ("red", "red", "matched", "dataset"),
    ("blue_v2", "blue", "recolor", "auto"),
    ("yellow_v2", "yellow", "recolor", "auto"),
)


def _checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    root = CHECKPOINT_ROOT.resolve()
    if not path.is_file() or path.suffix != ".pt" or root not in path.parents:
        raise ValueError(f"checkpoint must be a .pt file under {root}: {path}")
    if not (path.parent / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config.json missing: {path.parent}")
    return path


def _output(protocol: str, num_eval: int) -> Path:
    return OUTPUT_ROOT / protocol if num_eval == 50 else OUTPUT_ROOT / "smoke" / protocol


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    checkpoint = _checkpoint(args.checkpoint)
    commands = []
    for protocol, color, goal_type, input_mode in PROTOCOLS:
        output = _output(protocol, args.num_eval)
        command = [
            sys.executable,
            str(EVALUATOR),
            "--derived-coloraug",
            "--checkpoint", str(checkpoint),
            "--color", color,
            "--goal-type", goal_type,
            "--input-mode", input_mode,
            "--planner", "cem10",
            "--selector", "mean",
            "--num-eval", str(args.num_eval),
            "--output", str(output),
            "--video" if args.video else "--no-video",
        ]
        if args.overwrite:
            command.append("--overwrite")
        commands.append((protocol, command, output))
    return commands


def _read_result(path: Path, expected_num_eval: int) -> dict[str, Any]:
    payload = json.loads((path / "results.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    if int(metrics["num_eval"]) != expected_num_eval:
        raise RuntimeError(
            f"result count mismatch: expected={expected_num_eval}, actual={metrics['num_eval']}"
        )
    success_count = int(metrics["success_count"])
    success_rate = float(metrics["success_rate"])
    if abs(success_rate - success_count / expected_num_eval * 100.0) > 1e-9:
        raise RuntimeError("success count/rate mismatch")
    return {
        "status": "completed",
        "success_count": success_count,
        "num_eval": expected_num_eval,
        "success_rate": success_rate,
        "results_json": str((path / "results.json").resolve()),
    }


def run(args: argparse.Namespace) -> int:
    commands = build_commands(args)
    if args.dry_run:
        print(json.dumps({"ordered_protocols": [item[0] for item in commands], "commands": [item[1] for item in commands]}, indent=2))
        return 0

    summary_path = OUTPUT_ROOT / ("evaluation_summary.json" if args.num_eval == 50 else "smoke/evaluation_summary.json")
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"summary exists; pass --overwrite intentionally: {summary_path}")
    results: dict[str, dict[str, Any]] = {}
    execution_failed = False
    for protocol, command, output in commands:  # tuple order freezes red first.
        completed = subprocess.run(command, cwd=str(EVALUATOR.parent), check=False, shell=False)
        if completed.returncode != 0:
            execution_failed = True
            results[protocol] = {"status": "execution_failed", "returncode": completed.returncode}
            continue
        try:
            results[protocol] = _read_result(output, args.num_eval)
        except Exception as exc:
            execution_failed = True
            results[protocol] = {"status": "invalid_result", "error": f"{type(exc).__name__}: {exc}"}

    red = results.get("red", {})
    formal_gate_evaluated = args.num_eval == 50 and red.get("status") == "completed"
    red_gate_passed = bool(
        formal_gate_evaluated and red["success_rate"] >= QUALITY_GATE_PERCENT
    )
    all_completed = all(results.get(protocol, {}).get("status") == "completed" for protocol, *_ in PROTOCOLS)
    promotion_allowed = bool(formal_gate_evaluated and red_gate_passed and all_completed)
    summary = {
        "protocol_order": [item[0] for item in PROTOCOLS],
        "legacy_policy": {"planner": "cem10", "selector": "mean"},
        "num_eval": args.num_eval,
        "results": results,
        "red_quality_gate": {
            "threshold_percent": QUALITY_GATE_PERCENT,
            "comparison": "success_rate >= 69.0",
            "evaluated": formal_gate_evaluated,
            "passed": red_gate_passed,
        },
        "combination_promotion_allowed": promotion_allowed,
        "promotion_rule": "formal red gate passes and all three evaluations complete",
        "blue_yellow_run_even_if_red_gate_failed": True,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if execution_failed:
        return 1
    if args.num_eval == 50 and not red_gate_passed:
        return 2
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    p.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
