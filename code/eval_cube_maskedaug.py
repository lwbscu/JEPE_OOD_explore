#!/usr/bin/env python3
"""Evaluate Route2.1 Cube masked-augmentation checkpoints on fixed50 colors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


AILAB = Path(__file__).resolve().parent.parent
EVALUATOR = Path(__file__).resolve().parent / "eval_ood_color.py"
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-maskedaug"
OUTPUT_ROOT = AILAB / "outputs/eval/cube/maskedaug"
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
            "--derived-maskedaug",
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
        print(
            json.dumps(
                {
                    "ordered_protocols": [item[0] for item in commands],
                    "policy": {"planner": "cem10", "selector": "mean"},
                    "commands": [item[1] for item in commands],
                },
                indent=2,
            )
        )
        return 0

    summary_path = OUTPUT_ROOT / (
        "evaluation_summary.json" if args.num_eval == 50 else "smoke/evaluation_summary.json"
    )
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"summary exists; pass --overwrite intentionally: {summary_path}")
    results: dict[str, dict[str, Any]] = {}
    execution_failed = False
    for protocol, command, output in commands:
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
    summary = {
        "protocol_order": [item[0] for item in PROTOCOLS],
        "policy": {"planner": "cem10", "selector": "mean"},
        "goal_protocol": {
            "red": "unchanged dataset",
            "blue_v2": "blue recolor v2",
            "yellow_v2": "yellow recolor v2",
        },
        "num_eval": args.num_eval,
        "results": results,
        "all_completed": all(
            results.get(protocol, {}).get("status") == "completed"
            for protocol, *_ in PROTOCOLS
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if execution_failed else 0


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
