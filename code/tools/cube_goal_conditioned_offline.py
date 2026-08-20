#!/usr/bin/env python3
"""Offline replay and imagination scoring for G1T1/G1T2 candidate pools.

The numerical replay and MaskedAug probe implementation are reused byte-for-
byte from ``cube_trust_region_offline.py``.  This adapter redirects artifacts
to the goal-conditioned tree and adds retrieval alignment plus the exact
same-color T1/T2 reference to the diagnostic payload.  The physical 40 mm
gate itself is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cube_goal_conditioned_common as common  # noqa: E402
import cube_trust_region_offline as trust_offline  # noqa: E402


REFERENCE_MEDIAN_MM = {
    "g1t1": {"red": 10.763, "blue_v2": 11.910, "yellow_v2": 10.692},
    "g1t2": {"red": 21.115, "blue_v2": 22.672, "yellow_v2": 23.143},
}


def _install_adapter() -> None:
    trust_offline.common = common


def _capture_alignment(evaluation_root: Path) -> dict:
    results = json.loads(
        (evaluation_root.resolve() / "results.json").read_text(encoding="utf-8")
    )
    alignment = results.get("goal_alignment") or results.get("trace", {}).get(
        "goal_alignment"
    )
    if not isinstance(alignment, dict) or "aggregate" not in alignment:
        raise ValueError(
            "goal-conditioned capture lacks frozen alignment statistics: "
            f"expected=trace.goal_alignment.aggregate, actual={alignment}"
        )
    return alignment


def _postprocess_replay(args: argparse.Namespace) -> None:
    capture = trust_offline._load_capture(args.evaluation_root)
    output = (
        args.output
        or common.physical_cache_root(capture["protocol"], capture["condition"])
    ).resolve()
    manifest_path = output / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["format_version"] = "cube_goal_conditioned_physical_cache_v1"
    payload.setdefault("helper_provenance", {})["this_adapter"] = (
        common.file_identity(Path(__file__))
    )
    common.write_json(manifest_path, payload)


def _postprocess_score(args: argparse.Namespace) -> None:
    capture = trust_offline._load_capture(args.evaluation_root)
    protocol = capture["protocol"]
    condition = capture["condition"]
    output = (
        args.output or common.imagination_output_root(protocol, condition)
    ).resolve()
    alignment = _capture_alignment(args.evaluation_root)
    gate_path = output / "gate.json"
    summary_path = output / "summary.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    observed = float(gate["criterion"]["observed_median_mm"])
    reference = float(REFERENCE_MEDIAN_MM[protocol][condition])
    diagnostic = {
        "baseline_protocol": common.base_protocol(protocol).upper(),
        "same_color_baseline_median_E_roll_mm": reference,
        "observed_median_E_roll_mm": observed,
        "difference_mm": observed - reference,
        "ratio": observed / reference,
        "is_gate_condition": False,
        "note": "exact prior Trust-Region cell; scale diagnostic only",
    }
    gate["format_version"] = "cube_goal_conditioned_gate_v1"
    gate["goal_alignment"] = alignment["aggregate"]
    gate["same_color_trust_region_reference"] = diagnostic
    # Preserve the sole hard fail-stop criterion from Trust-Region: the full
    # fixed12x300 Masked E_roll median must remain <=40 mm.
    common.write_json(gate_path, gate)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["format_version"] = "cube_goal_conditioned_imagination_error_v1"
    summary["goal_alignment"] = alignment["aggregate"]
    summary["same_color_trust_region_reference"] = diagnostic
    summary["gate"] = gate
    summary.setdefault("helper_provenance", {})["this_adapter"] = (
        common.file_identity(Path(__file__))
    )
    common.write_json(summary_path, summary)


def main(argv: Sequence[str] | None = None) -> int:
    _install_adapter()
    parser = trust_offline.build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        return trust_offline.command_validate(args)
    if args.command == "replay":
        result = trust_offline.command_replay(args)
        _postprocess_replay(args)
        return result
    if args.command == "score":
        if args.rollout_batch_size <= 0 or args.encoder_batch_size <= 0:
            raise ValueError("score batch sizes must be positive")
        result = trust_offline.command_score(args)
        _postprocess_score(args)
        return result
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
