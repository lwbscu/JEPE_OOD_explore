#!/usr/bin/env python3
"""Validate, physically replay, and score new Trust-Region first-cycle pools.

Candidate indices in T1/T2 are new and must never be joined to legacy audit
outcomes.  ``score`` therefore requires a physical cache whose manifest hashes
the exact captured population.  If that cache is absent, scoring fails with an
explicit terminal-truth requirement.  ``replay`` is isolated and opt-in because
the fixed 12x300 MuJoCo branch evaluation is materially expensive.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
LEWM_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cube_cem_audit as audit  # noqa: E402
import cube_imagination_error_common as imag_common  # noqa: E402
import cube_trust_region_common as common  # noqa: E402
import run_cube_imagination_error as imagination  # noqa: E402


SCORE_FIELDS = (
    "protocol",
    "condition",
    "model",
    "env_idx",
    "dataset_row",
    "candidate_idx",
    "E_roll_mm",
    "E_enc_mm",
    "Delta_roll_minus_enc_mm",
    "E_imag_mm",
    "latent_l2",
    "latent_cosine_distance",
    "final_success",
    "ever_success",
    "terminal_x_m",
    "terminal_y_m",
    "terminal_z_m",
    "roll_x_m",
    "roll_y_m",
    "roll_z_m",
    "enc_x_m",
    "enc_y_m",
    "enc_z_m",
    "latent_cost",
)


def _load_capture(evaluation_root: Path) -> dict[str, Any]:
    evaluation_root = common.ensure_child(
        evaluation_root, common.OUTPUT_ROOT, "Trust-Region evaluation root"
    )
    results_path = evaluation_root / "results.json"
    manifest_path = evaluation_root / "first_cycle_pools/manifest.json"
    if not results_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"evaluation capture is incomplete: {results_path}, {manifest_path}"
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = results["protocol"]["id"]
    condition = results["protocol"]["condition"]
    if protocol not in common.PROTOCOLS or condition not in common.CONDITIONS:
        raise ValueError(f"invalid capture protocol: {protocol}/{condition}")
    if manifest.get("protocol") != protocol or manifest.get("condition") != condition:
        raise ValueError("capture manifest/results protocol mismatch")
    expected_envs = [
        env for env in common.AUDIT_ENVS if env < int(results["metrics"]["num_eval"])
    ]
    if manifest.get("saved_envs") != expected_envs:
        raise ValueError(
            f"captured env mismatch: expected={expected_envs}, actual={manifest.get('saved_envs')}"
        )
    cases = []
    for item in manifest["cases"]:
        case = evaluation_root / "first_cycle_pools" / common.case_name(
            int(item["env_idx"]), int(item["dataset_row"])
        )
        population = case / "population.npz"
        metadata = json.loads((case / "capture_meta.json").read_text(encoding="utf-8"))
        actual_sha = common.sha256_file(population)
        if actual_sha != metadata["population"]["sha256"]:
            raise ValueError(
                f"captured population hash mismatch: {population}; "
                f"expected={metadata['population']['sha256']}, actual={actual_sha}"
            )
        with np.load(population, allow_pickle=False) as data:
            shapes = {
                "candidates_normalized": data["candidates_normalized"].shape,
                "latent_costs": data["latent_costs"].shape,
                "initial_pixels": data["initial_pixels"].shape,
            }
        expected_shapes = {
            "candidates_normalized": (300, 5, 25),
            "latent_costs": (300,),
            "initial_pixels": (224, 224, 3),
        }
        if shapes != expected_shapes:
            raise ValueError(
                f"captured population shape mismatch: {population}; "
                f"expected={expected_shapes}, actual={shapes}"
            )
        cases.append(
            {
                "env_idx": int(item["env_idx"]),
                "dataset_row": int(item["dataset_row"]),
                "case": case,
                "population": population,
                "population_sha256": actual_sha,
            }
        )
    return {
        "root": evaluation_root,
        "results": results,
        "results_path": results_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "protocol": protocol,
        "condition": condition,
        "cases": cases,
    }


def _load_physical_cache(
    capture: Mapping[str, Any], physical_root: Path
) -> dict[int, dict[str, Any]]:
    physical_root = common.ensure_child(
        physical_root, common.OUTPUT_ROOT, "Trust-Region physical cache"
    )
    manifest_path = physical_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "new Trust-Region candidates have no terminal truth. Missing physical "
            f"cache manifest: {manifest_path}. Do not join legacy audit labels by "
            "candidate_idx; run the independently authorized 12x300 replay first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != capture["protocol"]
        or manifest.get("condition") != capture["condition"]
        or manifest.get("source_results_sha256")
        != common.sha256_file(capture["results_path"])
    ):
        raise ValueError("physical cache does not match the captured evaluation")
    outputs: dict[int, dict[str, Any]] = {}
    by_env = {int(item["env_idx"]): item for item in manifest["cases"]}
    for source in capture["cases"]:
        env_idx = source["env_idx"]
        if env_idx not in by_env:
            raise ValueError(f"physical cache missing captured env={env_idx}")
        item = by_env[env_idx]
        if item["source_population_sha256"] != source["population_sha256"]:
            raise ValueError(
                f"physical cache/pool hash mismatch: env={env_idx}; "
                f"expected={source['population_sha256']}, actual={item['source_population_sha256']}"
            )
        path = common.ensure_child(
            physical_root / item["relative_npz"],
            physical_root,
            f"Trust-Region physical outcome env={env_idx}",
        )
        actual_sha = common.sha256_file(path)
        if actual_sha != item["physical_npz_sha256"]:
            raise ValueError(
                f"physical cache file hash mismatch: {path}; "
                f"expected={item['physical_npz_sha256']}, actual={actual_sha}"
            )
        with np.load(path, allow_pickle=False) as physical:
            shapes = {
                "terminal_images": physical["terminal_images"].shape,
                "terminal_cube_position": physical["terminal_cube_position"].shape,
                "final_success": physical["final_success"].shape,
                "ever_success": physical["ever_success"].shape,
            }
        expected_shapes = {
            "terminal_images": (300, 224, 224, 3),
            "terminal_cube_position": (300, 3),
            "final_success": (300,),
            "ever_success": (300,),
        }
        if shapes != expected_shapes:
            raise ValueError(
                f"physical cache shape mismatch: {path}; expected={expected_shapes}, actual={shapes}"
            )
        outputs[env_idx] = {"path": path, "item": item}
    return outputs


def _require_formal_fixed12(capture: Mapping[str, Any], operation: str) -> None:
    """Reject smoke/partial captures before physical replay or scoring."""

    manifest = capture["manifest"]
    results = capture["results"]
    actual = {
        "mode": results.get("mode"),
        "num_eval": int(results["metrics"]["num_eval"]),
        "formal_full_50_order": bool(
            results["protocol"].get("formal_full_50_order", False)
        ),
        "full_formal_order_then_select_12": bool(
            manifest.get("full_formal_order_then_select_12", False)
        ),
        "saved_envs": manifest.get("saved_envs"),
    }
    expected = {
        "mode": "capture_only_no_env_step_no_video",
        "num_eval": 50,
        "formal_full_50_order": True,
        "full_formal_order_then_select_12": True,
        "saved_envs": list(common.AUDIT_ENVS),
    }
    if actual != expected:
        raise ValueError(
            f"{operation} requires a full 50-env solve in formal order followed "
            f"by the fixed 12 capture; expected={expected}, actual={actual}"
        )


def command_validate(args: argparse.Namespace) -> int:
    capture = _load_capture(args.evaluation_root)
    payload = {
        "protocol": capture["protocol"],
        "condition": capture["condition"],
        "num_cases": len(capture["cases"]),
        "saved_envs": [case["env_idx"] for case in capture["cases"]],
        "full_formal_order_then_select_12": capture["manifest"][
            "full_formal_order_then_select_12"
        ],
        "physical_truth_available": False,
        "imagination_scoring_available": False,
    }
    physical_root = args.physical_root or common.physical_cache_root(
        capture["protocol"], capture["condition"]
    )
    try:
        _load_physical_cache(capture, physical_root)
    except FileNotFoundError as error:
        payload["blocked_reason"] = str(error)
    else:
        payload["physical_truth_available"] = True
        payload["imagination_scoring_available"] = True
    print(json.dumps(common.jsonable(payload), indent=2, sort_keys=True))
    return 0


def command_replay(args: argparse.Namespace) -> int:
    common.configure_storage()
    if not args.authorize_physical_replay:
        raise PermissionError(
            "12x300 MuJoCo replay is not authorized. Re-run only after Leader "
            "approval with --authorize-physical-replay."
        )
    capture = _load_capture(args.evaluation_root)
    _require_formal_fixed12(capture, "physical replay")
    output = common.prepare_output(
        args.output
        or common.physical_cache_root(capture["protocol"], capture["condition"]),
        common.OUTPUT_ROOT,
        args.overwrite,
    )
    _, _, cube_color = common.condition_visual(capture["condition"])
    started = time.time()
    env = audit._make_replay_env()
    manifest_cases = []
    try:
        for source in capture["cases"]:
            with np.load(source["population"], allow_pickle=False) as loaded:
                data = {key: np.asarray(loaded[key]) for key in loaded.files}
            candidates = np.asarray(data["candidates_normalized"], dtype=np.float32)
            latent_costs = np.asarray(data["latent_costs"], dtype=np.float32)
            snapshot = audit._setup_case_env(env, data, cube_color)
            terminal_images = []
            rows = []
            for candidate_idx in range(common.NUM_SAMPLES):
                actions = audit._inverse_scale(
                    candidates[candidate_idx],
                    data["action_scaler_mean"],
                    data["action_scaler_scale"],
                )
                outcome, terminal, _ = audit._branch_rollout(
                    env,
                    snapshot,
                    actions,
                    np.asarray(data["goal_position"]),
                    collect_frames=False,
                    stop_on_success=False,
                )
                terminal_images.append(np.asarray(terminal, dtype=np.uint8))
                rows.append(
                    {
                        "candidate_idx": candidate_idx,
                        "latent_cost": float(latent_costs[candidate_idx]),
                        **outcome,
                    }
                )
            case_output = output / common.case_name(
                source["env_idx"], source["dataset_row"]
            )
            case_output.mkdir()
            physical_path = case_output / "physical_outcomes.npz"
            np.savez_compressed(
                physical_path,
                terminal_images=np.stack(terminal_images),
                terminal_cube_position=np.asarray(
                    [
                        [row["terminal_cube_x"], row["terminal_cube_y"], row["terminal_cube_z"]]
                        for row in rows
                    ],
                    dtype=np.float64,
                ),
                final_success=np.asarray([row["final_success"] for row in rows], dtype=bool),
                ever_success=np.asarray([row["ever_success"] for row in rows], dtype=bool),
                min_goal_distance_m=np.asarray([row["min_goal_distance_m"] for row in rows]),
                final_goal_distance_m=np.asarray([row["final_goal_distance_m"] for row in rows]),
            )
            common.write_csv(case_output / "candidate_outcomes.csv", rows, tuple(rows[0]))
            item = {
                "env_idx": source["env_idx"],
                "dataset_row": source["dataset_row"],
                "source_population_sha256": source["population_sha256"],
                "relative_npz": str(physical_path.relative_to(output)),
                "physical_npz_sha256": common.sha256_file(physical_path),
                "num_candidates": len(rows),
                "candidate_indices": "0..299 in the exact new captured pool",
                "legacy_audit_labels_used": False,
            }
            common.write_json(case_output / "replay_meta.json", item)
            manifest_cases.append(item)
    finally:
        env.close()
    manifest = {
        "format_version": "cube_trust_region_physical_cache_v1",
        "protocol": capture["protocol"],
        "condition": capture["condition"],
        "source_evaluation": str(capture["root"]),
        "source_results_sha256": common.sha256_file(capture["results_path"]),
        "source_pool_manifest_sha256": common.sha256_file(capture["manifest_path"]),
        "num_cases": len(manifest_cases),
        "num_candidates_per_case": common.NUM_SAMPLES,
        "physics": "MuJoCo 25-step raw-action replay; terminate_at_goal=False",
        "legacy_audit_labels_used": False,
        "elapsed_seconds": time.time() - started,
        "helper_provenance": {
            "cube_cem_audit": common.file_identity(Path(audit.__file__)),
            "this_tool": common.file_identity(Path(__file__)),
        },
        "cases": manifest_cases,
    }
    common.write_json(output / "manifest.json", manifest)
    print(output)
    return 0


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "E_roll_mm",
        "E_enc_mm",
        "Delta_roll_minus_enc_mm",
        "E_imag_mm",
        "latent_l2",
        "latent_cosine_distance",
    )
    result = {}
    for model in sorted({row["model"] for row in rows}):
        population = [row for row in rows if row["model"] == model]
        result[model] = {}
        for name, predicate in (
            ("all", lambda row: True),
            ("final_success", lambda row: row["final_success"]),
            ("final_failure", lambda row: not row["final_success"]),
            ("ever_success", lambda row: row["ever_success"]),
            ("ever_failure", lambda row: not row["ever_success"]),
        ):
            subset = [row for row in population if predicate(row)]
            result[model][name] = {
                "count": len(subset),
                **{
                    metric: common.distribution(
                        np.asarray([row[metric] for row in subset], dtype=np.float64)
                    )
                    for metric in metrics
                },
            }
    return result


def _encoder_floor_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    e_roll = np.asarray([row["E_roll_mm"] for row in rows], dtype=np.float64)
    e_enc = np.asarray([row["E_enc_mm"] for row in rows], dtype=np.float64)
    e_imag = np.asarray([row["E_imag_mm"] for row in rows], dtype=np.float64)
    encoder_floor_ok = e_enc <= 40.0
    return {
        "E_enc_mm": common.distribution(e_enc),
        "P_E_enc_gt_40mm": float(np.mean(e_enc > 40.0)),
        "P_E_roll_gt_40mm_and_E_enc_le_40mm": float(
            np.mean((e_roll > 40.0) & encoder_floor_ok)
        ),
        "E_enc_le_40mm_subset": {
            "count": int(np.count_nonzero(encoder_floor_ok)),
            "E_roll_mm": common.distribution(e_roll[encoder_floor_ok]),
            "E_imag_mm": common.distribution(e_imag[encoder_floor_ok]),
        },
        "encoder_floor_not_gate_condition": True,
    }


def command_score(args: argparse.Namespace) -> int:
    common.configure_storage()
    frozen_checkpoint = common.frozen_masked_checkpoint_contract()
    capture = _load_capture(args.evaluation_root)
    physical_root = args.physical_root or common.physical_cache_root(
        capture["protocol"], capture["condition"]
    )
    physical = _load_physical_cache(capture, physical_root)
    _require_formal_fixed12(capture, "imagination scoring")
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    labels = ["masked"]
    output_candidate = args.output or common.imagination_output_root(
        capture["protocol"], capture["condition"]
    )
    output_candidate = common.ensure_child(
        output_candidate, common.OUTPUT_ROOT, "Trust-Region imagination output"
    )
    if output_candidate.exists() and not output_candidate.is_dir():
        raise ValueError(f"output exists but is not a directory: {output_candidate}")
    if output_candidate.exists() and any(output_candidate.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is nonempty: {output_candidate}")
    rows: list[dict[str, Any]] = []
    model_provenance = {}
    for label in labels:
        model, probe, model_provenance[label] = imagination._load_bundle(
            label, args.device
        )
        for source in capture["cases"]:
            with np.load(source["population"], allow_pickle=False) as population:
                candidates = np.asarray(population["candidates_normalized"], dtype=np.float32)
                initial = np.asarray(population["initial_pixels"], dtype=np.uint8)
                latent_costs = np.asarray(population["latent_costs"], dtype=np.float64)
            with np.load(physical[source["env_idx"]]["path"], allow_pickle=False) as truth:
                terminal_images = np.asarray(truth["terminal_images"], dtype=np.uint8)
                terminal_xyz = np.asarray(truth["terminal_cube_position"], dtype=np.float64)
                final_success = np.asarray(truth["final_success"], dtype=bool)
                ever_success = np.asarray(truth["ever_success"], dtype=bool)
            predicted_latent = imagination.route1.exact_candidate_terminal_embeddings(
                model,
                initial,
                candidates,
                args.device,
                batch_size=args.rollout_batch_size,
            )
            actual_latent = imag_common.encode_uint8(
                model,
                terminal_images,
                args.device,
                args.encoder_batch_size,
            )
            with torch.inference_mode():
                roll_xyz = probe(predicted_latent).detach().cpu().numpy()
                enc_xyz = probe(actual_latent).detach().cpu().numpy()
            e_roll = imag_common.xyz_error_mm(roll_xyz, terminal_xyz)
            e_enc = imag_common.xyz_error_mm(enc_xyz, terminal_xyz)
            e_imag = imag_common.xyz_error_mm(roll_xyz, enc_xyz)
            drift = imag_common.latent_metrics(predicted_latent, actual_latent)
            for candidate_idx in range(common.NUM_SAMPLES):
                rows.append(
                    {
                        "protocol": capture["protocol"],
                        "condition": capture["condition"],
                        "model": label,
                        "env_idx": source["env_idx"],
                        "dataset_row": source["dataset_row"],
                        "candidate_idx": candidate_idx,
                        "E_roll_mm": float(e_roll[candidate_idx]),
                        "E_enc_mm": float(e_enc[candidate_idx]),
                        "Delta_roll_minus_enc_mm": float(e_roll[candidate_idx] - e_enc[candidate_idx]),
                        "E_imag_mm": float(e_imag[candidate_idx]),
                        "latent_l2": float(drift["latent_l2"][candidate_idx]),
                        "latent_cosine_distance": float(drift["latent_cosine_distance"][candidate_idx]),
                        "final_success": bool(final_success[candidate_idx]),
                        "ever_success": bool(ever_success[candidate_idx]),
                        "terminal_x_m": float(terminal_xyz[candidate_idx, 0]),
                        "terminal_y_m": float(terminal_xyz[candidate_idx, 1]),
                        "terminal_z_m": float(terminal_xyz[candidate_idx, 2]),
                        "roll_x_m": float(roll_xyz[candidate_idx, 0]),
                        "roll_y_m": float(roll_xyz[candidate_idx, 1]),
                        "roll_z_m": float(roll_xyz[candidate_idx, 2]),
                        "enc_x_m": float(enc_xyz[candidate_idx, 0]),
                        "enc_y_m": float(enc_xyz[candidate_idx, 1]),
                        "enc_z_m": float(enc_xyz[candidate_idx, 2]),
                        "latent_cost": float(latent_costs[candidate_idx]),
                    }
                )
        del model, probe
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    expected = len(labels) * len(common.AUDIT_ENVS) * common.NUM_SAMPLES
    if len(rows) != expected:
        raise RuntimeError(f"score row mismatch: expected={expected}, actual={len(rows)}")
    actual_checkpoint_sha = model_provenance["masked"]["checkpoint"]["sha256"]
    expected_checkpoint_sha = frozen_checkpoint["weights"]["sha256"]
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError(
            "Masked probe bundle checkpoint differs from frozen Trust-Region checkpoint: "
            f"expected={expected_checkpoint_sha}, actual={actual_checkpoint_sha}"
        )
    summary = _score_summary(rows)
    e_roll = np.asarray([row["E_roll_mm"] for row in rows], dtype=np.float64)
    if e_roll.shape != (len(common.AUDIT_ENVS) * common.NUM_SAMPLES,) or not np.isfinite(e_roll).all():
        raise ValueError(
            "gate requires 12x300 finite Masked E_roll values: "
            f"expected_shape={(len(common.AUDIT_ENVS) * common.NUM_SAMPLES,)}, "
            f"actual_shape={e_roll.shape}, finite={bool(np.isfinite(e_roll).all())}"
        )
    observed_median = float(np.median(e_roll))
    encoder_floor_diagnostic = _encoder_floor_diagnostic(rows)
    capture_hashes = {
        str(source["env_idx"]): source["population_sha256"]
        for source in capture["cases"]
    }
    old_reference = None
    old_path = imag_common.OUTPUT_ROOT / "measurement2.json"
    if old_path.is_file():
        old_payload = json.loads(old_path.read_text(encoding="utf-8"))
        old_metric = old_payload["by_model_condition_stratum"]["masked"][capture["condition"]]["all"]["E_roll_mm"]
        old_reference = {
            "source": common.file_identity(old_path),
            "protocol": "legacy unseeded CEM pool; report-only, not a gate condition",
            "E_roll_mm": old_metric,
        }
    gate = {
        "format_version": "cube_trust_region_gate_v1",
        "status": "PASS" if observed_median <= 40.0 else "FAIL",
        "protocol": capture["protocol"],
        "condition": capture["condition"],
        "primary_model": "masked",
        "criterion": {
            "metric": "E_roll_mm_median_fixed12x300",
            "count": int(e_roll.size),
            "observed_median_mm": observed_median,
            "threshold_mm": 40.0,
            "operator": "<=",
            "all_terminal_truth_hashes_complete": True,
            "all_values_finite": True,
        },
        "encoder_floor_diagnostic_report_only": encoder_floor_diagnostic,
        "encoder_floor_not_gate_condition": True,
        "checkpoint": frozen_checkpoint,
        "probe": model_provenance["masked"]["probe"],
        "capture": {
            "root": str(capture["root"]),
            "results": common.file_identity(capture["results_path"]),
            "pool_manifest": common.file_identity(capture["manifest_path"]),
            "population_sha256_by_env": capture_hashes,
        },
        "physical_cache_manifest": common.file_identity(Path(physical_root) / "manifest.json"),
        "old_unseeded_reference": old_reference,
        "old_reference_is_gate_condition": False,
    }
    output = common.prepare_output(
        output_candidate,
        common.OUTPUT_ROOT,
        args.overwrite,
    )
    common.write_csv(output / "candidate_imagination_error.csv", rows, SCORE_FIELDS)
    common.write_json(output / "gate.json", gate)
    common.write_json(
        output / "summary.json",
        {
            "format_version": "cube_trust_region_imagination_error_v1",
            "protocol": capture["protocol"],
            "condition": capture["condition"],
            "source_evaluation": common.file_identity(capture["results_path"]),
            "source_pool_manifest": common.file_identity(capture["manifest_path"]),
            "physical_cache_manifest": common.file_identity(
                Path(physical_root) / "manifest.json"
            ),
            "physical_terminal_truth_required": True,
            "legacy_audit_labels_used": False,
            "latent_cost_source": (
                "captured Trust-Region evaluator model cost for the exact new pool; "
                "not recomputed per scoring model"
            ),
            "models": model_provenance,
            "by_model_stratum": summary,
            "gate": gate,
            "encoder_floor_diagnostic_report_only": encoder_floor_diagnostic,
            "encoder_floor_not_gate_condition": True,
            "num_rows": len(rows),
        },
    )
    print(f"{output} gate={gate['status']} median_E_roll_mm={observed_median:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--evaluation-root", type=Path, required=True)
    validate.add_argument("--physical-root", type=Path)
    replay = sub.add_parser("replay")
    replay.add_argument("--evaluation-root", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--overwrite", action="store_true")
    replay.add_argument("--authorize-physical-replay", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("--evaluation-root", type=Path, required=True)
    score.add_argument("--physical-root", type=Path)
    score.add_argument("--output", type=Path)
    score.add_argument("--device", default="cuda")
    score.add_argument("--rollout-batch-size", type=int, default=300)
    score.add_argument("--encoder-batch-size", type=int, default=128)
    score.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        return command_validate(args)
    if args.command == "replay":
        return command_replay(args)
    if args.command == "score":
        if args.rollout_batch_size <= 0 or args.encoder_batch_size <= 0:
            raise ValueError("score batch sizes must be positive")
        return command_score(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
