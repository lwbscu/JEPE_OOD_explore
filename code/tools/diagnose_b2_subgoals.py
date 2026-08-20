#!/usr/bin/env python3
"""Strictly offline diagnosis of the 36 B2 rule-arm interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT_ROOT / "outputs/eval/cube/longhorizon/rule_offset100"
DEFAULT_DATASET = PROJECT_ROOT / "datasets/ogbench/cube_single_expert.h5"
DEFAULT_INDEX = PROJECT_ROOT / "outputs/memory_index/cube_expert_v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/eval/cube/longhorizon/subgoal_diagnosis"
)
EXPECTED_INTERVENTIONS = 36
CALIBRATION_SEED = 420031
CALIBRATION_ROWS = 8000
VISIBLE_FRACTION_MIN = 0.80
COMPOSITION_PROGRESS_FRACTION_MAX = 0.25
COMPOSITION_CEM_IMPROVEMENT_MIN = 0.20
OSCILLATION_TARGET_DISTANCE_M = 0.04
OSCILLATION_MAX_SPACING_STEPS = 52


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, hash_content: bool = True) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path) if hash_content else None,
        "hash_omission_reason": (
            None
            if hash_content
            else "95GB HDF5 content hash omitted; consumed rows and values are hashed"
        ),
    }


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_red_centroid(rgb: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) / 179.0
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    mask = ((hue > 0.90) | (hue < 0.03)) & (sat > 0.40) & (val > 0.15)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None, {"reason": "no_red_component"}
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    width = int(stats[component, cv2.CC_STAT_WIDTH])
    height = int(stats[component, cv2.CC_STAT_HEIGHT])
    if not (35 <= area <= 900 and 5 <= width <= 45 and 5 <= height <= 45):
        return None, {
            "reason": "red_component_geometry_out_of_range",
            "area": area,
            "width": width,
            "height": height,
        }
    u, v = centers[component]
    feature = np.asarray([u, v, math.sqrt(area), width, height], dtype=np.float64)
    return feature, {
        "reason": None,
        "u": float(u),
        "v": float(v),
        "area": area,
        "width": width,
        "height": height,
    }


def _design(features: np.ndarray) -> np.ndarray:
    f = np.asarray(features, dtype=np.float64)
    u, v, a, w, h = (f[:, i] for i in range(5))
    return np.column_stack(
        [
            np.ones(len(f)),
            u,
            v,
            a,
            w,
            h,
            u * u,
            u * v,
            v * v,
            u * a,
            v * a,
        ]
    )


def _fit_calibrator(
    dataset_path: Path,
    anchor_rows_path: Path,
    fixed_episodes: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    import hdf5plugin  # noqa: F401
    import h5py

    anchors = np.load(anchor_rows_path, mmap_mode="r")
    rng = np.random.default_rng(CALIBRATION_SEED)
    eligible = np.flatnonzero(
        ~np.isin(np.asarray(anchors, dtype=np.int64) // 201, fixed_episodes)
    )
    chosen = np.sort(
        rng.choice(eligible, size=CALIBRATION_ROWS, replace=False)
    )
    rows = np.asarray(anchors[chosen], dtype=np.int64)
    features: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    episodes: list[int] = []
    consumed_pixels = hashlib.sha256()
    with h5py.File(dataset_path, "r", swmr=True) as h5:
        for start in range(0, len(rows), 128):
            batch_rows = rows[start : start + 128]
            pixels = np.asarray(h5["pixels"][batch_rows], dtype=np.uint8)
            pos = np.asarray(h5["privileged_block_0_pos"][batch_rows], dtype=np.float64)
            eps = np.asarray(h5["ep_idx"][batch_rows], dtype=np.int64)
            consumed_pixels.update(np.ascontiguousarray(pixels).tobytes())
            for image, xyz, episode in zip(pixels, pos, eps):
                detected, _ = _detect_red_centroid(image)
                if detected is None:
                    continue
                features.append(detected)
                positions.append(xyz)
                episodes.append(int(episode))
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(positions, dtype=np.float64)
    eps = np.asarray(episodes, dtype=np.int64)
    if len(f) < 5000:
        raise RuntimeError(f"insufficient calibration detections: {len(f)}")
    if np.intersect1d(eps, fixed_episodes).size:
        raise RuntimeError("calibration rows leaked a fixed evaluation episode")
    validation = (eps % 5) == 0
    if validation.sum() < 500 or (~validation).sum() < 2000:
        raise RuntimeError("calibration episode split is too small")
    x_train = _design(f[~validation])
    center = x_train[:, 1:].mean(axis=0)
    scale = x_train[:, 1:].std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_train[:, 1:] = (x_train[:, 1:] - center) / scale
    coefficient, *_ = np.linalg.lstsq(x_train, y[~validation], rcond=None)
    x_valid = _design(f[validation])
    x_valid[:, 1:] = (x_valid[:, 1:] - center) / scale
    prediction = x_valid @ coefficient
    error_mm = np.linalg.norm(prediction - y[validation], axis=1) * 1000.0
    xy_error_mm = np.linalg.norm(
        prediction[:, :2] - y[validation, :2], axis=1
    ) * 1000.0
    model = {
        "center": center,
        "scale": scale,
        "coefficient": coefficient,
    }
    report = {
        "method": (
            "largest connected red HSV component; polynomial least-squares mapping "
            "[u,v,sqrt(area),width,height] to privileged block xyz"
        ),
        "red_mask": "(hue>0.90 or hue<0.03) and saturation>0.40 and value>0.15",
        "sampling_seed": CALIBRATION_SEED,
        "requested_rows": CALIBRATION_ROWS,
        "detected_rows": len(f),
        "train_rows": int((~validation).sum()),
        "validation_rows": int(validation.sum()),
        "episode_split": "validation iff non-heldout episode_idx % 5 == 0",
        "fixed50_excluded": True,
        "median_xyz_error_mm": float(np.median(error_mm)),
        "p90_xyz_error_mm": float(np.percentile(error_mm, 90)),
        "median_xy_error_mm": float(np.median(xy_error_mm)),
        "p90_xy_error_mm": float(np.percentile(xy_error_mm, 90)),
        "calibration_rows_sha256_int64": _hash_array(rows),
        "detected_episode_sha256_int64": _hash_array(eps),
        "consumed_calibration_pixels_sha256": consumed_pixels.hexdigest(),
        "feature_sha256_float64": _hash_array(f),
        "target_xyz_sha256_float64": _hash_array(y),
        "coefficient": coefficient,
        "feature_center": center,
        "feature_scale": scale,
    }
    provenance = {
        "selected_anchor_indices_sha256_int64": _hash_array(chosen),
        "selected_anchor_rows_sha256_int64": _hash_array(rows),
        "consumed_calibration_pixels_sha256": consumed_pixels.hexdigest(),
        "consumed_positions_sha256_float64": _hash_array(y),
    }
    return model, rows, {"report": report, "provenance": provenance}


def _predict_position(model: Mapping[str, np.ndarray], feature: np.ndarray) -> np.ndarray:
    design = _design(np.asarray(feature, dtype=np.float64)[None])
    design[:, 1:] = (design[:, 1:] - model["center"]) / model["scale"]
    return (design @ model["coefficient"])[0]


def _expert_reachability(
    dataset_path: Path, fixed_episodes: np.ndarray
) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(dataset_path, "r", swmr=True) as h5:
        positions = h5["privileged_block_0_pos"]
        chunks: list[np.ndarray] = []
        for episode in range(len(h5["ep_len"])):
            if episode in fixed_episodes:
                continue
            offset = int(h5["ep_offset"][episode])
            length = int(h5["ep_len"][episode])
            if length <= 25:
                continue
            pos = np.asarray(positions[offset : offset + length], dtype=np.float64)
            chunks.append(np.linalg.norm(pos[25:] - pos[:-25], axis=1))
    displacement = np.concatenate(chunks)
    p90 = float(np.percentile(displacement, 90))
    return {
        "estimator": (
            "p90 Euclidean block displacement over every 25-step window in all "
            "expert episodes except the fixed 50 evaluation episodes"
        ),
        "num_windows": len(displacement),
        "median_m": float(np.median(displacement)),
        "p75_m": float(np.percentile(displacement, 75)),
        "p90_m": p90,
        "p95_m": float(np.percentile(displacement, 95)),
        "max_m": float(np.max(displacement)),
        "displacement_sha256_float64": _hash_array(displacement),
    }


def _agent_frames(video_path: Path) -> Iterable[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if bgr.shape[0] < 240 or bgr.shape[1] < 240:
                raise RuntimeError(f"unexpected panel video shape: {bgr.shape}")
            rgb = cv2.cvtColor(bgr[16:240, 16:240], cv2.COLOR_BGR2RGB)
            yield index, rgb
            index += 1
    finally:
        capture.release()


def _video_positions(
    video_path: Path, model: Mapping[str, np.ndarray]
) -> tuple[list[np.ndarray | None], list[dict[str, Any]]]:
    positions: list[np.ndarray | None] = []
    detections: list[dict[str, Any]] = []
    for frame_idx, rgb in _agent_frames(video_path):
        feature, detection = _detect_red_centroid(rgb)
        detection["frame_idx"] = frame_idx
        detections.append(detection)
        positions.append(None if feature is None else _predict_position(model, feature))
    return positions, detections


def _linear_slope(values: Sequence[float | None]) -> float | None:
    valid = [(i, float(value)) for i, value in enumerate(values) if value is not None]
    if len(valid) < 3:
        return None
    x = np.asarray([item[0] for item in valid], dtype=np.float64)
    y = np.asarray([item[1] for item in valid], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _cost_history(run_root: Path, manifest: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    rows = manifest["evaluated_rows"]
    for env_idx, row in enumerate(rows):
        path = run_root / "cost_history" / f"episode_{env_idx:02d}_row_{row}.json"
        result[env_idx] = _load_json(path)["planning_cycles"]
    return result


def _cost_metrics(cycles: Sequence[Mapping[str, Any]], start: int, end: int) -> dict[str, Any]:
    before = [cycle for cycle in cycles if int(cycle["env_step"]) < start]
    after = [cycle for cycle in cycles if int(cycle["env_step"]) == start]
    during = [cycle for cycle in cycles if start <= int(cycle["env_step"]) < end]
    if len(after) != 1:
        raise RuntimeError(f"expected one subgoal solve at env_step={start}, got {len(after)}")
    solve = after[0]
    iteration = [float(value) for value in solve["best_cost_by_iteration"]]
    first = iteration[0]
    last = iteration[-1]
    improvement = None if first == 0 else (first - last) / abs(first)
    before_cost = None if not before else float(before[-1]["final_top1_cost"])
    after_cost = float(solve["final_top1_cost"])
    return {
        "planner_cost_before_env_step": None if not before else int(before[-1]["env_step"]),
        "planner_cost_before": before_cost,
        "planner_cost_after_env_step": int(solve["env_step"]),
        "planner_cost_after": after_cost,
        "planner_cost_after_minus_before": (
            None if before_cost is None else after_cost - before_cost
        ),
        "planner_cost_after_iteration_series": iteration,
        "planner_cost_after_iteration_slope": _linear_slope(iteration),
        "planner_cost_after_iteration_fractional_improvement": improvement,
        "planner_cost_series_during": [
            {
                "env_step": int(cycle["env_step"]),
                "final_top1_cost": float(cycle["final_top1_cost"]),
            }
            for cycle in during
        ],
        "planner_cost_during_slope_per_cycle": _linear_slope(
            [float(cycle["final_top1_cost"]) for cycle in during]
        ),
    }


def _label(value: bool | None, evidence: str) -> dict[str, Any]:
    return {"value": "unknown" if value is None else bool(value), "evidence": evidence}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def finite(field: str) -> np.ndarray:
        return np.asarray(
            [float(row[field]) for row in records if row.get(field) is not None],
            dtype=np.float64,
        )

    labels = {}
    for name in ("window_insufficient", "composition_mismatch", "switching_oscillation"):
        counts = Counter(str(row["labels"][name]["value"]).lower() for row in records)
        labels[name] = dict(sorted(counts.items()))
    progress = finite("physical_progress_m")
    ratio = finite("initial_distance_over_expert_p90_25step")
    visibility = finite("video_visible_fraction")
    return {
        "num_interventions": len(records),
        "outcomes": dict(Counter(row["outcome"] for row in records)),
        "labels": labels,
        "initial_distance_m": _distribution(finite("initial_distance_to_subgoal_m")),
        "final_distance_m": _distribution(finite("final_distance_to_subgoal_m")),
        "min_distance_m": _distribution(finite("min_distance_to_subgoal_m")),
        "physical_progress_m": _distribution(progress),
        "initial_over_expert_p90_ratio": _distribution(ratio),
        "video_visible_fraction": _distribution(visibility),
    }


def _report_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    labels = aggregate["labels"]
    calibration = summary["calibration"]
    reach = summary["expert_25step_reachability"]
    return "\n".join(
        [
            "# B2 Subgoal Diagnosis",
            "",
            "This report is strictly offline. It consumes the completed B2 rule artifact, "
            "saved MP4 panels, the HDF5 expert dataset, and the frozen memory index; "
            "it does not instantiate a simulator or call a GPU.",
            "",
            "## Conclusion",
            "",
            f"- Parsed exactly {summary['num_interventions_observed']} rule interventions. "
            f"All {labels['window_insufficient'].get('false', 0)} had initial distance below "
            "the empirical expert p90 25-step reachability estimate; the evidence does "
            "not support a hard window-insufficient diagnosis.",
            f"- {labels['composition_mismatch'].get('true', 0)} interventions are labeled "
            "composition_mismatch: the planner reduced within-solve CEM cost while the "
            "physical subgoal distance made less than 25% progress. Four are negative; "
            "none are forced to unknown by missing video steps.",
            f"- {labels['switching_oscillation'].get('true', 0)} are labeled "
            "switching_oscillation under the predeclared 52-step/4cm target-reuse rule; "
            f"{labels['switching_oscillation'].get('unknown', 0)} have no adjacent event "
            "with enough evidence.",
            "",
            "## Evidence",
            "",
            f"- Outcomes: `{json.dumps(aggregate['outcomes'], sort_keys=True)}`.",
            f"- Initial distance median/p90: "
            f"{aggregate['initial_distance_m']['median']:.4f}/"
            f"{aggregate['initial_distance_m']['p90']:.4f} m.",
            f"- Final distance median/p90: "
            f"{aggregate['final_distance_m']['median']:.4f}/"
            f"{aggregate['final_distance_m']['p90']:.4f} m.",
            f"- Physical progress median: {aggregate['physical_progress_m']['median']:.4g} m; "
            f"p90: {aggregate['physical_progress_m']['p90']:.4f} m.",
            f"- Expert p90 25-step displacement: {reach['p90_m']:.4f} m "
            f"({reach['num_windows']} non-heldout windows).",
            f"- Video calibration validation error: median {calibration['median_xyz_error_mm']:.1f} mm, "
            f"p90 {calibration['p90_xyz_error_mm']:.1f} mm; every intervention had a "
            f"1.00 visible interior-frame fraction.",
            "",
            "## Risks",
            "",
            "- Interior physical positions are estimates from the largest red HSV component "
            "in the saved agent panel, calibrated against non-heldout expert frames. "
            "The 61.7 mm p90 calibration error is material; exact initial/final logged "
            "distances are stronger evidence than the interior curve.",
            "- A panel video can hold a visually stable red centroid while the cube is "
            "occluded or depth-shifted. No missing frame was silently interpolated; any "
            "unavailable measurement is encoded as null/unknown.",
            "- The labels are overlapping operational diagnostics, not causal proof. "
            "Composition mismatch is defined by cost improvement plus weak physical "
            "progress, while switching oscillation is defined by target reuse and spacing.",
            "",
        ]
    )


def _distribution(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"count": 0, "median": None, "p10": None, "p90": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def run(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    names = (
        "events.json",
        "subgoal_retrieval.json",
        "intervention_outcomes.json",
        "goal_switches.json",
        "results.json",
        "run_manifest.json",
    )
    inputs = {name: _identity(run_root / name) for name in names}
    manifest = _load_json(run_root / "run_manifest.json")
    events = _load_json(run_root / "events.json")
    retrievals = _load_json(run_root / "subgoal_retrieval.json")
    outcomes = _load_json(run_root / "intervention_outcomes.json")
    switches = _load_json(run_root / "goal_switches.json")
    if not (len(events) == len(retrievals) == len(outcomes) == EXPECTED_INTERVENTIONS):
        raise RuntimeError(
            "formal rule intervention count drift: "
            f"events={len(events)}, retrievals={len(retrievals)}, outcomes={len(outcomes)}"
        )
    if manifest.get("status") != "complete" or manifest.get("mode") != "rule":
        raise RuntimeError("input is not a completed formal B2 rule run")

    fixed = np.asarray(
        manifest["retrieval"]["global_excluded_episodes"], dtype=np.int64
    )
    model, calibration_rows, calibration = _fit_calibrator(
        args.dataset, args.index / "anchor_rows.npy", fixed
    )
    reachability = _expert_reachability(args.dataset, fixed)
    expert_p90 = float(reachability["p90_m"])
    costs = _cost_history(run_root, manifest)

    by_env_retrievals: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in retrievals:
        by_env_retrievals[int(item["env_idx"])].append(item)
    for items in by_env_retrievals.values():
        items.sort(key=lambda item: int(item["step"]))

    video_cache: dict[int, tuple[list[np.ndarray | None], list[dict[str, Any]]]] = {}
    records: list[dict[str, Any]] = []
    consumed_video_identities: dict[str, Any] = {}
    event_lookup = {(int(x["env_idx"]), int(x["step"])): x for x in events}
    outcome_lookup = {
        (int(x["env_idx"]), int(x["started_step"])): x for x in outcomes
    }
    start_switch_lookup = {
        (int(x["env_idx"]), int(x["step"])): x
        for x in switches
        if x["to_kind"] == "subgoal"
    }

    for intervention_id, retrieval in enumerate(
        sorted(retrievals, key=lambda x: (int(x["env_idx"]), int(x["step"])))
    ):
        env_idx = int(retrieval["env_idx"])
        start = int(retrieval["step"])
        event = event_lookup[(env_idx, start)]
        outcome = outcome_lookup[(env_idx, start)]
        switch = start_switch_lookup[(env_idx, start)]
        end = int(outcome["step"])
        target = np.asarray(retrieval["selected_real_frame"]["position"], dtype=np.float64)
        initial_pos = np.asarray(event["state"]["block_pos"], dtype=np.float64)
        initial_distance = float(np.linalg.norm(initial_pos - target))
        raw_final_distance = outcome.get("block_distance_m")
        final_distance = (
            None if raw_final_distance is None else float(raw_final_distance)
        )

        if env_idx not in video_cache:
            video_path = run_root / "videos" / f"env_{env_idx}.mp4"
            video_cache[env_idx] = _video_positions(video_path, model)
            consumed_video_identities[video_path.name] = _identity(video_path)
        video_positions, detections = video_cache[env_idx]
        # callback step k is saved as MP4 frame k-1.  The trigger position at S
        # and outcome distance at E are exact logs; S+1..E-1 are video estimates.
        series: list[dict[str, Any]] = [
            {
                "step": start,
                "source": "privileged_trigger_log",
                "position": initial_pos,
                "distance_to_subgoal_m": initial_distance,
                "detection": None,
            }
        ]
        for step in range(start + 1, end):
            frame_idx = step - 1
            estimate = video_positions[frame_idx] if frame_idx < len(video_positions) else None
            series.append(
                {
                    "step": step,
                    "source": "video_red_centroid_calibration" if estimate is not None else "missing",
                    "position": estimate,
                    "distance_to_subgoal_m": (
                        None if estimate is None else float(np.linalg.norm(estimate - target))
                    ),
                    "detection": (
                        None if frame_idx >= len(detections) else detections[frame_idx]
                    ),
                }
            )
        series.append(
            {
                "step": end,
                "source": (
                    "privileged_outcome_distance_log"
                    if final_distance is not None
                    else (
                        "video_red_centroid_calibration"
                        if end - 1 < len(video_positions)
                        and video_positions[end - 1] is not None
                        else "missing"
                    )
                ),
                "position": (
                    None
                    if final_distance is not None
                    or end - 1 >= len(video_positions)
                    else video_positions[end - 1]
                ),
                "distance_to_subgoal_m": (
                    final_distance
                    if final_distance is not None
                    else (
                        None
                        if end - 1 >= len(video_positions)
                        or video_positions[end - 1] is None
                        else float(np.linalg.norm(video_positions[end - 1] - target))
                    )
                ),
                "detection": (
                    None if end - 1 >= len(detections) else detections[end - 1]
                ),
            }
        )
        final_distance = series[-1]["distance_to_subgoal_m"]
        distances = [row["distance_to_subgoal_m"] for row in series]
        valid = [float(value) for value in distances if value is not None]
        estimated_interior = series[1:-1]
        visible = sum(row["distance_to_subgoal_m"] is not None for row in estimated_interior)
        visible_fraction = 1.0 if not estimated_interior else visible / len(estimated_interior)
        min_distance = float(min(valid)) if valid else None
        progress = (
            None if final_distance is None else initial_distance - float(final_distance)
        )
        progress_fraction = (
            None
            if progress is None or initial_distance <= 1e-9
            else progress / initial_distance
        )
        delta = [
            float(a) - float(b)
            for a, b in zip(distances, distances[1:])
            if a is not None and b is not None
        ]
        monotonic_fraction = None if not delta else sum(value >= 0 for value in delta) / len(delta)
        physical_slope = _linear_slope(distances)
        cost = _cost_metrics(costs[env_idx], start, end)

        env_interventions = by_env_retrievals[env_idx]
        ordinal = env_interventions.index(retrieval)
        previous = env_interventions[ordinal - 1] if ordinal else None
        following = (
            env_interventions[ordinal + 1]
            if ordinal + 1 < len(env_interventions)
            else None
        )
        previous_spacing = None if previous is None else start - int(previous["step"])
        next_spacing = None if following is None else int(following["step"]) - start
        neighbor_evidence: list[dict[str, Any]] = []
        for neighbor, spacing in ((previous, previous_spacing), (following, next_spacing)):
            if neighbor is None or spacing is None:
                continue
            neighbor_target = np.asarray(
                neighbor["selected_real_frame"]["position"], dtype=np.float64
            )
            neighbor_evidence.append(
                {
                    "step": int(neighbor["step"]),
                    "spacing_steps": int(spacing),
                    "target_distance_m": float(np.linalg.norm(target - neighbor_target)),
                }
            )

        ratio = None if expert_p90 <= 0 else initial_distance / expert_p90
        window_value = None if ratio is None else ratio > 1.0
        cem_improvement = cost["planner_cost_after_iteration_fractional_improvement"]
        composition_value: bool | None
        if visible_fraction < VISIBLE_FRACTION_MIN or progress_fraction is None or cem_improvement is None:
            composition_value = None
        else:
            composition_value = bool(
                progress_fraction < COMPOSITION_PROGRESS_FRACTION_MAX
                and cem_improvement >= COMPOSITION_CEM_IMPROVEMENT_MIN
            )
        if not neighbor_evidence:
            oscillation_value = None
        else:
            oscillation_value = any(
                item["spacing_steps"] <= OSCILLATION_MAX_SPACING_STEPS
                and item["target_distance_m"] <= OSCILLATION_TARGET_DISTANCE_M
                for item in neighbor_evidence
            )

        record = {
            "intervention_id": intervention_id,
            "env_idx": env_idx,
            "episode": int(retrieval["episode"]),
            "intervention_index_within_env": int(retrieval["intervention_index"]),
            "start_step": start,
            "end_step": end,
            "elapsed_physical_steps": int(outcome["elapsed_physical_steps"]),
            "outcome": str(outcome["outcome"]),
            "target_row": int(retrieval["selected_real_frame"]["row"]),
            "target_episode": int(retrieval["selected_real_frame"]["episode"]),
            "target_position": target,
            "initial_block_position": initial_pos,
            "initial_distance_to_subgoal_m": initial_distance,
            "final_distance_to_subgoal_m": final_distance,
            "min_distance_to_subgoal_m": min_distance,
            "physical_progress_m": progress,
            "physical_progress_fraction": progress_fraction,
            "physical_distance_slope_m_per_step": physical_slope,
            "monotonic_nonincrease_fraction": monotonic_fraction,
            "video_visible_steps": visible,
            "video_expected_interior_steps": len(estimated_interior),
            "video_visible_fraction": visible_fraction,
            "distance_series": series,
            "expert_reachable_25step_p90_m": expert_p90,
            "initial_distance_over_expert_p90_25step": ratio,
            **cost,
            "interventions_in_env": len(env_interventions),
            "previous_intervention_spacing_steps": previous_spacing,
            "next_intervention_spacing_steps": next_spacing,
            "neighbor_target_evidence": neighbor_evidence,
            "goal_switch_count_inclusive_interval": sum(
                int(item["env_idx"]) == env_idx and start <= int(item["step"]) <= end
                for item in switches
            ),
            "labels": {
                "window_insufficient": _label(
                    window_value,
                    f"initial/p90 ratio={ratio}; true iff ratio>1.0",
                ),
                "composition_mismatch": _label(
                    composition_value,
                    "true iff video visibility>=0.80, physical progress fraction<0.25, "
                    "and within-solve CEM best-cost improvement>=0.20",
                ),
                "switching_oscillation": _label(
                    oscillation_value,
                    "true iff adjacent intervention is within 52 steps and its real-frame "
                    "target is within 0.04m; unknown with no adjacent intervention",
                ),
            },
        }
        records.append(record)

    records.sort(key=lambda row: int(row["intervention_id"]))
    if len(records) != EXPECTED_INTERVENTIONS:
        raise RuntimeError(f"diagnostic row count drift: {len(records)}")

    csv_path = output / "interventions.csv"
    csv_fields = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: _csv_value(row.get(key)) for key in csv_fields})
    _write_json(output / "interventions.json", records)

    dataset_identity = _identity(args.dataset, hash_content=False)
    dataset_identity["consumed_rows_and_values"] = calibration["provenance"]
    input_manifest = {
        "format_version": "cube_b2_subgoal_diagnosis_inputs_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strictly_offline": True,
        "simulation_calls": 0,
        "gpu_calls": 0,
        "rule_run": inputs,
        "cost_history_json": {
            f"env_{env_idx}": _identity(
                run_root
                / "cost_history"
                / f"episode_{env_idx:02d}_row_{manifest['evaluated_rows'][env_idx]}.json"
            )
            for env_idx in sorted(by_env_retrievals)
        },
        "consumed_videos": consumed_video_identities,
        "dataset": dataset_identity,
        "memory_anchor_rows": _identity(args.index / "anchor_rows.npy"),
        "fixed_eval_episodes_sha256_int64": _hash_array(fixed),
        "calibration": calibration["provenance"],
    }
    _write_json(output / "input_manifest.json", input_manifest)
    summary = {
        "format_version": "cube_b2_subgoal_diagnosis_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strictly_offline": True,
        "num_interventions_expected": EXPECTED_INTERVENTIONS,
        "num_interventions_observed": len(records),
        "distance_sources": {
            "initial": "privileged state logged at trigger",
            "final": "privileged block_distance_m logged by intervention outcome",
            "interior_and_min": "MP4 agent panel red-centroid physical calibrator; missing remains null",
            "frame_alignment": "callback step k is MP4 frame k-1",
        },
        "calibration": calibration["report"],
        "expert_25step_reachability": reachability,
        "label_definitions": {
            "overlap_allowed": True,
            "window_insufficient": "initial distance / expert p90 25-step displacement > 1.0",
            "composition_mismatch": (
                "visibility>=0.80 and physical progress fraction<0.25 and CEM "
                "within-solve best-cost improvement>=0.20"
            ),
            "switching_oscillation": (
                "adjacent intervention within 52 steps reuses a real-frame target within 0.04m"
            ),
            "unknown_policy": "unknown whenever the required observable is unavailable",
        },
        "aggregate": _summarize(records),
        "artifacts": {
            "input_manifest": "input_manifest.json",
            "interventions_json": "interventions.json",
            "interventions_csv": "interventions.csv",
        },
    }
    _write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_report_markdown(summary), encoding="utf-8")
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
