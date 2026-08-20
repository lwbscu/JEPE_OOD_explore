#!/usr/bin/env python3
"""Capture and physically replay the first Cube CEM population.

This tool answers a narrow audit question: for selected cases from the
published 50-case OGBench-Cube evaluation, how well does LeWM's latent CEM
cost rank the physical outcome of the 300 candidates sampled in the final
CEM iteration?

Two details are deliberate:

* ``capture`` solves all 50 cases in their original order, even though only
  twelve are saved.  CEM owns one seeded torch.Generator, so dropping an
  earlier environment would change the later environments' candidates.
* CEM executes the *updated elite mean*, not one of the final 300 samples.
  ``replay`` therefore evaluates the 300 samples and that mean separately.

The module keeps heavyweight imports inside subcommands.  Merely running
``--help`` or ``manifest`` never loads a model or initializes CUDA/MuJoCo.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "datasets/ogbench/cube_single_expert.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/audit/cube_cem_300"
RECOLOR_GOAL_ROOT = PROJECT_ROOT / "outputs/eval/cube/ood/goal_recolor"
COLOR_OUTPUTS = {
    "dataset": DEFAULT_OUTPUT,
    "blue": PROJECT_ROOT / "outputs/audit/cube_cem_300_blue_matched",
    "yellow": PROJECT_ROOT / "outputs/audit/cube_cem_300_yellow_matched",
}
RECOLOR_OUTPUTS = {
    "blue": PROJECT_ROOT / "outputs/audit/cube_cem_300_blue_v2",
    "yellow": PROJECT_ROOT / "outputs/audit/cube_cem_300_yellow_v2",
}
AUDIT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/audit"
DEFAULT_TMP = PROJECT_ROOT.parent / "tmp"

CUBE_RGB = {
    "blue": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    "yellow": np.asarray([1.0, 1.0, 0.0], dtype=np.float64),
}

FORMAL_SEED = 42
FORMAL_NUM_EVAL = 50
GOAL_OFFSET = 25
HORIZON = 5
ACTION_BLOCK = 5
ACTION_DIM = 5
NUM_SAMPLES = 300
N_STEPS = 10
TOPK = 30
SUCCESS_THRESHOLD_METERS = 0.04

# Fixed before this tool was written: six failures and six successes from the
# completed 50-case pretrained Cube evaluation.
AUDIT_CASES: tuple[dict[str, Any], ...] = (
    {"env_idx": 0, "dataset_row": 128267, "formal_50step_success": False},
    {"env_idx": 1, "dataset_row": 136513, "formal_50step_success": False},
    {"env_idx": 6, "dataset_row": 257500, "formal_50step_success": False},
    {"env_idx": 11, "dataset_row": 556268, "formal_50step_success": False},
    {"env_idx": 26, "dataset_row": 1269622, "formal_50step_success": False},
    {"env_idx": 38, "dataset_row": 1570913, "formal_50step_success": False},
    {"env_idx": 2, "dataset_row": 172735, "formal_50step_success": True},
    {"env_idx": 7, "dataset_row": 332101, "formal_50step_success": True},
    {"env_idx": 12, "dataset_row": 712588, "formal_50step_success": True},
    {"env_idx": 23, "dataset_row": 1058181, "formal_50step_success": True},
    {"env_idx": 37, "dataset_row": 1564529, "formal_50step_success": True},
    {"env_idx": 49, "dataset_row": 1960957, "formal_50step_success": True},
)


def _configure_storage() -> None:
    """Keep caches and temporary files on the persistent data disk."""

    defaults = {
        "STABLEWM_HOME": str(PROJECT_ROOT),
        "HF_HOME": str(PROJECT_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(PROJECT_ROOT.parent / ".cache/torch"),
        "TMPDIR": str(DEFAULT_TMP),
        "MUJOCO_GL": "egl",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    DEFAULT_TMP.mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dataset_identity(path: Path, h5: Any, formal_rows: np.ndarray) -> dict[str, Any]:
    """Cheap, reproducible identity without hashing the 95 GiB HDF5 file."""

    stat = path.stat()
    digest = hashlib.sha256()
    for key in ("ep_idx", "step_idx", "qpos", "qvel"):
        digest.update(np.ascontiguousarray(h5[key][formal_rows]).tobytes())
    goal_rows = formal_rows + GOAL_OFFSET
    digest.update(
        np.ascontiguousarray(h5["privileged_block_0_pos"][goal_rows]).tobytes()
    )
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "formal_rows_state_sha256": digest.hexdigest(),
    }


def _prepare_output(path: Path, overwrite: bool) -> None:
    raw_path = path.expanduser().absolute()
    if raw_path.is_symlink():
        raise ValueError(f"refusing symlink output path: {raw_path}")
    path = raw_path.resolve()
    audit_root = AUDIT_OUTPUT_ROOT.resolve()
    if path == audit_root or audit_root not in path.parents:
        raise ValueError(
            f"output must be a concrete child of audit root {audit_root}: {path}"
        )
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output is not empty: {path}; pass --overwrite intentionally"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _default_output(cube_color: str, goal_protocol: str) -> Path:
    if goal_protocol == "recolor":
        if cube_color not in RECOLOR_OUTPUTS:
            raise ValueError("recolor goal protocol requires blue or yellow cube color")
        return RECOLOR_OUTPUTS[cube_color]
    return COLOR_OUTPUTS[cube_color]


def _capture_output(args: argparse.Namespace) -> Path:
    goal_protocol = getattr(args, "goal_protocol", "matched")
    return Path(
        args.output or _default_output(args.cube_color, goal_protocol)
    ).resolve()


def _recolor_goal_path(cube_color: str) -> Path:
    return RECOLOR_GOAL_ROOT / f"{cube_color}_goal.npy"


def _load_recolor_goals(cube_color: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the frozen formal-env-indexed, cube-only recolored goal frames."""

    path = _recolor_goal_path(cube_color)
    if not path.is_file():
        raise FileNotFoundError(f"recolor goal array missing: {path}")
    goals = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = (FORMAL_NUM_EVAL, 224, 224, 3)
    if goals.shape != expected_shape:
        raise ValueError(
            f"recolor goal shape mismatch: expected={expected_shape}, "
            f"actual={goals.shape}, path={path}"
        )
    if goals.dtype != np.uint8:
        raise TypeError(
            f"recolor goal dtype mismatch: expected=uint8, actual={goals.dtype}, "
            f"path={path}"
        )
    metadata = {
        "kind": "offline_recolor_v2",
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(expected_shape),
        "dtype": "uint8",
        "index_contract": "axis0_env_idx_matches_verified_formal_rows",
        "source_semantics": (
            "original_hdf5_future_frame_with_only_cube_pixels_recolored"
        ),
    }
    return np.asarray(goals).copy(), metadata


def _visual_protocol(
    cube_color: str,
    goal_protocol: str,
    recolor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "cube_color": cube_color,
        "cube_rgb": (
            "original_hdf5_training_red"
            if cube_color == "dataset"
            else CUBE_RGB[cube_color]
        ),
        "goal_type": goal_protocol,
        "goal_render": (
            "original_hdf5_future_frame"
            if cube_color == "dataset"
            else "current_arm_cube_teleported_to_hdf5_goal_pose"
        ),
        "reference_protocol": "formal_red_hdf5_seed42_offset25",
        "selection_stratum_protocol": "red_hdf5_cem10_formal50",
    }
    if goal_protocol == "recolor":
        if recolor_metadata is None:
            raise ValueError("recolor protocol metadata is required")
        payload["goal_render"] = "frozen_offline_cube_only_recolor"
        payload["goal_recolor"] = recolor_metadata
    return payload


def _validate_audit_input(path: Path) -> Path:
    raw_path = path.expanduser().absolute()
    if raw_path.is_symlink():
        raise ValueError(f"refusing symlink audit input: {raw_path}")
    resolved = raw_path.resolve()
    audit_root = AUDIT_OUTPUT_ROOT.resolve()
    if resolved == audit_root or audit_root not in resolved.parents:
        raise ValueError(
            f"input must be a concrete child of audit root {audit_root}: {resolved}"
        )
    return resolved


def _refuse_active_training() -> None:
    """Do not compete with a live training process for CUDA/EGL resources."""

    own_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == own_pid:
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"train.py" not in cmdline:
            continue
        try:
            cwd = (proc / "cwd").resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            cwd = None
        if (
            cwd == PROJECT_ROOT
            or (cwd is not None and PROJECT_ROOT in cwd.parents)
            or b"le-wm" in cmdline
        ):
            raise RuntimeError(
                "active LeWM training detected; wait for it to exit before Cube audit"
            )


def _formal_rows(h5: Any) -> np.ndarray:
    """Reproduce eval.py's exact seed-42 selection, including its off-by-one."""

    episode_idx = np.asarray(h5["ep_idx"][:])
    step_idx = np.asarray(h5["step_idx"][:])
    lengths = np.asarray(h5["ep_len"][:])
    max_start = lengths - GOAL_OFFSET - 1
    valid = step_idx <= max_start[episode_idx]
    valid_indices = np.nonzero(valid)[0]
    rng = np.random.default_rng(FORMAL_SEED)
    sampled_positions = rng.choice(
        len(valid_indices) - 1, size=FORMAL_NUM_EVAL, replace=False
    )
    return np.sort(valid_indices[sampled_positions])


def _validate_manifest(h5: Any) -> np.ndarray:
    rows = _formal_rows(h5)
    for case in AUDIT_CASES:
        actual = int(rows[case["env_idx"]])
        expected = int(case["dataset_row"])
        if actual != expected:
            raise RuntimeError(
                "fixed manifest no longer matches dataset/eval selection: "
                f"env_{case['env_idx']} expected row {expected}, got {actual}"
            )
        goal_row = expected + GOAL_OFFSET
        if int(h5["ep_idx"][goal_row]) != int(h5["ep_idx"][expected]):
            raise RuntimeError(f"goal row crosses episode boundary: {expected}")
        delta = int(h5["step_idx"][goal_row]) - int(h5["step_idx"][expected])
        if delta != GOAL_OFFSET:
            raise RuntimeError(
                f"row {expected}: expected +{GOAL_OFFSET} steps, got +{delta}"
            )
    return rows


def command_manifest(args: argparse.Namespace) -> None:
    import hdf5plugin  # noqa: F401 - register HDF5 filters before h5py
    import h5py

    dataset = args.dataset.resolve()
    with h5py.File(dataset, "r", swmr=True) as h5:
        formal_rows = _validate_manifest(h5)
        payload = {
            "dataset": _dataset_identity(dataset, h5, formal_rows),
            "selection": {
                "seed": FORMAL_SEED,
                "num_eval": FORMAL_NUM_EVAL,
                "goal_offset": GOAL_OFFSET,
            },
            "formal_rows": formal_rows,
            "audit_cases": AUDIT_CASES,
        }
    if args.output:
        _write_json(args.output, payload)
        print(args.output.resolve())
    else:
        print(json.dumps(_jsonable(payload), indent=2))


class FinalPopulationRecorder:
    """CEM callback retaining only the last population of each env batch."""

    output_key = "cube_audit_final_population"

    def __init__(self, final_step: int) -> None:
        self.final_step = final_step
        self.records: list[dict[str, np.ndarray]] = []
        self.history: list[Any] = []

    def reset(self) -> None:
        self.records = []
        self.history = []

    def start_batch(self) -> None:
        return None

    def end_solve(self) -> None:
        self.history = [{"captured_batches": len(self.records)}]

    def __call__(self, **state: Any) -> None:
        if int(state["step"]) != self.final_step:
            return
        keys = (
            "candidates",
            "costs",
            "topk_vals",
            "topk_inds",
            "topk_candidates",
            "mean",
            "var",
            "prev_mean",
            "prev_var",
        )
        record = {
            key: state[key].detach().cpu().float().numpy()
            for key in keys
        }
        self.records.append(record)


def _image_transform(image_size: int) -> Any:
    import stable_pretraining as spt
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=image_size),
        ]
    )


def _set_cube_rgb(raw: Any, rgb: np.ndarray) -> None:
    """Change only the single physical cube/target visual material."""

    for geom_id in raw._cube_geom_ids_list[0]:
        raw._model.geom(geom_id).rgba[:3] = rgb
        raw._model.geom(geom_id).rgba[3] = 1.0
    for geom_id in raw._cube_target_geom_ids_list[0]:
        raw._model.geom(geom_id).rgba[:3] = rgb


def _render_color_inputs(
    h5: Any, formal_rows: np.ndarray, cube_color: str
) -> tuple[np.ndarray, np.ndarray]:
    """Render current and matched-goal inputs for every formal row.

    The goal keeps the arm at the current state and teleports only the cube
    to the HDF5 goal pose.  All 50 images are rendered so capture continues
    to solve the complete formal batch and consume CEM RNG in the same order.
    """

    if cube_color == "dataset":
        raise ValueError("dataset inputs must be loaded directly from HDF5")

    import mujoco

    rgb = CUBE_RGB[cube_color]
    current_images: list[np.ndarray] = []
    goal_images: list[np.ndarray] = []
    env = _make_replay_env()
    try:
        for row in formal_rows:
            row = int(row)
            goal_row = row + GOAL_OFFSET
            env.reset(seed=FORMAL_SEED)
            raw = _raw_env(env)
            qpos = np.asarray(h5["qpos"][row]).copy()
            qvel = np.asarray(h5["qvel"][row]).copy()
            goal_position = np.asarray(
                h5["privileged_block_0_pos"][goal_row]
            )
            goal_quaternion = np.asarray(
                h5["privileged_block_0_quat"][goal_row]
            )
            raw.set_state(qpos, qvel)
            raw.set_target_pos(0, goal_position, goal_quaternion)
            _set_cube_rgb(raw, rgb)
            mujoco.mj_forward(raw._model, raw._data)
            current_images.append(np.asarray(env.render(), dtype=np.uint8).copy())

            # Matched goal: preserve the current arm; move only the cube.
            cube_qpos = raw._data.joint("object_joint_0").qpos
            cube_qpos[:3] = goal_position
            cube_qpos[3:] = goal_quaternion
            mujoco.mj_forward(raw._model, raw._data)
            goal_images.append(np.asarray(env.render(), dtype=np.uint8).copy())

            # Restore for an explicit, auditable no-state-leak invariant.
            raw.set_state(qpos, qvel)
            _set_cube_rgb(raw, rgb)
            mujoco.mj_forward(raw._model, raw._data)
    finally:
        env.close()
    return np.stack(current_images), np.stack(goal_images)


def _load_capture_inputs(
    h5: Any,
    formal_rows: np.ndarray,
    cube_color: str = "dataset",
    recolor_goals: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    goal_rows = formal_rows + GOAL_OFFSET
    if cube_color == "dataset":
        pixels = np.asarray(h5["pixels"][formal_rows])
        goals = np.asarray(h5["pixels"][goal_rows])
    else:
        pixels, goals = _render_color_inputs(h5, formal_rows, cube_color)
    if recolor_goals is not None:
        expected_shape = (len(formal_rows), 224, 224, 3)
        if recolor_goals.shape != expected_shape or recolor_goals.dtype != np.uint8:
            raise ValueError(
                "invalid recolor goal array at capture input boundary: "
                f"expected={expected_shape}/uint8, "
                f"actual={recolor_goals.shape}/{recolor_goals.dtype}"
            )
        goals = recolor_goals.copy()
    return {
        "pixels": pixels[:, None, ...],
        "goal": goals[:, None, ...],
        "action": np.asarray(h5["action"][formal_rows])[:, None, ...],
    }


def command_capture(args: argparse.Namespace) -> None:
    """Run the formal first CEM solve and save final populations."""

    _configure_storage()
    _refuse_active_training()
    if args.goal_protocol == "recolor" and args.cube_color == "dataset":
        raise ValueError("recolor goal protocol requires blue or yellow cube color")
    recolor_goals = None
    recolor_metadata = None
    if args.goal_protocol == "recolor":
        recolor_goals, recolor_metadata = _load_recolor_goals(args.cube_color)
    visual_protocol = _visual_protocol(
        args.cube_color, args.goal_protocol, recolor_metadata
    )
    output = _capture_output(args)
    args.output = output
    _prepare_output(output, args.overwrite)

    import hdf5plugin  # noqa: F401 - register HDF5 filters before h5py
    import h5py
    import torch
    import stable_worldmodel as swm
    from gymnasium.spaces import Box
    from sklearn.preprocessing import StandardScaler

    if not torch.cuda.is_available():
        raise RuntimeError("capture requires CUDA; replay is CPU-only")
    if args.n_steps != N_STEPS or args.num_samples != NUM_SAMPLES:
        raise ValueError(
            "formal capture is frozen at n_steps=10 and num_samples=300"
        )

    with h5py.File(args.dataset, "r", swmr=True) as h5:
        formal_rows = _validate_manifest(h5)
        dataset_identity = _dataset_identity(args.dataset, h5, formal_rows)
        raw_inputs = _load_capture_inputs(
            h5, formal_rows, args.cube_color, recolor_goals
        )
        action_data = np.asarray(h5["action"][:])
        action_data = action_data[~np.isnan(action_data).any(axis=1)]
        action_scaler = StandardScaler().fit(action_data)
        manifest_details = [
            {
                **case,
                "episode_idx": int(h5["ep_idx"][case["dataset_row"]]),
                "start_step": int(h5["step_idx"][case["dataset_row"]]),
                "goal_row": int(case["dataset_row"] + GOAL_OFFSET),
            }
            for case in AUDIT_CASES
        ]
    del action_data

    model = swm.wm.utils.load_pretrained(
        args.checkpoint, cache_dir=str(PROJECT_ROOT)
    )
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    recorder = FinalPopulationRecorder(final_step=N_STEPS - 1)
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=NUM_SAMPLES,
        var_scale=1.0,
        n_steps=N_STEPS,
        topk=TOPK,
        device=args.device,
        seed=FORMAL_SEED,
        callbacks=[recorder],
    )
    config = swm.PlanConfig(
        horizon=HORIZON,
        receding_horizon=HORIZON,
        action_block=ACTION_BLOCK,
    )
    batched_action_space = Box(
        low=np.broadcast_to(-np.inf, (FORMAL_NUM_EVAL, ACTION_DIM)),
        high=np.broadcast_to(np.inf, (FORMAL_NUM_EVAL, ACTION_DIM)),
        dtype=np.float32,
    )
    solver.configure(
        action_space=batched_action_space,
        n_envs=FORMAL_NUM_EVAL,
        config=config,
    )

    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process={"action": action_scaler},
        transform={
            "pixels": _image_transform(args.image_size),
            "goal": _image_transform(args.image_size),
        },
    )
    prepared = policy._prepare_info(raw_inputs)

    with torch.inference_mode():
        outputs = solver(prepared, init_action=None)

    if len(recorder.records) != FORMAL_NUM_EVAL:
        raise RuntimeError(
            f"expected {FORMAL_NUM_EVAL} final populations, captured "
            f"{len(recorder.records)}"
        )
    final_means = outputs["actions"].detach().cpu().float().numpy()

    # Post-hoc mean costs are diagnostic only. They do not alter CEM's choice.
    mean_costs = np.full(FORMAL_NUM_EVAL, np.nan, dtype=np.float32)
    model_dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for env_idx in range(FORMAL_NUM_EVAL):
            one_info = {}
            for key, value in prepared.items():
                value = value[env_idx : env_idx + 1]
                if torch.is_tensor(value):
                    dtype = model_dtype if value.is_floating_point() else None
                    value = value.to(device=args.device, dtype=dtype).unsqueeze(1)
                elif isinstance(value, np.ndarray):
                    value = value[:, None, ...]
                one_info[key] = value
            candidate = outputs["actions"][env_idx : env_idx + 1].to(
                args.device
            )[:, None]
            mean_costs[env_idx] = float(
                model.get_cost(one_info, candidate).detach().cpu().item()
            )

    cases_by_env = {case["env_idx"]: case for case in manifest_details}
    for env_idx, case in cases_by_env.items():
        row = case["dataset_row"]
        goal_row = case["goal_row"]
        rec = recorder.records[env_idx]
        case_dir = output / f"env_{env_idx:02d}_row_{row}"
        case_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.dataset, "r", swmr=True) as h5:
            np.savez_compressed(
                case_dir / "population.npz",
                candidates_normalized=rec["candidates"][0],
                latent_costs=rec["costs"][0],
                topk_indices=rec["topk_inds"][0].astype(np.int64),
                topk_costs=rec["topk_vals"][0],
                final_mean_normalized=final_means[env_idx],
                final_mean_cost_posthoc=mean_costs[env_idx],
                final_variance=rec["var"][0],
                previous_mean=rec["prev_mean"][0],
                previous_variance=rec["prev_var"][0],
                action_scaler_mean=action_scaler.mean_,
                action_scaler_scale=action_scaler.scale_,
                initial_qpos=np.asarray(h5["qpos"][row]),
                initial_qvel=np.asarray(h5["qvel"][row]),
                initial_prev_qpos=np.asarray(h5["prev_qpos"][row]),
                initial_prev_qvel=np.asarray(h5["prev_qvel"][row]),
                goal_position=np.asarray(h5["privileged_block_0_pos"][goal_row]),
                goal_quaternion=np.asarray(
                    h5["privileged_block_0_quat"][goal_row]
                ),
                initial_pixels=np.asarray(raw_inputs["pixels"][env_idx, 0]),
                goal_pixels=np.asarray(raw_inputs["goal"][env_idx, 0]),
            )
        costs = rec["costs"][0]
        elite_indices = set(rec["topk_inds"][0].astype(np.int64).tolist())
        latent_ranks = _rank(costs)
        _write_csv(
            case_dir / "latent_population.csv",
            [
                {
                    "candidate_idx": idx,
                    "latent_cost": float(costs[idx]),
                    "latent_rank": int(latent_ranks[idx]),
                    "is_final_elite": idx in elite_indices,
                }
                for idx in range(NUM_SAMPLES)
            ],
        )
        _write_json(
            case_dir / "capture_meta.json",
            {
                **case,
                "checkpoint": args.checkpoint,
                "visual_protocol": visual_protocol,
                "formal_solver": {
                    "seed": FORMAL_SEED,
                    "batch_size": 1,
                    "num_samples": NUM_SAMPLES,
                    "n_steps": N_STEPS,
                    "topk": TOPK,
                    "horizon": HORIZON,
                    "action_block": ACTION_BLOCK,
                },
                "note": (
                    "final_mean is the CEM action selected for execution; it is "
                    "the updated elite mean and is not necessarily one of the 300 "
                    "sampled candidates"
                ),
            },
        )

    _write_json(
        output / "manifest.json",
        {
            "dataset": dataset_identity,
            "formal_rows": formal_rows,
            "audit_cases": manifest_details,
            "visual_protocol": visual_protocol,
            "capture_complete": True,
        },
    )
    print(f"captured {len(manifest_details)} cases in {output}")


@dataclass
class MujocoSnapshot:
    spec: int
    state: np.ndarray
    elapsed_steps: int | None
    reset_next_step: bool | None
    success: bool | None
    prev_qpos: np.ndarray | None
    prev_qvel: np.ndarray | None
    rng_state: dict[str, Any] | None


def _raw_env(env: Any) -> Any:
    return env.unwrapped


def _take_snapshot(env: Any) -> MujocoSnapshot:
    """Capture all MuJoCo integration state plus Python-side episode state."""

    import mujoco

    raw = _raw_env(env)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(raw._model, spec), dtype=np.float64)
    mujoco.mj_getState(raw._model, raw._data, state, spec)
    rng_state = None
    if getattr(raw, "np_random", None) is not None:
        rng_state = copy.deepcopy(raw.np_random.bit_generator.state)
    return MujocoSnapshot(
        spec=spec,
        state=state,
        elapsed_steps=getattr(env, "_elapsed_steps", None),
        reset_next_step=getattr(raw, "_reset_next_step", None),
        success=getattr(raw, "_success", None),
        prev_qpos=(
            np.asarray(raw._prev_qpos).copy()
            if hasattr(raw, "_prev_qpos")
            else None
        ),
        prev_qvel=(
            np.asarray(raw._prev_qvel).copy()
            if hasattr(raw, "_prev_qvel")
            else None
        ),
        rng_state=rng_state,
    )


def _restore_snapshot(env: Any, snapshot: MujocoSnapshot) -> None:
    import mujoco

    raw = _raw_env(env)
    expected = mujoco.mj_stateSize(raw._model, snapshot.spec)
    if snapshot.state.size != expected:
        raise RuntimeError(
            f"MuJoCo snapshot size changed: {snapshot.state.size} != {expected}"
        )
    mujoco.mj_setState(
        raw._model, raw._data, snapshot.state.copy(), snapshot.spec
    )
    mujoco.mj_forward(raw._model, raw._data)
    if snapshot.elapsed_steps is not None and hasattr(env, "_elapsed_steps"):
        env._elapsed_steps = snapshot.elapsed_steps
    if snapshot.reset_next_step is not None:
        raw._reset_next_step = snapshot.reset_next_step
    if snapshot.success is not None:
        raw._success = snapshot.success
    if snapshot.prev_qpos is not None:
        raw._prev_qpos = snapshot.prev_qpos.copy()
    if snapshot.prev_qvel is not None:
        raw._prev_qvel = snapshot.prev_qvel.copy()
    if snapshot.rng_state is not None:
        raw.np_random.bit_generator.state = copy.deepcopy(snapshot.rng_state)


def _save_snapshot(path: Path, snapshot: MujocoSnapshot) -> None:
    np.savez_compressed(
        path,
        spec=np.asarray(snapshot.spec, dtype=np.int64),
        state=snapshot.state,
        elapsed_steps=np.asarray(
            -1 if snapshot.elapsed_steps is None else snapshot.elapsed_steps
        ),
        reset_next_step=np.asarray(snapshot.reset_next_step),
        success=np.asarray(snapshot.success),
        prev_qpos=(
            np.asarray([], dtype=np.float64)
            if snapshot.prev_qpos is None
            else snapshot.prev_qpos
        ),
        prev_qvel=(
            np.asarray([], dtype=np.float64)
            if snapshot.prev_qvel is None
            else snapshot.prev_qvel
        ),
        rng_state_json=np.asarray(json.dumps(_jsonable(snapshot.rng_state))),
    )


def _make_replay_env() -> Any:
    _configure_storage()
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401 - registers swm environments

    return gym.make(
        "swm/OGBCube-v0",
        max_episode_steps=1000,
        render_mode="rgb_array",
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        # Full 25-step branch audit. We record first success separately instead
        # of terminating and accidentally resetting on the next action.
        terminate_at_goal=False,
    )


def _inverse_scale(
    normalized: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    # Match WorldModelPolicy: solver actions are cast to float32 before the
    # fitted StandardScaler inverse transform is applied.
    flat = np.asarray(normalized, dtype=np.float32).reshape(-1, ACTION_DIM).copy()
    flat *= np.asarray(scale)[None, :]
    flat += np.asarray(mean)[None, :]
    return flat


def _extract_value(info: dict[str, Any], names: Sequence[str]) -> float:
    for name in names:
        if name in info:
            value = np.asarray(info[name]).reshape(-1)
            if value.size:
                return float(value[0])
    return float("nan")


def _branch_rollout(
    env: Any,
    snapshot: MujocoSnapshot,
    actions: np.ndarray,
    goal_position: np.ndarray,
    collect_frames: bool = False,
    stop_on_success: bool = False,
) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray]]:
    _restore_snapshot(env, snapshot)
    raw = _raw_env(env)
    start_pos = raw._data.joint("object_joint_0").qpos[:3].copy()
    distances: list[float] = []
    frames = [np.asarray(env.render()).copy()] if collect_frames else []
    first_success_step: int | None = None
    last_info: dict[str, Any] = {}
    terminated = truncated = False

    for step_idx, action in enumerate(actions, start=1):
        _, _, terminated, truncated, last_info = env.step(action)
        cube_pos = raw._data.joint("object_joint_0").qpos[:3].copy()
        distance = float(np.linalg.norm(cube_pos - goal_position))
        distances.append(distance)
        if distance <= SUCCESS_THRESHOLD_METERS and first_success_step is None:
            first_success_step = step_idx
        if collect_frames:
            frames.append(np.asarray(env.render()).copy())
        if truncated or (stop_on_success and first_success_step is not None):
            break

    cube_pos = raw._data.joint("object_joint_0").qpos[:3].copy()
    terminal = np.asarray(env.render()).copy()
    if not distances:
        distances = [float(np.linalg.norm(start_pos - goal_position))]
    result = {
        "executed_steps": len(distances),
        "initial_goal_distance_m": float(np.linalg.norm(start_pos - goal_position)),
        "min_goal_distance_m": float(np.min(distances)),
        "final_goal_distance_m": float(distances[-1]),
        "final_success": bool(distances[-1] <= SUCCESS_THRESHOLD_METERS),
        "ever_success": first_success_step is not None,
        "first_success_step": first_success_step,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "cube_displacement_m": float(np.linalg.norm(cube_pos - start_pos)),
        "terminal_cube_x": float(cube_pos[0]),
        "terminal_cube_y": float(cube_pos[1]),
        "terminal_cube_z": float(cube_pos[2]),
        "gripper_contact": _extract_value(
            last_info,
            ("proprio/gripper_contact", "proprio_gripper_contact"),
        ),
        "gripper_opening": _extract_value(
            last_info,
            ("proprio/gripper_opening", "proprio_gripper_opening"),
        ),
    }
    return result, terminal, frames


def _rank(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values if ascending else -values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(x)[valid], np.asarray(y)[valid]).statistic)


def _save_image(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        path, quality=90, optimize=True
    )


def _make_contact_sheet(
    path: Path,
    images: Sequence[np.ndarray],
    rows: Sequence[dict[str, Any]],
    order: np.ndarray,
    title: str,
    columns: int = 15,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    thumb_w, thumb_h, label_h = 112, 112, 35
    count = len(order)
    n_rows = (count + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * thumb_w, 40 + n_rows * (thumb_h + label_h)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((8, 8), title, fill="black", font=title_font)
    for slot, candidate_idx in enumerate(order):
        result = rows[int(candidate_idx)]
        image = Image.fromarray(np.asarray(images[int(candidate_idx)], dtype=np.uint8))
        image.thumbnail((thumb_w, thumb_h))
        x = (slot % columns) * thumb_w
        y = 40 + (slot // columns) * (thumb_h + label_h)
        canvas.paste(image, (x, y))
        label = (
            f"#{candidate_idx} L{result['latent_rank']} P{result['physical_rank']}\n"
            f"c={result['latent_cost']:.3g} d={result['min_goal_distance_m']:.3f}"
        )
        draw.text((x + 2, y + thumb_h), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def _video_selection(
    spec: str, rows: Sequence[dict[str, Any]]
) -> list[tuple[str, int]]:
    """Return ``(label, candidate_idx)``; -1 denotes the CEM mean."""

    if not spec:
        return []
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if tokens == ["auto"]:
        tokens = [
            "mean",
            "latent:0",
            "latent:9",
            "latent:149",
            "latent:299",
            "physical:0",
        ]
    selected: list[tuple[str, int]] = []
    for token in tokens:
        if token == "all":
            selected.extend(
                (f"candidate_{idx:03d}", idx) for idx in range(NUM_SAMPLES)
            )
        elif token in {"mean", "cem_mean"}:
            selected.append(("cem_mean", -1))
        elif token.startswith("latent:"):
            rank = int(token.split(":", 1)[1])
            match = next(row for row in rows if row["latent_rank"] == rank)
            selected.append((f"latent_rank_{rank}", int(match["candidate_idx"])))
        elif token.startswith("physical:"):
            rank = int(token.split(":", 1)[1])
            match = next(row for row in rows if row["physical_rank"] == rank)
            selected.append((f"physical_rank_{rank}", int(match["candidate_idx"])))
        else:
            idx = int(token)
            if idx < 0 or idx >= NUM_SAMPLES:
                raise ValueError(f"candidate index out of range: {idx}")
            selected.append((f"candidate_{idx:03d}", idx))
    # Stable de-duplication.
    return list(dict.fromkeys(selected))


def _setup_case_env(env: Any, data: Any, cube_color: str = "dataset") -> MujocoSnapshot:
    env.reset(seed=FORMAL_SEED)
    raw = _raw_env(env)
    raw.set_state(data["initial_qpos"], data["initial_qvel"])
    raw.set_target_pos(
        cube_id=0,
        target_pos=data["goal_position"],
        target_quat=data["goal_quaternion"],
    )
    if cube_color != "dataset":
        _set_cube_rgb(raw, CUBE_RGB[cube_color])
    if hasattr(raw, "_prev_qpos"):
        raw._prev_qpos = data["initial_prev_qpos"].copy()
    if hasattr(raw, "_prev_qvel"):
        raw._prev_qvel = data["initial_prev_qvel"].copy()
    import mujoco

    mujoco.mj_forward(raw._model, raw._data)
    return _take_snapshot(env)


def _replay_case(
    case_dir: Path,
    env: Any,
    save_terminal_images: bool,
    video_spec: str,
    fps: int,
    stop_on_success: bool,
    cube_color: str = "dataset",
    visual_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = np.load(case_dir / "population.npz", allow_pickle=False)
    candidates = np.asarray(data["candidates_normalized"])
    latent_costs = np.asarray(data["latent_costs"])
    final_mean = np.asarray(data["final_mean_normalized"])
    mean_cost = float(np.asarray(data["final_mean_cost_posthoc"]))
    scale_mean = np.asarray(data["action_scaler_mean"])
    scale_scale = np.asarray(data["action_scaler_scale"])
    goal_position = np.asarray(data["goal_position"])
    if candidates.shape != (NUM_SAMPLES, HORIZON, ACTION_BLOCK * ACTION_DIM):
        raise RuntimeError(
            f"unexpected candidate shape in {case_dir}: {candidates.shape}"
        )

    snapshot = _setup_case_env(env, data, cube_color)
    _save_snapshot(case_dir / "mujoco_snapshot.npz", snapshot)
    derived_paths = (
        case_dir / "terminal_images",
        case_dir / "candidate_videos",
        case_dir / "contact_sheet_by_candidate.jpg",
        case_dir / "contact_sheet_by_latent_cost.jpg",
        case_dir / "video_index.csv",
    )
    for path in derived_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    initial_reference = np.asarray(env.render()).copy()
    _save_image(case_dir / "initial_reference.jpg", initial_reference)
    _save_image(case_dir / "goal_reference.jpg", np.asarray(data["goal_pixels"]))
    terminal_images: list[np.ndarray] = []
    results: list[dict[str, Any]] = []
    for candidate_idx in range(NUM_SAMPLES):
        actions = _inverse_scale(candidates[candidate_idx], scale_mean, scale_scale)
        outcome, terminal, _ = _branch_rollout(
            env,
            snapshot,
            actions,
            goal_position,
            stop_on_success=stop_on_success,
        )
        terminal_images.append(terminal)
        results.append(
            {
                "candidate_idx": candidate_idx,
                "latent_cost": float(latent_costs[candidate_idx]),
                **outcome,
            }
        )

    latent_ranks = _rank(latent_costs)
    physical_ranks = _rank(
        np.asarray([row["min_goal_distance_m"] for row in results])
    )
    endpoint_ranks = _rank(
        np.asarray([row["final_goal_distance_m"] for row in results])
    )
    for idx, row in enumerate(results):
        # Insert rankings next to cost/index for readable CSVs and contact sheets.
        row["latent_rank"] = int(latent_ranks[idx])
        row["physical_rank"] = int(physical_ranks[idx])
        row["endpoint_rank"] = int(endpoint_ranks[idx])

    mean_actions = _inverse_scale(final_mean, scale_mean, scale_scale)
    mean_outcome, mean_terminal, _ = _branch_rollout(
        env,
        snapshot,
        mean_actions,
        goal_position,
        stop_on_success=stop_on_success,
    )
    mean_result = {
        "candidate_idx": "cem_mean",
        "latent_cost": mean_cost,
        "latent_rank_among_samples": int(np.sum(latent_costs < mean_cost)),
        "physical_rank_among_samples": int(
            np.sum(
                np.asarray([r["min_goal_distance_m"] for r in results])
                < mean_outcome["min_goal_distance_m"]
            )
        ),
        "endpoint_rank_among_samples": int(
            np.sum(
                np.asarray([r["final_goal_distance_m"] for r in results])
                < mean_outcome["final_goal_distance_m"]
            )
        ),
        **mean_outcome,
    }

    _write_csv(case_dir / "candidate_outcomes.csv", results)
    _write_json(case_dir / "cem_mean_outcome.json", mean_result)
    np.savez_compressed(
        case_dir / "physical_outcomes.npz",
        terminal_images=np.stack(terminal_images),
        latent_costs=latent_costs,
        latent_ranks=latent_ranks,
        min_goal_distance_m=np.asarray(
            [row["min_goal_distance_m"] for row in results]
        ),
        final_goal_distance_m=np.asarray(
            [row["final_goal_distance_m"] for row in results]
        ),
        ever_success=np.asarray([row["ever_success"] for row in results]),
        terminal_cube_position=np.asarray(
            [
                [
                    row["terminal_cube_x"],
                    row["terminal_cube_y"],
                    row["terminal_cube_z"],
                ]
                for row in results
            ]
        ),
        cem_mean_terminal_image=mean_terminal,
    )

    if save_terminal_images:
        image_dir = case_dir / "terminal_images"
        for idx, image in enumerate(terminal_images):
            _save_image(image_dir / f"candidate_{idx:03d}.jpg", image)
        _save_image(image_dir / "cem_mean.jpg", mean_terminal)

    _make_contact_sheet(
        case_dir / "contact_sheet_by_candidate.jpg",
        terminal_images,
        results,
        np.arange(NUM_SAMPLES),
        "Cube final CEM population — candidate order",
    )
    _make_contact_sheet(
        case_dir / "contact_sheet_by_latent_cost.jpg",
        terminal_images,
        results,
        np.argsort(latent_costs, kind="stable"),
        "Cube final CEM population — best latent cost first",
    )

    selected_videos = _video_selection(video_spec, results)
    blind_rows: list[dict[str, Any]] = []
    for label, candidate_idx in selected_videos:
        normalized = final_mean if candidate_idx == -1 else candidates[candidate_idx]
        actions = _inverse_scale(normalized, scale_mean, scale_scale)
        _, _, frames = _branch_rollout(
            env,
            snapshot,
            actions,
            goal_position,
            collect_frames=True,
            stop_on_success=stop_on_success,
        )
        video_path = case_dir / "candidate_videos" / f"{label}.mp4"
        _save_video(video_path, frames, fps)
        blind_rows.append(
            {
                "video_file": str(video_path.relative_to(case_dir)),
                "candidate_idx": "cem_mean" if candidate_idx == -1 else candidate_idx,
            }
        )
    if blind_rows:
        _write_csv(case_dir / "video_index.csv", blind_rows)

    min_dist = np.asarray([row["min_goal_distance_m"] for row in results])
    final_dist = np.asarray([row["final_goal_distance_m"] for row in results])
    summary = {
        "case_dir": str(case_dir),
        "spearman_latent_vs_min_distance": _spearman(latent_costs, min_dist),
        "spearman_latent_vs_final_distance": _spearman(latent_costs, final_dist),
        "latent_best_candidate": int(np.argmin(latent_costs)),
        "latent_best_physical_rank": int(physical_ranks[np.argmin(latent_costs)]),
        "physical_best_candidate": int(np.argmin(min_dist)),
        "physical_best_latent_rank": int(latent_ranks[np.argmin(min_dist)]),
        "sample_success_count": int(
            sum(bool(row["ever_success"]) for row in results)
        ),
        "cem_mean": mean_result,
        "video_selection": selected_videos,
        "stop_on_success": stop_on_success,
        "visual_protocol": visual_protocol or _visual_protocol(
            cube_color, "matched"
        ),
    }
    _write_json(case_dir / "audit_summary.json", summary)
    return summary


def command_replay(args: argparse.Namespace) -> None:
    _configure_storage()
    _refuse_active_training()
    requested_color = getattr(args, "cube_color", None)
    requested_goal = getattr(args, "goal_protocol", None)
    input_path = args.input or _default_output(
        requested_color or "dataset", requested_goal or "matched"
    )
    root = _validate_audit_input(input_path)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"capture manifest missing: {root / 'manifest.json'}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    captured_color = manifest.get("visual_protocol", {}).get(
        "cube_color", "dataset"
    )
    visual_protocol = manifest.get("visual_protocol") or _visual_protocol(
        "dataset", "matched"
    )
    captured_goal = visual_protocol.get("goal_type", "matched")
    if captured_color not in COLOR_OUTPUTS:
        raise ValueError(f"unknown cube color in capture manifest: {captured_color}")
    if requested_color is not None and requested_color != captured_color:
        raise ValueError(
            f"--cube-color {requested_color} does not match captured "
            f"manifest color {captured_color}"
        )
    if requested_goal is not None and requested_goal != captured_goal:
        raise ValueError(
            f"--goal-protocol {requested_goal} does not match captured "
            f"manifest protocol {captured_goal}"
        )
    if captured_goal not in {"matched", "recolor"}:
        raise ValueError(f"unknown goal protocol in capture manifest: {captured_goal}")
    if captured_goal == "recolor":
        meta = visual_protocol.get("goal_recolor", {})
        required = {"source", "sha256", "shape", "dtype", "index_contract"}
        if not required.issubset(meta):
            raise ValueError("recolor capture manifest is missing goal provenance")
    cube_color = captured_color
    case_dirs = sorted(path for path in root.glob("env_*_row_*") if path.is_dir())
    if args.env_indices:
        wanted = {int(token) for token in args.env_indices.split(",")}
        case_dirs = [
            path
            for path in case_dirs
            if int(path.name.split("_", 2)[1]) in wanted
        ]
    if not case_dirs:
        raise RuntimeError("no captured case directories selected")

    env = _make_replay_env()
    summaries: list[dict[str, Any]] = []
    try:
        for case_dir in case_dirs:
            print(f"replaying {case_dir.name} ...", flush=True)
            summaries.append(
                _replay_case(
                    case_dir,
                    env,
                    save_terminal_images=not args.no_terminal_images,
                    video_spec=args.videos,
                    fps=args.fps,
                    stop_on_success=args.stop_on_success,
                    cube_color=cube_color,
                    visual_protocol=visual_protocol,
                )
            )
    finally:
        env.close()

    completed_summaries: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.glob("env_*_row_*") if path.is_dir()):
        summary_path = case_dir / "audit_summary.json"
        if summary_path.exists():
            completed_summaries.append(
                json.loads(summary_path.read_text(encoding="utf-8"))
            )
    aggregate = {
        "num_cases": len(completed_summaries),
        "visual_protocol": visual_protocol,
        "mean_spearman_latent_vs_min_distance": float(
            np.nanmean(
                [
                    item["spearman_latent_vs_min_distance"]
                    for item in completed_summaries
                ]
            )
        ),
        "mean_spearman_latent_vs_final_distance": float(
            np.nanmean(
                [
                    item["spearman_latent_vs_final_distance"]
                    for item in completed_summaries
                ]
            )
        ),
        "cases": completed_summaries,
    }
    _write_json(root / "aggregate_summary.json", aggregate)
    print(
        f"replayed {len(summaries)} selected cases; "
        f"{len(completed_summaries)} total completed in {root}"
    )


def command_all(args: argparse.Namespace) -> None:
    capture_args = argparse.Namespace(**vars(args))
    capture_args.output = args.output
    command_capture(capture_args)
    replay_args = argparse.Namespace(
        input=capture_args.output,
        cube_color=args.cube_color,
        goal_protocol=args.goal_protocol,
        env_indices="",
        no_terminal_images=args.no_terminal_images,
        videos=args.videos,
        fps=args.fps,
        stop_on_success=args.stop_on_success,
    )
    command_replay(replay_args)


def _add_capture_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="Cube HDF5 dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "audit output root; defaults to cube_cem_300 for dataset color or "
            "cube_cem_300_{blue,yellow}_{matched,v2} for color OOD"
        ),
    )
    parser.add_argument(
        "--cube-color",
        choices=("dataset", "blue", "yellow"),
        default="dataset",
        help=(
            "dataset keeps the formal red HDF5 inputs; blue/yellow use exact "
            "matched RGB [0,0,1]/[1,1,0] renders"
        ),
    )
    parser.add_argument(
        "--goal-protocol",
        choices=("matched", "recolor"),
        default="matched",
        help=(
            "matched renders a current-arm/cube-only synthetic goal; recolor "
            "loads the frozen HDF5-future-frame cube-only recolor array (v2)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="quentinll/lewm-cube",
        help="local checkpoint name/path or Hugging Face repo id",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing audit root"
    )


def _add_replay_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-terminal-images",
        action="store_true",
        help="keep terminal images only in compressed NPZ (contact sheets still saved)",
    )
    parser.add_argument(
        "--videos",
        default="all,mean",
        help=(
            "comma list: all, auto, mean, latent:RANK, physical:RANK, or candidate "
            "index; default all,mean writes 300 candidate videos plus the CEM mean "
            "per case"
        ),
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="stop a branch at first <=4cm success instead of executing all 25 actions",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit LeWM's final 300 CEM candidates on OGBench-Cube"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="validate/print the fixed 12 cases")
    manifest.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    manifest.add_argument("--output", type=Path)
    manifest.set_defaults(func=command_manifest)

    capture = sub.add_parser(
        "capture", help="GPU: capture first-plan final population for 12 cases"
    )
    _add_capture_options(capture)
    capture.set_defaults(func=command_capture)

    replay = sub.add_parser(
        "replay", help="CPU/MuJoCo: execute captured candidates from restored state"
    )
    replay.add_argument(
        "--input",
        type=Path,
        default=None,
        help="capture root; defaults from --cube-color (dataset when omitted)",
    )
    replay.add_argument(
        "--cube-color",
        choices=("dataset", "blue", "yellow"),
        default=None,
        help="optional assertion; normally inferred from capture manifest",
    )
    replay.add_argument(
        "--goal-protocol",
        choices=("matched", "recolor"),
        default=None,
        help="optional assertion; normally inferred from capture manifest",
    )
    replay.add_argument(
        "--env-indices", default="", help="optional comma list, e.g. 0,2,49"
    )
    _add_replay_options(replay)
    replay.set_defaults(func=command_replay)

    all_cmd = sub.add_parser("all", help="run capture followed by physical replay")
    _add_capture_options(all_cmd)
    _add_replay_options(all_cmd)
    all_cmd.set_defaults(func=command_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        if os.environ.get("CUBE_CEM_AUDIT_TRACEBACK") == "1":
            traceback.print_exc()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
