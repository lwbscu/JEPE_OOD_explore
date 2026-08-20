#!/usr/bin/env python3
"""Collect planner-in-the-loop Cube data for off-policy V2 training.

The collection is deliberately split in two phases:

``capture`` (CUDA, no environment stepping)
    Selects one state from each of 150 distinct expert episodes, globally
    excludes the formal 50 and the frozen measurement-one holdout episodes
    from both source selection and trajectory-memory retrieval, and runs the
    exact MaskedAug T2 CEM solve.  All ten post-injection 300-candidate pools,
    costs, elites, pre/post distribution parameters, and RNG provenance are
    written atomically to ``planner_trace.h5``.

``replay`` (CPU/MuJoCo only)
    Restores the same source snapshot for every branch and physically replays
    candidate 0 plus candidate indices 31..50 from every CEM iteration.  This
    is 150 * 10 * 21 = 31,500 fixed 25-env-step rollouts.  No extra action
    clipping is applied.  Six model-rate JPEG frames and privileged poses are
    stored in one atomic HDF5 shard per source state, making replay resumable.

The 20-rollout smoke uses one source state and candidate indices 0 and 31 in
each of ten iterations.  It exercises the exact capture path and output schema
without silently changing the CEM population size or iteration count.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
LEWM_ROOT = PROJECT_ROOT / "le-wm"
for _path in (TOOLS_ROOT, LEWM_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_cube_memory_index as memory  # noqa: E402
import collect_cube_offpolicy as v1  # noqa: E402
import cube_trust_region_common as trust  # noqa: E402
import eval_memory_seed as memory_eval  # noqa: E402
import eval_ood_color as ood  # noqa: E402


DATASET = PROJECT_ROOT / "datasets/ogbench/cube_single_expert.h5"
INDEX_ROOT = PROJECT_ROOT / "outputs/memory_index/cube_expert_v1"
FORMAL_MANIFEST = PROJECT_ROOT / "outputs/audit/cube_cem_manifest.json"
MEASUREMENT1_SEGMENTS = (
    PROJECT_ROOT / "outputs/eval/cube/imagination_error/measurement1_segments.json"
)
MASKED_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt"
)
V1_ROOT = PROJECT_ROOT / "datasets/offpolicy_cube_v1"
OUTPUT_ROOT = PROJECT_ROOT / "datasets/offpolicy_cube_v2"
SMOKE_ROOT = OUTPUT_ROOT / "smoke_20"
TMP_ROOT = PROJECT_ROOT.parent / "tmp"

FORMAT_VERSION = "cube_offpolicy_planner_rollout_hdf5_v2"
TRACE_FORMAT_VERSION = "cube_offpolicy_planner_trace_hdf5_v2"
SELECTION_FORMAT_VERSION = "cube_offpolicy_planner_selection_v2"
SEED = 2_026_081_6
FORMAL_STATES = 150
SMOKE_STATES = 1
N_STEPS = 10
NUM_SAMPLES = 300
TOPK = 30
HORIZON = 5
ACTION_BLOCK = 5
ACTION_DIM = 5
ENV_STEPS = HORIZON * ACTION_BLOCK
FRAME_COUNT = HORIZON + 1
GOAL_OFFSET = 25
MEMORY_SLOTS = 10
FORMAL_CANDIDATE_INDICES = np.asarray([0, *range(31, 51)], dtype=np.int16)
SMOKE_CANDIDATE_INDICES = np.asarray([0, 31], dtype=np.int16)
FORMAL_ROLLOUTS = FORMAL_STATES * N_STEPS * len(FORMAL_CANDIDATE_INDICES)
SMOKE_ROLLOUTS = SMOKE_STATES * N_STEPS * len(SMOKE_CANDIDATE_INDICES)
JPEG_QUALITY = 95
MAX_BYTES = 40 * (1 << 30)


def _configure_storage() -> None:
    values = {
        "STABLEWM_HOME": str(PROJECT_ROOT),
        "HF_HOME": str(PROJECT_ROOT.parent / ".cache/huggingface"),
        "TORCH_HOME": str(PROJECT_ROOT.parent / ".cache/torch"),
        "PIP_CACHE_DIR": str(PROJECT_ROOT.parent / ".cache/pip"),
        "TMPDIR": str(TMP_ROOT),
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        os.environ[key] = value
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    partial.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, sha256: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    result = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if sha256:
        result["sha256"] = _sha256(resolved)
    return result


def _root(scope: str) -> Path:
    return SMOKE_ROOT if scope == "smoke" else OUTPUT_ROOT


def _safe_root(scope: str) -> Path:
    lexical = _root(scope).expanduser().absolute()
    if lexical.is_symlink():
        raise ValueError(f"refusing symlink output root: {lexical}")
    resolved = lexical.resolve()
    frozen = OUTPUT_ROOT.resolve()
    if resolved != frozen and frozen not in resolved.parents:
        raise ValueError(f"V2 output must be {frozen} or a child: {resolved}")
    if PROJECT_ROOT.parent.resolve() not in resolved.parents:
        raise ValueError(f"V2 output is not on the data disk: {resolved}")
    return resolved


def _scope_contract(scope: str) -> tuple[int, np.ndarray, int]:
    if scope == "smoke":
        return SMOKE_STATES, SMOKE_CANDIDATE_INDICES.copy(), SMOKE_ROLLOUTS
    if scope == "formal":
        return FORMAL_STATES, FORMAL_CANDIDATE_INDICES.copy(), FORMAL_ROLLOUTS
    raise ValueError(scope)


def _load_exclusions(h5: Any) -> dict[str, np.ndarray]:
    manifest = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    formal_rows = np.asarray(manifest["formal_rows"], dtype=np.int64)
    if formal_rows.shape != (50,) or len(np.unique(formal_rows)) != 50:
        raise RuntimeError("formal manifest must contain exactly 50 unique rows")
    order = np.argsort(formal_rows)
    formal_sorted = np.asarray(h5["ep_idx"][formal_rows[order]], dtype=np.int64)
    formal_episodes = formal_sorted[np.argsort(order)]
    if len(np.unique(formal_episodes)) != 50:
        raise RuntimeError("formal rows do not map to 50 unique episodes")
    segments = json.loads(MEASUREMENT1_SEGMENTS.read_text(encoding="utf-8"))
    holdout = np.unique(np.asarray(segments["episode_indices"], dtype=np.int64))
    if len(np.asarray(segments["episode_indices"])) != 2000:
        raise RuntimeError("measurement-one segment artifact no longer contains 2000 segments")
    frozen_formal = np.asarray(segments["formal_episodes_excluded"], dtype=np.int64)
    if set(map(int, frozen_formal)) != set(map(int, formal_episodes)):
        raise RuntimeError("measurement-one and formal manifests disagree on excluded episodes")
    return {
        "formal_rows": formal_rows,
        "formal_episodes": formal_episodes,
        "measurement1_holdout_episodes": holdout,
    }


def _h5_indexed(dataset: Any, rows: np.ndarray) -> np.ndarray:
    """Gather arbitrary unique HDF5 rows while respecting h5py's sort rule."""

    indices = np.asarray(rows, dtype=np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    values = np.asarray(dataset[unique])
    return values[inverse].reshape(*indices.shape, *values.shape[1:])


def _select_sources(h5: Any, count: int, seed: int) -> dict[str, np.ndarray]:
    exclusions = _load_exclusions(h5)
    excluded = set(map(int, exclusions["formal_episodes"]))
    excluded.update(map(int, exclusions["measurement1_holdout_episodes"]))
    offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
    eligible = np.asarray(
        [
            episode
            for episode, length in enumerate(lengths)
            if episode not in excluded and int(length) > GOAL_OFFSET
        ],
        dtype=np.int64,
    )
    if count > len(eligible):
        raise RuntimeError(
            f"insufficient source episodes: expected_at_least={count}, actual={len(eligible)}"
        )
    rng = np.random.default_rng(seed)
    episodes = rng.choice(eligible, size=count, replace=False).astype(np.int64)
    starts = np.asarray(
        [int(rng.integers(0, int(lengths[episode]) - GOAL_OFFSET)) for episode in episodes],
        dtype=np.int64,
    )
    rows = offsets[episodes] + starts
    goals = rows + GOAL_OFFSET
    if len(np.unique(episodes)) != count:
        raise RuntimeError("source episode sampling unexpectedly used replacement")
    if not np.array_equal(_h5_indexed(h5["ep_idx"], rows), episodes):
        raise RuntimeError("source rows do not map to selected episodes")
    if not np.array_equal(_h5_indexed(h5["ep_idx"], goals), episodes):
        raise RuntimeError("one or more +25 goal rows crosses an episode boundary")
    if not np.array_equal(_h5_indexed(h5["step_idx"], goals) - starts, np.full(count, 25)):
        raise RuntimeError("goal rows are not exactly 25 environment steps after source rows")
    global_memory_excluded = np.unique(
        np.concatenate(
            [
                exclusions["formal_episodes"],
                exclusions["measurement1_holdout_episodes"],
                episodes,
            ]
        )
    )
    return {
        **exclusions,
        "source_rows": rows,
        "source_episodes": episodes,
        "source_steps": starts.astype(np.int16),
        "goal_rows": goals,
        "eligible_source_episodes": np.asarray(len(eligible), dtype=np.int64),
        "global_memory_excluded_episodes": global_memory_excluded,
        "seed": np.asarray(seed, dtype=np.int64),
    }


def _selection_path(root: Path) -> Path:
    return root / "selection.npz"


def _create_or_load_selection(
    root: Path, scope: str, overwrite: bool = False
) -> dict[str, np.ndarray]:
    import hdf5plugin  # noqa: F401
    import h5py

    count, _, _ = _scope_contract(scope)
    root.mkdir(parents=True, exist_ok=True)
    path = _selection_path(root)
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as data:
            selected = {name: np.asarray(data[name]) for name in data.files}
    else:
        with h5py.File(DATASET, "r", swmr=True) as h5:
            selected = _select_sources(h5, count, SEED)
        _atomic_npz(
            path,
            format_version=np.asarray(SELECTION_FORMAT_VERSION),
            scope=np.asarray(scope),
            **selected,
        )
    expected = {
        "source_rows": (count,),
        "source_episodes": (count,),
        "source_steps": (count,),
        "goal_rows": (count,),
    }
    for name, shape in expected.items():
        if name not in selected or selected[name].shape != shape:
            raise RuntimeError(
                f"selection field mismatch: field={name}, expected={shape}, "
                f"actual={None if name not in selected else selected[name].shape}"
            )
    if len(np.unique(selected["source_episodes"])) != count:
        raise RuntimeError("selection does not use distinct source episodes")
    if "format_version" in selected and str(selected["format_version"].item()) != SELECTION_FORMAT_VERSION:
        raise RuntimeError("selection format version mismatch")
    if "scope" in selected and str(selected["scope"].item()) != scope:
        raise RuntimeError("selection scope mismatch")
    if int(np.asarray(selected["seed"]).item()) != SEED:
        raise RuntimeError("selection seed mismatch")
    # A resume may happen long after selection.  Rebind its leakage contract
    # to the current frozen artifacts and source HDF5 before trusting it.
    with h5py.File(DATASET, "r", swmr=True) as h5:
        current = _load_exclusions(h5)
        for name in ("formal_rows", "formal_episodes", "measurement1_holdout_episodes"):
            if not np.array_equal(np.asarray(selected[name]), current[name]):
                raise RuntimeError(f"selection exclusion artifact changed: field={name}")
        if not np.array_equal(
            _h5_indexed(h5["ep_idx"], selected["source_rows"]),
            selected["source_episodes"],
        ):
            raise RuntimeError("selection source episode binding changed")
        if not np.array_equal(
            _h5_indexed(h5["step_idx"], selected["source_rows"]),
            selected["source_steps"],
        ):
            raise RuntimeError("selection source step binding changed")
    return selected


class _PlannerCaptureProxy:
    """Inject exact T2 slots and retain each complete post-injection pool."""

    def __init__(self, base: Any, contexts: list[dict[str, Any]]) -> None:
        self.base = base
        self.contexts = contexts
        self.call_index = 0
        self.generator: Any = None
        self.records: list[dict[str, np.ndarray]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def get_cost(self, info_dict: dict[str, Any], candidates: Any) -> Any:
        env_idx = self.call_index // N_STEPS
        iteration = self.call_index % N_STEPS
        if env_idx >= len(self.contexts):
            raise RuntimeError("CEM made more cost calls than expected")
        expected = (1, NUM_SAMPLES, HORIZON, ACTION_BLOCK * ACTION_DIM)
        if tuple(candidates.shape) != expected:
            raise RuntimeError(
                f"unexpected candidate shape: expected={expected}, actual={tuple(candidates.shape)}"
            )
        context = self.contexts[env_idx]
        before = candidates[0, 1:31].detach().cpu().float().numpy().copy()
        candidate0 = candidates[0, 0].detach().cpu().float().numpy().copy()
        import torch

        candidates[:, 1:11].copy_(
            torch.as_tensor(
                context["seed_actions_normalized"],
                device=candidates.device,
                dtype=candidates.dtype,
            ).unsqueeze(0)
        )
        candidates[:, 11:31].copy_(
            torch.as_tensor(
                context["noise_bundle"]["normalized"].reshape(20, HORIZON, 25),
                device=candidates.device,
                dtype=candidates.dtype,
            ).unsqueeze(0)
        )
        if not np.array_equal(
            candidates[0, 0].detach().cpu().float().numpy(), candidate0
        ):
            raise RuntimeError("T2 injection modified candidate0")
        costs = self.base.get_cost(info_dict, candidates)
        self.records.append(
            {
                "source_index": np.asarray(env_idx, dtype=np.int16),
                "cem_iteration": np.asarray(iteration, dtype=np.int8),
                "preinjection_slots_1_30": before,
                "candidates_normalized": candidates[0].detach().cpu().float().numpy().copy(),
                "latent_costs": costs[0].detach().cpu().float().numpy().copy(),
                "rng_state_after_sampling": self.generator.get_state().cpu().numpy().copy(),
            }
        )
        self.call_index += 1
        return costs


class _DistributionRecorder:
    """Capture CEM distribution parameters before and after each update."""

    output_key = "offpolicy_v2_distribution_trace"

    def __init__(self) -> None:
        self.records: list[dict[str, np.ndarray]] = []
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        # CEM calls reset exactly once at the start of every solve.  Both the
        # array payload used by _write_trace and the public callback history
        # must therefore be solve-local rather than accumulating on reuse.
        self.records = []
        self.history = []

    def start_batch(self) -> None:
        return None

    def __call__(self, **state: Any) -> None:
        self.records.append(
            {
                "topk_indices": state["topk_inds"][0].detach().cpu().numpy().astype(np.int16),
                "topk_costs": state["topk_vals"][0].detach().cpu().float().numpy().copy(),
                "mean_pre": state["prev_mean"][0].detach().cpu().float().numpy().copy(),
                "std_pre": state["prev_var"][0].detach().cpu().float().numpy().copy(),
                "mean_post": state["mean"][0].detach().cpu().float().numpy().copy(),
                "std_post": state["var"][0].detach().cpu().float().numpy().copy(),
            }
        )

    def end_solve(self) -> None:
        # Installed CEMSolver unconditionally reads ``callback.history`` at
        # the end of solve.  Keep that public payload small and JSON-safe;
        # the complete numeric records remain in ``self.records`` and are
        # written to planner_trace.h5 by the collector.
        self.history = [
            {
                "format_version": "cube_offpolicy_v2_distribution_callback_v1",
                "num_iteration_records": int(len(self.records)),
                "all_values_finite": bool(
                    all(
                        np.isfinite(record[name]).all()
                        for record in self.records
                        for name in (
                            "topk_costs",
                            "mean_pre",
                            "std_pre",
                            "mean_post",
                            "std_post",
                        )
                    )
                ),
            }
        ]


def _retrieval_contexts(
    selected: dict[str, np.ndarray], index: memory.CubeMemoryIndex
) -> list[dict[str, Any]]:
    import hdf5plugin  # noqa: F401
    import h5py

    excluded = set(map(int, selected["global_memory_excluded_episodes"]))
    contexts: list[dict[str, Any]] = []
    with h5py.File(DATASET, "r", swmr=True) as h5:
        for source_index, row in enumerate(selected["source_rows"]):
            query = memory.feature_chunk(h5, int(row), int(row) + 1)[0]
            retrieved = v1._query_memory_top10(index, query, excluded)
            if set(map(int, retrieved["episodes"])) & excluded:
                raise RuntimeError("globally excluded episode leaked into memory retrieval")
            bundle = index.action_seed_bundle(retrieved["rows"])
            noise = trust.noisy_seed_variants(
                bundle["raw"], index.action_mean, index.action_scale, int(row), 0
            )
            contexts.append(
                {
                    "source_index": source_index,
                    "source_row": int(row),
                    "query_feature_raw": query,
                    "retrieval_rows": retrieved["rows"],
                    "retrieval_episodes": retrieved["episodes"],
                    "retrieval_steps": retrieved["steps"],
                    "retrieval_distances": retrieved["distances"],
                    "retrieval_anchor_indices": retrieved["anchor_indices"],
                    "seed_actions_raw": bundle["raw"],
                    "seed_actions_normalized": bundle["normalized"],
                    "noise_bundle": noise,
                }
            )
    return contexts


def _write_trace(
    path: Path,
    scope: str,
    selected: dict[str, np.ndarray],
    contexts: list[dict[str, Any]],
    proxy: _PlannerCaptureProxy,
    recorder: _DistributionRecorder,
    rng_initial: np.ndarray,
    rng_final: np.ndarray,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
) -> None:
    import hdf5plugin  # noqa: F401
    import h5py

    state_count, _, _ = _scope_contract(scope)
    expected_records = state_count * N_STEPS
    if len(proxy.records) != expected_records or len(recorder.records) != expected_records:
        raise RuntimeError(
            "planner trace count mismatch: "
            f"expected={expected_records}, proxy={len(proxy.records)}, callback={len(recorder.records)}"
        )
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if partial.exists():
        partial.unlink()
    compression = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    try:
        with h5py.File(partial, "w", libver="latest") as out:
            out.attrs["format_version"] = TRACE_FORMAT_VERSION
            out.attrs["scope"] = scope
            out.attrs["seed"] = SEED
            out.attrs["solver_seed"] = trust.FORMAL_SEED
            out.attrs["protocol"] = "T2 exact injection; 10 seed + 20 sigma0.1 + 270 free"
            out.attrs["action_normalizer_mean"] = action_mean
            out.attrs["action_normalizer_scale"] = action_scale
            out.create_dataset("rng_state_initial", data=rng_initial)
            out.create_dataset("rng_state_final", data=rng_final)
            for name in (
                "source_rows",
                "source_episodes",
                "source_steps",
                "goal_rows",
                "global_memory_excluded_episodes",
            ):
                out.create_dataset(name, data=selected[name], **compression)
            context_fields = (
                "query_feature_raw",
                "retrieval_rows",
                "retrieval_episodes",
                "retrieval_steps",
                "retrieval_distances",
                "retrieval_anchor_indices",
                "seed_actions_raw",
                "seed_actions_normalized",
            )
            for name in context_fields:
                out.create_dataset(name, data=np.stack([item[name] for item in contexts]), **compression)
            noise_fields = {
                "noise_parent_seed_indices": "parent_indices",
                "noise_derived_seed": "derived_seed",
                "noise_seed_components": "seed_components",
                "noise_values_raw": "noise_raw",
                "noise_unclipped_actions_raw": "unclipped_raw",
                "noise_clipped_actions_raw": "clipped_raw",
                "noise_candidates_normalized": "normalized",
                "noise_clip_mask": "clip_mask",
            }
            for target, source in noise_fields.items():
                out.create_dataset(
                    target,
                    data=np.stack([item["noise_bundle"][source] for item in contexts]),
                    **compression,
                )
            for name in (
                "source_index",
                "cem_iteration",
                "preinjection_slots_1_30",
                "candidates_normalized",
                "latent_costs",
                "rng_state_after_sampling",
            ):
                values = np.stack([item[name] for item in proxy.records])
                values = values.reshape(state_count, N_STEPS, *values.shape[1:])
                out.create_dataset(name, data=values, **compression)
            for name in (
                "topk_indices",
                "topk_costs",
                "mean_pre",
                "std_pre",
                "mean_post",
                "std_post",
            ):
                values = np.stack([item[name] for item in recorder.records])
                values = values.reshape(state_count, N_STEPS, *values.shape[1:])
                out.create_dataset(name, data=values, **compression)
            # The installed solver calls this sampling scale ``var`` even
            # though it multiplies randn directly and updates it with std().
            # Retain explicit aliases so downstream audits can match either
            # the library name or the mathematically accurate name.
            out["cem_var_pre"] = out["std_pre"]
            out["cem_var_post"] = out["std_post"]
            out.flush()
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _capture(args: argparse.Namespace) -> None:
    _configure_storage()
    if args.scope == "formal" and not args.authorize_model_capture:
        raise PermissionError("formal 150-state CUDA capture requires --authorize-model-capture")
    root = _safe_root(args.scope)
    if args.overwrite:
        for path in (
            root / "selection.npz",
            root / "planner_trace.h5",
            root / "capture_manifest.json",
            root / "manifest.json",
            root / "progress.json",
        ):
            if path.exists():
                path.unlink()
        if (root / "shards").exists():
            shutil.rmtree(root / "shards")
    root.mkdir(parents=True, exist_ok=True)
    trace_path = root / "planner_trace.h5"
    if trace_path.exists():
        raise FileExistsError(f"planner capture already exists: {trace_path}")
    selected = _create_or_load_selection(root, args.scope, overwrite=False)

    import hdf5plugin  # noqa: F401
    import h5py
    import stable_worldmodel as swm
    import torch
    from gymnasium.spaces import Box

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("exact planner capture requires CUDA")
    checkpoint_contract = trust.frozen_masked_checkpoint_contract()
    index = memory.CubeMemoryIndex(INDEX_ROOT, DATASET)
    contexts = _retrieval_contexts(selected, index)
    with h5py.File(DATASET, "r", swmr=True) as h5:
        rows = selected["source_rows"]
        goals = selected["goal_rows"]
        raw_inputs = {
            "pixels": _h5_indexed(h5["pixels"], rows)[:, None, ...],
            "goal": _h5_indexed(h5["pixels"], goals)[:, None, ...],
            "action": _h5_indexed(h5["action"], rows)[:, None, ...],
        }
    model = swm.wm.utils.load_pretrained(str(MASKED_CHECKPOINT), cache_dir=str(PROJECT_ROOT))
    model = model.to(args.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    scaler = memory_eval._standard_scaler(index)
    proxy = _PlannerCaptureProxy(model, contexts)
    recorder = _DistributionRecorder()
    solver = swm.solver.CEMSolver(
        model=proxy,
        batch_size=1,
        num_samples=NUM_SAMPLES,
        var_scale=1.0,
        n_steps=N_STEPS,
        topk=TOPK,
        device=args.device,
        seed=trust.FORMAL_SEED,
        callbacks=[recorder],
    )
    proxy.generator = solver.torch_gen
    state_count, candidate_indices, rollout_count = _scope_contract(args.scope)
    config = swm.PlanConfig(
        horizon=HORIZON, receding_horizon=HORIZON, action_block=ACTION_BLOCK
    )
    solver.configure(
        action_space=Box(
            low=np.broadcast_to(-np.inf, (state_count, ACTION_DIM)),
            high=np.broadcast_to(np.inf, (state_count, ACTION_DIM)),
            dtype=np.float32,
        ),
        n_envs=state_count,
        config=config,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process={"action": scaler},
        transform={"pixels": ood._image_transform(224), "goal": ood._image_transform(224)},
    )
    prepared = policy._prepare_info(raw_inputs)
    rng_initial = solver.torch_gen.get_state().cpu().numpy().copy()
    started = time.time()
    with torch.inference_mode():
        solver(prepared, init_action=None)
    elapsed = time.time() - started
    rng_final = solver.torch_gen.get_state().cpu().numpy().copy()
    _write_trace(
        trace_path,
        args.scope,
        selected,
        contexts,
        proxy,
        recorder,
        rng_initial,
        rng_final,
        index.action_mean,
        index.action_scale,
    )
    payload = {
        "format_version": TRACE_FORMAT_VERSION,
        "scope": args.scope,
        "complete": True,
        "num_source_states": state_count,
        "num_cem_iterations": N_STEPS,
        "num_candidates_per_iteration": NUM_SAMPLES,
        "captured_postinjection_candidates": state_count * N_STEPS * NUM_SAMPLES,
        "physics_candidate_indices": candidate_indices,
        "planned_physics_rollouts": rollout_count,
        "protocol": {
            "id": "T2",
            "initial_mean": "zero",
            "initial_std": 1.0,
            "memory_slots": list(range(1, 11)),
            "noise_slots": list(range(11, 31)),
            "free_replay_slots": list(map(int, candidate_indices[candidate_indices != 0])),
            "candidate0": "pre-update current mean",
            "n_steps": N_STEPS,
            "topk": TOPK,
            "solver_seed": trust.FORMAL_SEED,
            "torch_rng_stream": "one exact legacy generator across source selection order",
            "optional_second_planning_cycle": (
                "not collected; V2 formal budget is the frozen 31,500 first-cycle rollouts"
            ),
        },
        "selection": _identity(root / "selection.npz"),
        "planner_trace": _identity(trace_path),
        "checkpoint": checkpoint_contract,
        "memory_index": _identity(INDEX_ROOT / "metadata.json"),
        "elapsed_seconds": elapsed,
    }
    _atomic_json(root / "capture_manifest.json", payload)
    print(json.dumps(_jsonable(payload), sort_keys=True))


def _target_quaternion_from_dataset(h5: Any, row: int) -> np.ndarray:
    if "privileged_block_0_quat" in h5:
        return np.asarray(h5["privileged_block_0_quat"][row], dtype=np.float64)
    yaw = float(np.asarray(h5["privileged_block_0_yaw"][row]).reshape(-1)[0])
    return v1._target_quaternion(yaw)


def _setup_snapshot(env: Any, h5: Any, source_row: int, goal_row: int, seed: int) -> Any:
    import cube_cem_audit as audit
    import mujoco

    env.reset(seed=int(seed))
    raw = env.unwrapped
    raw.set_state(
        np.asarray(h5["qpos"][source_row], dtype=np.float64),
        np.asarray(h5["qvel"][source_row], dtype=np.float64),
    )
    raw.set_target_pos(
        0,
        np.asarray(h5["privileged_block_0_pos"][goal_row], dtype=np.float64),
        _target_quaternion_from_dataset(h5, goal_row),
    )
    if hasattr(raw, "_prev_qpos"):
        raw._prev_qpos = np.asarray(h5["prev_qpos"][source_row], dtype=np.float64).copy()
    if hasattr(raw, "_prev_qvel"):
        raw._prev_qvel = np.asarray(h5["prev_qvel"][source_row], dtype=np.float64).copy()
    mujoco.mj_forward(raw._model, raw._data)
    snapshot = audit._take_snapshot(env)
    audit._restore_snapshot(env, snapshot)
    return snapshot


def _inverse_action(normalized: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    value = np.asarray(normalized, dtype=np.float32).reshape(ENV_STEPS, ACTION_DIM).copy()
    value *= np.asarray(scale, dtype=np.float64)
    value += np.asarray(mean, dtype=np.float64)
    if not np.isfinite(value).all():
        raise RuntimeError("inverse-scaled planner action contains nonfinite values")
    return value


def _empty_batch(count: int) -> dict[str, Any]:
    return {
        "pixels_jpeg": [[None] * FRAME_COUNT for _ in range(count)],
        "action_env": np.empty((count, ENV_STEPS, ACTION_DIM), dtype=np.float32),
        "action_model": np.empty((count, HORIZON, ACTION_BLOCK * ACTION_DIM), dtype=np.float32),
        "block_pos": np.empty((count, FRAME_COUNT, 3), dtype=np.float32),
        "block_quat": np.empty((count, FRAME_COUNT, 4), dtype=np.float32),
        "block_yaw": np.empty((count, FRAME_COUNT), dtype=np.float32),
        "terminated": np.empty((count, ENV_STEPS), dtype=bool),
        "truncated": np.empty((count, ENV_STEPS), dtype=bool),
        "latent_cost": np.empty(count, dtype=np.float32),
        "topk_rank": np.full(count, -1, dtype=np.int8),
        "initial_qpos": np.empty((count, 21), dtype=np.float64),
        "initial_qvel": np.empty((count, 20), dtype=np.float64),
        "initial_prev_qpos": np.empty((count, 21), dtype=np.float64),
        "initial_prev_qvel": np.empty((count, 20), dtype=np.float64),
        "goal_pos": np.empty((count, 3), dtype=np.float64),
        "goal_quat": np.empty((count, 4), dtype=np.float64),
        "source_h5_pixel_mae": np.empty(count, dtype=np.float32),
    }


_WORKER: dict[str, Any] = {}


def _worker_close() -> None:
    for key in ("env", "trace", "h5"):
        value = _WORKER.get(key)
        if value is not None:
            value.close()
    _WORKER.clear()


def _worker_init(trace_path: str) -> None:
    import atexit
    import hdf5plugin  # noqa: F401
    import h5py

    _configure_storage()
    _WORKER.update(
        {
            "h5": h5py.File(DATASET, "r", swmr=True),
            "trace": h5py.File(trace_path, "r", swmr=True),
            "env": v1._make_env(),
        }
    )
    atexit.register(_worker_close)


def _jpeg_mae(jpeg: np.ndarray, reference: np.ndarray) -> float:
    from PIL import Image

    rendered = np.asarray(
        Image.open(io.BytesIO(np.asarray(jpeg, dtype=np.uint8).tobytes())).convert("RGB"),
        dtype=np.float32,
    )
    return float(np.mean(np.abs(rendered - np.asarray(reference, dtype=np.float32))))


def _write_rollout_shard(
    path: Path,
    batch: dict[str, Any],
    rollout_ids: np.ndarray,
    source_index: int,
    source_row: int,
    source_episode: int,
    source_step: int,
    goal_row: int,
    iterations: np.ndarray,
    candidate_indices: np.ndarray,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401
    import h5py

    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if partial.exists():
        partial.unlink()
    compression = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    n = len(rollout_ids)
    try:
        with h5py.File(partial, "w", libver="latest") as out:
            out.attrs["format_version"] = FORMAT_VERSION
            out.attrs["jpeg"] = json.dumps(
                {"quality": JPEG_QUALITY, "subsampling": "4:4:4", "mode": "RGB"},
                sort_keys=True,
            )
            out.attrs["action_normalizer_mean"] = action_mean
            out.attrs["action_normalizer_scale"] = action_scale
            out.attrs["no_extra_action_clip"] = True
            out.create_dataset("rollout_id", data=rollout_ids)
            out.create_dataset("source_index", data=np.full(n, source_index, dtype=np.int16))
            out.create_dataset("source_row", data=np.full(n, source_row, dtype=np.int64))
            out.create_dataset("source_episode", data=np.full(n, source_episode, dtype=np.int64))
            out.create_dataset("source_step", data=np.full(n, source_step, dtype=np.int16))
            out.create_dataset("goal_row", data=np.full(n, goal_row, dtype=np.int64))
            out.create_dataset("cem_iteration", data=iterations.astype(np.int8))
            out.create_dataset("candidate_index", data=candidate_indices.astype(np.int16))
            jpeg_type = h5py.vlen_dtype(np.dtype("uint8"))
            pixels = out.create_dataset("pixels_jpeg", shape=(n, FRAME_COUNT), dtype=jpeg_type)
            for item in range(n):
                for frame in range(FRAME_COUNT):
                    pixels[item, frame] = batch["pixels_jpeg"][item][frame]
            for name, values in batch.items():
                if name != "pixels_jpeg":
                    out.create_dataset(name, data=values, **compression)
            out.flush()
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "source_index": source_index,
        "source_row": source_row,
        "source_episode": source_episode,
        "first_rollout_id": int(rollout_ids[0]),
        "last_rollout_id": int(rollout_ids[-1]),
        "num_rollouts": n,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _worker_state(task: tuple[Any, ...]) -> dict[str, Any]:
    root_text, source_index, candidate_values, rollout_start = task
    root = Path(root_text)
    trace = _WORKER["trace"]
    h5 = _WORKER["h5"]
    env = _WORKER["env"]
    source_index = int(source_index)
    source_row = int(trace["source_rows"][source_index])
    source_episode = int(trace["source_episodes"][source_index])
    source_step = int(trace["source_steps"][source_index])
    goal_row = int(trace["goal_rows"][source_index])
    candidate_values = np.asarray(candidate_values, dtype=np.int16)
    iterations = np.repeat(np.arange(N_STEPS, dtype=np.int8), len(candidate_values))
    candidates = np.tile(candidate_values, N_STEPS)
    n = len(iterations)
    rollout_ids = np.arange(int(rollout_start), int(rollout_start) + n, dtype=np.int64)
    batch = _empty_batch(n)
    state_seed = int.from_bytes(
        hashlib.sha256(
            f"cube-offpolicy-v2-physics|{SEED}|{source_index}|{source_row}".encode("ascii")
        ).digest()[:4],
        "little",
    )
    snapshot = _setup_snapshot(env, h5, source_row, goal_row, state_seed)
    mean = np.asarray(trace.attrs["action_normalizer_mean"], dtype=np.float64)
    scale = np.asarray(trace.attrs["action_normalizer_scale"], dtype=np.float64)
    topk = np.asarray(trace["topk_indices"][source_index], dtype=np.int16)
    reference = np.asarray(h5["pixels"][source_row], dtype=np.uint8)
    for local, (iteration, candidate_index) in enumerate(
        zip(iterations, candidates, strict=True)
    ):
        normalized = np.asarray(
            trace["candidates_normalized"][source_index, int(iteration), int(candidate_index)],
            dtype=np.float32,
        )
        raw = _inverse_action(normalized, mean, scale)
        result = v1._rollout(env, snapshot, raw)
        batch["pixels_jpeg"][local] = result["pixels_jpeg"]
        batch["action_env"][local] = raw
        batch["action_model"][local] = normalized
        for name in ("block_pos", "block_quat", "block_yaw", "terminated", "truncated"):
            batch[name][local] = result[name]
        batch["latent_cost"][local] = trace["latent_costs"][
            source_index, int(iteration), int(candidate_index)
        ]
        where = np.flatnonzero(topk[int(iteration)] == int(candidate_index))
        batch["topk_rank"][local] = int(where[0]) if len(where) else -1
        batch["initial_qpos"][local] = h5["qpos"][source_row]
        batch["initial_qvel"][local] = h5["qvel"][source_row]
        batch["initial_prev_qpos"][local] = h5["prev_qpos"][source_row]
        batch["initial_prev_qvel"][local] = h5["prev_qvel"][source_row]
        batch["goal_pos"][local] = h5["privileged_block_0_pos"][goal_row]
        batch["goal_quat"][local] = _target_quaternion_from_dataset(h5, goal_row)
        batch["source_h5_pixel_mae"][local] = _jpeg_mae(result["pixels_jpeg"][0], reference)
    shard = root / "shards" / f"state_{source_index:03d}_row_{source_row}.h5"
    return _write_rollout_shard(
        shard,
        batch,
        rollout_ids,
        source_index,
        source_row,
        source_episode,
        source_step,
        goal_row,
        iterations,
        candidates,
        mean,
        scale,
    )


def _distribution_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("distribution statistics require finite nonempty values")
    absolute = np.abs(array)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(np.max(absolute)),
    }


def _v1_diagnostic() -> dict[str, Any]:
    manifest_path = V1_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return {"available": False, "reason": f"missing {manifest_path}"}
    import hdf5plugin  # noqa: F401
    import h5py

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized: list[np.ndarray] = []
    raw: list[np.ndarray] = []
    for shard in manifest["storage"]["shards"]:
        path = V1_ROOT / "shards" / shard["filename"]
        with h5py.File(path, "r", swmr=True) as h5:
            normalized.append(np.asarray(h5["action_model"][:], dtype=np.float32).reshape(-1))
            raw.append(np.asarray(h5["action_env"][:], dtype=np.float32).reshape(-1))
    model_values = np.concatenate(normalized)
    raw_values = np.concatenate(raw)
    return {
        "available": True,
        "manifest": _identity(manifest_path),
        "num_rollouts": int(manifest["num_rollouts"]),
        "normalized_all_values": _distribution_stats(model_values),
        "raw_all_values": _distribution_stats(raw_values),
        "raw_outside_minus1_plus1_fraction": float(np.mean(np.abs(raw_values) > 1.0)),
    }


def _v2_diagnostic(root: Path, trace_path: Path, shards: list[dict[str, Any]]) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401
    import h5py

    normalized: list[np.ndarray] = []
    raw: list[np.ndarray] = []
    per_iteration: dict[str, dict[str, Any]] = {}
    raw_by_iteration: list[list[np.ndarray]] = [[] for _ in range(N_STEPS)]
    norm_by_iteration: list[list[np.ndarray]] = [[] for _ in range(N_STEPS)]
    for shard in shards:
        with h5py.File(root / "shards" / shard["filename"], "r", swmr=True) as h5:
            model = np.asarray(h5["action_model"][:], dtype=np.float32)
            action = np.asarray(h5["action_env"][:], dtype=np.float32)
            iterations = np.asarray(h5["cem_iteration"][:], dtype=np.int64)
            normalized.append(model.reshape(-1))
            raw.append(action.reshape(-1))
            for iteration in range(N_STEPS):
                mask = iterations == iteration
                norm_by_iteration[iteration].append(model[mask].reshape(-1))
                raw_by_iteration[iteration].append(action[mask].reshape(-1))
    normalized_all = np.concatenate(normalized)
    raw_all = np.concatenate(raw)
    with h5py.File(trace_path, "r", swmr=True) as trace:
        full_pool = np.asarray(trace["candidates_normalized"][:], dtype=np.float32)
        for iteration in range(N_STEPS):
            per_iteration[str(iteration)] = {
                "replayed_normalized": _distribution_stats(
                    np.concatenate(norm_by_iteration[iteration])
                ),
                "replayed_raw": _distribution_stats(np.concatenate(raw_by_iteration[iteration])),
                "complete_300_pool_normalized": _distribution_stats(full_pool[:, iteration]),
                "mean_pre_normalized": _distribution_stats(trace["mean_pre"][:, iteration]),
                "std_pre": _distribution_stats(trace["std_pre"][:, iteration]),
                "mean_post_normalized": _distribution_stats(trace["mean_post"][:, iteration]),
                "std_post": _distribution_stats(trace["std_post"][:, iteration]),
            }
    return {
        "num_rollouts": sum(int(item["num_rollouts"]) for item in shards),
        "candidate_subset": "candidate0 plus indices31..50 from every iteration",
        "normalized_all_values": _distribution_stats(normalized_all),
        "raw_all_values": _distribution_stats(raw_all),
        "raw_outside_minus1_plus1_fraction": float(np.mean(np.abs(raw_all) > 1.0)),
        "no_extra_clip": True,
        "per_cem_iteration": per_iteration,
    }


def _final_manifest(
    root: Path,
    scope: str,
    selected: dict[str, np.ndarray],
    candidate_indices: np.ndarray,
    shard_records: list[dict[str, Any]],
    elapsed: float,
    free_before: int,
) -> dict[str, Any]:
    trace_path = root / "planner_trace.h5"
    capture_manifest = root / "capture_manifest.json"
    total_bytes = sum(int(item["size_bytes"]) for item in shard_records)
    v2_diag = _v2_diagnostic(root, trace_path, shard_records)
    v1_diag = _v1_diagnostic()
    comparison: dict[str, Any] = {
        "v1": v1_diag,
        "v2": v2_diag,
        "scope_warning": (
            None
            if scope == "formal"
            else "smoke has one source state; distribution differences are diagnostic only"
        ),
    }
    if v1_diag.get("available"):
        comparison["delta_v2_minus_v1"] = {
            "normalized_std": (
                v2_diag["normalized_all_values"]["std"]
                - v1_diag["normalized_all_values"]["std"]
            ),
            "normalized_p95_abs": (
                v2_diag["normalized_all_values"]["p95_abs"]
                - v1_diag["normalized_all_values"]["p95_abs"]
            ),
            "raw_outside_minus1_plus1_fraction": (
                v2_diag["raw_outside_minus1_plus1_fraction"]
                - v1_diag["raw_outside_minus1_plus1_fraction"]
            ),
        }
    return {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "scope": scope,
        "num_rollouts": sum(int(item["num_rollouts"]) for item in shard_records),
        "num_model_transitions": sum(int(item["num_rollouts"]) for item in shard_records)
        * HORIZON,
        "num_environment_steps": sum(int(item["num_rollouts"]) for item in shard_records)
        * ENV_STEPS,
        "selection": {
            "source": "one random start from each distinct eligible expert episode",
            "seed": SEED,
            "num_source_states": len(selected["source_rows"]),
            "unique_source_episodes": len(np.unique(selected["source_episodes"])),
            "source_rows": selected["source_rows"],
            "source_episodes": selected["source_episodes"],
            "source_steps": selected["source_steps"],
            "goal_rows": selected["goal_rows"],
            "goal_contract": "same source episode at source row +25 privileged block pose",
            "excluded_eval_episodes": selected["formal_episodes"],
            "excluded_eval_episode_ids": selected["formal_episodes"],
            "measurement1_holdout_episodes": selected["measurement1_holdout_episodes"],
            "measurement1_holdout_unique_episode_count": len(
                selected["measurement1_holdout_episodes"]
            ),
            "globally_excluded_memory_episodes": selected[
                "global_memory_excluded_episodes"
            ],
            "source_and_memory_exclusion_overlap_count": int(
                np.intersect1d(
                    selected["source_episodes"],
                    selected["global_memory_excluded_episodes"],
                ).size
            ),
            "selection_file": _identity(root / "selection.npz"),
        },
        "planner_capture": {
            "protocol": "exact MaskedAug T2 first planning cycle",
            "complete_postinjection_pool_shape": [
                len(selected["source_rows"]),
                N_STEPS,
                NUM_SAMPLES,
                HORIZON,
                ACTION_BLOCK * ACTION_DIM,
            ],
            "physics_candidate_indices": candidate_indices,
            "candidate_selection_reason": (
                "candidate0 tracks the current mean; free indices31..50 avoid all 30 injected slots"
                if scope == "formal"
                else "candidate0 plus first free index31 for exact 20-rollout smoke"
            ),
            "trace": _identity(trace_path),
            "capture_manifest": _identity(capture_manifest),
        },
        "rollout_protocol": {
            "fixed_environment_steps": ENV_STEPS,
            "early_termination": False,
            "frames_per_rollout": FRAME_COUNT,
            "frame_steps": [0, 5, 10, 15, 20, 25],
            "pixels": f"JPEG quality={JPEG_QUALITY}, RGB, 4:4:4",
            "snapshot": "same complete MuJoCo/Python snapshot restored before every branch",
            "action_conversion": (
                "exact evaluation StandardScaler inverse in float32; no explicit clip before env.step"
            ),
        },
        "distribution_diagnostic_v2_vs_v1": comparison,
        "storage": {
            "root": str(root.resolve()),
            "budget_bytes": MAX_BYTES,
            "total_shard_bytes": total_bytes,
            "under_budget": total_bytes <= MAX_BYTES,
            "free_bytes_before": free_before,
            "free_bytes_after": int(shutil.disk_usage(root).free),
            "shards": shard_records,
        },
        "timing": {
            "physics_elapsed_seconds": elapsed,
            "rollouts_per_second": sum(int(item["num_rollouts"]) for item in shard_records)
            / elapsed,
        },
        "sources": {
            "dataset": _identity(DATASET, sha256=False),
            "formal_manifest": _identity(FORMAL_MANIFEST),
            "measurement1_segments": _identity(MEASUREMENT1_SEGMENTS),
            "memory_index": _identity(INDEX_ROOT / "metadata.json"),
            "masked_checkpoint": _identity(MASKED_CHECKPOINT),
            "v1_dataset_manifest": (
                _identity(V1_ROOT / "manifest.json")
                if (V1_ROOT / "manifest.json").is_file()
                else None
            ),
        },
        "schema": {
            "pixels_jpeg": ["N", FRAME_COUNT, "vlen uint8"],
            "action_env": ["N", ENV_STEPS, ACTION_DIM],
            "action_model": ["N", HORIZON, ACTION_BLOCK * ACTION_DIM],
            "block_pos": ["N", FRAME_COUNT, 3],
            "block_quat": ["N", FRAME_COUNT, 4],
            "block_yaw": ["N", FRAME_COUNT],
            "source_episode": ["N"],
            "cem_iteration": ["N"],
            "candidate_index": ["N"],
        },
    }


def _replay(args: argparse.Namespace) -> None:
    _configure_storage()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.scope == "formal" and not args.authorize_physics:
        raise PermissionError("formal 31,500-rollout replay requires --authorize-physics")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    root = _safe_root(args.scope)
    trace_path = root / "planner_trace.h5"
    capture_manifest = root / "capture_manifest.json"
    if not trace_path.is_file() or not capture_manifest.is_file():
        raise FileNotFoundError("capture must complete before physics replay")
    selected = _create_or_load_selection(root, args.scope, overwrite=False)
    capture_payload = json.loads(capture_manifest.read_text(encoding="utf-8"))
    if capture_payload.get("planner_trace", {}).get("sha256") != _sha256(trace_path):
        raise RuntimeError("planner trace identity differs from atomic capture manifest")
    state_count, candidate_indices, rollout_count = _scope_contract(args.scope)
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(trace_path, "r", swmr=True) as trace:
        for name in ("source_rows", "source_episodes", "source_steps", "goal_rows"):
            if not np.array_equal(np.asarray(trace[name][:]), selected[name]):
                raise RuntimeError(f"planner trace/selection mismatch: field={name}")
    shards_root = root / "shards"
    progress_path = root / "progress.json"
    if args.overwrite and shards_root.exists():
        shutil.rmtree(shards_root)
        if progress_path.exists():
            progress_path.unlink()
        if (root / "manifest.json").exists():
            (root / "manifest.json").unlink()
    shards_root.mkdir(parents=True, exist_ok=True)
    progress = None
    if args.resume:
        if not progress_path.is_file():
            raise FileNotFoundError(f"resume requested but progress is missing: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("trace_sha256") != _sha256(trace_path):
            raise RuntimeError("resume planner trace identity mismatch")
    elif progress_path.exists() or any(shards_root.iterdir()):
        raise FileExistsError("replay output exists; use --resume or --overwrite")
    records = list(progress.get("shards", [])) if progress else []
    completed = {int(item["source_index"]): item for item in records}
    for source_index, record in completed.items():
        path = shards_root / record["filename"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"resume shard identity mismatch: source_index={source_index}")
    per_state = N_STEPS * len(candidate_indices)
    tasks = [
        (str(root), source_index, candidate_indices, source_index * per_state)
        for source_index in range(state_count)
        if source_index not in completed
    ]
    free_before = int(shutil.disk_usage(root).free)
    if free_before < 2 * (1 << 30):
        raise RuntimeError(
            f"insufficient data-disk headroom: expected_at_least={2 * (1 << 30)}, "
            f"actual={free_before}"
        )
    prior_elapsed = float(progress.get("elapsed_seconds", 0.0)) if progress else 0.0
    started = time.time()
    if tasks:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with context.Pool(
            processes=min(args.workers, len(tasks)),
            initializer=_worker_init,
            initargs=(str(trace_path),),
        ) as pool:
            for record in pool.imap_unordered(_worker_state, tasks, chunksize=1):
                records.append(record)
                records.sort(key=lambda item: int(item["source_index"]))
                done = sum(int(item["num_rollouts"]) for item in records)
                bytes_written = sum(int(item["size_bytes"]) for item in records)
                projected = int(math.ceil(bytes_written / done * rollout_count))
                if projected > MAX_BYTES:
                    pool.terminate()
                    raise RuntimeError(
                        f"projected collection exceeds 40GB: projected={projected}"
                    )
                free_now = int(shutil.disk_usage(root).free)
                remaining = max(0, projected - bytes_written)
                if free_now < remaining + (1 << 30):
                    pool.terminate()
                    raise RuntimeError(
                        "insufficient data-disk free space for projected replay: "
                        f"expected_at_least={remaining + (1 << 30)}, actual={free_now}"
                    )
                elapsed = prior_elapsed + time.time() - started
                _atomic_json(
                    progress_path,
                    {
                        "format_version": FORMAT_VERSION,
                        "complete": done == rollout_count,
                        "scope": args.scope,
                        "trace_sha256": _sha256(trace_path),
                        "requested_rollouts": rollout_count,
                        "completed_rollouts": done,
                        "worker_count": args.workers,
                        "elapsed_seconds": elapsed,
                        "bytes_written": bytes_written,
                        "projected_bytes": projected,
                        "shards": records,
                    },
                )
                print(
                    f"atomic replay progress: {done}/{rollout_count} rollouts, bytes={bytes_written}",
                    flush=True,
                )
    if not progress_path.is_file():
        raise RuntimeError("replay produced no progress artifact")
    final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if int(final_progress["completed_rollouts"]) != rollout_count:
        raise RuntimeError(
            f"replay incomplete: expected={rollout_count}, actual={final_progress['completed_rollouts']}"
        )
    manifest = _final_manifest(
        root,
        args.scope,
        selected,
        candidate_indices,
        records,
        float(final_progress["elapsed_seconds"]),
        free_before,
    )
    _atomic_json(root / "manifest.json", manifest)
    print(json.dumps({"root": str(root), "num_rollouts": rollout_count}, sort_keys=True))


def _validate(args: argparse.Namespace) -> None:
    _configure_storage()
    root = _safe_root(args.scope)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_count, candidate_indices, rollout_count = _scope_contract(args.scope)
    if manifest.get("format_version") != FORMAT_VERSION or not manifest.get("complete"):
        raise RuntimeError("unsupported or incomplete V2 manifest")
    if int(manifest["num_rollouts"]) != rollout_count:
        raise RuntimeError("manifest rollout count mismatch")
    trace_path = root / "planner_trace.h5"
    if _sha256(trace_path) != manifest["planner_capture"]["trace"]["sha256"]:
        raise RuntimeError("planner trace changed after manifest creation")
    import hdf5plugin  # noqa: F401
    import h5py
    from PIL import Image

    excluded = set(map(int, manifest["selection"]["excluded_eval_episode_ids"]))
    holdout = set(map(int, manifest["selection"]["measurement1_holdout_episodes"]))
    global_memory = set(map(int, manifest["selection"]["globally_excluded_memory_episodes"]))
    expected_id = 0
    source_episodes: list[int] = []
    decoded = 0
    with h5py.File(trace_path, "r", swmr=True) as trace:
        if trace.attrs.get("format_version") != TRACE_FORMAT_VERSION:
            raise RuntimeError("planner trace format mismatch")
        expected_pool = (state_count, N_STEPS, NUM_SAMPLES, HORIZON, 25)
        if trace["candidates_normalized"].shape != expected_pool:
            raise RuntimeError(
                f"planner pool shape mismatch: expected={expected_pool}, "
                f"actual={trace['candidates_normalized'].shape}"
            )
        retrieved = set(map(int, np.asarray(trace["retrieval_episodes"][:]).reshape(-1)))
        if retrieved & global_memory:
            raise RuntimeError("globally excluded memory episode leaked into planner trace")
        mean = np.asarray(trace.attrs["action_normalizer_mean"], dtype=np.float64)
        scale = np.asarray(trace.attrs["action_normalizer_scale"], dtype=np.float64)
        for shard in manifest["storage"]["shards"]:
            path = root / "shards" / shard["filename"]
            if _sha256(path) != shard["sha256"]:
                raise RuntimeError(f"shard hash mismatch: {path}")
            with h5py.File(path, "r", swmr=True) as h5:
                if h5.attrs.get("format_version") != FORMAT_VERSION:
                    raise RuntimeError(f"shard format mismatch: {path}")
                ids = np.asarray(h5["rollout_id"][:], dtype=np.int64)
                expected = np.arange(expected_id, expected_id + len(ids), dtype=np.int64)
                if not np.array_equal(ids, expected):
                    raise RuntimeError(f"rollout id discontinuity: {path}")
                expected_id += len(ids)
                episodes = np.asarray(h5["source_episode"][:], dtype=np.int64)
                source_episodes.append(int(episodes[0]))
                if set(map(int, episodes)) & (excluded | holdout):
                    raise RuntimeError(f"training exclusion leaked into source shard: {path}")
                iterations = np.asarray(h5["cem_iteration"][:], dtype=np.int64)
                candidates = np.asarray(h5["candidate_index"][:], dtype=np.int64)
                expected_iterations = np.repeat(np.arange(N_STEPS), len(candidate_indices))
                expected_candidates = np.tile(candidate_indices, N_STEPS)
                if not np.array_equal(iterations, expected_iterations) or not np.array_equal(
                    candidates, expected_candidates
                ):
                    raise RuntimeError(f"iteration/candidate replay contract mismatch: {path}")
                model = np.asarray(h5["action_model"][:], dtype=np.float32)
                raw = np.asarray(h5["action_env"][:], dtype=np.float32)
                inverse = model.reshape(-1, ENV_STEPS, ACTION_DIM).copy()
                inverse *= scale
                inverse += mean
                if not np.array_equal(inverse, raw):
                    diff = float(np.max(np.abs(inverse.astype(np.float64) - raw)))
                    raise RuntimeError(
                        f"action inverse mismatch: path={path}, max_abs_diff={diff}"
                    )
                source_index = int(h5["source_index"][0])
                expected_model = np.stack(
                    [
                        trace["candidates_normalized"][source_index, int(it), int(candidate)]
                        for it, candidate in zip(iterations, candidates, strict=True)
                    ]
                ).astype(np.float32)
                if not np.array_equal(model, expected_model):
                    raise RuntimeError(f"replayed actions differ from captured planner pool: {path}")
                if decoded < args.jpeg_checks:
                    value = np.asarray(h5["pixels_jpeg"][0, 0], dtype=np.uint8)
                    image = np.asarray(Image.open(io.BytesIO(value.tobytes())).convert("RGB"))
                    if image.shape != (224, 224, 3):
                        raise RuntimeError(f"decoded JPEG shape mismatch: {image.shape}")
                    decoded += 1
    if expected_id != rollout_count:
        raise RuntimeError(f"rollout count mismatch: expected={rollout_count}, actual={expected_id}")
    if len(source_episodes) != state_count or len(set(source_episodes)) != state_count:
        raise RuntimeError("source shards do not map one-to-one to distinct episodes")
    result = {
        "valid": True,
        "scope": args.scope,
        "num_rollouts": rollout_count,
        "num_source_episodes": state_count,
        "decoded_jpeg_shards": decoded,
        "formal_episode_overlap": len(set(source_episodes) & excluded),
        "measurement1_holdout_overlap": len(set(source_episodes) & holdout),
        "memory_global_exclusion_overlap": 0,
        "no_extra_action_clip": True,
        "total_shard_bytes": manifest["storage"]["total_shard_bytes"],
        "under_40gb": manifest["storage"]["under_budget"],
    }
    _atomic_json(root / "validation.json", result)
    print(json.dumps(result, sort_keys=True))


def _selection_check(args: argparse.Namespace) -> None:
    _configure_storage()
    root = _safe_root(args.scope)
    selected = _create_or_load_selection(root, args.scope, overwrite=args.overwrite)
    result = {
        "scope": args.scope,
        "source_count": len(selected["source_rows"]),
        "unique_source_episodes": len(np.unique(selected["source_episodes"])),
        "formal_excluded": len(selected["formal_episodes"]),
        "measurement1_holdout_unique": len(selected["measurement1_holdout_episodes"]),
        "global_memory_excluded": len(selected["global_memory_excluded_episodes"]),
        "selection": _identity(root / "selection.npz"),
    }
    print(json.dumps(_jsonable(result), sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "capture", "replay", "validate"):
        item = sub.add_parser(command)
        item.add_argument("--scope", choices=("smoke", "formal"), default="smoke")
        if command in {"select", "capture", "replay"}:
            item.add_argument("--overwrite", action="store_true")
        if command == "capture":
            item.add_argument("--device", default="cuda")
            item.add_argument("--authorize-model-capture", action="store_true")
        if command == "replay":
            item.add_argument("--workers", type=int, default=4)
            item.add_argument("--resume", action="store_true")
            item.add_argument("--authorize-physics", action="store_true")
        if command == "validate":
            item.add_argument("--jpeg-checks", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        _selection_check(args)
    elif args.command == "capture":
        _capture(args)
    elif args.command == "replay":
        _replay(args)
    elif args.command == "validate":
        _validate(args)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
