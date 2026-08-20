#!/usr/bin/env python3
"""Cube goal-conditioned retrieval evaluation for frozen G1T1/G1T2.

This is a narrow adapter around ``eval_trust_region.py``.  It replaces only
the memory retriever: exact state-nearest raw anchors are truncated to 100
after globally excluding all 50 formal evaluation episodes, positive 25-step
goal progress is preferred, and any shortage is filled from that same top-100.
The underlying T1/T2 CEM distributions and all model/evaluation code remain
the frozen Trust-Region implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_cube_memory_index as memory  # noqa: E402
import cube_goal_conditioned_common as common  # noqa: E402
import eval_trust_region as trust  # noqa: E402


_BaseMemoryIndex = memory.CubeMemoryIndex
_trust_save_trace = trust._save_trace
_trust_save_first_cycle_pools = trust._save_first_cycle_pools
_ACTIVE_INDEX: "GoalConditionedMemoryIndex | None" = None


class GoalProtocol(str):
    """Serialize as G1 while satisfying the frozen T1/T2 branch predicates."""

    _BASE = {"g1t1": "t1", "g1t2": "t2"}

    def __new__(cls, value: str) -> "GoalProtocol":
        normalized = str(value).lower()
        if normalized not in cls._BASE:
            raise ValueError(f"invalid goal-conditioned protocol: {value}")
        result = super().__new__(cls, normalized)
        result.base = cls._BASE[normalized]
        return result

    def __eq__(self, other: object) -> bool:
        return str.__eq__(self, other) or other == self.base

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = str.__hash__


class GoalConditionedMemoryIndex(_BaseMemoryIndex):
    """Exact top-100 state retrieval with a fixed per-eval-episode goal."""

    def __init__(self, root: Path, dataset: Path | None = None) -> None:
        super().__init__(root, dataset)
        global _ACTIVE_INDEX
        _ACTIVE_INDEX = self
        manifest = json.loads(common.MANIFEST.read_text(encoding="utf-8"))
        formal_rows = np.asarray(manifest["formal_rows"], dtype=np.int64)
        if formal_rows.shape != (50,) or len(np.unique(formal_rows)) != 50:
            raise RuntimeError(
                "fixed evaluation manifest must contain 50 unique rows: "
                f"expected_shape=(50,), actual_shape={formal_rows.shape}, "
                f"unique={len(np.unique(formal_rows))}"
            )
        import hdf5plugin  # noqa: F401
        import h5py

        with h5py.File(self.dataset, "r", swmr=True) as h5:
            episodes = np.asarray(h5["ep_idx"][formal_rows], dtype=np.int64)
            goal_rows = formal_rows + common.GOAL_SCORE_HORIZON
            goals = np.asarray(
                h5["privileged_block_0_pos"][goal_rows], dtype=np.float64
            )
            goal_episodes = np.asarray(h5["ep_idx"][goal_rows], dtype=np.int64)
        if len(np.unique(episodes)) != 50:
            raise RuntimeError(
                "fixed 50 rows must belong to distinct episodes: "
                f"expected=50, actual={len(np.unique(episodes))}"
            )
        if not np.array_equal(episodes, goal_episodes):
            bad = np.flatnonzero(episodes != goal_episodes)
            raise RuntimeError(
                "formal goal row crosses an episode boundary: "
                f"expected=0 mismatches, actual={len(bad)}, positions={bad.tolist()}"
            )
        self.formal_rows = formal_rows
        self.excluded_eval_episodes = frozenset(int(value) for value in episodes)
        self.goal_by_episode = {
            int(episode): goals[idx].copy() for idx, episode in enumerate(episodes)
        }
        self._retrieval_queue: dict[
            tuple[int, tuple[int, ...]], list[dict[str, Any]]
        ] = {}

    def _exact_top100(
        self, raw_feature: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_query = np.asarray(raw_feature, dtype=np.float64).reshape(1, -1)
        query = self.normalize(raw_query)[0]
        k = 256
        candidates: list[tuple[float, int, int, int]] = []
        while True:
            distances, indices = self.tree.query(
                query, k=min(k, len(self.rows)), eps=0.0, workers=1
            )
            candidates = sorted(
                (
                    float(distance),
                    int(self.rows[int(index)]),
                    int(self.episodes[int(index)]),
                    int(index),
                )
                for distance, index in zip(
                    np.atleast_1d(distances), np.atleast_1d(indices), strict=True
                )
                if int(self.episodes[int(index)]) not in self.excluded_eval_episodes
            )
            if len(candidates) >= common.TOP100:
                cutoff = float(candidates[common.TOP100 - 1][0])
                ball = self.tree.query_ball_point(
                    query,
                    r=np.nextafter(cutoff, np.inf),
                    eps=0.0,
                    workers=1,
                )
                complete = sorted(
                    (
                        float(np.linalg.norm(self.features[int(index)] - query)),
                        int(self.rows[int(index)]),
                        int(self.episodes[int(index)]),
                        int(index),
                    )
                    for index in ball
                    if int(self.episodes[int(index)])
                    not in self.excluded_eval_episodes
                )
                if len(complete) >= common.TOP100:
                    candidates = complete[: common.TOP100]
                    break
            if k >= len(self.rows):
                raise RuntimeError(
                    "insufficient anchors after global fixed50 exclusion: "
                    f"expected={common.TOP100}, actual={len(candidates)}, "
                    f"query_position={raw_query[0].tolist()}"
                )
            k = min(k * 2, len(self.rows))
        return (
            np.asarray([item[0] for item in candidates], dtype=np.float64),
            np.asarray([item[1] for item in candidates], dtype=np.int64),
            np.asarray([item[2] for item in candidates], dtype=np.int64),
            np.asarray([item[1] % 201 for item in candidates], dtype=np.int64),
            np.asarray([item[3] for item in candidates], dtype=np.int64),
        )

    def _trajectory_positions(
        self, rows: np.ndarray, episodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import hdf5plugin  # noqa: F401
        import h5py

        future_rows = rows + common.GOAL_SCORE_HORIZON
        indices = np.concatenate([rows, future_rows])
        unique, inverse = np.unique(indices, return_inverse=True)
        with h5py.File(self.dataset, "r", swmr=True) as h5:
            unique_xyz = np.asarray(
                h5["privileged_block_0_pos"][unique], dtype=np.float64
            )
            unique_future, future_inverse = np.unique(
                future_rows, return_inverse=True
            )
            future_episodes = np.asarray(
                h5["ep_idx"][unique_future], dtype=np.int64
            )[future_inverse]
        all_xyz = unique_xyz[inverse]
        current_xyz, future_xyz = np.split(all_xyz, 2)
        valid = future_episodes == episodes
        return current_xyz, future_xyz, valid

    def retrieve(
        self,
        raw_feature: np.ndarray,
        exclude_episode: int,
        count: int = common.MEMORY_SLOTS,
    ) -> dict[str, np.ndarray]:
        if count != common.MEMORY_SLOTS:
            raise ValueError(
                "goal-conditioned retrieval is frozen to 10 seeds: "
                f"expected={common.MEMORY_SLOTS}, actual={count}"
            )
        excluded_episode = int(exclude_episode)
        if excluded_episode not in self.excluded_eval_episodes:
            raise ValueError(
                "query episode is outside the fixed50 exclusion contract: "
                f"expected_member=True, actual={excluded_episode}"
            )
        goal = self.goal_by_episode[excluded_episode]
        distances, rows, episodes, steps, anchor_indices = self._exact_top100(
            raw_feature
        )
        current_xyz, future_xyz, future_valid = self._trajectory_positions(
            rows, episodes
        )
        distinct_sources = len(np.unique(episodes))
        if distinct_sources < count:
            raise RuntimeError(
                "insufficient distinct source episodes inside fixed top-100: "
                f"expected={count}, actual={distinct_sources}, "
                f"query_position={np.asarray(raw_feature, dtype=np.float64).reshape(-1).tolist()}"
            )
        selected = common.select_goal_aligned(
            distances,
            rows,
            episodes,
            steps,
            current_xyz,
            future_xyz,
            future_valid,
            goal,
            count=count,
        )
        chosen = np.asarray(selected["selected_indices"], dtype=np.int64)
        selected_episodes = episodes[chosen]
        leaked = sorted(
            set(int(value) for value in selected_episodes)
            & self.excluded_eval_episodes
        )
        if leaked:
            raise RuntimeError(
                "goal-conditioned retrieval leaked fixed evaluation episodes: "
                f"expected=[], actual={leaked}, query_episode={excluded_episode}"
            )
        result = {
            "distances": distances[chosen],
            "rows": rows[chosen],
            "episodes": selected_episodes,
            "steps": steps[chosen],
            "anchor_indices": anchor_indices[chosen],
        }
        metadata = {
            "goal_position": goal.copy(),
            "top100_distances": distances,
            "top100_rows": rows,
            "top100_episodes": episodes,
            "top100_steps": steps,
            "top100_anchor_indices": anchor_indices,
            "top100_block_position_t": current_xyz,
            "top100_block_position_t_plus_25": future_xyz,
            "top100_future_valid": future_valid,
            "top100_goal_scores": np.asarray(selected["scores"]),
            "top100_is_aligned": np.asarray(selected["aligned"]),
            "selected_top100_indices": chosen,
            "selected_goal_scores": np.asarray(selected["scores"])[chosen],
            "selected_is_aligned": np.asarray(selected["selected_is_aligned"]),
            "selected_is_fallback": np.asarray(selected["selected_is_fallback"]),
            "alignment_raw_positive_count": int(selected["raw_positive_count"]),
            "alignment_raw_positive_rate": float(selected["raw_positive_rate"]),
            "alignment_selected_count": int(selected["selected_aligned_count"]),
            "alignment_selected_rate": float(selected["selected_aligned_rate"]),
            "alignment_fallback_count": int(selected["fallback_count"]),
            "alignment_fallback_rate": float(selected["fallback_rate"]),
            "global_excluded_eval_episodes": np.asarray(
                sorted(self.excluded_eval_episodes), dtype=np.int64
            ),
        }
        key = (excluded_episode, tuple(int(value) for value in result["rows"]))
        self._retrieval_queue.setdefault(key, []).append(metadata)
        print(
            "[goal-retrieval] "
            f"query_episode={excluded_episode} raw_positive="
            f"{metadata['alignment_raw_positive_count']}/{common.TOP100} "
            f"selected_aligned={metadata['alignment_selected_count']}/{common.MEMORY_SLOTS} "
            f"fallback={metadata['alignment_fallback_count']}/{common.MEMORY_SLOTS}"
        )
        return result

    def take_metadata(self, context: dict[str, Any]) -> dict[str, Any]:
        key = (
            int(context["excluded_eval_episode"]),
            tuple(int(value) for value in context["source_rows"]),
        )
        queue = self._retrieval_queue.get(key, [])
        if not queue:
            raise RuntimeError(
                "missing goal retrieval provenance for CEM context: "
                f"episode={key[0]}, rows={list(key[1])}"
            )
        metadata = queue.pop(0)
        if not queue:
            self._retrieval_queue.pop(key, None)
        return metadata


class GoalConditionedCostProxy(trust.TrustRegionCostProxy):
    """Attach retrieval provenance to every otherwise-frozen CEM trace row."""

    _ALIGNMENT_FIELDS = (
        "goal_position",
        "top100_distances",
        "top100_rows",
        "top100_episodes",
        "top100_steps",
        "top100_anchor_indices",
        "top100_block_position_t",
        "top100_block_position_t_plus_25",
        "top100_future_valid",
        "top100_goal_scores",
        "top100_is_aligned",
        "selected_top100_indices",
        "selected_goal_scores",
        "selected_is_aligned",
        "selected_is_fallback",
        "alignment_raw_positive_count",
        "alignment_raw_positive_rate",
        "alignment_selected_count",
        "alignment_selected_rate",
        "alignment_fallback_count",
        "alignment_fallback_rate",
        "global_excluded_eval_episodes",
    )

    def begin_solve(self, contexts: list[dict[str, Any]]) -> None:
        if _ACTIVE_INDEX is None:
            raise RuntimeError("goal-conditioned memory index was not initialized")
        for context in contexts:
            context.update(_ACTIVE_INDEX.take_metadata(context))
        super().begin_solve(contexts)

    def get_cost(self, info_dict: dict[str, Any], candidates: Any) -> Any:
        if self.pending is None:
            raise RuntimeError("goal-conditioned proxy called without context")
        context = self.pending[self.call_index // common.N_STEPS]
        result = super().get_cost(info_dict, candidates)
        record = self.trace[-1]
        for field in self._ALIGNMENT_FIELDS:
            record[field] = context[field]
        return result


def _save_trace(output: Path, proxy: GoalConditionedCostProxy) -> dict[str, Any]:
    summary = _trust_save_trace(output, proxy)
    records = proxy.trace
    array_fields = GoalConditionedCostProxy._ALIGNMENT_FIELDS
    arrays = {
        field: np.stack([np.asarray(row[field]) for row in records])
        for field in array_fields
    }
    arrays.update(
        {
            "env_idx": np.asarray([row["env_idx"] for row in records], dtype=np.int64),
            "planning_cycle": np.asarray(
                [row["planning_cycle"] for row in records], dtype=np.int64
            ),
            "cem_iteration": np.asarray(
                [row["cem_iteration"] for row in records], dtype=np.int64
            ),
        }
    )
    path = output / "goal_alignment_trace.npz"
    np.savez_compressed(path, **arrays)
    aggregate = common.alignment_distribution(records)
    common.write_json(
        output / "goal_alignment_trace.json",
        {
            "format_version": "cube_goal_conditioned_trace_v1",
            "definition": {
                "topn": common.TOP100,
                "score": "norm(block_t-goal)-norm(block_t_plus_25-goal)",
                "aligned": "future_valid and score>0",
                "selection": (
                    "aligned anchors in stable state order, one per episode; "
                    "shortage filled from the same top-100"
                ),
                "global_fixed50_episode_exclusion": True,
            },
            "aggregate": aggregate,
            "arrays": common.file_identity(path),
        },
    )
    summary["goal_alignment"] = {
        "aggregate": aggregate,
        "npz": common.file_identity(path),
        "json": common.file_identity(output / "goal_alignment_trace.json"),
    }
    return summary


def _save_first_cycle_pools(
    output: Path,
    proxy: GoalConditionedCostProxy,
    recorder: Any,
    rows: np.ndarray,
    eval_episodes: np.ndarray,
    raw_inputs: dict[str, np.ndarray],
    dataset_path: Path,
    scaler: Any,
    protocol: str,
    condition: str,
) -> dict[str, Any]:
    result = _trust_save_first_cycle_pools(
        output,
        proxy,
        recorder,
        rows,
        eval_episodes,
        raw_inputs,
        dataset_path,
        scaler,
        protocol,
        condition,
    )
    root = output / "first_cycle_pools"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_env = {int(item["env_idx"]): item for item in manifest["cases"]}
    for env_idx in common.AUDIT_ENVS:
        if env_idx >= len(rows):
            continue
        trace = proxy.first_cycle_final[int(env_idx)]
        row = int(rows[env_idx])
        case = root / common.case_name(env_idx, row)
        provenance_path = case / "goal_alignment_provenance.npz"
        np.savez_compressed(
            provenance_path,
            **{
                field: np.asarray(trace[field])
                for field in GoalConditionedCostProxy._ALIGNMENT_FIELDS
            },
        )
        alignment = {
            "raw_positive_count": int(trace["alignment_raw_positive_count"]),
            "raw_positive_rate": float(trace["alignment_raw_positive_rate"]),
            "selected_aligned_count": int(trace["alignment_selected_count"]),
            "selected_aligned_rate": float(trace["alignment_selected_rate"]),
            "fallback_count": int(trace["alignment_fallback_count"]),
            "fallback_rate": float(trace["alignment_fallback_rate"]),
        }
        meta_path = case / "capture_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["goal_alignment_provenance"] = common.file_identity(provenance_path)
        meta["goal_alignment"] = alignment
        meta["all_fixed50_eval_episodes_excluded"] = True
        common.write_json(meta_path, meta)
        by_env[env_idx]["goal_alignment_provenance"] = common.file_identity(
            provenance_path
        )
        by_env[env_idx]["goal_alignment"] = alignment
        by_env[env_idx]["all_fixed50_eval_episodes_excluded"] = True
    manifest["format_version"] = "cube_goal_conditioned_first_cycle_manifest_v1"
    manifest["retrieval"] = {
        "topn": common.TOP100,
        "global_fixed50_episode_exclusion": True,
        "fallback_scope": "same top-100 only",
    }
    common.write_json(manifest_path, manifest)
    return result


def _install_adapter() -> None:
    """Patch only this process's imported Trust-Region helper bindings."""

    trust.common = common
    trust.memory.CubeMemoryIndex = GoalConditionedMemoryIndex
    trust.TrustRegionCostProxy = GoalConditionedCostProxy
    trust._save_trace = _save_trace
    trust._save_first_cycle_pools = _save_first_cycle_pools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=common.PROTOCOLS, required=True)
    parser.add_argument("--condition", choices=common.CONDITIONS, required=True)
    parser.add_argument("--num-eval", type=int, choices=(2, 50), default=2)
    parser.add_argument("--mode", choices=("capture", "evaluate"), default="evaluate")
    parser.add_argument("--seed", type=int, default=common.FORMAL_SEED)
    parser.add_argument("--dataset", type=Path, default=common.DATASET)
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST)
    parser.add_argument("--index", type=Path, default=common.MEMORY_INDEX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authorize-capture", action="store_true")
    parser.add_argument("--authorize-formal", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.manifest.expanduser().resolve() != common.MANIFEST.resolve():
        raise ValueError(
            "goal-conditioned retrieval is frozen to the fixed50 manifest: "
            f"expected={common.MANIFEST.resolve()}, "
            f"actual={args.manifest.expanduser().resolve()}"
        )
    args.protocol = GoalProtocol(args.protocol)
    _install_adapter()
    result = trust.run(args)
    output = (
        args.output
        or (
            common.capture_output_root(args.protocol, args.condition)
            if args.mode == "capture"
            else common.default_eval_output(args.protocol, args.condition, args.num_eval)
        )
    ).resolve()
    results_path = output / "results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["format_version"] = (
        "cube_goal_conditioned_gate_capture_v1"
        if args.mode == "capture"
        else "cube_goal_conditioned_evaluation_v1"
    )
    payload["protocol"]["base_trust_region_protocol"] = common.base_protocol(
        args.protocol
    )
    payload["protocol"]["retrieval_contract"] = {
        "state_pool": "exact raw-anchor top-100 after global fixed50 exclusion",
        "goal_source": "fixed H5 formal-row+25 privileged_block_0_pos",
        "score": "norm(block_t-goal)-norm(block_t_plus_25-goal)",
        "positive_filter": "score>0",
        "unique_source_episode": True,
        "fallback": "stable state order within the same top-100 only",
    }
    helpers = payload["protocol"].setdefault("helper_provenance", {})
    helpers["trust_region_evaluator"] = helpers.get(
        "this_evaluator", common.file_identity(Path(trust.__file__))
    )
    helpers["this_evaluator"] = common.file_identity(Path(__file__))
    payload["goal_alignment"] = payload["trace"]["goal_alignment"]
    common.write_json(results_path, payload)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
