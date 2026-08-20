#!/usr/bin/env python3
"""Freeze B2 real-frame candidates and replay the 70 B1 calls offline.

Candidate construction is public so the online evaluator and offline replay use
one implementation.  Real API execution requires both ``--execute-api`` and an
explicit environment authorization latch.  Dry-run and self-test never touch
the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import brain_supervisor as b1
    from . import brain_supervisor_v2 as b2
except ImportError:
    import brain_supervisor as b1  # type: ignore[no-redef]
    import brain_supervisor_v2 as b2  # type: ignore[no-redef]


AILAB = Path(__file__).resolve().parents[2]
INDEX_ROOT = AILAB / "outputs/memory_index/cube_expert_v1"
DATASET = AILAB / "datasets/ogbench/cube_single_expert.h5"
B1_CALLS = AILAB / "outputs/eval/cube/longhorizon/brain_offset100/llm_calls.json"
B1_RUN_MANIFEST = AILAB / "outputs/eval/cube/longhorizon/brain_offset100/run_manifest.json"
B1_RESULTS = AILAB / "outputs/eval/cube/longhorizon/brain_offset100/results.json"
BASELINE_RUN_MANIFEST = AILAB / "outputs/eval/cube/longhorizon/baseline_offset100/run_manifest.json"
BASELINE_RESULTS = AILAB / "outputs/eval/cube/longhorizon/baseline_offset100/results.json"
FIXED_EVAL_MANIFEST = AILAB / "outputs/audit/cube_cem_manifest.json"
OUTPUT_ROOT = AILAB / "outputs/eval/cube/brain_b2/offline_prompt_replay"
INPUTS_PATH = OUTPUT_ROOT / "frozen_inputs.json"
INPUT_MANIFEST_PATH = OUTPUT_ROOT / "input_manifest.json"
CANDIDATE_AUDIT_PATH = OUTPUT_ROOT / "candidate_audit.json"
EPISODE_LENGTH = 201
RECOVERY_EE_BLOCK_MAX_M = 0.03
RECOVERY_CONTACT_MIN = 0.5
RECOVERY_OPENING_MAX = 0.60
RETREAT_MIN_DISPLACEMENT_M = 0.045
RETREAT_PREFERRED_MAX_DISPLACEMENT_M = 0.08
RETREAT_PREFERRED_REVERSE_PROJECTION_M = 0.04
AUTH_ENV = "BRAIN_B2_REAL_API_AUTHORIZED"
ROUND_ARTIFACT_FILES = {
    "api_manifest": "api_manifest.json",
    "llm_calls": "llm_calls.json",
    "summary": "summary.json",
    "system_prompt_text": "system_prompt.txt",
    "decisions": "decisions.json",
    "acceptance": "acceptance.json",
}
EXPECTED_SHA256 = {
    B1_CALLS: "c3dcae0704ecaaa8593cf016c7ee43950e28c43b2a9c4795d0f2eee628bdf35a",
    B1_RUN_MANIFEST: "9e1b09e22b763217946ac69801c3e88459c2a8390b0deee08f7b16ba381fbc62",
    B1_RESULTS: "0188a837d88a9ca8a8a3362d94a0a9c61c899ef9a8e22f0eba32a1060f1c4ae0",
    BASELINE_RUN_MANIFEST: "3f6bdebda785a53fcd50566fbe0f012047e1104e969826cb43e621b6408b85f7",
    BASELINE_RESULTS: "ca120d2fa2aeada19404718401c9348092d943fb9ae24f71fe7ef009d1a235d4",
    FIXED_EVAL_MANIFEST: "cf3f4cc8e9ff8efbb5d5f0a099bcedbef62e3662bd97e2714364d01325425d50",
    INDEX_ROOT / "metadata.json": "fdf7d064b20ae5ed3b3f013fd0aee314d44b8e1329325cfedac52570a23a37b3",
    INDEX_ROOT / "anchor_rows.npy": "51ddb7bb2045268fe25310ff8e97a6692d7102c71e34700e74f6a32d741332f1",
    INDEX_ROOT / "anchor_episodes.npy": "06b33d82803daff30e92f706ebbd4448fe2d46a4484ba77276a4a4faa7928db7",
    INDEX_ROOT / "anchor_features_z.npy": "38533535ff0488dc68c76e1d48076c4f7c9d8b12a7ae518986692324e3e8dd6d",
    INDEX_ROOT / "stats.npz": "63cf7d57a42b6045cecc218c72325ade968db0b68eafd097b64948dc0e687cf9",
}
EXPECTED_ROWS_SHA256_INT64 = "0cd9a6fd177d40f62c5d06d5632454f9cad4aeef357158dd393777db505a78ce"
EXPECTED_EPISODES_SHA256_INT64 = "f99be3d88d6af9f78bccf7f726541b30e65ee6c02f56ab9fdfbf2d11338186a4"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha_file(path),
    }


def _sha_int64(values: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes(order="C")).hexdigest()


def verify_memory_index_assets(index_root: Path, dataset: Path) -> dict[str, Any]:
    resolved_index = index_root.expanduser().resolve()
    resolved_dataset = dataset.expanduser().resolve()
    if resolved_index != INDEX_ROOT.resolve() or resolved_dataset != DATASET.resolve():
        raise RuntimeError("B2 requires the frozen cube_expert_v1 index and expert dataset")
    identities: dict[str, Any] = {}
    for name in (
        "metadata.json", "anchor_rows.npy", "anchor_episodes.npy",
        "anchor_features_z.npy", "stats.npz",
    ):
        path = resolved_index / name
        identity = _identity(path)
        expected = EXPECTED_SHA256[path]
        if identity["sha256"] != expected:
            raise RuntimeError(
                f"memory index SHA drift: path={path}, expected={expected}, "
                f"actual={identity['sha256']}"
            )
        identities[name] = identity
    metadata = _load_json(resolved_index / "metadata.json")
    for name, entry in metadata.get("files", {}).items():
        if name not in identities or entry.get("sha256") != identities[name]["sha256"]:
            raise RuntimeError(f"memory index metadata/file mismatch: {name}")
    dataset_entry = metadata.get("dataset", {})
    stat = resolved_dataset.stat()
    if (
        Path(dataset_entry.get("path", "")).resolve() != resolved_dataset
        or dataset_entry.get("size_bytes") != stat.st_size
        or dataset_entry.get("mtime_ns") != stat.st_mtime_ns
    ):
        raise RuntimeError("memory index dataset identity drift (path/size/mtime)")
    identities["dataset"] = {
        "path": str(resolved_dataset), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "sha256": None,
        "sha256_status": "not recorded by frozen index; path/size/mtime bound",
    }
    return identities


def verify_frozen_sources() -> dict[str, Any]:
    """Bind the replay to the exact completed B1 and memory-index assets."""

    identities: dict[str, Any] = {}
    for path, expected in EXPECTED_SHA256.items():
        identity = _identity(path)
        if identity["sha256"] != expected:
            raise RuntimeError(
                f"frozen asset SHA drift: path={path}, expected={expected}, "
                f"actual={identity['sha256']}"
            )
        identities[str(path.resolve())] = identity

    manifest = _load_json(B1_RUN_MANIFEST)
    expected_protocol = {
        "format_version": "cube_brain_b1_run_manifest_v1",
        "status": "complete", "mode": "brain", "goal_offset_steps": 100,
        "eval_budget": 200, "num_eval": 50, "seed": 42,
    }
    for key, expected in expected_protocol.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"B1 protocol drift at {key}: expected={expected!r}, actual={manifest.get(key)!r}"
            )
    selection = manifest.get("selection", {})
    identity_values = selection.get("identities", {})
    if identity_values.get("rows_sha256_int64") != EXPECTED_ROWS_SHA256_INT64:
        raise RuntimeError("B1 offset100 row identity drift")
    if identity_values.get("episodes_sha256_int64") != EXPECTED_EPISODES_SHA256_INT64:
        raise RuntimeError("B1 fixed episode identity drift")
    if _sha_int64(manifest.get("evaluated_rows", [])) != EXPECTED_ROWS_SHA256_INT64:
        raise RuntimeError("B1 evaluated_rows bytes do not match frozen identity")
    if _sha_int64(manifest.get("evaluated_episodes", [])) != EXPECTED_EPISODES_SHA256_INT64:
        raise RuntimeError("B1 evaluated_episodes bytes do not match frozen identity")
    brain = manifest.get("brain", {})
    required_brain = {
        "protocol": "cube_brain_b1_v1", "requested_model": b2.MODEL,
        "thinking": {"type": "disabled"}, "temperature": 0.1,
        "reasoning_effort": "omitted", "stream": False,
        "single_turn_no_history": True,
    }
    for key, expected in required_brain.items():
        if brain.get(key) != expected:
            raise RuntimeError(f"B1 API protocol drift at brain.{key}")
    embedded_assets = {
        "results": manifest.get("results"),
        "baseline_manifest": manifest.get("baseline_pairing", {}).get("manifest"),
        "baseline_results": manifest.get("baseline_pairing", {}).get("results"),
        "fixed_manifest": selection.get("fixed_manifest"),
        "index_metadata": manifest.get("retrieval", {}).get("index"),
    }
    actual_assets = {
        "results": identities[str(B1_RESULTS.resolve())],
        "baseline_manifest": identities[str(BASELINE_RUN_MANIFEST.resolve())],
        "baseline_results": identities[str(BASELINE_RESULTS.resolve())],
        "fixed_manifest": identities[str(FIXED_EVAL_MANIFEST.resolve())],
        "index_metadata": identities[str((INDEX_ROOT / "metadata.json").resolve())],
    }
    if embedded_assets != actual_assets:
        raise RuntimeError("B1 embedded asset identities no longer match frozen files")

    baseline = _load_json(BASELINE_RUN_MANIFEST)
    for key, expected in {
        "format_version": "cube_brain_b1_run_manifest_v1", "status": "complete",
        "mode": "baseline", "goal_offset_steps": 100, "eval_budget": 200,
        "num_eval": 50, "seed": 42,
    }.items():
        if baseline.get(key) != expected:
            raise RuntimeError(f"offset100 baseline protocol drift at {key}")
    if baseline.get("evaluated_rows") != manifest.get("evaluated_rows"):
        raise RuntimeError("B1 replay and offset100 baseline rows are not paired")
    if baseline.get("evaluated_episodes") != manifest.get("evaluated_episodes"):
        raise RuntimeError("B1 replay and offset100 baseline episodes are not paired")

    metadata = _load_json(INDEX_ROOT / "metadata.json")
    if metadata.get("format_version") != 1 or metadata.get("num_anchors") != 1_760_000:
        raise RuntimeError("memory index metadata protocol drift")
    for name, entry in metadata.get("files", {}).items():
        actual = identities.get(str((INDEX_ROOT / name).resolve()))
        if actual is None or entry.get("sha256") != actual["sha256"]:
            raise RuntimeError(f"memory index metadata/file mismatch: {name}")
    dataset_entry = metadata.get("dataset", {})
    dataset_stat = DATASET.stat()
    if (
        Path(dataset_entry.get("path", "")).resolve() != DATASET.resolve()
        or dataset_entry.get("size_bytes") != dataset_stat.st_size
        or dataset_entry.get("mtime_ns") != dataset_stat.st_mtime_ns
    ):
        raise RuntimeError("memory index dataset identity drift (path/size/mtime)")
    identities["dataset_from_index_metadata"] = {
        "path": str(DATASET.resolve()), "size": dataset_stat.st_size,
        "mtime_ns": dataset_stat.st_mtime_ns,
        "sha256": None,
        "sha256_status": "not recorded by frozen index; path/size/mtime bound",
    }

    calls = _load_json(B1_CALLS)
    if not isinstance(calls, list) or len(calls) != 70:
        raise RuntimeError("frozen B1 source must contain exactly 70 calls")
    per_env: Counter[str] = Counter()
    for index, call in enumerate(calls):
        if any((
            call.get("status") != "ok", call.get("protocol_failure") is not False,
            call.get("requested_model") != b2.MODEL, call.get("actual_model") != b2.MODEL,
            call.get("thinking") != {"type": "disabled"},
            call.get("temperature") != 0.1, call.get("reasoning_present") is not False,
            call.get("finish_reason") != "stop",
        )):
            raise RuntimeError(f"B1 call {index} violates frozen clean-call protocol")
        normalized = b1.validate_state_payload(call.get("payload"))
        if normalized["event"] != "STALLED":
            raise RuntimeError("B1 stage0 source event distribution drifted from 70 STALLED")
        if call.get("payload_sha256") != b1.sha256_text(b1.compact_json(normalized)):
            raise RuntimeError(f"B1 call {index} payload SHA mismatch")
        per_env[str(call.get("episode_id"))] += 1
    if len(per_env) != 14 or set(per_env.values()) != {5}:
        raise RuntimeError(f"B1 source call distribution drift: {dict(per_env)}")
    return {
        "identities": identities,
        "semantic_bindings": {
            "b1_status": "complete", "b1_mode": "brain", "offset": 100,
            "budget": 200, "seed": 42, "num_calls": 70,
            "events": {"STALLED": 70, "DROPPED": 0},
            "rows_sha256_int64": EXPECTED_ROWS_SHA256_INT64,
            "episodes_sha256_int64": EXPECTED_EPISODES_SHA256_INT64,
        },
    }


def _wrap_yaw(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def _clamp_position(value: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.clip(value[0], *b1.ID_X),
        np.clip(value[1], *b1.ID_Y),
        np.clip(value[2], *b1.ID_Z),
    ], dtype=np.float64)


def stalled_intents(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen 1/3, 2/3, and reverse-5cm geometric intents."""

    current = np.asarray(state["block"]["pos"], dtype=np.float64)
    target = np.asarray(state["target"]["pos"], dtype=np.float64)
    delta = target - current
    planar_delta = delta.copy()
    planar_delta[2] = 0.0
    planar_norm = float(np.linalg.norm(planar_delta))
    retreat_unit = (
        planar_delta / planar_norm if planar_norm > 1e-12
        else np.asarray([1.0, 0.0, 0.0])
    )
    yaw0 = float(state["block"]["yaw"])
    yaw_delta = _wrap_yaw(float(state["target"]["yaw"]) - yaw0)
    return [
        {
            "intent": "progress_one_third",
            "desired_block_pos": _clamp_position(current + delta / 3.0),
            "desired_yaw": _wrap_yaw(yaw0 + yaw_delta / 3.0),
        },
        {
            "intent": "progress_two_thirds",
            "desired_block_pos": _clamp_position(current + 2.0 * delta / 3.0),
            "desired_yaw": _wrap_yaw(yaw0 + 2.0 * yaw_delta / 3.0),
        },
        {
            "intent": "retreat_five_cm",
            "desired_block_pos": _clamp_position(current - 0.05 * retreat_unit),
            "desired_yaw": _wrap_yaw(yaw0),
        },
    ]


class RealFrameCandidateBuilder:
    """Stable real-anchor retrieval shared by replay and online evaluation."""

    def __init__(
        self,
        index_root: Path,
        dataset: Path,
        fixed_episodes: Sequence[int],
    ) -> None:
        from scipy.spatial import cKDTree

        self.index_root = index_root.expanduser().resolve()
        self.dataset = dataset.expanduser().resolve()
        self.asset_identities = verify_memory_index_assets(self.index_root, self.dataset)
        if _sha_int64([int(value) for value in fixed_episodes]) != EXPECTED_EPISODES_SHA256_INT64:
            raise ValueError("candidate retrieval requires the exact ordered 50 held-out episodes")
        self.fixed_episodes = frozenset(int(value) for value in fixed_episodes)
        if len(self.fixed_episodes) != 50:
            raise ValueError("candidate retrieval requires the exact 50 held-out episodes")
        self.rows = np.load(self.index_root / "anchor_rows.npy", mmap_mode="r")
        self.episodes = np.load(self.index_root / "anchor_episodes.npy", mmap_mode="r")
        self.features = np.load(self.index_root / "anchor_features_z.npy", mmap_mode="r")
        stats = np.load(self.index_root / "stats.npz")
        self.mean = np.asarray(stats["feature_mean"], dtype=np.float64)
        self.std = np.asarray(stats["feature_std"], dtype=np.float64)
        if self.features.shape != (len(self.rows), 9) or self.episodes.shape != self.rows.shape:
            raise RuntimeError("memory index array shape mismatch")
        self.block_dims = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        raw_block = (
            np.asarray(self.features[:, :3], dtype=np.float64) * self.std[:3]
            + self.mean[:3]
        )
        self.raw_block = raw_block
        block_mask = (
            (raw_block[:, 0] >= b1.ID_X[0]) & (raw_block[:, 0] <= b1.ID_X[1])
            & (raw_block[:, 1] >= b1.ID_Y[0]) & (raw_block[:, 1] <= b1.ID_Y[1])
            & (raw_block[:, 2] >= b1.ID_Z[0]) & (raw_block[:, 2] <= b1.ID_Z[1])
        )
        self.block_anchor_indices = np.flatnonzero(block_mask).astype(np.int64)
        if len(self.block_anchor_indices) < 3:
            raise RuntimeError("insufficient ID-box block anchors")
        self.block_tree = cKDTree(
            np.ascontiguousarray(self.features[self.block_anchor_indices][:, self.block_dims]),
            copy_data=False, balanced_tree=True, compact_nodes=True,
        )
        self._recovery_tree: Any | None = None
        self._recovery_anchor_indices: np.ndarray | None = None
        self._recovery_filter_count: int | None = None

    def _raw_feature(self, anchor_index: int) -> np.ndarray:
        return np.asarray(self.features[anchor_index], dtype=np.float64) * self.std + self.mean

    def _block_query(self, position: np.ndarray, yaw: float) -> np.ndarray:
        raw = np.concatenate((position, [math.sin(yaw), math.cos(yaw)]))
        return (raw - self.mean[self.block_dims]) / self.std[self.block_dims]

    def _nearest(
        self,
        tree: Any,
        query: np.ndarray,
        index_map: np.ndarray | None,
        current_episode: int,
        used_episodes: set[int],
        anchor_predicate: Any | None = None,
    ) -> tuple[int, float, bool, str | None]:
        forbidden = set(self.fixed_episodes) | {int(current_episode)}

        def search(disallow_used: bool) -> tuple[int, float] | None:
            k = min(64, int(tree.n))
            while True:
                distances, local_indices = tree.query(query, k=k, eps=0.0, workers=1)
                values: list[tuple[float, int, int, int]] = []
                for distance, local in zip(np.atleast_1d(distances), np.atleast_1d(local_indices)):
                    anchor = int(index_map[int(local)]) if index_map is not None else int(local)
                    episode = int(self.episodes[anchor])
                    if episode in forbidden or (disallow_used and episode in used_episodes):
                        continue
                    if anchor_predicate is not None and not anchor_predicate(anchor):
                        continue
                    values.append((float(distance), int(self.rows[anchor]), episode, anchor))
                if values:
                    provisional = min(values)
                    ball = tree.query_ball_point(
                        query, r=np.nextafter(provisional[0], np.inf), eps=0.0, workers=1
                    )
                    exact: list[tuple[float, int, int, int]] = []
                    for local in ball:
                        anchor = int(index_map[int(local)]) if index_map is not None else int(local)
                        episode = int(self.episodes[anchor])
                        if episode in forbidden or (disallow_used and episode in used_episodes):
                            continue
                        if anchor_predicate is not None and not anchor_predicate(anchor):
                            continue
                        distance = float(np.linalg.norm(np.asarray(tree.data[int(local)]) - query))
                        exact.append((distance, int(self.rows[anchor]), episode, anchor))
                    if exact:
                        best = min(exact)
                        return best[3], best[0]
                if k >= int(tree.n):
                    return None
                k = min(k * 2, int(tree.n))

        selected = search(True)
        if selected is not None:
            return selected[0], selected[1], False, None
        raise RuntimeError("no allowed real-frame candidate with a distinct source episode")

    def _candidate(
        self,
        candidate_id: int,
        anchor: int,
        target: np.ndarray,
        current_position: np.ndarray,
        current_distance: float,
        *,
        include_ee: bool,
    ) -> dict[str, Any]:
        raw = self._raw_feature(anchor)
        block_pos = raw[:3]
        yaw = math.atan2(float(raw[3]), float(raw[4]))
        result: dict[str, Any] = {
            "candidate_id": candidate_id,
            "anchor_row": int(self.rows[anchor]),
            "source_episode": int(self.episodes[anchor]),
            "block_pos": np.round(block_pos, 6).tolist(),
            "yaw": round(yaw, 6),
            "dist_to_target": round(
                float(np.linalg.norm(np.round(block_pos, 6) - target)), 6
            ),
            "current_to_candidate_distance": round(
                float(np.linalg.norm(np.round(block_pos, 6) - current_position)), 6
            ),
            "retreat_then_advance": False,
        }
        result["retreat_then_advance"] = (
            float(result["dist_to_target"]) > current_distance + 1e-6
        )
        if include_ee:
            result["ee_pos"] = np.round(raw[5:8], 6).tolist()
        return result

    def _nearest_retreat(
        self,
        desired: np.ndarray,
        current: np.ndarray,
        target: np.ndarray,
        target_unit: np.ndarray,
        current_distance: float,
        current_episode: int,
        used_episodes: set[int],
    ) -> tuple[int, float, bool, str | None]:
        displacement = self.raw_block - current
        displacement_norm = np.linalg.norm(displacement, axis=1)
        target_distance = np.linalg.norm(self.raw_block - target, axis=1)
        episodes = np.asarray(self.episodes, dtype=np.int64)
        forbidden = np.asarray(
            sorted(set(self.fixed_episodes) | {int(current_episode)} | set(used_episodes)),
            dtype=np.int64,
        )
        base = (
            (displacement_norm >= RETREAT_MIN_DISPLACEMENT_M)
            & (target_distance > current_distance + 1e-6)
            & ~np.isin(episodes, forbidden)
        )
        within_preferred_range = base & (
            displacement_norm <= RETREAT_PREFERRED_MAX_DISPLACEMENT_M
        )
        strict = within_preferred_range & (
            (displacement @ target_unit) <= -RETREAT_PREFERRED_REVERSE_PROJECTION_M
        )
        if np.any(strict):
            eligible_mask = strict
            fallback = False
            reason = None
        elif np.any(within_preferred_range):
            eligible_mask = within_preferred_range
            fallback = True
            reason = "no_strict_reverse_anchor"
        else:
            eligible_mask = base
            fallback = True
            reason = "no_qualified_detour_within_8cm"
        eligible = np.flatnonzero(eligible_mask)
        if len(eligible) == 0:
            raise RuntimeError("no qualified real-frame retreat candidate")
        distance = np.linalg.norm(self.raw_block[eligible] - desired, axis=1)
        minimum = float(np.min(distance))
        ties = eligible[distance == minimum]
        anchor = min((int(value) for value in ties), key=lambda value: int(self.rows[value]))
        return anchor, minimum, fallback, reason

    def build_stalled(
        self,
        state: Mapping[str, Any],
        current_episode: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        target = np.asarray(state["target"]["pos"], dtype=np.float64)
        current = np.asarray(state["block"]["pos"], dtype=np.float64)
        direction = target - current
        direction[2] = 0.0
        direction_norm = float(np.linalg.norm(direction))
        target_unit = (
            direction / direction_norm if direction_norm > 1e-12
            else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        )
        current_distance = float(state["dist_to_target"])
        used: set[int] = set()
        candidates: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        for candidate_id, intent in enumerate(stalled_intents(state)):
            query = self._block_query(intent["desired_block_pos"], float(intent["desired_yaw"]))
            if intent["intent"] == "retreat_five_cm":
                anchor, distance_z, fallback, reason = self._nearest_retreat(
                    np.asarray(intent["desired_block_pos"], dtype=np.float64),
                    current, target, target_unit, current_distance, current_episode, used,
                )
            else:
                anchor, distance_z, fallback, reason = self._nearest(
                    self.block_tree, query, self.block_anchor_indices,
                    current_episode, used,
                )
            candidate = self._candidate(
                candidate_id, anchor, target, current, current_distance, include_ee=False
            )
            if intent["intent"] == "retreat_five_cm":
                candidate["intent"] = "detour" if fallback else "retreat"
                candidate["strict_reverse"] = not fallback
                candidate["fallback_reason"] = reason
            else:
                candidate["intent"] = intent["intent"]
                candidate["strict_reverse"] = False
                candidate["fallback_reason"] = None
            used.add(int(candidate["source_episode"]))
            candidates.append(candidate)
            audit.append({
                "candidate_id": candidate_id,
                "intent": candidate["intent"],
                "strict_reverse": candidate["strict_reverse"],
                "desired_block_pos": np.round(intent["desired_block_pos"], 6).tolist(),
                "desired_yaw": round(float(intent["desired_yaw"]), 6),
                "retrieval_distance_z": distance_z,
                "retrieval_metric": (
                    "physical_block_xyz_m" if intent["intent"] == "retreat_five_cm"
                    else "standardized_block_xyz_sinyaw_cosyaw"
                ),
                "current_to_candidate_distance": candidate["current_to_candidate_distance"],
                "signed_progress_projection_m": round(float(np.dot(
                    np.asarray(candidate["block_pos"], dtype=np.float64) - current,
                    target_unit,
                )), 6),
                "retreat_requirements": (
                    {
                        "min_current_displacement_m": RETREAT_MIN_DISPLACEMENT_M,
                        "preferred_max_current_displacement_m": (
                            RETREAT_PREFERRED_MAX_DISPLACEMENT_M
                        ),
                        "preferred_minimum_reverse_projection_m": (
                            RETREAT_PREFERRED_REVERSE_PROJECTION_M
                        ),
                        "must_increase_final_target_distance": True,
                        "unqualified_fallback_forbidden": True,
                    }
                    if intent["intent"] == "retreat_five_cm" else None
                ),
                "fallback_used": fallback,
                "fallback_reason": reason,
                "anchor_row": candidate["anchor_row"],
                "source_episode": candidate["source_episode"],
            })
        normalized = b2.validate_candidate_payload(
            {"state": state, "retrieval_candidates": candidates}
        )["retrieval_candidates"]
        return normalized, audit

    def _ensure_recovery_tree(self) -> None:
        if self._recovery_tree is not None:
            return
        import hdf5plugin  # noqa: F401 - required before h5py reads compressed arrays
        import h5py
        from scipy.spatial import cKDTree

        with h5py.File(self.dataset, "r") as h5:
            block = np.asarray(h5["privileged_block_0_pos"], dtype=np.float64)
            ee = np.asarray(h5["proprio_effector_pos"], dtype=np.float64)
            contact = np.asarray(h5["proprio_gripper_contact"], dtype=np.float64).reshape(-1)
            opening = np.asarray(h5["proprio_gripper_opening"], dtype=np.float64).reshape(-1)
        rows = np.asarray(self.rows, dtype=np.int64)
        distance = np.linalg.norm(ee[rows] - block[rows], axis=1)
        mask = (
            (distance <= RECOVERY_EE_BLOCK_MAX_M)
            & (contact[rows] >= RECOVERY_CONTACT_MIN)
            & (opening[rows] <= RECOVERY_OPENING_MAX)
            & ~np.isin(np.asarray(self.episodes, dtype=np.int64), np.asarray(sorted(self.fixed_episodes)))
        )
        indices = np.flatnonzero(mask).astype(np.int64)
        if len(indices) < 3:
            raise RuntimeError(f"insufficient contact-qualified recovery anchors: {len(indices)}")
        self._recovery_anchor_indices = indices
        self._recovery_filter_count = int(len(indices))
        self._recovery_tree = cKDTree(
            np.ascontiguousarray(self.features[indices]),
            copy_data=False, balanced_tree=True, compact_nodes=True,
        )

    def build_dropped(
        self,
        state: Mapping[str, Any],
        current_episode: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._ensure_recovery_tree()
        assert self._recovery_tree is not None and self._recovery_anchor_indices is not None
        raw_query = np.asarray([
            *state["block"]["pos"],
            math.sin(float(state["block"]["yaw"])),
            math.cos(float(state["block"]["yaw"])),
            *state["ee_pos"],
            RECOVERY_OPENING_MAX,
        ], dtype=np.float64)
        query = (raw_query - self.mean) / self.std
        target = np.asarray(state["target"]["pos"], dtype=np.float64)
        current = np.asarray(state["block"]["pos"], dtype=np.float64)
        current_distance = float(state["dist_to_target"])
        used: set[int] = set()
        candidates: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []

        def inside_recovery(anchor: int) -> bool:
            raw = self._raw_feature(anchor)
            return _inside(raw[:3]) and _inside(raw[5:8])

        for candidate_id in range(3):
            anchor, distance_z, fallback, reason = self._nearest(
                self._recovery_tree, query, self._recovery_anchor_indices,
                current_episode, used, inside_recovery,
            )
            candidate = self._candidate(
                candidate_id, anchor, target, current, current_distance, include_ee=True
            )
            used.add(int(candidate["source_episode"]))
            candidates.append(candidate)
            audit.append({
                "candidate_id": candidate_id,
                "intent": "contact_qualified_recovery",
                "retrieval_distance_z": distance_z,
                "fallback_used": fallback,
                "fallback_reason": reason,
                "anchor_row": candidate["anchor_row"],
                "source_episode": candidate["source_episode"],
                "recovery_filter": {
                    "ee_block_max_m": RECOVERY_EE_BLOCK_MAX_M,
                    "contact_min": RECOVERY_CONTACT_MIN,
                    "opening_max": RECOVERY_OPENING_MAX,
                    "qualified_anchor_count": self._recovery_filter_count,
                    "fixed_50_excluded_in_pool": True,
                    "selection_requires_block_and_ee_inside_frozen_id_box": True,
                },
            })
        normalized = b2.validate_candidate_payload(
            {"state": state, "retrieval_candidates": candidates}
        )["retrieval_candidates"]
        return normalized, audit

    def build_candidates(
        self,
        state: Mapping[str, Any],
        current_episode: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        normalized_state = b1.validate_state_payload(state)
        if normalized_state["event"] == "STALLED":
            return self.build_stalled(normalized_state, current_episode)
        return self.build_dropped(normalized_state, current_episode)

    def resolve_landing(
        self,
        state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        decision: Mapping[str, Any],
        current_episode: int,
    ) -> dict[str, Any]:
        """Resolve a supervisor decision to one concrete real dataset frame.

        Exact candidate geometry preserves that candidate's ``anchor_row``.
        Any numeric adjustment is re-retrieved once from the same filtered
        index; callers must load the returned row's real pixels as the goal.
        """

        payload = b2.validate_candidate_payload({
            "state": state,
            "retrieval_candidates": list(candidates),
        })
        action = decision.get("decision")
        if action == "CONTINUE":
            return {
                "decision": "CONTINUE",
                "anchor_row": None,
                "source_episode": None,
                "reretrieval_performed": False,
                "landing_contract": "no_goal_change",
            }
        if action == "SUBGOAL":
            visible = {
                "decision": action,
                "candidate_id": decision.get("candidate_id"),
                "block_pos": decision.get("block_pos"),
                "yaw": decision.get("yaw"),
            }
        elif action == "RECOVER":
            visible = {
                "decision": action,
                "candidate_id": decision.get("candidate_id"),
                "strategy": decision.get("strategy"),
                "ee_pos": decision.get("ee_pos"),
            }
        else:
            raise b2.ProtocolError("landing decision must be CONTINUE, SUBGOAL, or RECOVER")
        validation = b2.validate_response(
            b1.compact_json(visible), payload["state"]["event"],
            payload["retrieval_candidates"],
        )
        if not validation.valid or validation.decision["decision"] == "CONTINUE":
            raise b2.ProtocolError(validation.error or "invalid intervention landing")
        applied, guard = b2.guard_and_bind(
            validation.decision, payload["retrieval_candidates"]
        )
        candidate_id = int(applied["candidate_id"])
        selected = payload["retrieval_candidates"][candidate_id]
        base = {
            "decision": action,
            "candidate_id": candidate_id,
            "selected_anchor_row": int(selected["anchor_row"]),
            "selected_source_episode": int(selected["source_episode"]),
            "requested_decision": guard["requested_decision"],
            "applied_decision": guard["applied_decision"],
            "clamp_applied": bool(guard["clamp_applied"]),
            "adjustment_clamp_applied": bool(guard["adjustment_clamp_applied"]),
            "yaw_wrap_applied": bool(guard["yaw_wrap_applied"]),
            "requested_position_delta_l2_m": guard["requested_position_delta_l2_m"],
            "applied_position_delta_l2_m": guard["applied_position_delta_l2_m"],
            "requested_yaw_delta_rad": guard["requested_yaw_delta_rad"],
            "applied_yaw_delta_rad": guard["applied_yaw_delta_rad"],
        }
        if not guard["reretrieve_required"]:
            return {
                **base,
                "anchor_row": int(selected["anchor_row"]),
                "source_episode": int(selected["source_episode"]),
                "block_pos": list(selected["block_pos"]),
                "yaw": float(selected["yaw"]),
                **({"ee_pos": list(selected["ee_pos"])} if action == "RECOVER" else {}),
                "retrieval_distance_z": 0.0,
                "reretrieval_performed": False,
                "landing_contract": "use_selected_anchor_row",
            }

        target = np.asarray(payload["state"]["target"]["pos"], dtype=np.float64)
        current = np.asarray(payload["state"]["block"]["pos"], dtype=np.float64)
        current_distance = float(payload["state"]["dist_to_target"])
        if action == "SUBGOAL":
            query = self._block_query(
                np.asarray(applied["block_pos"], dtype=np.float64), float(applied["yaw"])
            )
            anchor, distance_z, fallback, reason = self._nearest(
                self.block_tree, query, self.block_anchor_indices,
                int(current_episode), set(),
            )
            resolved = self._candidate(
                candidate_id, anchor, target, current, current_distance, include_ee=False
            )
        else:
            self._ensure_recovery_tree()
            assert self._recovery_tree is not None and self._recovery_anchor_indices is not None
            raw_query = np.asarray([
                *selected["block_pos"], math.sin(float(selected["yaw"])),
                math.cos(float(selected["yaw"])), *applied["ee_pos"],
                RECOVERY_OPENING_MAX,
            ], dtype=np.float64)
            query = (raw_query - self.mean) / self.std

            def inside_recovery(anchor_index: int) -> bool:
                raw = self._raw_feature(anchor_index)
                return _inside(raw[:3]) and _inside(raw[5:8])

            anchor, distance_z, fallback, reason = self._nearest(
                self._recovery_tree, query, self._recovery_anchor_indices,
                int(current_episode), set(), inside_recovery,
            )
            resolved = self._candidate(
                candidate_id, anchor, target, current, current_distance, include_ee=True
            )
        return {
            **base,
            "anchor_row": int(resolved["anchor_row"]),
            "source_episode": int(resolved["source_episode"]),
            "block_pos": list(resolved["block_pos"]),
            "yaw": float(resolved["yaw"]),
            **({"ee_pos": list(resolved["ee_pos"])} if action == "RECOVER" else {}),
            "retrieval_distance_z": float(distance_z),
            "reretrieval_performed": True,
            "landing_contract": "reretrieve_real_frame",
            "fallback_used": bool(fallback),
            "fallback_reason": reason,
        }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def prepare_inputs() -> dict[str, Any]:
    source_preflight = verify_frozen_sources()
    calls = _load_json(B1_CALLS)
    run_manifest = _load_json(B1_RUN_MANIFEST)
    if not isinstance(calls, list) or len(calls) != 70:
        raise RuntimeError("B1 replay source must contain exactly 70 calls")
    fixed = [int(value) for value in run_manifest["evaluated_episodes"]]
    builder = RealFrameCandidateBuilder(INDEX_ROOT, DATASET, fixed)
    frozen_inputs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    per_episode = Counter(str(row["episode_id"]) for row in calls)
    if len(per_episode) != 14 or set(per_episode.values()) != {5}:
        raise RuntimeError(f"B1 source call distribution drift: {dict(per_episode)}")
    for index, call in enumerate(calls):
        if call.get("requested_model") != b2.MODEL or call.get("status") != "ok":
            raise RuntimeError(f"B1 call {index} is not a clean frozen source")
        state = b1.validate_state_payload(call["payload"])
        env_idx = int(call["episode_id"])
        if not 0 <= env_idx < len(fixed):
            raise RuntimeError("B1 call episode_id is not a valid formal env index")
        episode = int(fixed[env_idx])
        candidates, candidate_audit = builder.build_candidates(state, episode)
        payload = b2.validate_candidate_payload(
            {"state": state, "retrieval_candidates": candidates}
        )
        replay_id = (
            f"b1_{index:03d}_env{env_idx}_ep{episode}_call{int(call['logical_call_index'])}"
        )
        frozen_inputs.append({
            "replay_id": replay_id,
            "env_idx": env_idx,
            "episode_id": str(episode),
            "b1_payload_sha256": call["payload_sha256"],
            "payload": payload,
        })
        audits.append({"replay_id": replay_id, "candidates": candidate_audit})
    builder._ensure_recovery_tree()
    assert builder._recovery_anchor_indices is not None
    recovery_episode_count = len(np.unique(
        np.asarray(builder.episodes[builder._recovery_anchor_indices], dtype=np.int64)
    ))
    retreat_breakdown = Counter(
        (
            row["candidates"][2]["intent"],
            row["candidates"][2]["fallback_reason"],
        )
        for row in audits
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_json(frozen_inputs)
    input_sha = b1.sha256_text(serialized)
    manifest = {
        "format_version": "cube_brain_b2_frozen_inputs_v1",
        "num_inputs": 70,
        "source_calls": _identity(B1_CALLS),
        "source_run_manifest": _identity(B1_RUN_MANIFEST),
        "source_preflight": source_preflight,
        "index": {
            name: _identity(INDEX_ROOT / name)
            for name in (
                "metadata.json", "anchor_rows.npy", "anchor_episodes.npy",
                "anchor_features_z.npy", "stats.npz",
            )
        },
        "fixed_episodes": fixed,
        "fixed_episodes_sha256": b1.sha256_text(b1.compact_json(fixed)),
        "fixed_episodes_sha256_int64": _sha_int64(fixed),
        "frozen_inputs_sha256": input_sha,
        "candidate_algorithm": {
            "STALLED": ["progress_one_third", "progress_two_thirds", "retreat_five_cm"],
            "STALLED_retreat": {
                "desired_displacement_m": 0.05,
                "minimum_actual_current_displacement_m": RETREAT_MIN_DISPLACEMENT_M,
                "preferred_maximum_actual_current_displacement_m": (
                    RETREAT_PREFERRED_MAX_DISPLACEMENT_M
                ),
                "desired_reverse_axis": "current-to-final planar xy unit vector",
                "preferred_minimum_reverse_projection_m": (
                    RETREAT_PREFERRED_REVERSE_PROJECTION_M
                ),
                "directional_fallback": (
                    "if strict reverse is absent, use nearest candidate that still satisfies "
                    "the displacement and farther-from-final qualifications; record fallback"
                ),
                "must_increase_final_target_distance": True,
                "nearest_metric": "physical block xyz meters then anchor row",
                "unqualified_fallback_forbidden": True,
            },
            "DROPPED_filter": {
                "ee_block_max_m": RECOVERY_EE_BLOCK_MAX_M,
                "contact_min": RECOVERY_CONTACT_MIN,
                "opening_max": RECOVERY_OPENING_MAX,
                "selection_requires_block_and_ee_inside_frozen_id_box": True,
                "fixed_50_excluded_in_pool": True,
                "qualified_anchor_count": builder._recovery_filter_count,
                "qualified_source_episode_count": recovery_episode_count,
            },
            "STALLED_filter": "retrieved block position inside frozen ID box",
            "stable_order": "intent order then exact nearest stable (distance,row)",
            "distinct_source_episode": "required; fail closed if unavailable",
            "exclusions": "all fixed 50 episodes plus current episode",
        },
        "id_box": {"x": list(b1.ID_X), "y": list(b1.ID_Y), "z": list(b1.ID_Z)},
        "all_candidate_rows_distinct": all(
            len({item["anchor_row"] for item in row["payload"]["retrieval_candidates"]}) == 3
            for row in frozen_inputs
        ),
        "all_candidate_episodes_distinct": all(
            len({item["source_episode"] for item in row["payload"]["retrieval_candidates"]}) == 3
            for row in frozen_inputs
        ),
        "fallback_count": sum(
            int(candidate["fallback_used"])
            for row in audits for candidate in row["candidates"]
        ),
        "retreat_candidate_breakdown": {
            (
                intent if reason is None else f"{intent}:{reason}"
            ): count
            for (intent, reason), count in sorted(
                retreat_breakdown.items(), key=lambda item: str(item[0])
            )
        },
    }
    for path, value in (
        (INPUTS_PATH, frozen_inputs),
        (CANDIDATE_AUDIT_PATH, audits),
        (INPUT_MANIFEST_PATH, manifest),
    ):
        if path.exists():
            existing = _load_json(path)
            if existing != value:
                raise RuntimeError(f"frozen B2 input artifact drift: {path}")
        else:
            b1.atomic_write_json(path, value)
    return manifest


def preflight_frozen_inputs() -> dict[str, Any]:
    """Rebuild and compare every frozen input before any external request."""

    manifest = prepare_inputs()
    inputs = _load_json(INPUTS_PATH)
    audits = _load_json(CANDIDATE_AUDIT_PATH)
    if not isinstance(inputs, list) or len(inputs) != 70:
        raise RuntimeError("frozen input preflight requires exactly 70 inputs")
    if not isinstance(audits, list) or len(audits) != 70:
        raise RuntimeError("candidate audit preflight requires exactly 70 rows")
    if b1.sha256_text(_canonical_json(inputs)) != manifest.get("frozen_inputs_sha256"):
        raise RuntimeError("frozen input semantic SHA mismatch")
    calls = _load_json(B1_CALLS)
    fixed = manifest["fixed_episodes"]
    seen_replay_ids: set[str] = set()
    for index, (row, audit) in enumerate(zip(inputs, audits)):
        replay_id = row.get("replay_id")
        if not isinstance(replay_id, str) or replay_id in seen_replay_ids:
            raise RuntimeError("frozen replay ids must be unique strings")
        seen_replay_ids.add(replay_id)
        if audit.get("replay_id") != replay_id:
            raise RuntimeError(f"candidate audit replay id mismatch at {index}")
        payload = b2.validate_candidate_payload(row.get("payload"))
        env_idx = int(row.get("env_idx"))
        if row.get("episode_id") != str(fixed[env_idx]):
            raise RuntimeError(f"frozen input episode mapping mismatch at {index}")
        if row.get("b1_payload_sha256") != calls[index].get("payload_sha256"):
            raise RuntimeError(f"frozen B1 payload binding mismatch at {index}")
        public = payload["retrieval_candidates"]
        audited = audit.get("candidates")
        if not isinstance(audited, list) or len(audited) != 3:
            raise RuntimeError(f"candidate audit shape mismatch at {index}")
        for candidate, evidence in zip(public, audited):
            if (
                candidate["candidate_id"] != evidence.get("candidate_id")
                or candidate["anchor_row"] != evidence.get("anchor_row")
                or candidate["source_episode"] != evidence.get("source_episode")
            ):
                raise RuntimeError(f"candidate/audit binding mismatch at {index}")
    return {
        "manifest": manifest,
        "inputs": inputs,
        "artifact_identities": {
            "frozen_inputs": _identity(INPUTS_PATH),
            "input_manifest": _identity(INPUT_MANIFEST_PATH),
            "candidate_audit": _identity(CANDIDATE_AUDIT_PATH),
        },
    }


def validate_previous_round(round_number: int, input_identities: Mapping[str, Any]) -> int:
    if round_number == 1:
        return 0
    previous_number = round_number - 1
    previous = OUTPUT_ROOT / f"round_{previous_number}"
    names = (*ROUND_ARTIFACT_FILES.values(), "round_manifest.json")
    paths = {name: previous / name for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"previous prompt round incomplete: {missing}")
    round_manifest = _load_json(paths["round_manifest.json"])
    if (
        round_manifest.get("format_version") != "cube_brain_b2_prompt_round_v1"
        or round_manifest.get("round") != previous_number
        or round_manifest.get("prompt_version") != previous_number
        or round_manifest.get("prompt_sha256")
        != b1.sha256_text(b2.PROMPTS[previous_number])
    ):
        raise RuntimeError("previous round manifest protocol drift")
    if round_manifest.get("frozen_input_artifacts") != dict(input_identities):
        raise RuntimeError("previous round does not bind the current frozen inputs")
    expected_code = {
        "supervisor_v2": _identity(Path(b2.__file__).resolve()),
        "replay_tool": _identity(Path(__file__).resolve()),
    }
    if round_manifest.get("code") != expected_code:
        raise RuntimeError("previous round code identity drift")
    artifacts = round_manifest.get("round_artifacts")
    expected_artifacts = {
        key: _identity(paths[name]) for key, name in ROUND_ARTIFACT_FILES.items()
    }
    if artifacts != expected_artifacts:
        raise RuntimeError("previous round artifact identity drift")
    acceptance = _load_json(paths["acceptance.json"])
    summary = _load_json(paths["summary.json"])
    if acceptance.get("passed"):
        raise RuntimeError("previous prompt passed; further iteration is forbidden")
    if acceptance.get("num_final_calls") != 70 or summary.get("logical_calls") != 70:
        raise RuntimeError("previous prompt round did not complete exact70 logical calls")
    return int(summary["budget"]["accounted_tokens"])


def acceptance_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finals = [record for record in records if record.get("status") != "transport_retry"]
    valid = [record for record in finals if record.get("status") == "ok"]
    decisions = Counter(record.get("applied_decision", {}).get("decision") for record in finals)
    coordinates_inside = True
    for record in valid:
        # Prompt acceptance evaluates what the model requested, before the
        # runtime safety clamp; otherwise an out-of-box prompt regression would
        # be hidden by the controller guard.
        decision = record["validated_response"]
        if decision["decision"] == "SUBGOAL":
            coordinates_inside &= _inside(decision["block_pos"])
        elif decision["decision"] == "RECOVER":
            coordinates_inside &= _inside(decision["ee_pos"])
    legal_fraction = len(valid) / 70.0
    non_continue = (len(finals) - decisions.get("CONTINUE", 0)) / 70.0
    checks = {
        "exactly_70_final_calls": len(finals) == 70,
        "legal_json_fraction_1_0": math.isclose(legal_fraction, 1.0),
        "non_continue_fraction_at_least_0_70": non_continue >= 0.70,
        "all_intervention_coordinates_in_id_box": bool(coordinates_inside),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "num_final_calls": len(finals),
        "num_valid": len(valid),
        "legal_json_fraction": legal_fraction,
        "non_continue_fraction": non_continue,
        "decision_counts": dict(decisions),
        "status_counts": dict(Counter(record.get("status") for record in records)),
    }


def _inside(position: Sequence[float]) -> bool:
    return (
        b1.ID_X[0] <= position[0] <= b1.ID_X[1]
        and b1.ID_Y[0] <= position[1] <= b1.ID_Y[1]
        and b1.ID_Z[0] <= position[2] <= b1.ID_Z[1]
    )


def execute_round(round_number: int) -> dict[str, Any]:
    if os.environ.get(AUTH_ENV) != "YES":
        raise RuntimeError(f"real API is locked; set {AUTH_ENV}=YES only after Leader authorization")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    output = OUTPUT_ROOT / f"round_{round_number}"
    if output.exists():
        raise FileExistsError(f"B2 round output must be a new isolated directory: {output}")
    preflight = preflight_frozen_inputs()
    inputs = preflight["inputs"]
    initial = validate_previous_round(round_number, preflight["artifact_identities"])
    request_upper_bounds = []
    placeholder = b2.BrainSupervisorV2.__new__(b2.BrainSupervisorV2)
    placeholder.config = b2.B2Config(prompt_version=round_number)  # type: ignore[attr-defined]
    placeholder.endpoint = b1.safe_endpoint(b2.DEFAULT_BASE_URL)  # type: ignore[attr-defined]
    for row in inputs:
        body = b2.BrainSupervisorV2.request_body(placeholder, row["payload"])
        request_upper_bounds.append(
            b1.conservative_token_upper_bound(b1.compact_json(body)) + b2.MAX_TOKENS
        )
    if initial + sum(request_upper_bounds) > b2.MAX_TOTAL_TOKENS:
        raise b1.BudgetExceeded(
            "insufficient cumulative budget for one attempt on all 70 frozen inputs"
        )
    supervisor = b2.BrainSupervisorV2(
        b2.B2Config(prompt_version=round_number, initial_tokens=initial), output
    )
    prompt_path = output / "system_prompt.txt"
    descriptor = os.open(prompt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(b2.PROMPTS[round_number] + "\n")
    results: list[dict[str, Any]] = []
    for row in inputs:
        result = supervisor.decide(
            row["payload"]["state"], row["payload"]["retrieval_candidates"],
            replay_id=row["replay_id"], episode_id=row["episode_id"],
        )
        results.append({"replay_id": row["replay_id"], "decision": result})
    acceptance = acceptance_from_records(supervisor.records)
    b1.atomic_write_json(output / "decisions.json", results)
    b1.atomic_write_json(output / "acceptance.json", acceptance)
    source_postflight = verify_frozen_sources()
    if source_postflight != preflight["manifest"]["source_preflight"]:
        raise RuntimeError("frozen B1/index source changed during API replay")
    post_input_identities = {
        "frozen_inputs": _identity(INPUTS_PATH),
        "input_manifest": _identity(INPUT_MANIFEST_PATH),
        "candidate_audit": _identity(CANDIDATE_AUDIT_PATH),
    }
    if post_input_identities != preflight["artifact_identities"]:
        raise RuntimeError("frozen B2 input artifacts changed during API replay")
    round_artifacts = {
        key: _identity(output / name) for key, name in ROUND_ARTIFACT_FILES.items()
    }
    b1.atomic_write_json(output / "round_manifest.json", {
        "format_version": "cube_brain_b2_prompt_round_v1",
        "round": round_number,
        "prompt_version": round_number,
        "prompt_sha256": b1.sha256_text(b2.PROMPTS[round_number]),
        "frozen_input_artifacts": post_input_identities,
        "code": {
            "supervisor_v2": _identity(Path(b2.__file__).resolve()),
            "replay_tool": _identity(Path(__file__).resolve()),
        },
        "api": supervisor.manifest(),
        "budget_preflight": {
            "initial_tokens": initial,
            "one_attempt_round_upper": sum(request_upper_bounds),
            "remaining_before_round": b2.MAX_TOTAL_TOKENS - initial,
            "sufficient_for_all_70_one_attempt": True,
        },
        "source_preflight": preflight["manifest"]["source_preflight"],
        "source_postflight_equal": True,
        "acceptance": acceptance,
        "summary": supervisor.summary(),
        "round_artifacts": round_artifacts,
        "completed_utc": b1.utc_now(),
    })
    return acceptance


def self_test() -> None:
    state = b2._synthetic_payload()["state"]
    intents = stalled_intents(state)
    assert [item["intent"] for item in intents] == [
        "progress_one_third", "progress_two_thirds", "retreat_five_cm"
    ]
    assert np.allclose(intents[0]["desired_block_pos"], [0.4166666667, 0.0, 0.02])
    assert np.allclose(intents[1]["desired_block_pos"], [0.4833333333, 0.0, 0.02])
    assert np.allclose(intents[2]["desired_block_pos"], [0.30, 0.0, 0.02])
    records = []
    for index in range(70):
        decision = (
            {"decision": "SUBGOAL", "candidate_id": 0, "block_pos": [0.4, 0, 0.02], "yaw": 0}
            if index < 49 else {"decision": "CONTINUE", "reason": "specific synthetic reason"}
        )
        records.append({
            "status": "ok", "validated_response": decision,
            "applied_decision": decision,
        })
    acceptance = acceptance_from_records(records)
    assert acceptance["passed"] and acceptance["non_continue_fraction"] == 0.7
    records[-1]["status"] = "protocol_failure"
    assert not acceptance_from_records(records)["passed"]
    assert AUTH_ENV == "BRAIN_B2_REAL_API_AUTHORIZED"

    scratch = Path("/root/autodl-tmp/tmp")
    scratch.mkdir(parents=True, exist_ok=True)
    original_output_root = OUTPUT_ROOT
    try:
        with tempfile.TemporaryDirectory(prefix="brain-b2-round-binding-", dir=scratch) as directory:
            globals()["OUTPUT_ROOT"] = Path(directory)
            previous = Path(directory) / "round_1"
            previous.mkdir()
            b1.atomic_write_json(previous / "api_manifest.json", {"requested_model": b2.MODEL})
            b1.atomic_write_json(previous / "llm_calls.json", [{"status": "ok"}] * 70)
            b1.atomic_write_json(
                previous / "summary.json",
                {"logical_calls": 70, "budget": {"accounted_tokens": 321}},
            )
            (previous / "system_prompt.txt").write_text(
                b2.PROMPTS[1] + "\n", encoding="utf-8"
            )
            b1.atomic_write_json(previous / "decisions.json", [{}] * 70)
            b1.atomic_write_json(
                previous / "acceptance.json",
                {"passed": False, "num_final_calls": 70},
            )
            input_identities = {"frozen_inputs": {"sha256": "synthetic"}}
            round_artifacts = {
                key: _identity(previous / name)
                for key, name in ROUND_ARTIFACT_FILES.items()
            }
            round_manifest = {
                "format_version": "cube_brain_b2_prompt_round_v1",
                "round": 1,
                "prompt_version": 1,
                "prompt_sha256": b1.sha256_text(b2.PROMPTS[1]),
                "frozen_input_artifacts": input_identities,
                "code": {
                    "supervisor_v2": _identity(Path(b2.__file__).resolve()),
                    "replay_tool": _identity(Path(__file__).resolve()),
                },
                "round_artifacts": round_artifacts,
            }
            b1.atomic_write_json(previous / "round_manifest.json", round_manifest)
            assert validate_previous_round(2, input_identities) == 321
            round_manifest["round_artifacts"].pop("system_prompt_text")
            b1.atomic_write_json(previous / "round_manifest.json", round_manifest)
            try:
                validate_previous_round(2, input_identities)
            except RuntimeError as exc:
                assert "artifact identity drift" in str(exc)
            else:
                raise AssertionError("tampered prior-round artifact map was accepted")
    finally:
        globals()["OUTPUT_ROOT"] = original_output_root
    print("replay_brain_b2_prompts self-test: PASS")


def dry_run(round_number: int) -> None:
    print(b1.compact_json({
        "round": round_number,
        "prompt_sha256": b1.sha256_text(b2.PROMPTS[round_number]),
        "source_calls": str(B1_CALLS),
        "source_call_count_expected": 70,
        "index": str(INDEX_ROOT),
        "output": str(OUTPUT_ROOT / f"round_{round_number}"),
        "real_api_authorization_env": AUTH_ENV,
        "real_api_authorized_now": os.environ.get(AUTH_ENV) == "YES",
        "external_request_sent": False,
    }))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute-api", action="store_true")
    value.add_argument("--round", type=int, choices=b2.PROMPT_VERSIONS, default=1)
    return value


def main(args: argparse.Namespace) -> int:
    if args.self_test:
        self_test()
    elif args.dry_run:
        dry_run(args.round)
    elif args.prepare:
        print(json.dumps(prepare_inputs(), indent=2, sort_keys=True))
    else:
        acceptance = execute_round(args.round)
        print(json.dumps(acceptance, indent=2, sort_keys=True))
        return 0 if acceptance["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
