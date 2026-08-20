#!/usr/bin/env python3
"""B2 goal-conditioned slow supervisor and strict DeepSeek transport.

This module is deliberately separate from the frozen B1 supervisor.  B2 adds
three real-frame geometric candidates, justified CONTINUE, and an auditable
candidate-row landing contract.  ``--self-test`` and ``--dry-run`` never read
an API key or send a network request.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import brain_supervisor as b1
except ImportError:
    import brain_supervisor as b1  # type: ignore[no-redef]


MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
THINKING = {"type": "disabled"}
TEMPERATURE = 0.1
MAX_TOKENS = 256
MAX_TOTAL_TOKENS = 1_000_000
MAX_LOGICAL_CALLS_PER_EPISODE = 5
MAX_TRANSPORT_RETRIES = 1
DEFAULT_TIMEOUT_SECONDS = 30.0
RETRYABLE_HTTP = frozenset({408, 409, 425, 429})
RECOVER_STRATEGIES = frozenset({"REGRASP", "LIFT_AND_RETRY", "REAPPROACH"})
PROMPT_VERSIONS = (1, 2, 3)
MAX_CONTINUE_REASON_CHARS = 240
FINE_TUNE_TOLERANCE = 1e-6
MAX_FINE_TUNE_POSITION_M = 0.03
MAX_FINE_TUNE_YAW_RAD = math.pi / 12.0


_BASE_POLICY = """You are the B2 slow supervisory controller for a cube task.
Input is one JSON object containing symbolic state and exactly three REAL expert-frame candidates. Never inspect, rank, or request planner actions, candidate trajectories, latents, images, or video. Every selected waypoint is executed as a real dataset frame: exact candidate geometry uses its anchor_row; modified coordinates are re-retrieved to another real frame. Never invent a goal image.

Event policy:
- STALLED: intervene with SUBGOAL and select candidate_id 0, 1, or 2. Candidate 0 is 1/3 progress, 1 is 2/3 progress, and 2 is the nearest real retreat/detour frame to the reverse-5cm intent (at least 4.5 cm actual displacement and farther from the final target). Compare current_to_candidate_distance and prefer progress unless retreat is needed to escape a bad approach.
- DROPPED: intervene with RECOVER and select one recovery candidate plus REGRASP, LIFT_AND_RETRY, or REAPPROACH.
- CONTINUE is exceptional, never free: it requires a specific non-empty reason explaining why every real-frame candidate is unsafe or geometrically unsuitable.

Return exactly one JSON object and no prose or markdown. Exact forms:
{"decision":"SUBGOAL","candidate_id":0,"block_pos":[x,y,z],"yaw":number}
{"decision":"RECOVER","candidate_id":0,"strategy":"REGRASP","ee_pos":[x,y,z]}
{"decision":"CONTINUE","reason":"specific reason"}
Use only candidate_id 0, 1, or 2. A coordinate fine-tune must stay within 0.03 m L2 of the selected candidate; a yaw fine-tune must stay within 15 degrees circular distance. Coordinates must stay inside x=[0.244,0.611], y=[-0.356,0.355], z=[0,0.35]. Output JSON only."""

_FEW_SHOTS_2 = """

Examples:
1) A STALLED block is 0.24 m from target, distance is flat, and progress candidate 1 is much closer to target. Output {"decision":"SUBGOAL","candidate_id":1,"block_pos":[0.49,0.08,0.02],"yaw":0.3}.
2) A DROPPED block is on the table and a nearby contact-qualified recovery frame is candidate 0. Output {"decision":"RECOVER","candidate_id":0,"strategy":"REGRASP","ee_pos":[0.40,-0.05,0.04]}."""

_FEW_SHOT_3 = """
3) A STALLED direct approach is blocked, while candidate 2 is marked retreat_then_advance=true. Output {"decision":"SUBGOAL","candidate_id":2,"block_pos":[0.31,0.12,0.02],"yaw":-0.4}."""

PROMPTS = {
    1: _BASE_POLICY + _FEW_SHOTS_2,
    2: _BASE_POLICY + _FEW_SHOTS_2 + _FEW_SHOT_3 + """

Decision checklist: first identify the event, then compare all three candidate distances and retreat flags, then emit the event-matched intervention. A stalled controller repeating the same state is evidence for changing the real-frame goal, not for CONTINUE.""",
    3: _BASE_POLICY + _FEW_SHOTS_2 + _FEW_SHOT_3 + """

Strong intervention rule: for STALLED, SUBGOAL is mandatory unless all three candidate frames violate a stated physical safety fact in the input. For DROPPED, RECOVER is mandatory. Generic uncertainty, low planner cost, or hope that replanning improves are not valid CONTINUE reasons. Copy the selected real candidate geometry unless a small coordinate adjustment is necessary.""",
}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionValidation:
    decision: dict[str, Any]
    valid: bool
    error: str | None
    event_policy_exception: bool = False


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite numeric")
    return result


def _vec3(value: Any, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ProtocolError(f"{name} must have length 3")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _exact(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ProtocolError(f"{name} exact keys mismatch: expected={sorted(keys)}, actual={actual}")
    return value


def _inside_id(position: Sequence[float]) -> bool:
    return (
        b1.ID_X[0] <= position[0] <= b1.ID_X[1]
        and b1.ID_Y[0] <= position[1] <= b1.ID_Y[1]
        and b1.ID_Z[0] <= position[2] <= b1.ID_Z[1]
    )


def _candidate_schema(event: str) -> set[str]:
    base = {
        "candidate_id", "anchor_row", "source_episode", "block_pos", "yaw",
        "dist_to_target", "current_to_candidate_distance", "retreat_then_advance",
    }
    return base | (
        {"ee_pos"} if event == "DROPPED"
        else {"intent", "strict_reverse", "fallback_reason"}
    )


def validate_candidate_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    root = _exact(value, {"state", "retrieval_candidates"}, "B2 payload")
    state = b1.validate_state_payload(root["state"])
    event = state["event"]
    candidates = root["retrieval_candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ProtocolError("retrieval_candidates must be an exact length-3 JSON array")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        item = _exact(raw, _candidate_schema(event), f"candidate[{index}]")
        candidate_id = item["candidate_id"]
        if isinstance(candidate_id, bool) or candidate_id != index:
            raise ProtocolError("candidate ids must be exact stable order [0,1,2]")
        for name in ("anchor_row", "source_episode"):
            if isinstance(item[name], bool) or not isinstance(item[name], int) or item[name] < 0:
                raise ProtocolError(f"candidate[{index}].{name} must be non-negative int")
        block_pos = _vec3(item["block_pos"], f"candidate[{index}].block_pos")
        if not _inside_id(block_pos):
            raise ProtocolError(f"candidate[{index}].block_pos outside frozen ID box")
        distance = _finite(item["dist_to_target"], f"candidate[{index}].dist_to_target")
        if distance < 0:
            raise ProtocolError("candidate distance cannot be negative")
        actual_distance = math.dist(block_pos, state["target"]["pos"])
        if not math.isclose(distance, actual_distance, rel_tol=0.0, abs_tol=2e-6):
            raise ProtocolError("candidate distance does not match real-frame geometry")
        current_distance = _finite(
            item["current_to_candidate_distance"],
            f"candidate[{index}].current_to_candidate_distance",
        )
        if current_distance < 0 or not math.isclose(
            current_distance, math.dist(block_pos, state["block"]["pos"]),
            rel_tol=0.0, abs_tol=2e-6,
        ):
            raise ProtocolError("candidate current distance does not match real-frame geometry")
        retreat = item["retreat_then_advance"]
        if not isinstance(retreat, bool):
            raise ProtocolError("retreat_then_advance must be bool")
        expected_retreat = distance > float(state["dist_to_target"]) + 1e-6
        if retreat != expected_retreat:
            raise ProtocolError("retreat flag was not recomputed from retrieved geometry")
        result = {
            "candidate_id": index,
            "anchor_row": int(item["anchor_row"]),
            "source_episode": int(item["source_episode"]),
            "block_pos": block_pos,
            "yaw": _finite(item["yaw"], f"candidate[{index}].yaw"),
            "dist_to_target": distance,
            "current_to_candidate_distance": current_distance,
            "retreat_then_advance": retreat,
        }
        if event == "STALLED":
            intent = item["intent"]
            strict_reverse = item["strict_reverse"]
            fallback_reason = item["fallback_reason"]
            if not isinstance(intent, str) or not isinstance(strict_reverse, bool):
                raise ProtocolError("STALLED candidate intent/strict_reverse types invalid")
            if index < 2:
                expected_intent = ("progress_one_third", "progress_two_thirds")[index]
                if intent != expected_intent or strict_reverse or fallback_reason is not None:
                    raise ProtocolError("progress candidate descriptor drift")
                if distance >= float(state["dist_to_target"]):
                    raise ProtocolError("progress candidate does not approach final target")
            elif intent == "retreat":
                if not strict_reverse or fallback_reason is not None:
                    raise ProtocolError("strict retreat descriptor drift")
            elif intent == "detour":
                if strict_reverse or fallback_reason not in {
                    "no_strict_reverse_anchor", "no_qualified_detour_within_8cm"
                }:
                    raise ProtocolError("detour fallback descriptor drift")
            else:
                raise ProtocolError("unknown STALLED candidate intent")
            if index == 2:
                if (
                    current_distance < 0.045 - 2e-6
                    or distance <= float(state["dist_to_target"]) + 1e-6
                ):
                    raise ProtocolError("retreat/detour hard geometry qualification failed")
                current = state["block"]["pos"]
                target = state["target"]["pos"]
                planar = [target[0] - current[0], target[1] - current[1]]
                planar_norm = math.hypot(*planar)
                unit = (
                    [planar[0] / planar_norm, planar[1] / planar_norm]
                    if planar_norm > 1e-12 else [1.0, 0.0]
                )
                signed_projection = (
                    (block_pos[0] - current[0]) * unit[0]
                    + (block_pos[1] - current[1]) * unit[1]
                )
                if intent == "retreat" and (
                    current_distance > 0.08 + 2e-6 or signed_projection > -0.04 + 2e-6
                ):
                    raise ProtocolError("strict retreat preference geometry failed")
                if fallback_reason == "no_strict_reverse_anchor" and current_distance > 0.08 + 2e-6:
                    raise ProtocolError("in-range detour exceeds 8cm")
                if fallback_reason == "no_qualified_detour_within_8cm" and current_distance <= 0.08:
                    raise ProtocolError("sparse detour fallback reason does not match geometry")
            result.update({
                "intent": intent,
                "strict_reverse": strict_reverse,
                "fallback_reason": fallback_reason,
            })
        else:
            ee_pos = _vec3(item["ee_pos"], f"candidate[{index}].ee_pos")
            if not _inside_id(ee_pos):
                raise ProtocolError(f"candidate[{index}].ee_pos outside frozen ID box")
            result["ee_pos"] = ee_pos
        normalized.append(result)
    if len({item["anchor_row"] for item in normalized}) != 3:
        raise ProtocolError("candidate anchor rows must be distinct")
    if len({item["source_episode"] for item in normalized}) != 3:
        raise ProtocolError("candidate source episodes must be distinct")
    encoded = b1.compact_json({"state": state, "retrieval_candidates": normalized})
    if b1.conservative_token_upper_bound(encoded) > 2400:
        raise ProtocolError("B2 input exceeds conservative 2400-token bound")
    return {"state": state, "retrieval_candidates": normalized}


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate response key: {key}")
        result[key] = value
    return result


def validate_response(content: str, event: str, candidates: Sequence[Mapping[str, Any]]) -> DecisionValidation:
    fallback = {"decision": "CONTINUE", "reason": "invalid provider response"}
    try:
        if not isinstance(content, str) or not content.strip():
            raise ProtocolError("empty response")
        value = json.loads(
            content,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ProtocolError(f"nonfinite {token}")),
        )
        if not isinstance(value, Mapping):
            raise ProtocolError("response must be one object")
        decision = value.get("decision")
        if decision == "CONTINUE":
            item = _exact(value, {"decision", "reason"}, "CONTINUE")
            reason = item["reason"]
            if not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_CONTINUE_REASON_CHARS:
                raise ProtocolError("CONTINUE requires a non-empty bounded reason")
            return DecisionValidation(
                {"decision": "CONTINUE", "reason": reason.strip()}, True, None, True
            )
        if decision == "SUBGOAL":
            if event != "STALLED":
                raise ProtocolError("SUBGOAL is only legal for STALLED")
            item = _exact(value, {"decision", "candidate_id", "block_pos", "yaw"}, "SUBGOAL")
            normalized = {
                "decision": "SUBGOAL",
                "candidate_id": item["candidate_id"],
                "block_pos": _vec3(item["block_pos"], "SUBGOAL.block_pos"),
                "yaw": _finite(item["yaw"], "SUBGOAL.yaw"),
            }
        elif decision == "RECOVER":
            if event != "DROPPED":
                raise ProtocolError("RECOVER is only legal for DROPPED")
            item = _exact(value, {"decision", "candidate_id", "strategy", "ee_pos"}, "RECOVER")
            if item["strategy"] not in RECOVER_STRATEGIES:
                raise ProtocolError("unknown recovery strategy")
            normalized = {
                "decision": "RECOVER",
                "candidate_id": item["candidate_id"],
                "strategy": item["strategy"],
                "ee_pos": _vec3(item["ee_pos"], "RECOVER.ee_pos"),
            }
        else:
            raise ProtocolError("unknown decision")
        candidate_id = normalized["candidate_id"]
        if isinstance(candidate_id, bool) or candidate_id not in {0, 1, 2}:
            raise ProtocolError("candidate_id must be 0, 1, or 2")
        if int(candidates[candidate_id]["candidate_id"]) != candidate_id:
            raise ProtocolError("candidate_id does not resolve in frozen candidate list")
        return DecisionValidation(normalized, True, None, False)
    except (json.JSONDecodeError, ProtocolError, TypeError, KeyError, IndexError) as exc:
        return DecisionValidation(fallback, False, str(exc), False)


def _wrap_yaw(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def guard_and_bind(
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = json.loads(b1.compact_json(dict(decision)))
    applied = json.loads(b1.compact_json(dict(decision)))
    audit: dict[str, Any] = {
        "clamp_applied": False,
        "adjustment_clamp_applied": False,
        "yaw_wrap_applied": False,
        "requested_position_delta_l2_m": None,
        "applied_position_delta_l2_m": None,
        "requested_yaw_delta_rad": None,
        "applied_yaw_delta_rad": None,
        "selected_anchor_row": None,
        "selected_source_episode": None,
        "fine_tuned": False,
        "reretrieve_required": False,
    }
    if decision["decision"] == "CONTINUE":
        return applied, audit
    candidate = candidates[int(decision["candidate_id"])]
    audit["selected_anchor_row"] = int(candidate["anchor_row"])
    audit["selected_source_episode"] = int(candidate["source_episode"])
    field = "block_pos" if decision["decision"] == "SUBGOAL" else "ee_pos"
    candidate_position = [float(value) for value in candidate[field]]
    requested_position = [float(value) for value in decision[field]]
    delta = [value - anchor for value, anchor in zip(requested_position, candidate_position)]
    delta_norm = math.sqrt(sum(value * value for value in delta))
    audit["requested_position_delta_l2_m"] = delta_norm
    if delta_norm > MAX_FINE_TUNE_POSITION_M:
        scale = MAX_FINE_TUNE_POSITION_M / delta_norm
        bounded_position = [
            anchor + value * scale for anchor, value in zip(candidate_position, delta)
        ]
        audit["adjustment_clamp_applied"] = True
    else:
        bounded_position = requested_position
    bounds = (b1.ID_X, b1.ID_Y, b1.ID_Z)
    applied[field] = [
        min(axis[1], max(axis[0], float(value)))
        for value, axis in zip(bounded_position, bounds)
    ]
    audit["clamp_applied"] = applied[field] != requested[field]
    audit["applied_position_delta_l2_m"] = math.dist(applied[field], candidate_position)
    if decision["decision"] == "SUBGOAL":
        requested_yaw = float(decision["yaw"])
        candidate_yaw = float(candidate["yaw"])
        yaw_delta = _wrap_yaw(requested_yaw - candidate_yaw)
        bounded_yaw_delta = min(
            MAX_FINE_TUNE_YAW_RAD, max(-MAX_FINE_TUNE_YAW_RAD, yaw_delta)
        )
        applied["yaw"] = _wrap_yaw(candidate_yaw + bounded_yaw_delta)
        audit["requested_yaw_delta_rad"] = yaw_delta
        audit["applied_yaw_delta_rad"] = _wrap_yaw(applied["yaw"] - candidate_yaw)
        yaw_bounded = not math.isclose(
            yaw_delta, bounded_yaw_delta, rel_tol=0.0, abs_tol=1e-12
        )
        audit["adjustment_clamp_applied"] = audit["adjustment_clamp_applied"] or yaw_bounded
        audit["yaw_wrap_applied"] = not math.isclose(
            applied["yaw"], requested_yaw, rel_tol=0.0, abs_tol=1e-12
        )
    candidate_field = candidate[field]
    coordinate_delta = max(abs(a - float(b)) for a, b in zip(applied[field], candidate_field))
    yaw_delta = 0.0
    if decision["decision"] == "SUBGOAL":
        yaw_delta = abs(_wrap_yaw(float(applied["yaw"]) - float(candidate["yaw"])))
    audit["fine_tuned"] = coordinate_delta > FINE_TUNE_TOLERANCE or yaw_delta > FINE_TUNE_TOLERANCE
    audit["reretrieve_required"] = audit["fine_tuned"]
    audit["landing_contract"] = (
        "reretrieve_real_frame" if audit["fine_tuned"] else "use_selected_anchor_row"
    )
    audit["requested_decision"] = requested
    audit["applied_decision"] = applied
    return applied, audit


@dataclass
class ReplayBudget:
    initial_tokens: int = 0
    max_total_tokens: int = MAX_TOTAL_TOKENS
    reported_tokens: int = 0
    unknown_token_upper: int = 0
    logical_by_episode: dict[str, int] = field(default_factory=dict)

    @property
    def accounted_tokens(self) -> int:
        return self.initial_tokens + self.reported_tokens + self.unknown_token_upper

    def begin(self, episode_id: str) -> int:
        count = self.logical_by_episode.get(episode_id, 0)
        if count >= MAX_LOGICAL_CALLS_PER_EPISODE:
            raise b1.BudgetExceeded("B2 per-episode five-call budget exhausted")
        self.logical_by_episode[episode_id] = count + 1
        return count + 1

    def reserve(self, upper: int) -> None:
        if self.accounted_tokens + upper > self.max_total_tokens:
            raise b1.BudgetExceeded("B2 cumulative one-million-token budget would be exceeded")

    def settle(self, upper: int, actual: int | None) -> None:
        if actual is None:
            self.unknown_token_upper += upper
        else:
            if actual < 0 or actual > upper:
                raise RuntimeError("provider usage exceeds conservative reservation")
            self.reported_tokens += actual

    def snapshot(self) -> dict[str, Any]:
        return {
            "initial_tokens": self.initial_tokens,
            "reported_tokens": self.reported_tokens,
            "unknown_token_upper": self.unknown_token_upper,
            "accounted_tokens": self.accounted_tokens,
            "logical_by_episode": dict(sorted(self.logical_by_episode.items())),
            "max_total_tokens": self.max_total_tokens,
        }


@dataclass(frozen=True)
class B2Config:
    prompt_version: int
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    initial_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_version not in PROMPT_VERSIONS:
            raise ValueError("prompt_version must be 1, 2, or 3")
        b1.safe_endpoint(self.base_url)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout must be finite positive")
        if isinstance(self.initial_tokens, bool) or not isinstance(self.initial_tokens, int) or self.initial_tokens < 0:
            raise ValueError("initial_tokens must be non-negative int")


class BrainSupervisorV2:
    def __init__(
        self,
        config: B2Config,
        output_dir: Path,
        *,
        api_key: str | None = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir.expanduser().absolute().resolve()
        data_root = Path("/root/autodl-tmp")
        if self.output_dir != data_root and data_root not in self.output_dir.parents:
            raise ValueError("B2 output must be on data disk")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.output_dir / "llm_calls.json"
        self.manifest_path = self.output_dir / "api_manifest.json"
        self.summary_path = self.output_dir / "summary.json"
        existing = [path for path in (self.calls_path, self.manifest_path, self.summary_path) if path.exists()]
        if existing:
            raise FileExistsError(f"B2 round is non-resumable: {existing}")
        self.api_key = api_key
        self.endpoint = b1.safe_endpoint(config.base_url)
        self.budget = ReplayBudget(initial_tokens=config.initial_tokens)
        self.records: list[dict[str, Any]] = []
        b1.atomic_write_json(self.manifest_path, self.manifest())

    def manifest(self) -> dict[str, Any]:
        prompt = PROMPTS[self.config.prompt_version]
        return {
            "protocol": "cube_brain_b2_prompt_replay_v1",
            "prompt_version": self.config.prompt_version,
            "prompt_sha256": b1.sha256_text(prompt),
            "provider": "deepseek",
            "requested_model": MODEL,
            "endpoint": self.endpoint,
            "thinking": dict(THINKING),
            "reasoning_effort": "omitted",
            "temperature": TEMPERATURE,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS,
            "single_turn_no_history": True,
            "max_logical_calls_per_episode": MAX_LOGICAL_CALLS_PER_EPISODE,
            "max_total_tokens_all_rounds": MAX_TOTAL_TOKENS,
            "initial_tokens_from_prior_rounds": self.config.initial_tokens,
            "id_box": {"x": list(b1.ID_X), "y": list(b1.ID_Y), "z": list(b1.ID_Z)},
            "candidate_ids": [0, 1, 2],
            "real_frame_landing": {
                "exact_candidate": "use selected anchor_row",
                "fine_tuned": "reretrieve a real frame; never synthesize goal pixels",
                "max_position_adjustment_l2_m": MAX_FINE_TUNE_POSITION_M,
                "max_yaw_adjustment_circular_rad": MAX_FINE_TUNE_YAW_RAD,
                "guard_order": "candidate-relative clamp, ID-box clamp, real-frame reretrieval",
            },
            "secrets_persisted": False,
            "reasoning_persisted": False,
        }

    def _key(self) -> str:
        key = self.api_key if self.api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise b1.FatalProviderError("missing DEEPSEEK_API_KEY")
        return key

    def request_body(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_candidate_payload(payload)
        return {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": PROMPTS[self.config.prompt_version]},
                {"role": "user", "content": b1.compact_json(normalized)},
            ],
            "stream": False,
            "thinking": dict(THINKING),
            "temperature": TEMPERATURE,
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS,
        }

    def _post_json(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=b1.compact_json(dict(body)).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds, context=ssl.create_default_context()
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_HTTP or exc.code >= 500:
                raise b1.RetryableTransportError(f"HTTP {exc.code}") from None
            raise b1.FatalProviderError(f"provider rejected request with HTTP {exc.code}") from None
        except (
            urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected,
            http.client.IncompleteRead, ConnectionResetError, BrokenPipeError,
        ) as exc:
            raise b1.RetryableTransportError(type(exc).__name__) from None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            raise ProtocolError("provider envelope was not JSON") from None
        if not isinstance(value, dict):
            raise ProtocolError("provider envelope was not object")
        return value

    @staticmethod
    def _response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        visible, truncated = b1._sanitize_visible_response(content)
        return {
            "actual_model": response.get("model") if isinstance(response.get("model"), str) else "unknown",
            "response_id": response.get("id") if isinstance(response.get("id"), str) else None,
            "system_fingerprint": response.get("system_fingerprint") if isinstance(response.get("system_fingerprint"), str) else None,
            "finish_reason": choice.get("finish_reason"),
            "reasoning_present": bool(message.get("reasoning_content")),
            "sanitized_response_text": visible,
            "response_text_truncated": truncated,
            "response_sha256": b1.sha256_text(content) if isinstance(content, str) else None,
            "usage": b1._usage(response.get("usage")),
        }

    @staticmethod
    def _extract_content(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        if choice.get("finish_reason") != "stop" or not isinstance(content, str):
            raise ProtocolError("provider response requires one string choice with finish_reason=stop")
        return content

    def decide(
        self,
        state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        *,
        replay_id: str,
        episode_id: str,
    ) -> dict[str, Any]:
        payload = validate_candidate_payload({"state": state, "retrieval_candidates": list(candidates)})
        logical_index = self.budget.begin(str(episode_id))
        body = self.request_body(payload)
        upper = b1.conservative_token_upper_bound(b1.compact_json(body)) + MAX_TOKENS
        last_error: str | None = None
        total_start = time.perf_counter()
        for retry in range(MAX_TRANSPORT_RETRIES + 1):
            self.budget.reserve(upper)
            started = time.perf_counter()
            response: dict[str, Any] | None = None
            metadata: dict[str, Any] = {"usage": {}, "actual_model": "unknown"}
            try:
                response = self._post_json(body)
                metadata = self._response_metadata(response)
                if metadata["actual_model"] != MODEL:
                    raise b1.FatalProviderError(
                        f"provider_model_mismatch: requested={MODEL}, actual={metadata['actual_model']}"
                    )
                if metadata["reasoning_present"]:
                    raise b1.FatalProviderError("non-empty reasoning_content with thinking disabled")
                content = self._extract_content(response)
                validation = validate_response(
                    content, payload["state"]["event"], payload["retrieval_candidates"]
                )
                self.budget.settle(upper, metadata["usage"].get("total_tokens"))
                applied, landing = guard_and_bind(validation.decision, payload["retrieval_candidates"])
                status = "ok" if validation.valid else "protocol_failure"
                record = self._record(
                    replay_id, episode_id, logical_index, retry + 1, payload, validation,
                    applied, landing, status, None if validation.valid else validation.error,
                    (time.perf_counter() - started) * 1000.0, metadata,
                )
                self._append(record)
                return {
                    **applied,
                    "landing": landing,
                    "call_record": {
                        "replay_id": replay_id,
                        "status": status,
                        "logical_call_index": logical_index,
                        "attempts": retry + 1,
                        "latency_ms": (time.perf_counter() - total_start) * 1000.0,
                        "usage": metadata["usage"],
                        "actual_model": metadata["actual_model"],
                        "response_sha256": metadata.get("response_sha256"),
                    },
                }
            except b1.RetryableTransportError as exc:
                last_error = str(exc)
                self.budget.settle(upper, None)
                self._append(self._record(
                    replay_id, episode_id, logical_index, retry + 1, payload,
                    DecisionValidation(
                        {"decision": "CONTINUE", "reason": "transport failure"}, False, last_error
                    ),
                    {"decision": "CONTINUE", "reason": "transport failure"}, {},
                    "transport_retry" if retry < MAX_TRANSPORT_RETRIES else "transport_failure",
                    last_error, (time.perf_counter() - started) * 1000.0, metadata,
                ))
                if retry < MAX_TRANSPORT_RETRIES:
                    time.sleep(0.5)
                    continue
                return {
                    "decision": "CONTINUE", "reason": "transport failure",
                    "landing": {},
                    "call_record": {
                        "replay_id": replay_id, "status": "transport_failure",
                        "logical_call_index": logical_index, "attempts": retry + 1,
                        "latency_ms": (time.perf_counter() - total_start) * 1000.0,
                        "usage": {}, "actual_model": "unknown", "response_sha256": None,
                    },
                }
            except ProtocolError as exc:
                usage = b1._usage(response.get("usage") if response else None)
                self.budget.settle(upper, usage.get("total_tokens"))
                last_error = str(exc)
                self._append(self._record(
                    replay_id, episode_id, logical_index, retry + 1, payload,
                    DecisionValidation(
                        {"decision": "CONTINUE", "reason": "protocol failure"}, False, last_error
                    ),
                    {"decision": "CONTINUE", "reason": "protocol failure"}, {},
                    "protocol_failure", last_error,
                    (time.perf_counter() - started) * 1000.0, metadata,
                ))
                return {
                    "decision": "CONTINUE", "reason": "protocol failure", "landing": {},
                    "call_record": {
                        "replay_id": replay_id, "status": "protocol_failure",
                        "logical_call_index": logical_index, "attempts": retry + 1,
                        "latency_ms": (time.perf_counter() - total_start) * 1000.0,
                        "usage": usage, "actual_model": metadata.get("actual_model", "unknown"),
                        "response_sha256": metadata.get("response_sha256"),
                    },
                }
            except b1.FatalProviderError as exc:
                usage = b1._usage(response.get("usage") if response else None)
                self.budget.settle(upper, usage.get("total_tokens"))
                self._append(self._record(
                    replay_id, episode_id, logical_index, retry + 1, payload,
                    DecisionValidation(
                        {"decision": "CONTINUE", "reason": "fatal provider failure"}, False, str(exc)
                    ),
                    {"decision": "CONTINUE", "reason": "fatal provider failure"}, {},
                    "fatal_provider_failure", str(exc),
                    (time.perf_counter() - started) * 1000.0, metadata,
                ))
                raise
        raise RuntimeError(last_error or "unreachable")

    def _record(
        self,
        replay_id: str,
        episode_id: str,
        logical_index: int,
        attempt: int,
        payload: Mapping[str, Any],
        validation: DecisionValidation,
        applied: Mapping[str, Any],
        landing: Mapping[str, Any],
        status: str,
        error: str | None,
        latency_ms: float,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "timestamp_utc": b1.utc_now(),
            "replay_id": replay_id,
            "episode_id": str(episode_id),
            "logical_call_index": logical_index,
            "attempt": attempt,
            "prompt_version": self.config.prompt_version,
            "prompt_sha256": b1.sha256_text(PROMPTS[self.config.prompt_version]),
            "requested_model": MODEL,
            "actual_model": metadata.get("actual_model", "unknown"),
            "endpoint": self.endpoint,
            "thinking": dict(THINKING),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "payload": dict(payload),
            "payload_sha256": b1.sha256_text(b1.compact_json(dict(payload))),
            "validated_response": validation.decision,
            "applied_decision": dict(applied),
            "event_policy_exception": validation.event_policy_exception,
            "landing": dict(landing),
            "status": status,
            "protocol_failure": not validation.valid,
            "error": error,
            "latency_ms": latency_ms,
            "usage": dict(metadata.get("usage", {})),
            "response_id": metadata.get("response_id"),
            "system_fingerprint": metadata.get("system_fingerprint"),
            "finish_reason": metadata.get("finish_reason"),
            "response_sha256": metadata.get("response_sha256"),
            "sanitized_response_text": metadata.get("sanitized_response_text"),
            "response_text_truncated": bool(metadata.get("response_text_truncated", False)),
            "reasoning_present": bool(metadata.get("reasoning_present", False)),
        }

    def _append(self, record: Mapping[str, Any]) -> None:
        b1._reject_sensitive_log_fields(record)
        self.records.append(dict(record))
        b1.atomic_write_json(self.calls_path, self.records)
        b1.atomic_write_json(self.summary_path, self.summary())

    def summary(self) -> dict[str, Any]:
        finals = [record for record in self.records if record["status"] != "transport_retry"]
        decisions = Counter(record["applied_decision"]["decision"] for record in finals)
        latencies = [float(record["latency_ms"]) for record in self.records]
        return {
            "prompt_version": self.config.prompt_version,
            "http_attempts": len(self.records),
            "logical_calls": len(finals),
            "status_counts": dict(Counter(record["status"] for record in self.records)),
            "decision_counts": dict(decisions),
            "non_continue_fraction": (
                (len(finals) - decisions.get("CONTINUE", 0)) / len(finals) if finals else None
            ),
            "mean_attempt_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "budget": self.budget.snapshot(),
        }


def _synthetic_payload(event: str = "STALLED") -> dict[str, Any]:
    state = b1.build_state_payload(
        event=event, step=20, budget=200,
        block_pos=[0.35, 0.0, 0.02], block_yaw=0.0,
        target_pos=[0.55, 0.0, 0.02], target_yaw=0.0,
        ee_pos=[0.35, 0.0, 0.05], gripper_opening=0.8,
        gripper_contact=event == "DROPPED", dist_to_target=0.2,
        dist_trend_5=[0.2] * 5, grasp_state="FREE", phase="APPROACH",
        planner_cost_trend=[1.0] * 5, calls_remaining=5,
    )
    positions = [[0.40, 0.0, 0.02], [0.48, 0.0, 0.02], [0.29, 0.0, 0.02]]
    candidates = []
    for index, position in enumerate(positions):
        item = {
            "candidate_id": index,
            "anchor_row": 1000 + index,
            "source_episode": 100 + index,
            "block_pos": position,
            "yaw": 0.0,
            "dist_to_target": round(math.dist(position, state["target"]["pos"]), 6),
            "current_to_candidate_distance": round(
                math.dist(position, state["block"]["pos"]), 6
            ),
            "retreat_then_advance": math.dist(position, state["target"]["pos"]) > 0.200001,
        }
        if event == "DROPPED":
            item["ee_pos"] = [position[0], position[1], 0.04]
        else:
            item.update({
                "intent": ("progress_one_third", "progress_two_thirds", "retreat")[index],
                "strict_reverse": index == 2,
                "fallback_reason": None,
            })
        candidates.append(item)
    return validate_candidate_payload({"state": state, "retrieval_candidates": candidates})


def self_test() -> None:
    payload = _synthetic_payload()
    body = BrainSupervisorV2.__new__(BrainSupervisorV2)
    body.config = B2Config(1)  # type: ignore[attr-defined]
    body.endpoint = b1.safe_endpoint(DEFAULT_BASE_URL)  # type: ignore[attr-defined]
    request = BrainSupervisorV2.request_body(body, payload)
    assert request["model"] == MODEL and request["thinking"] == {"type": "disabled"}
    assert request["temperature"] == 0.1 and "reasoning_effort" not in request
    assert len(request["messages"]) == 2
    good = validate_response(
        '{"decision":"SUBGOAL","candidate_id":1,"block_pos":[0.48,0,0.02],"yaw":0}',
        "STALLED", payload["retrieval_candidates"],
    )
    assert good.valid
    applied, landing = guard_and_bind(good.decision, payload["retrieval_candidates"])
    assert applied["decision"] == "SUBGOAL" and landing["selected_anchor_row"] == 1001
    assert landing["fine_tuned"] is False and landing["landing_contract"] == "use_selected_anchor_row"
    tuned = validate_response(
        '{"decision":"SUBGOAL","candidate_id":1,"block_pos":[9,0,0.02],"yaw":7}',
        "STALLED", payload["retrieval_candidates"],
    )
    assert tuned.valid
    applied, landing = guard_and_bind(tuned.decision, payload["retrieval_candidates"])
    assert math.isclose(
        math.dist(applied["block_pos"], payload["retrieval_candidates"][1]["block_pos"]),
        MAX_FINE_TUNE_POSITION_M,
    )
    assert abs(landing["applied_yaw_delta_rad"]) <= MAX_FINE_TUNE_YAW_RAD + 1e-12
    assert landing["adjustment_clamp_applied"] and landing["reretrieve_required"]
    continued = validate_response(
        '{"decision":"CONTINUE","reason":"all candidates collide with a known bound"}',
        "STALLED", payload["retrieval_candidates"],
    )
    assert continued.valid and continued.event_policy_exception
    assert not validate_response('{"decision":"CONTINUE"}', "STALLED", payload["retrieval_candidates"]).valid
    assert not validate_response(
        '{"decision":"RECOVER","candidate_id":0,"strategy":"REGRASP","ee_pos":[0.4,0,0.04]}',
        "STALLED", payload["retrieval_candidates"],
    ).valid
    dropped = _synthetic_payload("DROPPED")
    recover = validate_response(
        '{"decision":"RECOVER","candidate_id":0,"strategy":"REGRASP",'
        '"ee_pos":[0.41,0,0.04]}',
        "DROPPED", dropped["retrieval_candidates"],
    )
    assert recover.valid
    recovered, recovery_landing = guard_and_bind(
        recover.decision, dropped["retrieval_candidates"]
    )
    assert recovered["decision"] == "RECOVER"
    assert recovery_landing["selected_anchor_row"] == 1000
    assert recovery_landing["applied_position_delta_l2_m"] <= MAX_FINE_TUNE_POSITION_M + 1e-12
    bad = json.loads(b1.compact_json(payload))
    bad["retrieval_candidates"][0]["candidate_actions"] = []
    try:
        validate_candidate_payload(bad)
    except ProtocolError:
        pass
    else:
        raise AssertionError("candidate action field escaped exact schema")

    scratch = Path("/root/autodl-tmp/tmp")
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="brain-b2-selftest-", dir=scratch) as directory:
        root = Path(directory)
        supervisor = BrainSupervisorV2(B2Config(2), root / "valid", api_key="self-test-only")
        supervisor._post_json = lambda _: {  # type: ignore[method-assign]
            "id": "synthetic", "model": MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": (
                '{"decision":"SUBGOAL","candidate_id":0,'
                '"block_pos":[0.4,0,0.02],"yaw":0}'
            )}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        result = supervisor.decide(
            payload["state"], payload["retrieval_candidates"],
            replay_id="synthetic-0", episode_id="7",
        )
        assert result["decision"] == "SUBGOAL"
        assert supervisor.records[0]["actual_model"] == MODEL
        assert supervisor.records[0]["sanitized_response_text"].startswith('{"decision"')
        assert supervisor.summary()["logical_calls"] == 1
        mismatch = BrainSupervisorV2(B2Config(1), root / "mismatch", api_key="self-test-only")
        mismatch._post_json = lambda _: {  # type: ignore[method-assign]
            "model": "unexpected-model",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"decision":"CONTINUE","reason":"test"}'}}],
            "usage": {"total_tokens": 10},
        }
        try:
            mismatch.decide(
                payload["state"], payload["retrieval_candidates"],
                replay_id="synthetic-mismatch", episode_id="8",
            )
        except b1.FatalProviderError:
            pass
        else:
            raise AssertionError("provider model mismatch did not fail closed")
        assert mismatch.records[0]["status"] == "fatal_provider_failure"
        reasoning = BrainSupervisorV2(B2Config(1), root / "reasoning", api_key="self-test-only")
        reasoning._post_json = lambda _: {  # type: ignore[method-assign]
            "model": MODEL,
            "choices": [{"finish_reason": "stop", "message": {
                "content": '{"decision":"CONTINUE","reason":"test"}',
                "reasoning_content": "must not be persisted",
            }}],
            "usage": {"total_tokens": 10},
        }
        try:
            reasoning.decide(
                payload["state"], payload["retrieval_candidates"],
                replay_id="synthetic-reasoning", episode_id="9",
            )
        except b1.FatalProviderError:
            pass
        else:
            raise AssertionError("nonempty reasoning_content did not fail closed")
        assert reasoning.records[0]["reasoning_present"] is True
        assert "must not be persisted" not in (reasoning.calls_path.read_text())
    print("brain_supervisor_v2 self-test: PASS")


def dry_run(prompt_version: int) -> None:
    payload = _synthetic_payload()
    config = B2Config(prompt_version)
    placeholder = BrainSupervisorV2.__new__(BrainSupervisorV2)
    placeholder.config = config  # type: ignore[attr-defined]
    placeholder.endpoint = b1.safe_endpoint(config.base_url)  # type: ignore[attr-defined]
    body = BrainSupervisorV2.request_body(placeholder, payload)
    print(b1.compact_json({
        "prompt_version": prompt_version,
        "prompt_sha256": b1.sha256_text(PROMPTS[prompt_version]),
        "model": body["model"], "thinking": body["thinking"],
        "temperature": body["temperature"], "max_tokens": body["max_tokens"],
        "endpoint": placeholder.endpoint, "candidate_count": 3,
        "request_token_upper": b1.conservative_token_upper_bound(b1.compact_json(body)) + MAX_TOKENS,
        "external_request_sent": False,
    }))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    value.add_argument("--prompt-version", type=int, choices=PROMPT_VERSIONS, default=1)
    return value


def main(args: argparse.Namespace) -> int:
    if args.self_test:
        self_test()
    else:
        dry_run(args.prompt_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
