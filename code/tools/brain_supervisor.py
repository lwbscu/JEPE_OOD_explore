#!/usr/bin/env python3
"""Pure-text DeepSeek supervisor for the Cube B1 slow loop.

The supervisor sees only a compact symbolic-state JSON object and must never
rank or select planner candidates.  The module has no third-party dependencies
and its ``--self-test`` / ``--dry-run`` modes never read a key or contact the
provider.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
ENDPOINT_SUFFIX = "chat/completions"
THINKING = {"type": "disabled"}
TEMPERATURE = 0.1
MAX_TOKENS = 128
MAX_PAYLOAD_BYTES = 600
MAX_LOGGED_RESPONSE_CHARS = 2048
MAX_LOGICAL_CALLS_PER_EPISODE = 5
MAX_ATTEMPTS_PER_EPISODE = 5
MAX_TOTAL_TOKENS = 1_000_000
MAX_TRANSPORT_RETRIES = 1
DEFAULT_TIMEOUT_SECONDS = 30.0
RETRYABLE_HTTP = frozenset({408, 409, 425, 429})
RECOVER_STRATEGIES = frozenset({"REGRASP", "LIFT_AND_RETRY", "REAPPROACH"})
STATE_KEYS = frozenset({
    "v", "event", "step", "budget", "block", "target", "ee_pos", "gripper",
    "dist_to_target", "dist_trend_5", "grasp_state", "phase",
    "planner_cost_trend", "calls_remaining",
})
FORBIDDEN_STATE_KEY_PARTS = (
    "candidate", "action", "latent", "trajectory", "image", "video",
)
ID_X = (0.244, 0.611)
ID_Y = (-0.356, 0.355)
ID_Z = (0.0, 0.35)

SYSTEM = (
    "You are the slow supervisor for a cube manipulation task. You receive only "
    "symbolic state JSON; positions are metres and yaw is radians. Never rank, "
    "select, inspect, or discuss planner candidates or action trajectories. Return "
    "exactly one JSON object and no prose or markdown. Allowed JSON forms are "
    '{"decision":"CONTINUE"}, '
    '{"decision":"SUBGOAL","block_pos":[x,y,z],"yaw":number}, or '
    '{"decision":"RECOVER","strategy":"REGRASP|LIFT_AND_RETRY|REAPPROACH",'
    '"ee_pos":[x,y,z]}. Use CONTINUE when intervention is not clearly justified. '
    "Do not add keys."
)


class RetryableTransportError(RuntimeError):
    """A transport/provider condition for which one retry is allowed."""


class FatalProviderError(RuntimeError):
    """A non-retryable provider/configuration error."""


class BudgetExceeded(RuntimeError):
    """The hard per-episode or global call/token budget would be exceeded."""


class ProtocolValidationError(ValueError):
    """The response is JSON but does not match one exact decision schema."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def conservative_token_upper_bound(text: str) -> int:
    """Bound tokens by UTF-8 bytes; a token consumes at least one input byte."""

    return len(text.encode("utf-8"))


def safe_endpoint(base_url: str, suffix: str = ENDPOINT_SUFFIX) -> str:
    base = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("DeepSeek base URL must be credential-free HTTPS")
    if parsed.query or parsed.fragment:
        raise ValueError("DeepSeek base URL must not contain query or fragment")
    if base.endswith("/" + suffix):
        return base
    return base + "/" + suffix


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vec3(value: Sequence[Any], name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return [_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _finite_series(value: Sequence[Any], name: str, expected: int | None = None) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a finite-number sequence")
    if expected is not None and len(value) != expected:
        raise ValueError(f"{name} must contain exactly {expected} values")
    return [_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _rounded(value: float) -> float:
    return round(value, 6)


def _exact_keys(value: Any, expected: set[str] | frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected), key=str)
        raise ValueError(f"{name} schema mismatch: missing={missing}, extra={extra}")
    return value


def _reject_forbidden_state_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"state key at {'.'.join(path) or '<root>'} must be a string")
            lowered = raw_key.lower()
            if any(part in lowered for part in FORBIDDEN_STATE_KEY_PARTS):
                location = ".".join((*path, raw_key))
                raise ValueError(f"forbidden non-symbolic state key: {location}")
            _reject_forbidden_state_keys(child, (*path, raw_key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_state_keys(child, (*path, str(index)))


def validate_state_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact recursive B1 symbolic-state schema."""

    _reject_forbidden_state_keys(payload)
    root = _exact_keys(payload, STATE_KEYS, "payload")
    if isinstance(root["v"], bool) or root["v"] != 1:
        raise ValueError("payload.v must be integer 1")
    if root["event"] not in {"DROPPED", "STALLED"}:
        raise ValueError("payload.event must be DROPPED or STALLED")
    integers: dict[str, int] = {}
    for name in ("step", "budget", "calls_remaining"):
        value = root[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"payload.{name} must be a non-negative integer")
        integers[name] = value
    if integers["step"] > integers["budget"]:
        raise ValueError("payload.step cannot exceed payload.budget")
    if integers["calls_remaining"] > MAX_LOGICAL_CALLS_PER_EPISODE:
        raise ValueError("payload.calls_remaining exceeds the per-episode budget")

    block = _exact_keys(root["block"], {"pos", "yaw"}, "payload.block")
    target = _exact_keys(root["target"], {"pos", "yaw"}, "payload.target")
    gripper = _exact_keys(root["gripper"], {"opening", "contact"}, "payload.gripper")
    if not isinstance(gripper["contact"], bool):
        raise ValueError("payload.gripper.contact must be bool")
    for name in ("grasp_state", "phase"):
        value = root[name]
        if not isinstance(value, str) or not value or len(value) > 32:
            raise ValueError(f"payload.{name} must be a short non-empty string")

    normalized = {
        "v": 1,
        "event": root["event"],
        "step": integers["step"],
        "budget": integers["budget"],
        "block": {
            "pos": _vec3(block["pos"], "payload.block.pos"),
            "yaw": _finite_float(block["yaw"], "payload.block.yaw"),
        },
        "target": {
            "pos": _vec3(target["pos"], "payload.target.pos"),
            "yaw": _finite_float(target["yaw"], "payload.target.yaw"),
        },
        "ee_pos": _vec3(root["ee_pos"], "payload.ee_pos"),
        "gripper": {
            "opening": _finite_float(gripper["opening"], "payload.gripper.opening"),
            "contact": gripper["contact"],
        },
        "dist_to_target": _finite_float(root["dist_to_target"], "payload.dist_to_target"),
        "dist_trend_5": _finite_series(root["dist_trend_5"], "payload.dist_trend_5", 5),
        "grasp_state": root["grasp_state"],
        "phase": root["phase"],
        "planner_cost_trend": _finite_series(
            root["planner_cost_trend"], "payload.planner_cost_trend", 5
        ),
        "calls_remaining": integers["calls_remaining"],
    }
    serialized = compact_json(normalized)
    upper = conservative_token_upper_bound(serialized)
    if upper > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"symbolic payload exceeds conservative 600-token bound: bytes={upper}, limit={MAX_PAYLOAD_BYTES}"
        )
    return normalized


def build_state_payload(
    *,
    event: str,
    step: int,
    budget: int,
    block_pos: Sequence[Any],
    block_yaw: Any,
    target_pos: Sequence[Any],
    target_yaw: Any,
    ee_pos: Sequence[Any],
    gripper_opening: Any,
    gripper_contact: bool,
    dist_to_target: Any,
    dist_trend_5: Sequence[Any],
    grasp_state: str,
    phase: str,
    planner_cost_trend: Sequence[Any],
    calls_remaining: int,
) -> dict[str, Any]:
    """Build and size-check the complete symbolic-state payload."""

    if event not in {"DROPPED", "STALLED"}:
        raise ValueError("event must be DROPPED or STALLED")
    for name, value in (("step", step), ("budget", budget), ("calls_remaining", calls_remaining)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if step > budget:
        raise ValueError("step cannot exceed budget")
    if calls_remaining > MAX_LOGICAL_CALLS_PER_EPISODE:
        raise ValueError("calls_remaining exceeds the per-episode budget")
    if not isinstance(gripper_contact, bool):
        raise ValueError("gripper_contact must be bool")
    if not isinstance(grasp_state, str) or not grasp_state or len(grasp_state) > 32:
        raise ValueError("grasp_state must be a short non-empty string")
    if not isinstance(phase, str) or not phase or len(phase) > 32:
        raise ValueError("phase must be a short non-empty string")

    def rv(values: Sequence[Any], name: str) -> list[float]:
        return [_rounded(item) for item in _vec3(values, name)]

    payload = {
        "v": 1,
        "event": event,
        "step": step,
        "budget": budget,
        "block": {"pos": rv(block_pos, "block_pos"), "yaw": _rounded(_finite_float(block_yaw, "block_yaw"))},
        "target": {"pos": rv(target_pos, "target_pos"), "yaw": _rounded(_finite_float(target_yaw, "target_yaw"))},
        "ee_pos": rv(ee_pos, "ee_pos"),
        "gripper": {
            "opening": _rounded(_finite_float(gripper_opening, "gripper_opening")),
            "contact": gripper_contact,
        },
        "dist_to_target": _rounded(_finite_float(dist_to_target, "dist_to_target")),
        "dist_trend_5": [_rounded(item) for item in _finite_series(dist_trend_5, "dist_trend_5", 5)],
        "grasp_state": grasp_state,
        "phase": phase,
        "planner_cost_trend": [
            _rounded(item) for item in _finite_series(planner_cost_trend, "planner_cost_trend", 5)
        ],
        "calls_remaining": calls_remaining,
    }
    serialized = compact_json(payload)
    upper = conservative_token_upper_bound(serialized)
    if upper > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"symbolic payload exceeds conservative 600-token bound: bytes={upper}, limit={MAX_PAYLOAD_BYTES}"
        )
    return validate_state_payload(payload)


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(f"duplicate response key: {key}")
        result[key] = value
    return result


def _response_vec3(value: Any, name: str) -> list[float]:
    try:
        result = _vec3(value, name)
    except ValueError as exc:
        raise ProtocolValidationError(str(exc)) from None
    return result


@dataclass(frozen=True)
class ValidationResult:
    decision: dict[str, Any]
    valid: bool
    error: str | None = None

    @property
    def protocol_failure(self) -> bool:
        return not self.valid


def validate_response(content: str) -> ValidationResult:
    """Strictly validate one of the three schemas, otherwise fall back safely."""

    fallback = {"decision": "CONTINUE"}
    if not isinstance(content, str) or not content.strip():
        return ValidationResult(fallback, False, "empty or non-string response content")
    try:
        value = json.loads(
            content,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProtocolValidationError(f"non-finite JSON constant: {token}")
            ),
        )
        if not isinstance(value, dict):
            raise ProtocolValidationError("response must be one JSON object")
        decision = value.get("decision")
        if decision == "CONTINUE":
            if set(value) != {"decision"}:
                raise ProtocolValidationError("CONTINUE has unexpected or missing keys")
            normalized = fallback
        elif decision == "SUBGOAL":
            if set(value) != {"decision", "block_pos", "yaw"}:
                raise ProtocolValidationError("SUBGOAL has unexpected or missing keys")
            try:
                yaw = _finite_float(value["yaw"], "yaw")
            except ValueError as exc:
                raise ProtocolValidationError(str(exc)) from None
            normalized = {
                "decision": "SUBGOAL",
                "block_pos": _response_vec3(value["block_pos"], "block_pos"),
                "yaw": yaw,
            }
        elif decision == "RECOVER":
            if set(value) != {"decision", "strategy", "ee_pos"}:
                raise ProtocolValidationError("RECOVER has unexpected or missing keys")
            strategy = value["strategy"]
            if strategy not in RECOVER_STRATEGIES:
                raise ProtocolValidationError("RECOVER strategy is not allowed")
            normalized = {
                "decision": "RECOVER",
                "strategy": strategy,
                "ee_pos": _response_vec3(value["ee_pos"], "ee_pos"),
            }
        else:
            raise ProtocolValidationError("decision is not one of the three allowed values")
    except (json.JSONDecodeError, ProtocolValidationError, TypeError, KeyError) as exc:
        return ValidationResult(fallback, False, str(exc))
    return ValidationResult(normalized, True, None)


@dataclass
class AttemptReservation:
    episode_id: str
    attempt_index: int
    upper_tokens: int
    settled: bool = False


@dataclass
class EpisodeBudget:
    logical_calls: int = 0
    attempts: int = 0


@dataclass
class BudgetManager:
    max_total_tokens: int = MAX_TOTAL_TOKENS
    max_logical_calls_per_episode: int = MAX_LOGICAL_CALLS_PER_EPISODE
    max_attempts_per_episode: int = MAX_ATTEMPTS_PER_EPISODE
    reported_tokens: int = 0
    unknown_attempt_token_upper: int = 0
    _episodes: dict[str, EpisodeBudget] = field(default_factory=dict, repr=False)
    _reservations: dict[tuple[str, int], AttemptReservation] = field(default_factory=dict, repr=False)

    def _episode(self, episode_id: str) -> EpisodeBudget:
        return self._episodes.setdefault(str(episode_id), EpisodeBudget())

    @property
    def accounted_tokens(self) -> int:
        unsettled = sum(item.upper_tokens for item in self._reservations.values() if not item.settled)
        return self.reported_tokens + self.unknown_attempt_token_upper + unsettled

    def begin_logical_call(self, episode_id: str) -> int:
        episode = self._episode(episode_id)
        if episode.logical_calls >= self.max_logical_calls_per_episode:
            raise BudgetExceeded("per-episode logical-call budget exhausted")
        episode.logical_calls += 1
        return episode.logical_calls

    def reserve_attempt(self, episode_id: str, upper_tokens: int) -> AttemptReservation:
        if isinstance(upper_tokens, bool) or not isinstance(upper_tokens, int) or upper_tokens <= 0:
            raise ValueError("upper_tokens must be a positive integer")
        episode = self._episode(episode_id)
        if episode.attempts >= self.max_attempts_per_episode:
            raise BudgetExceeded("per-episode HTTP-attempt budget exhausted")
        if self.accounted_tokens + upper_tokens > self.max_total_tokens:
            raise BudgetExceeded("global one-million-token budget would be exceeded")
        episode.attempts += 1
        reservation = AttemptReservation(str(episode_id), episode.attempts, upper_tokens)
        self._reservations[(str(episode_id), episode.attempts)] = reservation
        return reservation

    def settle_attempt(self, reservation: AttemptReservation, reported_total_tokens: int | None) -> None:
        key = (reservation.episode_id, reservation.attempt_index)
        current = self._reservations.get(key)
        if current is None or current is not reservation or current.settled:
            raise ValueError("attempt reservation is missing or already settled")
        if reported_total_tokens is None:
            self.unknown_attempt_token_upper += reservation.upper_tokens
        else:
            if (
                isinstance(reported_total_tokens, bool)
                or not isinstance(reported_total_tokens, int)
                or reported_total_tokens < 0
            ):
                raise ValueError("reported_total_tokens must be a non-negative integer or None")
            if reported_total_tokens > reservation.upper_tokens:
                raise RuntimeError("provider usage exceeded the conservative request reservation")
            self.reported_tokens += reported_total_tokens
        current.settled = True

    def calls_remaining(self, episode_id: str) -> int:
        episode = self._episode(episode_id)
        return max(0, self.max_logical_calls_per_episode - episode.logical_calls)

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_total_tokens": self.max_total_tokens,
            "reported_tokens": self.reported_tokens,
            "unknown_attempt_token_upper": self.unknown_attempt_token_upper,
            "accounted_tokens": self.accounted_tokens,
            "episodes": {name: asdict(value) for name, value in sorted(self._episodes.items())},
        }


def atomic_write_json(path: Path, value: Any) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _sanitize_visible_response(content: Any) -> tuple[str | None, bool]:
    if not isinstance(content, str):
        return None, False
    truncated = len(content) > MAX_LOGGED_RESPONSE_CHARS
    value = content[:MAX_LOGGED_RESPONSE_CHARS]
    value = re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [REDACTED]", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED_API_KEY]", value)
    value = re.sub(
        r'(?i)("(?:api[_-]?key|authorization|auth[_-]?token|access[_-]?token|secret)"\s*:\s*)"[^"]*"',
        r'\1"[REDACTED]"',
        value,
    )
    return value, truncated


def _reject_sensitive_log_fields(value: Any, path: tuple[str, ...] = ()) -> None:
    usage_top = {
        "prompt_tokens", "completion_tokens", "total_tokens",
        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
    }
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError("log object keys must be strings")
            lowered = raw_key.lower()
            location = (*path, raw_key)
            if lowered in {"api_key", "apikey", "authorization", "auth_token", "access_token", "secret"}:
                raise ValueError(f"refusing sensitive log field: {'.'.join(location)}")
            if "token" in lowered:
                allowed = (
                    (not path and lowered == "max_tokens")
                    or (path == ("usage",) and lowered in usage_top)
                    or (path == ("usage",) and lowered == "completion_tokens_details")
                    or (
                        path == ("usage", "completion_tokens_details")
                        and lowered == "reasoning_tokens"
                    )
                )
                if not allowed:
                    raise ValueError(f"refusing non-whitelisted token log field: {'.'.join(location)}")
            _reject_sensitive_log_fields(child, location)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive_log_fields(child, (*path, str(index)))


@dataclass
class AtomicCallLogger:
    path: Path
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = self.path.expanduser().absolute()
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise ValueError("existing supervisor log must be a JSON list")
            self.records = value

    def append(self, record: Mapping[str, Any]) -> None:
        sanitized = dict(record)
        _reject_sensitive_log_fields(sanitized)
        self.records.append(sanitized)
        atomic_write_json(self.path, self.records)


def _usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    details = value.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool) and reasoning_tokens >= 0:
            result["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return result


def _fallback_result() -> dict[str, str]:
    return {"decision": "CONTINUE"}


@dataclass(frozen=True)
class BrainCallResult:
    decision: dict[str, Any]
    status: str
    protocol_failure: bool
    logical_call_index: int
    attempts: int
    total_latency_ms: float


class DeepSeekSupervisorClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        logger: AtomicCallLogger | None = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        self.endpoint = safe_endpoint(self.base_url)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.timeout_seconds = float(timeout_seconds)
        self.logger = logger

    def _key(self) -> str:
        key = self._api_key if self._api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise FatalProviderError("missing DEEPSEEK_API_KEY")
        return key

    def request_body(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_state_payload(payload)
        symbolic = compact_json(normalized)
        return {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": symbolic},
            ],
            "stream": False,
            "thinking": dict(THINKING),
            "temperature": TEMPERATURE,
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS,
        }

    def _post_json(self, body: Mapping[str, Any]) -> dict[str, Any]:
        encoded = compact_json(dict(body)).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=ssl.create_default_context(),
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_HTTP or exc.code >= 500:
                raise RetryableTransportError(f"HTTP {exc.code}") from None
            raise FatalProviderError(f"provider rejected request with HTTP {exc.code}") from None
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
            ConnectionResetError,
            BrokenPipeError,
        ) as exc:
            raise RetryableTransportError(type(exc).__name__) from None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            raise ProtocolValidationError("HTTP 200 provider envelope was not JSON") from None
        if not isinstance(value, dict):
            raise ProtocolValidationError("HTTP 200 provider envelope was not an object")
        return value

    @staticmethod
    def _response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        sanitized_response_text, response_text_truncated = _sanitize_visible_response(content)
        return {
            "actual_model": response.get("model") if isinstance(response.get("model"), str) else "unknown",
            "response_id": response.get("id") if isinstance(response.get("id"), str) else None,
            "system_fingerprint": (
                response.get("system_fingerprint")
                if isinstance(response.get("system_fingerprint"), str)
                else None
            ),
            "finish_reason": choice.get("finish_reason"),
            "reasoning_present": bool(message.get("reasoning_content")),
            "response_sha256": sha256_text(content) if isinstance(content, str) else None,
            "sanitized_response_text": sanitized_response_text,
            "response_text_truncated": response_text_truncated,
        }

    @staticmethod
    def _extract(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProtocolValidationError("provider response must contain exactly one choice")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason != "stop":
            raise ProtocolValidationError(f"finish_reason is not stop: {finish_reason!r}")
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProtocolValidationError("provider response has no string message.content")
        return message["content"]

    def call(
        self,
        payload: Mapping[str, Any],
        *,
        episode_id: str,
        budget: BudgetManager,
    ) -> BrainCallResult:
        normalized_payload = validate_state_payload(payload)
        body = self.request_body(normalized_payload)
        logical_call_index = budget.begin_logical_call(episode_id)
        # JSON request bytes upper-bound input tokens. MAX_TOKENS bounds output.
        request_upper_tokens = conservative_token_upper_bound(compact_json(body)) + MAX_TOKENS
        total_start = time.perf_counter()
        attempts_used = 0
        for retry_index in range(MAX_TRANSPORT_RETRIES + 1):
            reservation = budget.reserve_attempt(episode_id, request_upper_tokens)
            attempts_used += 1
            attempt_start = time.perf_counter()
            response: dict[str, Any] | None = None
            usage: dict[str, Any] = {}
            metadata: dict[str, Any] = {
                "actual_model": "unknown",
                "response_id": None,
                "system_fingerprint": None,
                "finish_reason": None,
                "reasoning_present": False,
                "response_sha256": None,
                "sanitized_response_text": None,
                "response_text_truncated": False,
            }
            try:
                response = self._post_json(body)
                usage = _usage(response.get("usage"))
                metadata = self._response_metadata(response)
                if metadata["actual_model"] != MODEL:
                    raise FatalProviderError(
                        f"provider_model_mismatch: requested={MODEL}, actual={metadata['actual_model']}"
                    )
                if metadata["reasoning_present"]:
                    raise FatalProviderError("provider returned non-empty reasoning_content with thinking disabled")
                content = self._extract(response)
                validation = validate_response(content)
                budget.settle_attempt(reservation, usage.get("total_tokens"))
                latency_ms = (time.perf_counter() - attempt_start) * 1000.0
                status = "ok" if validation.valid else "protocol_failure"
                self._log_attempt(
                    episode_id=episode_id,
                    logical_call_index=logical_call_index,
                    reservation=reservation,
                    payload=normalized_payload,
                    validation=validation,
                    status=status,
                    error=validation.error,
                    latency_ms=latency_ms,
                    usage=usage,
                    metadata=metadata,
                )
                return BrainCallResult(
                    validation.decision,
                    status,
                    validation.protocol_failure,
                    logical_call_index,
                    attempts_used,
                    (time.perf_counter() - total_start) * 1000.0,
                )
            except ProtocolValidationError as exc:
                budget.settle_attempt(reservation, _usage(response.get("usage") if response else None).get("total_tokens"))
                latency_ms = (time.perf_counter() - attempt_start) * 1000.0
                validation = ValidationResult(_fallback_result(), False, str(exc))
                self._log_attempt(
                    episode_id, logical_call_index, reservation, normalized_payload, validation,
                    "protocol_failure", str(exc), latency_ms, usage, metadata,
                )
                return BrainCallResult(
                    _fallback_result(), "protocol_failure", True, logical_call_index,
                    attempts_used, (time.perf_counter() - total_start) * 1000.0,
                )
            except RetryableTransportError as exc:
                budget.settle_attempt(reservation, None)
                latency_ms = (time.perf_counter() - attempt_start) * 1000.0
                is_final = retry_index >= MAX_TRANSPORT_RETRIES
                validation = ValidationResult(_fallback_result(), False, str(exc))
                self._log_attempt(
                    episode_id, logical_call_index, reservation, normalized_payload, validation,
                    "transport_failure" if is_final else "transport_retry", str(exc),
                    latency_ms, usage, metadata,
                )
                if is_final:
                    return BrainCallResult(
                        _fallback_result(), "transport_failure", False, logical_call_index,
                        attempts_used, (time.perf_counter() - total_start) * 1000.0,
                    )
                time.sleep(0.5)
            except FatalProviderError as exc:
                budget.settle_attempt(reservation, usage.get("total_tokens"))
                latency_ms = (time.perf_counter() - attempt_start) * 1000.0
                validation = ValidationResult(_fallback_result(), False, str(exc))
                self._log_attempt(
                    episode_id, logical_call_index, reservation, normalized_payload, validation,
                    "fatal_provider_failure", str(exc), latency_ms, usage, metadata,
                )
                raise
        raise AssertionError("unreachable retry loop")

    def _log_attempt(
        self,
        episode_id: str,
        logical_call_index: int,
        reservation: AttemptReservation,
        payload: Mapping[str, Any],
        validation: ValidationResult,
        status: str,
        error: str | None,
        latency_ms: float,
        usage: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        if self.logger is None:
            return
        self.logger.append({
            "timestamp_utc": utc_now(),
            "episode_id": str(episode_id),
            "logical_call_index": logical_call_index,
            "attempt_index": reservation.attempt_index,
            "provider": "deepseek",
            "requested_model": MODEL,
            "endpoint": self.endpoint,
            "thinking": dict(THINKING),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "payload": dict(payload),
            "payload_sha256": sha256_text(compact_json(dict(payload))),
            "validated_response": validation.decision,
            "protocol_failure": validation.protocol_failure,
            "status": status,
            "error": error,
            "latency_ms": latency_ms,
            "usage": dict(usage),
            "actual_model": metadata.get("actual_model", "unknown"),
            "response_id": metadata.get("response_id"),
            "system_fingerprint": metadata.get("system_fingerprint"),
            "finish_reason": metadata.get("finish_reason"),
            "response_sha256": metadata.get("response_sha256"),
            "sanitized_response_text": metadata.get("sanitized_response_text"),
            "response_text_truncated": bool(metadata.get("response_text_truncated", False)),
            # Never persist reasoning_content; only audit an unexpected presence.
            "reasoning_present": bool(metadata.get("reasoning_present", False)),
        })


@dataclass(frozen=True)
class SupervisorConfig:
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_total_tokens: int = MAX_TOTAL_TOKENS
    max_logical_calls_per_episode: int = MAX_LOGICAL_CALLS_PER_EPISODE
    max_attempts_per_episode: int = MAX_ATTEMPTS_PER_EPISODE

    def __post_init__(self) -> None:
        safe_endpoint(self.base_url)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        frozen = {
            "max_total_tokens": (self.max_total_tokens, MAX_TOTAL_TOKENS),
            "max_logical_calls_per_episode": (
                self.max_logical_calls_per_episode,
                MAX_LOGICAL_CALLS_PER_EPISODE,
            ),
            "max_attempts_per_episode": (
                self.max_attempts_per_episode,
                MAX_ATTEMPTS_PER_EPISODE,
            ),
        }
        changed = [name for name, (actual, expected) in frozen.items() if actual != expected]
        if changed:
            raise ValueError(f"B1 hard budgets are frozen and cannot be overridden: {changed}")

    @property
    def id_x(self) -> tuple[float, float]:
        return ID_X

    @property
    def id_y(self) -> tuple[float, float]:
        return ID_Y

    @property
    def id_z(self) -> tuple[float, float]:
        return ID_Z

    @classmethod
    def from_value(cls, value: "SupervisorConfig | Mapping[str, Any] | None") -> "SupervisorConfig":
        if value is None:
            return cls(base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("config must be SupervisorConfig, a mapping, or None")
        allowed = {
            "base_url", "timeout_seconds", "max_total_tokens",
            "max_logical_calls_per_episode", "max_attempts_per_episode",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown or forbidden supervisor config keys: {sorted(unknown)}")
        return cls(**dict(value))


def _clamp_scalar(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def _wrap_yaw(value: float) -> float:
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    return 0.0 if wrapped == -0.0 else wrapped


def clamp_decision(
    decision: Mapping[str, Any],
    config: SupervisorConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Clamp a validated decision and return requested, applied, and audit info."""

    requested = json.loads(compact_json(dict(decision)))
    applied = json.loads(compact_json(dict(decision)))
    position_changed = False
    yaw_changed = False
    position_field: str | None = None
    if decision.get("decision") == "SUBGOAL":
        position_field = "block_pos"
        bounds = (config.id_x, config.id_y, config.id_z)
        applied[position_field] = [
            _clamp_scalar(float(value), axis_bounds)
            for value, axis_bounds in zip(decision[position_field], bounds)
        ]
        position_changed = applied[position_field] != requested[position_field]
        applied["yaw"] = _wrap_yaw(float(decision["yaw"]))
        yaw_changed = not math.isclose(
            applied["yaw"], float(requested["yaw"]), rel_tol=0.0, abs_tol=1e-12
        )
    elif decision.get("decision") == "RECOVER":
        position_field = "ee_pos"
        bounds = (config.id_x, config.id_y, config.id_z)
        applied[position_field] = [
            _clamp_scalar(float(value), axis_bounds)
            for value, axis_bounds in zip(decision[position_field], bounds)
        ]
        position_changed = applied[position_field] != requested[position_field]
    audit = {
        "clamp_applied": position_changed or yaw_changed,
        "position_field": position_field,
        "position_clamp_applied": position_changed,
        "yaw_wrap_applied": yaw_changed,
    }
    return requested, applied, audit


class BrainSupervisor:
    """Evaluator-facing facade with frozen config, budgets, and atomic logs.

    ``decide`` is single-turn: every call sends exactly SYSTEM plus the current
    symbolic payload. No provider message history is retained or replayed.
    """

    def __init__(
        self,
        config: SupervisorConfig | Mapping[str, Any] | None,
        output_dir: Path,
        *,
        api_key: str | None = None,
    ) -> None:
        self.config = SupervisorConfig.from_value(config)
        self.output_dir = output_dir.expanduser().absolute().resolve()
        data_root = Path("/root/autodl-tmp")
        if self.output_dir != data_root and data_root not in self.output_dir.parents:
            raise ValueError("B1 supervisor output must be on /root/autodl-tmp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "llm_calls.json"
        manifest_path = self.output_dir / "brain_api_manifest.json"
        summary_path = self.output_dir / "brain_api_summary.json"
        existing = [path for path in (log_path, manifest_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError(f"B1 supervisor is non-resumable; artifacts already exist: {existing}")
        self.logger = AtomicCallLogger(log_path)
        self.budget = BudgetManager(
            max_total_tokens=self.config.max_total_tokens,
            max_logical_calls_per_episode=self.config.max_logical_calls_per_episode,
            max_attempts_per_episode=self.config.max_attempts_per_episode,
        )
        self.client = DeepSeekSupervisorClient(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout_seconds=self.config.timeout_seconds,
            logger=self.logger,
        )
        frozen = self.manifest()
        atomic_write_json(manifest_path, frozen)

    def manifest(self) -> dict[str, Any]:
        return {
            "protocol": "cube_brain_b1_v1",
            "provider": "deepseek",
            "requested_model": MODEL,
            "endpoint": self.client.endpoint,
            "single_turn_no_history": True,
            "thinking": dict(THINKING),
            "reasoning_effort": "omitted",
            "temperature": TEMPERATURE,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS,
            "system_sha256": sha256_text(SYSTEM),
            "max_payload_conservative_tokens": MAX_PAYLOAD_BYTES,
            "response_schemas": {
                "CONTINUE": ["decision"],
                "SUBGOAL": ["decision", "block_pos", "yaw"],
                "RECOVER": ["decision", "strategy", "ee_pos"],
                "recover_strategies": sorted(RECOVER_STRATEGIES),
            },
            "id_box": {
                "x": list(self.config.id_x),
                "y": list(self.config.id_y),
                "z": list(self.config.id_z),
            },
            "retryable_http": sorted(RETRYABLE_HTTP),
            "retryable_http_5xx": True,
            "max_transport_retries": MAX_TRANSPORT_RETRIES,
            "every_http_attempt_counts_against_budget": True,
            "max_logical_calls_per_episode": self.config.max_logical_calls_per_episode,
            "max_attempts_per_episode": self.config.max_attempts_per_episode,
            "max_total_tokens": self.config.max_total_tokens,
            "timeout_seconds": self.config.timeout_seconds,
            "secrets_persisted": False,
            "reasoning_persisted": False,
        }

    def decide(
        self,
        payload: Mapping[str, Any],
        event: str,
        env_idx: int | str,
        step: int,
    ) -> dict[str, Any]:
        """Return a normalized decision plus one compact logical call record."""

        normalized_payload = validate_state_payload(payload)
        if normalized_payload["event"] != event:
            raise ValueError("event argument does not match symbolic payload")
        if normalized_payload["step"] != step:
            raise ValueError("step argument does not match symbolic payload")
        first_record = len(self.logger.records)
        try:
            result = self.client.call(normalized_payload, episode_id=str(env_idx), budget=self.budget)
        except BaseException:
            atomic_write_json(self.output_dir / "brain_api_summary.json", self.summary())
            raise
        attempt_records = self.logger.records[first_record:]
        final_record = attempt_records[-1] if attempt_records else {}
        requested_decision, clamped_decision, clamp_audit = clamp_decision(result.decision, self.config)
        if final_record:
            final_record["requested_decision"] = requested_decision
            final_record["clamped_decision"] = clamped_decision
            final_record["validated_response"] = clamped_decision
            final_record.update(clamp_audit)
            _reject_sensitive_log_fields(final_record)
            atomic_write_json(self.logger.path, self.logger.records)
        call_record = {
            "event": event,
            "env_idx": env_idx,
            "step": step,
            "status": result.status,
            "protocol_failure": result.protocol_failure,
            "logical_call_index": result.logical_call_index,
            "attempts": result.attempts,
            "total_latency_ms": result.total_latency_ms,
            "payload_sha256": final_record.get("payload_sha256"),
            "response_sha256": final_record.get("response_sha256"),
            "usage": final_record.get("usage", {}),
            "actual_model": final_record.get("actual_model", "unknown"),
            "attempt_statuses": [record.get("status") for record in attempt_records],
            "requested_decision": requested_decision,
            "clamped_decision": clamped_decision,
            **clamp_audit,
        }
        value = dict(clamped_decision)
        value["call_record"] = call_record
        atomic_write_json(self.output_dir / "brain_api_summary.json", self.summary())
        return value

    def summary(self) -> dict[str, Any]:
        records = self.logger.records
        final_records = [record for record in records if record.get("status") != "transport_retry"]
        status_counts: dict[str, int] = {}
        decision_counts: dict[str, int] = {}
        for record in records:
            status = str(record.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        for record in final_records:
            response = record.get("validated_response")
            decision = response.get("decision") if isinstance(response, dict) else "unknown"
            decision_counts[str(decision)] = decision_counts.get(str(decision), 0) + 1
        latencies = [
            float(record["latency_ms"])
            for record in records
            if isinstance(record.get("latency_ms"), (int, float))
            and not isinstance(record.get("latency_ms"), bool)
        ]
        return {
            "logical_calls": sum(item.logical_calls for item in self.budget._episodes.values()),
            "http_attempts": len(records),
            "status_counts": status_counts,
            "decision_counts": decision_counts,
            "protocol_failures": status_counts.get("protocol_failure", 0),
            "provider_model_mismatches": sum(
                1 for record in records if "provider_model_mismatch" in str(record.get("error", ""))
            ),
            "average_attempt_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "budget": self.budget.snapshot(),
        }


def sample_payload() -> dict[str, Any]:
    return build_state_payload(
        event="STALLED",
        step=64,
        budget=150,
        block_pos=[0.401, -0.081, 0.026],
        block_yaw=0.12,
        target_pos=[0.553, 0.101, 0.026],
        target_yaw=-0.31,
        ee_pos=[0.398, -0.075, 0.092],
        gripper_opening=0.018,
        gripper_contact=True,
        dist_to_target=0.237,
        dist_trend_5=[0.243, 0.241, 0.239, 0.238, 0.237],
        grasp_state="HELD",
        phase="TRANSPORT",
        planner_cost_trend=[0.094, 0.093, 0.093, 0.092, 0.092],
        calls_remaining=4,
    )


def self_test() -> None:
    payload = sample_payload()
    serialized = compact_json(payload)
    assert conservative_token_upper_bound(serialized) == 402
    request = DeepSeekSupervisorClient(api_key="self-test-only").request_body(payload)
    assert request["model"] == MODEL
    assert request["thinking"] == {"type": "disabled"}
    assert request["temperature"] == 0.1 and request["stream"] is False
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 128 and "reasoning_effort" not in request
    assert len(request["messages"]) == 2 and request["messages"][1]["content"] == serialized
    assert safe_endpoint(DEFAULT_BASE_URL) == "https://api.deepseek.com/v1/chat/completions"
    assert validate_response('{"decision":"CONTINUE"}').valid
    assert validate_response(
        '{"decision":"SUBGOAL","block_pos":[0.4,0.0,0.03],"yaw":0.2}'
    ).valid
    assert validate_response(
        '{"decision":"RECOVER","strategy":"REGRASP","ee_pos":[0.4,0.0,0.1]}'
    ).valid
    assert validate_response(
        '{"decision":"SUBGOAL","block_pos":[9,-9,2],"yaw":13}'
    ).valid
    invalid = (
        "not json",
        '{"decision":"CONTINUE","extra":1}',
        '{"decision":"SUBGOAL","block_pos":[0.4,0,0],"yaw":NaN}',
        '{"decision":"RECOVER","strategy":"UNKNOWN","ee_pos":[0.4,0,0.1]}',
        '{"decision":"CONTINUE","decision":"SUBGOAL"}',
    )
    for content in invalid:
        result = validate_response(content)
        assert not result.valid and result.decision == _fallback_result()

    candidate_payload = json.loads(serialized)
    candidate_payload["candidate_actions"] = [[0, 0, 0]]
    try:
        DeepSeekSupervisorClient(api_key="self-test-only").request_body(candidate_payload)
    except ValueError as exc:
        assert "forbidden non-symbolic state key" in str(exc)
    else:
        raise AssertionError("candidate actions escaped the exact payload schema")
    nested_secret = json.loads(serialized)
    nested_secret["block"]["authorization"] = "synthetic-secret"
    try:
        validate_state_payload(nested_secret)
    except ValueError as exc:
        assert "schema" in str(exc) or "forbidden" in str(exc)
    else:
        raise AssertionError("nested authorization escaped the exact payload schema")
    visible, was_truncated = _sanitize_visible_response("x" * (MAX_LOGGED_RESPONSE_CHARS + 1))
    assert visible == "x" * MAX_LOGGED_RESPONSE_CHARS and was_truncated
    dummy_secret = "sk-" + "SELFTEST123456"
    visible, _ = _sanitize_visible_response(f'{{"api_key":"{dummy_secret}"}}')
    assert dummy_secret not in str(visible) and "[REDACTED]" in str(visible)
    try:
        SupervisorConfig.from_value({"id_x": [0.0, 1.0]})
    except ValueError as exc:
        assert "unknown or forbidden" in str(exc)
    else:
        raise AssertionError("frozen ID box was unexpectedly configurable")

    scratch_root = Path("/root/autodl-tmp/tmp")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="brain-supervisor-selftest-", dir=scratch_root) as directory:
        logger = AtomicCallLogger(Path(directory) / "calls.json")
        budget = BudgetManager()
        client = DeepSeekSupervisorClient(api_key="self-test-only", logger=logger)
        responses = iter([
            RetryableTransportError("synthetic timeout"),
            {
                "id": "synthetic",
                "model": MODEL,
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": '{"decision":"CONTINUE"}'},
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_cache_hit_tokens": 20,
                    "prompt_cache_miss_tokens": 80,
                },
            },
        ])

        def fake_post(_: Mapping[str, Any]) -> dict[str, Any]:
            value = next(responses)
            if isinstance(value, BaseException):
                raise value
            return value

        client._post_json = fake_post  # type: ignore[method-assign]
        result = client.call(payload, episode_id="smoke-0", budget=budget)
        assert result.status == "ok" and result.attempts == 2
        assert budget.snapshot()["episodes"]["smoke-0"] == {"logical_calls": 1, "attempts": 2}
        assert logger.records[0]["status"] == "transport_retry"
        assert logger.records[1]["usage"]["prompt_cache_hit_tokens"] == 20
        assert logger.records[1]["sanitized_response_text"] == '{"decision":"CONTINUE"}'
        assert all("reasoning_content" not in compact_json(record) for record in logger.records)

        protocol_client = DeepSeekSupervisorClient(api_key="self-test-only")
        protocol_client._post_json = lambda _: {  # type: ignore[method-assign]
            "model": MODEL,
            "choices": [{"finish_reason": "length", "message": {"content": "{}"}}],
            "usage": {"total_tokens": 7},
        }
        protocol = protocol_client.call(payload, episode_id="smoke-1", budget=budget)
        assert protocol.status == "protocol_failure" and protocol.decision == _fallback_result()

        mismatch_client = DeepSeekSupervisorClient(api_key="self-test-only", logger=logger)
        mismatch_client._post_json = lambda _: {  # type: ignore[method-assign]
            "id": "synthetic-mismatch",
            "model": "different-model",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"decision":"CONTINUE"}'},
            }],
            "usage": {"total_tokens": 9},
        }
        try:
            mismatch_client.call(payload, episode_id="smoke-2", budget=budget)
        except FatalProviderError:
            pass
        else:
            raise AssertionError("provider model mismatch was not fatal")
        assert logger.records[-1]["status"] == "fatal_provider_failure"
        assert logger.records[-1]["actual_model"] == "different-model"
        assert logger.records[-1]["response_sha256"] == sha256_text('{"decision":"CONTINUE"}')
        assert "provider_model_mismatch" in logger.records[-1]["error"]

        unknown_client = DeepSeekSupervisorClient(api_key="self-test-only", logger=logger)
        unknown_client._post_json = lambda _: {  # type: ignore[method-assign]
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"decision":"CONTINUE"}'},
            }],
            "usage": {"total_tokens": 8},
        }
        try:
            unknown_client.call(payload, episode_id="smoke-3", budget=budget)
        except FatalProviderError:
            pass
        else:
            raise AssertionError("unknown provider model was not fatal")
        assert logger.records[-1]["actual_model"] == "unknown"

        reasoning_client = DeepSeekSupervisorClient(api_key="self-test-only", logger=logger)
        reasoning_client._post_json = lambda _: {  # type: ignore[method-assign]
            "model": MODEL,
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": '{"decision":"CONTINUE"}',
                    "reasoning_content": "synthetic reasoning that must never be logged",
                },
            }],
            "usage": {"total_tokens": 10},
        }
        try:
            reasoning_client.call(payload, episode_id="smoke-4", budget=budget)
        except FatalProviderError:
            pass
        else:
            raise AssertionError("non-empty reasoning_content was not fatal")
        assert logger.records[-1]["reasoning_present"] is True
        assert "synthetic reasoning" not in compact_json(logger.records[-1])

        invalid_json_client = DeepSeekSupervisorClient(api_key="self-test-only", logger=logger)
        invalid_json_client._post_json = lambda _: {  # type: ignore[method-assign]
            "model": MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": "not json"}}],
            "usage": {"total_tokens": 6},
        }
        invalid_json = invalid_json_client.call(payload, episode_id="smoke-5", budget=budget)
        assert invalid_json.status == "protocol_failure"
        assert logger.records[-1]["sanitized_response_text"] == "not json"
        assert logger.records[-1]["response_sha256"] == sha256_text("not json")

        facade = BrainSupervisor(None, Path(directory) / "facade", api_key="self-test-only")
        facade.client._post_json = lambda _: {  # type: ignore[method-assign]
            "id": "synthetic-facade",
            "model": MODEL,
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": '{"decision":"SUBGOAL","block_pos":[9,-9,2],"yaw":13}'
                },
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 4, "total_tokens": 54},
        }
        facade_result = facade.decide(payload, "STALLED", 3, 64)
        assert facade_result["decision"] == "SUBGOAL"
        assert facade_result["block_pos"] == [ID_X[1], ID_Y[0], ID_Z[1]]
        assert -math.pi <= facade_result["yaw"] <= math.pi
        assert facade_result["call_record"]["clamp_applied"] is True
        assert facade.logger.records[-1]["requested_decision"]["block_pos"] == [9.0, -9.0, 2.0]
        assert facade.logger.records[-1]["clamped_decision"]["block_pos"] == [ID_X[1], ID_Y[0], ID_Z[1]]
        assert facade.logger.records[-1]["yaw_wrap_applied"] is True
        assert facade_result["call_record"]["actual_model"] == MODEL
        assert facade.manifest()["single_turn_no_history"] is True
        assert facade.summary()["logical_calls"] == 1
        try:
            BrainSupervisor(None, Path(directory) / "facade", api_key="self-test-only")
        except FileExistsError:
            pass
        else:
            raise AssertionError("formal facade unexpectedly allowed resume")

        try:
            logger.append({"payload": {"nested": {"authorization": "synthetic-secret"}}})
        except ValueError as exc:
            assert "sensitive log field" in str(exc)
        else:
            raise AssertionError("nested authorization escaped recursive log scanning")

        tiny = BudgetManager(max_total_tokens=1)
        tiny.begin_logical_call("tiny")
        try:
            tiny.reserve_attempt("tiny", 2)
        except BudgetExceeded:
            pass
        else:
            raise AssertionError("global hard token budget was not enforced")

        logical_cap = BudgetManager()
        for _ in range(MAX_LOGICAL_CALLS_PER_EPISODE):
            logical_cap.begin_logical_call("cap")
        try:
            logical_cap.begin_logical_call("cap")
        except BudgetExceeded:
            pass
        else:
            raise AssertionError("per-episode logical-call budget was not enforced")

    print("brain_supervisor self-test: PASS")


def dry_run() -> None:
    payload = sample_payload()
    client = DeepSeekSupervisorClient(api_key="dry-run-only")
    body = client.request_body(payload)
    print(compact_json({
        "provider": "deepseek",
        "model": MODEL,
        "endpoint": client.endpoint,
        "thinking": THINKING,
        "temperature": TEMPERATURE,
        "stream": False,
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_TOKENS,
        "payload_bytes": len(compact_json(payload).encode("utf-8")),
        "request_token_upper_bound": conservative_token_upper_bound(compact_json(body)) + MAX_TOKENS,
        "external_request_sent": False,
    }))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return value


def main(args: argparse.Namespace) -> int:
    if args.self_test:
        self_test()
    else:
        dry_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
