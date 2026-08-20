#!/usr/bin/env python3
"""Call blind Cube rerankers without persisting keys or chain-of-thought.

Only the public prompt package is read.  Each persisted response contains the
single validated PICK plus non-sensitive API metadata; raw provider responses
and reasoning fields are intentionally discarded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AILAB = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC = AILAB / "outputs/rerank_pilot/prompts/public"
DEFAULT_OUTPUT_ROOT = AILAB / "outputs/rerank_pilot/responses"
ALLOWED_ROOT = (AILAB / "outputs/rerank_pilot").resolve()
OPENAI_REASONING_OUTPUT_BUDGET = 8192
# Four clean DeepSeek responses exhausted 8192 tokens, and a second full-arm
# attempt exhausted 32768 on three consecutive prompts. A new full 36-prompt
# run therefore uses the largest budget that still fits beside the ~20k-token
# prompt in the model context and leaves room for the required one-line PICK.
DEEPSEEK_REASONING_OUTPUT_BUDGET = 65536
PICK_RE = re.compile(r"^PICK: ([0-9]|[1-9][0-9]|[12][0-9][0-9])$")
CODEX_COMPAT_USER_AGENT = (
    "codex_vscode/0.147.0-alpha.6.5 (Ubuntu 22.4.0; x86_64) "
    "dumb (codex_exec; 0.147.0-alpha.6.5)"
)
SYSTEM = (
    "Act as the blind robot-motion reranker described by the user prompt. "
        "Return exactly one line PICK: <integer candidate ID>, with no analysis or explanation."
)


class RetryableError(RuntimeError):
    """Provider failure safe to retry; message never contains response bodies."""


class FatalProviderError(RuntimeError):
    """Provider authentication/configuration failure that must stop the batch."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_output(path: Path) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"refusing symlink output: {raw}")
    resolved = raw.resolve()
    if resolved == ALLOWED_ROOT or ALLOWED_ROOT not in resolved.parents:
        raise ValueError(f"output must be a concrete child of {ALLOWED_ROOT}")
    return resolved


def _read_openai_sse(response: Any) -> dict[str, Any]:
    """Return only the terminal Responses object from an SSE stream.

    Reasoning and text delta events are deliberately ignored instead of being
    accumulated or persisted. Streaming avoids idle proxy disconnects during
    long xhigh reasoning requests.
    """
    terminal: dict[str, Any] | None = None
    for raw_line in response:
        if not isinstance(raw_line, bytes) or not raw_line.startswith(b"data:"):
            continue
        data = raw_line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            raise RetryableError("invalid JSON in Responses SSE stream") from None
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type in {"response.completed", "response.incomplete"}:
            value = event.get("response")
            if isinstance(value, dict):
                terminal = value
        elif event_type in {"response.failed", "error"}:
            raise RetryableError("provider reported a failed Responses SSE event")
    if terminal is None:
        raise RetryableError("Responses SSE stream ended without a terminal response")
    return terminal


def _endpoint(base: str, suffix: str) -> str:
    base = base.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("provider base URL must be HTTPS")
    if base.endswith("/" + suffix):
        return base
    return base + "/" + suffix


def _post(
    url: str,
    key: str,
    payload: dict[str, Any],
    timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
            content_type = response.headers.get_content_type()
            if content_type == "text/event-stream":
                return _read_openai_sse(response)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        # Never read or print the response body: it can contain model reasoning.
        if exc.code in {408, 409, 425, 429} or exc.code >= 500:
            raise RetryableError(f"HTTP {exc.code}") from None
        raise FatalProviderError(f"provider rejected request with HTTP {exc.code}") from None
    except (
        urllib.error.URLError,
        TimeoutError,
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        ConnectionResetError,
        BrokenPipeError,
    ) as exc:
        raise RetryableError(type(exc).__name__) from None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("HTTP 200 response was not JSON") from None
    if not isinstance(value, dict):
        raise RuntimeError("HTTP 200 response was not an object")
    return value


def _openai_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks)


def _deepseek_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    # reasoning_content is deliberately ignored and never persisted.
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _pick(text: str) -> str:
    match = PICK_RE.fullmatch(text)
    if not match:
        raise RuntimeError("response did not follow the exact single-PICK protocol")
    return match.group(1)


def _usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    safe = {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"):
        if isinstance(value.get(key), int):
            safe[key] = value[key]
    return safe


def _provider_spec(provider: str, prompt: str) -> tuple[str, dict[str, Any], str]:
    if provider == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        payload = {
            "model": "gpt-5.5",
            "instructions": SYSTEM,
            "input": prompt,
            "reasoning": {"effort": "xhigh"},
            "store": False,
            "stream": True,
            # Responses counts both visible output and hidden reasoning against
            # this limit.  A tiny value can exhaust xhigh reasoning before the
            # required one-line PICK is emitted.
            "max_output_tokens": OPENAI_REASONING_OUTPUT_BUDGET,
        }
        return _endpoint(base, "responses"), payload, "gpt-5.5"
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    payload = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        # DeepSeek thinking and the final content share the response budget.
        "max_tokens": DEEPSEEK_REASONING_OUTPUT_BUDGET,
    }
    return _endpoint(base, "chat/completions"), payload, "deepseek-v4-pro"


def _call_one(provider: str, prompt: str, attempts: int, timeout: float) -> dict[str, Any]:
    url, payload, requested_model = _provider_spec(provider, prompt)
    key = os.environ.get("OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY", "")
    if not key:
        env_name = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
        raise RuntimeError(f"missing required environment variable {env_name}")
    last_error = "unknown failure"
    for attempt in range(1, attempts + 1):
        try:
            # The user-provided OpenAI-compatible endpoint is configured for
            # Codex clients and rejects Python's default urllib User-Agent.
            # These two non-secret identity headers match the installed Codex
            # client; DeepSeek receives no Codex-specific headers.
            extra_headers = (
                {"User-Agent": CODEX_COMPAT_USER_AGENT, "originator": "codex_vscode"}
                if provider == "openai" else None
            )
            response = _post(url, key, payload, timeout, extra_headers)
            text = _openai_text(response) if provider == "openai" else _deepseek_text(response)
            # Persist only the final non-empty line, never preceding free-form
            # analysis. Exact protocol validation still applies to the full text.
            final_line = next((line for line in reversed(text.splitlines()) if line.strip()), "")
            final_hash = _sha_bytes(final_line.encode("utf-8"))
            metadata = {
                "provider": provider,
                "requested_model": requested_model,
                "actual_model": response.get("model") if isinstance(response.get("model"), str) else "unknown",
                "response_id": response.get("id") if isinstance(response.get("id"), str) else None,
                "system_fingerprint": response.get("system_fingerprint") if isinstance(response.get("system_fingerprint"), str) else None,
                "final_line": final_line,
                "final_line_sha256": final_hash,
                "attempt": attempt,
                "usage": _usage(response.get("usage")),
                "received_at_utc": datetime.now(timezone.utc).isoformat(),
                "reasoning_persisted": False,
            }
            metadata["sanitized_response_sha256"] = _sha_bytes(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            try:
                pick = _pick(text)
            except RuntimeError as exc:
                return {**metadata, "status": "protocol_failure", "pick_id": None, "failure": str(exc)}
            return {**metadata, "status": "ok", "pick_id": pick}
        except RetryableError as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(min(20.0, 1.5 * (2 ** (attempt - 1))))
        except FatalProviderError:
            raise
        except RuntimeError as exc:
            # HTTP 200 non-JSON/non-object is a terminal protocol failure. It has
            # no safely extracted final line to persist.
            return {
                "status": "protocol_failure", "provider": provider,
                "requested_model": requested_model, "actual_model": "unknown",
                "pick_id": None, "attempt": attempt, "failure": str(exc),
                "received_at_utc": datetime.now(timezone.utc).isoformat(),
                "reasoning_persisted": False,
            }
    return {
        "status": "transport_failure", "provider": provider,
        "requested_model": requested_model, "actual_model": "unknown",
        "pick_id": None, "attempt": attempts, "failure": last_error,
        "received_at_utc": datetime.now(timezone.utc).isoformat(),
        "reasoning_persisted": False,
    }


def _validate_public(public: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], str]]]:
    manifest_path = public / "manifest.json"
    manifest = _read_json(manifest_path)
    entries = manifest.get("prompts")
    if manifest.get("num_prompts") != 36 or not isinstance(entries, list) or len(entries) != 36:
        raise RuntimeError("public package does not contain exactly 36 prompts")
    result = []
    for entry in entries:
        prompt_path = (public / entry["prompt_file"]).resolve()
        if public.resolve() not in prompt_path.parents:
            raise RuntimeError("prompt path escapes public package")
        raw = prompt_path.read_bytes()
        if _sha_bytes(raw) != entry["prompt_sha256"]:
            raise RuntimeError(f"prompt hash mismatch: {entry['prompt_id']}")
        result.append((entry, raw.decode("utf-8")))
    return manifest, result


def main(args: argparse.Namespace) -> int:
    public = args.public.expanduser().resolve()
    manifest, prompts = _validate_public(public)
    providers = ["openai", "deepseek"] if args.provider == "both" else [args.provider]
    if args.dry_run:
        # Deliberately return before reading any API-key environment variable.
        for provider in providers:
            url, payload, model = _provider_spec(provider, "<prompt omitted>")
            payload.pop("input", None)
            payload.pop("messages", None)
            print(f"validated provider={provider} model={model} endpoint={url} prompts={len(prompts)}")
        return 0

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id) or args.run_id in {".", ".."}:
        raise ValueError("--run-id must be a single safe path component")
    if args.output_root.expanduser().resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError(f"--output-root is frozen to {DEFAULT_OUTPUT_ROOT}")
    output = _safe_output(args.output_root / args.run_id)
    if output.exists() and not args.resume:
        raise FileExistsError(f"run-id already exists; choose a new run-id: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_sha = _sha_file(public / "manifest.json")
    any_transport_failure = False
    for provider in providers:
        provider_dir = output / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        complete_path = provider_dir / "COMPLETE.json"
        if complete_path.exists():
            raise FileExistsError(f"frozen response set already exists: {complete_path}")
        pending: list[tuple[dict[str, Any], str, Path]] = []
        for entry, prompt in prompts:
            destination = provider_dir / f"{entry['prompt_id']}.json"
            if destination.exists():
                existing = _read_json(destination)
                if args.resume and existing.get("prompt_sha256") == entry["prompt_sha256"]:
                    if existing.get("status") != "transport_failure":
                        continue
                    destination.unlink()
                else:
                    raise FileExistsError(f"response exists (use --resume only for matching prompts): {destination}")
            pending.append((entry, prompt, destination))

        def process_one(item: tuple[dict[str, Any], str, Path]) -> tuple[str, str, Any]:
            entry, prompt, destination = item
            result = _call_one(provider, prompt, args.max_attempts, args.timeout)
            if result.get("status") == "ok" and int(result["pick_id"]) not in entry["displayed_candidate_ids"]:
                result.update({"status": "protocol_failure", "failure": "PICK was not a displayed candidate", "pick_id": None})
            result.update({"prompt_id": entry["prompt_id"], "prompt_sha256": entry["prompt_sha256"]})
            _write_json(destination, result)
            return entry["prompt_id"], result["status"], result["pick_id"]

        if args.workers == 1:
            completed = map(process_one, pending)
            for prompt_id, status, pick_id in completed:
                print(f"{provider}: {prompt_id} status={status} pick={pick_id}", flush=True)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
            items = iter(pending)
            futures: set[concurrent.futures.Future[tuple[str, str, Any]]] = set()
            for _ in range(args.workers):
                item = next(items, None)
                if item is not None:
                    futures.add(executor.submit(process_one, item))
            try:
                while futures:
                    done, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    # Resolve every completion before submitting replacements;
                    # a fatal error therefore cannot let an idle worker pull a
                    # fifth task from an already-populated executor queue.
                    results = [future.result() for future in done]
                    for prompt_id, status, pick_id in results:
                        print(f"{provider}: {prompt_id} status={status} pick={pick_id}", flush=True)
                    for _ in results:
                        item = next(items, None)
                        if item is not None:
                            futures.add(executor.submit(process_one, item))
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
        response_hashes = {
            entry["prompt_id"]: _sha_file(provider_dir / f"{entry['prompt_id']}.json")
            for entry, _ in prompts
        }
        statuses = [_read_json(provider_dir / f"{entry['prompt_id']}.json").get("status") for entry, _ in prompts]
        if "transport_failure" in statuses:
            any_transport_failure = True
            print(f"{provider}: network-interrupted run remains resumable; COMPLETE not written")
            continue
        _write_json(complete_path, {
            "provider": provider,
            "public_manifest_sha256": manifest_sha,
            "num_responses": len(response_hashes),
            "response_sha256": response_hashes,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"froze {len(response_hashes)} responses: {complete_path}")
    return 2 if any_transport_failure else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", required=True, help="New immutable run directory name")
    p.add_argument("--provider", choices=("openai", "deepseek", "both"), required=True)
    p.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
