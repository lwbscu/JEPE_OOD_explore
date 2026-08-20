#!/usr/bin/env python3
"""Fail closed secret and path scan for JEPE OOD release staging trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCANNER_VERSION = "jepe_release_secret_scan_v1"
CHUNK_BYTES = 8 << 20
OVERLAP_BYTES = 512

HARD_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("openai_style_key", re.compile(rb"sk-[A-Za-z0-9_-]{16,}")),
    ("huggingface_token", re.compile(rb"hf_[A-Za-z0-9]{20,}")),
    ("github_classic_token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("github_fine_grained_token", re.compile(rb"github_pat_[A-Za-z0-9_]{20,}")),
    ("authorization_header", re.compile(rb"(?i)authorization\s*:\s*(?:bearer|basic)\s+\S+")),
    ("private_key_header", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}")),
)

CONTEXT_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "credential_assignment",
        re.compile(
            rb"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
            rb"\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{12,}"
        ),
    ),
)

DENY_EXACT = {
    "llm_api.md",
    ".netrc",
    ".git-credentials",
    "credentials",
    "credentials.json",
}
DENY_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}

# These exact source revisions contain only self-test/dry-run api_key
# placeholders. Their assignment-shaped strings were reviewed without copying
# their values into this scanner. Any source edit changes the hash and returns
# the file to fail-closed context review.
REVIEWED_CONTEXT_HASHES = {
    "code/tools/brain_supervisor.py": (
        "57e49844d4cc0d2981a03c9b99fa143cfd565e9f1737bcb79cb8af8d8efd862e"
    ),
    "code/tools/brain_supervisor_v2.py": (
        "6c05a61a1312e85f3dedad8e94db2cca2656cffff3c19bcd9bd1281a64003f1e"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        for name in list(dirs):
            path = Path(base) / name
            if path.is_symlink():
                yield path
                dirs.remove(name)
        for name in files:
            yield Path(base) / name


def _denylisted_name(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in DENY_EXACT
        or name.startswith(".env")
        or "credential" in name
        or "token" in name
        or path.suffix.lower() in DENY_SUFFIXES
    )


def _scan_patterns(path: Path) -> dict[str, int]:
    counts = {rule: 0 for rule, _ in HARD_RULES + CONTEXT_RULES}
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            block = tail + chunk
            for rule, pattern in HARD_RULES + CONTEXT_RULES:
                counts[rule] += len(pattern.findall(block))
            tail = block[-OVERLAP_BYTES:]
    return {rule: count for rule, count in counts.items() if count}


def scan(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    findings: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    symlink_count = 0
    outside_root_count = 0
    total_bytes = 0
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlink_count += 1
            try:
                resolved = path.resolve(strict=True)
                if resolved != root and root not in resolved.parents:
                    outside_root_count += 1
            except OSError:
                outside_root_count += 1
            findings.append({"path": rel, "rule_id": "symlink_forbidden", "match_count": 1})
            continue
        stat = path.stat()
        total_bytes += stat.st_size
        digest = _sha256(path)
        manifest.append({"path": rel, "size": stat.st_size, "sha256": digest})
        if _denylisted_name(path):
            findings.append(
                {
                    "path": rel,
                    "rule_id": "denylisted_filename",
                    "match_count": 1,
                    "size": stat.st_size,
                    "sha256": digest,
                }
            )
        for rule, count in _scan_patterns(path).items():
            finding = {
                "path": rel,
                "rule_id": rule,
                "match_count": count,
                "size": stat.st_size,
                "sha256": digest,
            }
            if rule in {item[0] for item in CONTEXT_RULES} and REVIEWED_CONTEXT_HASHES.get(rel) == digest:
                finding["review_status"] = "known_noncredential_test_placeholder"
            findings.append(finding)
    manifest_blob = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    hard_ids = {rule for rule, _ in HARD_RULES}
    context_ids = {rule for rule, _ in CONTEXT_RULES}
    denylisted = sum(item["rule_id"] == "denylisted_filename" for item in findings)
    hard = sum(
        int(item["match_count"])
        for item in findings
        if item["rule_id"] in hard_ids
    )
    context = sum(
        int(item["match_count"])
        for item in findings
        if item["rule_id"] in context_ids
    )
    reviewed_context = sum(
        item.get("review_status") == "known_noncredential_test_placeholder"
        for item in findings
    )
    blocking = [
        item
        for item in findings
        if item.get("review_status") != "known_noncredential_test_placeholder"
    ]
    status = "PASS" if not blocking and not outside_root_count else "FAIL"
    return {
        "format_version": SCANNER_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root_realpath": str(root),
        "manifest_sha256": hashlib.sha256(manifest_blob).hexdigest(),
        "file_count": len(manifest),
        "total_bytes": total_bytes,
        "symlink_count": symlink_count,
        "outside_root_count": outside_root_count,
        "denylisted_name_count": denylisted,
        "hard_secret_match_count": hard,
        "context_match_count": context,
        "reviewed_context_count": reviewed_context,
        "blocking_finding_count": len(blocking),
        "findings": findings,
        "files": manifest,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = scan(args.root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in (
                "root_realpath", "file_count", "total_bytes", "manifest_sha256", "status"
            )},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
