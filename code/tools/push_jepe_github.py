#!/usr/bin/env python3
"""Create three semantic commits from scanned staging and push via askpass."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import scan_release_secrets as secret_scan


AILAB_ROOT = Path(__file__).resolve().parents[2]
STAGING = AILAB_ROOT / "release/github_staging/JEPE_OOD_explore"
SCAN_REPORT = AILAB_ROOT / "outputs/release/github_staging_scan_final.json"
PUSH_REPORT = AILAB_ROOT / "outputs/release/github_push_report.json"
REMOTE = "https://github.com/lwbscu/JEPE_OOD_explore.git"
BRANCH = "main"
TOKEN_ENV_NAMES = ("GH_TOKEN", "GITHUB_TOKEN")


def _run(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _token() -> str:
    found = [os.environ.get(name, "") for name in TOKEN_ENV_NAMES]
    found = [value for value in found if value]
    if len(found) != 1:
        raise RuntimeError("set exactly one of GH_TOKEN or GITHUB_TOKEN")
    return found[0]


def _assert_scan(staging: Path, report_path: Path) -> dict[str, Any]:
    recorded = _read_json(report_path)
    current = secret_scan.scan(staging)
    if recorded.get("status") != "PASS" or current.get("status") != "PASS":
        raise PermissionError("GitHub staging scan is not PASS")
    if recorded.get("manifest_sha256") != current.get("manifest_sha256"):
        raise ValueError("GitHub staging changed after the final scan")
    return current


def _scan_index(staging: Path) -> dict[str, Any]:
    temporary_root = AILAB_ROOT.parent / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jepe-index-", dir=temporary_root) as directory:
        export = Path(directory)
        prefix = str(export.resolve()) + "/"
        _run(
            ["git", "checkout-index", "--all", f"--prefix={prefix}"],
            cwd=staging,
        )
        report = secret_scan.scan(export)
        if report["status"] != "PASS":
            raise PermissionError("staged Git index failed the secret scan")
        return {
            key: report[key]
            for key in (
                "file_count",
                "total_bytes",
                "manifest_sha256",
                "status",
            )
        }


def _commit(staging: Path, paths: list[str], message: str) -> str:
    _run(["git", "add", "--", *paths], cwd=staging)
    _scan_index(staging)
    _run(["git", "commit", "-m", message], cwd=staging)
    return _run(["git", "rev-parse", "HEAD"], cwd=staging)


def run(args: argparse.Namespace) -> int:
    staging = args.staging.expanduser().resolve()
    if staging != STAGING.resolve():
        raise ValueError(f"GitHub staging root is frozen: expected={STAGING}, actual={staging}")
    scan = _assert_scan(staging, args.scan_report.expanduser().resolve())
    if (staging / ".git").exists():
        raise FileExistsError(f"staging already contains .git: {staging}")
    token = _token()
    _run(["git", "init", "-b", BRANCH], cwd=staging)
    _run(["git", "config", "user.name", "JEPE OOD Release"], cwd=staging)
    _run(["git", "config", "user.email", "release@local.invalid"], cwd=staging)
    _run(["git", "remote", "add", "origin", REMOTE], cwd=staging)

    commits = [
        _commit(
            staging,
            ["code", ".gitignore", "LICENSE", "NOTICE"],
            "Add reproducible experiment code",
        ),
        _commit(
            staging,
            ["README.md", "docs"],
            "Document controlled OOD experiments and results",
        ),
        _commit(
            staging,
            ["evidence"],
            "Add curated visual evidence",
        ),
    ]
    if _run(["git", "status", "--porcelain"], cwd=staging):
        raise RuntimeError("GitHub staging has uncommitted release files")

    temporary_root = AILAB_ROOT.parent / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    askpass = None
    try:
        fd, name = tempfile.mkstemp(prefix="jepe-askpass-", dir=temporary_root)
        askpass = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  *sername*) printf '%s\\n' \"x-access-token\" ;;\n"
                "  *) printf '%s\\n' \"$JEPE_GITHUB_TOKEN\" ;;\n"
                "esac\n"
            )
        askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        env = dict(os.environ)
        env.pop("GH_TOKEN", None)
        env.pop("GITHUB_TOKEN", None)
        env["JEPE_GITHUB_TOKEN"] = token
        env["GIT_ASKPASS"] = str(askpass)
        env["GIT_TERMINAL_PROMPT"] = "0"
        push_error = None
        for attempt in range(1, 4):
            try:
                _run(["git", "push", "-u", "origin", BRANCH], cwd=staging, env=env)
                push_error = None
                break
            except subprocess.CalledProcessError as error:
                push_error = error
                if attempt < 3:
                    time.sleep(5 * attempt)
        if push_error is not None:
            raise RuntimeError("GitHub push failed after 3 attempts") from push_error
    finally:
        if askpass is not None and askpass.exists():
            askpass.unlink()

    remote_head = _run(["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], cwd=staging)
    if not remote_head.startswith(commits[-1]):
        raise RuntimeError(
            f"remote HEAD mismatch: expected={commits[-1]}, actual={remote_head}"
        )
    report = {
        "format_version": "jepe_github_push_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "remote": REMOTE,
        "branch": BRANCH,
        "commits": commits,
        "remote_head": remote_head.split()[0],
        "staging_scan_manifest_sha256": scan["manifest_sha256"],
        "status": "PASS",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=STAGING)
    parser.add_argument("--scan-report", type=Path, default=SCAN_REPORT)
    parser.add_argument("--report", type=Path, default=PUSH_REPORT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
