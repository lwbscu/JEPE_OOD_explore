#!/usr/bin/env python3
"""Upload the scanned JEPE OOD staging tree to its frozen HF dataset repo.

Authentication is accepted only through an environment variable.  A scan
report is not treated as a timeless approval: the staging tree is rescanned
and its manifest hash must still match immediately before the first upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi

import scan_release_secrets as secret_scan


AILAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGING = AILAB_ROOT / "release/hf_staging/JEPE_OOD_explore"
DEFAULT_SCAN_REPORT = AILAB_ROOT / "outputs/release/hf_staging_scan_final.json"
DEFAULT_REPORT = AILAB_ROOT / "outputs/release/hf_upload_report.json"
REPO_ID = "scilwb/JEPE_OOD_explore"
REPO_TYPE = "dataset"
TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
SEGMENTS = (
    "weights",
    "datasets/offpolicy_cube_v1",
    "datasets/offpolicy_cube_v2",
    "memory_index",
    "reports",
    "evidence",
)
ROOT_FILES = ("README.md", "LICENSE", "NOTICE")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _token_from_environment() -> str:
    present = [(name, os.environ.get(name, "")) for name in TOKEN_ENV_NAMES]
    present = [(name, value) for name, value in present if value]
    if len(present) != 1:
        names = ", ".join(TOKEN_ENV_NAMES)
        raise RuntimeError(f"set exactly one authentication variable: {names}")
    return present[0][1]


def _local_files(root: Path, prefix: str | None = None) -> dict[str, int]:
    base = root if prefix is None else root / prefix
    if not base.exists():
        raise FileNotFoundError(base)
    paths: Iterable[Path] = [base] if base.is_file() else base.rglob("*")
    result: dict[str, int] = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError(f"symlinks are forbidden in HF staging: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.stat().st_size
    return result


def _remote_files(api: HfApi) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in api.list_repo_tree(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        recursive=True,
        expand=True,
    ):
        if getattr(item, "type", None) == "file" or item.__class__.__name__ == "RepoFile":
            size = getattr(item, "size", None)
            lfs = getattr(item, "lfs", None)
            result[str(item.path)] = {
                "size": int(size) if size is not None else -1,
                "blob_id": str(getattr(item, "blob_id", "")),
                "lfs_sha256": str(getattr(lfs, "sha256", "")) if lfs else "",
            }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_identity(path: Path, remote: dict[str, Any]) -> tuple[str, str, str]:
    remote_lfs = str(remote.get("lfs_sha256", ""))
    if remote_lfs:
        return "sha256", _sha256(path), remote_lfs
    return "git_blob_sha1", _git_blob_sha1(path), str(remote.get("blob_id", ""))


def _retry(label: str, operation: Any, attempts: int) -> Any:
    error = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as current:  # network/provider errors vary by version
            error = current
            if attempt == attempts:
                break
            delay = min(45, 5 * (3 ** (attempt - 1)))
            print(f"{label}: attempt {attempt}/{attempts} failed; retrying in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from error


def _verify_prefix(
    api: HfApi, staging: Path, prefix: str | None, attempts: int
) -> dict[str, Any]:
    expected = (
        {
            filename: (staging / filename).stat().st_size
            for filename in ROOT_FILES
        }
        if prefix is None
        else _local_files(staging, prefix)
    )

    def check() -> dict[str, Any]:
        remote = _remote_files(api)
        missing = sorted(set(expected) - set(remote))
        size_mismatch = sorted(
            [
                {
                    "path": path,
                    "expected": size,
                    "actual": remote[path]["size"],
                }
                for path, size in expected.items()
                if path in remote
                and remote[path]["size"] >= 0
                and remote[path]["size"] != size
            ],
            key=lambda item: item["path"],
        )
        if prefix is None:
            remote_scope = {
                path for path in remote if "/" not in path and path != ".gitattributes"
            }
        else:
            remote_scope = {
                path for path in remote if path == prefix or path.startswith(prefix + "/")
            }
        unexpected = sorted(remote_scope - set(expected))
        size_mismatch_paths = {item["path"] for item in size_mismatch}
        content_mismatch: list[dict[str, str]] = []
        for path in sorted(set(expected) & set(remote)):
            if path in size_mismatch_paths:
                continue
            algorithm, local_identity, remote_identity = _content_identity(
                staging / path, remote[path]
            )
            if not remote_identity or local_identity != remote_identity:
                content_mismatch.append(
                    {
                        "path": path,
                        "algorithm": algorithm,
                        "expected": local_identity,
                        "actual": remote_identity or "<missing>",
                    }
                )
        if missing or size_mismatch or unexpected or content_mismatch:
            raise RuntimeError(
                f"remote verification failed for {prefix or 'root'}: "
                f"missing={missing[:10]}, size_mismatch={size_mismatch[:10]}, "
                f"unexpected={unexpected[:10]}, "
                f"content_mismatch={content_mismatch[:10]}"
            )
        return {
            "prefix": prefix or "root",
            "file_count": len(expected),
            "total_bytes": int(sum(expected.values())),
            "missing_count": 0,
            "size_mismatch_count": 0,
            "unexpected_count": 0,
            "content_mismatch_count": 0,
        }

    return _retry(f"verify {prefix or 'root'}", check, attempts)


def _verify_complete(api: HfApi, staging: Path, attempts: int) -> dict[str, Any]:
    expected = _local_files(staging)

    def check() -> dict[str, Any]:
        remote = _remote_files(api)
        missing = sorted(set(expected) - set(remote))
        size_mismatch = sorted(
            [
                {
                    "path": path,
                    "expected": size,
                    "actual": remote[path]["size"],
                }
                for path, size in expected.items()
                if path in remote
                and remote[path]["size"] >= 0
                and remote[path]["size"] != size
            ],
            key=lambda item: item["path"],
        )
        unexpected = sorted(set(remote) - set(expected) - {".gitattributes"})
        size_mismatch_paths = {item["path"] for item in size_mismatch}
        content_mismatch: list[dict[str, str]] = []
        for path in sorted(set(expected) & set(remote)):
            if path in size_mismatch_paths:
                continue
            algorithm, local_identity, remote_identity = _content_identity(
                staging / path, remote[path]
            )
            if not remote_identity or local_identity != remote_identity:
                content_mismatch.append(
                    {
                        "path": path,
                        "algorithm": algorithm,
                        "expected": local_identity,
                        "actual": remote_identity or "<missing>",
                    }
                )
        if missing or size_mismatch or unexpected or content_mismatch:
            raise RuntimeError(
                "complete remote verification failed: "
                f"missing={missing[:10]}, size_mismatch={size_mismatch[:10]}, "
                f"unexpected={unexpected[:10]}, "
                f"content_mismatch={content_mismatch[:10]}"
            )
        return {
            "prefix": "complete_staging",
            "file_count": len(expected),
            "total_bytes": int(sum(expected.values())),
            "missing_count": 0,
            "size_mismatch_count": 0,
            "unexpected_count": 0,
            "content_mismatch_count": 0,
        }

    return _retry("verify complete staging", check, attempts)


def run(args: argparse.Namespace) -> int:
    staging = args.staging.expanduser().resolve()
    scan_report_path = args.scan_report.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if AILAB_ROOT.parent.resolve() not in staging.parents:
        raise ValueError(f"staging must be on /root/autodl-tmp: {staging}")
    recorded = _read_json(scan_report_path)
    current = secret_scan.scan(staging)
    if recorded.get("status") != "PASS" or current.get("status") != "PASS":
        raise PermissionError("HF staging secret scan is not PASS")
    if Path(str(recorded.get("root_realpath"))).resolve() != staging:
        raise ValueError("scan report was produced for a different staging root")
    if recorded.get("manifest_sha256") != current.get("manifest_sha256"):
        raise ValueError(
            "staging changed after secret scan: "
            f"expected_manifest={recorded.get('manifest_sha256')}, "
            f"actual_manifest={current.get('manifest_sha256')}"
        )
    token = _token_from_environment()
    api = HfApi(token=token)
    who = api.whoami()
    identity = str(who.get("name") or who.get("fullname") or "")
    if identity != "scilwb":
        raise PermissionError(f"unexpected HF identity: {identity!r}")
    info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    if bool(getattr(info, "private", True)):
        raise PermissionError("target HF dataset repo must remain public")

    selected = list(SEGMENTS) if args.segment == "all" else [args.segment]
    uploads: list[dict[str, Any]] = []
    for filename in ROOT_FILES:
        source = staging / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        _retry(
            f"upload {filename}",
            lambda source=source, filename=filename: api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Add release metadata: {filename}",
            ),
            args.attempts,
        )
    uploads.append(_verify_prefix(api, staging, None, args.attempts))

    for prefix in selected:
        local = staging / prefix
        _retry(
            f"upload {prefix}",
            lambda local=local, prefix=prefix: api.upload_folder(
                folder_path=str(local),
                path_in_repo=prefix,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Upload JEPE OOD release segment: {prefix}",
            ),
            args.attempts,
        )
        uploads.append(_verify_prefix(api, staging, prefix, args.attempts))
        print(json.dumps(uploads[-1], sort_keys=True))

    if selected == list(SEGMENTS):
        uploads.append(_verify_complete(api, staging, args.attempts))

    report = {
        "format_version": "jepe_hf_upload_v2",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "authenticated_identity": identity,
        "staging": str(staging),
        "scan_manifest_sha256": current["manifest_sha256"],
        "segments": selected,
        "remote_content_verification": {
            "lfs": "sha256",
            "git_files": "git_blob_sha1",
            "unexpected_files": "rejected except repository .gitattributes",
        },
        "verifications": uploads,
        "status": "PASS",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--scan-report", type=Path, default=DEFAULT_SCAN_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--segment", choices=("all", *SEGMENTS), default="all")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    if args.attempts < 1 or args.attempts > 5:
        parser.error("--attempts must be between 1 and 5")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
