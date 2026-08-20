# Phase 0 Authentication and Secret Boundary Report

Generated: 2026-08-20 UTC

## Authentication

- GitHub target: `lwbscu/JEPE_OOD_explore`
- GitHub authenticated identity: `lwbscu`
- GitHub repository exists, is public, and the supplied credential reports
  `push=true` and `admin=true` for the target repository.
- Hugging Face target: `scilwb/JEPE_OOD_explore` (`dataset`)
- Hugging Face authenticated identity: `scilwb`
- The dataset repository exists, is public, is not gated, and currently has one
  repository file.

The `gh` CLI is not installed and the HF CLI has no persisted login. This is not
an upload blocker: GitHub will use a temporary in-memory `GIT_ASKPASS` flow and
Hugging Face will use `HfApi(token=...)`. Credentials will not be put in remotes,
command arguments, logs, cache files, reports, or staging trees.

## Secret Boundary

`/root/autodl-tmp/ailab/llm_api.md` is a confirmed credential source and is
permanently excluded from both release trees. It will not be copied, linked,
committed, uploaded, or quoted. Other blocked names include `.env*`, `.netrc`,
Git credential files, token/credential files, private keys, and PEM key files.

The final scanner will process the exact GitHub and HF staging manifests,
including binary byte streams. It will reject escaping symlinks and the following
hard patterns without recording matched text:

- OpenAI-style `sk-` credentials;
- Hugging Face `hf_` credentials;
- GitHub classic/fine-grained token prefixes;
- Authorization headers and private-key headers;
- common cloud access-key forms.

Findings contain only relative path, rule identifier, match count, size, and file
SHA-256. Both staging trees and the Git index must pass before any upload.

## Storage and Runtime Baseline

- Workspace: `/root/autodl-tmp/ailab` on the persistent data disk.
- Data disk at start: 250 GiB total, 183 GiB used, 68 GiB available.
- System disk at start: 30 GiB total, 18 GiB available.
- GPU at start: 0 MiB allocated, 0% utilization.
- No training, evaluation, upload, MuJoCo, or FFmpeg process was active.

Status: `PASS`. Experiment execution is authorized; upload remains contingent on
the final staging scans.
