#!/usr/bin/env python3
"""Render final GitHub/HF release documentation from verified artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GITHUB_ROOT = ROOT / "release/github_staging/JEPE_OOD_explore"
HF_ROOT = ROOT / "release/hf_staging/JEPE_OOD_explore"
GITHUB_URL = "https://github.com/lwbscu/JEPE_OOD_explore"
HF_URL = "https://huggingface.co/datasets/scilwb/JEPE_OOD_explore"
CONDITIONS = ("red", "blue_v2", "yellow_v2")


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric(path: Path) -> float:
    metrics = _json(path).get("metrics", {})
    value = float(metrics["success_rate"])
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"invalid success rate in {path}: {value}")
    return value


def _rates(root: Path) -> list[float]:
    return [_metric(root / condition / "results.json") for condition in CONDITIONS]


def _row(label: str, rates: list[float]) -> str:
    macro = sum(rates) / len(rates)
    return (
        f"| {label} | {rates[0]:.0f}% | {rates[1]:.0f}% | "
        f"{rates[2]:.0f}% | {macro:.2f}% |"
    )


def _probe_rows() -> tuple[list[str], dict[str, Any]]:
    summary = _json(ROOT / "outputs/eval/cube/probe_goal_cost/summary.json")
    rows = []
    labels = {
        "in_box": "In-box",
        "plus_05cm": "+5 cm",
        "fallback_max": "Fallback support (median 5.57 cm)",
    }
    for tier, label in labels.items():
        arms = summary["tiers"][tier]
        latent = float(arms["latent"]["success_rate"])
        probe = float(arms["probe"]["success_rate"])
        rows.append(f"| {label} | {latent:.0f}% | {probe:.0f}% | {probe-latent:+.0f} pp |")
    return rows, summary


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_rows() -> list[str]:
    rows = []
    weight_root = HF_ROOT / "weights"
    for path in sorted(weight_root.rglob("*.pt")):
        rel = path.relative_to(HF_ROOT).as_posix()
        rows.append(
            f"| `{rel}` | {path.stat().st_size / (1024**2):.1f} MiB | "
            f"`{_sha(path)[:16]}...` |"
        )
    return rows


def _license_text() -> str:
    lewm = (ROOT / "le-wm/LICENSE").read_text(encoding="utf-8").rstrip()
    stable = """MIT License

Copyright (c) stable-worldmodel authors: Lucas Maes, Quentin Le Lidec, and Randall Balestriero

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
    return (
        "JEPE_OOD_explore combines derived experiment code with MIT-licensed "
        "LeWM and stable-worldmodel interfaces.\n\n"
        "LeWM license (verbatim from the local upstream checkout):\n\n"
        f"{lewm}\n\n"
        "stable-worldmodel license notice (the upstream 0.1.1 pyproject "
        "declares MIT; its checkout does not include a standalone LICENSE):\n\n"
        f"{stable}\n"
    )


def _notice_text() -> str:
    return f"""# Third-Party and Derivative Work Notice

This release contains experiment code derived from LeWM and calls the
stable-worldmodel API. The modifications add Cube OOD evaluation, targeted
visual augmentation, trust-region CEM seeding, off-policy experiments,
long-horizon supervisors, no-augmentation controls, probe-goal planning,
geometric waypoint evaluation, and official Play-data fine-tuning.

- LeWM: https://github.com/lucas-maes/le-wm (MIT; local license reproduced in LICENSE)
- stable-worldmodel 0.1.1: https://github.com/galilai-group/stable-worldmodel
  (MIT as declared by upstream package metadata; authors Lucas Maes,
  Quentin Le Lidec, Randall Balestriero)
- OGBench: https://github.com/seohongpark/ogbench (MIT, 2024 OGBench Authors)
- Original Cube model: https://huggingface.co/quentinll/lewm-cube (MIT)
- Original Cube data: https://huggingface.co/datasets/quentinll/lewm-cube (MIT)

The off-policy datasets are newly simulated derivative trajectories whose
initial states originate from the OGBench Cube dataset. The original 101.9 GB
Cube HDF5, PushT source data, and original Quentinll checkpoint are not
redistributed. See {HF_URL} for the released derivative artifacts.
"""


def render() -> None:
    report = ROOT / "outputs/eval/cube/CONTROL_AND_PROBEGOAL_REPORT.md"
    if not report.is_file():
        raise FileNotFoundError(report)
    for required in (
        ROOT / "outputs/eval/cube/PLANNING_PROBLEM_ANALYSIS.md",
        ROOT / "outputs/eval/cube/FINAL_DOCUMENT_VALIDATION.md",
        ROOT / "outputs/eval/cube/PLAY_LINE_VERDICT.md",
        ROOT / "outputs/eval/cube/waypoint_probe/WAYPOINT_REPORT.md",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    masked = _rates(ROOT / "outputs/eval/cube/trust_region/T2")
    robust = _rates(ROOT / "outputs/eval/cube/robust_v1")
    control_12732 = _rates(ROOT / "outputs/eval/cube/control_noaugment/step_12732")
    control_16732 = _rates(ROOT / "outputs/eval/cube/control_noaugment/step_16732")
    probe_rows, _ = _probe_rows()

    result_rows = "\n".join(
        (
            _row("MaskedAug + T2", masked),
            _row("Robust v1 + T2", robust),
            _row("No augmentation, 12,732 steps", control_12732),
            _row("No augmentation, 16,732 cumulative steps", control_16732),
        )
    )
    github_readme = f"""# JEPE OOD Explore

Reproducible code, reports, and evidence for a controlled study of OOD behavior
in a Cube JEPA world-model planner. Large artifacts are hosted in the
[Hugging Face dataset repository]({HF_URL}).

## Project in one page

**Research question.** Which part of a visual world-model planner limits
closed-loop OOD control: goal construction, visual perception, candidate
generation, ranking, slow-loop intervention, or learned dynamics?

**Best deployable result.** Robust v1 + T2 reaches **92/92/86%** on paired
Red/Blue-v2/Yellow-v2 evaluation, macro **90.00%**. The 94% Red probe result is
a privileged-coordinate diagnostic, not a three-color deployment result.

| Line | Main intervention | Quantitative result | Final status |
|---|---|---|---|
| Goal and perception | Real frame + controlled recolor + EE continuity; MaskedAug then Robust v1 | 66.00% baseline to 90.00% macro | Improved; floor, light, and camera remain sensitive |
| Candidate generation | Memory Seed and T2 | 74.00% then 87.33% macro | Effective planning-side intervention |
| Ranking and navigation | top-1, blind LLM, probe XYZ cost, waypoint chain | Probe 32/18/12% to 50/58/52%, but 4 cm chain falls to 16/12/18% | Ranking-only and waypoint remedies rejected |
| Slow loop | B1/B2 rule and LLM intervention | 72% baseline versus 70% rule and 70% LLM | Archived |
| Dynamics training | Off-policy V1/V2/V3 and official Play v1 | Expert depth-5 5.16 mm; planner candidates 85.24/118.04/124.55 mm | Four-round line archived |

The final evidence supports a structural mismatch between the JEPA one-step
training objective and the multi-step prediction required by planning. See the
[final planning analysis](docs/eval/cube/PLANNING_PROBLEM_ANALYSIS.md),
[Play verdict](docs/eval/cube/PLAY_LINE_VERDICT.md), and
[waypoint report](docs/eval/cube/waypoint_probe/WAYPOINT_REPORT.md).

## Main three-color matrix

| Model / training arm | Red | Blue v2 | Yellow v2 | Macro |
|---|---:|---:|---:|---:|
{result_rows}

### Intermediate probe-goal diagnostic

| Target tier | Robust latent cost | Probe XYZ cost | Delta |
|---|---:|---:|---:|
{chr(10).join(probe_rows)}

The probe changes the cost interface and improves each paired tier, but does
not solve navigation: a 4 cm waypoint chain scores 16/12/18% versus direct
probe 50/58/52%. Full paired flips and provenance are in
[`CONTROL_AND_PROBEGOAL_REPORT.md`](docs/eval/cube/CONTROL_AND_PROBEGOAL_REPORT.md)
and [`WAYPOINT_REPORT.md`](docs/eval/cube/waypoint_probe/WAYPOINT_REPORT.md).

## Repository map

- `code/`: LeWM experiment and tool scripts, preserving their original layout.
- `docs/`: experiment reports, verdicts, analyses, and validation records.
- `evidence/`: 100 retained MP4s plus comparison/contact-sheet images.
- `LICENSE` and `NOTICE`: upstream MIT terms and derivative-work attribution.

## Reproduction entry points

```bash
# Zero-augmentation control: 12,732 steps plus a fresh 4,000-step continuation
python code/train_cube_control_noaugment.py --run-id control_noaugment_seed3072 --phase all --num-workers 6

# T2 paired control evaluation (run once for each released control checkpoint)
python code/eval_control_noaugment.py --checkpoint <control.pt> --condition all --num-eval 50 --authorize-formal

# Robust-specific embedding dataset and strict XYZ probe
python code/tools/build_cube_probe_dataset.py --checkpoint <robust.pt> --output <embedding_dir> --max-frames 400000 --sampling-mode episode_blocks
python code/tools/train_cube_xyz_probe.py --dataset <embedding_dir> --device cuda

# Paired latent/probe goal-cost evaluation
python code/eval_probe_goal_ood.py --checkpoint <robust.pt> --probe <probe.pt> --probe-dataset-metadata <embedding_dir>/metadata.json --tier all --mode both --num-eval 50 --authorize-formal

# Official Play conversion, mixed one-step training, and fail-stop offline gate
python code/tools/prepare_cube_play_v1.py --help
python code/train_cube_play_v1.py --help
python code/tools/evaluate_cube_play_v1.py --help
```

The original runs used absolute `/root/autodl-tmp/ailab/...` paths. Historical
reports preserve those paths as provenance; map them to your checkout and data
root when reproducing. The 20,566 previously validated bulk videos were deleted
for disk recovery and can be regenerated with the corresponding evaluators.

## Data and limitations

- Fixed evaluation episodes are excluded from training, probes, memory lookup,
  and released off-policy generation where each protocol requires it.
- Real Cube frames support only about 7.02 cm outside the nominal target box.
  Requested +10 cm and +20 cm tiers therefore share a documented fallback
  support point with median distance about 5.57 cm; they are not claimed as
  true +10/+20 cm measurements.
- The original Cube HDF5, official Play source, PushT data, and Quentinll base
  checkpoint are not redistributed. Follow the source links in `NOTICE`.
- Play v1 passed expert-retention checks but failed all candidate-pool gates, so
  no online Play evaluation was authorized.
- Negative off-policy and long-horizon results are retained for scientific
  completeness.

## Future work

The next controlled direction is an explicitly multi-step training target that
matches the planner's joint off-policy action sequences while retaining the
expert manifold. Any slow loop should validate goal continuity and model
uncertainty before action search, instead of attempting repeated post-hoc
recovery after an unreliable plan is active.

## Links

- Code and reports: {GITHUB_URL}
- Weights, derivative data, memory index, reports, and evidence: {HF_URL}
"""

    hf_readme = f"""---
license: mit
pretty_name: JEPE OOD Explore
task_categories:
- reinforcement-learning
tags:
- robotics
- world-models
- out-of-distribution
- model-based-planning
---

# JEPE OOD Explore artifact bundle

This dataset repository accompanies [{GITHUB_URL}]({GITHUB_URL}). It contains
portable experiment weights, two derivative off-policy datasets, the frozen
Cube expert memory index, final reports, and curated visual evidence.

## Project summary

Robust v1 + T2 is the best deployable three-color configuration at
**92/92/86%** (macro **90.00%**). Probe XYZ cost raises paired target-space
success to 50/58/52%, but waypoint decomposition falls to 16/12/18%. Official
Play training preserves expert behavior (`-16.34%` stopline; 5.16 mm expert
depth-5 error) while leaving planner-candidate error at
85.24/118.04/124.55 mm, so the off-policy training line is a completed negative
result. The final diagnosis is in `reports/PLANNING_PROBLEM_ANALYSIS.md`.

## Contents

- `weights/`: ColorAug, MaskedAug, Robust v1, two no-augmentation control
  checkpoints, off-policy v1 and Play v1 negative-result checkpoints, and the
  robust XYZ probe. Play v1 is retained for reproduction and is not recommended
  for deployment.
- `datasets/offpolicy_cube_v1/`: synthetic-noise off-policy rollouts.
- `datasets/offpolicy_cube_v2/`: planner-in-the-loop off-policy rollouts.
- `memory_index/cube_expert_v1/`: expert retrieval index with fixed-50 exclusion metadata.
- `reports/`: final planning analysis, validation record, and terminal reports.
- `evidence/`: the same 100 MP4s and comparison/contact-sheet images as GitHub.

## Released weights

| Path | Size | SHA-256 prefix |
|---|---:|---|
{chr(10).join(_weight_rows())}

## Main paired results

| Model / training arm | Red | Blue v2 | Yellow v2 | Macro |
|---|---:|---:|---:|---:|
{result_rows}

| Target tier | Robust latent cost | Probe XYZ cost | Delta |
|---|---:|---:|---:|
{chr(10).join(probe_rows)}

The probe table is an intermediate diagnostic, not the final navigation result:
the paired 4 cm waypoint chain scores 16/12/18%. See the GitHub reports for
exact 50-environment vectors, flip tables, loss curves, model/probe provenance,
and the complete causal interpretation.

## Provenance and exclusions

The derivative trajectories use OGBench Cube initial states but were newly
simulated for this project. Every released training/index artifact records the
fixed-50 evaluation exclusion. The original 101.9 GB Cube HDF5, PushT source
data, and original Quentinll checkpoint are intentionally not included:

- https://huggingface.co/datasets/quentinll/lewm-cube
- https://huggingface.co/quentinll/lewm-cube
- https://github.com/seohongpark/ogbench

Requested +10/+20 cm target tiers exceeded the real dataset support and are
reported only at their common documented fallback (median about 5.57 cm).
Absolute paths inside manifests/reports are immutable provenance from the
originating machine, not required installation paths.

The official Play source dataset and its 11 GiB local conversion workspace are
not redistributed. Only the portable 68.9 MiB final negative-result weight,
configuration, code, and audit report are included.

## Citation

```bibtex
@misc{{jepe_ood_explore_2026,
  title={{JEPE OOD Explore: Controlled OOD Studies for a Cube World Model}},
  author={{LWB SCU}},
  year={{2026}},
  howpublished={{{GITHUB_URL}}}
}}
```
"""

    for root, readme in ((GITHUB_ROOT, github_readme), (HF_ROOT, hf_readme)):
        if not root.is_dir():
            raise FileNotFoundError(root)
        (root / "README.md").write_text(readme, encoding="utf-8")
        (root / "LICENSE").write_text(_license_text(), encoding="utf-8")
        (root / "NOTICE").write_text(_notice_text(), encoding="utf-8")
    (GITHUB_ROOT / ".gitignore").write_text(
        "__pycache__/\n"
        "*.py[cod]\n"
        "*.ckpt\n"
        "*.pt\n"
        "*.h5\n"
        "*.npz\n"
        "*.npy\n"
        "*.safetensors\n"
        "*.tmp\n"
        ".env*\n",
        encoding="utf-8",
    )
    print(GITHUB_ROOT)
    print(HF_ROOT)


if __name__ == "__main__":
    render()
