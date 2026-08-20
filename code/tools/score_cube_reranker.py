#!/usr/bin/env python3
"""Score frozen single-PICK Cube reranker responses against physical audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


AILAB = Path(__file__).resolve().parents[2]
PILOT = AILAB / "outputs/rerank_pilot"
DEFAULT_PROMPTS = PILOT / "prompts"
DEFAULT_RESPONSES = PILOT / "responses"
REPORT_MD = PILOT / "RERANK_PILOT_REPORT.md"
RESULTS_CSV = PILOT / "rerank_36_results.csv"
RESULTS_JSON = PILOT / "rerank_36_results.json"
AUDIT = AILAB / "outputs/audit"
AUDIT_ROOTS = {
    "red": AUDIT / "cube_cem_300",
    "blue_v2": AUDIT / "cube_cem_300_blue_v2",
    "yellow_v2": AUDIT / "cube_cem_300_yellow_v2",
}
# Frozen 12-case top-30 baselines: sum of successful candidates, cost-top1,
# and best-of-30 oracle, each as (ever, final).
EXPECTED = {
    "red": {"pool": (151, 144), "cost_top1": (6, 5), "oracle": (7, 7)},
    "blue_v2": {"pool": (148, 134), "cost_top1": (5, 5), "oracle": (8, 8)},
    "yellow_v2": {"pool": (143, 122), "cost_top1": (5, 3), "oracle": (7, 7)},
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _load_cases(prompts_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    public = _json(prompts_root / "public/manifest.json")
    private = _json(prompts_root / "private/mapping.json")
    by_id = {entry["prompt_id"]: entry for entry in public["prompts"]}
    cases = []
    for secret in private["prompts"]:
        prompt_id = secret["prompt_id"]
        shown = [int(value) for value in by_id[prompt_id]["displayed_candidate_ids"]]
        root = AUDIT_ROOTS[secret["protocol"]]
        case_dir = root / f"env_{int(secret['env_idx']):02d}_row_{int(secret['row'])}"
        population = case_dir / "population.npz"
        if _sha(population) != secret["population_sha256"]:
            raise RuntimeError(f"frozen population changed: {prompt_id}")
        outcomes_path = case_dir / "candidate_outcomes.csv"
        with outcomes_path.open(newline="", encoding="utf-8") as handle:
            rows = {int(row["candidate_idx"]): row for row in csv.DictReader(handle)}
        if len(shown) != 30 or len(set(shown)) != 30 or any(value not in rows for value in shown):
            raise RuntimeError(f"invalid displayed candidate set: {prompt_id}")
        selected = [rows[value] for value in shown]
        cases.append({**secret, "shown": shown, "outcomes": rows, "selected": selected,
                      "candidate_outcomes_sha256": _sha(outcomes_path)})
    if len(cases) != 36:
        raise RuntimeError(f"expected 36 cases, got {len(cases)}")
    return public, cases


def _baselines(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        protocol = case["protocol"]
        selected = case["selected"]
        totals[protocol]["prompts"] += 1
        totals[protocol]["pool_ever"] += sum(_bool(row["ever_success"]) for row in selected)
        totals[protocol]["pool_final"] += sum(_bool(row["final_success"]) for row in selected)
        top1 = min(selected, key=lambda row: float(row["latent_cost"]))
        totals[protocol]["top1_ever"] += _bool(top1["ever_success"])
        totals[protocol]["top1_final"] += _bool(top1["final_success"])
        totals[protocol]["oracle_ever"] += any(_bool(row["ever_success"]) for row in selected)
        totals[protocol]["oracle_final"] += any(_bool(row["final_success"]) for row in selected)
    result = {key: dict(value) for key, value in totals.items()}
    for protocol, expected in EXPECTED.items():
        got = result[protocol]
        actual = {
            "pool": (got["pool_ever"], got["pool_final"]),
            "cost_top1": (got["top1_ever"], got["top1_final"]),
            "oracle": (got["oracle_ever"], got["oracle_final"]),
        }
        if got["prompts"] != 12 or actual != expected:
            raise RuntimeError(f"frozen baseline mismatch for {protocol}: expected={expected}, actual={actual}")
        got["uniform_random_expected_ever"] = got["pool_ever"] / 30.0
        got["uniform_random_expected_final"] = got["pool_final"] / 30.0
    return result


def _score_provider(provider: str, run: Path, public: dict[str, Any], cases: list[dict[str, Any]],
                    public_manifest_sha256: str) -> dict[str, Any]:
    provider_dir = run / provider
    complete = _json(provider_dir / "COMPLETE.json")
    if complete.get("num_responses") != 36:
        raise RuntimeError(f"incomplete frozen response set: {provider}")
    if complete.get("public_manifest_sha256") != public_manifest_sha256:
        raise RuntimeError(f"response set was produced from a different public manifest: {provider}")
    case_by_id = {case["prompt_id"]: case for case in cases}
    per_protocol: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failures = []
    picks: dict[str, dict[str, Any]] = {}
    for entry in public["prompts"]:
        prompt_id = entry["prompt_id"]
        path = provider_dir / f"{prompt_id}.json"
        if complete["response_sha256"].get(prompt_id) != _sha(path):
            raise RuntimeError(f"response hash mismatch: {provider}/{prompt_id}")
        response = _json(path)
        if response.get("prompt_sha256") != entry["prompt_sha256"]:
            raise RuntimeError(f"prompt hash mismatch in response: {provider}/{prompt_id}")
        case = case_by_id[prompt_id]
        protocol = case["protocol"]
        per_protocol[protocol]["prompts"] += 1
        if response.get("status") != "ok":
            failures.append({"prompt_id": prompt_id, "status": response.get("status")})
            picks[prompt_id] = {"status": response.get("status"), "candidate_idx": None,
                                "actual_model": response.get("actual_model")}
            continue
        candidate_idx = int(response["pick_id"])
        if candidate_idx not in case["shown"]:
            raise RuntimeError(f"undisplayed PICK: {provider}/{prompt_id}/{candidate_idx}")
        outcome = case["outcomes"][candidate_idx]
        ever = _bool(outcome["ever_success"])
        final = _bool(outcome["final_success"])
        per_protocol[protocol]["valid_picks"] += 1
        per_protocol[protocol]["ever_success"] += ever
        per_protocol[protocol]["final_success"] += final
        picks[prompt_id] = {
            "status": "ok", "actual_model": response.get("actual_model"),
            "candidate_idx": candidate_idx, "ever_success": ever, "final_success": final,
            "min_goal_distance_m": float(outcome["min_goal_distance_m"]),
            "final_goal_distance_m": float(outcome["final_goal_distance_m"]),
            "physical_rank": int(outcome["physical_rank"]),
            "endpoint_rank": int(outcome["endpoint_rank"]),
        }
    return {
        "provider": provider,
        "single_pick": {key: dict(value) for key, value in per_protocol.items()},
        "failures": failures,
        "picks": picks,
        "success_at_3": "not_available",
        "success_at_5": "not_available",
        "mrr": "not_available",
        "unavailable_reason": "The frozen protocol requests exactly one PICK per prompt.",
    }


def main(args: argparse.Namespace) -> int:
    public, cases = _load_cases(args.prompts.expanduser().resolve())
    baselines = _baselines(cases)
    if args.validate_baselines_only:
        print(json.dumps(baselines, indent=2, sort_keys=True))
        return 0
    run = (args.responses_root / args.run_id).expanduser().resolve()
    providers = ["openai", "deepseek"] if args.provider == "both" else [args.provider]
    public_manifest_sha256 = _sha(args.prompts.expanduser().resolve() / "public/manifest.json")
    scores = [
        _score_provider(provider, run, public, cases, public_manifest_sha256)
        for provider in providers
    ]
    by_provider = {score["provider"]: score for score in scores}
    rows = []
    for case in cases:
        selected = case["selected"]
        top1 = min(selected, key=lambda row: float(row["latent_cost"]))
        row = {
            "prompt_id": case["prompt_id"], "color": case["protocol"],
            "cost_top1_candidate_idx": int(top1["candidate_idx"]),
            "cost_top1_ever_success": _bool(top1["ever_success"]),
            "cost_top1_final_success": _bool(top1["final_success"]),
            "random_ever_successes_of_30": sum(_bool(value["ever_success"]) for value in selected),
            "random_final_successes_of_30": sum(_bool(value["final_success"]) for value in selected),
            "random_expected_ever": sum(_bool(value["ever_success"]) for value in selected) / 30.0,
            "random_expected_final": sum(_bool(value["final_success"]) for value in selected) / 30.0,
            "oracle_ever_success": any(_bool(value["ever_success"]) for value in selected),
            "oracle_final_success": any(_bool(value["final_success"]) for value in selected),
        }
        for provider in ("openai", "deepseek"):
            pick = by_provider.get(provider, {}).get("picks", {}).get(case["prompt_id"], {"status": "not_run"})
            for key, value in pick.items():
                row[f"{provider}_{key}"] = value
        rows.append(row)
    report = {
        "protocol": "single PICK from shuffled top-30",
        "run_id": args.run_id,
        "baselines": baselines,
        "providers": scores,
        "candidate_outcomes_sha256": {
            case["prompt_id"]: case["candidate_outcomes_sha256"] for case in cases
        },
        "rows": rows,
        "success_at_3": "not_available",
        "success_at_5": "not_available",
        "mrr": "not_available",
    }
    for path in (REPORT_MD, RESULTS_CSV, RESULTS_JSON):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"result exists: {path}; pass --overwrite")
    RESULTS_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Cube blind reranker score", "", f"Run: `{args.run_id}`", "",
             "Success@3, Success@5, and MRR: **not available** (one frozen PICK per prompt).", ""]
    lines.extend(["| Color | Metric | GPT-5.5 | DeepSeek | Cost top-1 | Random expectation | Oracle | GPT-5.5 - top1 | DeepSeek - top1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for protocol in ("red", "blue_v2", "yellow_v2"):
        base = baselines[protocol]
        for metric in ("ever", "final"):
            top1 = base[f"top1_{metric}"]
            random_expected = base[f"uniform_random_expected_{metric}"]
            oracle = base[f"oracle_{metric}"]
            values = {}
            for provider in ("openai", "deepseek"):
                stats = by_provider.get(provider, {}).get("single_pick", {}).get(protocol, {})
                values[provider] = stats.get(f"{metric}_success")
            def show(value: Any) -> str:
                return "protocol failure" if value is None else f"{value}/12"
            codex_delta = "—" if values["openai"] is None else f"{values['openai'] - top1:+d}"
            deepseek_delta = "—" if values["deepseek"] is None else f"{values['deepseek'] - top1:+d}"
            lines.append(
                f"| {protocol} | {metric} | {show(values['openai'])} | {show(values['deepseek'])} | "
                f"{top1}/12 | {random_expected:.3f}/12 | {oracle}/12 | "
                f"{codex_delta} | {deepseek_delta} |"
            )
    lines.extend([
        "",
        "## 36 个盲化单元",
        "",
        "`E/F` 分别表示 25 步内曾成功/第 25 步最终成功；`PF` 表示严格输出协议失败，按未选中计入 12 个单元的分母。",
        "",
        "| Prompt | Color | GPT-5.5 PICK | GPT E/F | DeepSeek PICK | DeepSeek E/F | Cost top-1 E/F | Random E/F | Oracle E/F |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    def pick_cell(row: dict[str, Any], provider: str) -> tuple[str, str]:
        if row.get(f"{provider}_status") != "ok":
            return "PF", "0/0"
        return (
            str(row[f"{provider}_candidate_idx"]),
            f"{int(bool(row[f'{provider}_ever_success']))}/{int(bool(row[f'{provider}_final_success']))}",
        )

    for row in rows:
        openai_pick, openai_ef = pick_cell(row, "openai")
        deepseek_pick, deepseek_ef = pick_cell(row, "deepseek")
        top1_ef = f"{int(bool(row['cost_top1_ever_success']))}/{int(bool(row['cost_top1_final_success']))}"
        random_ef = f"{row['random_expected_ever']:.3f}/{row['random_expected_final']:.3f}"
        oracle_ef = f"{int(bool(row['oracle_ever_success']))}/{int(bool(row['oracle_final_success']))}"
        lines.append(
            f"| {row['prompt_id']} | {row['color']} | {openai_pick} | {openai_ef} | "
            f"{deepseek_pick} | {deepseek_ef} | {top1_ef} | {random_ef} | {oracle_ef} |"
        )

    lines.extend([
        "",
        "## 结论",
        "",
        "- 全 36 个单元汇总：GPT-5.5 ever/final 为 17/15，DeepSeek 为 18/13，cost top-1 为 16/13，Top-30 均匀随机期望为 14.733/13.333，hindsight oracle 为 22/22。",
        "- GPT-5.5 的 ever-success 为 Red 7/12、Blue-v2 5/12、Yellow-v2 5/12；相对 cost top-1 分别为 +1、0、0。它在 Red 达到 Top-30 oracle，但没有显示 OOD 颜色越重、相对优势越大的趋势。",
        "- DeepSeek 的 ever-success 为 6/12、6/12、6/12；相对 cost top-1 分别为 0、+1、+1。这个 OOD 增益只有每色 1 个样本，且 Blue-v2 有 2 个、Yellow-v2 有 1 个协议失败，证据不足以认定为稳定优势。",
        "- 最终成功并未随 ever-success 一致改善：GPT-5.5 为 7/12、4/12、4/12，DeepSeek 为 6/12、3/12、4/12。说明文本 LLM 有时能找到短暂进入目标阈值的轨迹，但对保持稳定性的判断仍弱。",
        "- 本实验是固定 12 个首周期、25 步 open-loop 候选的离线机制试点；同一 12 个 env 跨色重复，不能把 36 个单元当独立总体样本，也不能直接等价为 50-env 闭环提升。",
        "",
        "DeepSeek 协议失败：`cube_015`、`cube_016`、`cube_027`；均保留为正式失败结果，没有选择性重问。",
    ])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote frozen score report: {REPORT_MD}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--responses-root", type=Path, default=DEFAULT_RESPONSES)
    p.add_argument("--run-id")
    p.add_argument("--provider", choices=("openai", "deepseek", "both"), default="both")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate-baselines-only", action="store_true")
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    if not parsed.validate_baselines_only and not parsed.run_id:
        parser().error("--run-id is required unless --validate-baselines-only is used")
    raise SystemExit(main(parsed))
