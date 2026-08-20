#!/usr/bin/env python3
"""Build leakage-resistant Cube top-30 reranking prompts.

This builder deliberately reads only the frozen HDF5, each audit manifest, and
each case's population.npz.  Physical outcomes are joined only by the scorer,
after model responses have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import hdf5plugin  # noqa: F401  # Register compression filters before h5py.
import h5py
import numpy as np


AILAB = Path(__file__).resolve().parents[2]
DATASET = AILAB / "datasets/ogbench/cube_single_expert.h5"
AUDIT = AILAB / "outputs/audit"
DEFAULT_OUTPUT = AILAB / "outputs/rerank_pilot/prompts"
PROTOCOLS = {
    "red": AUDIT / "cube_cem_300",
    "blue_v2": AUDIT / "cube_cem_300_blue_v2",
    "yellow_v2": AUDIT / "cube_cem_300_yellow_v2",
}
ACTION_SCALE = np.asarray([0.05, 0.05, 0.05, 0.3, 1.0], dtype=np.float64)
TOP_K = 30
FORMAT_VERSION = "cube-rerank-prompt-v1"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def _prepare_output(path: Path, overwrite: bool) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"refusing symlink output: {raw}")
    path = raw.resolve()
    allowed = (AILAB / "outputs/rerank_pilot").resolve()
    if path == allowed or allowed not in path.parents:
        raise ValueError(f"output must be a concrete child of {allowed}: {path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"nonempty output: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _f(value: np.ndarray | float) -> str:
    array = np.asarray(value).reshape(-1)
    return "[" + ",".join(f"{float(x):+.5f}" for x in array) + "]"


def _action_matrix(actions: np.ndarray) -> str:
    """Encode all 25 physical actions compactly, without dropping dimensions."""
    physical = np.asarray(actions, dtype=np.float64).copy()
    if physical.shape != (25, 5):
        raise RuntimeError(f"expected a 25x5 physical action matrix, got {physical.shape}")
    physical[:, :3] *= 100.0  # metres -> centimetres
    physical[:, 3] = np.degrees(physical[:, 3])
    rows = ["[" + ",".join(f"{float(value):.5f}" for value in row) + "]" for row in physical]
    return "[" + ",".join(rows) + "]"


def _prompt(color: str, state: dict[str, np.ndarray], candidates: list[dict[str, Any]]) -> str:
    lines = [
        "Blindly select one robot motion sequence for one cube.",
        "Criterion: pick one sequence most likely to put cube center within 0.04 m of target at any time in 25 steps; cube yaw is context only.",
        f"color={color}",
        f"current: cube_pos_m={_f(state['block_pos'])};cube_yaw_rad={_f(state['block_yaw'])};"
        f"ee_pos_m={_f(state['ee_pos'])};ee_yaw_rad={_f(state['ee_yaw'])};"
        f"gripper_opening={_f(state['gripper_opening'])};gripper_contact={_f(state['gripper_contact'])}",
        f"target: cube_pos_m={_f(state['goal_pos'])};cube_yaw_rad={_f(state['goal_yaw'])}",
        "action columns=[dx_cm,dy_cm,dz_cm,dyaw_deg,dgripper]; each ID has exactly 25 rows.",
        "yaw: positive=counterclockwise, negative=clockwise. dgripper: >+0.05=open, <-0.05=close, otherwise=hold.",
        "Do not average/modify sequences. Reply exactly: PICK: <integer candidate ID>",
        "shuffled candidates; format ID:[[step1 five values],...,[step25 five values]]",
    ]
    for candidate in candidates:
        lines.append(f"{candidate['candidate_idx']}:{_action_matrix(candidate['physical_actions'])}")
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> int:
    manifests = {name: _json(root / "manifest.json") for name, root in PROTOCOLS.items()}
    reference_rows = manifests["red"]["formal_rows"]
    dataset_stat = args.dataset.stat()
    for name, manifest in manifests.items():
        if not manifest.get("capture_complete") or manifest["formal_rows"] != reference_rows:
            raise RuntimeError(f"unfrozen/inconsistent manifest: {name}")
        if len(manifest["audit_cases"]) != 12:
            raise RuntimeError(f"expected 12 audit cases: {name}")
        identity = manifest.get("dataset", {})
        if (identity.get("size_bytes") != dataset_stat.st_size
                or identity.get("mtime_ns") != dataset_stat.st_mtime_ns):
            raise RuntimeError(f"dataset identity differs from frozen manifest: {name}")

    public_entries: list[dict[str, Any]] = []
    private_entries: list[dict[str, Any]] = []
    rendered_prompts: dict[str, str] = {}
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        prompt_number = 0
        for protocol, root in PROTOCOLS.items():
            color = {"red": "red", "blue_v2": "blue", "yellow_v2": "yellow"}[protocol]
            for case in manifests[protocol]["audit_cases"]:
                prompt_number += 1
                env_idx = int(case["env_idx"])
                row = int(case["dataset_row"])
                goal_row = int(case["goal_row"])
                case_dir = root / f"env_{env_idx:02d}_row_{row}"
                population_path = case_dir / "population.npz"
                with np.load(population_path, allow_pickle=False) as data:
                    population = np.asarray(data["candidates_normalized"], dtype=np.float64)
                    top_indices = np.asarray(data["topk_indices"], dtype=np.int64)
                    latent_costs = np.asarray(data["latent_costs"], dtype=np.float64)
                    scaler_mean = np.asarray(data["action_scaler_mean"], dtype=np.float64)
                    scaler_scale = np.asarray(data["action_scaler_scale"], dtype=np.float64)
                if (population.shape != (300, 5, 25) or top_indices.shape != (TOP_K,)
                        or latent_costs.shape != (300,) or scaler_mean.shape != (5,)
                        or scaler_scale.shape != (5,)):
                    raise RuntimeError(f"bad population shape: {population_path}")
                if (len(np.unique(top_indices)) != TOP_K or np.any(top_indices < 0)
                        or np.any(top_indices >= 300)):
                    raise RuntimeError(f"invalid top-30 indices: {population_path}")
                if not all(np.all(np.isfinite(value)) for value in
                           (population, latent_costs, scaler_mean, scaler_scale)):
                    raise RuntimeError(f"non-finite population data: {population_path}")
                expected_top = np.argsort(latent_costs, kind="stable")[:TOP_K]
                if set(top_indices.tolist()) != set(expected_top.tolist()):
                    raise RuntimeError(f"top-30 indices do not match latent costs: {population_path}")

                shuffle_material = f"{FORMAT_VERSION}|{protocol}|{env_idx}|" + ",".join(map(str, sorted(top_indices.tolist())))
                shuffle_sha = hashlib.sha256(shuffle_material.encode()).hexdigest()
                shuffled = top_indices.tolist()
                random.Random(int(shuffle_sha, 16)).shuffle(shuffled)
                prompt_id = f"cube_{prompt_number:03d}"
                prompt_candidates = []
                for candidate_idx in shuffled:
                    normalized = population[candidate_idx].reshape(25, 5)
                    env_action = normalized * scaler_scale + scaler_mean
                    physical = env_action * ACTION_SCALE
                    prompt_candidates.append({"candidate_idx": int(candidate_idx), "physical_actions": physical})

                state = {
                    "block_pos": np.asarray(h5["privileged_block_0_pos"][row]),
                    "block_yaw": np.asarray(h5["privileged_block_0_yaw"][row]),
                    "ee_pos": np.asarray(h5["proprio_effector_pos"][row]),
                    "ee_yaw": np.asarray(h5["proprio_effector_yaw"][row]),
                    "gripper_opening": np.asarray(h5["proprio_gripper_opening"][row]),
                    "gripper_contact": np.asarray(h5["proprio_gripper_contact"][row]),
                    "goal_pos": np.asarray(h5["privileged_block_0_pos"][goal_row]),
                    "goal_yaw": np.asarray(h5["privileged_block_0_yaw"][goal_row]),
                }
                prompt_text = _prompt(color, state, prompt_candidates)
                forbidden = ("latent_cost", "latent rank", "success", "dataset_row", "env_idx", str(AILAB))
                if any(token.lower() in prompt_text.lower() for token in forbidden):
                    raise RuntimeError(f"prompt leakage token detected: {prompt_id}")
                prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                rendered_prompts[prompt_id] = prompt_text
                public_entries.append({
                    "prompt_id": prompt_id,
                    "color": color,
                    "candidate_count": TOP_K,
                    "actions_per_candidate": 25,
                    "action_dims": 5,
                    "action_text_encoding": "compact 25x5 arrays; columns dx_cm,dy_cm,dz_cm,dyaw_deg,dgripper; signed decimal with 5 places",
                    "prompt_file": f"prompts/{prompt_id}.txt",
                    "prompt_sha256": prompt_sha,
                    "shuffle_sha256": shuffle_sha,
                    "displayed_candidate_ids": [int(index) for index in shuffled],
                })
                private_entries.append({
                    "prompt_id": prompt_id,
                    "protocol": protocol,
                    "env_idx": env_idx,
                    "row": row,
                    "population_sha256": _sha(population_path),
                })

    if len(public_entries) != 36:
        raise RuntimeError(f"expected 36 prompts, built {len(public_entries)}")

    # Do not touch an existing output until every source and rendered prompt has
    # passed validation above.
    output = _prepare_output(args.output, args.overwrite)
    public_dir = output / "public/prompts"
    private_dir = output / "private"
    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True, mode=0o700)
    for prompt_id, prompt_text in rendered_prompts.items():
        (public_dir / f"{prompt_id}.txt").write_text(prompt_text, encoding="utf-8")

    public_manifest = {
        "format_version": FORMAT_VERSION,
        "blind_protocol": "single PICK from a shuffled 30-candidate subset",
        "num_prompts": len(public_entries),
        "dataset_identity": {"size_bytes": dataset_stat.st_size, "mtime_ns": dataset_stat.st_mtime_ns},
        "prompts": public_entries,
    }
    private_mapping = {"format_version": FORMAT_VERSION, "num_prompts": len(private_entries), "prompts": private_entries}
    _write_json(output / "public/manifest.json", public_manifest)
    _write_json(private_dir / "mapping.json", private_mapping, 0o600)
    print(f"built 36 blind prompts: {output / 'public'}")
    print(f"private mapping (do not send to models): {private_dir / 'mapping.json'}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(build(parser().parse_args()))
