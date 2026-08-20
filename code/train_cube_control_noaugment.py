#!/usr/bin/env python3
"""Two-phase no-augmentation control for the Cube LeWM experiments.

Phase A reproduces the 12,732-step MaskedAug optimization schedule while
removing the image intervention. Phase B reloads that exported model and
restarts AdamW plus a 4,000-step scheduler, matching robust_v1's continuation
budget. The original training entry points and pretrained weights are never
modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_cube_coloraug as base  # noqa: E402


AILAB = HERE.parent
DATASET = AILAB / "datasets/ogbench/cube_single_expert.h5"
MANIFEST = AILAB / "outputs/audit/cube_cem_manifest.json"
ORIGINAL_WEIGHTS = AILAB / "checkpoints/models--quentinll--lewm-cube/weights.pt"
OUTPUT_ROOT = AILAB / "outputs/train/control_noaugment"
CHECKPOINT_ROOT = AILAB / "checkpoints/lewm-cube-control_noaugment"
TENSORBOARD_ROOT = AILAB / "logs/tensorboard/control_noaugment"

PHASE_A_STEPS = 12_732
PHASE_B_STEPS = 4_000
FORMAL_CUMULATIVE_STEPS = PHASE_A_STEPS + PHASE_B_STEPS

_ORIGINAL_PLAN = base._plan
_CONTROL_RUN_ID = ""
_PHASE_NAME = ""
_EXPORT_FILENAME = ""
_CUMULATIVE_STEP_END = 0
_PHASE_EXPECTED_STEPS = 0


class IdentityNoAugmentation:
    """Signature-compatible image transform that is exactly the identity."""

    def __init__(self, source: str = "pixels", **_: Any) -> None:
        self.source = source

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        return sample


def _control_plan(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    plan = _ORIGINAL_PLAN(args, protocol)
    phase_run_id = plan["run_id"]
    plan.update(
        {
            "route": "control_noaugment_two_phase",
            "run_id": _CONTROL_RUN_ID,
            "phase": _PHASE_NAME,
            "phase_run_id": phase_run_id,
            "cumulative_step_end": _CUMULATIVE_STEP_END,
            "augmentation": {
                "enabled": False,
                "pixel_interventions": [],
                "train": "deterministic ImageNet conversion and resize only",
                "validation": "deterministic ImageNet conversion and resize only",
                "identity_transform_returns_input_sample": True,
            },
            "checkpoint_export": {
                "filename": _EXPORT_FILENAME,
                "directory": str((CHECKPOINT_ROOT / _CONTROL_RUN_ID).resolve()),
            },
        }
    )
    plan["training"]["optimizer_state_source"] = (
        "new AdamW state from Quentinll weights"
        if _PHASE_NAME == "phase_a"
        else "new AdamW state after reloading Phase A exported weights"
    )
    plan["schedule"]["comparison_alignment"] = (
        "MaskedAug 12732-step schedule"
        if _PHASE_NAME == "phase_a"
        else "robust_v1 4000-step continuation schedule"
    )
    return plan


class SaveControlWeights:
    """Export exactly one portable model artifact at each phase boundary."""

    @staticmethod
    def create(_phase_run_id: str, model_config: Any):
        from lightning.pytorch.callbacks import Callback
        from stable_worldmodel.wm.utils import save_pretrained

        class CallbackImpl(Callback):
            def on_train_end(self, trainer: Any, module: Any) -> None:
                if int(trainer.global_step) != _PHASE_EXPECTED_STEPS:
                    raise RuntimeError(
                        f"{_PHASE_NAME} ended at step {trainer.global_step}; "
                        f"expected {_PHASE_EXPECTED_STEPS}"
                    )
                if trainer.is_global_zero:
                    save_pretrained(
                        module.model,
                        run_name=f"lewm-cube-control_noaugment/{_CONTROL_RUN_ID}",
                        config=model_config,
                        filename=_EXPORT_FILENAME,
                        cache_dir=str(AILAB),
                    )

        return CallbackImpl()


def _phase_spec(control_run_id: str, phase: str, smoke: bool) -> dict[str, Any]:
    if phase == "phase_a":
        steps = 2 if smoke else PHASE_A_STEPS
        return {
            "name": phase,
            "steps": steps,
            "cumulative_step_end": steps,
            "warm_start": ORIGINAL_WEIGHTS,
            "export_filename": "weights_step_12732.pt" if not smoke else "weights_smoke_phase_a.pt",
        }
    if phase == "phase_b":
        steps = 2 if smoke else PHASE_B_STEPS
        phase_a_name = "weights_smoke_phase_a.pt" if smoke else "weights_step_12732.pt"
        return {
            "name": phase,
            "steps": steps,
            "cumulative_step_end": (2 if smoke else PHASE_A_STEPS) + steps,
            "warm_start": CHECKPOINT_ROOT / control_run_id / phase_a_name,
            "export_filename": "weights_final.pt" if not smoke else "weights_smoke_final.pt",
        }
    raise ValueError(f"unknown phase: {phase}")


def _configure_phase(control_run_id: str, spec: dict[str, Any]) -> None:
    global _CONTROL_RUN_ID, _PHASE_NAME, _EXPORT_FILENAME
    global _CUMULATIVE_STEP_END, _PHASE_EXPECTED_STEPS
    _CONTROL_RUN_ID = control_run_id
    _PHASE_NAME = str(spec["name"])
    _EXPORT_FILENAME = str(spec["export_filename"])
    _CUMULATIVE_STEP_END = int(spec["cumulative_step_end"])
    _PHASE_EXPECTED_STEPS = int(spec["steps"])

    base.OUTPUT_ROOT = OUTPUT_ROOT / control_run_id
    base.CHECKPOINT_ROOT = CHECKPOINT_ROOT / control_run_id / "training_state"
    base.TENSORBOARD_ROOT = TENSORBOARD_ROOT / control_run_id
    base.RandomHueRotation = IdentityNoAugmentation
    base.SaveRoute2Weights = SaveControlWeights
    base._plan = _control_plan


def _base_args(args: argparse.Namespace, spec: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=spec["name"],
        dataset=args.dataset,
        manifest=args.manifest,
        warm_start=spec["warm_start"],
        seed=args.seed,
        max_epochs=1,
        max_steps=spec["steps"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        train_split=args.train_split,
        # The identity transform accepts these compatibility arguments but
        # never reads them or consumes RNG.
        hue_probability=0.0,
        max_hue_delta=0.5,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sigreg_weight=args.sigreg_weight,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        resume=(
            args.resume
            and (
                CHECKPOINT_ROOT
                / args.run_id
                / "training_state"
                / spec["name"]
                / "lightning/last.ckpt"
            ).is_file()
        ),
        dry_run=False,
    )


def _phase_output(control_run_id: str, phase: str) -> Path:
    return OUTPUT_ROOT / control_run_id / phase


def _phase_complete(control_run_id: str, spec: dict[str, Any]) -> bool:
    completed_path = _phase_output(control_run_id, spec["name"]) / "completed.json"
    export_path = CHECKPOINT_ROOT / control_run_id / spec["export_filename"]
    if not completed_path.is_file() or not export_path.is_file():
        return False
    payload = json.loads(completed_path.read_text(encoding="utf-8"))
    return int(payload.get("global_step", -1)) == int(spec["steps"])


def _write_master_plan(args: argparse.Namespace, specs: list[dict[str, Any]]) -> None:
    payload = {
        "experiment": "control_noaugment",
        "run_id": args.run_id,
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
        "manifest": str(args.manifest.resolve()),
        "augmentation": "none",
        "formal_cumulative_steps": FORMAL_CUMULATIVE_STEPS,
        "phase_policy": (
            "Phase A matches MaskedAug's 12732-step optimizer schedule; Phase B reloads "
            "the Phase A weights with fresh AdamW and a fresh 4000-step robust_v1 schedule."
        ),
        "phases": [
            {
                "name": spec["name"],
                "steps": spec["steps"],
                "cumulative_step_end": spec["cumulative_step_end"],
                "warm_start": str(Path(spec["warm_start"]).resolve()),
                "export": str(
                    (CHECKPOINT_ROOT / args.run_id / spec["export_filename"]).resolve()
                ),
            }
            for spec in specs
        ],
    }
    base._write_json(OUTPUT_ROOT / args.run_id / "run_plan.json", payload)


def _patch_phase_completion(
    control_run_id: str, spec: dict[str, Any], wall_seconds: float
) -> None:
    path = _phase_output(control_run_id, spec["name"]) / "completed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    export_path = CHECKPOINT_ROOT / control_run_id / spec["export_filename"]
    if not export_path.is_file():
        raise FileNotFoundError(f"phase export missing: {export_path}")
    payload.update(
        {
            "phase": spec["name"],
            "phase_steps": spec["steps"],
            "cumulative_step_end": spec["cumulative_step_end"],
            "exported_weights": str(export_path.resolve()),
            "exported_weights_sha256": base._sha(export_path),
            "wall_seconds": wall_seconds,
            "augmentation": "none",
        }
    )
    # The inherited field points at the training-state directory because this
    # control deliberately exports both phase weights into one release folder.
    payload.pop("final_weights", None)
    base._write_json(path, payload)


def _run_phase(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    if _phase_complete(args.run_id, spec):
        print(f"{spec['name']} already complete; reusing verified phase export")
        return
    _configure_phase(args.run_id, spec)
    phase_args = _base_args(args, spec)
    started = time.monotonic()
    result = base.run(phase_args)
    if result != 0:
        raise RuntimeError(f"{spec['name']} returned status {result}")
    _patch_phase_completion(args.run_id, spec, time.monotonic() - started)


def _dry_run(args: argparse.Namespace, specs: list[dict[str, Any]]) -> int:
    protocol = base._heldout_protocol(args.dataset.resolve(), args.manifest.resolve())
    plans = []
    for spec in specs:
        _configure_phase(args.run_id, spec)
        phase_args = _base_args(args, spec)
        if Path(spec["warm_start"]).is_file():
            plans.append(_control_plan(phase_args, protocol))
            continue
        # Phase B's weight intentionally does not exist until Phase A finishes.
        if plans:
            plan = copy.deepcopy(plans[0])
        else:
            template_args = copy.copy(phase_args)
            template_args.warm_start = ORIGINAL_WEIGHTS
            plan = _control_plan(template_args, protocol)
        plan.update(
            {
                "phase": spec["name"],
                "phase_run_id": spec["name"],
                "cumulative_step_end": spec["cumulative_step_end"],
                "warm_start": str(Path(spec["warm_start"]).resolve()),
                "warm_start_sha256": "pending_phase_a_export",
            }
        )
        plan["schedule"].update(
            {
                "estimated_total_steps": spec["steps"],
                "warmup_steps": max(1, int(0.01 * int(spec["steps"]))),
                "max_steps": spec["steps"],
                "comparison_alignment": "robust_v1 4000-step continuation schedule",
            }
        )
        plan["training"]["optimizer_state_source"] = (
            "new AdamW state after reloading Phase A exported weights"
        )
        plan["checkpoint_export"] = {
            "filename": spec["export_filename"],
            "directory": str((CHECKPOINT_ROOT / args.run_id).resolve()),
        }
        plan["paths"] = {
            "checkpoint": str(
                (CHECKPOINT_ROOT / args.run_id / "training_state" / spec["name"]).resolve()
            ),
            "output": str((OUTPUT_ROOT / args.run_id / spec["name"]).resolve()),
            "tensorboard": str((TENSORBOARD_ROOT / args.run_id / spec["name"]).resolve()),
        }
        plans.append(plan)
    print(json.dumps({"phases": plans}, indent=2, sort_keys=True))
    return 0


def _self_test() -> int:
    pixels = torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8)
    sample = {"pixels": pixels, "action": torch.ones(1)}
    transformed = IdentityNoAugmentation(probability=1.0, max_delta=0.5)(sample)
    assert transformed is sample
    assert transformed["pixels"] is pixels
    assert torch.equal(transformed["pixels"], pixels)
    assert PHASE_A_STEPS + PHASE_B_STEPS == FORMAL_CUMULATIVE_STEPS
    print("control_noaugment self-test: PASS")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="control_noaugment_seed3072")
    p.add_argument("--phase", choices=("all", "phase_a", "phase_b"), default="all")
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--precision", choices=("bf16-mixed",), default="bf16-mixed")
    p.add_argument("--limit-val-batches", type=int, default=50)
    p.add_argument("--log-every-n-steps", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> int:
    p = parser()
    args = p.parse_args()
    if args.self_test:
        return _self_test()
    args.run_id = base._safe_run_id(args.run_id)
    if args.smoke and "smoke" not in args.run_id.lower():
        p.error("--smoke requires 'smoke' in --run-id to protect formal outputs")
    if args.batch_size < 1 or args.num_workers < 0:
        p.error("batch size/workers are invalid")
    if not 0.0 < args.train_split < 1.0:
        p.error("--train-split must be in (0,1)")
    if args.learning_rate <= 0 or args.gradient_clip_val <= 0:
        p.error("learning rate and gradient clip must be positive")

    ordered = ["phase_a", "phase_b"] if args.phase == "all" else [args.phase]
    specs = [_phase_spec(args.run_id, phase, args.smoke) for phase in ordered]
    if args.dry_run:
        return _dry_run(args, specs)

    _write_master_plan(args, specs)
    for spec in specs:
        _run_phase(args, spec)

    complete = all(_phase_complete(args.run_id, spec) for spec in specs)
    base._write_json(
        OUTPUT_ROOT / args.run_id / "completed.json",
        {
            "run_id": args.run_id,
            "requested_phases": ordered,
            "complete": complete,
            "formal_cumulative_steps": FORMAL_CUMULATIVE_STEPS,
            "exports": [
                str((CHECKPOINT_ROOT / args.run_id / spec["export_filename"]).resolve())
                for spec in specs
            ],
        },
    )
    if not complete:
        raise RuntimeError("one or more requested phases did not complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
