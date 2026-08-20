#!/usr/bin/env python3
"""Robust-v1 Cube finetune using the frozen Route21 training lifecycle."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_cube_maskedaug as base  # noqa: E402
from cube_robustaug import RobustVisualAugmentation, save_robust_qc_artifacts  # noqa: E402


base.OUTPUT_ROOT = base.AILAB / "outputs/train/robust_v1"
base.CHECKPOINT_ROOT = base.AILAB / "checkpoints/lewm-cube-robust_v1"
base.TENSORBOARD_ROOT = base.AILAB / "logs/tensorboard/robust_v1"
base.WARM_WEIGHTS = (
    base.AILAB
    / "checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt"
)
base.RandomMaskedHueRotation = RobustVisualAugmentation
base.save_qc_artifacts = save_robust_qc_artifacts
_base_plan = base._plan


def _robust_plan(args: object, protocol: dict[str, object]) -> dict[str, object]:
    plan = _base_plan(args, protocol)
    plan["route"] = "robust_v1_full_visual_axis_float32"
    plan["augmentation"]["axes"] = [
        "masked_cube_hue_shift",
        "low_saturation_dark_background_hue_shift",
        "full_frame_brightness_gamma",
    ]
    plan["augmentation"]["space_and_precision"] = "float32 HSV pixel space"
    plan["augmentation"]["scope"] = "cube/background masks plus full-frame gamma"
    plan["augmentation"]["background_mask"] = {
        "space": "HSV float32",
        "saturation_lt": 0.40,
        "value_lt": 0.35,
    }
    plan["augmentation"]["gamma_range"] = [0.7, 1.4]
    plan["augmentation"]["implementation"] = "single_float32_hsv_pass"
    plan["stopline"] = {
        "metric": "heldout expert validation pred_loss",
        "relative_increase_limit": 0.10,
        "policy": "posthoc comparison against MaskedAug baseline; no checkpoint promotion on breach",
    }
    return plan


base._plan = _robust_plan


class SaveRobustWeights:
    @staticmethod
    def create(run_id: str, model_config: object):
        from lightning.pytorch.callbacks import Callback
        from stable_worldmodel.wm.utils import save_pretrained

        class CallbackImpl(Callback):
            def _save(self, trainer: object, module: object, filename: str) -> None:
                if trainer.is_global_zero:
                    save_pretrained(
                        module.model,
                        run_name=f"lewm-cube-robust_v1/{run_id}",
                        config=model_config,
                        filename=filename,
                        cache_dir=str(base.AILAB),
                    )

            def on_train_epoch_end(self, trainer: object, module: object) -> None:
                self._save(trainer, module, f"weights_epoch_{trainer.current_epoch + 1}.pt")

            def on_train_end(self, trainer: object, module: object) -> None:
                self._save(trainer, module, "weights_final.pt")

        return CallbackImpl()


base.SaveRoute21Weights = SaveRobustWeights


if __name__ == "__main__":
    raise SystemExit(base.main())
