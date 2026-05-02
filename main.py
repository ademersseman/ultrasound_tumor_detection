"""Backward-compatible entry point.

Prefer the installed console command:

    utd-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ultrasound_tumor_detection import BUSIDataset, UNet, bce_dice_loss, dice_score, visualize_predictions
from ultrasound_tumor_detection.pipeline import EPOCHS, MODEL_PATH, main, run_pipeline

__all__ = [
    "BUSIDataset",
    "EPOCHS",
    "MODEL_PATH",
    "UNet",
    "bce_dice_loss",
    "dice_score",
    "main",
    "run_pipeline",
    "visualize_predictions",
]


if __name__ == "__main__":
    main()
