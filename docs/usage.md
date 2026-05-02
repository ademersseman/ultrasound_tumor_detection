# Usage

Install the project before importing or running commands:

```bash
pip install -e ".[dev]"
```

Common commands:

```bash
utd-run --epochs 10
utd-train --data-dir Dataset_BUSI_with_GT --model-path models/unet_model.pth
utd-evaluate --data-dir Dataset_BUSI_with_GT --model-path models/unet_model.pth --no-show
pytest
```

Common imports:

```python
from ultrasound_tumor_detection import UNet
from ultrasound_tumor_detection.data import BUSIDataset, download_busi_dataset
from ultrasound_tumor_detection.losses import bce_dice_loss, dice_score
from ultrasound_tumor_detection.pipeline import run_pipeline
```
