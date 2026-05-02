# Ultrasound Tumor Detection

U-Net segmentation model for detecting tumor regions in breast ultrasound images. This project uses the BUSI dataset and provides reusable Python package modules plus command-line entry points for training, evaluation, and visualization.

This code was built for Cornell ORIE5270 coursework.

## Project Layout

```text
.
├── src/ultrasound_tumor_detection/
│   ├── data.py            # BUSI dataset download/loading
│   ├── model.py           # U-Net architecture
│   ├── losses.py          # BCE + Dice loss and Dice score
│   ├── pipeline.py        # train/evaluate/full pipeline functions
│   ├── visualization.py   # prediction plotting helpers
│   └── cli.py             # command-line entry points
├── scripts/               # thin script wrappers
├── tests/                 # pytest suite
├── pyproject.toml         # package metadata and install config
└── README.md
```

Large local artifacts such as datasets, model checkpoints, coverage reports, and Python caches are ignored by git.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

For development and testing tools:

```bash
pip install -e ".[dev]"
```

For GPU training, install the PyTorch build that matches your CUDA version before installing this package. See the official PyTorch install selector for the correct command.

## Run The Pipeline

Run the full download, train-or-load, evaluate, and visualization flow:

```bash
utd-run
```

By default this downloads the Kaggle BUSI dataset with `kagglehub`, saves model weights to `models/unet_model.pth`, and shows prediction plots.

Useful options:

```bash
utd-run --epochs 10
utd-run --data-dir Dataset_BUSI_with_GT --model-path models/unet_model.pth
utd-run --no-show
```

The old command still works after installation:

```bash
python3 main.py
```

## Train

Train when no checkpoint exists:

```bash
utd-train --epochs 10 --model-path models/unet_model.pth
```

Use a local BUSI dataset instead of downloading:

```bash
utd-train --data-dir Dataset_BUSI_with_GT --epochs 10
```

## Evaluate

Evaluate an existing checkpoint:

```bash
utd-evaluate --data-dir Dataset_BUSI_with_GT --model-path models/unet_model.pth --no-show
```

## Import From Python

Use the model directly:

```python
from ultrasound_tumor_detection import UNet

model = UNet()
```

Load a BUSI-formatted dataset:

```python
from ultrasound_tumor_detection.data import BUSIDataset

dataset = BUSIDataset("Dataset_BUSI_with_GT")
```

Download and load the standard Kaggle dataset:

```python
from ultrasound_tumor_detection.data import BUSIDataset, download_busi_dataset

data_dir = download_busi_dataset()
dataset = BUSIDataset(data_dir)
```

Use the loss and metric:

```python
from ultrasound_tumor_detection.losses import bce_dice_loss, dice_score

loss = bce_dice_loss(predictions, masks)
score = dice_score(predictions, masks)
```

Run the pipeline from Python:

```python
from ultrasound_tumor_detection.pipeline import run_pipeline

run_pipeline(data_dir="Dataset_BUSI_with_GT", model_path="models/unet_model.pth", epochs=10)
```

## Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=ultrasound_tumor_detection
```

## Dataset

Dataset page: <https://scholar.cu.edu.eg/?q=afahmy/pages/dataset>

Citation:

Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
