# Ultrasound Tumor Detection

U-Net segmentation for breast tumor regions in grayscale ultrasound images. The project is packaged under `src/ultrasound_tumor_detection` and provides reusable Python modules, command-line entry points for the BUSI training/evaluation pipeline, and a PyQt5 desktop GUI for interactive inference.

This code was built for Cornell ORIE5270 coursework.

## Overview

The model takes a single-channel breast ultrasound image and outputs a one-channel segmentation map with pixel values in `[0, 1]` (sigmoid applied inside the model). The BUSI dataset classes (`benign`, `malignant`, and `normal`) are used as image categories, but the learning task is binary pixel segmentation: tumor mask versus background. Normal images are expected to have empty/no-tumor masks.

At inference time, pixel probabilities above `0.65` are counted as tumor pixels. The GUI uses the same threshold for tumor-pixel counting and heatmap display.

## Project Layout

```text
.
├── Dataset_BUSI_with_GT/              # BUSI data in source checkout, when present
├── models/                            # optional user-trained checkpoints
├── scripts/                           # thin wrappers around package CLI functions
├── src/ultrasound_tumor_detection/
│   ├── __init__.py                    # public imports
│   ├── __main__.py                    # PyQt5 GUI application
│   ├── assets/
│   │   ├── unet_model.pth             # bundled pretrained checkpoint
│   │   └── sample_image.png           # default image loaded on GUI startup
│   ├── cli.py                         # utd-run, utd-train, utd-evaluate, utd-gui
│   ├── data.py                        # BUSI download/loading
│   ├── losses.py                      # BCE + Dice loss and Dice score
│   ├── model.py                       # U-Net architecture
│   └── pipeline.py                    # train/evaluate/full pipeline functions
├── tests/                             # pytest suite
├── pyproject.toml                     # package metadata and install config
└── README.md
```

Large local artifacts such as user-trained model checkpoints, coverage reports, and Python caches are ignored by git.

## Architecture

The model is a 3-level U-Net with skip connections:

```text
Input (B, 1, H, W), where H and W are divisible by 8
  |
  |-- Encoder
  |     enc1: Conv block 1 -> 64
  |     enc2: Conv block 64 -> 128
  |     enc3: Conv block 128 -> 256
  |
  |-- Bottleneck: Conv block 256 -> 512 + Dropout2d(0.3)
  |
  |-- Decoder
        up3 + dec3: ConvTranspose + Conv block 512 -> 256
        up2 + dec2: ConvTranspose + Conv block 256 -> 128
        up1 + dec1: ConvTranspose + Conv block 128 -> 64
        final: Conv 64 -> 1 + Sigmoid

Output (B, 1, H, W) probabilities in [0, 1]
```

Each convolution block is two `Conv2d + BatchNorm2d + ReLU` layers. Sigmoid is applied inside `forward()` as the final step, so the model always outputs probabilities — no external sigmoid is needed.

`UNet.forward()` validates its input and raises:

- `ValueError` if the tensor is not 4-D or does not have exactly one channel.
- `RuntimeError` if height/width are not divisible by 8 or if a skip-connection shape mismatch is detected.

## Dataset

The project uses BUSI, the Breast Ultrasound Images Dataset: 780 grayscale PNG images with pixel-level ground-truth masks.

| Class | Meaning |
| --- | --- |
| `benign` | Non-cancerous tumor images |
| `malignant` | Cancerous tumor images |
| `normal` | No tumor present |

`BUSIDataset` expects the standard folder layout:

```text
Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/
```

For each image, it looks for a matching `_mask.png` file, skips orphan images and unreadable/shape-mismatched pairs, resizes images and masks to `256 x 256`, normalizes values to `[0, 1]`, and returns `(image, mask)` tensors with shape `(1, 256, 256)` and dtype `float32`. It raises `ValueError` if no valid image-mask pairs are found.

Dataset page: <https://scholar.cu.edu.eg/?q=afahmy/pages/dataset>

Kaggle mirror used by `kagglehub`: <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>

Citation: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: `10.1016/j.dib.2019.104863`.

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

For development and tests:

```bash
pip install -e ".[dev]"
```

For the desktop GUI:

```bash
pip install -e ".[gui]"
```

For everything:

```bash
pip install -e ".[dev,gui]"
```

For GPU training, install the PyTorch build that matches your CUDA version before installing this package. See the official PyTorch install selector for the correct command.

If you want the pipeline to download BUSI automatically, configure Kaggle credentials for `kagglehub` first, usually by placing `kaggle.json` at `~/.kaggle/kaggle.json` or by setting `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Command-Line Usage

Run the full download, train-or-load, and evaluate flow:

```bash
utd-run
```

Checkpoint behavior depends on whether you use the default model path or pass your own `--model-path`:

| Command state | What happens |
| --- | --- |
| Default path `models/unet_model.pth` exists | Load that local/user checkpoint and skip training. |
| Default path is missing, bundled `assets/unet_model.pth` exists | Load the bundled pretrained checkpoint and skip training. |
| Default path and bundled checkpoint are both missing | Train from scratch and save to `models/unet_model.pth`. |
| Custom `--model-path` exists | Load that custom checkpoint and skip training. |
| Custom `--model-path` is missing | Train from scratch and save to that custom path. The bundled checkpoint is not used for custom missing paths. |

This means the default command prefers a user-trained local model, then falls back to the bundled pretrained model. To force new training, pass a new custom `--model-path` or remove/rename the existing local checkpoint you want to replace.

Useful options:

```bash
utd-run --epochs 10
utd-run --data-dir Dataset_BUSI_with_GT --model-path models/unet_model.pth
```

Train/evaluate entry points are thin wrappers around the same pipeline:

```bash
utd-train --data-dir Dataset_BUSI_with_GT --epochs 10 --model-path models/unet_model.pth
utd-evaluate --data-dir Dataset_BUSI_with_GT --model-path models/unet_model.pth
```

To train your own checkpoint without replacing the default local model path, choose another path that does not already exist:

```bash
utd-run --data-dir Dataset_BUSI_with_GT --epochs 10 --model-path models/my_unet_model.pth
```

The source-checkout script wrappers are also available:

```bash
python scripts/run_pipeline.py --data-dir Dataset_BUSI_with_GT
python scripts/train_model.py --data-dir Dataset_BUSI_with_GT
python scripts/evaluate_model.py --data-dir Dataset_BUSI_with_GT
```

## GUI Usage

Launch the desktop GUI with:

```bash
python -m ultrasound_tumor_detection
```

or via the CLI entry point:

```bash
utd-gui
```

The GUI is a PyQt5 desktop app titled `Ultrasound Tumor Detection - Multi-Image Analysis`. It loads a `UNet`, runs on CUDA when available, and uses the default checkpoint resolution:

```text
1. models/unet_model.pth
2. bundled assets/unet_model.pth
```

The package includes the pretrained model as package data, so the GUI can be used immediately after installing the GUI extra. If both the local and bundled checkpoints are missing, the GUI prints a warning and runs with an untrained model. To use your own trained model in the GUI, save it to `models/unet_model.pth`.

On startup, the GUI automatically loads a bundled `sample_image.png` so there is always something to view.

GUI features:

- Load one image or multiple PNG/JPG/BMP images.
- Open each loaded image in a closeable tab.
- Resize each image to `256 x 256`, normalize to `[0, 1]`, and run model inference immediately.
- Show the raw grayscale image and a probability heatmap side-by-side with a `hot` colormap and colorbar.
- Display tumor status, confidence bar, tumor pixel count, coverage percentage, and max model confidence.
- Draw a rectangle to zoom the raw image and heatmap together; left-click-drag to pan while zoomed.
- Reset zoom per image, close individual tabs, or clear all loaded images.

The GUI tumor status counts pixels with probability greater than `0.65`; more than `1000` such pixels is reported as detected.

GUI layout:

```text
Top controls: Load Image | Load Multiple Images | Clear All | Images loaded: N
Main area:    closeable image tabs
Per tab:      left metrics/control panel + right raw-image and heatmap canvases
```

## Training Details

| Hyperparameter | Value |
| --- | --- |
| Input size | `256 x 256` grayscale |
| Train/test split | 80/20 via `torch.utils.data.random_split` |
| Batch size | 8 by default |
| Optimizer | Adam, `lr=1e-4` |
| Epochs | 10 by default |
| Loss | BCE + Dice loss (on sigmoid probabilities) |
| Prediction threshold | 0.65 for masks and GUI counting |

Loss (both terms operate on probabilities, since sigmoid is in the model):

```text
loss = BCE(pred_probs, target_mask) + (1 - Dice(pred_probs, target_mask))
```

The evaluation metric is average Dice score on the held-out test loader.

## Python API

Use the model directly:

```python
from ultrasound_tumor_detection import UNet

model = UNet()
# model(x) returns probabilities in [0, 1] — no external sigmoid needed
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

run_pipeline(
    data_dir="Dataset_BUSI_with_GT",
    model_path="models/unet_model.pth",
    epochs=10,
)
```

## Tests

Run the test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=ultrasound_tumor_detection
```

The tests cover dataset parsing and filtering, model shape/input validation and persistence, loss/metric behavior, training mechanics, pipeline behavior, and GUI widget state, inference logic, zoom/pan interactions, and tab management.

## AI Disclosure

AI was used to generate boilerplate code and components of this documentation.
