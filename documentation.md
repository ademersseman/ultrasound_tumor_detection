# Ultrasound Tumor Detection

The goal of this project is the segmentation of breast tumors in ultrasound images using a U-Net trained on [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

---

## Overview

The model takes a grayscale breast ultrasound image(ultrasound capture) and outputs a segmentation heatmap identifying the tumor region with confidence levels. It handles three tissue classes in the dataset — **benign**, **malignant**, and **normal** — treating them uniformly as a segmentation task (tumor present vs. absent).

---

## Architecture

A 3-level U-Net with skip connections:

```
Input (1×256×256)
  │
  ├─ Encoder
  │    enc1: Conv(1→64)   ──────────────────────────────┐
  │    enc2: Conv(64→128) ───────────────────────┐      │  skip
  │    enc3: Conv(128→256)────────────┐          │      │  connections
  │                                   ↓          ↓      ↓
  ├─ Bottleneck: Conv(256→512) + Dropout2d(0.3)
  │
  └─ Decoder
       up3 + dec3: ConvTranspose + Conv(512→256)
       up2 + dec2: ConvTranspose + Conv(256→128)
       up1 + dec1: ConvTranspose + Conv(128→64)
       final: Conv(64→1)  [raw logits]

Output (1×256×256)  [raw logits; sigmoid applied at inference/loss time]
```

Each encoder/decoder block is two Conv2d + BatchNorm2d + ReLU layers. Skip connections concatenate encoder feature maps with upsampled decoder maps at each level.

`forward()` validates its input and raises:
- `ValueError` if the input is not 4-D, does not have exactly 1 channel, or spatial dimensions are not divisible by 8.
- `RuntimeError` if a skip-connection shape mismatch is detected.

---

## Dataset

**BUSI — Breast Ultrasound Images Dataset**  
780 grayscale PNG images (average size ~500×500) with pixel-level ground-truth masks.

| Class | Description |
|---|---|
| benign | Non-cancerous tumors |
| malignant | Cancerous tumors |
| normal | No tumor present |

The dataset is included in the github and cited in the readme and below

`BUSIDataset` silently skips:
- **Orphan images** — images with no corresponding `_mask.png`.
- **Corrupt files** — images/masks that OpenCV cannot decode or whose shapes do not match.

It raises `ValueError` if no valid pairs are found after filtering.

---

## Installation

```bash
# CPU
pip install torch opencv-python numpy matplotlib kagglehub PyQt5

# GPU (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy matplotlib kagglehub PyQt5
```

Or from the requirements file:

```bash
pip install -r requirements.txt
```

Set up Kaggle credentials before running (required for dataset download):

```bash
# Place kaggle.json at ~/.kaggle/kaggle.json
# or set environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## Usage

### `main.py` — training and evaluation pipeline

```bash
python main.py
```

**First run** — downloads the dataset, trains for 50 epochs, saves `unet_model.pth`, evaluates on the test set, and displays prediction visualizations.

**Subsequent runs** — loads `unet_model.pth`, skips training, evaluates, and visualizes.

### `gui.py` — interactive desktop app

Requires `unet_model.pth` to exist — run `main.py` at least once first.

```bash
python gui.py
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Input size | 256 × 256 (grayscale) |
| Train / test split | 80 / 20 |
| Batch size | 8 |
| Optimizer | Adam, lr = 1e-4 |
| Epochs | 10 |
| Loss | BCE (with logits) + Dice |
| Prediction threshold | 0.65 |

**Loss function** — combined Binary Cross-Entropy and Dice loss operating on raw logits:

```
L = BCEWithLogits(pred, target) + (1 - Dice(sigmoid(pred), target))
```

The model outputs raw logits; sigmoid is applied inside both the loss function and `dice_score`.

**Evaluation metric** — Dice score on the held-out test set (higher is better, max = 1.0).

---

## Output (command-line)

After evaluation, the script displays a three-panel figure for each test image:

| Panel | Content |
|---|---|
| Left | Original ultrasound |
| Center | Ground-truth mask |
| Right | Predicted mask (sigmoid + threshold 0.65) |

---

## GUI

`gui.py` provides a PyQt5 desktop application for interactive inference on arbitrary images.

### Features

- **Multi-image tabbed interface** — each loaded image opens in its own closeable tab.
- **Load single or multiple images** — PNG, JPG, and BMP are supported.
- **Per-image analysis panel** showing:
  - Raw grayscale image
  - Probability heatmap (`hot` colormap, `[0, 1]` range) with colorbar
  - Tumor status (detected / not detected) with threshold at 0.65
  - Confidence bar and percentage
  - Tumor pixel count and coverage percentage
  - Max model confidence
- **Interactive zoom** — drag a rectangle on either the raw image or heatmap to zoom both views simultaneously.
- **Pan** — click-drag while zoomed to pan both views.
- **Reset Zoom** — restores full-image view and re-enables the zoom selector.
- **Clear All** — removes all tabs and resets the image count.

### Layout

```
┌─────────────────────────────────────────────┐
│  [Load Image]  [Load Multiple]  [Clear All] │
│  Images loaded: N                           │
├─────────────────────────────────────────────┤
│  Tab 1  │  Tab 2  │  Tab 3  │  ...          │
│ ┌──────────────────────────────────────────┐│
│ │ Left panel (controls)                    ││
│ │  • Image name                            ││
│ │  • Confidence label + progress bar       ││
│ │  • Tumor status                          ││
│ │  • Tumor pixel count                     ││
│ │  • Coverage / max confidence stats       ││
│ │  • Reset Zoom button                     ││
│ │                                          ││
│ │ Right panel                              ││
│ │  Raw Image (zoomable/pannable)           ││
│ │  Heatmap   (zoomable/pannable)           ││
│ └──────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

---

## Tests

Tests are organized under the `tests/` directory with shared fixtures in `conftest.py`.

```bash
pip install pytest
python -m pytest tests/ -v
```

### `conftest.py` — shared fixtures

| Fixture | Description |
|---|---|
| `fake_dataset_dir` | Temp directory with 2 images + masks per class (benign, malignant, normal) |
| `dataset` | `BUSIDataset` built from `fake_dataset_dir` |
| `model` | `UNet()` in eval mode |

### `test_data.py` — `BUSIDataset`

Tests the data loading pipeline using the synthetic dataset from `conftest.py`.

**Shape and dtype**
- `test_sample_shape` — `(img, mask)` both have shape `(1, 256, 256)`; grayscale channel confirmed.
- `test_dtype` — both tensors are `float32`.

**Value invariants**
- `test_image_range` — image pixels are finite and in `[0.0, 1.0]`.
- `test_mask_range_and_binary` — mask values are finite, in `[0.0, 1.0]`, and only `{0.0, 1.0}`.

**Dataset-level**
- `test_non_empty` — dataset has at least one sample.
- `test_all_samples_valid` — no NaN values; image and mask shapes match across all items.

**Alignment correctness**
- `test_image_mask_alignment` — each image path and mask path share the same base name.

**Edge cases**
- `test_orphan_image_skipped` — a PNG with no `_mask.png` counterpart is excluded from `image_paths` and does not break iteration.
- `test_corrupt_file_skipped` — a file containing invalid image data is excluded and does not break iteration.

**DataLoader compatibility**
- `test_dataloader_batching` — `DataLoader(batch_size=2)` produces correctly shaped batches.
- `test_deterministic_access` — `dataset[0]` returns identical tensors on repeated calls.

### `test_model.py` — `UNet`

Tests the model architecture using random tensors; no dataset needed.

**`TestUNetArchitecture`**
- `test_output_shape_matches_input` — `(1,1,256,256)` → `(1,1,256,256)`.
- `test_output_is_probability` — output is in `[0,1]` (sigmoid applied).
- `test_no_nan_or_inf` — output is finite for random input.
- `test_extreme_input_values` — output remains finite with very large inputs (1e6).
- `test_batch_size_agnostic` — parametrized over batch sizes 1, 2, 8.
- `test_output_varies_with_input` — blank and all-white inputs produce different predictions.
- `test_valid_resolutions` — parametrized over 64×64, 128×128, 256×256.
- `test_invalid_resolution_raises` — 250×250 (not divisible by 8) raises `RuntimeError`.
- `test_eval_mode_deterministic` — two forward passes in eval mode produce identical outputs.
- `test_train_mode_not_deterministic` — two forward passes in train mode differ (Dropout2d).

**`TestUNetPersistence`**
- `test_save_and_load_produces_identical_output` — saves weights with `torch.save`, loads into a fresh model, verifies bit-for-bit identical output in eval mode.

### `test_loss.py` — `bce_dice_loss` and `dice_score`

**`TestDiceScoreCorrectness`**
- `test_perfect_overlap` — `dice_score(target, target) ≈ 1.0`.
- `test_zero_overlap` — all-ones target vs. all-zeros prediction → score < 0.1.
- `test_all_zero_case` — both all-zero → score ≈ 1.0 (smooth denominator prevents division by zero).
- `test_output_in_zero_one_range` — score stays in `[0.0, 1.0]` for arbitrary inputs.
- `test_threshold_effect` — lower threshold accepts more positives and yields higher score.
- `test_symmetry` — `dice_score(a, b) ≈ dice_score(b, a)`.
- `test_invalid_inputs_raise` — mismatched shapes raise `ValueError`.

**`TestBceDiceLossShape`**
- `test_returns_scalar` — loss is a 0-dimensional tensor.
- `test_no_nan` — loss is never NaN.
- `test_invalid_inputs_raise` — mismatched shapes raise `ValueError`.

**`TestBceDiceLossValues`**
- `test_non_negative` — loss is always ≥ 0.
- `test_loss_finite` — loss is always finite.
- `test_all_zero_edge_case` — all-zero pred + all-zero target does not produce NaN or inf.
- `test_perfect_prediction_lower_than_random` — `loss(target, target) < loss(random, target)`.
- `test_loss_increases_as_prediction_worsens` — prediction at 0.8 incurs less loss than prediction at 0.1 toward a target of 0.9.
- `test_worst_case_all_zeros_vs_all_ones_is_high` — all-zeros pred vs. all-ones target produces loss > 5.0.

**`TestBceDiceLossGradients`**
- `test_loss_is_differentiable` — `.backward()` populates non-NaN gradients.

### `test_training.py` — training mechanics

**`TestTrainingStepMechanics`**
- `test_weights_update_after_optimizer_step` — `enc1` weights change after one Adam step.
- `test_weights_remain_finite_after_step` — all parameters stay finite after an update.
- `test_gradients_exist_and_finite` — all requires-grad parameters receive finite, non-zero gradients.
- `test_zero_grad_clears_all_gradients` — `optimizer.zero_grad()` zeroes every parameter gradient.

**`TestReproducibility`**
- `test_fixed_seed_produces_same_output` — same manual seed → same model weights and input → identical output.
- `test_different_seed_different_output` — different seeds → different initialized weights → different outputs.

**`TestIntegration`**
- `test_two_training_batches_no_crash` — two full training batches (data load → forward → loss → backward → step) complete without error.
- `test_overfit_tiny_dataset_loss_decreases` — 15 gradient steps on a 6-sample dataset reduce loss (confirms the model can learn).

---

## GUI Test Plan (in development)

The following test scenarios are planned for `tests/test_gui.py`. PyQt5 widget testing requires `pytest-qt` (`pip install pytest-qt`).

### Startup and model loading
- Application starts without error when `unet_model.pth` is present.
- Application starts (with a printed warning) when `unet_model.pth` is absent; no crash.
- Window title is `"Ultrasound Tumor Detection - Multi-Image Analysis"`.
- Initial image count label reads `"Images loaded: 0"`.

### Image loading
- **Single image** — clicking "Load Image" and selecting a valid PNG creates exactly one new tab; count label increments to 1.
- **Multiple images** — "Load Multiple Images" with N files creates N tabs; count label shows N.
- **Invalid / unreadable file** — selecting a non-image file does not crash the application.
- **Duplicate loads** — loading the same file twice creates two independent tabs.

### Tab management
- Clicking the close button on a tab removes it and decrements the count label.
- Closing all tabs individually leaves count at 0.
- "Clear All" removes all tabs at once and resets the count to 0.

### Per-tab analysis panel (`ImageAnalysisTab`)
- After loading, the raw image canvas is non-empty (not blank).
- After loading, the heatmap canvas is non-empty.
- Tumor status label contains either `"Tumor Detected"` or `"No Tumor Detected"`.
- Confidence bar value is in `[0, 100]`.
- Tumor pixel label text matches the format `"Tumor Pixels: X / Y"`.
- Statistics label contains `"Coverage"` and `"Max Confidence"`.

### Confidence metric logic (unit-testable without Qt)
- `update_confidence_metrics` with `current_pred_raw` all above 0.65 → tumor detected branch.
- `update_confidence_metrics` with `current_pred_raw` all below 0.65 → no tumor branch.
- Confidence bar value is clamped to `[0, 100]`.

### Zoom and pan
- After drag-selecting a rectangle, both raw image and heatmap axes limits change.
- "Reset Zoom" restores axes limits to `(0, W)` × `(0, H)` and re-enables the rectangle selector.
- Panning is only active when `is_zoomed()` returns `True`.
- `is_zoomed()` returns `False` at full extent and `True` after a zoom.

### Preprocessing contract
- `add_image` resizes any input image to 256×256 before storing.
- The tensor sent to the model has shape `(1, 1, 256, 256)` and dtype `float32`.
- Pixel values are divided by 255 before inference.

---

## File Structure

```
ultrasound_tumor_detection/
├── main.py            # Data loading, model, training, evaluation pipeline
├── gui.py             # PyQt5 desktop GUI for interactive inference
├── unet_model.pth     # Saved model weights (generated on first run of main.py)
├── requirements.txt
├── documentation.md
└── tests/
    ├── conftest.py    # Shared fixtures (fake dataset, model)
    ├── test_data.py   # BUSIDataset tests
    ├── test_loss.py   # bce_dice_loss and dice_score tests
    ├── test_model.py  # UNet architecture and persistence tests
    └── test_training.py  # Training step, reproducibility, and integration tests
```

## AI Disclosure

AI was used to generate boilerplate code and components of this document.