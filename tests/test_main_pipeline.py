"""
Integration tests for main(), mocking the external I/O with synthetic data, 
covering the training path with 1 epoch.
"""

import os
import shutil
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

import main as main_module
from main import UNet, main

_real_exists = os.path.exists


# -------------------------
# Helpers
# -------------------------

# os.path.exists that returns False only for the model checkpoint
def _fake_exists_no_model(path):
    if path == "unet_model.pth":
        return False
    return _real_exists(path)

# os.path.exists that returns True only for the model checkpoint
def _fake_exists_with_model(path):
    if path == "unet_model.pth":
        return True
    return _real_exists(path)


# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
# randomly generate a dataset
def kaggle_dir():
    tmp = tempfile.mkdtemp()
    busi = os.path.join(tmp, "Dataset_BUSI_with_GT")

    for cls in ["benign", "malignant", "normal"]:
        cls_dir = os.path.join(busi, cls)
        os.makedirs(cls_dir)

        for i in range(1, 3):
            img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            mask = (img > 128).astype(np.uint8) * 255

            cv2.imwrite(os.path.join(cls_dir, f"image ({i}).png"), img)
            cv2.imwrite(os.path.join(cls_dir, f"image ({i})_mask.png"), mask)

    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(0)


# -------------------------
# Running helpers
# -------------------------

# run main() with no saved model (aka we want to train)
def run_no_model(kaggle_dir):
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir) as mock_dl, \
         patch("main.os.path.exists", side_effect=_fake_exists_no_model), \
         patch("main.torch.save") as mock_save, \
         patch("main.visualize_predictions") as mock_viz, \
         patch("main.EPOCHS", 1):
        main()
    return mock_dl, mock_save, mock_viz

# run main() with an existing saved model (skip training, use model)
def run_with_model(kaggle_dir):
    state_dict = UNet().state_dict()
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("main.os.path.exists", side_effect=_fake_exists_with_model), \
         patch("main.torch.load", return_value=state_dict) as mock_load, \
         patch("main.torch.save") as mock_save, \
         patch("main.visualize_predictions") as mock_viz:
        main()
    return mock_load, mock_save, mock_viz


# -------------------------
# Training path (no saved model)
# -------------------------

# full pipeline runs without raising
def test_no_model_runs_without_crash(kaggle_dir):
    run_no_model(kaggle_dir)

# model weights are written to disk after training
def test_torch_save_called_once(kaggle_dir):
    _, mock_save, _ = run_no_model(kaggle_dir)
    assert mock_save.call_count == 1

# weights are saved under the expected filename
def test_model_saved_to_correct_path(kaggle_dir):
    _, mock_save, _ = run_no_model(kaggle_dir)
    assert mock_save.call_args[0][1] == "unet_model.pth"

# dataset is fetched with the correct Kaggle slug
def test_kaggle_slug_correct(kaggle_dir):
    mock_dl, _, _ = run_no_model(kaggle_dir)
    mock_dl.assert_called_once_with("aryashah2k/breast-ultrasound-images-dataset")

# visualize_predictions is called once at the end
def test_no_model_visualize_called(kaggle_dir):
    _, _, mock_viz = run_no_model(kaggle_dir)
    assert mock_viz.call_count == 1


# -------------------------
# Skip-training path 
# -------------------------

# full pipeline runs without raising when a checkpoint is present
def test_with_model_runs_without_crash(kaggle_dir):
    run_with_model(kaggle_dir)

# no new checkpoint is written when a saved model is loaded
def test_torch_save_not_called(kaggle_dir):
    _, mock_save, _ = run_with_model(kaggle_dir)
    mock_save.assert_not_called()

# existing checkpoint is loaded from disk
def test_torch_load_called(kaggle_dir):
    mock_load, _, _ = run_with_model(kaggle_dir)
    assert mock_load.called

# checkpoint is loaded from the expected path
def test_torch_load_uses_model_path(kaggle_dir):
    mock_load, _, _ = run_with_model(kaggle_dir)
    assert mock_load.call_args[0][0] == "unet_model.pth"

# visualize_predictions is called once even without training
def test_with_model_visualize_called(kaggle_dir):
    _, _, mock_viz = run_with_model(kaggle_dir)
    assert mock_viz.call_count == 1

# -------------------------
# Data flow
# -------------------------

# BUSIDataset is constructed with <download_path>/Dataset_BUSI_with_GT
def test_busi_dataset_receives_correct_path(kaggle_dir):
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("main.os.path.exists", side_effect=_fake_exists_no_model), \
         patch("main.torch.save"), \
         patch("main.visualize_predictions"), \
         patch("main.EPOCHS", 1), \
         patch("main.BUSIDataset", wraps=main_module.BUSIDataset) as mock_ds:
        main()
    assert mock_ds.call_args[0][0] == os.path.join(kaggle_dir, "Dataset_BUSI_with_GT")

# tensors passed to visualize_predictions are on the CPU
def test_visualize_receives_cpu_tensors(kaggle_dir):
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("main.os.path.exists", side_effect=_fake_exists_no_model), \
         patch("main.torch.save"), \
         patch("main.visualize_predictions") as mock_viz, \
         patch("main.EPOCHS", 1):
        main()
    imgs, masks, preds = mock_viz.call_args[0]
    assert imgs.device.type == "cpu"
    assert masks.device.type == "cpu"
    assert preds.device.type == "cpu"

# all three tensors are 4-D (B, C, H, W)
def test_visualize_receives_4d_tensors(kaggle_dir):
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("main.os.path.exists", side_effect=_fake_exists_no_model), \
         patch("main.torch.save"), \
         patch("main.visualize_predictions") as mock_viz, \
         patch("main.EPOCHS", 1):
        main()
    imgs, masks, preds = mock_viz.call_args[0]
    assert imgs.ndim == 4
    assert masks.ndim == 4
    assert preds.ndim == 4

# imgs, masks, preds share the same shape
def test_visualize_tensors_have_matching_shapes(kaggle_dir):
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("main.os.path.exists", side_effect=_fake_exists_no_model), \
         patch("main.torch.save"), \
         patch("main.visualize_predictions") as mock_viz, \
         patch("main.EPOCHS", 1):
        main()
    imgs, masks, preds = mock_viz.call_args[0]
    assert imgs.shape == masks.shape == preds.shape

# images are single-channel (grayscale)
def test_visualize_single_channel_images(kaggle_dir):
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("main.os.path.exists", side_effect=_fake_exists_no_model), \
         patch("main.torch.save"), \
         patch("main.visualize_predictions") as mock_viz, \
         patch("main.EPOCHS", 1):
        main()
    imgs, _, _ = mock_viz.call_args[0]
    assert imgs.shape[1] == 1

# model outputs probabilities in [0, 1] (sigmoid applied in forward)
def test_prediction_values_in_zero_one_range(kaggle_dir):
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("main.os.path.exists", side_effect=_fake_exists_no_model), \
         patch("main.torch.save"), \
         patch("main.visualize_predictions") as mock_viz, \
         patch("main.EPOCHS", 1):
        main()
    _, _, preds = mock_viz.call_args[0]
    assert preds.min().item() >= 0.0
    assert preds.max().item() <= 1.0

# dataset corruption and failure must raise
def test_dataset_failure_raises(kaggle_dir):
    empty_root = kaggle_dir
    shutil.rmtree(os.path.join(empty_root, "Dataset_BUSI_with_GT"))
    os.makedirs(os.path.join(empty_root, "Dataset_BUSI_with_GT"))

    with patch("main.kagglehub.dataset_download", return_value=empty_root), \
         patch("main.os.path.exists", side_effect=_fake_exists_no_model):

        with pytest.raises(ValueError):
            main()


def test_corrupt_checkpoint_raises(kaggle_dir):
    with patch("main.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("main.os.path.exists", side_effect=_fake_exists_with_model), \
         patch("main.torch.load", return_value={"bad": "state"}):

        with pytest.raises(RuntimeError):
            main()
