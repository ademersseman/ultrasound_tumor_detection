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

import ultrasound_tumor_detection.pipeline as main_module
from ultrasound_tumor_detection.pipeline import MODEL_PATH, main


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
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir) as mock_dl, \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False) as mock_load_checkpoint, \
         patch("ultrasound_tumor_detection.pipeline.torch.save") as mock_save, \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1):
        main()
    return mock_dl, mock_load_checkpoint, mock_save

# run main() with an existing saved model (skip training, use model)
def run_with_model(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=True) as mock_load_checkpoint, \
         patch("ultrasound_tumor_detection.pipeline.torch.save") as mock_save:
        main()
    return mock_load_checkpoint, mock_save


# -------------------------
# Training path (no saved model)
# -------------------------

# full pipeline runs without raising
def test_no_model_runs_without_crash(kaggle_dir):
    run_no_model(kaggle_dir)

# model weights are written to disk after training
def test_torch_save_called_once(kaggle_dir):
    _, _, mock_save = run_no_model(kaggle_dir)
    assert mock_save.call_count == 1

# weights are saved under the expected filename
def test_model_saved_to_correct_path(kaggle_dir):
    _, _, mock_save = run_no_model(kaggle_dir)
    assert mock_save.call_args[0][1] == MODEL_PATH

# dataset is fetched with the correct Kaggle slug
def test_kaggle_slug_correct(kaggle_dir):
    mock_dl, _, _ = run_no_model(kaggle_dir)
    mock_dl.assert_called_once_with("aryashah2k/breast-ultrasound-images-dataset")


# -------------------------
# Skip-training path
# -------------------------

# full pipeline runs without raising when a checkpoint is present
def test_with_model_runs_without_crash(kaggle_dir):
    run_with_model(kaggle_dir)

# no new checkpoint is written when a saved model is loaded
def test_torch_save_not_called(kaggle_dir):
    _, mock_save = run_with_model(kaggle_dir)
    mock_save.assert_not_called()

# existing checkpoint is loaded from disk
def test_load_checkpoint_called(kaggle_dir):
    mock_load_checkpoint, _ = run_with_model(kaggle_dir)
    assert mock_load_checkpoint.called

# checkpoint loader receives the expected path
def test_load_checkpoint_uses_model_path(kaggle_dir):
    mock_load_checkpoint, _ = run_with_model(kaggle_dir)
    assert mock_load_checkpoint.call_args.kwargs["model_path"] == MODEL_PATH


# -------------------------
# Data flow
# -------------------------

# BUSIDataset is constructed with <download_path>/Dataset_BUSI_with_GT
def test_busi_dataset_receives_correct_path(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False), \
         patch("ultrasound_tumor_detection.pipeline.torch.save"), \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1), \
         patch("ultrasound_tumor_detection.pipeline.BUSIDataset", wraps=main_module.BUSIDataset) as mock_ds:
        main()
    assert mock_ds.call_args[0][0] == os.path.join(kaggle_dir, "Dataset_BUSI_with_GT")

# dataset corruption and failure must raise
def test_dataset_failure_raises(kaggle_dir):
    empty_root = kaggle_dir
    shutil.rmtree(os.path.join(empty_root, "Dataset_BUSI_with_GT"))
    os.makedirs(os.path.join(empty_root, "Dataset_BUSI_with_GT"))

    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=empty_root), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False):

        with pytest.raises(ValueError):
            main()


def test_corrupt_checkpoint_raises(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", side_effect=RuntimeError):

        with pytest.raises(RuntimeError):
            main()
