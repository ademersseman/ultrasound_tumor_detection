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
from ultrasound_tumor_detection.pipeline import MODEL_PATH, load_checkpoint, main
from ultrasound_tumor_detection.model import UNet


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
         patch("ultrasound_tumor_detection.pipeline.visualize_predictions") as mock_viz, \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1):
        main()
    return mock_dl, mock_load_checkpoint, mock_save, mock_viz

# run main() with an existing saved model (skip training, use model)
def run_with_model(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=True) as mock_load_checkpoint, \
         patch("ultrasound_tumor_detection.pipeline.torch.save") as mock_save, \
         patch("ultrasound_tumor_detection.pipeline.visualize_predictions") as mock_viz:
        main()
    return mock_load_checkpoint, mock_save, mock_viz


# -------------------------
# Training path (no saved model)
# -------------------------

# full pipeline runs without raising
def test_no_model_runs_without_crash(kaggle_dir):
    run_no_model(kaggle_dir)

# model weights are written to disk after training
def test_torch_save_called_once(kaggle_dir):
    _, _, mock_save, _ = run_no_model(kaggle_dir)
    assert mock_save.call_count == 1

# weights are saved under the expected filename
def test_model_saved_to_correct_path(kaggle_dir):
    _, _, mock_save, _ = run_no_model(kaggle_dir)
    assert mock_save.call_args[0][1] == MODEL_PATH

# dataset is fetched with the correct Kaggle slug
def test_kaggle_slug_correct(kaggle_dir):
    mock_dl, _, _, _ = run_no_model(kaggle_dir)
    mock_dl.assert_called_once_with("aryashah2k/breast-ultrasound-images-dataset")

# visualize_predictions is called once at the end
def test_no_model_visualize_called(kaggle_dir):
    _, _, _, mock_viz = run_no_model(kaggle_dir)
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
def test_load_checkpoint_called(kaggle_dir):
    mock_load_checkpoint, _, _ = run_with_model(kaggle_dir)
    assert mock_load_checkpoint.called

# checkpoint loader receives the expected path
def test_load_checkpoint_uses_model_path(kaggle_dir):
    mock_load_checkpoint, _, _ = run_with_model(kaggle_dir)
    assert mock_load_checkpoint.call_args.kwargs["model_path"] == MODEL_PATH

# visualize_predictions is called once even without training
def test_with_model_visualize_called(kaggle_dir):
    _, _, mock_viz = run_with_model(kaggle_dir)
    assert mock_viz.call_count == 1

# -------------------------
# Data flow
# -------------------------

# BUSIDataset is constructed with <download_path>/Dataset_BUSI_with_GT
def test_busi_dataset_receives_correct_path(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False), \
         patch("ultrasound_tumor_detection.pipeline.torch.save"), \
         patch("ultrasound_tumor_detection.pipeline.visualize_predictions"), \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1), \
         patch("ultrasound_tumor_detection.pipeline.BUSIDataset", wraps=main_module.BUSIDataset) as mock_ds:
        main()
    assert mock_ds.call_args[0][0] == os.path.join(kaggle_dir, "Dataset_BUSI_with_GT")

# tensors passed to visualize_predictions are on the CPU
def test_visualize_receives_cpu_tensors(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False), \
         patch("ultrasound_tumor_detection.pipeline.torch.save"), \
         patch("ultrasound_tumor_detection.pipeline.visualize_predictions") as mock_viz, \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1):
        main()
    imgs, masks, preds = mock_viz.call_args[0]
    assert imgs.device.type == "cpu"
    assert masks.device.type == "cpu"
    assert preds.device.type == "cpu"

# all three tensors are 4-D (B, C, H, W)
def test_visualize_receives_4d_tensors(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False), \
         patch("ultrasound_tumor_detection.pipeline.torch.save"), \
         patch("ultrasound_tumor_detection.pipeline.visualize_predictions") as mock_viz, \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1):
        main()
    imgs, masks, preds = mock_viz.call_args[0]
    assert imgs.ndim == 4
    assert masks.ndim == 4
    assert preds.ndim == 4

# imgs, masks, preds share the same shape
def test_visualize_tensors_have_matching_shapes(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False), \
         patch("ultrasound_tumor_detection.pipeline.torch.save"), \
         patch("ultrasound_tumor_detection.pipeline.visualize_predictions") as mock_viz, \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1):
        main()
    imgs, masks, preds = mock_viz.call_args[0]
    assert imgs.shape == masks.shape == preds.shape

# images are single-channel (grayscale)
def test_visualize_single_channel_images(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False), \
         patch("ultrasound_tumor_detection.pipeline.torch.save"), \
         patch("ultrasound_tumor_detection.pipeline.visualize_predictions") as mock_viz, \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1):
        main()
    imgs, _, _ = mock_viz.call_args[0]
    assert imgs.shape[1] == 1

# model outputs raw logits, which should remain finite
def test_prediction_values_are_finite(kaggle_dir):
    with patch("ultrasound_tumor_detection.data.kagglehub.dataset_download", return_value=kaggle_dir), \
         patch("ultrasound_tumor_detection.pipeline.load_checkpoint", return_value=False), \
         patch("ultrasound_tumor_detection.pipeline.torch.save"), \
         patch("ultrasound_tumor_detection.pipeline.visualize_predictions") as mock_viz, \
         patch("ultrasound_tumor_detection.pipeline.EPOCHS", 1):
        main()
    _, _, preds = mock_viz.call_args[0]
    assert torch.isfinite(preds).all()

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


# -------------------------
# load_checkpoint unit tests
# -------------------------

class TestLoadCheckpoint:
    def test_loads_local_checkpoint_when_present(self, tmp_path):
        model = UNet()
        ckpt_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), ckpt_path)

        fresh = UNet()
        result = load_checkpoint(fresh, torch.device("cpu"), model_path=str(ckpt_path))
        assert result is True

    def test_returns_false_when_no_local_and_bundled_disallowed(self, tmp_path):
        model = UNet()
        missing = str(tmp_path / "nonexistent.pth")
        result = load_checkpoint(model, torch.device("cpu"), model_path=missing, allow_bundled=False)
        assert result is False

    def test_returns_false_when_nothing_exists(self, tmp_path):
        model = UNet()
        missing = str(tmp_path / "nonexistent.pth")
        with patch("ultrasound_tumor_detection.pipeline.bundled_model_resource") as mock_res:
            mock_res.return_value.is_file.return_value = False
            result = load_checkpoint(model, torch.device("cpu"), model_path=missing, allow_bundled=True)
        assert result is False

    def test_loads_bundled_when_no_local(self, tmp_path):
        model = UNet()
        state = model.state_dict()
        bundled_file = tmp_path / "bundled.pth"
        torch.save(state, bundled_file)

        missing = str(tmp_path / "nonexistent.pth")
        with patch("ultrasound_tumor_detection.pipeline.bundled_model_resource") as mock_res:
            mock_res.return_value.is_file.return_value = True
            mock_res.return_value.__enter__ = lambda s: None
            mock_res.return_value.__exit__ = lambda s, *a: None
            with patch("ultrasound_tumor_detection.pipeline.as_file") as mock_as_file, \
                 patch("ultrasound_tumor_detection.pipeline.torch.load", return_value=state) as mock_tload:
                mock_as_file.return_value.__enter__ = lambda s: bundled_file
                mock_as_file.return_value.__exit__ = lambda s, *a: None
                result = load_checkpoint(model, torch.device("cpu"), model_path=missing, allow_bundled=True)

        assert result is True

    def test_local_checkpoint_updates_model_weights(self, tmp_path):
        source = UNet().eval()
        ckpt_path = tmp_path / "model.pth"
        torch.save(source.state_dict(), ckpt_path)

        target = UNet()
        load_checkpoint(target, torch.device("cpu"), model_path=str(ckpt_path))
        target.eval()

        x = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            assert torch.allclose(source(x), target(x))
