"""
Data pipeline tests. 
Tests that the dataset (BUSI) 
This file unit tests with the synthetic testing dataset from conftest.py.
Tests include dataset loading, expected shape, mask/image shape alignment,
datatype and values for different unexpected input types in dataset.
"""

import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from main import BUSIDataset


# -------------------------
# Fixtures
# -------------------------
# Regular image dataset
@pytest.fixture
def dataset(fake_dataset_dir):
    return BUSIDataset(fake_dataset_dir)

# Has image with no mask
@pytest.fixture
def dataset_with_orphan(fake_dataset_dir):
    orphan = os.path.join(fake_dataset_dir, "benign", "orphan.png")
    cv2.imwrite(orphan, np.zeros((64, 64), dtype=np.uint8))
    return BUSIDataset(fake_dataset_dir), orphan

# Has corrupt image
@pytest.fixture
def dataset_with_corrupt(fake_dataset_dir):
    corrupt_path = os.path.join(fake_dataset_dir, "benign", "corrupt.png")
    with open(corrupt_path, "w") as f:
        f.write("not an image")

    return BUSIDataset(fake_dataset_dir), corrupt_path

# Helper
def iterate_samples(dataset):
    for i in range(len(dataset)):
        yield dataset[i]


# -------------------------
# Shape and datatype invariants
# -------------------------

@pytest.mark.parametrize("idx", [0, 1])
def test_sample_shape(dataset, idx):
    img, mask = dataset[idx]
    assert img.ndim == 3
    assert mask.ndim == 3
    assert img.shape == mask.shape

     # grayscale channel
    assert img.shape[0] == 1 


@pytest.mark.parametrize("idx", [0, 1])
def test_dtype(dataset, idx):
    img, mask = dataset[idx]
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32

# -------------------------
# Value invariants
# -------------------------
@pytest.mark.parametrize("idx", [0, 1])
def test_image_range(dataset, idx):
    img, _ = dataset[idx]
    assert torch.isfinite(img).all()
    assert 0.0 <= img.min() <= 1.0
    assert 0.0 <= img.max() <= 1.0


@pytest.mark.parametrize("idx", [0, 1])
def test_mask_range_and_binary(dataset, idx):
    _, mask = dataset[idx]
    assert torch.isfinite(mask).all()
    assert 0.0 <= mask.min() <= 1.0
    assert 0.0 <= mask.max() <= 1.0

    unique_vals = torch.unique(mask)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})

# -------------------------
# Dataset-level invariants
# -------------------------
def test_non_empty(dataset):
    assert len(dataset) > 0

def test_all_samples_valid(dataset):
    for img, mask in iterate_samples(dataset):
        assert img.shape == mask.shape
        assert not torch.isnan(img).any()
        assert not torch.isnan(mask).any()
# -------------------------
# Alignment correctness
# -------------------------

def test_image_mask_alignment(dataset):
    if hasattr(dataset, "image_paths") and hasattr(dataset, "mask_paths"):
        for img_path, mask_path in zip(dataset.image_paths, dataset.mask_paths):
            img_name = os.path.basename(img_path).replace(".png", "")
            mask_name = os.path.basename(mask_path).replace("_mask.png", "")
            assert img_name == mask_name

# -------------------------
# Orphan handling
# ------------------------
def test_orphan_image_skipped(dataset_with_orphan):
    ds, orphan = dataset_with_orphan

    if hasattr(ds, "image_paths"):
        assert orphan not in ds.image_paths

    for img, mask in iterate_samples(ds):
        assert img is not None
        assert mask is not None

# -------------------------
# Corrupt file handling
# -------------------------
def test_corrupt_file_skipped(dataset_with_corrupt):
    ds, corrupt_path = dataset_with_corrupt

    if hasattr(ds, "image_paths"):
        assert corrupt_path not in ds.image_paths

    for img, mask in ds:
        assert img is not None
        assert mask is not None

# -------------------------
# DataLoader compatibility testing
# -------------------------
def test_dataloader_batching(dataset):
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    imgs, masks = next(iter(loader))

    assert imgs.shape[0] == 2
    assert imgs.shape == masks.shape

def test_deterministic_access(dataset):
    img1, mask1 = dataset[0]
    img2, mask2 = dataset[0]

    assert torch.allclose(img1, img2)
    assert torch.allclose(mask1, mask2)