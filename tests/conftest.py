"""
Configures mock dataset for unit testing.
"""

import os
import sys
import shutil
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import BUSIDataset, UNet

def make_fake_dataset(root: str, n_per_class: int = 2) -> None:
    for cls in ["benign", "malignant", "normal"]:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(1, n_per_class + 1):
            img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
            mask = (img > 128).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(cls_dir, f"image ({i}).png"), img)
            cv2.imwrite(os.path.join(cls_dir, f"image ({i})_mask.png"), mask)


@pytest.fixture
def fake_dataset_dir():
    tmp = tempfile.mkdtemp()
    make_fake_dataset(tmp, n_per_class=2)
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def dataset(fake_dataset_dir):
    return BUSIDataset(fake_dataset_dir)


@pytest.fixture
def model():
    return UNet().eval()
