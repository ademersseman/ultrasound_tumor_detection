"""Dataset utilities for BUSI breast ultrasound segmentation data."""

import os

import cv2
import kagglehub
import torch
from torch.utils.data import Dataset


BUSI_KAGGLE_DATASET = "aryashah2k/breast-ultrasound-images-dataset"
BUSI_FOLDER_NAME = "Dataset_BUSI_with_GT"
BUSI_CLASSES = ("benign", "malignant", "normal")


def download_busi_dataset(dataset_slug: str = BUSI_KAGGLE_DATASET) -> str:
    """Download the BUSI dataset and return the image/mask folder path."""
    path = kagglehub.dataset_download(dataset_slug)
    return os.path.join(path, BUSI_FOLDER_NAME)


class BUSIDataset(Dataset):
    """PyTorch dataset for BUSI image and segmentation-mask pairs."""

    def __init__(self, root_dir):
        self.image_paths = []
        self.mask_paths = []
        skipped_orphans = 0
        skipped_corrupt = 0
        total_examples = 0

        for cls in BUSI_CLASSES:
            folder = os.path.join(root_dir, cls)
            if not os.path.isdir(folder):
                continue

            for file in os.listdir(folder):
                if "_mask" in file:
                    continue

                img_path = os.path.join(folder, file)
                mask_path = img_path.replace(".png", "_mask.png")

                if not os.path.exists(mask_path):
                    skipped_orphans += 1
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if img is None or mask is None:
                    skipped_corrupt += 1
                    continue

                if img.shape != mask.shape:
                    skipped_corrupt += 1
                    continue

                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                total_examples += 1

        if len(self.image_paths) != total_examples or len(self.image_paths) != len(self.mask_paths):
            raise ValueError("BUSIDataset: Unexpected parsing error.")
        if len(self.image_paths) == 0:
            raise ValueError("BUSIDataset: No valid image-mask pairs found after filtering")

        print(f"Orphans: {skipped_orphans} \n Corrupt: {skipped_corrupt}")
        print(f"Total: {total_examples}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Failure to read index {idx}: ")

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask
