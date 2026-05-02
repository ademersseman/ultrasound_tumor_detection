"""Losses and segmentation metrics."""

import torch
import torch.nn.functional as F


def bce_dice_loss(pred, target, smooth=1):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    if pred.numel() == 0:
        raise ValueError("Empty tensors passed to loss")

    pred = pred.clamp(1e-7, 1 - 1e-7)
    bce = F.binary_cross_entropy(pred, target)
    intersection = (pred * target).sum()
    dice = 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return bce + dice


def dice_score(pred, target, smooth=1, threshold=0.65):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    if pred.numel() == 0:
        raise ValueError("Empty tensors")

    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
