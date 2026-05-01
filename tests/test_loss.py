"""
Loss function tests — known inputs → known outputs, edge cases.
Especially important for medical segmentation where loss shape matters.
"""

import pytest
import torch

from main import bce_dice_loss


class TestBceDiceLossShape:
    def test_returns_scalar(self):
        pred, target = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
        assert bce_dice_loss(pred, target).shape == ()

    def test_no_nan(self):
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32).round()
        assert not torch.isnan(bce_dice_loss(pred, target))


class TestBceDiceLossValues:
    def test_non_negative(self):
        pred, target = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
        assert bce_dice_loss(pred, target).item() >= 0.0

    def test_perfect_prediction_lower_than_random(self):
        target = torch.rand(2, 1, 32, 32)
        perfect_loss = bce_dice_loss(target.clone(), target)
        random_loss = bce_dice_loss(torch.rand_like(target), target)
        assert perfect_loss.item() < random_loss.item()

    def test_loss_increases_as_prediction_worsens(self):
        target = torch.ones(1, 1, 16, 16) * 0.9
        close_pred = torch.ones(1, 1, 16, 16) * 0.8
        far_pred = torch.ones(1, 1, 16, 16) * 0.1
        assert bce_dice_loss(close_pred, target).item() < bce_dice_loss(far_pred, target).item()

    def test_worst_case_all_zeros_vs_all_ones_is_high(self):
        # all-ones target, all-zeros prediction — BCE alone drives this very high
        target = torch.ones(1, 1, 16, 16)
        pred = torch.zeros(1, 1, 16, 16)
        assert bce_dice_loss(pred, target).item() > 5.0


class TestBceDiceLossGradients:
    def test_loss_is_differentiable(self):
        pred = torch.rand(1, 1, 16, 16, requires_grad=True)
        target = torch.rand(1, 1, 16, 16)
        loss = bce_dice_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
