"""
Loss function and dice tests — ensure correctness on known cases, numeric stability
Includes different edge cases for bce/dice loss and dice score
"""

import pytest
import torch

from ultrasound_tumor_detection.losses import bce_dice_loss, dice_score

# -------------------------
# Dice Score Tests
# -------------------------
class TestDiceScoreCorrectness:
    # test perfect prediction
    def test_perfect_overlap(self):
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()
        score = dice_score(target, target)
        assert score.item() == pytest.approx(1.0, abs=1e-5)

    # test that if masks are disjoint, dice is near 0
    def test_zero_overlap(self):
        target = torch.ones(1, 1, 16, 16)
        pred = torch.zeros_like(target)
        score = dice_score(pred, target)
        assert score.item() < 0.1
    
    # if everything is 0 dice = 1 (perfect prediction edge case)
    def test_all_zero_case(self):
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.zeros(1, 1, 8, 8)

        score = dice_score(pred, target)
        assert torch.isfinite(score)
        assert score.item() == pytest.approx(1.0, abs=1e-5)
    
    # test for range of dice score in [0,1]
    def test_output_in_zero_one_range(self):
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32).round()
        score = dice_score(pred, target).item()
        assert 0.0 <= score <= 1.0

    # test that increasing threshold should reduce our positives
    def test_threshold_effect(self):
        target = torch.ones(1, 1, 8, 8)
        pred = torch.full_like(target, 0.6)

        low_thresh = dice_score(pred, target, threshold=0.5)
        high_thresh = dice_score(pred, target, threshold=0.7)

        assert low_thresh > high_thresh
    
    # dice(a,b) = dice(b,a)
    def test_symmetry(self):
        pred = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 16, 16)

        s1 = dice_score(pred, target)
        s2 = dice_score(target, pred)

        assert abs(s1.item() - s2.item()) < 1e-5

    # test that our input shapes should raise errors if incorrect
    def test_invalid_inputs_raise(self):
        pred = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 8, 8)
        
        with pytest.raises(ValueError):
            dice_score(pred, target)


# -------------------------
# Loss Tests
# -------------------------
class TestBceDiceLossShape:
    # must be scalar
    def test_returns_scalar(self):
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)

        loss = bce_dice_loss(pred, target)
        assert loss.ndim == 0

    # must not be nan
    def test_no_nan(self):
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32).round()
        assert not torch.isnan(bce_dice_loss(pred, target))

    # input shape mismatch raises error
    def test_invalid_inputs_raise(self):
        pred = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 8, 8)

        with pytest.raises(ValueError):
            bce_dice_loss(pred, target)

class TestBceDiceLossValues:
    # must not be negative
    def test_non_negative(self):
        pred, target = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
        assert bce_dice_loss(pred, target).item() >= 0.0

    # must be finite
    def test_loss_finite(self):
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)

        loss = bce_dice_loss(pred, target)
        assert torch.isfinite(loss)

    # edge case with all 0s (testing smoothing)
    def test_all_zero_edge_case(self):
        pred = torch.zeros(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)

        loss = bce_dice_loss(pred, target)
        assert torch.isfinite(loss)
    
    # perfect prediction loss < random prediction loss
    def test_perfect_prediction_lower_than_random(self):
        target = torch.rand(2, 1, 32, 32)
        perfect_loss = bce_dice_loss(target.clone(), target)
        random_loss = bce_dice_loss(torch.rand_like(target), target)
        assert perfect_loss.item() < random_loss.item()

    # loss reflects prediction quality 
    def test_loss_increases_as_prediction_worsens(self):
        target = torch.ones(1, 1, 16, 16) * 0.9
        close_pred = torch.ones(1, 1, 16, 16) * 0.8
        far_pred = torch.ones(1, 1, 16, 16) * 0.1
        assert bce_dice_loss(close_pred, target).item() < bce_dice_loss(far_pred, target).item()

    # edge case with loss reflecting prediction quality - worst case
    def test_worst_case_all_zeros_vs_all_ones_is_high(self):
        target = torch.ones(1, 1, 16, 16)
        pred = torch.zeros(1, 1, 16, 16)
        assert bce_dice_loss(pred, target).item() > 5.0
    
    
class TestBceDiceLossGradients:
    # loss must be differentiable
    def test_loss_is_differentiable(self):
        pred = torch.rand(1, 1, 16, 16, requires_grad=True)
        target = torch.rand(1, 1, 16, 16)
        loss = bce_dice_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).all()



