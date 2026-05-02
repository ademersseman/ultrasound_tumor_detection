"""
Training step tests  --- check for gradients updating, model ability to fit, etc. 
"""

import pytest
import torch
import torch.optim as optim

from ultrasound_tumor_detection.data import BUSIDataset
from ultrasound_tumor_detection.losses import bce_dice_loss
from ultrasound_tumor_detection.model import UNet


class TestTrainingStepMechanics:

    # optimizer should cause weight updates
    def test_weights_update_after_optimizer_step(self):
        model = UNet().train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        x = torch.rand(1, 1, 256, 256)
        target = torch.rand(1, 1, 256, 256)

        before = model.enc1[0].weight.data.clone()
        loss = bce_dice_loss(model(x), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.equal(before, model.enc1[0].weight.data)

    #params should remain finite after update
    def test_weights_remain_finite_after_step(self):
        model = UNet().train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        x = torch.rand(2, 1, 256, 256)
        target = torch.rand(2, 1, 256, 256)

        loss = bce_dice_loss(model(x), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            assert torch.isfinite(p).all()

    #grads should exist, be valid
    def test_gradients_exist_and_finite(self):
        model = UNet().train()
        x = torch.rand(1, 1, 256, 256)
        target = torch.rand(1, 1, 256, 256)

        bce_dice_loss(model(x), target).backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]

        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum().item() > 0 for g in grads)

    # testing if zero grad works
    def test_zero_grad_clears_all_gradients(self):
        model = UNet().train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        x, target = torch.rand(1, 1, 256, 256), torch.rand(1, 1, 256, 256)

        bce_dice_loss(model(x), target).backward()
        optimizer.zero_grad()

        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum().item() == 0.0

class TestReproducibility:
    # the same seed should produce the same output
    def test_fixed_seed_produces_same_output(self):
        def run(seed):
            torch.manual_seed(seed)
            m = UNet().eval()
            torch.manual_seed(seed)
            x = torch.rand(1, 1, 256, 256)
            with torch.no_grad():
                return m(x)

        assert torch.allclose(run(42), run(42))

    # different seeds should produce different outputs
    def test_different_seed_different_output(self):
        torch.manual_seed(1)
        m1 = UNet().eval()

        torch.manual_seed(2)
        m2 = UNet().eval()

        x = torch.rand(1, 1, 256, 256)

        with torch.no_grad():
            o1 = m1(x)
            o2 = m2(x)

        assert not torch.allclose(o1, o2)

class TestIntegration:
    # testing full pipeline
    def test_two_training_batches_no_crash(self, fake_dataset_dir):
        ds = BUSIDataset(fake_dataset_dir)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

        model = UNet().train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for i, (imgs, masks) in enumerate(loader):
            loss = bce_dice_loss(model(imgs), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i >= 1:
                break
    
    # model overfits a super tiny dataset
    def test_overfit_tiny_dataset_loss_decreases(self, fake_dataset_dir):
        torch.manual_seed(69)
        ds = BUSIDataset(fake_dataset_dir)
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)

        model = UNet().train()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        imgs, masks = next(iter(loader))

        initial_loss = bce_dice_loss(model(imgs), masks).item()
        for _ in range(15):
            loss = bce_dice_loss(model(imgs), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_loss = bce_dice_loss(model(imgs), masks).item()

        assert final_loss < initial_loss, (
            f"Bug -- loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}. "
        )
