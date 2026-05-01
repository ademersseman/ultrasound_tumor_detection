"""
Model architecture tests — verify tensor flow and parameter structure.
"""

import os
import shutil
import tempfile

import pytest
import torch

from main import UNet

class TestUNetArchitecture:
    # output spatial dims check
    def test_output_shape_matches_input(self, model):
        x = torch.zeros(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    # output must be in [0,1]
    def test_output_is_probability(self, model):
        # sigmoid final layer guarantees [0, 1]
        x = torch.rand(2, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    # cannot be nan or inf in output
    def test_no_nan_or_inf(self, model):
        x = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    # test edge case: extreme inputs 
    def test_extreme_input_values(self, model):
        x = torch.full((1, 1, 256, 256), 1e6)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    # must support varying batch sizes
    @pytest.mark.parametrize("batch", [1, 2, 8])
    def test_batch_size_agnostic(self, model, batch):
        x = torch.rand(batch, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == batch

    # output probs must vary with input
    def test_output_varies_with_input(self, model):
        x1 = torch.zeros(1, 1, 256, 256)
        x2 = torch.ones(1, 1, 256, 256)
        with torch.no_grad():
            o1, o2 = model(x1), model(x2)
        assert not torch.allclose(o1, o2)
    
    # spatial dims divisible by 8 should work 
    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_valid_resolutions(self, size):
        m = UNet().eval()
        x = torch.rand(1, 1, size, size)
        with torch.no_grad():
            out = m(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    # invalid spatial size should fail 
    def test_invalid_resolution_raises(self):
        m = UNet().eval()
        x = torch.rand(1, 1, 250, 250) 

        with pytest.raises(RuntimeError):
            _ = m(x)
    
    # eval mode should lead to deterministic outputs
    def test_eval_mode_deterministic(self):
        m = UNet().eval()
        x = torch.rand(1, 1, 256, 256)

        with torch.no_grad():
            o1 = m(x)
            o2 = m(x)

        assert torch.allclose(o1, o2)

    # train mode, dropout should introduce variation
    def test_train_mode_not_deterministic(self):
        m = UNet().train()
        x = torch.rand(1, 1, 256, 256)

        o1 = m(x)
        o2 = m(x)

        assert not torch.allclose(o1, o2)

class TestUNetPersistence:
    def test_save_and_load_produces_identical_output(self, model):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "unet.pth")
        try:
            torch.save(model.state_dict(), path)

            loaded = UNet()
            loaded.load_state_dict(torch.load(path, map_location="cpu"))
            loaded.eval()

            x = torch.rand(1, 1, 256, 256)
            with torch.no_grad():
                out_orig = model(x)
                out_loaded = loaded(x)

            assert torch.allclose(out_orig, out_loaded)
        finally:
            shutil.rmtree(tmp)
