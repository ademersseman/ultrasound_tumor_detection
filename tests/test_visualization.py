"""
Visualization tests.
Tests the visualize_predictions function, which renders one three-panel
matplotlib figure per sample (image | ground-truth mask | thresholded prediction).
All tests mock matplotlib.pyplot so no display window is opened.
Tests cover call counts, figure layout, data shape, sigmoid + threshold logic,
and per-sample panel ordering.
"""

from unittest.mock import patch

import pytest
import torch

from main import visualize_predictions

THRESHOLD = 0.65


# -------------------------
# Fixtures
# -------------------------

# Two-sample batch of random tensors (logits for preds)
@pytest.fixture
def batch():
    torch.manual_seed(0)
    imgs  = torch.rand(2, 1, 256, 256)
    masks = torch.randint(0, 2, (2, 1, 256, 256)).float()
    preds = torch.randn(2, 1, 256, 256)
    return imgs, masks, preds

# Single-sample batch
@pytest.fixture
def single():
    torch.manual_seed(1)
    imgs  = torch.rand(1, 1, 256, 256)
    masks = torch.randint(0, 2, (1, 1, 256, 256)).float()
    preds = torch.randn(1, 1, 256, 256)
    return imgs, masks, preds


# -------------------------
# Figure / show call counts
# -------------------------

# one plt.figure() call per sample in the batch
def test_one_figure_per_sample(batch):
    imgs, masks, preds = batch
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    assert mock_plt.figure.call_count == len(imgs)

# plt.show() fires exactly once per sample
def test_show_called_once_per_sample(batch):
    imgs, masks, preds = batch
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    assert mock_plt.show.call_count == len(imgs)

# three imshow calls per sample: image, mask, prediction
def test_three_imshow_calls_per_sample(batch):
    imgs, masks, preds = batch
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    assert mock_plt.imshow.call_count == 3 * len(imgs)

# three subplot calls per sample (one per panel)
def test_three_subplot_calls_per_sample(batch):
    imgs, masks, preds = batch
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    assert mock_plt.subplot.call_count == 3 * len(imgs)

# three title calls per sample
def test_three_title_calls_per_sample(batch):
    imgs, masks, preds = batch
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    assert mock_plt.title.call_count == 3 * len(imgs)


# -------------------------
# Figure layout
# -------------------------

# figure is created with the expected dimensions
def test_figsize(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    mock_plt.figure.assert_called_with(figsize=(10, 3))

# subplots are arranged in a 1×3 grid at indices 1, 2, 3
def test_subplot_indices(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    subplot_calls = [c.args for c in mock_plt.subplot.call_args_list]
    assert (1, 3, 1) in subplot_calls
    assert (1, 3, 2) in subplot_calls
    assert (1, 3, 3) in subplot_calls

# panels are titled Image, Ground Truth, Prediction in order
def test_panel_titles(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    titles = [c.args[0] for c in mock_plt.title.call_args_list]
    assert titles == ["Image", "Ground Truth", "Prediction"]

# all three panels use the grayscale colormap
def test_all_panels_use_gray_colormap(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    for c in mock_plt.imshow.call_args_list:
        assert c.kwargs.get("cmap") == "gray"


# -------------------------
# Data passed to imshow
# -------------------------

# image panel receives a 2-D tensor (channel dim squeezed)
def test_image_panel_is_2d(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    img_data = mock_plt.imshow.call_args_list[0].args[0]
    assert img_data.ndim == 2

# image panel spatial size matches 256×256 input
def test_image_panel_shape_matches_input(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    img_data = mock_plt.imshow.call_args_list[0].args[0]
    assert img_data.shape == (256, 256)

# mask panel receives a 2-D tensor
def test_mask_panel_is_2d(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    mask_data = mock_plt.imshow.call_args_list[1].args[0]
    assert mask_data.ndim == 2

# prediction panel is a boolean tensor after sigmoid + threshold
def test_prediction_panel_is_boolean(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    pred_data = mock_plt.imshow.call_args_list[2].args[0]
    assert pred_data.dtype == torch.bool

# prediction panel contains only True / False
def test_prediction_panel_values_are_binary(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    pred_data = mock_plt.imshow.call_args_list[2].args[0]
    unique = set(pred_data.flatten().tolist())
    assert unique.issubset({True, False})

# prediction panel is 2-D
def test_prediction_panel_is_2d(single):
    imgs, masks, preds = single
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    pred_data = mock_plt.imshow.call_args_list[2].args[0]
    assert pred_data.ndim == 2


# -------------------------
# Sigmoid + threshold behaviour
# -------------------------

# very large positive logit (>>0) sigmoid to ~1, exceeds threshold → all True
def test_large_positive_logit_maps_to_true(single):
    imgs, masks, _ = single
    preds = torch.full((1, 1, 256, 256), 10.0)
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    pred_data = mock_plt.imshow.call_args_list[2].args[0]
    assert pred_data.all()

# very large negative logit (<<0) sigmoids to ~0, below threshold → all False
def test_large_negative_logit_maps_to_false(single):
    imgs, masks, _ = single
    preds = torch.full((1, 1, 256, 256), -10.0)
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    pred_data = mock_plt.imshow.call_args_list[2].args[0]
    assert not pred_data.any()

# logit just above the 0.65 boundary maps to True; just below maps to False
def test_default_threshold_is_0_65(single):
    imgs, masks, _ = single
    logit_at_boundary = torch.log(torch.tensor(0.65 / 0.35))
    just_above = logit_at_boundary + 0.01
    just_below = logit_at_boundary - 0.01

    preds_above = torch.full((1, 1, 256, 256), just_above.item())
    preds_below = torch.full((1, 1, 256, 256), just_below.item())

    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds_above)
    data_above = mock_plt.imshow.call_args_list[2].args[0]

    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds_below)
    data_below = mock_plt.imshow.call_args_list[2].args[0]

    assert data_above.all()
    assert not data_below.any()

# custom threshold parameter is respected
def test_custom_threshold_respected(single):
    imgs, masks, _ = single
    preds = torch.zeros(1, 1, 256, 256)   # sigmoid(0) = 0.5

    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds, threshold=0.49)
    data_low = mock_plt.imshow.call_args_list[2].args[0]

    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds, threshold=0.51)
    data_high = mock_plt.imshow.call_args_list[2].args[0]

    assert data_low.all()
    assert not data_high.any()

# sigmoid is applied — extreme logit values still produce a valid boolean mask
def test_sigmoid_applied_not_raw_logits():
    imgs  = torch.rand(1, 1, 256, 256)
    masks = torch.zeros(1, 1, 256, 256)
    preds = torch.randn(1, 1, 256, 256) * 10

    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    pred_data = mock_plt.imshow.call_args_list[2].args[0]
    assert pred_data.dtype == torch.bool


# -------------------------
# Panel ordering across samples
# -------------------------

# each figure's image panel matches the corresponding batch element
def test_correct_image_data_per_sample(batch):
    imgs, masks, preds = batch
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    calls = mock_plt.imshow.call_args_list
    for i in range(len(imgs)):
        img_data = calls[i * 3].args[0]
        assert torch.allclose(img_data, imgs[i].squeeze())

# each figure's mask panel matches the corresponding batch element
def test_correct_mask_data_per_sample(batch):
    imgs, masks, preds = batch
    with patch("main.plt") as mock_plt:
        visualize_predictions(imgs, masks, preds)
    calls = mock_plt.imshow.call_args_list
    for i in range(len(masks)):
        mask_data = calls[i * 3 + 1].args[0]
        assert torch.allclose(mask_data, masks[i].squeeze())
