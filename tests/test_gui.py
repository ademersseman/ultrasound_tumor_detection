"""
Tests the GUI widgets in __main__.py — widget state, confidence metrics,
zoom/pan interactions, image loading, and tab management.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PyQt5.QtWidgets import QApplication

from ultrasound_tumor_detection.__main__ import ImageAnalysisTab, UltrasoundGUI


# -------------------------
# Helpers
# -------------------------

def _make_mock_model():
    """Return a model mock whose __call__ returns a zero-logit tensor."""
    model = MagicMock()
    model.return_value = torch.zeros(1, 1, 256, 256)
    model.to.return_value = model
    model.eval.return_value = model
    return model


class _Event:
    """Minimal stand-in for a matplotlib MouseEvent."""
    def __init__(self, button=1, xdata=10.0, ydata=20.0):
        self.button = button
        self.xdata = xdata
        self.ydata = ydata


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def image_data():
    np.random.seed(0)
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    return {'image': img, 'name': 'test.png'}


@pytest.fixture
def mock_model():
    return _make_mock_model()


@pytest.fixture
def tab(qapp, mock_model, image_data):
    return ImageAnalysisTab(mock_model, torch.device('cpu'), image_data, tab_index=1)


@pytest.fixture
def gui(qapp):
    mock_mdl = _make_mock_model()
    with patch('ultrasound_tumor_detection.__main__.UNet', return_value=mock_mdl), \
         patch('ultrasound_tumor_detection.__main__.load_checkpoint', return_value=True):
        g = UltrasoundGUI()
    return g


# -------------------------
# ImageAnalysisTab — widget creation
# -------------------------

class TestImageAnalysisTabWidgets:
    def test_conf_label_created(self, tab):
        assert tab.conf_label is not None

    def test_tumor_label_created(self, tab):
        assert tab.tumor_label is not None

    def test_pixels_label_created(self, tab):
        assert tab.pixels_label is not None

    def test_stats_label_created(self, tab):
        assert tab.stats_label is not None

    def test_reset_btn_created(self, tab):
        assert tab.reset_btn is not None

    def test_orig_canvas_created(self, tab):
        assert tab.orig_canvas is not None

    def test_heatmap_canvas_created(self, tab):
        assert tab.heatmap_canvas is not None

    def test_title_label_includes_index_and_name(self, qapp, mock_model):
        np.random.seed(1)
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        data = {'image': img, 'name': 'scan42.png'}
        t = ImageAnalysisTab(mock_model, torch.device('cpu'), data, tab_index=42)
        from PyQt5.QtWidgets import QLabel
        label_texts = [lbl.text() for lbl in t.findChildren(QLabel)]
        assert any("42" in txt and "scan42.png" in txt for txt in label_texts)

    def test_panning_starts_false(self, tab):
        assert tab.panning is False

    def test_current_pred_raw_populated_after_init(self, tab):
        # predict() is called during initUI
        assert tab.current_pred_raw is not None

    def test_current_pred_raw_is_numpy(self, tab):
        assert isinstance(tab.current_pred_raw, np.ndarray)

    def test_current_pred_raw_shape(self, tab):
        assert tab.current_pred_raw.shape == (256, 256)


# -------------------------
# ImageAnalysisTab — predict
# -------------------------

class TestPredict:
    def test_predict_populates_pred_raw(self, tab):
        tab.current_pred_raw = None
        tab.predict()
        assert tab.current_pred_raw is not None

    def test_predict_skips_when_image_is_none(self, tab):
        tab.image_data['image'] = None
        tab.current_pred_raw = None
        tab.predict()
        assert tab.current_pred_raw is None

    def test_pred_raw_values_in_zero_one(self, tab):
        assert tab.current_pred_raw.min() >= 0.0
        assert tab.current_pred_raw.max() <= 1.0

    def test_model_called_once_per_predict(self, qapp, mock_model, image_data):
        mock_model.reset_mock()
        tab = ImageAnalysisTab(mock_model, torch.device('cpu'), image_data, tab_index=1)
        calls_after_init = mock_model.call_count
        tab.current_pred_raw = None
        tab.predict()
        assert mock_model.call_count == calls_after_init + 1


# -------------------------
# ImageAnalysisTab — update_confidence_metrics
# -------------------------

class TestConfidenceMetrics:
    def test_skips_when_pred_raw_is_none(self, tab):
        tab.current_pred_raw = None
        tab.update_confidence_metrics()  # must not raise

    def test_tumor_detected_when_many_high_confidence_pixels(self, tab):
        pred = np.zeros((256, 256), dtype=np.float32)
        pred[:40, :30] = 0.9   # 1 200 pixels > 0.65
        tab.current_pred_raw = pred
        tab.update_confidence_metrics()
        assert "Tumor" in tab.tumor_label.text()
        assert "Detected" in tab.tumor_label.text()

    def test_no_tumor_when_few_high_confidence_pixels(self, tab):
        pred = np.zeros((256, 256), dtype=np.float32)
        pred[:10, :10] = 0.9   # only 100 pixels
        tab.current_pred_raw = pred
        tab.update_confidence_metrics()
        assert "No Tumor" in tab.tumor_label.text()

    def test_all_pixels_above_threshold_shows_tumor(self, tab):
        tab.current_pred_raw = np.full((256, 256), 0.9, dtype=np.float32)
        tab.update_confidence_metrics()
        assert "Detected" in tab.tumor_label.text()

    def test_no_pixels_above_threshold_shows_no_tumor(self, tab):
        tab.current_pred_raw = np.full((256, 256), 0.1, dtype=np.float32)
        tab.update_confidence_metrics()
        assert "No Tumor" in tab.tumor_label.text()

    def test_conf_bar_value_in_range(self, tab):
        tab.current_pred_raw = np.full((256, 256), 0.5, dtype=np.float32)
        tab.update_confidence_metrics()
        assert 0 <= tab.conf_bar.value() <= 100

    def test_pixels_label_shows_correct_tumor_count(self, tab):
        pred = np.zeros((256, 256), dtype=np.float32)
        pred[:10, :10] = 0.9   # 100 pixels
        tab.current_pred_raw = pred
        tab.update_confidence_metrics()
        assert "100" in tab.pixels_label.text()

    def test_pixels_label_shows_total_pixel_count(self, tab):
        tab.current_pred_raw = np.zeros((256, 256), dtype=np.float32)
        tab.update_confidence_metrics()
        assert str(256 * 256) in tab.pixels_label.text()

    def test_stats_label_contains_coverage(self, tab):
        tab.current_pred_raw = np.full((256, 256), 0.3, dtype=np.float32)
        tab.update_confidence_metrics()
        assert "Coverage" in tab.stats_label.text()

    def test_stats_label_contains_max_confidence(self, tab):
        tab.current_pred_raw = np.full((256, 256), 0.7, dtype=np.float32)
        tab.update_confidence_metrics()
        assert "Max Confidence" in tab.stats_label.text()


# -------------------------
# ImageAnalysisTab — zoom region
# -------------------------

class TestZoomRegion:
    def test_zoom_region_sets_xlim(self, tab):
        tab.zoom_region(10, 20, 100, 150)
        assert tab.orig_ax.get_xlim() == (10, 100)

    def test_zoom_region_sets_inverted_ylim(self, tab):
        tab.zoom_region(10, 20, 100, 150)
        lo, hi = tab.orig_ax.get_ylim()
        assert lo > hi  # image y-axis is inverted

    def test_zoom_region_syncs_both_canvases(self, tab):
        tab.zoom_region(5, 5, 50, 50)
        assert tab.orig_ax.get_xlim() == tab.heatmap_ax.get_xlim()
        assert tab.orig_ax.get_ylim() == tab.heatmap_ax.get_ylim()

    def test_zoom_region_handles_reversed_input_coords(self, tab):
        tab.zoom_region(100, 150, 10, 20)
        xmin, xmax = tab.orig_ax.get_xlim()
        assert xmin < xmax

    def test_is_zoomed_false_after_reset(self, tab):
        tab.reset_zoom()
        assert tab.is_zoomed() is False

    def test_is_zoomed_true_after_zoom_region(self, tab):
        tab.reset_zoom()
        tab.zoom_region(10, 10, 50, 50)
        assert tab.is_zoomed() is True

    def test_reset_zoom_restores_full_width(self, tab):
        tab.zoom_region(10, 10, 50, 50)
        tab.reset_zoom()
        h, w = tab.image_data['image'].shape
        assert tab.orig_ax.get_xlim() == (0, w)

    def test_reset_zoom_restores_full_height(self, tab):
        tab.zoom_region(10, 10, 50, 50)
        tab.reset_zoom()
        h, w = tab.image_data['image'].shape
        assert tab.orig_ax.get_ylim() == (h, 0)

    def test_reset_zoom_reactivates_selectors(self, tab):
        tab.selector.set_active(False)
        tab.heatmap_selector.set_active(False)
        tab.reset_zoom()
        assert tab.selector.get_active()
        assert tab.heatmap_selector.get_active()


# -------------------------
# ImageAnalysisTab — on_select
# -------------------------

class TestOnSelect:
    def test_on_select_zooms_to_selection(self, tab):
        tab.reset_zoom()
        tab.on_select(_Event(xdata=10.0, ydata=20.0), _Event(xdata=100.0, ydata=150.0))
        assert tab.is_zoomed() is True

    def test_on_select_returns_early_when_xdata_none(self, tab):
        tab.reset_zoom()
        tab.on_select(_Event(xdata=None, ydata=10.0), _Event(xdata=100.0, ydata=150.0))
        assert tab.is_zoomed() is False  # state unchanged

    def test_on_select_deactivates_selectors(self, tab):
        tab.selector.set_active(True)
        tab.heatmap_selector.set_active(True)
        tab.on_select(_Event(xdata=10.0, ydata=10.0), _Event(xdata=100.0, ydata=100.0))
        assert not tab.selector.get_active()
        assert not tab.heatmap_selector.get_active()


# -------------------------
# ImageAnalysisTab — panning events
# -------------------------

class TestPanningEvents:
    def test_on_release_clears_panning_flag(self, tab):
        tab.panning = True
        tab.on_release(_Event(button=1))
        assert tab.panning is False

    def test_on_release_clears_press_coordinates(self, tab):
        tab.panning = True
        tab.press_x = 5.0
        tab.press_y = 5.0
        tab.on_release(_Event(button=1))
        assert tab.press_x is None
        assert tab.press_y is None

    def test_on_release_ignores_non_left_button(self, tab):
        tab.panning = True
        tab.on_release(_Event(button=3))
        assert tab.panning is True  # unchanged

    def test_on_press_starts_panning_when_zoomed(self, tab):
        tab.reset_zoom()
        tab.zoom_region(10, 10, 100, 100)
        tab.selector.set_active(False)
        tab.heatmap_selector.set_active(False)
        tab.on_press(_Event(button=1, xdata=50.0, ydata=50.0))
        assert tab.panning is True

    def test_on_press_does_not_pan_when_not_zoomed(self, tab):
        tab.reset_zoom()
        tab.selector.set_active(False)
        tab.heatmap_selector.set_active(False)
        tab.on_press(_Event(button=1, xdata=50.0, ydata=50.0))
        assert tab.panning is False

    def test_on_press_does_not_pan_when_selector_active(self, tab):
        tab.zoom_region(10, 10, 100, 100)
        tab.selector.set_active(True)
        tab.on_press(_Event(button=1, xdata=50.0, ydata=50.0))
        assert tab.panning is False

    def test_on_press_ignores_none_coordinates(self, tab):
        tab.on_press(_Event(button=1, xdata=None, ydata=None))
        assert tab.panning is False

    def test_on_motion_shifts_xlim_while_panning(self, tab):
        tab.reset_zoom()
        tab.zoom_region(10, 10, 100, 100)
        tab.selector.set_active(False)
        tab.heatmap_selector.set_active(False)
        tab.on_press(_Event(button=1, xdata=50.0, ydata=50.0))
        xlim_before = tab.orig_ax.get_xlim()
        tab.on_motion(_Event(button=1, xdata=40.0, ydata=40.0))
        assert tab.orig_ax.get_xlim() != xlim_before

    def test_on_motion_no_change_when_not_panning(self, tab):
        tab.panning = False
        xlim_before = tab.orig_ax.get_xlim()
        tab.on_motion(_Event(button=1, xdata=40.0, ydata=40.0))
        assert tab.orig_ax.get_xlim() == xlim_before

    def test_on_motion_ignores_none_coordinates(self, tab):
        tab.panning = True
        tab.press_x = 50.0
        tab.press_y = 50.0
        tab.orig_xlim = tab.orig_ax.get_xlim()
        tab.orig_ylim = tab.orig_ax.get_ylim()
        xlim_before = tab.orig_ax.get_xlim()
        tab.on_motion(_Event(button=1, xdata=None, ydata=None))
        assert tab.orig_ax.get_xlim() == xlim_before


# -------------------------
# UltrasoundGUI — initial state
# -------------------------

class TestUltrasoundGUIInitialState:
    def test_window_title_identifies_application(self, gui):
        assert "Ultrasound Tumor Detection" in gui.windowTitle()

    def test_images_list_is_empty(self, gui):
        assert gui.images == []

    def test_count_label_shows_zero(self, gui):
        assert "0" in gui.image_count_label.text()

    def test_no_tabs_on_startup(self, gui):
        assert gui.tabs.count() == 0

    def test_tabs_have_close_buttons(self, gui):
        assert gui.tabs.tabsClosable()


# -------------------------
# UltrasoundGUI — add_image
# -------------------------

class TestAddImage:
    @staticmethod
    def _patch_cv2(fake):
        return patch('ultrasound_tumor_detection.__main__.cv2.imread', return_value=fake), \
               patch('ultrasound_tumor_detection.__main__.cv2.resize', return_value=fake)

    def test_add_image_appends_to_images_list(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        rd, rz = self._patch_cv2(fake)
        with rd, rz:
            gui.add_image('/fake/img.png')
        assert len(gui.images) == 1

    def test_add_image_creates_one_tab(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        rd, rz = self._patch_cv2(fake)
        with rd, rz:
            gui.add_image('/fake/img.png')
        assert gui.tabs.count() == 1

    def test_add_image_updates_count_label(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        rd, rz = self._patch_cv2(fake)
        with rd, rz:
            gui.add_image('/fake/img.png')
        assert "1" in gui.image_count_label.text()

    def test_add_image_stores_basename(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        rd, rz = self._patch_cv2(fake)
        with rd, rz:
            gui.add_image('/some/dir/scan.png')
        assert gui.images[-1]['name'] == 'scan.png'

    def test_add_multiple_images_each_gets_tab(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        rd, rz = self._patch_cv2(fake)
        with rd, rz:
            gui.add_image('/fake/a.png')
            gui.add_image('/fake/b.png')
        assert len(gui.images) == 2
        assert gui.tabs.count() == 2

    def test_add_image_error_does_not_raise(self, gui):
        with patch('ultrasound_tumor_detection.__main__.cv2.imread', side_effect=Exception("I/O error")):
            gui.add_image('/nonexistent/path.png')  # exception caught internally


# -------------------------
# UltrasoundGUI — close_tab
# -------------------------

class TestCloseTab:
    def _setup(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        with patch('ultrasound_tumor_detection.__main__.cv2.imread', return_value=fake), \
             patch('ultrasound_tumor_detection.__main__.cv2.resize', return_value=fake):
            gui.add_image('/fake/a.png')
            gui.add_image('/fake/b.png')

    def test_close_tab_removes_one_tab(self, gui):
        self._setup(gui)
        before = gui.tabs.count()
        gui.close_tab(0)
        assert gui.tabs.count() == before - 1

    def test_close_tab_removes_corresponding_image(self, gui):
        self._setup(gui)
        before = len(gui.images)
        gui.close_tab(0)
        assert len(gui.images) == before - 1

    def test_close_tab_updates_count_label(self, gui):
        self._setup(gui)
        gui.close_tab(0)
        assert "1" in gui.image_count_label.text()

    def test_close_last_tab_empties_gui(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        with patch('ultrasound_tumor_detection.__main__.cv2.imread', return_value=fake), \
             patch('ultrasound_tumor_detection.__main__.cv2.resize', return_value=fake):
            gui.add_image('/fake/only.png')
        gui.close_tab(0)
        assert len(gui.images) == 0
        assert gui.tabs.count() == 0


# -------------------------
# UltrasoundGUI — clear_all
# -------------------------

class TestClearAll:
    def _setup(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        with patch('ultrasound_tumor_detection.__main__.cv2.imread', return_value=fake), \
             patch('ultrasound_tumor_detection.__main__.cv2.resize', return_value=fake):
            gui.add_image('/fake/a.png')
            gui.add_image('/fake/b.png')

    def test_clear_all_empties_images_list(self, gui):
        self._setup(gui)
        gui.clear_all()
        assert gui.images == []

    def test_clear_all_removes_all_tabs(self, gui):
        self._setup(gui)
        gui.clear_all()
        assert gui.tabs.count() == 0

    def test_clear_all_resets_count_label(self, gui):
        self._setup(gui)
        gui.clear_all()
        assert "0" in gui.image_count_label.text()

    def test_clear_all_on_empty_gui_is_safe(self, gui):
        gui.clear_all()  # must not raise
        assert gui.images == []


# -------------------------
# UltrasoundGUI — file dialogs
# -------------------------

class TestFileDialogs:
    def test_load_image_adds_image_when_file_selected(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        with patch('ultrasound_tumor_detection.__main__.QFileDialog') as mock_dlg, \
             patch('ultrasound_tumor_detection.__main__.cv2.imread', return_value=fake), \
             patch('ultrasound_tumor_detection.__main__.cv2.resize', return_value=fake):
            mock_dlg.getOpenFileName.return_value = ('/fake/img.png', '')
            gui.load_image()
        assert len(gui.images) == 1

    def test_load_image_no_op_when_dialog_cancelled(self, gui):
        with patch('ultrasound_tumor_detection.__main__.QFileDialog') as mock_dlg:
            mock_dlg.getOpenFileName.return_value = ('', '')
            gui.load_image()
        assert len(gui.images) == 0

    def test_load_multiple_images_adds_each_selected_file(self, gui):
        fake = np.zeros((256, 256), dtype=np.uint8)
        with patch('ultrasound_tumor_detection.__main__.QFileDialog') as mock_dlg, \
             patch('ultrasound_tumor_detection.__main__.cv2.imread', return_value=fake), \
             patch('ultrasound_tumor_detection.__main__.cv2.resize', return_value=fake):
            mock_dlg.getOpenFileNames.return_value = (['/fake/a.png', '/fake/b.png'], '')
            gui.load_multiple_images()
        assert len(gui.images) == 2

    def test_load_multiple_images_no_op_when_cancelled(self, gui):
        with patch('ultrasound_tumor_detection.__main__.QFileDialog') as mock_dlg:
            mock_dlg.getOpenFileNames.return_value = ([], '')
            gui.load_multiple_images()
        assert len(gui.images) == 0
