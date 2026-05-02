"""Tests for the CLI entry points in cli.py."""

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from ultrasound_tumor_detection.cli import _parser, evaluate, gui, run
from ultrasound_tumor_detection.pipeline import MODEL_PATH


# -------------------------
# _parser
# -------------------------

class TestParser:
    def test_default_data_dir_is_none(self):
        args = _parser("test").parse_args([])
        assert args.data_dir is None

    def test_default_model_path(self):
        args = _parser("test").parse_args([])
        assert args.model_path == MODEL_PATH

    def test_default_epochs_is_none(self):
        args = _parser("test").parse_args([])
        assert args.epochs is None

    def test_default_batch_size(self):
        args = _parser("test").parse_args([])
        assert args.batch_size == 8

    def test_data_dir_flag(self):
        args = _parser("test").parse_args(["--data-dir", "/tmp/data"])
        assert args.data_dir == "/tmp/data"

    def test_model_path_flag(self):
        args = _parser("test").parse_args(["--model-path", "/tmp/model.pth"])
        assert args.model_path == "/tmp/model.pth"

    def test_epochs_flag(self):
        args = _parser("test").parse_args(["--epochs", "5"])
        assert args.epochs == 5

    def test_batch_size_flag(self):
        args = _parser("test").parse_args(["--batch-size", "16"])
        assert args.batch_size == 16


# -------------------------
# run()
# -------------------------

class TestRun:
    def test_run_calls_run_pipeline(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            run([])
        mock_rp.assert_called_once()

    def test_run_passes_default_args(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            run([])
        _, kwargs = mock_rp.call_args
        assert kwargs["data_dir"] is None
        assert kwargs["model_path"] == MODEL_PATH
        assert kwargs["epochs"] is None
        assert kwargs["batch_size"] == 8

    def test_run_passes_custom_data_dir(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            run(["--data-dir", "/my/data"])
        _, kwargs = mock_rp.call_args
        assert kwargs["data_dir"] == "/my/data"

    def test_run_passes_custom_epochs(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            run(["--epochs", "3"])
        _, kwargs = mock_rp.call_args
        assert kwargs["epochs"] == 3

    def test_run_passes_custom_batch_size(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            run(["--batch-size", "4"])
        _, kwargs = mock_rp.call_args
        assert kwargs["batch_size"] == 4

    def test_run_passes_custom_model_path(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            run(["--model-path", "/tmp/custom.pth"])
        _, kwargs = mock_rp.call_args
        assert kwargs["model_path"] == "/tmp/custom.pth"


# -------------------------
# evaluate()
# -------------------------

class TestEvaluate:
    def test_evaluate_calls_run_pipeline(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            evaluate([])
        mock_rp.assert_called_once()

    def test_evaluate_passes_default_args(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            evaluate([])
        _, kwargs = mock_rp.call_args
        assert kwargs["data_dir"] is None
        assert kwargs["model_path"] == MODEL_PATH

    def test_evaluate_passes_custom_model_path(self):
        with patch("ultrasound_tumor_detection.cli.run_pipeline") as mock_rp:
            evaluate(["--model-path", "/tmp/eval.pth"])
        _, kwargs = mock_rp.call_args
        assert kwargs["model_path"] == "/tmp/eval.pth"


# -------------------------
# gui()
# -------------------------

class TestGui:
    def test_gui_creates_qapplication(self):
        mock_app = MagicMock()
        mock_app.exec_.return_value = 0
        mock_window = MagicMock()

        with patch("ultrasound_tumor_detection.cli.QApplication", return_value=mock_app) as mock_qapp, \
             patch("ultrasound_tumor_detection.cli.UltrasoundGUI", return_value=mock_window), \
             patch("ultrasound_tumor_detection.cli.sys.exit"):
            gui([])

        mock_qapp.assert_called_once()

    def test_gui_shows_window(self):
        mock_app = MagicMock()
        mock_app.exec_.return_value = 0
        mock_window = MagicMock()

        with patch("ultrasound_tumor_detection.cli.QApplication", return_value=mock_app), \
             patch("ultrasound_tumor_detection.cli.UltrasoundGUI", return_value=mock_window), \
             patch("ultrasound_tumor_detection.cli.sys.exit"):
            gui([])

        mock_window.show.assert_called_once()

    def test_gui_exits_with_exec_return_code(self):
        mock_app = MagicMock()
        mock_app.exec_.return_value = 42
        mock_window = MagicMock()

        with patch("ultrasound_tumor_detection.cli.QApplication", return_value=mock_app), \
             patch("ultrasound_tumor_detection.cli.UltrasoundGUI", return_value=mock_window), \
             patch("ultrasound_tumor_detection.cli.sys.exit") as mock_exit:
            gui([])

        mock_exit.assert_called_once_with(42)
