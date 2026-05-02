"""Command-line interface for training and evaluating the model."""

import argparse

from ultrasound_tumor_detection.pipeline import MODEL_PATH, run_pipeline


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", default=None, help="Path to Dataset_BUSI_with_GT. Downloads via Kaggle if omitted.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to model checkpoint.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs when no checkpoint exists.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for dataloaders.")
    parser.add_argument("--no-show", action="store_true", help="Skip prediction plots.")
    return parser


def run(argv=None):
    parser = _parser("Run the BUSI download/train/evaluate/visualize pipeline.")
    args = parser.parse_args(argv)
    run_pipeline(
        data_dir=args.data_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        show_predictions=not args.no_show,
    )


def train(argv=None):
    run(argv)


def evaluate(argv=None):
    parser = _parser("Evaluate an existing checkpoint on BUSI data.")
    args = parser.parse_args(argv)
    run_pipeline(
        data_dir=args.data_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        show_predictions=not args.no_show,
    )
