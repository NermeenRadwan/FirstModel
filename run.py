"""Application entry point to execute training or prediction workflows."""

import argparse
import sys

from production_app.main import run_prediction_pipeline, run_training_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Production ML Pipeline Entry Point (Breast Cancer Wisconsin Classification)",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default=None,
        choices=["train", "predict", "all"],
        help="Action to perform: 'train', 'predict', or 'all' (default)",
    )
    parser.add_argument(
        "--mode",
        dest="mode_flag",
        choices=["train", "predict", "all"],
        default=None,
        help="Alternative flag: --mode train or --mode predict",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point execution."""
    args = parse_args()
    mode = args.mode_flag or args.action or "all"

    if mode == "train":
        print("\n--> Starting Training Workflow...")
        metrics = run_training_pipeline()
        print(f"--> Training completed. Metrics: {metrics}")

    elif mode == "predict":
        print("\n--> Starting Prediction Workflow...")
        predictions = run_prediction_pipeline()
        print(f"--> Prediction completed. Output: {predictions.tolist()}")

    elif mode == "all":
        print(
            "\n--> Running Full End-to-End Workflow (Train -> Save -> Load -> Predict)..."
        )
        metrics = run_training_pipeline()
        print(f"--> Training completed. Accuracy: {metrics.get('accuracy', 0):.4f}")

        predictions = run_prediction_pipeline()
        print(
            f"--> Prediction completed on saved model. Predictions: {predictions.tolist()}"
        )

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
