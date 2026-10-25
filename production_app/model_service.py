"""Model serialization, loading, and prediction service."""

import os
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from production_app.config import SELECTED_FEATURES, load_settings
from production_app.data import load_data
from production_app.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def save_model(pipeline: Pipeline, model_path: str) -> None:
    """Save trained model pipeline to file."""
    try:
        destination_path = Path(model_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving model to %s", destination_path)
        joblib.dump(pipeline, model_path)
        logger.info("MODEL pipeline saved successfully")
    except Exception as exc:
        logger.error("Failed to save model pipeline to %s : %s", model_path, exc)
        raise


def load_model(model_path: str) -> Pipeline:
    """Load model pipeline from file."""
    if not os.path.exists(model_path):
        logger.error("Model file not found at %s", model_path)
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        logger.info("Loading model from %s", model_path)
        pipeline = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return pipeline
    except Exception as exc:
        logger.error("Failed to load model pipeline from %s : %s", model_path, exc)
        raise


def predict(pipeline: Pipeline, data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict using the loaded model pipeline."""
    if data is None or len(data) == 0:
        logger.error("Prediction Failed : input data is empty")
        raise ValueError("Cannot predict on empty input data")

    try:
        logger.info("Starting predictions on %d samples", len(data))
        predictions = cast(np.ndarray, pipeline.predict(data))
        logger.info("Prediction completed successfully: %s", predictions.tolist())
        return predictions
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise


def predict_proba(pipeline: Pipeline, data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict class probabilities using the loaded model pipeline."""
    if data is None or len(data) == 0:
        logger.error("Prediction Failed : input data is empty")
        raise ValueError("Cannot predict on empty input data")

    try:
        logger.info("Starting probability predictions on %d samples", len(data))
        probabilities = cast(np.ndarray, pipeline.predict_proba(data))
        return probabilities
    except Exception as exc:
        logger.error("Probability prediction failed: %s", exc)
        raise


def run_prediction_pipeline(
    sample_data: pd.DataFrame | np.ndarray | None = None,
) -> np.ndarray:
    """Load the saved model artifact and generate predictions without retraining."""
    settings = load_settings()
    configure_logging(level=settings.log_level)
    logger.info("=== Starting Prediction Pipeline ===")

    # Load persisted model
    model = load_model(settings.model_path)

    # If no sample provided, take the first 3 samples from the dataset as a demonstration
    if sample_data is None:
        logger.info("No sample data provided; loading 3 test samples from dataset")
        X, _ = load_data()
        sample_data = X[SELECTED_FEATURES]

    predictions = predict(model, sample_data)
    logger.info("Prediction completed: %s", predictions.tolist())
    return predictions


if __name__ == "__main__":
    run_prediction_pipeline()
