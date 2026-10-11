"""Model serialization, loading, and prediction service."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from production_app.logging_config import get_logger

logger = get_logger(__name__)


def save_model(pipeline: Pipeline, model_path: str) -> None:
    """Save trained model pipeline to file."""

    try:
        destination_path = Path(model_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {destination_path}")
        joblib.dump(pipeline, model_path)
        logger.info("MODEL pipline saved successfully ")
    except Exception as exc:
        logger.error("Failed to save model pipeline to %s : %s ", model_path, exc)
        raise


def load_model(model_path: str) -> Pipeline:
    """Load model pipeline from file."""
    if not os.path.exists(model_path):
        logger.error("Model file not found at %s", model_path)
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        logger.info(f"Loading model from {model_path}")
        pipeline = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return pipeline
    except Exception as exc:
        logger.error("Failed to load model pipeline from %s : %s ", model_path, exc)
        raise


def predict(pipeline: Pipeline, data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict using the loaded model pipeline."""
    if data is None or len(data) == 0:
        logger.error("Prediction Failed : input data is empty")
        raise ValueError("Cannot predict on empty input data")

    try:
        logger.info("Starting predictions on %d samples", len(data))

        predictions = pipeline.predict(data)
        logger.info("Prediction completed successfully: %s", predictions.tolist())
        return predictions
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise
