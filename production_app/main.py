import numpy as np
import pandas as pd

from production_app.config import load_settings
from production_app.data import load_data, split_data
from production_app.evaluation import evaluate_model
from production_app.logging_config import configure_logging, get_logger
from production_app.model_service import load_model, predict, save_model
from production_app.training import create_pipeline, train_model

logger = get_logger(__name__)


def run_training_pipeline() -> dict[str, float]:
    """Execute the end-to-end training, evaluation, and saving pipeline."""
    settings = load_settings()
    configure_logging(level=settings.log_level)
    logger.info("=== Starting Training Pipeline ===")

    # 1. Load and split data
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
    )

    # 2. Build and train pipeline
    pipeline = create_pipeline(random_state=settings.random_seed)
    fitted_pipeline = train_model(pipeline, X_train, y_train)

    # 3. Evaluate model
    metrics = evaluate_model(fitted_pipeline, X_test, y_test)

    # 4. Save the fitted artifact
    save_model(fitted_pipeline, settings.model_path)
    logger.info("=== Training Pipeline Completed Successfully ===")
    return metrics


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
        sample_data = X.head(3)

    predictions = predict(model, sample_data)
    logger.info("=== Prediction Pipeline Completed Successfully ===")
    return predictions
