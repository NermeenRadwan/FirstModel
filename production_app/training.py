"""Model creation, pipeline assembly, training, and full training pipeline orchestration."""

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.config import SELECTED_FEATURES, load_settings
from production_app.data import load_data, split_data
from production_app.evaluation import evaluate_model
from production_app.logging_config import configure_logging, get_logger
from production_app.model_service import save_model
from production_app.preprocessing import create_preprocessor

logger = get_logger(__name__)


def create_pipeline(random_state: int = 42, max_iter: int = 1000) -> Pipeline:
    """Create a complete scikit-learn pipeline combining preprocessing and classifier.

    Args:
        random_state: Random state for reproducibility.
        max_iter: Maximum number of iterations for solver convergence.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    logger.info(
        "Creating ML pipeline with RobustScaler and LogisticRegression (random_state=%d)",
        random_state,
    )
    preprocessor = create_preprocessor(scaler_type="robust")
    classifier = LogisticRegression(random_state=random_state, max_iter=max_iter)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the pipeline on training data.

    Args:
        pipeline: Unfitted pipeline.
        X_train: Training features.
        y_train: Training target labels.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    logger.info("Training started on %d samples", len(X_train))
    pipeline.fit(X_train, y_train)
    logger.info("Training completed successfully")
    return pipeline


def run_training_pipeline() -> dict[str, Any]:
    """Execute the end-to-end training, evaluation, and saving pipeline.

    Loads the dataset, selects features, splits stratified data,
    trains the pipeline, evaluates performance, and persists the model artifact.

    Returns:
        Evaluation metrics dictionary.
    """
    settings = load_settings()
    configure_logging(level=settings.log_level)
    logger.info("=== Starting Training Pipeline ===")

    # 1. Load data and filter to selected features
    X, y = load_data()
    X = X[SELECTED_FEATURES]

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


if __name__ == "__main__":
    run_training_pipeline()
