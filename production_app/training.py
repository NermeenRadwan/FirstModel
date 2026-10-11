"""Model creation, pipeline assembly, and model fitting."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.logging_config import get_logger
from production_app.preprocessing import create_preprocessor

logger = get_logger(__name__)


def create_pipeline(random_state: int = 42, max_iter: int = 1000) -> Pipeline:
    """Create a complete scikit-learn pipeline combining preprocessing and classifier.
    Args:random_state: Random state for reproducibility.
        max_iter: Maximum number of iterations for solver convergence.
    Returns: Unfitted scikit-learn Pipeline.
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


def train_model(pipeline: Pipeline,X_train: pd.DataFrame,y_train: pd.Series) -> Pipeline:

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
