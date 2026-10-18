import logging

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_data():
    """Load the Breast Cancer dataset."""
    logger.info("Loading Breast Cancer dataset")

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    logger.info("Dataset loaded successfully: %s rows", len(X))

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Data split completed: train=%s, test=%s",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test

