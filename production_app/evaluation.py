"""Model evaluation and metrics reporting."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from production_app.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    logger.info("Evaluating model on %d test samples", len(X_test))
    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    logger.info("Evaluation Resultls:")
    logger.info("Accuracy: %.2f%%", acc * 100)
    logger.info("Precision: %.2f%%", precision * 100)
    logger.info("Recall: %.2f%%", recall * 100)
    logger.info("F1 Score: %.2f%%", f1 * 100)
    logger.info("Confusion Matrix: %s", cm)

    class_report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": class_report,
    }
