"""Automated tests for ML model loading and prediction."""

import pandas as pd
from sklearn.pipeline import Pipeline

from production_app.config import SELECTED_FEATURES, load_settings
from production_app.model_service import load_model, predict
from production_app.schemas import EXAMPLE_FEATURES_DICT


def test_model_loads_and_predicts():
    """Requirement 01: Verify trained artifact loads and predicts with selected features without retraining."""
    settings = load_settings()
    model = load_model(settings.model_path)
    assert isinstance(model, Pipeline)

    # Test prediction on selected features
    df = pd.DataFrame([EXAMPLE_FEATURES_DICT], columns=SELECTED_FEATURES)
    predictions = predict(model, df)

    assert len(predictions) == 1
    assert int(predictions[0]) in (0, 1)
