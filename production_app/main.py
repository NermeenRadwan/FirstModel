"""FastAPI application entry point with all diagnostic REST API endpoints."""

import pandas as pd

# pyrefly: ignore[missing-import]
from fastapi import FastAPI, HTTPException

from production_app.config import load_settings
from production_app.logging_config import get_logger
from production_app.model_service import load_model, predict_proba
from production_app.model_service import predict as model_predict
from production_app.schemas import (
    REQUIRED_FEATURES,
    PredictionRequest,
    PredictionResponse,
)

logger = get_logger(__name__)

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Production REST API for Breast Cancer Wisconsin diagnostic prediction.",
    version="0.1.0",
)

settings = load_settings()


def get_loaded_model():
    """Attempt to load the model artifact, returning None if not found or corrupted."""
    try:
        return load_model(settings.model_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model artifact not ready at %s: %s", settings.model_path, exc)
        return None


# Initialize model on startup
model = get_loaded_model()


@app.get("/")
def hello():
    """Root welcoming endpoint."""
    return {"message": "Hello from the Breast Cancer Prediction API!"}


@app.get("/health")
def health():
    """Health check endpoint indicating service and model readiness."""
    global model
    if model is None:
        model = get_loaded_model()

    return {
        "status": "healthy" if model is not None else "not ready",
        "model_loaded": model is not None,
    }


@app.get("/check-model")
def check_model():
    """Check if the ML model artifact is loaded and ready for predictions."""
    global model
    if model is None:
        model = get_loaded_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure the model artifact exists.",
        )
    return {"message": "Model is ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict breast cancer diagnosis (benign vs malignant) from input features."""
    global model
    if model is None:
        model = get_loaded_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure the model has been trained and saved.",
        )

    # If sent as a named dictionary, strictly validate all required features are present and no extra features exist
    if isinstance(request.features, dict):
        missing = set(REQUIRED_FEATURES) - set(request.features.keys())
        extra = set(request.features.keys()) - set(REQUIRED_FEATURES)
        if missing or extra:
            error_details = []
            if missing:
                error_details.append(f"Missing required features: {sorted(missing)}")
            if extra:
                error_details.append(f"Unexpected extra features: {sorted(extra)}")
            raise HTTPException(
                status_code=422,
                detail="; ".join(error_details),
            )
        values = [request.features[col] for col in REQUIRED_FEATURES]
        input_df = pd.DataFrame([values], columns=REQUIRED_FEATURES)
    else:
        if len(request.features) != len(REQUIRED_FEATURES):
            raise HTTPException(
                status_code=422,
                detail=f"Expected {len(REQUIRED_FEATURES)} features, got {len(request.features)}",
            )
        input_df = pd.DataFrame([request.features], columns=REQUIRED_FEATURES)

    predictions = model_predict(model, input_df)
    pred_val = int(predictions[0])
    label = "benign" if pred_val == 1 else "malignant"

    # Compute probability distribution
    probs = predict_proba(model, input_df)[0]
    prob_malignant = float(probs[0])
    prob_benign = float(probs[1])

    # Probability of the predicted class
    top_prob = prob_benign if pred_val == 1 else prob_malignant
    conf_pct = f"{top_prob * 100:.2f}%"

    risk_level = "Low Risk" if pred_val == 1 else "High Risk"
    clinical_rec = (
        "Routine follow-up screening recommended."
        if pred_val == 1
        else "High risk of malignancy detected. Urgent biopsy confirmation and oncology consultation recommended."
    )

    return PredictionResponse(
        prediction=pred_val,
        label=label,
        probability=round(top_prob, 4),
        confidence_percentage=conf_pct,
        risk_level=risk_level,
        probabilities={
            "malignant": round(prob_malignant, 4),
            "benign": round(prob_benign, 4),
        },
        clinical_recommendation=clinical_rec,
    )


if __name__ == "__main__":
    # pyrefly: ignore[missing-import]
    import uvicorn

    uvicorn.run("production_app.main:app", host="127.0.0.1", port=8000, reload=True)
