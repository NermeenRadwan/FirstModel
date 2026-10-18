from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from production_app.config import SELECTED_FEATURES

REQUIRED_FEATURES = SELECTED_FEATURES

# Dummy data mapping each feature name to its realistic clinical value
# Type annotation: dict[str, Any] prevents dictionary invariance errors in Pydantic schemas
EXAMPLE_FEATURES_DICT: dict[str, Any] = {
    "mean texture": 10.38,
    "mean compactness": 0.2776,
    "mean concavity": 0.3001,
    "mean concave points": 0.1471,
    "radius error": 1.0950,
    "area error": 153.40,
    "compactness error": 0.04904,
    "concave points error": 0.01587,
    "fractal dimension error": 0.006193,
    "worst radius": 25.38,
    "worst texture": 17.33,
    "worst perimeter": 184.60,
    "worst area": 2019.0,
    "worst smoothness": 0.1622,
    "worst concavity": 0.7119,
    "worst concave points": 0.2654,
    "worst symmetry": 0.4601,
    "worst fractal dimension": 0.11890,
}


class PredictionRequest(BaseModel):
    # Accepts either a dictionary with feature names or a list of values
    features: dict[str, float] | list[float] = Field(
        ...,
        description="A dictionary mapping feature names to numerical values, or an ordered list",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": EXAMPLE_FEATURES_DICT
            }
        }
    )

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for malignant, 1 for benign")
    label: str = Field(..., description="Diagnostic prediction label")
    probability: float = Field(..., description="Confidence score between 0 and 1")
    confidence_percentage: str = Field(..., description="Confidence formatted as percentage")
    risk_level: str = Field(..., description="Clinical risk category")
    probabilities: dict[str, float] = Field(..., description="Probability per diagnostic class")
    clinical_recommendation: str = Field(..., description="Actionable next step recommendation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 0,
                "label": "malignant",
                "probability": 0.9452,
                "confidence_percentage": "94.52%",
                "risk_level": "High Risk",
                "probabilities": {
                    "malignant": 0.9452,
                    "benign": 0.0548,
                },
                "clinical_recommendation": (
                    "High risk of malignancy detected. Recommend urgent biopsy confirmation "
                    "and oncology specialist consultation."
                ),
            }
        }
    )