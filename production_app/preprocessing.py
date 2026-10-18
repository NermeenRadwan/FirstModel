"""Define reusable preprocessing steps.
Learned transformations are fitted only on training data to prevent data leakage.
"""

from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

from production_app.logging_config import get_logger

logger = get_logger(__name__)


def create_preprocessor(scaler_type: str = "robust") -> TransformerMixin:
    """Create a scaler preprocessor.
    Args:scaler_type: Type of scaler ('robust' or 'standard').
    Returns:Unfitted scaler transformer.
    """
    logger.info("Initializing preprocessor with scaler_type=%s", scaler_type)

    if scaler_type == "standard":
        return StandardScaler()
    return RobustScaler()
