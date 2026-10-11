"""Application configuration."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str
    model_path: str
    log_level: str
    test_size: float
    random_seed: int


def load_settings() -> Settings:
    """Load runtime settings from environment variables."""
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        model_path=os.getenv("MODEL_PATH", "models/model.joblib"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
    )
