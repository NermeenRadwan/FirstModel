"""FastAPI application endpoints module.

Endpoints and app instance have moved to `production_app.main` for clean project organization.
This module re-exports `app` and endpoint handlers for backward compatibility.
"""

from production_app.main import app, check_model, health, hello, predict

__all__ = ["app", "check_model", "health", "hello", "predict"]
