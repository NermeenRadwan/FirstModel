FROM python:3.12-slim

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install dependencies
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

# Copy application and model
COPY production_app/ production_app/
COPY models/ models/
COPY run.py ./

EXPOSE 8000

CMD ["uvicorn", "production_app.main:app", "--host", "0.0.0.0", "--port", "8000"]