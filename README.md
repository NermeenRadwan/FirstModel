# Breast Cancer Classification: From Notebook to Production Python

A complete, reproducible, and modular Machine Learning classification project built with VS Code and Antigravity IDE    , Python 3.12, and `uv`. This project transitions exploratory data analysis from a Jupyter Notebook into a clean, tested, production-ready Python application.

---

## 1. Project Goal

The primary objective of this project is to apply production software engineering best practices to a Machine Learning workflow:
- **Separation of Concerns**: Decomposing monolithic notebook logic into modular, single-responsibility Python files (`data`, `preprocessing`, `training`, `evaluation`, `model_service`, `main`).
- **Reproducibility**: Managing deterministic dependencies with `uv` (`pyproject.toml` and `uv.lock`).
- **Preventing Data Leakage**: Enforcing strict separation between training and test sets using `scikit-learn`'s `Pipeline`, ensuring preprocessing parameters are fitted only on training data.
- **Artifact Serialization**: Saving the fitted preprocessing + model pipeline to disk (`models/model.joblib`) and loading it for inference without retraining.
- **Production Foundations**: Centralized logging, immutable configuration via environment variables, strict type annotations, error handling, and automated code formatting with Ruff.

---

## 2. Dataset

- **Name**: Breast Cancer Wisconsin (Diagnostic) Dataset (`sklearn.datasets.load_breast_cancer`).
- **Instances**: 569 patient samples (212 Malignant `0`, 357 Benign `1`).
- **Features**: 30 numeric continuous features computed from digitized images of fine needle aspirate (FNA) of breast masses (e.g., mean radius, mean texture, mean smoothness, worst concavity).
- **Target**: Binary classification (0 = Malignant, 1 = Benign).

---

## 3. ML Scope & Model Choice

- **Chosen Model**: `LogisticRegression(max_iter=1000, random_state=42)` combined with `RobustScaler`.
- **Rationale**: 
  - The dataset consists entirely of continuous numeric features with varying scales and potential outliers. `RobustScaler` scales features using the median and interquartile range (IQR), making it resilient to extreme values.
  - `LogisticRegression` serves as an optimal baseline model: highly interpretable, fast to train, probabilistic, and achieves exceptional linear separability on this feature set.

---

## 4. Machine Learning Pipeline Architecture

The end-to-end pipeline follows a strict, leak-free design:

```
┌─────────────────────────┐
│       Load Data         │ (data.py: load_breast_cancer)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Stratified Split      │ (data.py: 80% train, 20% test, stratified by y)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     Assemble Pipeline   │ (training.py: Pipeline([
│  - RobustScaler         │    ('preprocessor', RobustScaler()),
│  - LogisticRegression   │    ('classifier', LogisticRegression())
└────────────┬────────────┘  ]))
             │
             ▼
┌─────────────────────────┐
│       Fit Pipeline      │ (Fitted ONLY on X_train to prevent data leakage)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      Evaluate Model     │ (evaluation.py: Accuracy, Precision, Recall, F1, Confusion Matrix)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      Save Pipeline      │ (model_service.py: joblib.dump -> models/model.joblib)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Reload & Predict      │ (model_service.py: joblib.load & predict on new inputs without retraining)
└─────────────────────────┘
```

---

## 5. Feature Selection Research & Evaluation Results

### Feature Selection Technique Families Comparison
| Family | Mechanism | Example | Key Strength | Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Filter** | Statistical scores ranking features independently of model fitting | ANOVA F-test, Pearson correlation, `SelectKBest` | Fast, computationally lightweight, model-agnostic | Ignores feature interactions and model-specific nuances |
| **Wrapper** *(Chosen)* | Evaluates feature subsets iteratively by training an estimator | **RFECV** (Recursive Feature Elimination with CV) | Captures feature dependencies, optimizes target metric directly | Computationally intensive |
| **Embedded** | Feature selection performed intrinsically during model training | L1 / Lasso regularization, Tree-based Gini importance | Balances model complexity and feature selection simultaneously | Tied strictly to algorithms with built-in regularization |

*Rationale for Choosing Wrapper (RFECV)*: The Wisconsin Diagnostic dataset has strong multicollinearity among cell nuclei dimensions (e.g., radius, perimeter, and area). RFECV with 5-fold cross-validation systematically eliminated 12 redundant features without degrading diagnostic performance.

### Baseline vs. Selected Features Model Comparison
Evaluated on the independent test split (114 hold-out samples, 20% stratified test size):

| Metric | Baseline Model (All 30 Features) | Selected Model (18 Features) | Δ Change |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 98.25% | **99.12%** | **+0.87%** |
| **Precision** | 98.61% | **98.63%** | **+0.02%** |
| **Recall** | 98.61% | **100.00%** | **+1.39%** |
| **F1-Score** | 98.61% | **99.31%** | **+0.70%** |
| **Feature Count** | 30 | **18** | **-40% complexity** |

### Selected 18 Features (Preserved Order):
`mean texture`, `mean compactness`, `mean concavity`, `mean concave points`, `radius error`, `area error`, `compactness error`, `concave points error`, `fractal dimension error`, `worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst smoothness`, `worst concavity`, `worst concave points`, `worst symmetry`, `worst fractal dimension`.

### Confusion Matrix (18 Features):
```
                     Predicted Malignant (0)   Predicted Benign (1)
Actual Malignant (0)           41                         1
Actual Benign (1)               0                        72
```

---

## 6. Project Structure

```
First-ML-project/
├── notebooks/
│   └── 01_exploration.ipynb    # Stage 01: Exploratory analysis & baseline experiments
├── production_app/
│   ├── __init__.py             # Package marker
│   ├── config.py               # Immutable application settings loaded from .env
│   ├── logging_config.py       # Centralized logging configuration & reusable loggers
│   ├── data.py                 # Dataset loading and stratified train/test split
│   ├── preprocessing.py        # Reusable transformers preventing data leakage
│   ├── training.py             # Complete training pipeline and model fitting
│   ├── evaluation.py           # Metric calculation (accuracy, precision, recall, F1, CM)
│   ├── model_service.py        # Serialization (joblib), artifact reloading, prediction pipeline
│   ├── schemas.py              # Pydantic schemas for REST API validation
│   ├── main.py                 # FastAPI application with all REST API endpoints
│   ├── api.py                  # Compatibility re-export module
│   └── ui/
│       ├── __init__.py         # Package marker
│       └── streamlit.py        # Streamlit interactive web dashboard module
├── models/
│   ├── model.joblib            # Persisted trained pipeline artifact
│   ├── metrics.json            # Saved evaluation metrics
│   └── selected_features.json  # Recorded selected and dropped feature lists
├── tests/
│   ├── test_model.py           # Model artifact loading and prediction tests
│   └── test_api.py             # Endpoint tests (/health, /predict, 422, 503)
├── streamlit_app.py            # FastAPI Streamlit HTTP client UI
├── Dockerfile                  # Container definition for FastAPI backend
├── .dockerignore               # Docker build exclusions
├── run.py                      # Application entry point CLI
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git exclusions (.venv, caches, artifacts, secrets)
├── .pre-commit-config.yaml     # Pre-commit configuration with Ruff hooks
├── pyproject.toml              # Declarative project dependencies and build metadata
├── uv.lock                     # Reproducible lockfile
└── README.md                   # Project documentation and execution instructions
```

---

## 7. Installation & Execution Guide

### Prerequisites
- Python `>= 3.12`
- `uv` package manager ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))

### Step 1: Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd First-ML-project
```

### Step 2: Install Virtual Environment & Dependencies
Use `uv` to automatically create the `.venv` and install all pinned dependencies:
```bash
uv sync
```

### Step 3: Configure Environment Variables
Create your local `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Default configuration values in `.env`:
```env
APP_ENV=development
LOG_LEVEL=INFO
MODEL_PATH=models/model.joblib
TEST_SIZE=0.2
RANDOM_SEED=42
```

### Step 4: Run the Application

#### Option A: Full End-to-End Workflow (Train → Evaluate → Save → Load → Predict)
```bash
uv run python run.py
```

#### Option B: Train & Save Pipeline Only
```bash
uv run python run.py train
```
*Fits the model on `X_train`, logs evaluation metrics, and saves the pipeline artifact to `models/model.joblib`.*

#### Option C: Predict Using Saved Model (No Retraining)
```bash
uv run python run.py predict
```
*Loads the serialized pipeline from disk and predicts on input samples without retraining.*

#### Option D: Launch FastAPI REST Server
Start the production FastAPI server:
```bash
uv run uvicorn production_app.main:app --reload --host 127.0.0.1 --port 8000
```
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Example prediction using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
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
         "worst fractal dimension": 0.11890
       }
     }'
```

#### Option E: Launch Streamlit Web UI (HTTP Client)
Launch the interactive clinical dashboard:
```bash
uv run streamlit run streamlit_app.py
```
- **Local URL**: [http://localhost:8501](http://localhost:8501)
- **Architecture**: Acts strictly as a client making HTTP POST requests to the FastAPI backend service; does not load or retrain models locally.
- **Features**: Interactive biomarker sliders, 1-click clinical presets (Benign/Malignant/Borderline), real-time probability distribution gauges, relative biomarker comparison bar charts, and JSON report export.

#### Option F: Build and Run Docker API Container
Build the production Docker image and serve the FastAPI application inside a container:
```bash
# 1. Build the Docker image
docker build -t breast-cancer-api:latest .

# 2. Run the container with port mapping 8000:8000
docker run -d --name cancer-api -p 8000:8000 breast-cancer-api:latest

# 3. Verify healthcheck endpoint on mapped port
curl http://127.0.0.1:8000/health

# 4. Stop and remove the container
docker stop cancer-api && docker rm cancer-api
```

#### Option G: Run Automated Pytest Suite
Run the test suite verifying model artifact loading, feature order validation, and API endpoint behavior:
```bash
uv run pytest -v
```

---

## 8. API Contract & Validation Rules

| Endpoint | Method | Expected Input | Success Response | Error Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `/` | `GET` | None | `HTTP 200` {"message": "..."} | N/A |
| `/health` | `GET` | None | `HTTP 200` {"status": "healthy", "model_loaded": true} | N/A |
| `/check-model`| `GET` | None | `HTTP 200` {"message": "Model is ready"} | `HTTP 503` if model file missing |
| `/predict` | `POST` | Dict of 18 features OR list of 18 floats | `HTTP 200` with prediction, probability, clinical recommendation | `HTTP 422` on missing/extra/malformed features; `HTTP 503` if model unready |

---

## 9. Code Quality & Pre-commit

Format and lint the codebase using Ruff:
```bash
# Check and auto-fix linting issues
uv run ruff check --fix

# Format code according to PEP 8 standards
uv run ruff format
```

Install pre-commit hooks for automated validation before git commits:
```bash
uv run pre-commit install
```