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

## 5. Evaluation Results

Evaluated on the independent test split (114 hold-out samples, 20% stratified test size):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.25%** |
| **Precision** | **98.61%** |
| **Recall** | **98.61%** |
| **F1-Score** | **98.61%** |

### Confusion Matrix
```
                     Predicted Malignant (0)   Predicted Benign (1)
Actual Malignant (0)           41                         1
Actual Benign (1)               1                        71
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
│   ├── training.py             # Pipeline definition and model training
│   ├── evaluation.py           # Metric calculation (accuracy, precision, recall, F1, CM)
│   ├── model_service.py        # Serialization (joblib), artifact reloading, prediction
│   └── main.py                 # Orchestration of training and prediction workflows
├── models/
│   └── model.joblib            # Persisted trained pipeline artifact
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

---

## 8. Code Quality & Pre-commit

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