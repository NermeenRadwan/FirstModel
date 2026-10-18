"""OncoPredict AI - Clinical Diagnostic & Risk Assessment System.

Interactive Streamlit interface for Breast Cancer Wisconsin Diagnostic Classification.
Integrates with FastAPI backend with automatic fallback to local pipeline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import streamlit as st

# Configure wide page layout with medical icon
st.set_page_config(
    page_title="OncoPredict AI | Breast Cancer Diagnostic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Clinical Feature Metadata and Preset Profiles
# -----------------------------------------------------------------------------

FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Mean Cell Features": [
        "mean texture",
        "mean compactness",
        "mean concavity",
        "mean concave points",
    ],
    "Variation & Error Metrics": [
        "radius error",
        "area error",
        "compactness error",
        "concave points error",
        "fractal dimension error",
    ],
    "Worst / Extreme Values": [
        "worst radius",
        "worst texture",
        "worst perimeter",
        "worst area",
        "worst smoothness",
        "worst concavity",
        "worst concave points",
        "worst symmetry",
        "worst fractal dimension",
    ],
}

FEATURE_CONFIG: dict[str, dict[str, Any]] = {
    "mean texture": {
        "label": "Mean Texture",
        "min": 10.0,
        "max": 40.0,
        "step": 0.05,
        "unit": "grayscale std dev",
        "desc": "Standard deviation of gray-scale values across cell nuclei",
    },
    "mean compactness": {
        "label": "Mean Compactness",
        "min": 0.019,
        "max": 0.35,
        "step": 0.001,
        "unit": "perimeter² / area - 1.0",
        "desc": "Nuclear perimeter squared divided by nuclear area minus 1.0",
    },
    "mean concavity": {
        "label": "Mean Concavity",
        "min": 0.0,
        "max": 0.45,
        "step": 0.001,
        "unit": "severity score",
        "desc": "Severity and depth of concave portions of cell nuclei contour",
    },
    "mean concave points": {
        "label": "Mean Concave Points",
        "min": 0.0,
        "max": 0.22,
        "step": 0.001,
        "unit": "count/ratio",
        "desc": "Number and density of concave contour portions",
    },
    "radius error": {
        "label": "Radius Error",
        "min": 0.1,
        "max": 3.0,
        "step": 0.01,
        "unit": "SE (μm)",
        "desc": "Standard error of nuclei radius distances from center to perimeter",
    },
    "area error": {
        "label": "Area Error",
        "min": 5.0,
        "max": 550.0,
        "step": 0.5,
        "unit": "SE (μm²)",
        "desc": "Standard error of cell nuclei area measurements",
    },
    "compactness error": {
        "label": "Compactness Error",
        "min": 0.001,
        "max": 0.15,
        "step": 0.001,
        "unit": "SE",
        "desc": "Standard error for nuclear shape compactness",
    },
    "concave points error": {
        "label": "Concave Points Error",
        "min": 0.0,
        "max": 0.06,
        "step": 0.0005,
        "unit": "SE",
        "desc": "Standard error for concave portions count",
    },
    "fractal dimension error": {
        "label": "Fractal Dimension Error",
        "min": 0.0005,
        "max": 0.035,
        "step": 0.0005,
        "unit": "SE",
        "desc": "Standard error for coastline approximation / perimeter roughness",
    },
    "worst radius": {
        "label": "Worst Radius",
        "min": 7.0,
        "max": 38.0,
        "step": 0.1,
        "unit": "μm (largest)",
        "desc": "Mean of the three largest nuclei radii",
    },
    "worst texture": {
        "label": "Worst Texture",
        "min": 10.0,
        "max": 50.0,
        "step": 0.1,
        "unit": "grayscale (worst)",
        "desc": "Mean of the three largest nuclei textures",
    },
    "worst perimeter": {
        "label": "Worst Perimeter",
        "min": 50.0,
        "max": 255.0,
        "step": 0.5,
        "unit": "μm (largest)",
        "desc": "Mean of the three largest cell perimeters",
    },
    "worst area": {
        "label": "Worst Area",
        "min": 180.0,
        "max": 4300.0,
        "step": 5.0,
        "unit": "μm² (largest)",
        "desc": "Mean of the three largest nuclear areas",
    },
    "worst smoothness": {
        "label": "Worst Smoothness",
        "min": 0.07,
        "max": 0.23,
        "step": 0.001,
        "unit": "local variation",
        "desc": "Local variation in radius lengths among largest nuclei",
    },
    "worst concavity": {
        "label": "Worst Concavity",
        "min": 0.0,
        "max": 1.30,
        "step": 0.005,
        "unit": "severity (worst)",
        "desc": "Highest concavity depths observed across nuclei",
    },
    "worst concave points": {
        "label": "Worst Concave Points",
        "min": 0.0,
        "max": 0.30,
        "step": 0.001,
        "unit": "count (worst)",
        "desc": "Highest density of concave points observed",
    },
    "worst symmetry": {
        "label": "Worst Symmetry",
        "min": 0.15,
        "max": 0.70,
        "step": 0.005,
        "unit": "symmetry index",
        "desc": "Highest nuclear asymmetry observed",
    },
    "worst fractal dimension": {
        "label": "Worst Fractal Dimension",
        "min": 0.05,
        "max": 0.22,
        "step": 0.001,
        "unit": "coastline index",
        "desc": "Highest boundary roughness observed",
    },
}

PRESETS: dict[str, dict[str, float]] = {
    "Typical Benign Case": {
        "mean texture": 14.36,
        "mean compactness": 0.08129,
        "mean concavity": 0.06664,
        "mean concave points": 0.04781,
        "radius error": 0.2699,
        "area error": 23.56,
        "compactness error": 0.0146,
        "concave points error": 0.01315,
        "fractal dimension error": 0.0023,
        "worst radius": 15.11,
        "worst texture": 19.26,
        "worst perimeter": 99.7,
        "worst area": 711.2,
        "worst smoothness": 0.144,
        "worst concavity": 0.239,
        "worst concave points": 0.1288,
        "worst symmetry": 0.2977,
        "worst fractal dimension": 0.07259,
    },
    "High-Risk Malignant Case": {
        "mean texture": 10.38,
        "mean compactness": 0.2776,
        "mean concavity": 0.3001,
        "mean concave points": 0.1471,
        "radius error": 1.095,
        "area error": 153.4,
        "compactness error": 0.04904,
        "concave points error": 0.01587,
        "fractal dimension error": 0.006193,
        "worst radius": 25.38,
        "worst texture": 17.33,
        "worst perimeter": 184.6,
        "worst area": 2019.0,
        "worst smoothness": 0.1622,
        "worst concavity": 0.7119,
        "worst concave points": 0.2654,
        "worst symmetry": 0.4601,
        "worst fractal dimension": 0.1189,
    },
    "Borderline Case": {
        "mean texture": 18.5,
        "mean compactness": 0.12,
        "mean concavity": 0.11,
        "mean concave points": 0.065,
        "radius error": 0.45,
        "area error": 45.0,
        "compactness error": 0.026,
        "concave points error": 0.012,
        "fractal dimension error": 0.0035,
        "worst radius": 17.5,
        "worst texture": 25.0,
        "worst perimeter": 115.0,
        "worst area": 920.0,
        "worst smoothness": 0.138,
        "worst concavity": 0.32,
        "worst concave points": 0.135,
        "worst symmetry": 0.31,
        "worst fractal dimension": 0.085,
    },
}

# -----------------------------------------------------------------------------
# Custom Theme Styling (CSS)
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Header Card */
.hero-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    color: #F8FAFC;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero-subtitle {
    font-size: 0.98rem;
    color: #94A3B8;
    margin-bottom: 14px;
}

/* Badges and Chips */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.chip-blue {
    background: rgba(56, 189, 248, 0.12);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.25);
}
.chip-green {
    background: rgba(16, 185, 129, 0.12);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.chip-purple {
    background: rgba(168, 85, 247, 0.12);
    color: #C084FC;
    border: 1px solid rgba(168, 85, 247, 0.25);
}

/* Result Cards */
.result-card-benign {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.3) 0%, rgba(4, 120, 87, 0.15) 100%);
    border: 2px solid #10B981;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 0 35px rgba(16, 185, 129, 0.2);
    margin-bottom: 20px;
}
.result-card-malignant {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(185, 28, 28, 0.2) 100%);
    border: 2px solid #EF4444;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 0 35px rgba(239, 68, 68, 0.25);
    margin-bottom: 20px;
}

.status-headline {
    font-size: 1.6rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.benign-text {
    color: #34D399;
}
.malignant-text {
    color: #F87171;
}

/* Confidence Meters */
.metric-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 14px;
}
.metric-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

/* Recommendation Box */
.recommendation-card {
    background: rgba(30, 41, 59, 0.5);
    border-left: 4px solid #38BDF8;
    border-radius: 8px 14px 14px 8px;
    padding: 16px 20px;
    margin-top: 16px;
    font-size: 0.92rem;
    line-height: 1.5;
}
.recommendation-title {
    font-weight: 700;
    color: #38BDF8;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Feature category headers */
.category-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid rgba(56, 189, 248, 0.3);
}

/* Clean Streamlit tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 600;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State Initialization
# -----------------------------------------------------------------------------

if "features" not in st.session_state:
    st.session_state.features = dict(PRESETS["Typical Benign Case"])


def apply_preset(preset_name: str) -> None:
    """Load a predefined clinical profile into the active inputs."""
    st.session_state.features = dict(PRESETS[preset_name])


# -----------------------------------------------------------------------------
# Prediction Engine (FastAPI with Automatic Pipeline Fallback)
# -----------------------------------------------------------------------------

def query_prediction(
    features: dict[str, float],
    api_url: str = "http://127.0.0.1:8000/predict",
) -> dict[str, Any]:
    """Execute prediction via FastAPI REST service or fallback to local pipeline."""
    payload = json.dumps({"features": features}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            data["backend"] = "FastAPI REST Server (Port 8000)"
            return data
    except (urllib.error.URLError, TimeoutError, OSError):
        # Fallback to local model service seamlessly
        try:
            from production_app.config import load_settings
            from production_app.model_service import load_model, predict, predict_proba

            settings = load_settings()
            pipeline = load_model(settings.model_path)
            input_df = pd.DataFrame([features])

            pred_arr = predict(pipeline, input_df)
            probs_arr = predict_proba(pipeline, input_df)[0]

            pred_val = int(pred_arr[0])
            label = "benign" if pred_val == 1 else "malignant"
            prob_mal = float(probs_arr[0])
            prob_ben = float(probs_arr[1])
            top_prob = prob_ben if pred_val == 1 else prob_mal

            clinical_rec = (
                "Routine follow-up screening recommended."
                if pred_val == 1
                else "High risk of malignancy detected. Urgent biopsy confirmation and oncology consultation recommended."
            )

            return {
                "prediction": pred_val,
                "label": label,
                "probability": round(top_prob, 4),
                "confidence_percentage": f"{top_prob * 100:.2f}%",
                "risk_level": "Low Risk" if pred_val == 1 else "High Risk",
                "probabilities": {
                    "malignant": round(prob_mal, 4),
                    "benign": round(prob_ben, 4),
                },
                "clinical_recommendation": clinical_rec,
                "backend": "Local Scikit-Learn Pipeline (Direct)",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "error": f"Failed to predict via both API and local fallback: {exc}",
                "backend": "None",
            }


# -----------------------------------------------------------------------------
# Top Hero Header
# -----------------------------------------------------------------------------

st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">🔬 OncoPredict AI</div>
    <div class="hero-subtitle">
        Clinical Diagnostic & Risk Assessment System &bull; Breast Cancer Wisconsin Fine-Needle Aspirate (FNA) Analysis
    </div>
    <div class="badge-container">
        <span class="chip chip-blue">🤖 LogisticRegression + RobustScaler</span>
        <span class="chip chip-green">🎯 99.12% Test Accuracy</span>
        <span class="chip chip-purple">⚡ 18 Clinically Selected Biomarkers</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Presets and Configuration
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📋 Clinical Presets")
    st.caption("Load verified test profiles to observe diagnostic response:")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🟢 Benign Profile", use_container_width=True):
            apply_preset("Typical Benign Case")
            st.rerun()

    with col_b2:
        if st.button("🔴 Malignant Profile", use_container_width=True):
            apply_preset("High-Risk Malignant Case")
            st.rerun()

    if st.button("⚖️ Borderline Case", use_container_width=True):
        apply_preset("Borderline Case")
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    api_endpoint = st.text_input(
        "FastAPI Server URL",
        value=os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict"),
        help="REST endpoint for inference. Automatically falls back to in-memory model if offline.",
    )

    st.markdown("---")
    st.markdown("### ℹ️ About the Model")
    st.markdown(
        """
        - **Dataset**: WDBC (Wisconsin Diagnostic)
        - **Pipeline**: `RobustScaler` $\\rightarrow$ `LogisticRegression`
        - **Metric Highlights**:
          - **Accuracy**: `99.12%`
          - **Recall (Sensitivity)**: `100.00%`
          - **Precision**: `98.63%`
          - **F1 Score**: `99.31%`
        """
    )
    st.caption("Developed for clinical decision-support paired with expert pathology.")


# -----------------------------------------------------------------------------
# Main Application Layout: Input Controls (Left) & Diagnostic Output (Right)
# -----------------------------------------------------------------------------

left_col, right_col = st.columns([1.1, 1.0], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN: Interactive Feature Sliders
# -----------------------------------------------------------------------------

with left_col:
    st.markdown("### 🔬 Patient Cell Biometric Measurements")
    st.caption("Adjust cellular biomarkers obtained from fine-needle aspirate (FNA) digital imaging:")

    tabs = st.tabs(list(FEATURE_CATEGORIES.keys()))

    for tab_idx, (cat_name, feature_list) in enumerate(FEATURE_CATEGORIES.items()):
        with tabs[tab_idx]:
            for feat_name in feature_list:
                cfg = FEATURE_CONFIG[feat_name]
                current_val = float(st.session_state.features.get(feat_name, cfg["min"]))

                # Ensure current_val is clamped within valid bounds
                clamped_val = max(cfg["min"], min(cfg["max"], current_val))

                val = st.slider(
                    label=f"{cfg['label']} ({cfg['unit']})",
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(clamped_val),
                    step=float(cfg["step"]),
                    help=cfg["desc"],
                    key=f"slider_{feat_name}",
                )
                st.session_state.features[feat_name] = val

# -----------------------------------------------------------------------------
# RIGHT COLUMN: Diagnostic Results & Intelligence Cards
# -----------------------------------------------------------------------------

with right_col:
    st.markdown("### 📊 Diagnostic Intelligence & Risk Assessment")

    # Run Prediction
    result = query_prediction(st.session_state.features, api_url=api_endpoint)

    if "error" in result:
        st.error(f"Prediction Error: {result['error']}")
    else:
        is_benign = result["prediction"] == 1
        label = result["label"].upper()
        prob_pct = result["confidence_percentage"]
        risk_level = result["risk_level"]
        prob_benign = result["probabilities"]["benign"] * 100
        prob_malignant = result["probabilities"]["malignant"] * 100
        backend_name = result.get("backend", "Unknown")

        # 1. Big Diagnostic Banner
        if is_benign:
            st.markdown(
                f"""
            <div class="result-card-benign">
                <div class="status-headline benign-text">
                    <span>🛡️ DIAGNOSIS: {label}</span>
                </div>
                <div style="font-size: 1.05rem; color: #D1FAE5; margin-bottom: 12px;">
                    No evidence of malignancy detected in sampled cellular morphology.
                </div>
                <div class="badge-container">
                    <span class="chip chip-green">✓ {risk_level}</span>
                    <span class="chip chip-blue">Confidence: {prob_pct}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="result-card-malignant">
                <div class="status-headline malignant-text">
                    <span>⚠️ DIAGNOSIS: {label}</span>
                </div>
                <div style="font-size: 1.05rem; color: #FEE2E2; margin-bottom: 12px;">
                    Morphological anomalies consistent with malignant neoplasm detected.
                </div>
                <div class="badge-container">
                    <span class="chip" style="background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4);">
                        🚨 {risk_level}
                    </span>
                    <span class="chip chip-blue">Confidence: {prob_pct}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # 2. Probability Distribution Gauges
        st.markdown("#### 📈 Probability Distribution")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric(
                label="Benign Probability",
                value=f"{prob_benign:.2f}%",
                delta=f"{'+' if is_benign else '-'}{abs(prob_benign - 50.0):.1f}% vs baseline",
            )
            st.progress(float(result["probabilities"]["benign"]))

        with col_m2:
            st.metric(
                label="Malignant Probability",
                value=f"{prob_malignant:.2f}%",
                delta=f"{'+' if not is_benign else '-'}{abs(prob_malignant - 50.0):.1f}% vs baseline",
                delta_color="inverse",
            )
            st.progress(float(result["probabilities"]["malignant"]))

        # 3. Clinical Recommendation Box
        st.markdown(
            f"""
        <div class="recommendation-card">
            <div class="recommendation-title">
                🩺 Clinical Protocol & Next Steps:
            </div>
            <div style="color: #CBD5E1;">
                {result["clinical_recommendation"]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Biomarker Comparison Chart
        st.markdown("#### 🔬 Relative Biomarker Profile (Key Indicators)")
        chart_features = [
            "worst radius",
            "worst perimeter",
            "worst area",
            "worst concavity",
            "worst concave points",
        ]
        # Normalize each selected feature relative to max scale for clean comparative visual
        norm_data = []
        for feat in chart_features:
            cfg = FEATURE_CONFIG[feat]
            val = st.session_state.features[feat]
            ratio = (val - cfg["min"]) / (cfg["max"] - cfg["min"]) * 100.0
            norm_data.append({"Biomarker": cfg["label"], "Intensity Score (%)": round(ratio, 1)})

        chart_df = pd.DataFrame(norm_data).set_index("Biomarker")
        st.bar_chart(chart_df, color="#38BDF8")
        st.caption("Bar chart displays relative biomarker intensity against normalized population spectrum.")

        # 5. Export and Raw Payload
        st.markdown("---")
        col_exp1, col_exp2 = st.columns([1, 1])

        with col_exp1:
            export_payload = {
                "assessment_result": result,
                "input_biomarkers": st.session_state.features,
                "metadata": {
                    "model": "RobustScaler + LogisticRegression",
                    "validation_accuracy": "99.12%",
                },
            }
            st.download_button(
                label="📥 Export Clinical Report (JSON)",
                data=json.dumps(export_payload, indent=2),
                file_name="oncopredict_diagnostic_report.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_exp2:
            st.caption(f"Connected Backend: `{backend_name}`")

        with st.expander("🔍 View Raw API Request & Response"):
            st.json(export_payload)
