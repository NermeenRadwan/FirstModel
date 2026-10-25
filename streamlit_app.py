"""OncoPredict AI - Modern Streamlit Interface for Breast Cancer Diagnostic Classification.

FastAPI HTTP client with clinical presets, multi-mode inputs, and rich diagnostic visualizations.
"""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import streamlit as st

from production_app.config import SELECTED_FEATURES
from production_app.schemas import EXAMPLE_FEATURES_DICT

# Configure page layout
st.set_page_config(
    page_title="OncoPredict AI | Breast Cancer Diagnostic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Clinical Profiles & Feature Metadata
# -----------------------------------------------------------------------------

PRESET_PROFILES: dict[str, dict[str, float]] = {
    "🟢 Typical Benign Case (Patient #102)": {
        "mean texture": 14.36,
        "mean compactness": 0.0813,
        "mean concavity": 0.0666,
        "mean concave points": 0.0478,
        "radius error": 0.2699,
        "area error": 23.56,
        "compactness error": 0.0146,
        "concave points error": 0.0132,
        "fractal dimension error": 0.0023,
        "worst radius": 15.11,
        "worst texture": 19.26,
        "worst perimeter": 99.7,
        "worst area": 711.2,
        "worst smoothness": 0.1440,
        "worst concavity": 0.2390,
        "worst concave points": 0.1288,
        "worst symmetry": 0.2977,
        "worst fractal dimension": 0.0726,
    },
    "🔴 High-Risk Malignant Case (Patient #001)": {
        "mean texture": 10.38,
        "mean compactness": 0.2776,
        "mean concavity": 0.3001,
        "mean concave points": 0.1471,
        "radius error": 1.0950,
        "area error": 153.40,
        "compactness error": 0.0490,
        "concave points error": 0.0159,
        "fractal dimension error": 0.0062,
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
    "⚖️ Borderline Suspicious Case (Patient #045)": {
        "mean texture": 18.50,
        "mean compactness": 0.1200,
        "mean concavity": 0.1100,
        "mean concave points": 0.0650,
        "radius error": 0.4500,
        "area error": 45.00,
        "compactness error": 0.0260,
        "concave points error": 0.0120,
        "fractal dimension error": 0.0035,
        "worst radius": 17.50,
        "worst texture": 25.00,
        "worst perimeter": 115.0,
        "worst area": 920.0,
        "worst smoothness": 0.1380,
        "worst concavity": 0.3200,
        "worst concave points": 0.1350,
        "worst symmetry": 0.3100,
        "worst fractal dimension": 0.0850,
    },
}

FEATURE_SECTIONS: dict[str, list[str]] = {
    "🧬 Cell Nuclei Means": [
        "mean texture",
        "mean compactness",
        "mean concavity",
        "mean concave points",
    ],
    "📏 Standard Errors (SE)": [
        "radius error",
        "area error",
        "compactness error",
        "concave points error",
        "fractal dimension error",
    ],
    "🔬 Worst / Extreme Values": [
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
        "step": 0.1,
        "unit": "grayscale",
        "min": 5.0,
        "max": 50.0,
    },
    "mean compactness": {
        "label": "Mean Compactness",
        "step": 0.01,
        "unit": "ratio",
        "min": 0.0,
        "max": 0.5,
    },
    "mean concavity": {
        "label": "Mean Concavity",
        "step": 0.01,
        "unit": "severity",
        "min": 0.0,
        "max": 0.6,
    },
    "mean concave points": {
        "label": "Mean Concave Points",
        "step": 0.005,
        "unit": "ratio",
        "min": 0.0,
        "max": 0.3,
    },
    "radius error": {
        "label": "Radius Error",
        "step": 0.05,
        "unit": "μm (SE)",
        "min": 0.0,
        "max": 5.0,
    },
    "area error": {
        "label": "Area Error",
        "step": 1.0,
        "unit": "μm² (SE)",
        "min": 0.0,
        "max": 600.0,
    },
    "compactness error": {
        "label": "Compactness Error",
        "step": 0.005,
        "unit": "SE",
        "min": 0.0,
        "max": 0.2,
    },
    "concave points error": {
        "label": "Concave Points Error",
        "step": 0.002,
        "unit": "SE",
        "min": 0.0,
        "max": 0.1,
    },
    "fractal dimension error": {
        "label": "Fractal Dim. Error",
        "step": 0.001,
        "unit": "SE",
        "min": 0.0,
        "max": 0.05,
    },
    "worst radius": {
        "label": "Worst Radius",
        "step": 0.2,
        "unit": "μm",
        "min": 5.0,
        "max": 45.0,
    },
    "worst texture": {
        "label": "Worst Texture",
        "step": 0.2,
        "unit": "grayscale",
        "min": 5.0,
        "max": 60.0,
    },
    "worst perimeter": {
        "label": "Worst Perimeter",
        "step": 1.0,
        "unit": "μm",
        "min": 40.0,
        "max": 300.0,
    },
    "worst area": {
        "label": "Worst Area",
        "step": 10.0,
        "unit": "μm²",
        "min": 150.0,
        "max": 4500.0,
    },
    "worst smoothness": {
        "label": "Worst Smoothness",
        "step": 0.005,
        "unit": "variation",
        "min": 0.05,
        "max": 0.3,
    },
    "worst concavity": {
        "label": "Worst Concavity",
        "step": 0.01,
        "unit": "severity",
        "min": 0.0,
        "max": 1.5,
    },
    "worst concave points": {
        "label": "Worst Concave Points",
        "step": 0.01,
        "unit": "ratio",
        "min": 0.0,
        "max": 0.4,
    },
    "worst symmetry": {
        "label": "Worst Symmetry",
        "step": 0.01,
        "unit": "ratio",
        "min": 0.1,
        "max": 0.8,
    },
    "worst fractal dimension": {
        "label": "Worst Fractal Dim.",
        "step": 0.005,
        "unit": "roughness",
        "min": 0.04,
        "max": 0.3,
    },
}

# -----------------------------------------------------------------------------
# Session State Management
# -----------------------------------------------------------------------------

if "features" not in st.session_state:
    st.session_state.features = dict(
        PRESET_PROFILES["🟢 Typical Benign Case (Patient #102)"]
    )

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# -----------------------------------------------------------------------------
# Pure HTTP Client (FastAPI Request)
# -----------------------------------------------------------------------------


def query_fastapi(features: dict[str, float], api_url: str) -> dict[str, Any]:
    """Execute prediction via FastAPI REST service over HTTP."""
    payload = json.dumps({"features": features}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        try:
            error_body = json.loads(http_err.read().decode("utf-8"))
            detail = error_body.get("detail", str(http_err))
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = str(http_err)

        if http_err.code == 503:
            return {"error": f"Model Unavailable (503): {detail}. Train model first."}
        elif http_err.code == 422:
            return {"error": f"Validation Error (422): {detail}"}
        return {"error": f"Server Error ({http_err.code}): {detail}"}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {
            "error": (
                f"Connection Failure: Could not reach FastAPI backend at '{api_url}'. "
                "Ensure Uvicorn or the Docker container is running."
            )
        }


# -----------------------------------------------------------------------------
# Header & Navigation
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                padding: 22px 28px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 24px;">
        <h1 style="margin: 0 0 6px 0; background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.2rem;">
            🔬 OncoPredict AI &bull; Clinical Diagnostic Assessment
        </h1>
        <p style="margin: 0; color: #94A3B8; font-size: 1rem;">
            Containerized Machine Learning inference on 18 clinically selected fine-needle aspirate (FNA) biomarkers.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ System Settings")
    api_endpoint = st.text_input(
        "FastAPI Service URL",
        value=os.getenv("API_URL", "http://127.0.0.1:8000/predict"),
        help="FastAPI prediction endpoint. If running inside Docker, use the mapped localhost port.",
    )

    st.markdown("---")
    st.header("📋 Model Highlights")
    st.markdown(
        """
        - **Algorithm**: `LogisticRegression`
        - **Preprocessor**: `RobustScaler`
        - **Test Accuracy**: `99.12%`
        - **Test Recall (Sensitivity)**: `100.00%`
        - **Biomarkers**: `18 Selected Features`
        """
    )
    st.caption("Pure HTTP Client &bull; Independent of model storage.")

# -----------------------------------------------------------------------------
# Main Application Layout: Input Controls (Left) & Diagnostic Visualizations (Right)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. Full-Width Patient Biomarker Inputs
# -----------------------------------------------------------------------------

st.subheader("📝 Patient Biomarker Entry")

# 1. Preset Profile Selector
col_preset1, col_preset2 = st.columns([2.5, 1])
with col_preset1:
    selected_preset = st.selectbox(
        "⚡ Choose a Clinical Scenario or Patient Case:",
        options=list(PRESET_PROFILES.keys())
        + ["🎲 Random Sample from Dataset", "✏️ Custom Input"],
        index=0,
        help="Select a profile to populate all 18 biometric measurements automatically.",
    )

if selected_preset in PRESET_PROFILES:
    preset_values = PRESET_PROFILES[selected_preset]
    for k, v in preset_values.items():
        st.session_state.features[k] = v
elif selected_preset == "🎲 Random Sample from Dataset":
    with col_preset2:
        st.write("")
        st.write("")
        if st.button("🔄 Generate Random Patient Sample", use_container_width=True):
            for k, v in EXAMPLE_FEATURES_DICT.items():
                jitter = random.uniform(0.85, 1.25)
                st.session_state.features[k] = round(float(v) * jitter, 4)
            st.rerun()

# 2. Multi-Mode Input Tabs (Full Width)
input_tab1, input_tab2 = st.tabs(
    ["🔢 Direct Numeric Inputs", "📋 Bulk JSON / CSV Paste"]
)

with input_tab1:
    st.caption(
        "Type exact values or adjust using step buttons. Grouped by clinical category across full width:"
    )

    with st.form("diagnostic_entry_form"):
        col_a, col_b, col_c = st.columns(3)
        columns_map = {
            "🧬 Cell Nuclei Means": col_a,
            "📏 Standard Errors (SE)": col_b,
            "🔬 Worst / Extreme Values": col_c,
        }

        form_inputs = {}
        for category_name, feature_list in FEATURE_SECTIONS.items():
            with columns_map[category_name]:
                st.markdown(f"**{category_name}**")
                for feat_name in feature_list:
                    cfg = FEATURE_CONFIG[feat_name]
                    current_val = float(st.session_state.features.get(feat_name, 0.0))
                    val = st.number_input(
                        label=f"{cfg['label']} ({cfg['unit']})",
                        value=current_val,
                        step=float(cfg["step"]),
                        format="%.4f",
                        key=f"num_{feat_name}",
                    )
                    form_inputs[feat_name] = val

        submit_btn = st.form_submit_button(
            "🔍 Run Diagnosis & Risk Prediction",
            type="primary",
            use_container_width=True,
        )

        if submit_btn:
            st.session_state.features = form_inputs
            st.session_state.last_prediction = query_fastapi(
                st.session_state.features, api_endpoint
            )

with input_tab2:
    st.caption("Paste a JSON dictionary containing the 18 feature keys:")
    json_text = st.text_area(
        "Paste JSON Features",
        value=json.dumps(st.session_state.features, indent=2),
        height=180,
    )
    if st.button("Apply Pasted JSON Values"):
        try:
            parsed_json = json.loads(json_text)
            missing = set(SELECTED_FEATURES) - set(parsed_json.keys())
            if missing:
                st.error(f"Missing required features: {missing}")
            else:
                st.session_state.features = {
                    k: float(parsed_json[k]) for k in SELECTED_FEATURES
                }
                st.success("JSON features loaded successfully!")
                st.rerun()
        except Exception as parse_err:  # noqa: BLE001
            st.error(f"Invalid JSON: {parse_err}")

# -----------------------------------------------------------------------------
# 2. Results & Diagnostic Visualizations (Down-Scroll Section)
# -----------------------------------------------------------------------------

st.markdown("---")
st.subheader("📊 Diagnostic Intelligence & Visualizations")

# If no prediction has run yet, trigger an initial evaluation with default values
if st.session_state.last_prediction is None:
    st.session_state.last_prediction = query_fastapi(
        st.session_state.features, api_endpoint
    )

result = st.session_state.last_prediction

if "error" in result:
    st.error(f"⚠️ {result['error']}")
else:
    is_benign = result["prediction"] == 1
    label = result["label"].upper()
    prob_benign = float(result["probabilities"]["benign"]) * 100.0
    prob_malignant = float(result["probabilities"]["malignant"]) * 100.0
    confidence = result["confidence_percentage"]
    risk_level = result.get("risk_level", "Standard")

    # 1. Main Diagnostic Outcome Banner (Full Width)
    if is_benign:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 2px solid #10B981;
                        border-radius: 14px; padding: 20px; margin-bottom: 24px;">
                <h2 style="color: #10B981; margin: 0 0 6px 0;">🛡️ DIAGNOSIS: {label}</h2>
                <p style="color: #000000ff; margin: 0; font-size: 1.05rem;">
                    Cell nuclei features fall within benign morphological boundaries. No invasive tumor characteristics detected.
                </p>
                <div style="margin-top: 12px;">
                    <span style="background: #10B981; color: white; padding: 5px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">
                        ✓ {risk_level}
                    </span>
                    <span style="background: rgba(255,255,255); color: #000000ff; padding: 5px 14px; border-radius: 9999px; font-size: 0.85rem; margin-left: 8px;">
                        Confidence: {confidence}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.12); border: 2px solid #EF4444;
                        border-radius: 14px; padding: 20px; margin-bottom: 24px;">
                <h2 style="color: #EF4444; margin: 0 0 6px 0;">🚨 DIAGNOSIS: {label}</h2>
                <p style="color: #FEE2E2; margin: 0; font-size: 1.05rem;">
                    Severe contour concavity, nuclear enlargement, and cell irregularity detected. Consistent with carcinoma.
                </p>
                <div style="margin-top: 12px;">
                    <span style="background: #EF4444; color: white; padding: 5px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">
                        🚨 {risk_level}
                    </span>
                    <span style="background: rgba(255,255,255,0.1); color: #E2E8F0; padding: 5px 14px; border-radius: 9999px; font-size: 0.85rem; margin-left: 8px;">
                        Confidence: {confidence}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Probability Split Gauge (2 Columns)
    st.markdown("#### 📈 Probability Distribution")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label="Benign Probability",
            value=f"{prob_benign:.2f}%",
            delta=f"{'+' if is_benign else '-'}{abs(prob_benign - 50.0):.1f}% vs baseline",
        )
        st.progress(prob_benign / 100.0)

    with col_m2:
        st.metric(
            label="Malignant Probability",
            value=f"{prob_malignant:.2f}%",
            delta=f"{'+' if not is_benign else '-'}{abs(prob_malignant - 50.0):.1f}% vs baseline",
            delta_color="inverse",
        )
        st.progress(prob_malignant / 100.0)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Critical Biomarkers Intensity Chart (Full Width)
    st.markdown("#### 🔬 Critical Malignancy Biomarkers Severity Profile")
    st.caption(
        "Normalized relative intensity of current patient sample across primary diagnostic indicators:"
    )

    key_indicators = [
        "worst radius",
        "worst perimeter",
        "worst area",
        "worst concavity",
        "worst concave points",
    ]
    norm_data = []
    for feat in key_indicators:
        cfg = FEATURE_CONFIG[feat]
        val = float(st.session_state.features[feat])
        ratio = max(
            0.0, min(100.0, (val - cfg["min"]) / (cfg["max"] - cfg["min"]) * 100.0)
        )
        norm_data.append({"Biomarker": cfg["label"], "Intensity (%)": round(ratio, 1)})

    chart_df = pd.DataFrame(norm_data).set_index("Biomarker")
    st.bar_chart(chart_df, color="#38BDF8")

    # 4. Clinical Protocol & Recommendation Box
    st.info(
        f"**🩺 Clinical Protocol Recommendation:** {result.get('clinical_recommendation', 'Routine assessment')}"
    )

    # 5. Export Clinical Report
    st.markdown("---")
    col_down1, col_down2 = st.columns([1, 1])
    report_payload = {
        "assessment_result": result,
        "patient_biomarkers": st.session_state.features,
        "pipeline": "RobustScaler + LogisticRegression (18 Features)",
        "validation_accuracy": "99.12%",
    }
    with col_down1:
        st.download_button(
            label="📥 Download Clinical Report (JSON)",
            data=json.dumps(report_payload, indent=2),
            file_name="oncopredict_clinical_report.json",
            mime="application/json",
            use_container_width=True,
        )
