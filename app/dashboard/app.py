from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------
# Paths
# ---------------------------
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "phase5"
PROCESSED = ROOT / "data" / "processed"

METRICS_JSON = REPORTS / "metrics.json"
UNCERTAINTY_JSON = REPORTS / "uncertainty.json"
EXPLAIN_SUMMARY_JSON = REPORTS / "explainability_summary.json"

OFFICIAL_PRED_CSV = REPORTS / "official_test_predictions.csv"
INTERNAL_PRED_CSV = REPORTS / "internal_test_predictions_calibrated.csv"
VALIDATION_PRED_CSV = REPORTS / "validation_predictions_calibrated.csv"

GLOBAL_IMPORTANCE_CSV = REPORTS / "explanations" / "global_feature_importance.csv"
LOCAL_EXPLANATIONS_CSV = REPORTS / "explanations" / "local_explanations.csv"
SPLIT_CSV = PROCESSED / "mendeley_internal_split.csv"

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="EV Battery Thermal Anomaly Control Room",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 50%, #0ea5e9 100%);
    color: white;
    padding: 1.4rem 1.5rem;
    border-radius: 22px;
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.18);
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    font-size: 2rem;
    line-height: 1.15;
}

.hero p {
    margin: 0.4rem 0 0 0;
    color: rgba(255,255,255,0.88);
    font-size: 0.98rem;
}

.status-pill {
    display: inline-block;
    margin-top: 0.85rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.20);
}

.metric-card {
    background: #ffffff;
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 18px;
    padding: 1rem 1rem 0.9rem 1rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    min-height: 110px;
}

.metric-title {
    font-size: 0.82rem;
    font-weight: 800;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.metric-subtitle {
    font-size: 0.86rem;
    color: #475569;
}

.section-card {
    background: #ffffff;
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}

.check-ok { color: #16a34a; font-weight: 700; }
.check-miss { color: #dc2626; font-weight: 700; }
.small-note { color: #64748b; font-size: 0.88rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(x, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"


def pct(x, digits: int = 2) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "NA"


def metric_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">{title}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-subtitle">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def confusion_heatmap(cm: np.ndarray, title: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred Normal", "Pred Abnormal"],
            y=["True Normal", "True Abnormal"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=420, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def roc_pr_figures(df: pd.DataFrame):
    if df.empty or "label" not in df.columns or "anomaly_probability" not in df.columns:
        return None, None

    y_true = df["label"].astype(int).values
    y_prob = df["anomaly_probability"].astype(float).values
    if len(np.unique(y_true)) < 2:
        return None, None

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420, margin=dict(l=20, r=20, t=50, b=20))

    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
    pr_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=420, margin=dict(l=20, r=20, t=50, b=20))

    return roc_fig, pr_fig


def reliability_fig(y_true: np.ndarray, y_prob: np.ndarray):
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
    fig.update_layout(title="Reliability Curve", xaxis_title="Mean predicted probability", yaxis_title="Fraction of positives", height=420, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def confidence_hist(df: pd.DataFrame):
    if df.empty or "confidence" not in df.columns:
        return None
    fig = px.histogram(df, x="confidence", nbins=25, title="Confidence Distribution")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def feature_importance_fig(df: pd.DataFrame):
    if df.empty or "feature" not in df.columns or "importance" not in df.columns:
        return None
    top = df.sort_values("importance", ascending=False).head(20).sort_values("importance")
    fig = px.bar(top, x="importance", y="feature", orientation="h", title="Top 20 Global Feature Importances")
    fig.update_layout(height=620, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def local_explanation_fig(df: pd.DataFrame, sample_id):
    if df.empty or "sample_id" not in df.columns:
        return None, pd.DataFrame()
    sdf = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    if sdf.empty:
        return None, pd.DataFrame()
    sdf = sdf.sort_values("weighted_score", ascending=False).head(12)
    fig = px.bar(sdf.sort_values("weighted_score", ascending=True), x="weighted_score", y="feature", orientation="h", title=f"Local Explanation — Sample {sample_id}")
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
    return fig, sdf


def safety_score(metrics: dict, uncertainty: dict) -> float:
    f1 = float(metrics.get("internal_test", {}).get("f1", 0.0) or 0.0)
    ece = float(uncertainty.get("ece", 1.0) or 1.0)
    return max(0.0, min(100.0, 0.6 * f1 * 100.0 + 0.4 * (1.0 - ece) * 100.0))


def threshold_preview(df: pd.DataFrame, threshold: float) -> dict:
    if df.empty or "anomaly_probability" not in df.columns:
        return {}

    y_pred = (df["anomaly_probability"].astype(float).values >= threshold).astype(int)
    out = {
        "predicted_abnormal": int((y_pred == 1).sum()),
        "predicted_normal": int((y_pred == 0).sum()),
    }

    if "label" in df.columns and df["label"].notna().any():
        y_true = df["label"].astype(int).values
        out.update(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            }
        )
    return out


# ---------------------------
# Load artifacts
# ---------------------------
metrics = load_json(METRICS_JSON)
uncertainty = load_json(UNCERTAINTY_JSON)
explain_summary = load_json(EXPLAIN_SUMMARY_JSON)

split_df = load_csv(SPLIT_CSV)
validation_df = load_csv(VALIDATION_PRED_CSV)
internal_df = load_csv(INTERNAL_PRED_CSV)
official_df = load_csv(OFFICIAL_PRED_CSV)
global_imp = load_csv(GLOBAL_IMPORTANCE_CSV)
local_exp = load_csv(LOCAL_EXPLANATIONS_CSV)

best_threshold = float(metrics.get("best_threshold", 0.51) or 0.51)
confidence_cutoff = float(uncertainty.get("confidence_cutoff", 0.80) or 0.80)
readiness = safety_score(metrics, uncertainty)
status_text = "READY FOR DEMO" if readiness >= 85 else "NEEDS REVIEW"

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.markdown("## Control Panel")
st.sidebar.caption("Dashboard only. No retraining.")

threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, float(best_threshold), 0.01)
top_n = st.sidebar.slider("Top anomalies to show", 5, 50, 10)

sample_choices = []
if not local_exp.empty and "sample_id" in local_exp.columns:
    # Prefer the most anomalous samples for demo
    ranked = (
        local_exp.groupby("sample_id", as_index=False)["anomaly_probability"]
        .max()
        .sort_values("anomaly_probability", ascending=False)
    )
    sample_choices = ranked["sample_id"].astype(str).tolist()

sample_id = None
if sample_choices:
    sample_id = st.sidebar.selectbox("Sample for local explanation", options=sample_choices, index=0)

with st.sidebar.expander("Artifact check", expanded=False):
    checks = [
        ("metrics.json", METRICS_JSON),
        ("uncertainty.json", UNCERTAINTY_JSON),
        ("explainability_summary.json", EXPLAIN_SUMMARY_JSON),
        ("official_test_predictions.csv", OFFICIAL_PRED_CSV),
        ("global_feature_importance.csv", GLOBAL_IMPORTANCE_CSV),
        ("local_explanations.csv", LOCAL_EXPLANATIONS_CSV),
    ]
    for label, path in checks:
        status = "<span class='check-ok'>OK</span>" if path.exists() else "<span class='check-miss'>Missing</span>"
        st.markdown(f"- {label}: {status}", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Phase-5 readiness score:** `{readiness:.1f}/100`")
st.sidebar.markdown("Built from saved artifacts only.")

# ---------------------------
# Hero
# ---------------------------
st.markdown(
    f"""
<div class="hero">
    <h1>Edge AI EV Battery Thermal Anomaly Early Warning System</h1>
    <p>Supervised labeled anomaly detection + uncertainty-aware diagnosis + explainability.</p>
    <div class="status-pill">{status_text}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# KPI Cards
# ---------------------------
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("Internal Test F1", fmt(metrics.get("internal_test", {}).get("f1", 0.0)))
with m2:
    metric_card("ROC-AUC", fmt(metrics.get("internal_test", {}).get("roc_auc", 0.0)))
with m3:
    metric_card("PR-AUC", fmt(metrics.get("internal_test", {}).get("pr_auc", 0.0)))
with m4:
    metric_card("ECE", fmt(uncertainty.get("ece", 0.0)))
with m5:
    metric_card("Selective Accuracy", fmt(uncertainty.get("selective_accuracy_at_confidence_cutoff", 0.0)))

s1, s2, s3, s4 = st.columns(4)
with s1:
    metric_card("Best Threshold", fmt(best_threshold), "Chosen from validation F1")
with s2:
    metric_card("Confidence Cutoff", fmt(confidence_cutoff), "For selective decisions")
with s3:
    metric_card("Brier Score", fmt(uncertainty.get("brier_score", 0.0)))
with s4:
    metric_card("Coverage", pct(uncertainty.get("coverage_at_confidence_cutoff", 0.0)))

tabs = st.tabs(["Overview", "Performance", "Uncertainty", "Explainability", "Predictions"])

# ---------------------------
# Overview
# ---------------------------
with tabs[0]:
    c1, c2 = st.columns([1.15, 0.85])

    with c1:
        st.markdown("### Project Story")
        st.markdown(
            """
- **Goal:** detect EV battery thermal anomalies early.
- **Core methods:** supervised anomaly detection, calibration, uncertainty estimation, explainability.
- **Demo value:** show what is wrong, how confident the model is, and which features drive the alarm.
- **Tata fit:** a working edge-AI battery health prototype with measurable results.
"""
        )

        if not official_df.empty and "anomaly_probability" in official_df.columns:
            top = official_df.sort_values("anomaly_probability", ascending=False).iloc[0]
            st.warning(
                f"Top risk sample: **{top['sample_id']}** | "
                f"Probability: **{float(top['anomaly_probability']):.4f}** | "
                f"Confidence: **{float(top['confidence']):.4f}**"
            )

        if not split_df.empty and "internal_split" in split_df.columns:
            split_counts = split_df["internal_split"].value_counts().reset_index()
            split_counts.columns = ["split", "count"]
            fig = px.bar(split_counts, x="split", y="count", color="split", title="Stratified Internal Split Size")
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dataset split summary unavailable.")

    with c2:
        st.markdown("### Readiness Checklist")
        checklist = [
            ("Dataset acquired", True),
            ("Model trained", METRICS_JSON.exists()),
            ("Uncertainty ready", UNCERTAINTY_JSON.exists()),
            ("Explainability ready", GLOBAL_IMPORTANCE_CSV.exists() and LOCAL_EXPLANATIONS_CSV.exists()),
            ("Official inference ready", OFFICIAL_PRED_CSV.exists()),
        ]
        for name, ok in checklist:
            status = "<span class='check-ok'>OK</span>" if ok else "<span class='check-miss'>Missing</span>"
        st.markdown(f"- {name}: {status}", unsafe_allow_html=True)

        st.code(
            "\n".join(
                [
                    str(METRICS_JSON),
                    str(UNCERTAINTY_JSON),
                    str(EXPLAIN_SUMMARY_JSON),
                    str(OFFICIAL_PRED_CSV),
                    str(GLOBAL_IMPORTANCE_CSV),
                    str(LOCAL_EXPLANATIONS_CSV),
                ]
            ),
            language="text",
        )

        st.markdown(
            f"""
<div class="section-card">
    <div class="metric-title">Demo Status</div>
    <div class="metric-value">{"READY FOR DEMO" if readiness >= 85 else "NEEDS REVIEW"}</div>
    <div class="metric-subtitle">No retraining. Visualization layer only.</div>
</div>
""",
            unsafe_allow_html=True,
        )

# ---------------------------
# Performance
# ---------------------------
with tabs[1]:
    st.markdown("### Model Performance")

    p1, p2, p3 = st.columns(3)
    with p1:
        metric_card("Validation Accuracy", fmt(metrics.get("validation", {}).get("accuracy", 0.0)))
    with p2:
        metric_card("Validation F1", fmt(metrics.get("validation", {}).get("f1", 0.0)))
    with p3:
        metric_card("Internal Test F1", fmt(metrics.get("internal_test", {}).get("f1", 0.0)))

    if not internal_df.empty and "label" in internal_df.columns and "anomaly_probability" in internal_df.columns:
        y_true = internal_df["label"].astype(int).values
        y_prob = internal_df["anomaly_probability"].astype(float).values
        y_pred = internal_df["prediction"].astype(int).values if "prediction" in internal_df.columns else (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        left, right = st.columns([0.95, 1.05])

        with left:
            st.plotly_chart(confusion_heatmap(cm, "Internal Test Confusion Matrix"), use_container_width=True)

        with right:
            roc_fig, pr_fig = roc_pr_figures(internal_df)
            if roc_fig is not None and pr_fig is not None:
                st.plotly_chart(roc_fig, use_container_width=True)
                st.plotly_chart(pr_fig, use_container_width=True)
            else:
                st.warning("ROC/PR curves unavailable because labels are missing or single-class.")

        st.markdown("### Threshold Preview on Internal Test")
        preview = threshold_preview(internal_df, threshold)
        a, b, c, d = st.columns(4)
        with a:
            metric_card("Predicted Abnormal", str(preview.get("predicted_abnormal", 0)))
        with b:
            metric_card("Predicted Normal", str(preview.get("predicted_normal", 0)))
        with c:
            metric_card("Current F1", fmt(preview.get("f1", 0.0)))
        with d:
            metric_card("Current Accuracy", fmt(preview.get("accuracy", 0.0)))
    else:
        st.warning("Internal prediction file missing or incomplete.")

# ---------------------------
# Uncertainty
# ---------------------------
with tabs[2]:
    st.markdown("### Uncertainty & Calibration")

    u1, u2, u3, u4 = st.columns(4)
    with u1:
        metric_card("Brier Score", fmt(uncertainty.get("brier_score", 0.0)))
    with u2:
        metric_card("Log Loss", fmt(uncertainty.get("log_loss", 0.0)))
    with u3:
        metric_card("ECE", fmt(uncertainty.get("ece", 0.0)))
    with u4:
        metric_card("Avg Confidence", fmt(uncertainty.get("avg_confidence", 0.0)))

    u5, u6 = st.columns(2)
    with u5:
        metric_card("Coverage", pct(uncertainty.get("coverage_at_confidence_cutoff", 0.0)))
    with u6:
        metric_card("Selective F1", fmt(uncertainty.get("selective_f1_at_confidence_cutoff", 0.0)))

    if not internal_df.empty and "confidence" in internal_df.columns:
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = confidence_hist(internal_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            y_true = internal_df["label"].astype(int).values if "label" in internal_df.columns else np.zeros(len(internal_df))
            y_prob = np.clip(internal_df["anomaly_probability"].astype(float).values, 1e-6, 1 - 1e-6)
            st.plotly_chart(reliability_fig(y_true, y_prob), use_container_width=True)

        st.markdown("### Lowest-confidence samples")
        st.dataframe(internal_df.sort_values("confidence", ascending=True).head(15), use_container_width=True, hide_index=True)

# ---------------------------
# Explainability
# ---------------------------
with tabs[3]:
    st.markdown("### Explainability")

    e1, e2 = st.columns([1.1, 0.9])

    with e1:
        if not global_imp.empty:
            fig = feature_importance_fig(global_imp)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Global feature importance file is missing.")

    with e2:
        st.markdown("### Local Explanation Explorer")
        if sample_id is not None and not local_exp.empty:
            fig, sample_top = local_explanation_fig(local_exp, sample_id)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("#### Top contributing features")
                st.dataframe(sample_top, use_container_width=True, hide_index=True)
            else:
                st.warning("No local explanation found for the selected sample.")
        else:
            st.info("No sample selected or local explanations are missing.")

    st.markdown("### Explanation Summary")
    if explain_summary:
        st.json(explain_summary)
    else:
        st.info("Explainability summary not found.")

    if not local_exp.empty:
        st.markdown("### Top anomalous local explanations")
        st.dataframe(local_exp.sort_values("weighted_score", ascending=False).head(15), use_container_width=True, hide_index=True)

# ---------------------------
# Predictions
# ---------------------------
with tabs[4]:
    st.markdown("### Prediction Explorer")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Official Demo Predictions")
        if not official_df.empty and "anomaly_probability" in official_df.columns:
            off = official_df.copy()
            off["prediction_at_threshold"] = (off["anomaly_probability"].astype(float) >= threshold).astype(int)
            off["confidence"] = np.maximum(off["anomaly_probability"].astype(float), 1 - off["anomaly_probability"].astype(float))
            st.dataframe(off.sort_values("anomaly_probability", ascending=False), use_container_width=True, hide_index=True)

            st.download_button(
                "Download official predictions CSV",
                data=off.to_csv(index=False).encode("utf-8"),
                file_name="official_test_predictions.csv",
                mime="text/csv",
            )
        else:
            st.warning("official_test_predictions.csv is missing or incomplete.")

    with right:
        st.markdown("#### Internal High-Risk Samples")
        if not internal_df.empty and "anomaly_probability" in internal_df.columns:
            ranked = internal_df.copy()
            ranked["prediction_at_threshold"] = (ranked["anomaly_probability"].astype(float) >= threshold).astype(int)
            ranked = ranked.sort_values("anomaly_probability", ascending=False).head(top_n)
            st.dataframe(ranked, use_container_width=True, hide_index=True)

            st.download_button(
                "Download top anomalies",
                data=ranked.to_csv(index=False).encode("utf-8"),
                file_name="top_internal_anomalies.csv",
                mime="text/csv",
            )
        else:
            st.warning("Internal predictions file is missing or incomplete.")

    st.markdown("### Threshold Effect")
    if not internal_df.empty and "label" in internal_df.columns and "anomaly_probability" in internal_df.columns:
        y_true = internal_df["label"].astype(int).values
        probs = internal_df["anomaly_probability"].astype(float).values
        sweep = []
        for thr in np.linspace(0.05, 0.95, 91):
            pred = (probs >= thr).astype(int)
            sweep.append(
                {
                    "threshold": thr,
                    "precision": precision_score(y_true, pred, zero_division=0),
                    "recall": recall_score(y_true, pred, zero_division=0),
                    "f1": f1_score(y_true, pred, zero_division=0),
                }
            )
        sweep_df = pd.DataFrame(sweep)
        fig = px.line(sweep_df, x="threshold", y=["precision", "recall", "f1"], title="Threshold Sweep on Internal Test")
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built from saved phase-5 artifacts only. No retraining. This dashboard is ready for screenshots and demo recording.")
