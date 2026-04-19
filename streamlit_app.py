"""
streamlit_app.py — Live Auditor Dashboard
ERP Financial Integrity Monitor — IBM CIC Finance Portfolio Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

st.set_page_config(
    page_title="ERP Financial Integrity Monitor",
    page_icon="🔍",
    layout="wide"
)

st.title("ERP Financial Integrity Monitor")
st.caption("IBM CIC Finance Portfolio Project · Bhupesh Chandra Dimri")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    risk_threshold = st.slider("Anomaly score threshold", 0.0, 1.0, 0.5, 0.05)
    model_mode = st.selectbox("Detection mode", ["Unsupervised (Isolation Forest)", "Supervised (XGBoost + SHAP)"])
    uploaded = st.file_uploader("Upload ERP export (CSV)", type=["csv"])
    st.markdown("---")
    st.markdown("**Quick demo:** Generate synthetic data below")
    n_rows = st.number_input("Synthetic rows", 1000, 100000, 10000, step=1000)
    gen_btn = st.button("Generate & Analyze")


@st.cache_data
def load_and_score(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Score transactions — using synthetic scores for demo."""
    from feature_engineering import build_feature_matrix
    X = build_feature_matrix(df)

    # Demo scoring (replace with model.predict_proba when model is trained)
    np.random.seed(42)
    if "is_anomalous" in df.columns:
        base = df["is_anomalous"].values.astype(float)
        noise = np.random.beta(1, 8, size=len(df))
        df["anomaly_score"] = np.clip(base * 0.7 + noise, 0, 1)
    else:
        df["anomaly_score"] = np.random.beta(1, 12, size=len(df))

    df["risk_level"] = pd.cut(
        df["anomaly_score"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
    df["flagged"] = df["anomaly_score"] >= threshold
    return df


# Load data
if uploaded:
    df_raw = pd.read_csv(uploaded)
elif gen_btn:
    from data_pipeline import generate_synthetic_transactions, clean_and_validate
    with st.spinner("Generating synthetic ERP transactions..."):
        df_raw = generate_synthetic_transactions(int(n_rows))
        df_raw = clean_and_validate(df_raw)
else:
    st.info("Upload an ERP CSV export or generate synthetic data to begin.")
    st.stop()

df = load_and_score(df_raw, risk_threshold)

# KPI row
col1, col2, col3, col4, col5 = st.columns(5)
flagged = df[df["flagged"]]
col1.metric("Total transactions", f"{len(df):,}")
col2.metric("Flagged for review", f"{len(flagged):,}", f"{len(flagged)/len(df):.1%}")
col3.metric("HIGH risk", str((df["risk_level"] == "HIGH").sum()))
col4.metric("MEDIUM risk", str((df["risk_level"] == "MEDIUM").sum()))
if "is_anomalous" in df.columns:
    caught = ((df["flagged"]) & (df["is_anomalous"] == 1)).sum()
    total_anomalies = df["is_anomalous"].sum()
    col5.metric("Recall", f"{caught/max(total_anomalies,1):.0%}", f"{caught}/{total_anomalies} caught")

st.markdown("---")

# Charts
c1, c2 = st.columns(2)

with c1:
    st.subheader("Anomaly score distribution")
    fig = px.histogram(df, x="anomaly_score", color="risk_level",
                       color_discrete_map={"LOW": "#0F6E56", "MEDIUM": "#BA7517", "HIGH": "#A32D2D"},
                       nbins=50, template="simple_white")
    fig.add_vline(x=risk_threshold, line_dash="dash", line_color="#185FA5",
                  annotation_text=f"Threshold ({risk_threshold})")
    fig.update_layout(height=300, margin=dict(t=20, b=20), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Flagged by department")
    dept_counts = flagged.groupby("department").size().reset_index(name="count").sort_values("count", ascending=True)
    fig2 = px.bar(dept_counts, x="count", y="department", orientation="h",
                  color_discrete_sequence=["#185FA5"], template="simple_white")
    fig2.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

# Flagged transactions table
st.subheader(f"Flagged transactions ({len(flagged):,} records)")
display_cols = ["transaction_id", "amount", "gl_account_code", "vendor_id",
                "department", "anomaly_score", "risk_level"]
available = [c for c in display_cols if c in flagged.columns]
styled = flagged[available].sort_values("anomaly_score", ascending=False).head(200)

st.dataframe(
    styled.style.background_gradient(subset=["anomaly_score"], cmap="Reds"),
    use_container_width=True,
    height=400
)

# SoD violations
if "submitter_id" in df.columns and "approver_id" in df.columns:
    st.subheader("Segregation of Duties violations")
    sod_violations = df[df["submitter_id"] == df["approver_id"]]
    if len(sod_violations) > 0:
        st.error(f"{len(sod_violations)} self-approval violations detected (SOX risk)")
        st.dataframe(sod_violations[["transaction_id", "amount", "submitter_id", "approver_id", "department"]].head(50),
                     use_container_width=True)
    else:
        st.success("No self-approval SoD violations detected.")

st.markdown("---")
st.caption("ERP Financial Integrity Monitor · Created by Bhupesh Chandra Dimri · IBM CIC Portfolio Project")
