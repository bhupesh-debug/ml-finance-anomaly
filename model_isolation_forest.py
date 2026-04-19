"""
model_isolation_forest.py
Unsupervised anomaly detection using Isolation Forest.
Recommended for new IBM CIC client onboarding — no labeled data required.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from feature_engineering import build_feature_matrix


def train(data_path: str, contamination: float = 0.012, output_path: str = "models/isolation_forest.pkl"):
    df = pd.read_csv(data_path)
    X = build_feature_matrix(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Isolation Forest returns -1 (anomaly) or 1 (normal)
    raw_scores = model.decision_function(X_scaled)
    # Normalize to 0–1 anomaly probability (higher = more anomalous)
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    df["anomaly_score"] = anomaly_scores
    df["predicted_anomaly"] = (model.predict(X_scaled) == -1).astype(int)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": list(X.columns)}, output_path)
    print(f"Model saved to {output_path}")

    if "is_anomalous" in df.columns:
        from evaluate import classification_report_summary
        classification_report_summary(df["is_anomalous"], df["predicted_anomaly"], df["anomaly_score"], "Isolation Forest")

    return model, scaler


def predict(transaction: dict, model_path: str = "models/isolation_forest.pkl") -> dict:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]

    df = pd.DataFrame([transaction])
    X = build_feature_matrix(df)
    X_scaled = scaler.transform(X.reindex(columns=artifact["feature_cols"], fill_value=0))

    raw_score = model.decision_function(X_scaled)[0]
    is_anomaly = model.predict(X_scaled)[0] == -1

    return {
        "is_anomaly": bool(is_anomaly),
        "raw_score": float(raw_score),
        "risk_level": "HIGH" if is_anomaly else "NORMAL"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--contamination", type=float, default=0.012)
    parser.add_argument("--output", default="models/isolation_forest.pkl")
    args = parser.parse_args()
    train(args.data, args.contamination, args.output)
