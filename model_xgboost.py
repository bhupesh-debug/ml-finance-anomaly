"""
model_xgboost.py
Supervised anomaly classifier using XGBoost + SHAP explainability.
Use when labeled transaction data is available (post-go-live audit history).
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from feature_engineering import build_feature_matrix


def train(data_path: str, output_path: str = "models/xgboost_model.pkl"):
    df = pd.read_csv(data_path)
    assert "is_anomalous" in df.columns, "Labeled column 'is_anomalous' required for supervised training."

    X = build_feature_matrix(df)
    y = df["is_anomalous"]

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )

    # Stratified K-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_aucs, f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        pr_aucs.append(average_precision_score(y_val, probs))
        f1s.append(f1_score(y_val, preds))
        print(f"  Fold {fold+1}: PR-AUC={pr_aucs[-1]:.3f}  F1={f1s[-1]:.3f}")

    print(f"\nMean PR-AUC: {np.mean(pr_aucs):.3f} ± {np.std(pr_aucs):.3f}")
    print(f"Mean F1:     {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # Final fit on all data
    model.fit(X, y)

    # SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Top 10 features by mean |SHAP|
    shap_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False)
    print("\nTop 10 SHAP features:")
    print(shap_importance.head(10))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "explainer": explainer,
        "feature_cols": list(X.columns),
        "shap_importance": shap_importance
    }, output_path)
    print(f"\nModel + SHAP saved to {output_path}")
    return model, explainer


def predict_with_explanation(transaction: dict, model_path: str = "models/xgboost_model.pkl") -> dict:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    explainer = artifact["explainer"]
    feature_cols = artifact["feature_cols"]

    df = pd.DataFrame([transaction])
    X = build_feature_matrix(df).reindex(columns=feature_cols, fill_value=0)

    prob = float(model.predict_proba(X)[0, 1])
    is_anomaly = prob >= 0.5
    shap_vals = explainer.shap_values(X)[0]

    top_features = pd.Series(shap_vals, index=feature_cols).abs().sort_values(ascending=False).head(5)
    explanation = [
        {"feature": feat, "shap_value": round(float(shap_vals[feature_cols.index(feat)]), 4)}
        for feat in top_features.index
    ]

    return {
        "anomaly_probability": round(prob, 4),
        "is_anomaly": is_anomaly,
        "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.5 else "NORMAL",
        "top_drivers": explanation
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/xgboost_model.pkl")
    args = parser.parse_args()
    train(args.data, args.output)
