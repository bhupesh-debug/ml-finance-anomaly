"""
evaluate.py
Model evaluation — PR curves, business KPIs, and model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, precision_recall_curve, roc_auc_score
)
from pathlib import Path


def classification_report_summary(y_true, y_pred, y_prob, model_name: str):
    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    fpr = 1 - prec  # approximation

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  PR-AUC:    {pr_auc:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  Est. FPR:  {fpr:.3f}")
    print(f"{'='*50}")

    return {"model": model_name, "pr_auc": pr_auc, "f1": f1, "precision": prec, "recall": rec}


def plot_pr_curves(results: list, output_path: str = "reports/pr_curves.png"):
    """
    results: list of dicts with keys: name, y_true, y_prob
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#185FA5", "#0F6E56", "#854F0B", "#A32D2D"]

    for i, r in enumerate(results):
        prec, rec, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        ap = average_precision_score(r["y_true"], r["y_prob"])
        ax.plot(rec, prec, color=colors[i % len(colors)],
                label=f"{r['name']} (AP={ap:.2f})", linewidth=2)

    baseline = r["y_true"].mean()
    ax.axhline(y=baseline, color="#888780", linestyle="--", linewidth=1, label=f"Baseline (AP={baseline:.2f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — ERP Anomaly Detection", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"PR curves saved to {output_path}")
    return fig


def business_kpi_report(df: pd.DataFrame, predicted_col: str = "predicted_anomaly",
                         true_col: str = "is_anomalous", amount_col: str = "amount"):
    """
    Frame results in business terms — $ caught, auditor workload, etc.
    """
    flagged = df[df[predicted_col] == 1]
    true_positives = df[(df[predicted_col] == 1) & (df[true_col] == 1)]
    false_positives = df[(df[predicted_col] == 1) & (df[true_col] == 0)]
    false_negatives = df[(df[predicted_col] == 0) & (df[true_col] == 1)]

    total_txns = len(df)
    flag_rate = len(flagged) / total_txns
    value_caught = true_positives[amount_col].sum()
    value_missed = false_negatives[amount_col].sum()
    auditor_review_reduction = 1 - flag_rate

    print("\n=== BUSINESS KPI REPORT ===")
    print(f"Total transactions:        {total_txns:,}")
    print(f"Flagged for review:        {len(flagged):,} ({flag_rate:.1%} of total)")
    print(f"True anomalies caught:     {len(true_positives):,}")
    print(f"False alarms:              {len(false_positives):,}")
    print(f"Anomalies missed:          {len(false_negatives):,}")
    print(f"$ value caught:            ${value_caught:,.0f}")
    print(f"$ value missed:            ${value_missed:,.0f}")
    print(f"Auditor workload vs manual: {auditor_review_reduction:.0%} reduction")
    print(f"Catch rate (recall):       {len(true_positives)/(len(true_positives)+len(false_negatives)):.1%}")
