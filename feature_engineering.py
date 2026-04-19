"""
feature_engineering.py
All feature transforms for the ERP Financial Integrity Monitor.
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["submit_datetime"] = pd.to_datetime(df["submit_datetime"])
    df["approval_datetime"] = pd.to_datetime(df["approval_datetime"])

    df["day_of_week"] = df["submit_datetime"].dt.dayofweek
    df["hour_of_day"] = df["submit_datetime"].dt.hour

    # Cyclical encoding (prevents 23->0 discontinuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["weekend_submission_flag"] = (df["day_of_week"] >= 5).astype(int)
    df["after_hours_flag"] = ((df["hour_of_day"] < 7) | (df["hour_of_day"] > 20)).astype(int)
    df["approver_delta_hours"] = (df["approval_datetime"] - df["submit_datetime"]).dt.total_seconds() / 3600
    df["approver_delta_hours"] = df["approver_delta_hours"].clip(lower=0)

    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])

    # Round number detection (common in fictitious invoices)
    df["amount_round_number"] = (
        (df["amount"] % 1000 == 0) | (df["amount"] % 500 == 0) | (df["amount"] % 100 == 0)
    ).astype(int)

    # GL account z-score (context-aware outlier)
    gl_stats = df.groupby("gl_account_code")["amount"].agg(["mean", "std"]).reset_index()
    gl_stats.columns = ["gl_account_code", "gl_mean", "gl_std"]
    df = df.merge(gl_stats, on="gl_account_code", how="left")
    df["gl_account_amount_zscore"] = (df["amount"] - df["gl_mean"]) / (df["gl_std"] + 1e-6)
    df = df.drop(columns=["gl_mean", "gl_std"])

    return df


def add_vendor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("submit_datetime").reset_index(drop=True)

    # 7-day spend velocity per vendor
    df["submit_date"] = df["submit_datetime"].dt.date
    vendor_daily = df.groupby(["vendor_id", "submit_date"])["amount"].sum().reset_index()
    vendor_daily.columns = ["vendor_id", "submit_date", "daily_spend"]

    # Rolling 7-day vs 90-day baseline
    df["spend_velocity_7d"] = df.groupby("vendor_id")["amount"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )
    df["prior_30d_vendor_spend"] = df.groupby("vendor_id")["amount"].transform(
        lambda x: x.rolling(30, min_periods=1).sum()
    )
    df["vendor_frequency"] = df.groupby("vendor_id")["transaction_id"].transform("count")

    return df


def add_sod_features(df: pd.DataFrame) -> pd.DataFrame:
    """Segregation of Duties features."""
    df = df.copy()

    # Self-approval ratio per submitter
    df["is_self_approved"] = (df["submitter_id"] == df["approver_id"]).astype(int)
    sod_ratio = df.groupby("submitter_id")["is_self_approved"].mean().reset_index()
    sod_ratio.columns = ["submitter_id", "approver_self_ratio"]
    df = df.merge(sod_ratio, on="submitter_id", how="left")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Frequency encode vendor and approver
    for col in ["vendor_id", "approver_id", "submitter_id"]:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)

    # Target-mean encode GL account (use overall mean for safety)
    if "is_anomalous" in df.columns:
        gl_target = df.groupby("gl_account_code")["is_anomalous"].mean()
        df["gl_account_target_enc"] = df["gl_account_code"].map(gl_target)
    else:
        df["gl_account_target_enc"] = df["gl_account_code"].astype("category").cat.codes / df["gl_account_code"].nunique()

    dept_freq = df["department"].value_counts(normalize=True)
    df["department_freq"] = df["department"].map(dept_freq)

    return df


FEATURE_COLS = [
    "log_amount", "amount_round_number", "gl_account_amount_zscore",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "weekend_submission_flag", "after_hours_flag", "approver_delta_hours",
    "spend_velocity_7d", "prior_30d_vendor_spend", "vendor_frequency",
    "is_self_approved", "approver_self_ratio", "is_duplicate_flag",
    "vendor_id_freq", "approver_id_freq", "submitter_id_freq",
    "gl_account_target_enc", "department_freq"
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_vendor_features(df)
    df = add_sod_features(df)
    df = encode_categoricals(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].fillna(0)
