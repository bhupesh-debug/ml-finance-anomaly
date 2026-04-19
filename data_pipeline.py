"""
data_pipeline.py
Ingestion, cleaning, schema validation, and synthetic ERP data generation.
"""

import argparse
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

GL_ACCOUNTS = [
    "6100-TRAVEL", "6200-VENDOR-SVCS", "6300-UTILITIES", "6400-PAYROLL",
    "6500-CAPEX", "7100-AP-TRADE", "7200-AR-RECV", "8100-GL-ADJ"
]
DEPARTMENTS = [
    "Finance", "Operations", "HR", "IT", "Legal", "Marketing",
    "Procurement", "Compliance", "Executive", "Audit"
]
VENDORS = [f"VENDOR_{i:04d}" for i in range(1, 201)]
APPROVERS = [f"EMP_{i:04d}" for i in range(1, 51)]


def generate_synthetic_transactions(n_rows: int = 50000, anomaly_rate: float = 0.012) -> pd.DataFrame:
    """
    Generate realistic synthetic ERP financial transaction records.
    Anomaly rate ~1.2% to reflect real-world ERP fraud incidence.
    """
    records = []
    n_anomalies = int(n_rows * anomaly_rate)
    n_normal = n_rows - n_anomalies

    # Normal transactions
    for _ in range(n_normal):
        submitter = random.choice(APPROVERS)
        approver = random.choice([a for a in APPROVERS if a != submitter])
        submit_dt = fake.date_time_between(start_date="-2y", end_date="now")
        approval_dt = submit_dt + timedelta(hours=random.uniform(2, 72))
        records.append({
            "transaction_id": fake.uuid4(),
            "amount": round(random.lognormormal(7.5, 1.2), 2),
            "gl_account_code": random.choice(GL_ACCOUNTS),
            "vendor_id": random.choice(VENDORS),
            "department": random.choice(DEPARTMENTS),
            "submitter_id": submitter,
            "approver_id": approver,
            "submit_datetime": submit_dt,
            "approval_datetime": approval_dt,
            "is_anomalous": 0,
            "anomaly_type": None
        })

    # Anomalous transactions
    anomaly_types = ["duplicate", "self_approved", "round_number", "after_hours", "vendor_burst"]
    for i in range(n_anomalies):
        atype = random.choice(anomaly_types)
        submitter = random.choice(APPROVERS)
        submit_dt = fake.date_time_between(start_date="-2y", end_date="now")

        if atype == "duplicate":
            amount = round(random.lognormormal(7.5, 1.2), 2)
            approver = random.choice([a for a in APPROVERS if a != submitter])
        elif atype == "self_approved":
            amount = round(random.lognormormal(8.0, 0.8), 2)
            approver = submitter  # SoD violation
        elif atype == "round_number":
            amount = float(random.choice([1000, 2500, 5000, 10000, 25000, 50000]))
            approver = random.choice([a for a in APPROVERS if a != submitter])
        elif atype == "after_hours":
            submit_dt = submit_dt.replace(hour=random.randint(22, 23))
            amount = round(random.lognormormal(9.0, 0.5), 2)
            approver = random.choice([a for a in APPROVERS if a != submitter])
        else:  # vendor_burst
            amount = round(random.uniform(50000, 250000), 2)
            approver = random.choice([a for a in APPROVERS if a != submitter])

        approval_dt = submit_dt + timedelta(hours=random.uniform(0.1, 1.0))

        records.append({
            "transaction_id": fake.uuid4(),
            "amount": amount,
            "gl_account_code": random.choice(GL_ACCOUNTS),
            "vendor_id": random.choice(VENDORS[:20]),  # concentrated vendor pool
            "department": random.choice(DEPARTMENTS),
            "submitter_id": submitter,
            "approver_id": approver,
            "submit_datetime": submit_dt,
            "approval_datetime": approval_dt,
            "is_anomalous": 1,
            "anomaly_type": atype
        })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated {len(df)} transactions | Anomalies: {df['is_anomalous'].sum()} ({df['is_anomalous'].mean():.2%})")
    return df


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Schema validation and cleaning."""
    required_cols = ["transaction_id", "amount", "gl_account_code", "vendor_id",
                     "department", "submitter_id", "approver_id",
                     "submit_datetime", "approval_datetime"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["amount", "transaction_id"])
    df = df[df["amount"] > 0]
    df["submit_datetime"] = pd.to_datetime(df["submit_datetime"])
    df["approval_datetime"] = pd.to_datetime(df["approval_datetime"])

    # Dedup hash
    df["dedup_hash"] = df.apply(
        lambda r: hashlib.md5(f"{r['amount']}{r['vendor_id']}{r['gl_account_code']}".encode()).hexdigest(),
        axis=1
    )
    df["is_duplicate_flag"] = df["dedup_hash"].duplicated(keep=False).astype(int)

    print(f"Cleaned: {len(df)} rows | Duplicates flagged: {df['is_duplicate_flag'].sum()}")
    return df


def save(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


def lognormormal(mean, sigma):
    return np.random.lognormal(mean, sigma)


# Monkey-patch for convenience
random.lognormormal = lambda m, s: round(float(np.random.lognormal(m, s)), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--n-rows", type=int, default=50000)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/processed/transactions.csv")
    args = parser.parse_args()

    if args.generate_synthetic:
        df = generate_synthetic_transactions(args.n_rows)
        save(df, "data/synthetic/transactions_raw.csv")
        df = clean_and_validate(df)
        save(df, args.output)
    elif args.input:
        df = pd.read_csv(args.input)
        df = clean_and_validate(df)
        save(df, args.output)
    else:
        print("Use --generate-synthetic or --input <file.csv>")
