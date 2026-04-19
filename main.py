"""
main.py — FastAPI REST endpoint for ERP Financial Integrity Monitor
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from model_xgboost import predict_with_explanation
from model_isolation_forest import predict as iso_predict

app = FastAPI(
    title="ERP Financial Integrity Monitor",
    description="ML-powered transaction anomaly detection — IBM CIC Finance Portfolio Project",
    version="1.0.0"
)


class Transaction(BaseModel):
    transaction_id: str
    amount: float
    gl_account_code: str
    vendor_id: str
    department: str
    submitter_id: str
    approver_id: str
    submit_datetime: str
    approval_datetime: str
    is_duplicate_flag: Optional[int] = 0


class PredictionResponse(BaseModel):
    transaction_id: str
    anomaly_probability: float
    is_anomaly: bool
    risk_level: str
    top_drivers: list
    model_used: str


@app.get("/")
def root():
    return {"status": "ok", "service": "ERP Financial Integrity Monitor v1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    try:
        payload = txn.dict()
        result = predict_with_explanation(payload, model_path="models/xgboost_model.pkl")
        return PredictionResponse(
            transaction_id=txn.transaction_id,
            model_used="XGBoost + SHAP",
            **result
        )
    except FileNotFoundError:
        # Fallback to Isolation Forest if XGBoost not trained yet
        try:
            payload = txn.dict()
            result = iso_predict(payload, model_path="models/isolation_forest.pkl")
            return PredictionResponse(
                transaction_id=txn.transaction_id,
                anomaly_probability=1.0 if result["is_anomaly"] else 0.0,
                is_anomaly=result["is_anomaly"],
                risk_level=result["risk_level"],
                top_drivers=[],
                model_used="Isolation Forest (unsupervised)"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/batch-predict")
def batch_predict(transactions: list[Transaction]):
    results = []
    for txn in transactions:
        try:
            r = predict(txn)
            results.append(r)
        except Exception as e:
            results.append({"transaction_id": txn.transaction_id, "error": str(e)})
    return {"results": results, "total": len(results), "flagged": sum(1 for r in results if isinstance(r, dict) and r.get("is_anomaly"))}
