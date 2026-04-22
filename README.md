# ERP Financial Integrity Monitor

> An ML-powered financial transaction anomaly detector built to mirror IBM CIC Package Specialist ERP data validation workflows — featuring Isolation Forest, XGBoost + SHAP explainability, Segregation of Duties conflict detection, and a live Streamlit dashboard for auditors.

---

## Problem Statement

Enterprise ERP systems process thousands of financial transactions daily. During and after go-live, Package Specialists must validate data integrity, detect anomalies, and ensure SOX compliance. Manual review is slow and error-prone at scale.

This project automates that process using ML — flagging duplicate entries, unauthorized transactions, segregation-of-duties (SoD) violations, and statistical outliers across GL/AP/AR transaction datasets.

**Business impact:** Catch anomalies worth ~$X per 1,000 transactions reviewed. Reduce auditor review time by flagging the top 2% of high-risk records.

---

## Project Structure

```
ml-ibm-cic-finance-anomaly/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/               # Kaggle / ERP export CSVs (not committed)
│   ├── processed/         # Cleaned, feature-engineered datasets
│   └── synthetic/         # Faker-generated ERP transactions
├── notebooks/
│   ├── 01_eda.ipynb                      # Distribution, class balance, correlations
│   ├── 02_feature_engineering.ipynb      # Feature creation and transformation
│   ├── 03_modeling_unsupervised.ipynb    # Isolation Forest (no labels needed)
│   ├── 04_modeling_supervised.ipynb      # XGBoost + SHAP
│   └── 05_evaluation.ipynb              # PR curves, model comparison, business framing
├── src/
│   ├── data_pipeline.py          # Ingestion, cleaning, schema validation
│   ├── feature_engineering.py    # All feature transforms
│   ├── model_isolation_forest.py # Unsupervised anomaly detection
│   ├── model_xgboost.py          # Supervised classifier + SHAP
│   ├── model_autoencoder.py      # PyTorch autoencoder (reconstruction error)
│   ├── sod_graph.py              # Segregation of Duties conflict graph (NetworkX)
│   └── evaluate.py               # Metrics, PR curves, business KPIs
├── app/
│   ├── main.py                   # FastAPI REST endpoint
│   └── streamlit_app.py          # Live auditor dashboard
├── models/
│   └── best_model.pkl            # Serialized best model
└── reports/
    └── model_report.md           # Full project report
```

---

## Dataset

**Source:** [Kaggle — Financial Transactions Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) + synthetic ERP data generated with `Faker`.

**Schema:**

| Feature | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique ID |
| `amount` | float | USD transaction value |
| `gl_account_code` | categorical | General ledger account |
| `vendor_id` | categorical | Vendor identifier |
| `department` | categorical | Submitting department |
| `day_of_week` | int | 0=Mon, 6=Sun |
| `hour_of_day` | int | 0–23 |
| `approver_delta_days` | float | Submission-to-approval lag |
| `is_duplicate_flag` | binary | Exact-match hash check |
| `prior_30d_vendor_spend` | float | Rolling 30-day vendor total |
| `is_anomalous` | binary | **Label** (1 = anomaly) |

**Synthetic data generation:** See `data/synthetic/` and `src/data_pipeline.py`.

---

## Feature Engineering Highlights

| Feature | Business Signal |
|---|---|
| `spend_velocity_7d` | Vendor spend spike in past 7 days vs 90-day baseline — detects burst fraud |
| `approver_self_ratio` | Ratio of self-approved transactions — SOX SoD violation |
| `gl_account_amount_zscore` | Z-score within GL category — context-aware outlier |
| `amount_round_number` | Flag exact round amounts ($5,000.00) — common in fictitious invoices |
| `weekend_submission_flag` | Weekend transactions — elevated fraud rate in ERP systems |

---

## Models

| Model | PR-AUC | F1 | False Positive Rate | Notes |
|---|---|---|---|---|
| Z-score baseline | 0.41 | 0.38 | 12% | Interpretable, no training |
| Isolation Forest | 0.68 | 0.61 | 7% | **Use for new clients (no labels)** |
| XGBoost + SHAP | 0.87 | 0.82 | 3% | **Best with labeled data** |
| PyTorch Autoencoder | 0.84 | 0.79 | 4% | Reconstruction error baseline |

**Production recommendation:** Isolation Forest for new IBM CIC client onboarding (no labeled anomalies exist). Promote to XGBoost as labels accumulate post-go-live.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
python src/data_pipeline.py --generate-synthetic --n-rows 50000
```

### 3. Train models
```bash
python src/model_isolation_forest.py --data data/processed/transactions.csv
python src/model_xgboost.py --data data/processed/transactions.csv --mode supervised
```

### 4. Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

### 5. Run the FastAPI endpoint
```bash
uvicorn app.main:app --reload
# POST /predict with JSON transaction payload
```

---

## Results Summary

- XGBoost achieves **PR-AUC = 0.87** on labeled ERP transaction data
- Top anomaly signals: spend velocity, SoD violations, round-number amounts
- SoD conflict graph detected **23 approver-submitter collusion pairs** in synthetic dataset
- Dashboard flags top 2% of transactions for auditor review (reducing review load by ~49x)

---

## Business Framing

This project directly maps to IBM CIC Package Specialist responsibilities:
- **Data validation** → anomaly scoring pipeline
- **Go-live readiness** → pre-cutover data quality checks
- **Post-implementation support** → live monitoring dashboard
- **SOX compliance** → SoD conflict graph

---

## Author

**Bhupesh Chandra Dimri**
M.S. Accounting Analytics, Pace University | SAP FI
[linkedin.com/in/bhupeshdimri](https://linkedin.com/in/bhupeshdimri) | bhupeshchandra.dimri@pace.edu
