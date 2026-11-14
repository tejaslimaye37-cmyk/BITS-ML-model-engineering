# Mini Production ML System: Customer Churn

An end-to-end churn scoring system built for the BITS Pilani WILP ML Model Engineering assignment (Option A).
It covers leak-free preprocessing, baselines, a class-weighted LightGBM model, serialized artifacts, an online
FastAPI service, a batch CLI, latency and throughput profiling, and a PSI/KS drift monitor. No Docker is needed.

**Author:** Tejas Limaye (2025em1200003)

## Problem

Predict whether a telecom subscriber will cancel before the next billing cycle, using the IBM Telco Customer
Churn dataset (7,043 customers, 26.5% churners). The model supports a retention campaign: a missed churner costs
5 cost units, an unnecessary retention offer costs 1. The primary serving mode is **batch** (nightly scoring of the
subscriber base), with an **online** endpoint for agent-assist and checkout flows.

## Quickstart

Tested on Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate    
pip install -r requirements.txt

python -m src.train          
python -m src.evaluate       
python -m src.drift_demo   
pytest -q                   
```

The trained artifacts are already committed in `artifacts/`, so the serving commands below work without retraining.

### Online path (FastAPI)

```bash
uvicorn serving.app:app --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
  "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}'
```

Interactive docs are served at `http://127.0.0.1:8000/docs`. `tenure`, `MonthlyCharges` and `Contract` are required;
all other fields may be omitted and are imputed. Malformed payloads return HTTP 422, and missing artifacts return 503.

### Batch path (CLI)

```bash
python -m serving.predict_batch --input data/sample_input.csv --output scored.csv --log-file batch.log
```

Writes `record_id, churn_probability, churn_flag, threshold, scoring_source, warnings`. Invalid rows are not scored;
they are written with a reason to `scored.rejects.csv`. Exit codes: 0 success, 1 startup failure, 2 missing columns.

### Profiling and monitoring

```bash
python -m serving.benchmark            # 1000 HTTP requests (p50/p95/p99) + batch throughput -> reports/benchmark_results.json
python -m src.monitor --input data/sample_input.csv    # PSI / KS drift report against the training reference
```

## Repository layout

```
README.md, requirements.txt
data/          telco_churn.csv, sample_input.csv
artifacts/     preprocessor.joblib, model.joblib (calibrated LightGBM), lgbm_raw.joblib, metadata.json, reference_profile.json
src/           config.py, data_pipeline.py, heuristic.py, train.py, evaluate.py, monitor.py, drift_demo.py
serving/       app.py, schemas.py, inference.py, predict_batch.py, benchmark.py, edge_cases.py
tests/         pipeline leakage, schemas, inference, API and batch CLI tests
reports/       metrics.json, benchmark_results.json, edge_case_results.json, drift_demo.json, figures/
```

## Design notes

- **Leakage control:** stratified 60/20/20 split (no timestamp exists in this dataset, so a temporal split is not
  possible). The imputers, scaler and one-hot encoder are fitted on the training partition only. Threshold and
  calibration use the validation partition. The test partition is used once, for reporting.
- **Imbalance:** class-weighted loss (`scale_pos_weight`), Platt calibration on validation data, and a decision
  threshold chosen to minimise expected cost. No oversampling.
- **Decoupling:** `serving/inference.py` loads artifacts once and is shared by the API and the batch CLI.
- **Fallback:** if model scoring raises, the scorer returns a rule-based risk score mapped to the empirical churn
  rate of the training data, and marks the response `scoring_source = fallback_heuristic`.
