# Customer Churn Prediction: XGBoost + MLflow + FastAPI

Predicts whether a telecom customer will churn, tracks training runs in MLflow, and serves the model through a FastAPI endpoint.

## Problem
Given customer account features, predict churn (yes/no).

## Data
IBM Telco Customer Churn dataset (about 7,000 customers). `train.py` downloads and cleans it automatically. 80/20 train/test split, `random_state=42`. Churn rate in the test set is 26.6%.

## Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Majority class (always "no churn") | 0.734 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.788 | 0.621 | 0.516 | 0.564 |
| XGBoost (100 estimators, depth 5, lr 0.1) | 0.797 | 0.651 | 0.508 | 0.571 |

What this means:
- Because 73% of customers don't churn, the majority-class baseline already scores 73.4% accuracy. XGBoost improves on it by about 6 points.
- XGBoost is only 0.9 accuracy points better than logistic regression on a single split, which is not a meaningful difference.
- Recall is 0.51: the model misses about half of customers who actually churn. When it flags a customer, it is right about 65% of the time.
- Parameters are fixed, not tuned. Class-imbalance handling is not yet applied.

Reproduce the table: `python baseline.py`

![MLflow runs](docs/mlflow-runs.png)

Each run logs its parameters and metrics to MLflow; recent runs also log precision, recall, F1 and the model files as artifacts.

## Inference latency
Measured with `latency_test.py`: 100 sequential requests to a local uvicorn server on a MacBook Pro (Python 3.11).

| p50 | p95 |
|---|---|
| 7.8 ms | 10.7 ms |

This is single-client latency on localhost. It is not a concurrency or production load test.

## Run it

    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python train.py                       # trains, logs to MLflow, writes churn_model.json
    uvicorn api:app --port 8000           # serve the API
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Interactive API docs are at http://127.0.0.1:8000/docs. The `/predict` endpoint accepts `{"data": {<customer fields>}}`; the API one-hot encodes the fields and aligns them to the 30 training features.

## Limitations
- One public dataset, not real customer data.
- Input is a free-form dictionary with no field validation.
- Not deployed to a cloud service; no monitoring or retraining.
- Results come from one train/test split, not cross-validation.

## Files
`train.py` training and MLflow logging, `baseline.py` baseline comparison, `api.py` FastAPI service, `app.py` Streamlit dashboard, `latency_test.py` latency benchmark, `churn_model.json` and `expected_features.json` model and feature list.
