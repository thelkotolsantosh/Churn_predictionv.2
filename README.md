# Churn_predictionv.2
# 📉 Customer Churn Prediction
> **Identify at-risk customers before they leave** — a production-grade machine learning system for telecom churn prediction, complete with a REST API and an interactive Streamlit dashboard.

---

## 🧩 Problem Statement

Customer churn — when a subscriber cancels their service — is one of the most expensive problems in subscription businesses. In telecom, **acquiring a new customer costs 5–25× more than retaining an existing one**. Even a 1 % reduction in churn rate can translate to millions in annual revenue.

This project builds an end-to-end ML pipeline that:
- Predicts the probability a customer will churn in the next billing cycle
- Surfaces the top drivers of churn (feature importance)
- Exposes predictions via a low-latency REST API for CRM integration
- Provides an interactive dashboard for Customer Success teams

---

## 🏗️ Architecture

```
Raw CSV data
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Data Pipeline (src/)               │
│                                                 │
│  generate_dataset.py  →  churn_data.csv         │
│         │                                       │
│  data_preprocessing.py                          │
│    ├── load_raw_data()                          │
│    ├── clean_data()          (missing values,   │
│    │                          type coercion)    │
│    └── encode_categoricals() (one-hot, binary)  │
│         │                                       │
│  feature_engineering.py                         │
│    ├── charges_per_month                        │
│    ├── long_tenure_flag                         │
│    ├── support_rate                             │
│    └── contract_risk                            │
│         │                                       │
│  split_and_scale()  →  StandardScaler           │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│              Model Training (src/)              │
│                                                 │
│  SMOTE oversampling   (class imbalance fix)     │
│  GridSearchCV         (hyperparameter tuning)   │
│                                                 │
│  ┌───────────────────┐  ┌──────────────────┐   │
│  │ Logistic          │  │  Random Forest   │   │
│  │ Regression        │  │  (200 trees)     │   │
│  └───────────────────┘  └──────────────────┘   │
│           └──────────┬───────────┘             │
│                       ▼                        │
│              Evaluation & Comparison            │
│         (AUC, F1, Precision, Recall)           │
│                       │                        │
│              Best model → models/best_model.pkl│
└─────────────┬───────────────────────────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌──────────┐    ┌────────────────┐
│  FastAPI │    │   Streamlit    │
│  /predict│    │   Dashboard    │
│  :8000   │    │   :8501        │
└──────────┘    └────────────────┘
```

---

## 📁 Project Structure

```
churn_prediction/
│
├── data/
│   ├── generate_dataset.py     # Synthetic telecom dataset generator
│   └── churn_data.csv          # Generated dataset (after running train.py)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Load, clean, encode, split, scale
│   ├── feature_engineering.py  # Domain-driven feature creation
│   ├── model_training.py       # Train LR + RF with SMOTE & GridSearchCV
│   └── evaluation.py           # Metrics, confusion matrix, ROC, feat. importance
│
├── app/
│   ├── app.py                  # FastAPI REST API  (uvicorn)
│   └── streamlit_ui.py         # Streamlit interactive dashboard
│
├── models/                     # Saved models (auto-created by train.py)
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── plots/                  # Evaluation plots
│       ├── confusion_matrix_*.png
│       ├── roc_curves.png
│       └── feature_importance_*.png
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA notebook
│
├── tests/
│   └── test_pipeline.py        # pytest test suite
│
├── train.py                    # One-command training orchestrator
├── sample_input.json           # Example API request body
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- pip

### 1. Clone & install

```bash
git clone https://github.com/thelkotolsantosh/churn-prediction.git
cd churn-prediction
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the models

```bash
python train.py
```

This will:
- Generate the synthetic dataset (`data/churn_data.csv`)
- Run feature engineering + preprocessing
- Train Logistic Regression and Random Forest with hyperparameter tuning
- Apply SMOTE to handle class imbalance
- Save the best model and all evaluation plots to `models/`

Expected output:
```
==============================================================
  Customer Churn Prediction – Training Pipeline
==============================================================
...
  Model Comparison (test set)
══════════════════════════════════════════════════════════════
                      accuracy  precision  recall     f1  roc_auc
model
random_forest           0.8124     0.6531  0.7843 0.7128   0.8761
logistic_regression     0.7893     0.6187  0.7612 0.6824   0.8512
==============================================================
  Best model : random_forest
  ROC-AUC    : 0.8761
==============================================================
```

### 3. Start the API

```bash
uvicorn app.app:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_ui.py
```

Dashboard: http://localhost:8501

---

## 🔌 API Reference

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "random_forest"
}
```

### `POST /predict`

**Request body:**
```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 5,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber Optic",
  "online_security": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "paperless_billing": "Yes",
  "contract_type": "Month-to-Month",
  "payment_method": "Electronic Check",
  "monthly_charges": 89.10,
  "total_charges": 445.50,
  "support_calls": 3
}
```

**Response:**
```json
{
  "churn_probability": 0.7834,
  "churn_label": "Churn",
  "risk_tier": "High",
  "model_used": "random_forest"
}
```

**cURL example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

**Python example:**
```python
import requests, json
payload = json.load(open("sample_input.json"))
payload.pop("_comment", None)
payload.pop("_business_context", None)
resp = requests.post("http://localhost:8000/predict", json=payload)
print(resp.json())
# {'churn_probability': 0.7834, 'churn_label': 'Churn', 'risk_tier': 'High', ...}
```

### `POST /predict/batch`

Send up to 500 customers in one request (same schema as `/predict` but wrapped in a list).

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Random Forest       | 0.8124   | 0.6531    | 0.7843 | 0.7128 | **0.8761** |
| Logistic Regression | 0.7893   | 0.6187    | 0.7612 | 0.6824 | 0.8512  |

> **Why ROC-AUC?** Churn data is imbalanced (~26 % positive). ROC-AUC measures the model's ability to *rank* churners above non-churners, regardless of the decision threshold — making it the most reliable single metric for retention use cases.

---

## 🔑 Key Churn Drivers

Based on Random Forest feature importances:

| Rank | Feature               | Business Insight |
|------|-----------------------|-----------------|
| 1    | `tenure`              | Newer customers churn far more |
| 2    | `monthly_charges`     | Higher bills → higher price sensitivity |
| 3    | `contract_type`       | Month-to-Month has no lock-in |
| 4    | `internet_service`    | Fiber Optic users report more issues |
| 5    | `payment_method`      | Electronic Check correlates with lower engagement |
| 6    | `support_calls`       | Frustrated customers call more before leaving |
| 7    | `total_charges`       | Low total = short relationship |
| 8    | `online_security`     | Unprotected users feel less value |

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔧 Class Imbalance Handling

The raw dataset has ~26 % churn (positive class). Two strategies are used:

1. **SMOTE** (`imbalanced-learn`) — synthetic oversampling of the minority class before training.
2. **`class_weight="balanced"`** — built-in sklearn option for Logistic Regression and Random Forest as a fallback.

---

## 📓 Exploratory Analysis

Open the notebook for data exploration, distribution plots, and correlation analysis:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 🚀 Deployment Notes

- **Docker**: Wrap `uvicorn app.app:app` in a `Dockerfile` with the `models/` directory copied in.
- **CI/CD**: Add `pytest tests/` as a GitHub Actions step before any deployment.
- **Model refresh**: Re-run `train.py` monthly as new customer data accumulates. Pin the model version in `model_info.json` for rollback capability.
