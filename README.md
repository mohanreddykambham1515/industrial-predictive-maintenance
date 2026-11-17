
# Industrial Predictive Maintenance System

## Project Overview

An end-to-end Machine Learning application that predicts
industrial machine failures using sensor data.

The system uses a Random Forest classifier to predict
machine failure based on temperature, rotational speed,
torque, and tool wear.

## Business Objective

Reduce unexpected machine downtime by identifying
potential failures before they occur.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- FastAPI
- Pydantic
- Docker
- Pytest
- GitHub Actions

## Project Architecture

```text
industrial-predictive-maintenance/
│
├── data/
│   └── ai4i2020.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── api/
│   ├── __init__.py
│   ├── app.py
│   └── schemas.py
│
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── artifacts/
│   ├── predictive_maintenance_pipeline.joblib
│   └── metrics.json
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd industrial-predictive-maintenance
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download the AI4I 2020 Predictive Maintenance Dataset
from the UCI Machine Learning Repository.

Place the CSV file inside:

```text
data/ai4i2020.csv
```

## Train Model

```bash
python -m src.train
```

## Evaluate Model

```bash
python -m src.evaluate
```

## Run FastAPI

```bash
python -m uvicorn api.app:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

## Run Tests

```bash
python -m pytest tests/ -v
```

## Docker

Build image:

```bash
docker build -t predictive-maintenance .
```

Run container:

```bash
docker run -p 8000:8000 predictive-maintenance
```

## Prediction Features

- Product Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

## Model Evaluation

The project evaluates:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

Recall and PR-AUC are particularly important
because machine failure is a minority class.

## Future Improvements

- XGBoost and LightGBM
- Hyperparameter optimization
- SHAP explainability
- MLflow experiment tracking
- Data drift monitoring
- Prometheus and Grafana
- Cloud deployment

## Author

Industrial Predictive Maintenance
End-to-End Machine Learning Project
