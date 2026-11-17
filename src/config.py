
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

# Model artifacts
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "predictive_maintenance_pipeline.joblib"

METRICS_PATH = ARTIFACT_DIR / "metrics.json"


# Target variable
TARGET = "Machine failure"


# Numerical features
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# Categorical features
CATEGORICAL_FEATURES = [
    "Type"
]


# Combined model features
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# Columns excluded from training
EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]
