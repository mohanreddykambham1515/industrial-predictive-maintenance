
import pandas as pd

from src.config import DATA_PATH, FEATURES, TARGET


def load_dataset(path=DATA_PATH) -> pd.DataFrame:

    # Check whether dataset exists
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place ai4i2020.csv inside the data folder."
        )

    # Load dataset
    df = pd.read_csv(path)

    # Validate required columns
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select only required columns
    df = df[required_columns].copy()

    # Validate target variable
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise"
    ).astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target variable must contain only 0 and 1."
        )

    return df
