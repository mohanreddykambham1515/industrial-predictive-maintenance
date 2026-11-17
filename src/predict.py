
import joblib
import pandas as pd

from src.config import MODEL_PATH, FEATURES


def predict_machine(payload: dict) -> dict:

    # 1. Check model availability
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.train first."
        )

    # 2. Load trained model
    model_bundle = joblib.load(MODEL_PATH)

    model = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]

    # 3. Convert input into DataFrame
    input_data = pd.DataFrame(
        [payload],
        columns=FEATURES
    )

    # 4. Generate failure probability
    probability = model.predict_proba(input_data)[0][1]

    # 5. Apply decision threshold
    prediction = int(probability >= threshold)

    # 6. Return prediction
    return {
        "machine_failure_prediction": prediction,
        "failure_probability": float(probability),
        "threshold": threshold,
        "status": (
            "Failure Risk"
            if prediction == 1
            else "No Failure Predicted"
        )
    }
