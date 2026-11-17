
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.config import MODEL_PATH, FEATURES, TARGET
from src.data import load_dataset


def evaluate_model():

    # 1. Load saved model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.train first."
        )

    model_bundle = joblib.load(MODEL_PATH)

    model = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]

    # 2. Load dataset
    df = load_dataset()

    X = df[FEATURES]
    y = df[TARGET]

    # 3. Generate predictions
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # 4. Calculate metrics
    print("\nMODEL EVALUATION")
    print("----------------")

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred, zero_division=0))
    print("Recall:", recall_score(y, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y, y_pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y, y_prob))
    print("PR-AUC:", average_precision_score(y, y_prob))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y,
        y_pred,
        zero_division=0
    ))


if __name__ == "__main__":
    evaluate_model()
