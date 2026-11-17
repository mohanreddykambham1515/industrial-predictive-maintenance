import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

from src.config import(
    FEATURES,
    TARGET,
    ARTIFACT_DIR,
    MODEL_PATH,
    METRICS_PATH
)
from src.data import load_dataset
from src.preprocessing import build_preprocessor

def train_model():

    df = load_dataset()

    X = df[FEATURES]
    y = df[TARGET]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )



    # 3. Build complete ML pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # 4. Train model
    model_pipeline.fit(X_train, y_train)

    # 5. Generate predictions
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    # 6. Evaluate model
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(
            y_test, y_pred
        ).tolist()
    }

    # 7. Display results
    print("\nMODEL PERFORMANCE")
    print("-----------------")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    # 8. Save model and metrics
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": model_pipeline,
            "threshold": threshold,
            "features": FEATURES
        },
        MODEL_PATH
    )

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4)
    )

    print("\nModel saved successfully!")
    print(f"Model path: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()