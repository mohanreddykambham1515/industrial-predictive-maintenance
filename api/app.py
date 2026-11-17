
from fastapi import FastAPI, HTTPException

from api.schemas import MachineInput, PredictionResponse
from src.config import MODEL_PATH
from src.predict import predict_machine

app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description="Machine Failure Prediction System",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Industrial Predictive Maintenance API",
        "status": "Running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_available": MODEL_PATH.exists()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: MachineInput):

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run python -m src.train first."
        )

    try:
        input_data = payload.model_dump(by_alias=True)
        return predict_machine(input_data)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
