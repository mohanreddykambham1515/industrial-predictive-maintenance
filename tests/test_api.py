
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["status"] == "Running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert "model_available" in response.json()


def test_prediction_endpoint():

    payload = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    result = response.json()

    assert "machine_failure_prediction" in result
    assert "failure_probability" in result
    assert "threshold" in result
    assert "status" in result

    assert result["machine_failure_prediction"] in [0, 1]


def test_invalid_input():

    payload = {
        "Type": "INVALID",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422
