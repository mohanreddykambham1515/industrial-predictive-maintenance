
from pydantic import BaseModel, Field, ConfigDict


class MachineInput(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    Type: str = Field(pattern="^[LMH]$")

    air_temperature: float = Field(
        alias="Air temperature [K]"
    )

    process_temperature: float = Field(
        alias="Process temperature [K]"
    )

    rotational_speed: int = Field(
        alias="Rotational speed [rpm]",
        gt=0
    )

    torque: float = Field(
        alias="Torque [Nm]",
        ge=0
    )

    tool_wear: int = Field(
        alias="Tool wear [min]",
        ge=0
    )


class PredictionResponse(BaseModel):

    machine_failure_prediction: int

    failure_probability: float

    threshold: float

    status: str
