"""Shared contract for failure-attribution methods."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, field_validator

from data.final_schema import Trajectory


class FailureAttributionOutput(BaseModel):
    """Predicted failure step and model-call cost for one trajectory."""

    pred_step_index: int  # 1-based position; -1 means no error found.
    latency: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int

    @field_validator("pred_step_index")
    @classmethod
    def validate_step_index(cls, value: int) -> int:
        if value != -1 and value < 1:
            raise ValueError("pred_step_index must be 1-based or -1")
        return value


class FailureAttributionMethod(ABC):
    """Interface implemented by each failure-attribution method."""

    @abstractmethod
    def predict(self, trajectory: Trajectory) -> FailureAttributionOutput:
        """Predict the failure step for a normalized trajectory."""
        raise NotImplementedError()
