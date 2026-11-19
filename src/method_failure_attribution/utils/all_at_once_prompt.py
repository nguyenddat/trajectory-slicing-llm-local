"""Prompt and response parser for all-at-once failure attribution."""

from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import PydanticOutputParser


class AllAtOnceInput(BaseModel):
    task: str
    trajectory_content: str


class AllAtOnceResponse(BaseModel):
    pred_step_index: int = Field(
        ge=-1,
        description=(
            "One-based index in the original trajectory of the first important "
            "mistake in the provided part; -1 if it contains no important mistake."
        ),
    )

    @field_validator("pred_step_index")
    @classmethod
    def validate_step_index(cls, value: int) -> int:
        if value == 0:
            raise ValueError("pred_step_index must be 1-based or -1")
        return value


all_at_once_parser = PydanticOutputParser(pydantic_object=AllAtOnceResponse)


all_at_once_prompt = """
You are analyzing a multi-agent trajectory for a problem-solving task.

Identify the first behavior in the provided trajectory in which an agent made
an important mistake that could directly lead to an incorrect final result.
Each behavior is labeled with its one-based index in the original trajectory.

Rules:
- Return the one-based trajectory index of the first important mistake in this input.
- Ignore minor wording issues and harmless inaccuracies.
- If later mistakes occur, still return the first important mistake.
- If this input contains no important mistake, return -1.
- Base your judgment only on the task and trajectory below.

Task:
{task}

Provided trajectory:
{trajectory_content}

Answer strictly in the following JSON format:
"""
