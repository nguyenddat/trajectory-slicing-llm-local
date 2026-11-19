from langchain_core.output_parsers import PydanticOutputParser
from typing import Annotated

from pydantic import BaseModel, Field


class SlicingInput(BaseModel):
    trajectory_content: str


class SlicingResponse(BaseModel):
    end_indices: list[Annotated[int, Field(ge=1)]] = Field(
        description=(
            "One-based indices of the last AgentBehavior in each slice, "
            "in strictly increasing order. Include the final trajectory index."
        )
    )


response_parser = PydanticOutputParser(pydantic_object=SlicingResponse)


slicing_prompt = """
You are dividing a multi-agent trajectory into consecutive, coherent slices.

Each item is one indivisible AgentBehavior, identified by its one-based index.
Place a boundary before an item when the work changes to a distinct subtask,
phase, or topic. Keep related consecutive items together. Do not create a
boundary for a minor wording change or merely because a new agent speaks.

Return only the one-based index of the last item in each slice, in increasing
order. Include the final item of the trajectory. If no meaningful boundary
exists, return a list containing only the final index. Every item must belong
to exactly one slice, in its original order.

Trajectory:
{trajectory_content}

Answer strictly in the following JSON format:
"""
