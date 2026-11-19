"""Prompt and response parser for step-by-step failure attribution."""

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class StepByStepInput(BaseModel):
    task: str
    target_step_content: str
    context_content: str


class StepByStepResponse(BaseModel):
    error_found: bool = Field(
        description="Whether the target step is an important mistake."
    )


step_by_step_parser = PydanticOutputParser(pydantic_object=StepByStepResponse)


_PROMPT_TEMPLATE = """
You are examining one target behavior in a multi-agent trajectory to determine
whether it is the first important mistake that could directly lead to an
incorrect final result.

The context is provided only to interpret the target behavior. Do not
identify a mistake in another behavior as a mistake in the target behavior.

Rules:
- Judge only the target behavior.
- Return error_found=true only when the target behavior itself introduces an
  important mistake.
- Ignore minor wording issues and harmless inaccuracies.
- Return error_found=false when the target behavior is not an important
  mistake, even if another behavior in the local context is mistaken.

{context_assumption}

Task:
{{task}}

Target behavior:
{{target_step_content}}

Context:
{{context_content}}

Answer strictly in the following JSON format:
"""


def _with_context_assumption(assumption: str) -> str:
    return _PROMPT_TEMPLATE.format(context_assumption=assumption)


step_by_step_centered_prompt = _with_context_assumption(
    "Context assumption: This is a local window centered on the target. It can "
    "include behaviors before and after the target, and may include the target "
    "again in the context. Later behaviors are retrospective evidence."
)
step_by_step_past_prompt = _with_context_assumption(
    "Context assumption: Only behaviors before the target are provided as "
    "context. They are the history available before the target; later behaviors "
    "are not available in this view."
)
step_by_step_future_prompt = _with_context_assumption(
    "Context assumption: Only behaviors after the target are provided as "
    "context. They are retrospective evidence, not information available to "
    "the agent at the target step. Use them to understand consequences."
)
step_by_step_semantic_prompt = _with_context_assumption(
    "Context assumption: The context is the semantic slice containing the "
    "target. It may include behaviors before and after it. Later behaviors are "
    "retrospective evidence."
)

# Preserve the original public name for callers that expect a default prompt.
step_by_step_prompt = step_by_step_centered_prompt
