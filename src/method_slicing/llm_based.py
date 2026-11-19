import json
from typing import List

from core.chat_models import get_chat_model
from data.final_schema import AgentBehavior

from .base import SlicingMethod
from .utils.chat import get_chat_completion
from .utils.llm_based_prompt import SlicingInput, response_parser, slicing_prompt


class LlmBasedSlicing(SlicingMethod):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = get_chat_model(self.model_name)
        self.last_cost = {
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }

    def slice(self, trajectory: List[AgentBehavior]) -> List[List[AgentBehavior]]:
        self.last_cost = {
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }
        if not trajectory:
            return []
        if len(trajectory) == 1:
            return [trajectory]

        trajectory_content = json.dumps(
            [
                {
                    "index": index,
                    "step_index": behavior.step_index,
                    "agent_name": behavior.agent_name,
                    "role": behavior.role,
                    "content": behavior.format_content(),
                }
                for index, behavior in enumerate(trajectory, start=1)
            ],
            ensure_ascii=False,
        )
        response, self.last_cost = get_chat_completion(
            model=self.model,
            prompt_template=slicing_prompt,
            parser=response_parser,
            prompt_params=SlicingInput(trajectory_content=trajectory_content),
        )
        end_indices = response["end_indices"]

        if (
            not end_indices
            or end_indices != sorted(set(end_indices))
            or end_indices[-1] != len(trajectory)
            or any(not 1 <= end <= len(trajectory) for end in end_indices)
        ):
            raise ValueError("Model returned invalid slice end indices")

        slices = []
        start = 0
        for end in end_indices:
            slices.append(trajectory[start:end])
            start = end
        return slices
