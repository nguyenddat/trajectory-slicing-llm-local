import json

from core.chat_models import get_chat_model
from data.final_schema import Trajectory
from method_slicing.base import SlicingMethod

from .base import FailureAttributionMethod, FailureAttributionOutput
from .utils.chat import get_chat_completion
from .utils.all_at_once_prompt import AllAtOnceInput, all_at_once_parser, all_at_once_prompt


class AllAtOnceMethod(FailureAttributionMethod):
    def __init__(
        self, model_name: str, slicing_method: SlicingMethod | None = None
    ) -> None:
        self.method_name = "all at once"
        self.slicing_method = slicing_method
        self.model = get_chat_model(model_name)
        self.last_slices = []

    def predict(self, trajectory: Trajectory) -> FailureAttributionOutput:
        total_cost = {
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }
        if not trajectory.behaviors:
            self.last_slices = []
            return FailureAttributionOutput(pred_step_index=-1, **total_cost)

        segments = (
            self.slicing_method.slice(trajectory.behaviors)
            if self.slicing_method is not None
            else [trajectory.behaviors]
        )
        slicing_cost = getattr(self.slicing_method, "last_cost", {})
        for key in total_cost:
            total_cost[key] += slicing_cost.get(key, 0)
        self.last_slices = segments
        if not segments or any(not segment for segment in segments):
            raise ValueError("Slicing method returned an empty segment")

        original_indices = {
            id(behavior): index
            for index, behavior in enumerate(trajectory.behaviors, start=1)
        }
        for segment in segments:
            try:
                segment_indices = [original_indices[id(behavior)] for behavior in segment]
            except KeyError as error:
                raise ValueError(
                    "Slicing method returned a behavior outside the trajectory"
                ) from error
            trajectory_content = json.dumps(
                [
                    {
                        "index": original_indices[id(behavior)],
                        "step_index": behavior.step_index,
                        "agent_name": behavior.agent_name,
                        "role": behavior.role,
                        "content": behavior.format_content(),
                    }
                    for behavior in segment
                ],
                ensure_ascii=False,
            )
            response, cost = get_chat_completion(
                model=self.model,
                prompt_template=all_at_once_prompt,
                parser=all_at_once_parser,
                prompt_params=AllAtOnceInput(
                    task=trajectory.task,
                    trajectory_content=trajectory_content,
                ),
            )
            for key in total_cost:
                total_cost[key] += cost[key]

            pred_step_index = response["pred_step_index"]
            if pred_step_index == -1:
                continue
            if pred_step_index not in segment_indices:
                raise ValueError(f"Model returned invalid step index: {pred_step_index}")
            return FailureAttributionOutput(pred_step_index=pred_step_index, **total_cost)

        return FailureAttributionOutput(pred_step_index=-1, **total_cost)
