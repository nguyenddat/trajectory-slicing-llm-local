import json

from core.chat_models import get_chat_model
from data.final_schema import AgentBehavior, Trajectory
from method_slicing.base import SlicingMethod
from method_slicing.fixed_step import FixedStepSlicing

from .base import FailureAttributionMethod, FailureAttributionOutput
from .utils.chat import get_chat_completion
from .utils.step_by_step_prompt import (
    StepByStepInput,
    step_by_step_parser,
    step_by_step_centered_prompt,
    step_by_step_future_prompt,
    step_by_step_past_prompt,
    step_by_step_semantic_prompt,
)


class PastOnlyFixedStepSlicing(FixedStepSlicing):
    """Fixed-width prior context for step-by-step attribution."""

    context_direction = "past"


class FutureOnlyFixedStepSlicing(FixedStepSlicing):
    """Fixed-width future context for step-by-step attribution."""

    context_direction = "future"


class StepByStepMethod(FailureAttributionMethod):
    """Evaluate each step with a directional or slicer-defined context."""

    def __init__(
        self,
        model_name: str,
        slicing_method: SlicingMethod | None = None,
    ) -> None:
        self.method_name = "step by step"
        self.slicing_method = slicing_method
        self.model = get_chat_model(model_name)
        self.last_slices = []

    def predict(
        self,
        trajectory: Trajectory,
        precomputed_slice_ranges: list[list[int]] | None = None,
    ) -> FailureAttributionOutput:
        total_cost = {
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }
        behaviors = trajectory.behaviors
        if not behaviors:
            self.last_slices = []
            return FailureAttributionOutput(pred_step_index=-1, **total_cost)

        if self.slicing_method is None:
            # Baseline: provide the full trajectory prefix before the target.
            contexts = [
                behaviors[:target_position]
                for target_position in range(len(behaviors))
            ]
            self.last_slices = contexts
            prompt_template = step_by_step_past_prompt
        elif isinstance(self.slicing_method, FixedStepSlicing):
            if self.slicing_method.overlapping:
                raise ValueError("Directional fixed-step context does not support overlap")

            width = self.slicing_method.max_steps_per_slice
            context_direction = getattr(
                self.slicing_method, "context_direction", "centered"
            )
            if context_direction == "centered":
                context_radius = width // 2
                contexts = [
                    behaviors[
                        max(0, target_position - context_radius) : min(
                            len(behaviors), target_position + context_radius + 1
                        )
                    ]
                    for target_position in range(len(behaviors))
                ]
            elif context_direction == "past":
                contexts = [
                    behaviors[max(0, target_position - width) : target_position]
                    for target_position in range(len(behaviors))
                ]
            else:
                contexts = [
                    behaviors[target_position + 1 : target_position + 1 + width]
                    for target_position in range(len(behaviors))
                ]
            self.last_slices = contexts
            prompt_template = {
                "centered": step_by_step_centered_prompt,
                "past": step_by_step_past_prompt,
                "future": step_by_step_future_prompt,
            }[context_direction]
        else:
            semantic_slices = _slices_from_ranges(
                behaviors,
                precomputed_slice_ranges,
            )
            if semantic_slices is None:
                semantic_slices = self.slicing_method.slice(behaviors)
                slicing_cost = getattr(self.slicing_method, "last_cost", {})
            else:
                slicing_cost = {}
            for key in total_cost:
                total_cost[key] += slicing_cost.get(key, 0)

            context_for_behavior = {}
            for current_slice in semantic_slices:
                if not current_slice:
                    raise ValueError("Slicing method returned an empty slice")
                for behavior in current_slice:
                    behavior_id = id(behavior)
                    if behavior_id in context_for_behavior:
                        raise ValueError(
                            "Semantic slices must assign each behavior to one slice"
                        )
                    context_for_behavior[behavior_id] = current_slice

            if set(context_for_behavior) != {id(behavior) for behavior in behaviors}:
                raise ValueError(
                    "Slicing method must assign every trajectory behavior to a slice"
                )
            contexts = [context_for_behavior[id(behavior)] for behavior in behaviors]
            self.last_slices = semantic_slices
            prompt_template = step_by_step_semantic_prompt

        formatted_behaviors = [behavior.format_content() for behavior in behaviors]

        for target_position, current_slice in enumerate(contexts):
            response, cost = get_chat_completion(
                model=self.model,
                prompt_template=prompt_template,
                parser=step_by_step_parser,
                prompt_params=StepByStepInput(
                    task=trajectory.task,
                    target_step_content=formatted_behaviors[target_position],
                    context_content=json.dumps(
                        [behavior.format_content() for behavior in current_slice],
                        ensure_ascii=False,
                    ),
                ),
            )
            for key in total_cost:
                total_cost[key] += cost[key]

            if response["error_found"]:
                return FailureAttributionOutput(
                    pred_step_index=target_position + 1,
                    **total_cost,
                )

        return FailureAttributionOutput(pred_step_index=-1, **total_cost)


def _slices_from_ranges(
    behaviors: list[AgentBehavior],
    slice_ranges: list[list[int]] | None,
) -> list[list[AgentBehavior]] | None:
    if not slice_ranges:
        return None

    slices = []
    next_start = 0
    for current_range in slice_ranges:
        if not isinstance(current_range, list) or len(current_range) != 2:
            return None
        start, end = current_range
        if (
            type(start) is not int
            or type(end) is not int
            or start != next_start
            or end <= start
            or end > len(behaviors)
        ):
            return None
        slices.append(behaviors[start:end])
        next_start = end

    return slices if next_start == len(behaviors) else None
