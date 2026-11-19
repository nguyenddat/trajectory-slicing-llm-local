from typing import List

from data.final_schema import AgentBehavior

from .base import SlicingMethod


class FixedStepSlicing(SlicingMethod):
    def __init__(
        self,
        max_steps_per_slice: int,
        overlapping: int = 0,
    ):
        if max_steps_per_slice <= 0:
            raise ValueError("max_steps_per_slice must be positive")

        if not 0 <= overlapping < max_steps_per_slice:
            raise ValueError("overlapping must be between 0 and max_steps_per_slice - 1")

        self.max_steps_per_slice = max_steps_per_slice
        self.overlapping = overlapping

    def slice(self, trajectory: List[AgentBehavior]) -> List[List[AgentBehavior]]:
        slices = []
        stride = self.max_steps_per_slice - self.overlapping

        for start in range(0, len(trajectory), stride):
            slices.append(trajectory[start : start + self.max_steps_per_slice])
            if start + self.max_steps_per_slice >= len(trajectory):
                break

        return slices
