from typing import List

from data.final_schema import AgentBehavior

from .base import SlicingMethod

from utils.token import count_tokens


class FixedTokenSlicing(SlicingMethod):
    def __init__(
        self,
        max_tokens_per_slice: int,
        overlapping: int = 0,
    ):
        if max_tokens_per_slice <= 0:
            raise ValueError("max_tokens_per_slice must be positive")
        if not 0 <= overlapping < max_tokens_per_slice:
            raise ValueError("overlapping must be between 0 and max_tokens_per_slice - 1")

        self.max_tokens_per_slice = max_tokens_per_slice
        self.overlapping = overlapping

    def slice(self, trajectory: List[AgentBehavior]) -> List[List[AgentBehavior]]:
        token_counts = [count_tokens(behavior.format_content()) for behavior in trajectory]
        slices = []
        start = 0

        while start < len(trajectory):
            end = start
            total_tokens = 0
            while (
                end < len(trajectory)
                and total_tokens + token_counts[end] <= self.max_tokens_per_slice
            ):
                total_tokens += token_counts[end]
                end += 1

            # A single step can exceed the budget; keep it intact as a one-step slice.
            if end == start:
                end += 1

            slices.append(trajectory[start:end])
            if end == len(trajectory):
                break

            next_start = end
            overlap_tokens = 0
            # Behaviors are indivisible steps: include a prior step only when the
            # whole step fits both the overlap budget and the next slice budget.
            while self.overlapping > 0 and next_start > start + 1:
                previous_tokens = token_counts[next_start - 1]
                if (
                    overlap_tokens + previous_tokens > self.overlapping
                    or overlap_tokens + previous_tokens + token_counts[end]
                    > self.max_tokens_per_slice
                ):
                    break
                overlap_tokens += previous_tokens
                next_start -= 1

            start = next_start
        return slices
