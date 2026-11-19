from abc import ABC, abstractmethod
from typing import List

from data.final_schema import AgentBehavior


def add_extra_context(
    trajectory: List[AgentBehavior],
    slices: List[List[AgentBehavior]],
    extra_context: int,
) -> List[List[AgentBehavior]]:
    """Expand each contiguous slice by up to N neighboring steps at both ends."""
    if extra_context == 0:
        return slices

    expanded = []
    start = 0
    for current_slice in slices:
        end = start + len(current_slice)
        expanded.append(
            trajectory[
                max(0, start - extra_context) : min(len(trajectory), end + extra_context)
            ]
        )
        start = end
    return expanded


class SlicingMethod(ABC):
    @abstractmethod
    def slice(self, trajectory: List[AgentBehavior]) -> List[List[AgentBehavior]]:
        """Split a trajectory into slices of agent behaviors."""
        raise NotImplementedError()
