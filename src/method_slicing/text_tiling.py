from typing import List

import numpy as np

from data.final_schema import AgentBehavior

from .base import SlicingMethod, add_extra_context
from .tokenization.bow import BOW
from .tokenization.transformer import Transformer
from .utils.preprocessing import preprocess


class TextTilingSlice(SlicingMethod):
    def __init__(
        self,
        tokenizer: BOW | Transformer,
        window_step: int,
        smoothing_width: int,
        smoothing_rounds: int,
        min_slice_steps: int,
        extra_context: int = 0,
    ):
        if window_step <= 0:
            raise ValueError("window_step must be positive")
        if smoothing_width <= 0:
            raise ValueError("smoothing_width must be positive")
        if smoothing_rounds < 0:
            raise ValueError("smoothing_rounds must be non-negative")
        if min_slice_steps <= 0:
            raise ValueError("min_slice_steps must be positive")
        if extra_context < 0:
            raise ValueError("extra_context must be non-negative")

        self.tokenizer = tokenizer
        self.window_step = window_step
        self.smoothing_width = smoothing_width
        self.smoothing_rounds = smoothing_rounds
        self.min_slice_steps = min_slice_steps
        self.extra_context = extra_context

    def slice(self, trajectory: List[AgentBehavior]) -> List[List[AgentBehavior]]:
        if not trajectory:
            return []

        gaps = list(range(self.window_step, len(trajectory) - self.window_step + 1))
        if not gaps:
            return [trajectory]

        left_contents = [
            "\n".join(
                behavior.format_content()
                for behavior in trajectory[gap - self.window_step : gap]
            )
            for gap in gaps
        ]
        right_contents = [
            "\n".join(
                behavior.format_content()
                for behavior in trajectory[gap : gap + self.window_step]
            )
            for gap in gaps
        ]

        if isinstance(self.tokenizer, BOW):
            words = [
                word
                for behavior in trajectory
                for word in preprocess(behavior.format_content())
            ]
            self.tokenizer.fit(words)
            vectors = [
                self.tokenizer.transform(content)
                for content in left_contents + right_contents
            ]
        elif isinstance(self.tokenizer, Transformer):
            vectors = self.tokenizer.transform_many(left_contents + right_contents)
        else:
            raise TypeError(f"Unsupported tokenizer: {type(self.tokenizer).__name__}")

        similarities = []
        for left_vector, right_vector in zip(
            vectors[: len(gaps)], vectors[len(gaps) :]
        ):
            denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
            similarities.append(
                float(np.dot(left_vector, right_vector) / denominator)
                if denominator
                else 0.0
            )

        scores = np.asarray(similarities, dtype=np.float32)
        radius = self.smoothing_width // 2
        for _ in range(self.smoothing_rounds):
            previous = scores.copy()
            for index in range(len(scores)):
                window = previous[max(0, index - radius) : index + radius + 1]
                scores[index] = float(np.mean(window))

        depth_scores = np.zeros(len(scores), dtype=np.float32)
        for index in range(len(scores)):
            left = index
            while left > 0 and scores[left - 1] >= scores[left]:
                left -= 1

            right = index
            while right < len(scores) - 1 and scores[right + 1] >= scores[right]:
                right += 1

            depth_scores[index] = scores[left] + scores[right] - 2 * scores[index]

        threshold = float(np.mean(depth_scores) - np.std(depth_scores) / 2)
        candidates = sorted(
            (index for index, depth in enumerate(depth_scores) if depth > threshold),
            key=lambda index: depth_scores[index],
            reverse=True,
        )

        boundaries = []
        for index in candidates:
            boundary = gaps[index]
            if (
                boundary >= self.min_slice_steps
                and len(trajectory) - boundary >= self.min_slice_steps
                and all(
                    abs(boundary - accepted) >= self.min_slice_steps
                    for accepted in boundaries
                )
            ):
                boundaries.append(boundary)

        boundaries.sort()
        starts = [0, *boundaries]
        ends = [*boundaries, len(trajectory)]
        slices = [trajectory[start:end] for start, end in zip(starts, ends)]
        return add_extra_context(trajectory, slices, self.extra_context)
