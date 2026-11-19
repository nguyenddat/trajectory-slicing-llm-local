from typing import List

import numpy as np

from data.final_schema import AgentBehavior

from .base import SlicingMethod, add_extra_context
from .tokenization.bow import BOW
from .tokenization.transformer import Transformer
from .utils.preprocessing import preprocess


class C99Slice(SlicingMethod):
    def __init__(
        self,
        tokenizer: BOW | Transformer,
        rank_window: int = 11,
        min_slice_steps: int = 1,
        extra_context: int = 0,
    ):
        if rank_window <= 1 or rank_window % 2 == 0:
            raise ValueError("rank_window must be odd and greater than 1")
        if min_slice_steps <= 0:
            raise ValueError("min_slice_steps must be positive")
        if extra_context < 0:
            raise ValueError("extra_context must be non-negative")

        self.tokenizer = tokenizer
        self.rank_window = rank_window
        self.min_slice_steps = min_slice_steps
        self.extra_context = extra_context

    def slice(self, trajectory: List[AgentBehavior]) -> List[List[AgentBehavior]]:
        count = len(trajectory)
        if not count:
            return []
        if count < 2 * self.min_slice_steps:
            return [trajectory]

        contents = [behavior.format_content() for behavior in trajectory]
        if isinstance(self.tokenizer, BOW):
            words = [word for content in contents for word in preprocess(content)]
            self.tokenizer.fit(words)
            vectors = np.asarray(
                [self.tokenizer.transform(content) for content in contents]
            )
        elif isinstance(self.tokenizer, Transformer):
            vectors = np.asarray(self.tokenizer.transform_many(contents))
        else:
            raise TypeError(f"Unsupported tokenizer: {type(self.tokenizer).__name__}")

        norms = np.linalg.norm(vectors, axis=1)
        denominators = np.outer(norms, norms)
        similarity = np.divide(
            vectors @ vectors.T,
            denominators,
            out=np.zeros((count, count), dtype=float),
            where=denominators != 0,
        )

        radius = self.rank_window // 2
        ranks = np.zeros_like(similarity)
        for row in range(count):
            row_start = max(0, row - radius)
            row_end = min(count, row + radius + 1)
            for column in range(count):
                neighbors = similarity[
                    row_start:row_end,
                    max(0, column - radius) : min(count, column + radius + 1),
                ]
                ranks[row, column] = np.count_nonzero(
                    neighbors < similarity[row, column]
                ) / neighbors.size

        if not np.any(ranks):
            return [trajectory]

        prefix = np.pad(ranks, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

        def block_sum(start: int, end: int) -> float:
            return float(
                prefix[end, end]
                - prefix[start, end]
                - prefix[end, start]
                + prefix[start, start]
            )

        segments = [(0, count)]
        segmentations = [segments.copy()]
        total_sum = block_sum(0, count)
        total_area = count * count
        densities = [total_sum / total_area]

        while True:
            best_density = float("-inf")
            best_split = None
            for segment_index, (start, end) in enumerate(segments):
                old_sum = block_sum(start, end)
                old_area = (end - start) ** 2
                for boundary in range(
                    start + self.min_slice_steps,
                    end - self.min_slice_steps + 1,
                ):
                    left_area = (boundary - start) ** 2
                    right_area = (end - boundary) ** 2
                    area = total_area - old_area + left_area + right_area
                    rank_sum = (
                        total_sum
                        - old_sum
                        + block_sum(start, boundary)
                        + block_sum(boundary, end)
                    )
                    density = rank_sum / area
                    if density > best_density:
                        best_density = density
                        best_split = (
                            segment_index,
                            start,
                            boundary,
                            end,
                            rank_sum,
                            area,
                        )

            if best_split is None:
                break

            segment_index, start, boundary, end, total_sum, total_area = best_split
            segments[segment_index : segment_index + 1] = [
                (start, boundary),
                (boundary, end),
            ]
            segmentations.append(segments.copy())
            densities.append(best_density)

        gains = np.diff(densities)
        if not len(gains):
            return [trajectory]

        smoothing_mask = np.array([1, 2, 4, 8, 4, 2, 1], dtype=float)
        smoothing_mask /= smoothing_mask.sum()
        smoothed_gains = np.convolve(
            np.pad(gains, (3, 3), mode="edge"), smoothing_mask, mode="valid"
        )
        threshold = float(np.mean(smoothed_gains) + 1.2 * np.std(smoothed_gains))
        split_count = 0
        for gain in smoothed_gains:
            if gain <= threshold:
                break
            split_count += 1

        chosen_segments = segmentations[split_count]
        slices = [trajectory[start:end] for start, end in chosen_segments]
        return add_extra_context(trajectory, slices, self.extra_context)
