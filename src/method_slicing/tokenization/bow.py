from collections import Counter

import numpy as np

from ..utils.preprocessing import preprocess


class BOW:
    def __init__(self):
        self.word_dictionary: list[str] = []

    def fit(self, words: list[str]) -> None:
        self.word_dictionary = list(dict.fromkeys(words))

    def transform(self, content: str) -> np.ndarray:
        word_counts = Counter(preprocess(content))
        return np.array(
            [word_counts.get(word, 0) for word in self.word_dictionary],
            dtype=np.float32,
        )
