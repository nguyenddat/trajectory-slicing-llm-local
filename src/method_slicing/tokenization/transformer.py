import numpy as np

from core.embedding_models import get_embedding_model


class Transformer:
    def __init__(self, model_name: str = "openai/text-embedding-3-small"):
        self.model = get_embedding_model(model_name)

    def transform(self, content: str) -> np.ndarray:
        return self.transform_many([content])[0]

    def transform_many(self, contents: list[str]) -> list[np.ndarray]:
        if not contents:
            return []
        return [
            np.asarray(embedding, dtype=np.float32)
            for embedding in self.model.embed_documents(contents)
        ]
