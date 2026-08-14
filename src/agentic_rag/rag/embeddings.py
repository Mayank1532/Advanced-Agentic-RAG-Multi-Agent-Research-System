from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate normalized embeddings using a local SentenceTransformer."""

    def __init__(self, model_path: Path) -> None:
        self.model = SentenceTransformer(
            str(model_path),
            local_files_only=True,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)
