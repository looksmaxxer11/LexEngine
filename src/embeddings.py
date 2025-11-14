from __future__ import annotations

from typing import List, Optional


class EmbeddingGenerator:
    """
    Sentence-Transformers embedding generator with lazy model load.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    "sentence-transformers is required for embeddings."
                ) from e
        return self._model

    def encode(self, text: str) -> List[float]:
        model = self._ensure_model()
        vec = model.encode(text or "", normalize_embeddings=True)
        return [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]


__all__ = ["EmbeddingGenerator"]
