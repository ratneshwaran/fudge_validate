from sentence_transformers import SentenceTransformer
import numpy as np

from .types import IntentBucket


class EmbeddingCache:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            embeddings = self.model.encode(uncached, normalize_embeddings=True)
            for t, e in zip(uncached, embeddings):
                self._cache[t] = e
        return np.array([self._cache[t] for t in texts])

    def intent_centroid(self, bucket: IntentBucket) -> np.ndarray:
        """Paper Eq 10: e_Br = mean of embeddings of all utterances in the bucket."""
        embeddings = self.encode_batch(bucket.utterances)
        centroid = embeddings.mean(axis=0)
        return centroid / np.linalg.norm(centroid)
