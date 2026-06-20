import numpy as np

from .types import IntentBucket, Utterance
from .embeddings import EmbeddingCache

INF = 1e9  # don't use float('inf') — causes NaN in DP min()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity. Range [0, 2]. 0 = identical."""
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return 1.0 - sim


class FudgeCosts:
    def __init__(self, embeddings: EmbeddingCache, all_buckets: list[IntentBucket],
                 method: str = 'centroid', alpha: float = 0.5):
        self.emb = embeddings
        self.all_buckets = all_buckets
        self.method = method
        self.alpha = alpha

        # Precompute all intent centroids
        self._centroids: dict[int, np.ndarray] = {
            id(b): self.emb.intent_centroid(b) for b in all_buckets
        }
        # Cache B* lookups: (actor, text) -> IntentBucket
        self._best_bucket_cache: dict[tuple[str, str], IntentBucket | None] = {}
        # Cache d1 values: (bucket_id, actor, text) -> float
        self._d1_cache: dict[tuple[int, str, str], float] = {}
        # Cache substitution costs: (bucket_id, actor, text) -> float
        self._sub_cache: dict[tuple[int, str, str], float] = {}

    def _get_centroid(self, bucket: IntentBucket) -> np.ndarray:
        """Get centroid for a bucket, computing lazily if not precomputed."""
        bid = id(bucket)
        if bid not in self._centroids:
            self._centroids[bid] = self.emb.intent_centroid(bucket)
        return self._centroids[bid]

    def _d1(self, bucket: IntentBucket, utterance: Utterance) -> float:
        """Intent-utterance distance (Eq 11)."""
        if bucket.actor != utterance.actor:
            return INF

        cache_key = (id(bucket), utterance.actor, utterance.text)
        if cache_key in self._d1_cache:
            return self._d1_cache[cache_key]

        # Use a precomputed embedding when the utterance carries one (collapsed
        # segments from segment_conversation); otherwise encode the text.
        u_emb = (utterance.embedding if utterance.embedding is not None
                 else self.emb.encode(utterance.text))

        if self.method == 'centroid':
            result = cosine_distance(self._get_centroid(bucket), u_emb)
        elif self.method == 'min':
            dists = [cosine_distance(self.emb.encode(u), u_emb) for u in bucket.utterances]
            result = min(dists)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._d1_cache[cache_key] = result
        return result

    def _d2(self, bucket_a: IntentBucket, bucket_b: IntentBucket) -> float:
        """Intent-intent distance (Eq 12)."""
        if bucket_a.actor != bucket_b.actor:
            return INF
        return cosine_distance(self._get_centroid(bucket_a), self._get_centroid(bucket_b))

    def _find_best_bucket(self, utterance: Utterance) -> IntentBucket | None:
        """Find B* = intent bucket closest to utterance u (same actor only)."""
        cache_key = (utterance.actor, utterance.text)
        if cache_key in self._best_bucket_cache:
            return self._best_bucket_cache[cache_key]

        best_bucket = None
        best_dist = INF
        for b in self.all_buckets:
            d = self._d1(b, utterance)
            if d < best_dist:
                best_dist = d
                best_bucket = b

        self._best_bucket_cache[cache_key] = best_bucket
        return best_bucket

    def substitution_cost(self, bucket: IntentBucket, utterance: Utterance) -> float:
        """Paper Eq 8: cost_sub(B_r, u) = alpha * (d1(B_r, u) + d2(B_r, B*))"""
        if bucket.actor != utterance.actor:
            return INF

        cache_key = (id(bucket), utterance.actor, utterance.text)
        if cache_key in self._sub_cache:
            return self._sub_cache[cache_key]

        d1_val = self._d1(bucket, utterance)
        b_star = self._find_best_bucket(utterance)

        if b_star is None:
            self._sub_cache[cache_key] = INF
            return INF

        d2_val = self._d2(bucket, b_star)
        result = self.alpha * (d1_val + d2_val)
        self._sub_cache[cache_key] = result
        return result

    def insertion_cost(self, utterance: Utterance) -> float:
        return 1.0

    def deletion_cost(self, bucket: IntentBucket) -> float:
        return 1.0
