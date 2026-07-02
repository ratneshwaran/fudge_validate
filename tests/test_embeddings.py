import numpy as np
import pytest
from fudge.embeddings import EmbeddingCache
from fudge.types import IntentBucket


@pytest.fixture(scope="module")
def emb():
    return EmbeddingCache()


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_similar_sentences_high_similarity(emb):
    a = emb.encode("I want to book a hotel")
    b = emb.encode("I need a hotel room")
    sim = cosine_sim(a, b)
    assert sim > 0.7, f"Expected > 0.7, got {sim}"


def test_dissimilar_sentences_low_similarity(emb):
    a = emb.encode("I want to book a hotel")
    b = emb.encode("What's the weather today")
    sim = cosine_sim(a, b)
    assert sim < 0.4, f"Expected < 0.4, got {sim}"


def test_intent_centroid_is_normalized(emb):
    bucket = IntentBucket(
        actor="user",
        utterances=["I want to book a hotel", "Book me a room", "Reserve a hotel"],
        label="book_hotel",
    )
    centroid = emb.intent_centroid(bucket)
    norm = float(np.linalg.norm(centroid))
    assert abs(norm - 1.0) < 1e-5, f"Centroid not normalized: norm={norm}"


def test_intent_centroid_empty_bucket_raises(emb):
    with pytest.raises(ValueError, match="empty bucket"):
        emb.intent_centroid(IntentBucket(actor="agent", utterances=[], label="empty"))


def test_cache_returns_same_result(emb):
    a1 = emb.encode("test caching")
    a2 = emb.encode("test caching")
    assert np.array_equal(a1, a2)
