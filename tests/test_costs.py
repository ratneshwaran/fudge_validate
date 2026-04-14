import pytest
from fudge.types import IntentBucket, Utterance
from fudge.embeddings import EmbeddingCache
from fudge.costs import FudgeCosts, cosine_distance


@pytest.fixture(scope="module")
def emb():
    return EmbeddingCache()


@pytest.fixture(scope="module")
def buckets():
    return [
        IntentBucket(actor="user", utterances=["I want to book a hotel", "Book me a room"], label="book_hotel"),
        IntentBucket(actor="user", utterances=["What is the weather"], label="ask_weather"),
        IntentBucket(actor="agent", utterances=["How can I help you", "What can I do for you"], label="greeting"),
        IntentBucket(actor="agent", utterances=["Your booking is confirmed"], label="confirm_booking"),
    ]


@pytest.fixture(scope="module")
def costs(emb, buckets):
    return FudgeCosts(emb, buckets)


def test_same_actor_similar(costs):
    bucket = IntentBucket(actor="user", utterances=["I want to book a hotel", "Book me a room"], label="book_hotel")
    utt = Utterance(actor="user", text="I need a hotel reservation")
    cost = costs.substitution_cost(bucket, utt)
    assert cost < 0.5, f"Similar same-actor should be low, got {cost}"


def test_same_actor_dissimilar(costs):
    bucket = IntentBucket(actor="user", utterances=["I want to book a hotel"], label="book_hotel")
    utt = Utterance(actor="user", text="What is the meaning of life")
    cost = costs.substitution_cost(bucket, utt)
    assert cost > 0.3, f"Dissimilar should be higher, got {cost}"


def test_actor_mismatch(costs):
    bucket = IntentBucket(actor="agent", utterances=["How can I help you"], label="greeting")
    utt = Utterance(actor="user", text="How can I help you")
    cost = costs.substitution_cost(bucket, utt)
    assert cost >= 1e8, f"Actor mismatch should be INF, got {cost}"


def test_cosine_distance_identical():
    import numpy as np
    a = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_distance(a, a)) < 1e-6


def test_cosine_distance_orthogonal():
    import numpy as np
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_distance(a, b) - 1.0) < 1e-6


def test_insertion_deletion_cost(costs):
    utt = Utterance(actor="user", text="anything")
    bucket = IntentBucket(actor="user", utterances=["anything"], label="x")
    assert costs.insertion_cost(utt) == 1.0
    assert costs.deletion_cost(bucket) == 1.0


def test_substitution_symmetry_property(costs):
    """Similar utterance to matching bucket should cost less than dissimilar."""
    bucket = IntentBucket(actor="user", utterances=["I want to book a hotel", "Book me a room"], label="book_hotel")
    close_utt = Utterance(actor="user", text="Reserve a hotel room please")
    far_utt = Utterance(actor="user", text="Tell me about quantum physics")
    close_cost = costs.substitution_cost(bucket, close_utt)
    far_cost = costs.substitution_cost(bucket, far_utt)
    assert close_cost < far_cost, f"Close ({close_cost}) should be < far ({far_cost})"
