import numpy as np
import pytest

from fudge.types import Conversation, Utterance, IntentBucket
from fudge.embeddings import EmbeddingCache
from fudge.segment import segment_conversation, _rle_with_smoothing


@pytest.fixture(scope="module")
def emb():
    return EmbeddingCache()


@pytest.fixture(scope="module")
def buckets():
    return [
        IntentBucket(actor="agent",
                     utterances=["What is your name?", "How old are you?",
                                 "Where do you live?"],
                     label="ask_personal"),
        IntentBucket(actor="agent",
                     utterances=["Let me explain how this therapy works.",
                                 "Here is the rationale for the procedure."],
                     label="explain_rationale"),
        IntentBucket(actor="user",
                     utterances=["My name is John.", "I am thirty years old.",
                                 "I live in London."],
                     label="give_personal"),
    ]


def _conv(*pairs):
    return Conversation(utterances=[Utterance(actor=a, text=t) for a, t in pairs])


# --- pure smoothing logic (no embeddings) ---

def test_rle_collapses_runs():
    assert _rle_with_smoothing([0, 0, 0], min_run=2) == [[0, 1, 2]]


def test_rle_absorbs_flanked_noise():
    # the lone `1` is flanked by `0` on both sides -> absorbed into one run
    assert _rle_with_smoothing([0, 0, 1, 0], min_run=2) == [[0, 1, 2, 3]]


def test_rle_keeps_distinct_labels():
    assert _rle_with_smoothing([0, 1, 2], min_run=2) == [[0], [1], [2]]


def test_rle_does_not_absorb_when_neighbours_differ():
    # mid `1` is flanked by different labels (0 then 2) -> kept separate
    assert _rle_with_smoothing([0, 1, 2], min_run=2) == [[0], [1], [2]]


# --- full segmentation with real embeddings ---

def test_consecutive_same_bucket_collapses(emb, buckets):
    conv = _conv(("agent", "What's your name?"),
                 ("agent", "And your age?"),
                 ("agent", "Which city are you in?"))
    seg = segment_conversation(conv, buckets, emb, min_run=2)
    assert len(seg.utterances) == 1
    assert seg.utterances[0].actor == "agent"


def test_noise_turn_is_absorbed(emb, buckets):
    conv = _conv(("agent", "What's your name?"),
                 ("agent", "And your age?"),
                 ("agent", "Here is the rationale for the procedure."),  # noise
                 ("agent", "Which city are you in?"))
    seg = segment_conversation(conv, buckets, emb, min_run=2)
    assert len(seg.utterances) == 1  # flanked noise absorbed


def test_distinct_buckets_not_merged(emb, buckets):
    conv = _conv(("agent", "What's your name?"),
                 ("agent", "Let me explain how the therapy works."))
    seg = segment_conversation(conv, buckets, emb, min_run=2)
    assert len(seg.utterances) == 2  # different stages stay separate


def test_per_actor_collapse_and_interleave(emb, buckets):
    conv = _conv(("agent", "What's your name?"),
                 ("user", "My name is Alice."),
                 ("agent", "How old are you?"),
                 ("user", "I'm twenty-nine."))
    seg = segment_conversation(conv, buckets, emb, min_run=2)
    # agent stream collapses to one segment, user stream to one -> 2 total
    assert len(seg.utterances) == 2
    assert [u.actor for u in seg.utterances] == ["agent", "user"]


def test_segments_are_unit_norm_and_not_longer(emb, buckets):
    conv = _conv(("agent", "What's your name?"),
                 ("agent", "And your age?"),
                 ("user", "My name is Bob."))
    seg = segment_conversation(conv, buckets, emb, min_run=2)
    assert len(seg.utterances) <= len(conv.utterances)
    for u in seg.utterances:
        assert u.embedding is not None
        assert abs(float(np.linalg.norm(u.embedding)) - 1.0) < 1e-5


def test_metadata_preserved(emb, buckets):
    conv = Conversation(
        utterances=[Utterance("agent", "What's your name?")],
        task="P5", dialogue_id=123,
    )
    seg = segment_conversation(conv, buckets, emb, min_run=2)
    assert seg.task == "P5"
    assert seg.dialogue_id == 123
