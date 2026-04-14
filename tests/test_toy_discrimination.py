"""
Hand-crafted sanity check: FuDGE should give lower scores to conversations
that match a flow and higher scores to conversations that don't.
"""
import pytest
from fudge.types import IntentBucket, Utterance, Conversation, DialogueFlow
from fudge.embeddings import EmbeddingCache
from fudge.costs import FudgeCosts
from fudge.fudge_efficient import fudge_efficient


@pytest.fixture(scope="module")
def emb():
    return EmbeddingCache()


def build_hotel_flow():
    """A simple hotel booking flow."""
    buckets = {
        "greet_u": IntentBucket("user", ["hello", "hi", "good morning"], "greet_user"),
        "greet_a": IntentBucket("agent", ["hello how can I help you", "hi there"], "greet_agent"),
        "ask_hotel": IntentBucket("user", ["I want to book a hotel", "need a hotel room", "book a room"], "ask_hotel"),
        "ask_dates": IntentBucket("agent", ["what dates would you like", "when do you arrive"], "ask_dates"),
        "provide_dates": IntentBucket("user", ["next monday", "march 1 to march 5", "this weekend"], "provide_dates"),
        "confirm": IntentBucket("agent", ["booking confirmed", "all set", "done"], "confirm"),
    }
    flow = DialogueFlow()
    for nid, bucket in buckets.items():
        flow.add_node(nid, bucket)
    flow.add_edge(flow.root, "greet_u")
    flow.add_edge("greet_u", "greet_a")
    flow.add_edge("greet_a", "ask_hotel")
    flow.add_edge("ask_hotel", "ask_dates")
    flow.add_edge("ask_dates", "provide_dates")
    flow.add_edge("provide_dates", "confirm")
    return flow, list(buckets.values())


def test_in_task_lower_than_out_of_task(emb):
    """Hotel conversation vs hotel flow should score lower than weather conversation."""
    flow, all_buckets = build_hotel_flow()
    costs = FudgeCosts(emb, all_buckets)

    # In-task: hotel booking conversation
    hotel_conv = Conversation(utterances=[
        Utterance("user", "hi there"),
        Utterance("agent", "hello how can I help"),
        Utterance("user", "I want to reserve a hotel room"),
        Utterance("agent", "when would you like to stay"),
        Utterance("user", "this friday"),
        Utterance("agent", "your booking is confirmed"),
    ])

    # Out-of-task: weather conversation
    weather_conv = Conversation(utterances=[
        Utterance("user", "what is the weather like today"),
        Utterance("agent", "it is sunny and 72 degrees"),
        Utterance("user", "will it rain tomorrow"),
        Utterance("agent", "there is a 30 percent chance of rain"),
        Utterance("user", "should I bring an umbrella"),
        Utterance("agent", "yes I would recommend it"),
    ])

    hotel_score = fudge_efficient(hotel_conv, flow, costs)
    weather_score = fudge_efficient(weather_conv, flow, costs)

    print(f"\nHotel score (in-task):   {hotel_score:.4f}")
    print(f"Weather score (out-task): {weather_score:.4f}")
    print(f"Ratio: {weather_score / hotel_score:.2f}x")

    assert hotel_score < weather_score, \
        f"In-task ({hotel_score:.4f}) should be < out-of-task ({weather_score:.4f})"


def test_multiple_out_of_task_all_higher(emb):
    """All out-of-task conversations should score higher than the in-task one."""
    flow, all_buckets = build_hotel_flow()
    costs = FudgeCosts(emb, all_buckets)

    in_task = Conversation(utterances=[
        Utterance("user", "hello"),
        Utterance("agent", "hi how can I help you"),
        Utterance("user", "I need to book a hotel"),
        Utterance("agent", "what dates"),
        Utterance("user", "next week"),
        Utterance("agent", "done"),
    ])

    out_of_task = [
        Conversation(utterances=[
            Utterance("user", "what is the capital of France"),
            Utterance("agent", "the capital of France is Paris"),
        ]),
        Conversation(utterances=[
            Utterance("user", "I need to report a fraudulent charge"),
            Utterance("agent", "I can help with that, what is your account number"),
            Utterance("user", "my account number is 12345"),
            Utterance("agent", "I have flagged the transaction"),
        ]),
        Conversation(utterances=[
            Utterance("user", "can you order me a pizza"),
            Utterance("agent", "sure what toppings do you want"),
            Utterance("user", "pepperoni and mushrooms"),
            Utterance("agent", "your order has been placed"),
        ]),
    ]

    in_score = fudge_efficient(in_task, flow, costs)
    for i, conv in enumerate(out_of_task):
        out_score = fudge_efficient(conv, flow, costs)
        assert in_score < out_score, \
            f"In-task ({in_score:.4f}) should be < out-of-task[{i}] ({out_score:.4f})"
