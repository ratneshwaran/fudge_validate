"""
Critical correctness test: all three FuDGE scorers must agree on all inputs.
fudge_dag is the production Step-2 scorer, so it is part of the oracle matrix.
"""
import pytest
from fudge.types import IntentBucket, Utterance, Conversation, DialogueFlow
from fudge.embeddings import EmbeddingCache
from fudge.costs import FudgeCosts
from fudge.fudge_naive import fudge_naive, path_edit_distance
from fudge.fudge_efficient import fudge_efficient, fudge_dag


@pytest.fixture(scope="module")
def emb():
    return EmbeddingCache()


# --- Helper: build various flow shapes ---

def make_buckets():
    """A reusable set of intent buckets."""
    return {
        "greet_user": IntentBucket(actor="user", utterances=["hello", "hi there", "good morning"], label="greet_user"),
        "greet_agent": IntentBucket(actor="agent", utterances=["hello how can I help", "hi what do you need"], label="greet_agent"),
        "ask_hotel": IntentBucket(actor="user", utterances=["I want to book a hotel", "need a room"], label="ask_hotel"),
        "ask_food": IntentBucket(actor="user", utterances=["I want to order food", "get me pizza"], label="ask_food"),
        "confirm": IntentBucket(actor="agent", utterances=["done", "confirmed", "all set"], label="confirm"),
        "ask_details": IntentBucket(actor="agent", utterances=["what dates", "when do you arrive"], label="ask_details"),
        "provide_dates": IntentBucket(actor="user", utterances=["next monday", "from march 1 to march 5"], label="provide_dates"),
        "goodbye_user": IntentBucket(actor="user", utterances=["thanks bye", "goodbye"], label="goodbye_user"),
        "goodbye_agent": IntentBucket(actor="agent", utterances=["goodbye", "have a nice day"], label="goodbye_agent"),
    }


def build_linear_flow(buckets):
    """root -> greet_user -> greet_agent -> ask_hotel -> confirm"""
    flow = DialogueFlow()
    for nid in ["greet_user", "greet_agent", "ask_hotel", "confirm"]:
        flow.add_node(nid, buckets[nid])
    flow.add_edge(flow.root, "greet_user")
    flow.add_edge("greet_user", "greet_agent")
    flow.add_edge("greet_agent", "ask_hotel")
    flow.add_edge("ask_hotel", "confirm")
    return flow


def build_branching_flow(buckets):
    """root -> greet_user -> greet_agent -> {ask_hotel, ask_food} -> confirm"""
    flow = DialogueFlow()
    for nid in ["greet_user", "greet_agent", "ask_hotel", "ask_food", "confirm"]:
        flow.add_node(nid, buckets[nid])
    flow.add_edge(flow.root, "greet_user")
    flow.add_edge("greet_user", "greet_agent")
    flow.add_edge("greet_agent", "ask_hotel")
    flow.add_edge("greet_agent", "ask_food")
    flow.add_edge("ask_hotel", "confirm")
    flow.add_edge("ask_food", "confirm")
    return flow


def build_diamond_flow(buckets):
    """root -> {greet_user, ask_hotel} -> confirm"""
    flow = DialogueFlow()
    for nid in ["greet_user", "ask_hotel", "confirm"]:
        flow.add_node(nid, buckets[nid])
    flow.add_edge(flow.root, "greet_user")
    flow.add_edge(flow.root, "ask_hotel")
    flow.add_edge("greet_user", "confirm")
    flow.add_edge("ask_hotel", "confirm")
    return flow


def build_deep_flow(buckets):
    """root -> greet_user -> greet_agent -> ask_hotel -> ask_details -> provide_dates -> confirm -> goodbye_agent"""
    flow = DialogueFlow()
    chain = ["greet_user", "greet_agent", "ask_hotel", "ask_details", "provide_dates", "confirm", "goodbye_agent"]
    for nid in chain:
        flow.add_node(nid, buckets[nid])
    flow.add_edge(flow.root, chain[0])
    for i in range(len(chain) - 1):
        flow.add_edge(chain[i], chain[i + 1])
    return flow


def build_nested_diamond_flow(buckets):
    """Two serial diamonds with reconvergent merges — the shape fudge_dag exists
    for (route count doubles per diamond; the DFS scorer re-expands at merges).

    root -> greet_user -> {greet_agent, ask_details} -> ask_hotel (merge 1)
         -> {confirm, goodbye_agent} -> goodbye_user (merge 2)
    """
    flow = DialogueFlow()
    for nid in ["greet_user", "greet_agent", "ask_details", "ask_hotel",
                "confirm", "goodbye_agent", "goodbye_user"]:
        flow.add_node(nid, buckets[nid])
    flow.add_edge(flow.root, "greet_user")
    flow.add_edge("greet_user", "greet_agent")
    flow.add_edge("greet_user", "ask_details")
    flow.add_edge("greet_agent", "ask_hotel")
    flow.add_edge("ask_details", "ask_hotel")
    flow.add_edge("ask_hotel", "confirm")
    flow.add_edge("ask_hotel", "goodbye_agent")
    flow.add_edge("confirm", "goodbye_user")
    flow.add_edge("goodbye_agent", "goodbye_user")
    return flow


def build_wide_branching_flow(buckets):
    """root -> greet_user -> {greet_agent, ask_details}, greet_agent -> {ask_hotel, ask_food}, ask_hotel -> confirm, ask_food -> confirm, ask_details -> provide_dates"""
    flow = DialogueFlow()
    for nid in ["greet_user", "greet_agent", "ask_details", "ask_hotel", "ask_food", "confirm", "provide_dates"]:
        flow.add_node(nid, buckets[nid])
    flow.add_edge(flow.root, "greet_user")
    flow.add_edge("greet_user", "greet_agent")
    flow.add_edge("greet_user", "ask_details")
    flow.add_edge("greet_agent", "ask_hotel")
    flow.add_edge("greet_agent", "ask_food")
    flow.add_edge("ask_hotel", "confirm")
    flow.add_edge("ask_food", "confirm")
    flow.add_edge("ask_details", "provide_dates")
    return flow


# --- Conversations ---

CONVS = [
    Conversation(utterances=[
        Utterance("user", "hi"),
        Utterance("agent", "hello how can I help you"),
        Utterance("user", "I need to book a hotel room"),
        Utterance("agent", "confirmed"),
    ], task="hotel_book"),
    Conversation(utterances=[
        Utterance("user", "hey there"),
        Utterance("agent", "what can I do for you"),
        Utterance("user", "I want to order a pizza"),
        Utterance("agent", "all set"),
    ], task="food_order"),
    Conversation(utterances=[
        Utterance("user", "good morning"),
        Utterance("agent", "hi what do you need"),
        Utterance("user", "I want a hotel"),
        Utterance("agent", "what dates"),
        Utterance("user", "next monday"),
        Utterance("agent", "done"),
        Utterance("agent", "have a nice day"),
    ], task="hotel_book_long"),
    Conversation(utterances=[
        Utterance("user", "hello"),
    ], task="short"),
    Conversation(utterances=[], task="empty"),
]


FLOW_BUILDERS = [
    build_linear_flow,
    build_branching_flow,
    build_diamond_flow,
    build_nested_diamond_flow,
    build_deep_flow,
    build_wide_branching_flow,
]


@pytest.mark.parametrize("flow_builder", FLOW_BUILDERS,
                         ids=["linear", "branching", "diamond", "nested_diamond",
                              "deep", "wide_branching"])
@pytest.mark.parametrize("conv_idx", range(len(CONVS)),
                         ids=["hotel", "food", "hotel_long", "short", "empty"])
def test_all_scorers_agree(emb, flow_builder, conv_idx):
    """All three algorithms MUST return the same distance on the same inputs."""
    buckets = make_buckets()
    flow = flow_builder(buckets)
    conv = CONVS[conv_idx]
    all_buckets = list(buckets.values())
    costs = FudgeCosts(emb, all_buckets)

    naive_score = fudge_naive(conv, flow, costs)
    efficient_score = fudge_efficient(conv, flow, costs)
    dag_score = fudge_dag(conv, flow, costs)

    assert abs(naive_score - efficient_score) < 1e-6, \
        f"Naive ({naive_score:.6f}) != Efficient ({efficient_score:.6f})"
    assert abs(naive_score - dag_score) < 1e-6, \
        f"Naive ({naive_score:.6f}) != DAG ({dag_score:.6f})"


# --- Additional sanity tests ---

def test_empty_path(emb):
    buckets = make_buckets()
    costs = FudgeCosts(emb, list(buckets.values()))
    conv = Conversation(utterances=[Utterance("user", "hello"), Utterance("agent", "hi")])
    dist = path_edit_distance(conv, [], costs)
    assert dist == 2.0


def test_empty_conversation(emb):
    buckets = make_buckets()
    costs = FudgeCosts(emb, list(buckets.values()))
    conv = Conversation(utterances=[])
    path = [buckets["greet_user"], buckets["greet_agent"]]
    dist = path_edit_distance(conv, path, costs)
    assert dist == 2.0


def test_good_match_scores_lower_than_bad(emb):
    """Hotel conversation vs hotel flow should score lower than food conversation vs hotel flow."""
    buckets = make_buckets()
    flow = build_linear_flow(buckets)
    costs = FudgeCosts(emb, list(buckets.values()))

    hotel_conv = CONVS[0]  # hotel conversation
    food_conv = CONVS[1]   # food conversation

    hotel_score = fudge_naive(hotel_conv, flow, costs)
    food_score = fudge_naive(food_conv, flow, costs)

    assert hotel_score < food_score, \
        f"Hotel ({hotel_score:.4f}) should be < Food ({food_score:.4f}) on hotel flow"
