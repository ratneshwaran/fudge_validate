from fudge.types import IntentBucket, DialogueFlow


def test_simple_flow_paths():
    """root -> A -> B, root -> A -> C. Paths: [[A,B], [A,C]]."""
    flow = DialogueFlow()
    flow.add_node("A", IntentBucket(actor="user", utterances=["hi"], label="greet"))
    flow.add_node("B", IntentBucket(actor="agent", utterances=["hello"], label="respond"))
    flow.add_node("C", IntentBucket(actor="agent", utterances=["bye"], label="farewell"))
    flow.add_edge(flow.root, "A")
    flow.add_edge("A", "B")
    flow.add_edge("A", "C")

    leaves = sorted(flow.get_leaf_nodes())
    assert leaves == ["B", "C"]

    paths = flow.get_all_paths()
    paths_sorted = sorted([sorted(p) for p in paths])
    assert paths_sorted == [["A", "B"], ["A", "C"]]
    # Also check ordering within paths (A always first)
    for p in paths:
        assert p[0] == "A"

    assert flow.num_nodes == 3


def test_linear_flow():
    """root -> A -> B -> C. Single path [A, B, C]."""
    flow = DialogueFlow()
    flow.add_node("A", IntentBucket(actor="user", utterances=["a"], label="a"))
    flow.add_node("B", IntentBucket(actor="agent", utterances=["b"], label="b"))
    flow.add_node("C", IntentBucket(actor="user", utterances=["c"], label="c"))
    flow.add_edge(flow.root, "A")
    flow.add_edge("A", "B")
    flow.add_edge("B", "C")

    assert flow.get_leaf_nodes() == ["C"]
    paths = flow.get_all_paths()
    assert len(paths) == 1
    assert paths[0] == ["A", "B", "C"]


def test_diamond_flow():
    """root -> A, root -> B, A -> C, B -> C. Paths: [[A,C], [B,C]]."""
    flow = DialogueFlow()
    flow.add_node("A", IntentBucket(actor="user", utterances=["a"], label="a"))
    flow.add_node("B", IntentBucket(actor="user", utterances=["b"], label="b"))
    flow.add_node("C", IntentBucket(actor="agent", utterances=["c"], label="c"))
    flow.add_edge(flow.root, "A")
    flow.add_edge(flow.root, "B")
    flow.add_edge("A", "C")
    flow.add_edge("B", "C")

    assert flow.get_leaf_nodes() == ["C"]
    paths = flow.get_all_paths()
    assert len(paths) == 2
    path_sets = [tuple(p) for p in paths]
    assert ("A", "C") in path_sets
    assert ("B", "C") in path_sets


def test_empty_flow():
    """Just root, no other nodes."""
    flow = DialogueFlow()
    assert flow.num_nodes == 0
    assert flow.get_leaf_nodes() == []
    assert flow.get_all_paths() == []


def test_get_bucket():
    flow = DialogueFlow()
    bucket = IntentBucket(actor="user", utterances=["hello"], label="greet")
    flow.add_node("A", bucket)
    assert flow.get_bucket("A") is bucket


def test_get_children():
    flow = DialogueFlow()
    flow.add_node("A", IntentBucket(actor="user", utterances=["a"], label="a"))
    flow.add_node("B", IntentBucket(actor="agent", utterances=["b"], label="b"))
    flow.add_node("C", IntentBucket(actor="agent", utterances=["c"], label="c"))
    flow.add_edge(flow.root, "A")
    flow.add_edge("A", "B")
    flow.add_edge("A", "C")
    assert sorted(flow.get_children("A")) == ["B", "C"]
    assert flow.get_children("B") == []
