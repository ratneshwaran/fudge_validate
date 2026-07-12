"""Rule-2 guard: build_flow_from_llm_dag(label_fallback=False) drops nodes that
win no training utterance and rewires parents->children, instead of seeding the
bucket with the node LABEL string (which would make FuDGE score utterances
against a label-string embedding). Uses a fake embedding cache so the test is
fast and deterministic — build_flow only ever calls encode_batch.
"""
import numpy as np

from fudge.types import Conversation, Utterance
from fudge.llm_dag import build_flow_from_llm_dag


class FakeEmb:
    """encode_batch maps each known text to a fixed unit vector."""

    def __init__(self, vecs):
        self._v = vecs

    def encode_batch(self, texts):
        return np.array([self._v[t] for t in texts], dtype=float)


# Three agent nodes A -> B -> C. Every training utterance lands on A or C;
# B's label anchor is orthogonal to all of them, so B wins nothing.
_E = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0],
      "ua": [1.0, 0.0, 0.0], "uc": [0.0, 0.0, 1.0]}
_DAG = {"nodes": [{"id": "A", "actor": "agent", "label": "A"},
                  {"id": "B", "actor": "agent", "label": "B"},
                  {"id": "C", "actor": "agent", "label": "C"}],
        "edges": [{"from": "A", "to": "B"}, {"from": "B", "to": "C"}]}
_TRAIN = [Conversation(utterances=[Utterance("agent", "ua"), Utterance("agent", "uc")])]


def test_label_fallback_default_keeps_empty_node_as_label_bucket():
    flow, buckets, stats = build_flow_from_llm_dag(_DAG, _TRAIN, FakeEmb(_E))
    assert stats["label_fallback"] is True
    assert stats["n_nodes"] == 3
    assert stats["n_empty_buckets"] == 1
    assert stats["n_empty_dropped"] == 0
    by_label = {b.label: b for b in buckets}
    # the empty node keeps a bucket whose only "utterance" is its label string
    assert by_label["B"].utterances == ["B"]
    assert set(flow.graph.nodes) == {flow.root, "A", "B", "C"}


def test_drop_empty_nodes_rewires_parents_to_children():
    flow, buckets, stats = build_flow_from_llm_dag(
        _DAG, _TRAIN, FakeEmb(_E), label_fallback=False)
    assert stats["label_fallback"] is False
    assert stats["n_nodes"] == 2
    assert stats["n_empty_dropped"] == 1
    assert stats["n_empty_buckets"] == 0
    assert {b.label for b in buckets} == {"A", "C"}
    # no surviving bucket is a label-string fallback
    assert all(not (len(b.utterances) == 1 and b.utterances[0] == b.label)
               for b in buckets)
    # B excised; A now feeds C directly and A is root-wired (reachability kept)
    assert "B" not in flow.graph.nodes
    assert flow.graph.has_edge("A", "C")
    assert flow.graph.has_edge(flow.root, "A")
