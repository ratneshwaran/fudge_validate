"""Unit tests for the label_source path in build_flow_from_conversations.

Covers:
  - load_llm_labels reads <dialogue_id>.json files and skips non-numeric stems
  - build_flow_from_conversations replaces _intent_sequence labels when a
    label_source is provided (actor + text preserved, label swapped)
  - mismatched label/utterance counts raise loudly
  - dialogues without a matching label_source entry are silently skipped
"""
import json

import pytest

from fudge.data_loader import build_flow_from_conversations, load_llm_labels
from fudge.types import Conversation, Utterance


def _make_conv(dialogue_id, utterances):
    conv = Conversation(
        utterances=[Utterance(actor=a, text=t) for a, t in utterances],
        task="t",
        dialogue_id=dialogue_id,
    )
    conv._intent_sequence = [
        (f"heuristic_label_{i}", a, t) for i, (a, t) in enumerate(utterances)
    ]
    return conv


def test_load_llm_labels_reads_directory(tmp_path):
    """Legacy layout: no sibling taxonomy.json -> version checking skipped."""
    (tmp_path / "1.json").write_text(json.dumps({"utterance_labels": ["a", "b"]}))
    (tmp_path / "42.json").write_text(json.dumps({"utterance_labels": ["c"]}))

    out = load_llm_labels(tmp_path)
    assert out == {1: ["a", "b"], 42: ["c"]}


def _hash_taxonomy(taxonomy: dict) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(taxonomy, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def test_load_llm_labels_validates_taxonomy_version(tmp_path):
    """Standard layout: <tax_dir>/taxonomy.json + <tax_dir>/<lab>/<id>.json.
    Files matching the active taxonomy load; mismatches raise."""
    tax_dir = tmp_path / "single_prompt"
    label_dir = tax_dir / "whole"
    label_dir.mkdir(parents=True)

    taxonomy = {"user": [{"label": "u_a", "description": "x"}],
                "agent": [{"label": "a_a", "description": "y"}]}
    (tax_dir / "taxonomy.json").write_text(json.dumps(taxonomy))
    version = _hash_taxonomy(taxonomy)

    (label_dir / "1.json").write_text(
        json.dumps({"utterance_labels": ["u_a", "a_a"], "taxonomy_version": version})
    )
    (label_dir / "2.json").write_text(
        json.dumps({"utterance_labels": ["u_a"], "taxonomy_version": version})
    )

    out = load_llm_labels(label_dir)
    assert out == {1: ["u_a", "a_a"], 2: ["u_a"]}


def test_load_llm_labels_raises_on_stale_version(tmp_path):
    tax_dir = tmp_path / "single_prompt"
    label_dir = tax_dir / "whole"
    label_dir.mkdir(parents=True)

    taxonomy = {"user": [{"label": "u_a", "description": "x"}],
                "agent": [{"label": "a_a", "description": "y"}]}
    (tax_dir / "taxonomy.json").write_text(json.dumps(taxonomy))
    fresh_version = _hash_taxonomy(taxonomy)

    # One up-to-date file, one with a stale version.
    (label_dir / "1.json").write_text(
        json.dumps({"utterance_labels": ["u_a"], "taxonomy_version": fresh_version})
    )
    (label_dir / "2.json").write_text(
        json.dumps({"utterance_labels": ["u_a"], "taxonomy_version": "stale_hash_xx"})
    )

    with pytest.raises(ValueError, match="Stale taxonomy_version"):
        load_llm_labels(label_dir)


def test_label_source_replaces_labels():
    convs = [
        _make_conv(1, [("user", "hi"), ("agent", "hello")]),
        _make_conv(2, [("user", "hi"), ("agent", "hello")]),
    ]
    label_source = {
        1: ["llm_user_greet", "llm_agent_greet"],
        2: ["llm_user_greet", "llm_agent_greet"],
    }

    flow, buckets = build_flow_from_conversations(convs, label_source=label_source)

    bucket_keys = sorted({(b.actor, b.label) for b in buckets})
    assert bucket_keys == [("agent", "llm_agent_greet"), ("user", "llm_user_greet")]
    # The trie collapsed both conversations into one path because they had
    # identical (actor, label) sequences after substitution.
    assert flow.num_nodes == 2


def test_label_source_count_mismatch_raises():
    convs = [_make_conv(1, [("user", "hi"), ("agent", "hello")])]
    with pytest.raises(ValueError, match="Label count mismatch"):
        build_flow_from_conversations(convs, label_source={1: ["only_one"]})


def test_label_source_skips_unmapped_dialogues():
    convs = [
        _make_conv(1, [("user", "hi"), ("agent", "hello")]),
        _make_conv(2, [("user", "hi"), ("agent", "hello")]),
    ]
    flow, buckets = build_flow_from_conversations(
        convs, label_source={1: ["a", "b"]}
    )
    # Only conv #1 contributed to the flow.
    assert flow.num_nodes == 2
    assert {b.label for b in buckets} == {"a", "b"}


def test_no_label_source_uses_existing_intent_sequence():
    convs = [_make_conv(1, [("user", "hi"), ("agent", "hello")])]
    flow, buckets = build_flow_from_conversations(convs)
    # Falls back to the heuristic labels stashed on _intent_sequence.
    assert {b.label for b in buckets} == {"heuristic_label_0", "heuristic_label_1"}
