"""Pure-logic tests for scripts/llm_judge.py (no network, no embeddings)."""
import asyncio
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import llm_judge as judge  # noqa: E402
from fudge.types import Conversation, Utterance  # noqa: E402


class _StubClient:
    """Stands in for JudgeClient; returns canned verdicts."""

    def __init__(self, verdicts):
        self._verdicts = verdicts

    async def call(self, stage, messages, schema_name, schema):
        return {"verdicts": self._verdicts}


RULES = [{"id": "r1", "kind": "presence", "text": "t1"},
         {"id": "r2", "kind": "transition", "text": "t2"}]
CONV = Conversation([Utterance("agent", "hello"), Utterance("user", "hi")])


def _judge(verdicts):
    return asyncio.run(judge.judge_conversation(_StubClient(verdicts), CONV, RULES))


def test_verdicts_aligned_to_rule_order():
    got = _judge([{"rule_id": "r2", "score": -1, "justification": "x"},
                  {"rule_id": "r1", "score": 1, "justification": "y"}])
    assert [v["rule_id"] for v in got] == ["r1", "r2"]
    assert [v["score"] for v in got] == [1, -1]


def test_omitted_rule_becomes_na():
    # judge returned the right count but graded a bogus id instead of r2
    got = _judge([{"rule_id": "r1", "score": 1, "justification": "y"},
                  {"rule_id": "bogus", "score": -1, "justification": "z"}])
    assert [v["rule_id"] for v in got] == ["r1", "r2"]
    assert got[1]["score"] == 0  # omitted rule -> N/A, not silently mis-mapped


def test_duplicate_rule_id_raises():
    with pytest.raises(ValueError, match="duplicate"):
        _judge([{"rule_id": "r1", "score": 1, "justification": "a"},
                {"rule_id": "r1", "score": -1, "justification": "b"}])


def test_score_session_excludes_na():
    verdicts = [{"score": 1}, {"score": 0}, {"score": -1}, {"score": 1}]
    assert judge.score_session(verdicts) == pytest.approx(1 / 3)


def test_circularity_guard_compares_slugs():
    judge_slug = judge.JUDGE_REGISTRY["claude-sonnet"]["slug"]
    # same underlying model registered as a generator under a DIFFERENT name
    clashing = {"my-gen": {"slug": judge_slug, "base_url": "x", "api_key_env": "K"}}
    with pytest.raises(SystemExit, match="Circularity"):
        judge.assert_not_circular("claude-sonnet", clashing)
    # a genuinely different model passes
    judge.assert_not_circular(
        "claude-sonnet",
        {"gen": {"slug": "other/model", "base_url": "x", "api_key_env": "K"}})


def test_unknown_judge_model_rejected():
    with pytest.raises(SystemExit, match="unknown judge model"):
        judge.assert_not_circular("nope", {})
