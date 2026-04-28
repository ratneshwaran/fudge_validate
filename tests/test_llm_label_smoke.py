"""Smoke test for scripts/llm_label_star.py.

Mocks openai.AsyncOpenAI so no real API calls are made. Uses the real
EmbeddingCache (consistent with the rest of tests/) for clustering and the
embedding-based off-taxonomy fallback.

Verifies:
  - taxonomy.json is written with {user, agent} structure of {label, description}
  - per-dialogue label files have one label per utterance
  - rerunning the pipeline with the same inputs causes zero additional API calls
    (the disk cache short-circuits everything)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make src/ importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from fudge.embeddings import EmbeddingCache  # noqa: E402
from fudge.types import Conversation, Utterance  # noqa: E402

import llm_label_star as M  # noqa: E402


# ---------------------------------------------------------------------------
# Fake OpenAI client
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt=120, completion=30)


class CallCounter:
    def __init__(self) -> None:
        self.n = 0


def _build_fake_openai(counter: CallCounter):
    async def fake_create(**kwargs):
        counter.n += 1
        rf = kwargs["response_format"]["json_schema"]
        name = rf["name"]
        schema = rf["schema"]

        if name == "cluster_label":
            # Differentiate user vs agent by inspecting the user message.
            msg = kwargs["messages"][-1]["content"]
            actor = "user" if "by the user" in msg else "agent"
            # Make label unique per call so cluster-merge doesn't collapse
            # everything into one entry.
            content = json.dumps(
                {
                    "label": f"{actor}_intent_{counter.n}",
                    "description": f"Auto-generated {actor} intent {counter.n}.",
                }
            )
        elif name == "dialogue_labels":
            enum = schema["properties"]["labels"]["items"]["properties"]["label"]["enum"]
            n = schema["properties"]["labels"]["minItems"]
            content = json.dumps(
                {"labels": [{"index": i, "label": enum[0]} for i in range(n)]}
            )
        elif name == "utterance_label":
            enum = schema["properties"]["label"]["enum"]
            content = json.dumps({"label": enum[0]})
        else:
            raise ValueError(f"unknown schema name in fake: {name}")
        return _FakeResponse(content)

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            chat = MagicMock()
            chat.completions = MagicMock()
            chat.completions.create = fake_create
            self.chat = chat

        async def close(self):
            pass

    return FakeAsyncOpenAI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder() -> EmbeddingCache:
    return EmbeddingCache()


def _make_conv(dialogue_id: int, turns: list[tuple[str, str]]) -> Conversation:
    conv = Conversation(utterances=[Utterance(actor=a, text=t) for a, t in turns])
    conv.dialogue_id = dialogue_id
    conv.task = "smoke_task"
    return conv


@pytest.fixture
def conversations() -> list[Conversation]:
    return [
        _make_conv(101, [
            ("user", "hi I want to book a hotel"),
            ("agent", "could I get your name please"),
            ("user", "my name is John"),
            ("agent", "what dates would you like"),
            ("user", "next monday to friday"),
            ("agent", "your booking is confirmed"),
        ]),
        _make_conv(102, [
            ("user", "hello I need a hotel room"),
            ("agent", "may I have your name"),
            ("user", "I'm Sarah"),
            ("agent", "when do you need it"),
            ("user", "this weekend"),
            ("agent", "all set you are booked"),
        ]),
    ]


def _make_config(method: str, tmp_path: Path) -> M.PipelineConfig:
    return M.PipelineConfig(
        task="smoke_task",
        method=method,
        model="gpt-5-mini",
        window_size=5,
        concurrency=4,
        cluster_algo="agglo",
        cluster_threshold=0.55,
        cluster_k=None,
        n_samples_per_cluster=3,
        merge_threshold=0.85,
        skip_taxonomy=False,
        limit=None,
        dry_run=False,
        star_dir=tmp_path / "star",
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "logs" / "smoke.jsonl",
    )


def _run(cfg, convs, embedder, counter, monkeypatch):
    """Run the pipeline once with mocked OpenAI; return the summary."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cls = _build_fake_openai(counter)

    # The script does `from openai import AsyncOpenAI` inside __aenter__.
    # Patch the symbol on the openai package so the late import resolves to ours.
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", fake_cls, raising=False)

    cache = M.LLMCache(cfg.cache_dir)
    logger = M.CallLogger(cfg.log_path)

    async def go():
        async with M.LLMClient(cfg.model, cache, logger, cfg.dry_run, cfg.concurrency) as client:
            return await M.run_pipeline(cfg, convs, embedder, client)

    return asyncio.run(go())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_taxonomy_file_structure(tmp_path, embedder, conversations, monkeypatch):
    cfg = _make_config("whole", tmp_path)
    counter = CallCounter()
    summary = _run(cfg, conversations, embedder, counter, monkeypatch)

    taxonomy_path = cfg.out_dir / "smoke_task" / "taxonomy.json"
    assert taxonomy_path.exists(), "taxonomy.json was not written"
    with open(taxonomy_path, encoding="utf-8") as f:
        tax = json.load(f)

    assert set(tax.keys()) == {"user", "agent"}
    assert isinstance(tax["user"], list) and len(tax["user"]) >= 1
    assert isinstance(tax["agent"], list) and len(tax["agent"]) >= 1
    for side in ("user", "agent"):
        for item in tax[side]:
            assert set(item.keys()) == {"label", "description"}
            assert isinstance(item["label"], str) and item["label"]
            assert isinstance(item["description"], str) and item["description"]

    assert summary["n_user_labels"] == len(tax["user"])
    assert summary["n_agent_labels"] == len(tax["agent"])


def test_per_dialogue_files_one_label_per_utterance(tmp_path, embedder, conversations, monkeypatch):
    cfg = _make_config("whole", tmp_path)
    _run(cfg, conversations, embedder, CallCounter(), monkeypatch)

    method_dir = cfg.out_dir / "smoke_task" / "whole"
    assert method_dir.exists()

    for conv in conversations:
        label_path = method_dir / f"{conv.dialogue_id}.json"
        assert label_path.exists(), f"missing label file for dialogue {conv.dialogue_id}"
        with open(label_path, encoding="utf-8") as f:
            payload = json.load(f)

        assert "utterance_labels" in payload
        assert "taxonomy_version" in payload
        assert len(payload["utterance_labels"]) == len(conv.utterances)
        for lbl in payload["utterance_labels"]:
            assert isinstance(lbl, str) and lbl


def test_per_dialogue_files_one_label_per_utterance_window(tmp_path, embedder, conversations, monkeypatch):
    cfg = _make_config("window", tmp_path)
    _run(cfg, conversations, embedder, CallCounter(), monkeypatch)

    method_dir = cfg.out_dir / "smoke_task" / "window"
    for conv in conversations:
        with open(method_dir / f"{conv.dialogue_id}.json", encoding="utf-8") as f:
            payload = json.load(f)
        assert len(payload["utterance_labels"]) == len(conv.utterances)


def test_cache_prevents_second_api_call(tmp_path, embedder, conversations, monkeypatch):
    cfg = _make_config("whole", tmp_path)

    counter1 = CallCounter()
    _run(cfg, conversations, embedder, counter1, monkeypatch)
    first_run_calls = counter1.n
    assert first_run_calls > 0, "fake client was never invoked on the first run"

    # Second run: same inputs, same cache_dir. Every prompt should be a hit.
    counter2 = CallCounter()
    _run(cfg, conversations, embedder, counter2, monkeypatch)
    assert counter2.n == 0, (
        f"expected zero API calls on the cached rerun, got {counter2.n} "
        f"(first run made {first_run_calls})"
    )
