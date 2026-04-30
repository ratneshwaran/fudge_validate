"""LLM-based intent labeling for STAR dialogues.

Two-stage closed-taxonomy pipeline:

  Stage 1 (taxonomy bootstrap):
    --taxonomy-method single_prompt (default):
      Send the LLM all unique utterances for one actor in a single call and
      ask for a unified taxonomy. Eliminates the synonym / near-duplicate
      problem the per-cluster naming approach has, since the model produces
      every label in one pass with full context.
    --taxonomy-method cluster:
      SBERT-cluster utterances, name each cluster individually, then merge
      synonyms post-hoc. Kept for the planned 3-method ablation.

  Stage 2 (per-utterance labeling):
    --method whole   : send full dialogue + both taxonomies, LLM returns one
                       label per utterance.
    --method window  : send a window centered on each target utterance + the
                       taxonomy for that actor; LLM returns one label.

Outputs are saved per dialogue at
  data/STAR_llm_labels/<task>/<method>/<dialogue_id>.json
as {"utterance_labels": [...], "taxonomy_version": "<sha>"}.

`build_flow_from_conversations(label_source=...)` consumes the files via
`fudge.data_loader.load_llm_labels`.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Block tensorflow as the rest of the codebase does (see conftest.py).
sys.modules["tensorflow"] = None

# Make src/ importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from fudge.data_loader import load_star_dialogues  # noqa: E402
from fudge.embeddings import EmbeddingCache  # noqa: E402
from fudge.types import Conversation, Utterance  # noqa: E402

# Per-million-token prices (USD). These are display-only — edit to match
# whatever prices apply at the time of running.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}

# If a method=whole prompt is estimated to exceed this many tokens, we fall
# back to the window method for that single dialogue and emit a warning.
WHOLE_METHOD_TOKEN_BUDGET = 60_000


# ---------------------------------------------------------------------------
# STAR loader (filtered by task)
# ---------------------------------------------------------------------------

def load_star_for_task(star_dir: str, task: str) -> list[Conversation]:
    """Load STAR dialogues filtered to one task.

    `dialogue_id` is now a real field on Conversation (set by
    data_loader.load_star_dialogues), so no extra walk is needed.
    """
    convs = load_star_dialogues(star_dir, filter_unlabeled=True)
    return [c for c in convs if c.task == task]


# ---------------------------------------------------------------------------
# On-disk LLM cache
# ---------------------------------------------------------------------------

class LLMCache:
    """Content-addressed cache of (request, response, parsed, usage) on disk."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(model: str, stage: str, payload: dict) -> str:
        blob = json.dumps(
            {"model": model, "stage": stage, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path_for(key)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def put(self, key: str, entry: dict) -> None:
        with open(self.path_for(key), "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Per-call JSONL logger
# ---------------------------------------------------------------------------

class CallLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ---------------------------------------------------------------------------
# Async OpenAI wrapper with cache + retry + cost tracking
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: dict) -> None:
        self.input_tokens += int(other.get("input_tokens", 0))
        self.output_tokens += int(other.get("output_tokens", 0))


class DryRunCacheMiss(Exception):
    pass


class LLMClient:
    """Async OpenAI client wrapper.

    Usage:
        client = LLMClient(model, cache, logger, dry_run=False, concurrency=10)
        async with client:
            parsed = await client.call(stage, messages, schema_name, schema)
    """

    def __init__(
        self,
        model: str,
        cache: LLMCache,
        logger: CallLogger,
        dry_run: bool,
        concurrency: int,
    ):
        self.model = model
        self.cache = cache
        self.logger = logger
        self.dry_run = dry_run
        self.semaphore = asyncio.Semaphore(concurrency)
        self.usage = Usage()
        self.cache_hits = 0
        self.api_calls = 0
        self._async_client = None  # lazy

    async def __aenter__(self):
        if not self.dry_run:
            from openai import AsyncOpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Export it in your environment "
                    "before running (or use --dry-run for cache-only)."
                )
            self._async_client = AsyncOpenAI(api_key=api_key)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._async_client is not None:
            await self._async_client.close()

    async def call(
        self,
        stage: str,
        messages: list[dict],
        schema_name: str,
        schema: dict,
        max_retries: int = 5,
    ) -> dict:
        payload = {"messages": messages, "schema_name": schema_name, "schema": schema}
        key = LLMCache.make_key(self.model, stage, payload)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            await self.logger.write(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": stage,
                    "cache_hit": True,
                    "model": self.model,
                    "messages": messages,
                    "parsed": hit.get("parsed"),
                }
            )
            return hit["parsed"]

        if self.dry_run:
            raise DryRunCacheMiss(
                f"--dry-run set but no cache entry for stage={stage} key={key[:12]}…"
            )

        # API call with exponential backoff.
        async with self.semaphore:
            delay = 1.0
            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    response = await self._async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "strict": True,
                                "schema": schema,
                            },
                        },
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    cls = type(e).__name__
                    if attempt == max_retries - 1:
                        raise
                    # Retry on rate limit / connection / 5xx
                    if any(s in cls for s in ("RateLimit", "APIConnection", "APITimeout", "InternalServerError", "APIError")):
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30.0)
                        continue
                    raise
            else:  # pragma: no cover - loop always breaks or raises
                raise last_error  # type: ignore[misc]

        raw_content = response.choices[0].message.content
        parsed = json.loads(raw_content)
        usage = response.usage
        usage_dict = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        self.usage.add(usage_dict)
        self.api_calls += 1

        entry = {
            "request": {"model": self.model, "messages": messages, "schema_name": schema_name},
            "raw_content": raw_content,
            "parsed": parsed,
            "usage": usage_dict,
        }
        self.cache.put(key, entry)

        await self.logger.write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "cache_hit": False,
                "model": self.model,
                "messages": messages,
                "raw_content": raw_content,
                "parsed": parsed,
                "usage": usage_dict,
            }
        )
        return parsed

    def estimated_cost_usd(self) -> float:
        rates = MODEL_PRICES.get(self.model)
        if rates is None:
            return 0.0
        in_rate, out_rate = rates
        return (
            self.usage.input_tokens * in_rate / 1_000_000.0
            + self.usage.output_tokens * out_rate / 1_000_000.0
        )


# ---------------------------------------------------------------------------
# Stage 1: clustering
# ---------------------------------------------------------------------------

def cluster_texts(
    texts: list[str],
    embedder: EmbeddingCache,
    algo: str,
    threshold: float,
    k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cluster_labels, embeddings).

    algo: 'agglo' | 'kmeans'
    For 'agglo', uses cosine + average linkage with distance_threshold = 1 - threshold.
    For 'kmeans', if k is None, picks k via silhouette over [3..min(20, n-1)].
    """
    if not texts:
        return np.array([], dtype=int), np.zeros((0, 0))

    embeddings = embedder.encode_batch(texts)
    if len(texts) == 1:
        return np.array([0]), embeddings

    if algo == "agglo":
        from sklearn.cluster import AgglomerativeClustering

        # 'threshold' here is the cosine *similarity* cut (>= threshold => same cluster).
        # AgglomerativeClustering with metric='cosine' uses cosine *distance* (= 1 - sim).
        cosine_distance_threshold = 1.0 - threshold
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=cosine_distance_threshold,
        )
        labels = clusterer.fit_predict(embeddings)
        return labels, embeddings

    if algo == "kmeans":
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        if k is not None:
            km = KMeans(n_clusters=min(k, len(texts)), random_state=42, n_init="auto")
            return km.fit_predict(embeddings), embeddings

        upper = min(20, len(texts) - 1)
        if upper < 3:
            km = KMeans(n_clusters=max(2, upper), random_state=42, n_init="auto")
            return km.fit_predict(embeddings), embeddings

        best_k = 3
        best_score = -1.0
        best_labels: np.ndarray | None = None
        for kc in range(3, upper + 1):
            km = KMeans(n_clusters=kc, random_state=42, n_init="auto")
            labels_k = km.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels_k, metric="cosine")
            if score > best_score:
                best_k, best_score, best_labels = kc, score, labels_k
        return best_labels if best_labels is not None else np.zeros(len(texts), dtype=int), embeddings

    raise ValueError(f"Unknown clustering algo: {algo}")


def cluster_representatives(
    texts: list[str],
    cluster_labels: np.ndarray,
    embeddings: np.ndarray,
    n_per_cluster: int,
) -> dict[int, list[str]]:
    """For each cluster, return the n texts whose embeddings are closest to the centroid."""
    out: dict[int, list[str]] = {}
    unique_clusters = sorted(set(int(c) for c in cluster_labels.tolist()))
    for cid in unique_clusters:
        idxs = [i for i, c in enumerate(cluster_labels) if int(c) == cid]
        if not idxs:
            continue
        cluster_embs = embeddings[idxs]
        centroid = cluster_embs.mean(axis=0)
        norm = np.linalg.norm(centroid) or 1.0
        centroid = centroid / norm
        sims = cluster_embs @ centroid
        order = np.argsort(-sims)
        chosen = [texts[idxs[int(o)]] for o in order[:n_per_cluster]]
        out[cid] = chosen
    return out


# ---------------------------------------------------------------------------
# Stage 1: LLM-driven taxonomy bootstrap
# ---------------------------------------------------------------------------

TAXONOMY_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "description"],
    "properties": {
        "label": {"type": "string"},
        "description": {"type": "string"},
    },
}


def _name_cluster_messages(actor: str, sample_texts: list[str]) -> list[dict]:
    rendered = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(sample_texts))
    return [
        {
            "role": "system",
            "content": (
                "You assign a single shared intent label to a cluster of "
                "task-oriented dialogue utterances. Output strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"These {len(sample_texts)} utterances were all spoken by the "
                f"{actor}. Identify what the {actor} is doing in all of them.\n\n"
                f"UTTERANCES:\n{rendered}\n\n"
                "Return:\n"
                "  label       — short snake_case identifier, 1–4 words, no spaces. "
                "Examples: ask_name, provide_arrival_date, confirm_booking.\n"
                "  description — one sentence describing the intent."
            ),
        },
    ]


async def name_clusters(
    client: LLMClient,
    actor: str,
    representatives: dict[int, list[str]],
) -> dict[int, dict]:
    """Returns cluster_id -> {label, description}."""

    async def _name(cid: int, samples: list[str]) -> tuple[int, dict]:
        messages = _name_cluster_messages(actor, samples)
        parsed = await client.call(
            stage=f"taxonomy.name.{actor}",
            messages=messages,
            schema_name="cluster_label",
            schema=TAXONOMY_ITEM_SCHEMA,
        )
        return cid, parsed

    tasks = [_name(cid, samples) for cid, samples in representatives.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)


def merge_taxonomy(
    items: list[dict],
    embedder: EmbeddingCache,
    similarity_threshold: float,
) -> list[dict]:
    """Merge taxonomy entries whose label-embeddings are above `similarity_threshold`.

    Connected-components on the threshold graph; representative = first entry by
    insertion order.
    """
    if len(items) <= 1 or similarity_threshold >= 1.0:
        return items

    label_strings = [f"{it['label']}. {it['description']}" for it in items]
    embs = embedder.encode_batch(label_strings)
    sim = embs @ embs.T

    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= similarity_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[dict] = []
    used_labels: set[str] = set()
    for _, members in sorted(groups.items()):
        rep = items[members[0]]
        label = rep["label"]
        # Disambiguate label collisions across actors / merged pairs.
        base = label
        suffix = 2
        while label in used_labels:
            label = f"{base}_{suffix}"
            suffix += 1
        used_labels.add(label)
        merged.append({"label": label, "description": rep["description"]})
    return merged


async def bootstrap_taxonomy_cluster(
    actor: str,
    utterances: list[str],
    client: LLMClient,
    embedder: EmbeddingCache,
    cluster_algo: str,
    threshold: float,
    k: int | None,
    n_samples_per_cluster: int,
    merge_threshold: float,
) -> list[dict]:
    """Cluster-then-name taxonomy. Kept for the planned 3-method ablation."""
    if not utterances:
        return []
    cluster_labels, embeddings = cluster_texts(utterances, embedder, cluster_algo, threshold, k)
    reps = cluster_representatives(utterances, cluster_labels, embeddings, n_samples_per_cluster)
    named = await name_clusters(client, actor, reps)
    items = [named[cid] for cid in sorted(named.keys())]
    return merge_taxonomy(items, embedder, merge_threshold)


# ---------------------------------------------------------------------------
# Stage 1 (alt): single-prompt taxonomy bootstrap
# ---------------------------------------------------------------------------

SINGLE_PROMPT_TAXONOMY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["taxonomy"],
    "properties": {
        "taxonomy": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "description"],
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        }
    },
}


def _single_prompt_messages(
    actor: str,
    utterances: list[str],
    target_size_min: int,
    target_size_max: int,
) -> list[dict]:
    rendered = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(utterances))
    return [
        {
            "role": "system",
            "content": (
                "You design a closed intent taxonomy for one side of a "
                "task-oriented dialogue. Your taxonomy must cover every "
                "utterance you are shown. Output strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Below are {len(utterances)} utterances spoken by the "
                f"{actor} in a task-oriented dialogue. Produce a SINGLE "
                f"taxonomy of distinct intents that together cover all of "
                f"these utterances.\n\n"
                f"REQUIREMENTS:\n"
                f"- Aim for {target_size_min}–{target_size_max} intents. Use "
                f"the smallest taxonomy that still distinguishes meaningfully "
                f"different actions.\n"
                f"- Each intent has a snake_case label (1–4 words) and a one-"
                f"sentence description.\n"
                f"- Do NOT produce near-synonyms or two intents that differ "
                f"only by slot values (e.g. city, date). Generalize: "
                f"`provide_arrival_date`, not `provide_arrival_date_monday`.\n"
                f"- Labels must be unique within the taxonomy.\n\n"
                f"UTTERANCES:\n{rendered}"
            ),
        },
    ]


def _sample_for_single_prompt(
    utterances: list[str],
    max_chars: int,
) -> list[str]:
    """Truncate the utterance list to fit a rough char budget by uniform sampling.

    No sampling for short corpora. Stable: deterministic given input.
    """
    total = sum(len(t) + 8 for t in utterances)  # +8 for numbering/newlines
    if total <= max_chars:
        return utterances
    keep = max(50, int(len(utterances) * (max_chars / total)))
    if keep >= len(utterances):
        return utterances
    # Even-spaced sample preserves ordering and picks across the distribution.
    step = len(utterances) / keep
    indices = sorted({int(i * step) for i in range(keep)})
    return [utterances[i] for i in indices if i < len(utterances)]


def _dedup_taxonomy(items: list[dict]) -> list[dict]:
    """Drop entries whose label has already been seen (defensive — prompt requires uniqueness)."""
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        lbl = it["label"]
        if lbl in seen:
            continue
        seen.add(lbl)
        out.append({"label": lbl, "description": it["description"]})
    return out


async def bootstrap_taxonomy_single_prompt(
    actor: str,
    utterances: list[str],
    client: LLMClient,
    target_size_min: int,
    target_size_max: int,
    max_prompt_chars: int,
) -> list[dict]:
    """Send all unique utterances for an actor in one call; return unified taxonomy."""
    if not utterances:
        return []
    sampled = _sample_for_single_prompt(utterances, max_prompt_chars)
    messages = _single_prompt_messages(actor, sampled, target_size_min, target_size_max)
    parsed = await client.call(
        stage=f"taxonomy.single.{actor}",
        messages=messages,
        schema_name="unified_taxonomy",
        schema=SINGLE_PROMPT_TAXONOMY_SCHEMA,
    )
    return _dedup_taxonomy(parsed.get("taxonomy", []))


async def bootstrap_taxonomy_hybrid(
    actor: str,
    utterances: list[str],
    client: LLMClient,
    embedder: EmbeddingCache,
    cluster_algo: str,
    threshold: float,
    k: int | None,
    n_reps_per_cluster: int,
    target_size_min: int,
    target_size_max: int,
) -> list[dict]:
    """Cluster utterances cheaply, then send 1-2 representatives per cluster
    in ONE LLM call asking for a unified taxonomy.

    Long-tail coverage from clustering + no synonyms from single-call naming.
    """
    if not utterances:
        return []
    cluster_labels, embeddings = cluster_texts(utterances, embedder, cluster_algo, threshold, k)
    reps_dict = cluster_representatives(utterances, cluster_labels, embeddings, n_reps_per_cluster)
    # Flatten reps in deterministic cluster_id order so the cache key is stable.
    flat_reps = [t for cid in sorted(reps_dict.keys()) for t in reps_dict[cid]]
    messages = _single_prompt_messages(actor, flat_reps, target_size_min, target_size_max)
    parsed = await client.call(
        stage=f"taxonomy.hybrid.{actor}",
        messages=messages,
        schema_name="unified_taxonomy",
        schema=SINGLE_PROMPT_TAXONOMY_SCHEMA,
    )
    return _dedup_taxonomy(parsed.get("taxonomy", []))


# ---------------------------------------------------------------------------
# Off-taxonomy fallback
# ---------------------------------------------------------------------------

def nearest_taxonomy_label(
    returned: str,
    taxonomy: list[dict],
    embedder: EmbeddingCache,
) -> str:
    """Embed `returned` and pick the closest taxonomy label by cosine similarity."""
    if not taxonomy:
        raise ValueError("Cannot fall back: empty taxonomy")
    labels = [t["label"] for t in taxonomy]
    label_strings = [f"{t['label']}. {t['description']}" for t in taxonomy]
    label_embs = embedder.encode_batch(label_strings)
    q_emb = embedder.encode(returned)
    sims = label_embs @ q_emb
    return labels[int(np.argmax(sims))]


# ---------------------------------------------------------------------------
# Stage 2: per-utterance labeling
# ---------------------------------------------------------------------------

def render_taxonomy_block(name: str, items: list[dict]) -> str:
    if not items:
        return f"{name}: (empty)"
    lines = [f"  - {it['label']}: {it['description']}" for it in items]
    return f"{name}:\n" + "\n".join(lines)


def whole_method_schema(taxonomy_user: list[dict], taxonomy_agent: list[dict], n: int) -> dict:
    enum = sorted({t["label"] for t in taxonomy_user} | {t["label"] for t in taxonomy_agent})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {
            "labels": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "label"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 0, "maximum": max(0, n - 1)},
                        "label": {"type": "string", "enum": enum or ["__none__"]},
                    },
                },
            }
        },
    }


def window_method_schema(taxonomy: list[dict]) -> dict:
    enum = sorted({t["label"] for t in taxonomy})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label"],
        "properties": {
            "label": {"type": "string", "enum": enum or ["__none__"]},
        },
    }


def _render_dialogue(conv: Conversation) -> str:
    return "\n".join(
        f"  [{i}] ({u.actor}) {u.text}" for i, u in enumerate(conv.utterances)
    )


def _whole_method_messages(
    conv: Conversation,
    taxonomy_user: list[dict],
    taxonomy_agent: list[dict],
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You label every utterance in a task-oriented dialogue with an "
                "intent from a closed taxonomy. Each utterance must use a label "
                "from the taxonomy matching its actor. Output strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{render_taxonomy_block('USER TAXONOMY', taxonomy_user)}\n\n"
                f"{render_taxonomy_block('AGENT TAXONOMY', taxonomy_agent)}\n\n"
                "DIALOGUE:\n"
                f"{_render_dialogue(conv)}\n\n"
                f"Return one entry per utterance, indices 0..{len(conv.utterances) - 1}, "
                "in any order. Each label must come from the taxonomy for that "
                "utterance's actor."
            ),
        },
    ]


def _window_method_messages(
    conv: Conversation,
    target_idx: int,
    taxonomy: list[dict],
    window_size: int,
) -> list[dict]:
    half = window_size // 2
    start = max(0, target_idx - half)
    end = min(len(conv.utterances), start + window_size)
    start = max(0, end - window_size)  # keep a full window when possible
    actor = conv.utterances[target_idx].actor

    rendered_lines = []
    for i in range(start, end):
        marker = "  <-- TARGET" if i == target_idx else ""
        u = conv.utterances[i]
        rendered_lines.append(f"  [{i}] ({u.actor}) {u.text}{marker}")
    rendered = "\n".join(rendered_lines)

    return [
        {
            "role": "system",
            "content": (
                "You label a single target utterance in a task-oriented dialogue "
                "using a closed taxonomy for that actor. Output strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{render_taxonomy_block(f'TAXONOMY ({actor})', taxonomy)}\n\n"
                f"CONTEXT WINDOW:\n{rendered}\n\n"
                f"Return the label for the TARGET utterance (index {target_idx}, "
                f"actor {actor})."
            ),
        },
    ]


def _validate_label(
    returned_label: str,
    utterance: Utterance,
    taxonomy_for_actor: list[dict],
    embedder: EmbeddingCache,
    warnings: list[str],
    where: str,
) -> str:
    """Return `returned_label` if valid for the actor; otherwise embed the
    *utterance text* (not the bad label string) and pick the nearest taxonomy
    entry. Embedding the bad label can land on a semantically opposite intent
    that just happens to share tokens (e.g., model returns agent `ask_name`
    on a user turn, label-string fallback picks user `provide_name` — opposite
    intent). Embedding the utterance text matches by what the speaker
    actually did.
    """
    valid = {t["label"] for t in taxonomy_for_actor}
    if returned_label in valid:
        return returned_label
    fallback = nearest_taxonomy_label(utterance.text, taxonomy_for_actor, embedder)
    warnings.append(
        f"{where}: model returned '{returned_label}' (not in {utterance.actor} "
        f"taxonomy) -> utterance-text fallback '{fallback}'"
    )
    return fallback


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough char/4 estimate; fine for budget gating."""
    total = sum(len(m.get("content", "")) for m in messages)
    return total // 4


async def label_dialogue_whole(
    conv: Conversation,
    taxonomy_user: list[dict],
    taxonomy_agent: list[dict],
    client: LLMClient,
    embedder: EmbeddingCache,
) -> tuple[list[str], list[str]]:
    n = len(conv.utterances)
    messages = _whole_method_messages(conv, taxonomy_user, taxonomy_agent)
    schema = whole_method_schema(taxonomy_user, taxonomy_agent, n)
    parsed = await client.call(
        stage="label.whole",
        messages=messages,
        schema_name="dialogue_labels",
        schema=schema,
    )

    warnings: list[str] = []
    by_index: dict[int, str] = {}
    for entry in parsed.get("labels", []):
        i = int(entry["index"])
        if 0 <= i < n:
            by_index[i] = entry["label"]

    out_labels: list[str] = []
    for i, u in enumerate(conv.utterances):
        tax = taxonomy_user if u.actor == "user" else taxonomy_agent
        if i not in by_index:
            warnings.append(f"dialogue {getattr(conv, 'dialogue_id', '?')} idx {i}: missing in response")
            # Pick best taxonomy label for this utterance via embedding.
            out_labels.append(nearest_taxonomy_label(u.text, tax, embedder))
            continue
        out_labels.append(
            _validate_label(
                by_index[i], u, tax, embedder, warnings,
                where=f"dialogue {getattr(conv, 'dialogue_id', '?')} idx {i}",
            )
        )
    return out_labels, warnings


async def label_dialogue_window(
    conv: Conversation,
    taxonomy_user: list[dict],
    taxonomy_agent: list[dict],
    client: LLMClient,
    embedder: EmbeddingCache,
    window_size: int,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []

    async def _one(i: int) -> str:
        u = conv.utterances[i]
        tax = taxonomy_user if u.actor == "user" else taxonomy_agent
        if not tax:
            raise ValueError(f"Empty taxonomy for actor {u.actor}; cannot label")
        messages = _window_method_messages(conv, i, tax, window_size)
        schema = window_method_schema(tax)
        parsed = await client.call(
            stage="label.window",
            messages=messages,
            schema_name="utterance_label",
            schema=schema,
        )
        return _validate_label(
            parsed["label"], u, tax, embedder, warnings,
            where=f"dialogue {getattr(conv, 'dialogue_id', '?')} idx {i}",
        )

    labels = await asyncio.gather(*[_one(i) for i in range(len(conv.utterances))])
    return list(labels), warnings


def _chunk_starts(n: int, chunk_size: int, stride: int) -> list[int]:
    """Compute chunk start offsets covering all n utterances.

    Each chunk is `[start, start + chunk_size)` (clamped to n).
    Last chunk is anchored to `n - chunk_size` so it reaches the end.
    For overlap zones, later chunks have more right-context so we want
    them processed last (caller relies on iteration order).
    """
    if n <= 0:
        return []
    if n <= chunk_size:
        return [0]
    starts = list(range(0, n - chunk_size + 1, stride))
    if not starts or starts[-1] + chunk_size < n:
        starts.append(n - chunk_size)
    return sorted(set(starts))


async def label_dialogue_chunk(
    conv: Conversation,
    taxonomy_user: list[dict],
    taxonomy_agent: list[dict],
    client: LLMClient,
    embedder: EmbeddingCache,
    chunk_size: int,
    stride: int,
) -> tuple[list[str], list[str]]:
    """Label a dialogue in overlapping chunks of `chunk_size`, stride `stride`.

    Each chunk is sent to the LLM as a sub-dialogue (reusing the whole-method
    prompt + schema). For positions in overlap zones, the later chunk's label
    wins — it has more right-context for those utterances.
    """
    n = len(conv.utterances)
    if n == 0:
        return [], []
    starts = _chunk_starts(n, chunk_size, stride)
    out: list[str | None] = [None] * n
    warnings: list[str] = []

    for start in starts:
        end = min(start + chunk_size, n)
        sub_utts = list(conv.utterances[start:end])
        sub_conv = Conversation(
            utterances=sub_utts,
            task=conv.task,
            dialogue_id=conv.dialogue_id,
        )
        messages = _whole_method_messages(sub_conv, taxonomy_user, taxonomy_agent)
        schema = whole_method_schema(taxonomy_user, taxonomy_agent, len(sub_utts))
        parsed = await client.call(
            stage="label.chunk",
            messages=messages,
            schema_name="dialogue_labels",
            schema=schema,
        )

        by_index: dict[int, str] = {}
        for entry in parsed.get("labels", []):
            i = int(entry["index"])
            if 0 <= i < len(sub_utts):
                by_index[i] = entry["label"]

        for i, u in enumerate(sub_utts):
            tax = taxonomy_user if u.actor == "user" else taxonomy_agent
            abs_idx = start + i
            if i not in by_index:
                warnings.append(
                    f"dialogue {conv.dialogue_id} chunk@{start} idx {i} (abs {abs_idx}): missing in response"
                )
                out[abs_idx] = nearest_taxonomy_label(u.text, tax, embedder)
                continue
            out[abs_idx] = _validate_label(
                by_index[i], u, tax, embedder, warnings,
                where=f"dialogue {conv.dialogue_id} chunk@{start} idx {i} (abs {abs_idx})",
            )

    if any(o is None for o in out):
        missing = [i for i, o in enumerate(out) if o is None]
        raise RuntimeError(
            f"chunk method left positions unlabeled for dialogue "
            f"{conv.dialogue_id}: {missing}"
        )
    return [o for o in out if o is not None], warnings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    task: str
    method: str  # 'whole' | 'window' | 'chunk'
    model: str
    window_size: int
    chunk_size: int
    chunk_stride: int
    concurrency: int
    taxonomy_method: str  # 'single_prompt' | 'hybrid' | 'cluster'
    # Cluster-method knobs (used by 'cluster' and 'hybrid')
    cluster_algo: str
    cluster_threshold: float
    cluster_k: int | None
    n_samples_per_cluster: int
    merge_threshold: float
    # Hybrid-only
    hybrid_reps_per_cluster: int
    # Single-prompt and hybrid LLM-call knobs
    target_size_min: int
    target_size_max: int
    max_prompt_chars: int
    # Common
    skip_taxonomy: bool
    limit: int | None
    dry_run: bool
    star_dir: Path
    out_dir: Path
    cache_dir: Path
    log_path: Path


def collect_utterances_by_actor(convs: list[Conversation]) -> tuple[list[str], list[str]]:
    user_set: dict[str, None] = {}
    agent_set: dict[str, None] = {}
    for c in convs:
        for u in c.utterances:
            if u.actor == "user":
                user_set[u.text] = None
            else:
                agent_set[u.text] = None
    return list(user_set.keys()), list(agent_set.keys())


async def run_pipeline(
    cfg: PipelineConfig,
    convs: list[Conversation],
    embedder: EmbeddingCache,
    client: LLMClient,
) -> dict:
    """Returns a small summary dict for the smoke test / CLI."""
    # Layout: <out>/<task>/<taxonomy_method>/taxonomy.json
    #         <out>/<task>/<taxonomy_method>/<label_method>/<dialogue_id>.json
    out_dir = cfg.out_dir / cfg.task / cfg.taxonomy_method
    out_dir.mkdir(parents=True, exist_ok=True)
    method_dir = out_dir / cfg.method
    method_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_path = out_dir / "taxonomy.json"

    # Stage 1: taxonomy.
    if cfg.skip_taxonomy and taxonomy_path.exists():
        with open(taxonomy_path, encoding="utf-8") as f:
            taxonomy = json.load(f)
        print(f"[stage1] Reusing existing taxonomy: {taxonomy_path}")
    else:
        user_texts, agent_texts = collect_utterances_by_actor(convs)
        print(
            f"[stage1] Bootstrapping taxonomy ({cfg.taxonomy_method}) from "
            f"{len(user_texts)} unique user utterances and "
            f"{len(agent_texts)} unique agent utterances"
        )
        if cfg.taxonomy_method == "single_prompt":
            user_taxonomy, agent_taxonomy = await asyncio.gather(
                bootstrap_taxonomy_single_prompt(
                    "user", user_texts, client,
                    cfg.target_size_min, cfg.target_size_max, cfg.max_prompt_chars,
                ),
                bootstrap_taxonomy_single_prompt(
                    "agent", agent_texts, client,
                    cfg.target_size_min, cfg.target_size_max, cfg.max_prompt_chars,
                ),
            )
        elif cfg.taxonomy_method == "hybrid":
            user_taxonomy, agent_taxonomy = await asyncio.gather(
                bootstrap_taxonomy_hybrid(
                    "user", user_texts, client, embedder,
                    cfg.cluster_algo, cfg.cluster_threshold, cfg.cluster_k,
                    cfg.hybrid_reps_per_cluster,
                    cfg.target_size_min, cfg.target_size_max,
                ),
                bootstrap_taxonomy_hybrid(
                    "agent", agent_texts, client, embedder,
                    cfg.cluster_algo, cfg.cluster_threshold, cfg.cluster_k,
                    cfg.hybrid_reps_per_cluster,
                    cfg.target_size_min, cfg.target_size_max,
                ),
            )
        elif cfg.taxonomy_method == "cluster":
            user_taxonomy, agent_taxonomy = await asyncio.gather(
                bootstrap_taxonomy_cluster(
                    "user", user_texts, client, embedder,
                    cfg.cluster_algo, cfg.cluster_threshold, cfg.cluster_k,
                    cfg.n_samples_per_cluster, cfg.merge_threshold,
                ),
                bootstrap_taxonomy_cluster(
                    "agent", agent_texts, client, embedder,
                    cfg.cluster_algo, cfg.cluster_threshold, cfg.cluster_k,
                    cfg.n_samples_per_cluster, cfg.merge_threshold,
                ),
            )
        else:
            raise ValueError(f"Unknown --taxonomy-method: {cfg.taxonomy_method}")
        taxonomy = {"user": user_taxonomy, "agent": agent_taxonomy}
        with open(taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, ensure_ascii=False, indent=2)
        print(
            f"[stage1] Wrote {taxonomy_path} "
            f"({len(taxonomy['user'])} user + {len(taxonomy['agent'])} agent labels)"
        )

    taxonomy_blob = json.dumps(taxonomy, sort_keys=True, ensure_ascii=False)
    taxonomy_version = hashlib.sha256(taxonomy_blob.encode("utf-8")).hexdigest()[:16]

    # Stage 2: per-dialogue labeling.
    target_convs = convs[: cfg.limit] if cfg.limit else convs
    print(f"[stage2] Labeling {len(target_convs)} dialogues with method={cfg.method}")

    async def _label_one(conv: Conversation) -> tuple[int, list[str], list[str]]:
        if cfg.method == "whole":
            messages = _whole_method_messages(conv, taxonomy["user"], taxonomy["agent"])
            est = _estimate_tokens(messages)
            if est > WHOLE_METHOD_TOKEN_BUDGET:
                # Hard fail rather than silently switching methods. A silent
                # switch would write `window` results into the `whole/` output
                # tree and make ablations non-reproducible — the user should
                # explicitly choose --method chunk or --method window for
                # datasets with long dialogues.
                raise RuntimeError(
                    f"Dialogue {getattr(conv, 'dialogue_id', '?')} estimated "
                    f"at {est} tokens (> WHOLE_METHOD_TOKEN_BUDGET="
                    f"{WHOLE_METHOD_TOKEN_BUDGET}). The whole method dilutes "
                    "attention beyond this point. Re-run with --method chunk "
                    "or --method window, or raise WHOLE_METHOD_TOKEN_BUDGET "
                    "if you've verified the model handles it."
                )

        if cfg.method == "whole":
            labels, warnings = await label_dialogue_whole(
                conv, taxonomy["user"], taxonomy["agent"], client, embedder,
            )
        elif cfg.method == "chunk":
            labels, warnings = await label_dialogue_chunk(
                conv, taxonomy["user"], taxonomy["agent"], client, embedder,
                cfg.chunk_size, cfg.chunk_stride,
            )
        elif cfg.method == "window":
            labels, warnings = await label_dialogue_window(
                conv, taxonomy["user"], taxonomy["agent"], client, embedder, cfg.window_size,
            )
        else:
            raise ValueError(f"Unknown --method: {cfg.method}")

        out_path = method_dir / f"{conv.dialogue_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "utterance_labels": labels,
                    "taxonomy_version": taxonomy_version,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return conv.dialogue_id, labels, warnings

    results = await asyncio.gather(*[_label_one(c) for c in target_convs])

    total_warnings = sum(len(w) for _, _, w in results)
    if total_warnings:
        print(f"[stage2] {total_warnings} warnings (off-taxonomy / missing index). See log.")

    return {
        "taxonomy_path": taxonomy_path,
        "taxonomy_version": taxonomy_version,
        "n_user_labels": len(taxonomy["user"]),
        "n_agent_labels": len(taxonomy["agent"]),
        "n_dialogues_labeled": len(results),
        "warnings": total_warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="STAR task (e.g., hotel_book)")
    p.add_argument("--method", required=True, choices=["whole", "window", "chunk"])
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--window-size", type=int, default=7,
                   help="Window size for --method window (utterances around target)")
    p.add_argument("--chunk-size", type=int, default=5,
                   help="Chunk size for --method chunk (utterances per LLM call)")
    p.add_argument("--chunk-stride", type=int, default=4,
                   help="Stride between chunk starts for --method chunk. "
                   "Stride < chunk-size produces overlap; later chunks win on overlap positions.")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument(
        "--taxonomy-method",
        choices=["single_prompt", "hybrid", "cluster"],
        default="single_prompt",
        help="single_prompt (default): one LLM call returns the unified taxonomy. "
        "hybrid: cluster cheaply, send 1-2 reps per cluster in one LLM call. "
        "cluster: SBERT-cluster + per-cluster naming + post-merge.",
    )
    p.add_argument("--hybrid-reps-per-cluster", type=int, default=2,
                   help="Representatives per cluster for the hybrid method")
    p.add_argument("--target-size-min", type=int, default=12,
                   help="Min taxonomy size hint (single_prompt and hybrid methods)")
    p.add_argument("--target-size-max", type=int, default=30,
                   help="Max taxonomy size hint (single_prompt and hybrid methods)")
    p.add_argument("--max-prompt-chars", type=int, default=120_000,
                   help="Soft cap on the bootstrap prompt size; uniformly samples "
                   "utterances if exceeded (single_prompt method)")
    p.add_argument("--cluster-algo", choices=["agglo", "kmeans"], default="agglo")
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.55,
        help="For agglo: min cosine similarity to keep in same cluster. "
        "For kmeans: ignored (use --cluster-k).",
    )
    p.add_argument("--cluster-k", type=int, default=None, help="Force k for kmeans (else silhouette)")
    p.add_argument("--centroid-samples", type=int, default=8)
    p.add_argument("--merge-threshold", type=float, default=0.85)
    p.add_argument("--skip-taxonomy", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Label only first N dialogues")
    p.add_argument("--dry-run", action="store_true", help="Run against cache only; fail on miss")
    p.add_argument("--star-dir", default="data/STAR")
    p.add_argument("--out-dir", default="data/STAR_llm_labels")
    p.add_argument("--cache-dir", default=".llm_cache")
    p.add_argument("--log-dir", default="logs")
    return p


def _make_log_path(log_dir: Path, task: str, taxonomy_method: str, label_method: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"llm_label_{task}_{taxonomy_method}_{label_method}_{ts}.jsonl"


async def _async_main(args: argparse.Namespace) -> None:
    cfg = PipelineConfig(
        task=args.task,
        method=args.method,
        model=args.model,
        window_size=args.window_size,
        chunk_size=args.chunk_size,
        chunk_stride=args.chunk_stride,
        concurrency=args.concurrency,
        taxonomy_method=args.taxonomy_method,
        cluster_algo=args.cluster_algo,
        cluster_threshold=args.cluster_threshold,
        cluster_k=args.cluster_k,
        n_samples_per_cluster=args.centroid_samples,
        merge_threshold=args.merge_threshold,
        hybrid_reps_per_cluster=args.hybrid_reps_per_cluster,
        target_size_min=args.target_size_min,
        target_size_max=args.target_size_max,
        max_prompt_chars=args.max_prompt_chars,
        skip_taxonomy=args.skip_taxonomy,
        limit=args.limit,
        dry_run=args.dry_run,
        star_dir=Path(args.star_dir),
        out_dir=Path(args.out_dir),
        cache_dir=Path(args.cache_dir),
        log_path=_make_log_path(Path(args.log_dir), args.task, args.taxonomy_method, args.method),
    )

    convs = load_star_for_task(str(cfg.star_dir), cfg.task)
    print(f"Loaded {len(convs)} {cfg.task} dialogues from {cfg.star_dir}")
    if not convs:
        raise SystemExit(f"No conversations found for task={cfg.task}")

    embedder = EmbeddingCache()
    cache = LLMCache(cfg.cache_dir)
    logger = CallLogger(cfg.log_path)

    t0 = time.time()
    async with LLMClient(cfg.model, cache, logger, cfg.dry_run, cfg.concurrency) as client:
        summary = await run_pipeline(cfg, convs, embedder, client)
        elapsed = time.time() - t0
        print()
        print(f"Done in {elapsed:.1f}s.")
        print(f"  Taxonomy: {summary['n_user_labels']} user + {summary['n_agent_labels']} agent")
        print(f"  Dialogues labeled: {summary['n_dialogues_labeled']}")
        print(f"  Cache hits: {client.cache_hits} | API calls: {client.api_calls}")
        print(
            f"  Tokens: {client.usage.input_tokens} in / "
            f"{client.usage.output_tokens} out  |  "
            f"est. ${client.estimated_cost_usd():.4f} "
            f"(prices for {cfg.model} from MODEL_PRICES)"
        )
        print(f"  Log: {cfg.log_path}")


def main() -> None:
    # Load .env into os.environ before LLMClient checks for OPENAI_API_KEY.
    # Existing env vars take precedence over .env values.
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env", override=False)
    except ImportError:
        pass

    args = _build_argparser().parse_args()
    if args.dry_run:
        print("[dry-run] Cache-only mode; will fail on any cache miss.")
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
