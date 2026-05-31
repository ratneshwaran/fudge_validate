"""LLM-based intent labeling for Thousand Voices of Trauma (TV).

Adapts the STAR labeling pipeline (scripts/llm_label_star.py) to TV with two
TV-specific changes (per supervisor 2026-05-17):

  1. Label ONLY agent (Therapist) turns. User (Client) turns are uncontrollable
     and don't add structure to the agent-driven dialogue flow. User positions
     get a sentinel label `_user_turn` so the label list stays aligned with
     `Conversation.utterances` and `build_flow_from_conversations` works
     unchanged — the trie just has one user bucket per branch level.

  2. Per-phase taxonomy. Each PE phase (P5..P11) has its own agent-intent
     vocabulary because the protocol differs across stages (orientation vs
     SUDS monitoring vs full exposure vs processing). A unified taxonomy
     would muddle phase-specific moves and destroy the cross-phase
     discrimination signal we need in TODO 4.

Output layout:
  data/TV_llm_labels/<phase>/<taxonomy_method>/taxonomy.json
  data/TV_llm_labels/<phase>/<taxonomy_method>/<method>/<dialogue_id>.json

Files consumed by `fudge.data_loader.load_llm_labels` exactly like STAR labels.

Run:
  python scripts/llm_label_tv.py --phase P10 --taxonomy-method single_prompt \\
      --method whole --limit 5   # smoke test
  python scripts/llm_label_tv.py --phase P10                                 # all P10
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make src/ importable and let us reuse the STAR script as a module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

sys.modules["tensorflow"] = None

import llm_label_star as star  # noqa: E402

from fudge.data_loader import load_thousand_voices_dialogues  # noqa: E402
from fudge.embeddings import EmbeddingCache  # noqa: E402
from fudge.types import Conversation  # noqa: E402

USER_SENTINEL = "_user_turn"


# ---------------------------------------------------------------------------
# TV loader
# ---------------------------------------------------------------------------

def load_tv_for_phase(tv_dir: str, phase: str) -> list[Conversation]:
    """Load TV conversations restricted to one phase (e.g. 'P10')."""
    convs = load_thousand_voices_dialogues(tv_dir, require_phases=(phase,))
    return convs


def collect_agent_utterances(convs: list[Conversation]) -> list[str]:
    """Unique agent texts across all loaded conversations, in first-seen order."""
    seen: dict[str, None] = {}
    for c in convs:
        for u in c.utterances:
            if u.actor == "agent":
                seen.setdefault(u.text, None)
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Agent-only Stage 2 prompts + schemas
# ---------------------------------------------------------------------------

def _agent_only_whole_messages(
    conv: Conversation,
    taxonomy_agent: list[dict],
) -> list[dict]:
    rendered = "\n".join(
        f"  [{i}] ({u.actor}) {u.text}" for i, u in enumerate(conv.utterances)
    )
    agent_indices = [i for i, u in enumerate(conv.utterances) if u.actor == "agent"]
    return [
        {
            "role": "system",
            "content": (
                "You label each Therapist turn in a Prolonged Exposure therapy "
                "dialogue with an intent from a closed agent taxonomy. The "
                "Client turns are shown for context only — do NOT label them. "
                "Output strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{star.render_taxonomy_block('AGENT TAXONOMY', taxonomy_agent)}\n\n"
                "DIALOGUE:\n"
                f"{rendered}\n\n"
                f"Return one entry per Therapist (agent) utterance. The agent "
                f"indices in this dialogue are: {agent_indices}. Each label "
                f"must come from AGENT TAXONOMY. Do not include entries for "
                f"Client (user) indices."
            ),
        },
    ]


def _agent_only_whole_schema(taxonomy_agent: list[dict], agent_indices: list[int]) -> dict:
    enum = sorted({t["label"] for t in taxonomy_agent})
    n_agent = len(agent_indices)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {
            "labels": {
                "type": "array",
                "minItems": n_agent,
                "maxItems": n_agent,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "label"],
                    "properties": {
                        "index": {"type": "integer"},
                        "label": {"type": "string", "enum": enum or ["__none__"]},
                    },
                },
            }
        },
    }


async def label_dialogue_agent_only_whole(
    conv: Conversation,
    taxonomy_agent: list[dict],
    client: star.LLMClient,
    embedder: EmbeddingCache,
) -> tuple[list[str], list[str]]:
    """Label every agent turn via one LLM call; user positions get sentinel."""
    agent_indices = [i for i, u in enumerate(conv.utterances) if u.actor == "agent"]
    warnings: list[str] = []

    if not agent_indices:
        return [USER_SENTINEL] * len(conv.utterances), warnings

    messages = _agent_only_whole_messages(conv, taxonomy_agent)
    schema = _agent_only_whole_schema(taxonomy_agent, agent_indices)
    parsed = await client.call(
        stage="label.whole.tv",
        messages=messages,
        schema_name="agent_dialogue_labels",
        schema=schema,
    )

    by_index: dict[int, str] = {}
    for entry in parsed.get("labels", []):
        i = int(entry["index"])
        if i in set(agent_indices):
            by_index[i] = entry["label"]

    out: list[str] = []
    for i, u in enumerate(conv.utterances):
        if u.actor == "user":
            out.append(USER_SENTINEL)
            continue
        if i not in by_index:
            warnings.append(
                f"dialogue {conv.dialogue_id} idx {i}: missing in agent response"
            )
            out.append(
                star.nearest_taxonomy_label(u.text, taxonomy_agent, embedder)
            )
            continue
        out.append(
            star._validate_label(
                by_index[i], u, taxonomy_agent, embedder, warnings,
                where=f"dialogue {conv.dialogue_id} idx {i}",
            )
        )
    return out, warnings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_phase(
    phase: str,
    convs: list[Conversation],
    taxonomy_method: str,
    model: str,
    out_root: Path,
    cache_dir: Path,
    log_dir: Path,
    embedder: EmbeddingCache,
    target_size_min: int,
    target_size_max: int,
    max_prompt_chars: int,
    concurrency: int,
    limit: int | None,
    dry_run: bool,
) -> dict:
    phase_dir = out_root / phase / taxonomy_method
    phase_dir.mkdir(parents=True, exist_ok=True)
    label_dir = phase_dir / "whole"
    label_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_path = phase_dir / "taxonomy.json"

    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"llm_label_tv_{phase}_{taxonomy_method}_whole_{ts}.jsonl"

    cache = star.LLMCache(cache_dir)
    logger = star.CallLogger(log_path)

    async with star.LLMClient(model, cache, logger, dry_run, concurrency) as client:
        # Stage 1: agent taxonomy
        if taxonomy_path.exists():
            with open(taxonomy_path, encoding="utf-8") as f:
                taxonomy = json.load(f)
            print(f"[{phase}] reusing taxonomy at {taxonomy_path}")
        else:
            agent_texts = collect_agent_utterances(convs)
            print(
                f"[{phase}] bootstrapping {taxonomy_method} taxonomy from "
                f"{len(agent_texts)} unique agent utterances"
            )
            if taxonomy_method == "single_prompt":
                agent_taxonomy = await star.bootstrap_taxonomy_single_prompt(
                    "agent", agent_texts, client,
                    target_size_min, target_size_max, max_prompt_chars,
                )
            elif taxonomy_method == "hybrid":
                agent_taxonomy = await star.bootstrap_taxonomy_hybrid(
                    "agent", agent_texts, client, embedder,
                    cluster_algo="agglo", threshold=0.55, k=None,
                    n_reps_per_cluster=2,
                    target_size_min=target_size_min,
                    target_size_max=target_size_max,
                )
            else:
                raise ValueError(f"taxonomy_method {taxonomy_method} not supported")

            taxonomy = {"agent": agent_taxonomy, "user": [
                {"label": USER_SENTINEL,
                 "description": "Client turn (not labelled; flow is agent-driven)"}
            ]}
            with open(taxonomy_path, "w", encoding="utf-8") as f:
                json.dump(taxonomy, f, ensure_ascii=False, indent=2)
            print(f"[{phase}] wrote taxonomy with {len(taxonomy['agent'])} agent labels")

        taxonomy_blob = json.dumps(taxonomy, sort_keys=True, ensure_ascii=False)
        taxonomy_version = hashlib.sha256(taxonomy_blob.encode("utf-8")).hexdigest()[:16]

        # Stage 2: per-dialogue labeling
        target = convs[:limit] if limit else convs
        print(f"[{phase}] labeling {len(target)} dialogues")

        async def _one(c: Conversation):
            labels, warnings = await label_dialogue_agent_only_whole(
                c, taxonomy["agent"], client, embedder
            )
            with open(label_dir / f"{c.dialogue_id}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "utterance_labels": labels,
                        "taxonomy_version": taxonomy_version,
                    },
                    f, ensure_ascii=False, indent=2,
                )
            return c.dialogue_id, len(warnings)

        results = await asyncio.gather(*[_one(c) for c in target])
        total_warnings = sum(w for _, w in results)

        return {
            "phase": phase,
            "n_agent_labels": len(taxonomy["agent"]),
            "n_dialogues_labeled": len(results),
            "warnings": total_warnings,
            "cache_hits": client.cache_hits,
            "api_calls": client.api_calls,
            "tokens_in": client.usage.input_tokens,
            "tokens_out": client.usage.output_tokens,
            "est_cost_usd": client.estimated_cost_usd(),
            "log_path": str(log_path),
        }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--phase", action="append", required=True,
                   help="PE phase to label (P5, P6, P7, P8, P10, P11). Repeat for multiple.")
    p.add_argument("--taxonomy-method", choices=["single_prompt", "hybrid"],
                   default="single_prompt")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--target-size-min", type=int, default=8)
    p.add_argument("--target-size-max", type=int, default=20)
    p.add_argument("--max-prompt-chars", type=int, default=120_000)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--limit", type=int, default=None,
                   help="Label only first N dialogues per phase (for smoke testing)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tv-dir",
                   default="data/thousand-voices-trauma/ThousandVoicesOfTrauma")
    p.add_argument("--out-dir", default="data/TV_llm_labels")
    p.add_argument("--cache-dir", default=".llm_cache")
    p.add_argument("--log-dir", default="logs")
    return p


async def _async_main(args: argparse.Namespace) -> None:
    embedder = EmbeddingCache()
    out_root = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    log_dir = Path(args.log_dir)

    summaries = []
    for phase in args.phase:
        convs = load_tv_for_phase(args.tv_dir, phase)
        print(f"\n=== {phase}: loaded {len(convs)} conversations ===")
        if not convs:
            print(f"  no convs for {phase}, skipping")
            continue
        t0 = time.time()
        summary = await run_phase(
            phase=phase,
            convs=convs,
            taxonomy_method=args.taxonomy_method,
            model=args.model,
            out_root=out_root,
            cache_dir=cache_dir,
            log_dir=log_dir,
            embedder=embedder,
            target_size_min=args.target_size_min,
            target_size_max=args.target_size_max,
            max_prompt_chars=args.max_prompt_chars,
            concurrency=args.concurrency,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        summary["elapsed_s"] = round(time.time() - t0, 1)
        summaries.append(summary)
        print(
            f"[{phase}] done in {summary['elapsed_s']}s | "
            f"{summary['n_agent_labels']} labels | "
            f"{summary['n_dialogues_labeled']} convs | "
            f"{summary['warnings']} warnings | "
            f"cache_hits={summary['cache_hits']} api_calls={summary['api_calls']} | "
            f"tokens={summary['tokens_in']}in/{summary['tokens_out']}out | "
            f"~${summary['est_cost_usd']:.4f}"
        )

    print("\n=== Run summary ===")
    total = sum(s["est_cost_usd"] for s in summaries)
    print(f"Total estimated cost: ${total:.4f}")


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env", override=False)
    except ImportError:
        pass
    args = _build_argparser().parse_args()
    if args.dry_run:
        print("[dry-run] cache-only mode; failures on cache miss.")
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
