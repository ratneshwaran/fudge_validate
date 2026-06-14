"""TODO 5 — LLM DAG generation for the cross-LLM comparison.

For each model x prompt-variant x PE phase, drive an LLM through the
DAG-generation prompts in `prompts.yaml` and save the resulting dialogue-flow
DAG as Mermaid (`dag.mmd`) plus a parsed `dag.json` (nodes/edges), per the
format in PIPELINE.md.

Routing (option C, per user 2026-06-02): all four models go through OpenRouter
with a single OPENROUTER_API_KEY. Edit MODEL_REGISTRY to repoint a model at a
different OpenAI-compatible endpoint — the client only needs base_url + an
api-key env var, so swapping providers is a one-line change.

Prompt variants (EXPLAINER.md TODO 5):
  v1  = prompt 1 alone (one call)
  v2  = prompts 1-5 fused into a single call
  v3  = prompts 1-5 run sequentially as a multi-turn conversation

prompts 2-5 only run for v2/v3. Prompt 5 ("merge with data examples") gets its
{{thousand_voices_data}} slot filled with N randomly sampled TV *training*
conversations for the phase (from data/splits/TV_v1.json, so test data never
leaks into DAG construction).

Output layout:
  data/dags/<model>/<variant>/<phase>/dag.mmd
  data/dags/<model>/<variant>/<phase>/dag.json
  data/dags/<model>/<variant>/<phase>/transcript.json

Run:
  python scripts/generate_llm_dags.py --model gpt-oss-20b --variant v1 \\
      --phase P6 --limit-examples 8                 # cheap pilot
  python scripts/generate_llm_dags.py --all-models --variant v3 --phase P5 P6 P7
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.modules["tensorflow"] = None

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import llm_label_star as star  # reuse LLMCache / CallLogger / Usage  # noqa: E402

from fudge.data_loader import load_thousand_voices_dialogues  # noqa: E402
from fudge.splits import load_split, split_conversations  # noqa: E402
from fudge.types import Conversation  # noqa: E402

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("pyyaml is required: pip install pyyaml") from e


# ---------------------------------------------------------------------------
# Model registry — all OpenRouter (option C)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# friendly name -> {slug, base_url, api_key_env}. Add/repoint freely.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "deepseek-v3.2": {
        "slug": "deepseek/deepseek-v3.2-exp",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "kimi-k2-0905": {
        "slug": "moonshotai/kimi-k2-0905",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "gpt-oss-20b": {
        "slug": "openai/gpt-oss-20b",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "gpt-5.1": {
        "slug": "openai/gpt-5.1",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
}

VARIANTS = ("v1", "v2", "v3")
DEFAULT_PHASES = ("P5", "P6", "P7")


# ---------------------------------------------------------------------------
# Free-text OpenAI-compatible client (DAG output is Mermaid, not JSON schema)
# ---------------------------------------------------------------------------

class DAGClient:
    """Async OpenAI-compatible client for free-text generation.

    Mirrors star.LLMClient (cache + retry + cost tracking) but returns raw
    text rather than schema-validated JSON, and supports a custom base_url so
    one client class serves any OpenAI-compatible provider.
    """

    def __init__(
        self,
        model_name: str,
        slug: str,
        base_url: str,
        api_key_env: str,
        cache: star.LLMCache,
        logger: star.CallLogger,
        dry_run: bool,
        concurrency: int,
    ):
        self.model_name = model_name
        self.slug = slug
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.cache = cache
        self.logger = logger
        self.dry_run = dry_run
        self.semaphore = asyncio.Semaphore(concurrency)
        self.usage = star.Usage()
        self.cache_hits = 0
        self.api_calls = 0
        self._client = None

    async def __aenter__(self):
        if not self.dry_run:
            from openai import AsyncOpenAI

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.api_key_env} is not set. Export it (or .env) before "
                    f"running, or use --dry-run for cache-only."
                )
            self._client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.close()

    async def complete(self, stage: str, messages: list[dict], max_retries: int = 5) -> str:
        # Cache key keys on the slug (the thing that actually determines output).
        payload = {"messages": messages}
        key = star.LLMCache.make_key(self.slug, stage, payload)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            await self.logger.write({
                "ts": datetime.now(timezone.utc).isoformat(),
                "stage": stage, "cache_hit": True, "model": self.slug,
                "messages": messages, "content": hit.get("content"),
            })
            return hit["content"]

        if self.dry_run:
            raise star.DryRunCacheMiss(
                f"--dry-run set but no cache entry for stage={stage} key={key[:12]}…"
            )

        async with self.semaphore:
            delay = 1.0
            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    response = await self._client.chat.completions.create(
                        model=self.slug,
                        messages=messages,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    cls = type(e).__name__
                    if attempt == max_retries - 1:
                        raise
                    if any(s in cls for s in (
                        "RateLimit", "APIConnection", "APITimeout",
                        "InternalServerError", "APIError",
                    )):
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30.0)
                        continue
                    raise
            else:  # pragma: no cover
                raise last_error  # type: ignore[misc]

        content = response.choices[0].message.content or ""
        usage = response.usage
        usage_dict = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        self.usage.add(usage_dict)
        self.api_calls += 1

        self.cache.put(key, {
            "request": {"model": self.slug, "messages": messages},
            "content": content, "usage": usage_dict,
        })
        await self.logger.write({
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage, "cache_hit": False, "model": self.slug,
            "messages": messages, "content": content, "usage": usage_dict,
        })
        return content


# ---------------------------------------------------------------------------
# Prompt loading + Prompt-5 data injection
# ---------------------------------------------------------------------------

def load_prompts(path: Path) -> tuple[dict[int, str], dict[str, dict]]:
    """Return ({prompt_id: text}, {phase_id: {name, description, moves}})."""
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    prompts = {int(p["id"]): p["text"] for p in doc["prompts"]}
    phases = doc.get("phases", {}) or {}
    return prompts, phases


def render_phase_slots(text: str, phase: str, phases: dict[str, dict]) -> str:
    """Fill {{phase_*}} slots for one phase. No-op for prompts without slots."""
    meta = phases.get(phase)
    if not meta:
        raise SystemExit(
            f"prompts.yaml has no `phases` entry for {phase}; cannot phase-condition. "
            f"Add it (see the phases: block) before generating."
        )
    return (text
            .replace("{{phase_id}}", phase)
            .replace("{{phase_name}}", str(meta.get("name", phase)))
            .replace("{{phase_description}}", str(meta.get("description", "")).strip())
            .replace("{{phase_moves}}", str(meta.get("moves", "")).strip()))


def render_tv_examples(convs: list[Conversation], max_chars_per_conv: int) -> str:
    """Render sampled TV training conversations as plain transcript text."""
    blocks: list[str] = []
    for i, c in enumerate(convs, 1):
        lines = []
        for u in c.utterances:
            speaker = "Therapist" if u.actor == "agent" else "Client"
            lines.append(f"{speaker}: {u.text}")
        text = "\n".join(lines)
        if max_chars_per_conv and len(text) > max_chars_per_conv:
            text = text[:max_chars_per_conv] + "\n…[truncated]"
        blocks.append(f"--- Example conversation {i} ---\n{text}")
    return "\n\n".join(blocks)


def sample_phase_training_convs(
    tv_dir: str,
    split: dict,
    phase: str,
    n: int,
    seed: int,
) -> list[Conversation]:
    """Sample n TV *training* conversations for a phase from the locked split."""
    convs = load_thousand_voices_dialogues(tv_dir, task_field="type", require_phases=(phase,))
    train, _ = split_conversations(convs, split, phase)
    if not train:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))[: min(n, len(train))]
    return [train[int(i)] for i in idx]


def fill_prompt5(prompt5: str, tv_examples: str, design_guidelines: str) -> str:
    return (prompt5
            .replace("{{design_guidelines}}", design_guidelines)
            .replace("{{thousand_voices_data}}", tv_examples))


DEFAULT_DESIGN_GUIDELINES = (
    "Preserve every clinically distinct therapist action observed in the data "
    "examples below. Keep the directed-acyclic, bot/user-alternating structure. "
    "Merge only nodes that represent the same high-level dialogue action."
)


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are an expert in Prolonged Exposure (PE) therapy and in designing "
    "dialogue-flow graphs. Follow the user's instructions precisely and, when "
    "asked for a flow, output a valid mermaid.js graph in a ```mermaid fenced "
    "code block."
)


async def run_variant_v1(client: DAGClient, prompts: dict[int, str], phase: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompts[1]},
    ]
    out = await client.complete(stage=f"dag.v1.{phase}.p1", messages=messages)
    messages.append({"role": "assistant", "content": out})
    return {"final": out, "messages": messages}


async def run_variant_v2(
    client: DAGClient, prompts: dict[int, str], phase: str, prompt5_filled: str
) -> dict:
    fused = "\n\n".join([
        "Perform the following five steps in order and return ONLY the final "
        "merged dialogue flow as a single ```mermaid fenced code block.",
        f"STEP 1 — GENERATE:\n{prompts[1]}",
        f"STEP 2 — CRITIQUE:\n{prompts[2]}",
        f"STEP 3 — REVISE:\n{prompts[3]}",
        f"STEP 4 — FINALISE:\n{prompts[4]}",
        f"STEP 5 — MERGE WITH DATA:\n{prompt5_filled}",
    ])
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": fused},
    ]
    out = await client.complete(stage=f"dag.v2.{phase}.fused", messages=messages)
    messages.append({"role": "assistant", "content": out})
    return {"final": out, "messages": messages}


async def run_variant_v3(
    client: DAGClient, prompts: dict[int, str], phase: str, prompt5_filled: str
) -> dict:
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    steps = [(1, prompts[1]), (2, prompts[2]), (3, prompts[3]),
             (4, prompts[4]), (5, prompt5_filled)]
    final = ""
    for pid, text in steps:
        messages.append({"role": "user", "content": text})
        out = await client.complete(stage=f"dag.v3.{phase}.p{pid}", messages=messages)
        messages.append({"role": "assistant", "content": out})
        final = out
    return {"final": final, "messages": messages}


# ---------------------------------------------------------------------------
# Mermaid extraction + parsing -> nodes/edges JSON
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:mermaid)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Node decls: ID["label"] / ID[label] / ID(label) / ID{label} (also ([ ]) etc.)
_NODE_RE = re.compile(r"\b([A-Za-z][\w]*)\s*([\[\({])+\s*\"?(.*?)\"?\s*[\]\)}]+")
# A node label attached to an id, in any bracket style, for stripping.
_LABEL_RE = re.compile(r"(\[[^\]]*\]|\([^)]*\)|\{[^}]*\})")
# An edge-label like -->|text| or -- text --> (pipe form); strip the pipes.
_EDGE_LABEL_RE = re.compile(r"\|[^|]*\|")
# Any mermaid arrow run (-->, ---, -.->, ==>, --x, --o, <-->, ...).
_ARROW_RE = re.compile(r"<?[-.=]{2,}[->ox]?")


def extract_mermaid(text: str) -> str:
    """Pull the last mermaid block; fall back to a `graph`/`flowchart` region."""
    blocks = _FENCE_RE.findall(text)
    candidates = [b.strip() for b in blocks
                  if re.search(r"\b(graph|flowchart)\b", b, re.IGNORECASE)]
    if candidates:
        return candidates[-1]
    if blocks:
        return blocks[-1].strip()
    m = re.search(r"((?:graph|flowchart)\b.*)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _infer_actor(node_id: str) -> str:
    # Dominant convention: the id starts with the actor letter (B#/U#), incl.
    # suffixed variants (BEnd1, B_CRISIS, U3a, ...). Some models prefix ids
    # (e.g. "nodeB1"/"nodeU1"); fall back to the first actor letter that is
    # immediately followed by a digit so those still resolve instead of going
    # "unknown" (which would silently drop the node from actor-matched alignment).
    head = node_id[:1].upper()
    if head == "B":
        return "agent"
    if head == "U":
        return "user"
    m = re.search(r"([BUbu])\d", node_id)
    if m:
        return "agent" if m.group(1).upper() == "B" else "user"
    return "unknown"


def parse_mermaid_dag(mmd: str) -> dict:
    """Parse a mermaid flowchart into {nodes, edges}.

    nodes: [{id, actor, label}]; edges: [{from, to}]. Best-effort and tolerant
    of the formatting variation different LLMs emit; downstream TODO 6/7 can
    re-validate. `style`/`classDef`/`linkStyle`/`subgraph` lines are ignored.
    """
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    skip = ("style ", "classdef ", "linkstyle ", "class ", "subgraph",
            "direction ", "%%", "end")

    for raw in mmd.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(skip) or low.startswith(("graph", "flowchart")):
            # still scan graph header line? labels rarely there; skip safely
            if low.startswith(("graph", "flowchart")):
                continue
            if low.startswith(("style", "classdef", "linkstyle", "class ",
                               "subgraph", "direction", "%%")) or low == "end":
                continue

        for nid, _br, label in _NODE_RE.findall(line):
            if label.strip():
                nodes[nid] = label.strip()
            else:
                nodes.setdefault(nid, "")

        # Strip node labels + edge labels, normalise arrows to a sentinel, then
        # walk the token stream so chains (A --> B --> C) and inline label+edge
        # decls (B1[..] --> U1[..]) both yield edges.
        cleaned = _LABEL_RE.sub(" ", line)
        cleaned = _EDGE_LABEL_RE.sub(" ", cleaned)
        cleaned = _ARROW_RE.sub(" >> ", cleaned)
        prev_id: str | None = None
        pending = False
        for tok in cleaned.split():
            if tok == ">>":
                pending = True
                continue
            if not re.fullmatch(r"[A-Za-z]\w*", tok):
                prev_id = None
                pending = False
                continue
            if pending and prev_id is not None:
                edges.append((prev_id, tok))
                nodes.setdefault(prev_id, nodes.get(prev_id, ""))
                nodes.setdefault(tok, nodes.get(tok, ""))
            prev_id = tok
            pending = False

    # de-dup edges, preserve order
    seen = set()
    uniq_edges = []
    for a, b in edges:
        if (a, b) not in seen:
            seen.add((a, b))
            uniq_edges.append({"from": a, "to": b})

    node_list = [
        {"id": nid, "actor": _infer_actor(nid), "label": label}
        for nid, label in nodes.items()
    ]
    return {"nodes": node_list, "edges": uniq_edges}


def check_dag_validity(dag: dict) -> dict:
    """Structural sanity of a parsed DAG (TODO 5 guard for #2).

    Reports rather than mutates — downstream alignment needs an acyclic, single-
    component graph, so a failing DAG should be flagged for a reroll/prune before
    it reaches TODO 7. Checks: acyclicity, weak-connectivity (one component),
    unknown-actor nodes, and bot/user alternation violations.
    """
    import networkx as nx

    nodes = dag["nodes"]
    edges = dag["edges"]
    actor = {n["id"]: n["actor"] for n in nodes}
    g = nx.DiGraph()
    g.add_nodes_from(actor)
    g.add_edges_from((e["from"], e["to"]) for e in edges)

    acyclic = nx.is_directed_acyclic_graph(g)
    n_cycles = 0 if acyclic else len(list(nx.simple_cycles(g)))
    n_components = nx.number_weakly_connected_components(g) if g.number_of_nodes() else 0
    n_unknown = sum(1 for a in actor.values() if a == "unknown")
    # same-actor adjacency (B->B or U->U) breaks strict alternation
    n_alt_violations = sum(
        1 for e in edges
        if actor.get(e["from"]) == actor.get(e["to"]) and actor.get(e["from"]) in ("agent", "user")
    )
    ok = acyclic and n_components <= 1 and n_unknown == 0
    return {
        "ok": ok,
        "acyclic": acyclic,
        "n_cycles": n_cycles,
        "n_components": n_components,
        "n_unknown_actor": n_unknown,
        "n_alternation_violations": n_alt_violations,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def generate_one(
    model_name: str,
    variant: str,
    phase: str,
    prompts: dict[int, str],
    phases: dict[str, dict],
    tv_dir: str,
    split: dict,
    out_root: Path,
    cache: star.LLMCache,
    logger: star.CallLogger,
    dry_run: bool,
    concurrency: int,
    n_examples: int,
    max_chars_per_conv: int,
    seed: int,
) -> dict:
    spec = MODEL_REGISTRY[model_name]
    out_dir = out_root / model_name / variant / phase
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase-condition every prompt (fills {{phase_*}} slots) before use.
    prompts = {pid: render_phase_slots(text, phase, phases) for pid, text in prompts.items()}

    prompt5_filled = ""
    n_used = 0
    if variant in ("v2", "v3"):
        sampled = sample_phase_training_convs(tv_dir, split, phase, n_examples, seed)
        n_used = len(sampled)
        tv_examples = render_tv_examples(sampled, max_chars_per_conv)
        prompt5_filled = fill_prompt5(prompts[5], tv_examples, DEFAULT_DESIGN_GUIDELINES)

    async with DAGClient(
        model_name, spec["slug"], spec["base_url"], spec["api_key_env"],
        cache, logger, dry_run, concurrency,
    ) as client:
        if variant == "v1":
            result = await run_variant_v1(client, prompts, phase)
        elif variant == "v2":
            result = await run_variant_v2(client, prompts, phase, prompt5_filled)
        elif variant == "v3":
            result = await run_variant_v3(client, prompts, phase, prompt5_filled)
        else:
            raise ValueError(f"unknown variant {variant}")

        mmd = extract_mermaid(result["final"])
        dag = parse_mermaid_dag(mmd)
        validity = check_dag_validity(dag)

        (out_dir / "dag.mmd").write_text(mmd, encoding="utf-8")
        with open(out_dir / "dag.json", "w", encoding="utf-8") as f:
            json.dump(dag, f, ensure_ascii=False, indent=2)
        with open(out_dir / "validity.json", "w", encoding="utf-8") as f:
            json.dump(validity, f, ensure_ascii=False, indent=2)
        with open(out_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump({
                "model": model_name, "slug": spec["slug"], "variant": variant,
                "phase": phase, "n_examples": n_used,
                "messages": result["messages"],
            }, f, ensure_ascii=False, indent=2)

        return {
            "model": model_name, "variant": variant, "phase": phase,
            "n_nodes": len(dag["nodes"]), "n_edges": len(dag["edges"]),
            "n_examples": n_used, "mmd_chars": len(mmd),
            "validity": validity,
            "cache_hits": client.cache_hits, "api_calls": client.api_calls,
            "tokens_in": client.usage.input_tokens,
            "tokens_out": client.usage.output_tokens,
        }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", nargs="+", choices=list(MODEL_REGISTRY),
                   help="Model(s) to run, space-separated.")
    p.add_argument("--all-models", action="store_true",
                   help="Run every model in MODEL_REGISTRY.")
    p.add_argument("--variant", nargs="+", choices=list(VARIANTS),
                   help="Prompt variant(s), space-separated. Default: all three.")
    p.add_argument("--phase", nargs="+", default=list(DEFAULT_PHASES),
                   help="PE phase(s) to generate for (P5 P6 P7 P8 P10 P11).")
    p.add_argument("--n-examples", type=int, default=10,
                   help="TV training conversations to sample for Prompt 5.")
    p.add_argument("--max-chars-per-conv", type=int, default=4000,
                   help="Truncate each Prompt-5 example conversation to this many chars.")
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="Cache-only; fails on cache miss. No API key needed.")
    p.add_argument("--tv-dir",
                   default="data/thousand-voices-trauma/ThousandVoicesOfTrauma")
    p.add_argument("--split", default="data/splits/TV_v1.json")
    p.add_argument("--prompts", default="prompts.yaml")
    p.add_argument("--out-dir", default="data/dags")
    p.add_argument("--cache-dir", default=".llm_cache")
    p.add_argument("--log-dir", default="logs")
    return p


async def _async_main(args: argparse.Namespace) -> None:
    models = list(MODEL_REGISTRY) if args.all_models else (args.model or [])
    if not models:
        raise SystemExit("Specify --model NAME (repeatable) or --all-models.")
    variants = args.variant or list(VARIANTS)

    prompts, phases = load_prompts(Path(args.prompts))
    split = load_split(args.split)

    cache = star.LLMCache(Path(args.cache_dir))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = star.CallLogger(log_dir / f"generate_llm_dags_{ts}.jsonl")
    out_root = Path(args.out_dir)

    summaries = []
    for model_name in models:
        for variant in variants:
            for phase in args.phase:
                t0 = time.time()
                s = await generate_one(
                    model_name=model_name, variant=variant, phase=phase,
                    prompts=prompts, phases=phases, tv_dir=args.tv_dir, split=split,
                    out_root=out_root, cache=cache, logger=logger,
                    dry_run=args.dry_run, concurrency=args.concurrency,
                    n_examples=args.n_examples,
                    max_chars_per_conv=args.max_chars_per_conv, seed=args.seed,
                )
                s["elapsed_s"] = round(time.time() - t0, 1)
                summaries.append(s)
                v = s["validity"]
                flag = "ok" if v["ok"] else (
                    "BAD[" + ",".join(
                        x for x, on in [
                            (f"{v['n_cycles']}cyc", not v["acyclic"]),
                            (f"{v['n_components']}comp", v["n_components"] > 1),
                            (f"{v['n_unknown_actor']}unk", v["n_unknown_actor"] > 0),
                        ] if on
                    ) + "]"
                )
                print(
                    f"[{model_name}/{variant}/{phase}] "
                    f"{s['n_nodes']} nodes, {s['n_edges']} edges | "
                    f"valid={flag} alt_viol={v['n_alternation_violations']} | "
                    f"ex={s['n_examples']} | "
                    f"calls={s['api_calls']} hits={s['cache_hits']} | "
                    f"tok={s['tokens_in']}in/{s['tokens_out']}out | "
                    f"{s['elapsed_s']}s"
                )

    print("\n=== Run summary ===")
    tot_in = sum(s["tokens_in"] for s in summaries)
    tot_out = sum(s["tokens_out"] for s in summaries)
    print(f"{len(summaries)} DAGs generated | tokens {tot_in}in/{tot_out}out")
    empties = [s for s in summaries if s["n_nodes"] == 0]
    if empties:
        print(f"[warn] {len(empties)} DAG(s) parsed to 0 nodes — inspect dag.mmd:")
        for s in empties:
            print(f"   {s['model']}/{s['variant']}/{s['phase']}")


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
