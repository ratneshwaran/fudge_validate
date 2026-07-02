"""LLM-as-judge second metric (Option B / AutoEval-ToD Domain Compliance).

SCAFFOLD — wired and import-clean, but not yet run (needs OPENROUTER_API_KEY +
the TV data/splits). The parallel, independent-failure-mode metric to FuDGE
(METRIC_OPTIONS.md, archived). FuDGE measures geometric alignment; this judge
reads multi-turn stretches holistically, so summary-level DAGs work directly and
the length confound never arises.

Grounded in AutoEval-ToD (Jain et al., NAACL 2025, pp. 10133-10148). Their
"Domain Compliance" metric scores each chatbot response against predefined
domain rules with an LLM (Prompt H.3, scale 1=compliant / 0=non-compliant /
-1=N/A) and reports % adherence (their Table 4). Crucially their rules check
response-level ADHERENCE, not dialogue ORDER. A dialogue-flow DAG's whole value
is order, so we extend the recipe with explicit TRANSITION rules (see below).

Pipeline:
  1. dag_to_checklist: turn an LLM DAG into a checklist of NL rules — both
     'presence' rules (an action happened) AND 'transition' rules (X before Y),
     the latter derived from the DAG edges. Without transition rules the judge
     degrades into a bag-of-intents content checker and the flow signal is lost.
  2. judge_conversation: give the judge the rules + full transcript; per rule it
     returns +1 satisfied / 0 N/A / -1 violated with a one-line justification.
     (We use METRIC_OPTIONS.md's +1/0/-1 = satisfied/N/A/violated convention;
     note the AutoEval-ToD H.3 encoding differs — only "exclude N/A from the
     mean" matters, not the integer labels.)
  3. score_session: mean of rule scores excluding N/A. Range [-1, 1], higher =
     better compliance (OPPOSITE direction to FuDGE, where lower = better fit).

Locked guards:
  - Circularity: the judge must NOT be any generator model under comparison
    (assert_not_circular). AutoEval-ToD used a fixed Claude judge; we keep the
    judge in a SEPARATE registry from the generators for exactly this reason.
  - Determinism: temperature 0 + on-disk cache (reuses the labelling cache).

Validation (this metric's Step-1 equivalent): hand-score ~20 sessions and
compare to the judge. AutoEval-ToD report raw ACCURACY (94-97%) and a 96%
inter-annotator agreement; raw accuracy can overstate agreement on imbalanced
labels, so ALSO report Cohen's/weighted kappa or Krippendorff's alpha
(see LITERATURE_SCAN.md). Target the AutoEval-ToD ballpark.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.modules["tensorflow"] = None

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import llm_label_star as star  # noqa: E402  reuse LLMCache / CallLogger / Usage
import generate_llm_dags as gen  # noqa: E402  reuse MODEL_REGISTRY / prompt loaders
from fudge.types import Conversation  # noqa: E402

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Judge registry — kept SEPARATE from gen.MODEL_REGISTRY (the generators) so the
# circularity guard is meaningful. Must be a strong model that is NOT a
# generator under comparison. AutoEval-ToD used Claude-3-Sonnet. Verify the exact
# OpenRouter slug at https://openrouter.ai/models before a real run.
JUDGE_REGISTRY: dict[str, dict[str, str]] = {
    "claude-sonnet": {
        "slug": "anthropic/claude-sonnet-4.6",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "claude-opus": {
        "slug": "anthropic/claude-opus-4.8",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
}

DEFAULT_JUDGE_MODEL = "claude-sonnet"     # strong, fixed, and NOT a generator
DEFAULT_TV_DIR = "data/thousand-voices-trauma/ThousandVoicesOfTrauma"
SPLIT_PATH = "data/splits/TV_v1.json"
DAGS_ROOT = "data/dags"
JUDGE_ROOT = "data/judge"

# --------------------------------------------------------------------------- #
# JSON schemas (strict) — the judge is forced into these shapes.
# --------------------------------------------------------------------------- #

CHECKLIST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["presence", "transition"]},
                    "text": {"type": "string"},
                },
                "required": ["id", "kind", "text"],
            },
        }
    },
    "required": ["rules"],
}


def verdict_schema(n_rules: int) -> dict:
    """Per-rule +1/0/-1 verdict, exactly one entry per rule."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": n_rules,
                "maxItems": n_rules,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rule_id": {"type": "string"},
                        "score": {"type": "integer", "enum": [-1, 0, 1]},
                        "justification": {"type": "string"},
                    },
                    "required": ["rule_id", "score", "justification"],
                },
            }
        },
        "required": ["verdicts"],
    }


# --------------------------------------------------------------------------- #
# OpenRouter client with JSON-schema output (DAGClient routing + LLMClient schema)
# --------------------------------------------------------------------------- #

class JudgeClient:
    """Async OpenAI-compatible client returning schema-validated JSON.

    Combines generate_llm_dags.DAGClient's provider routing (custom base_url +
    api-key env) with llm_label_star.LLMClient's json_schema response_format and
    temperature 0. Cache + retry come from the shared star infra.
    """

    def __init__(self, model_name, slug, base_url, api_key_env,
                 cache, logger, dry_run, concurrency, temperature=0.0):
        self.model_name = model_name
        self.slug = slug
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.cache = cache
        self.logger = logger
        self.dry_run = dry_run
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(concurrency)
        self.usage = star.Usage()
        self.cache_hits = 0
        self.api_calls = 0
        self._client = None

    @classmethod
    def from_registry(cls, model_name, **kw):
        entry = JUDGE_REGISTRY.get(model_name)
        if entry is None:
            raise SystemExit(f"unknown judge model {model_name!r}; "
                             f"known: {sorted(JUDGE_REGISTRY)}")
        return cls(model_name, entry["slug"], entry["base_url"],
                   entry["api_key_env"], **kw)

    async def __aenter__(self):
        if not self.dry_run:
            from openai import AsyncOpenAI
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} is not set (or use --dry-run).")
            self._client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        return self

    async def __aexit__(self, *exc):
        if self._client is not None:
            await self._client.close()

    async def call(self, stage, messages, schema_name, schema, max_retries=5):
        payload = {"messages": messages, "schema_name": schema_name, "schema": schema,
                   "temperature": self.temperature}
        key = star.LLMCache.make_key(self.slug, stage, payload)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit["parsed"]
        if self.dry_run:
            raise star.DryRunCacheMiss(f"--dry-run, no cache for stage={stage} {key[:12]}")

        async with self.semaphore:
            delay = 1.0
            for attempt in range(max_retries):
                try:
                    resp = await self._client.chat.completions.create(
                        model=self.slug, messages=messages, temperature=self.temperature,
                        response_format={"type": "json_schema", "json_schema": {
                            "name": schema_name, "strict": True, "schema": schema}},
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    cls = type(e).__name__
                    if attempt == max_retries - 1:
                        raise
                    if any(s in cls for s in ("RateLimit", "APIConnection", "APITimeout",
                                              "InternalServerError", "APIError")):
                        await asyncio.sleep(delay); delay = min(delay * 2, 30.0); continue
                    raise

        parsed = json.loads(resp.choices[0].message.content)
        usage = resp.usage
        usage_dict = {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                      "output_tokens": getattr(usage, "completion_tokens", 0) or 0}
        self.usage.add(usage_dict); self.api_calls += 1
        self.cache.put(key, {"request": {"model": self.slug, "messages": messages},
                             "parsed": parsed, "usage": usage_dict})
        await self.logger.write({"ts": datetime.now(timezone.utc).isoformat(),
                                 "stage": stage, "model": self.slug, "parsed": parsed})
        return parsed


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

CHECKLIST_SYSTEM = (
    "You convert a dialogue-flow DAG for a Prolonged-Exposure (PE) therapy phase "
    "into a checklist of verifiable rules for grading a real session. Output STRICT "
    "JSON matching the schema."
)


def checklist_user(dag: dict, phase: str, phases: dict) -> str:
    info = phases.get(phase, {})
    nodes = "\n".join(f"  {n['id']} ({n.get('actor','?')}): {n.get('label','')}"
                      for n in dag.get("nodes", []))
    edges = "\n".join(f"  {e['from']} -> {e['to']}" for e in dag.get("edges", []))
    return (
        f"PE phase {phase} — {info.get('name','')}\n{info.get('description','')}\n\n"
        f"DAG nodes:\n{nodes}\n\nDAG edges (order):\n{edges}\n\n"
        "Produce 6-15 rules. Two kinds:\n"
        "- 'presence': a clinically meaningful therapist action happened "
        "(e.g. 'The therapist explained the imaginal-exposure rationale.').\n"
        "- 'transition': ORDER between actions, derived from the edges "
        "(e.g. 'The therapist elicited a baseline SUDS rating BEFORE starting the "
        "narrative.'). At least a THIRD of the rules must be 'transition' rules — "
        "order is the whole point of a flow DAG. Avoid rules that are trivially "
        "always true. Each rule needs a short snake_case id."
    )


JUDGE_SYSTEM = (
    "You are an expert evaluator of Prolonged-Exposure therapy dialogue. For each "
    "rule, return +1 if the transcript clearly satisfies it, -1 if it clearly "
    "violates/contradicts it, and 0 if not applicable or genuinely undecidable. Be "
    "conservative: when unsure, use 0. Justify each verdict in one line citing turn "
    "indices [i]. Output STRICT JSON matching the schema."
)


def judge_user(conv: Conversation, rules: list[dict]) -> str:
    rule_lines = "\n".join(f"  {r['id']} [{r['kind']}]: {r['text']}" for r in rules)
    transcript = "\n".join(f"[{i}] ({u.actor}) {u.text}"
                           for i, u in enumerate(conv.utterances))
    return f"RULES:\n{rule_lines}\n\nTRANSCRIPT:\n{transcript}"


# --------------------------------------------------------------------------- #
# Core operations
# --------------------------------------------------------------------------- #

async def dag_to_checklist(client: JudgeClient, dag: dict, phase: str, phases: dict) -> list[dict]:
    msgs = [{"role": "system", "content": CHECKLIST_SYSTEM},
            {"role": "user", "content": checklist_user(dag, phase, phases)}]
    out = await client.call("checklist", msgs, "checklist", CHECKLIST_SCHEMA)
    return out["rules"]


async def judge_conversation(client: JudgeClient, conv: Conversation, rules: list[dict]) -> list[dict]:
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_user(conv, rules)}]
    out = await client.call("judge", msgs, "verdicts", verdict_schema(len(rules)))
    # The schema enforces the verdict COUNT but not rule_id identity, so map the
    # verdicts back onto the input rules: duplicates are an error, an omitted
    # rule scores 0 (N/A) with a warning, unknown ids are dropped with a warning.
    by_id: dict[str, dict] = {}
    for v in out["verdicts"]:
        if v["rule_id"] in by_id:
            raise ValueError(f"judge returned duplicate verdict for rule {v['rule_id']!r}")
        by_id[v["rule_id"]] = v
    aligned = []
    for r in rules:
        v = by_id.pop(r["id"], None)
        if v is None:
            print(f"[warn] judge omitted rule {r['id']!r}; treating as N/A (0)")
            v = {"rule_id": r["id"], "score": 0,
                 "justification": "omitted by judge; treated as N/A"}
        aligned.append(v)
    if by_id:
        print(f"[warn] judge returned unknown rule ids (ignored): {sorted(by_id)}")
    return aligned


def score_session(verdicts: list[dict]) -> float:
    """Mean of rule scores excluding N/A (0). Range [-1, 1]; higher = better.

    (AutoEval-ToD report this as % adherence = #compliant / (#compliant+#violated);
    that is just (mean+1)/2 of this score.)
    """
    scores = [v["score"] for v in verdicts if v["score"] != 0]
    return sum(scores) / len(scores) if scores else float("nan")


def assert_not_circular(judge_model: str, generator_registry: dict) -> None:
    """Locked guard: the judge must not be a model whose DAGs it grades.

    Compares the underlying OpenRouter SLUGS, not registry key names — the two
    registries share no keyspace, so a name comparison could never fire.
    """
    entry = JUDGE_REGISTRY.get(judge_model)
    if entry is None:
        raise SystemExit(f"unknown judge model {judge_model!r}; "
                         f"known: {sorted(JUDGE_REGISTRY)}")
    judge_slug = entry["slug"]
    clashes = [name for name, e in generator_registry.items()
               if e["slug"] == judge_slug]
    if clashes:
        raise SystemExit(
            f"Circularity violation: judge {judge_model!r} ({judge_slug}) is the same "
            f"model as generator(s) {clashes} under comparison. Pick a judge outside "
            f"that set.")


# --------------------------------------------------------------------------- #
# CLI — build a checklist for one DAG cell (the only step runnable without the
# full split). Judging/discrimination is wired via the functions above and
# specified in LLM_JUDGE_DESIGN.md; it needs the TV split + data to run.
# --------------------------------------------------------------------------- #

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, choices=list(JUDGE_REGISTRY))
    ap.add_argument("--gen-model", required=True, help="generator whose DAG to convert")
    ap.add_argument("--variant", default="v3")
    ap.add_argument("--phase", default="P6")
    ap.add_argument("--dags-root", default=DAGS_ROOT)
    ap.add_argument("--prompts", default="prompts.yaml")
    ap.add_argument("--out-dir", default=JUDGE_ROOT)
    ap.add_argument("--cache-dir", default=".llm_cache")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    return ap


async def _amain(args) -> None:
    # Judge must be external to the generators being compared.
    assert_not_circular(args.judge_model, gen.MODEL_REGISTRY)
    _, phases = gen.load_prompts(Path(args.prompts))
    dag_path = Path(args.dags_root) / args.gen_model / args.variant / args.phase / "dag.json"
    dag = json.load(open(dag_path, encoding="utf-8"))

    cache = star.LLMCache(Path(args.cache_dir))
    logger = star.CallLogger(Path(args.log_dir) / "llm_judge.jsonl")
    client = JudgeClient.from_registry(args.judge_model, cache=cache, logger=logger,
                                       dry_run=args.dry_run, concurrency=args.concurrency)
    async with client:
        rules = await dag_to_checklist(client, dag, args.phase, phases)

    out = Path(args.out_dir) / args.judge_model / args.gen_model / args.variant / args.phase
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"judge_model": args.judge_model, "gen_model": args.gen_model,
               "variant": args.variant, "phase": args.phase, "rules": rules},
              open(out / "checklist.json", "w", encoding="utf-8"), indent=2)
    n_trans = sum(1 for r in rules if r["kind"] == "transition")
    print(f"Wrote {out/'checklist.json'} — {len(rules)} rules ({n_trans} transition)")


def main() -> None:
    asyncio.run(_amain(_build_argparser().parse_args()))


if __name__ == "__main__":
    main()
