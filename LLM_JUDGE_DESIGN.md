# LLM-judge second metric — design (Option B / AutoEval-ToD)

**Date:** 2026-06-20. **Status:** designed + scaffolded (`scripts/llm_judge.py`), not yet run
(needs `OPENROUTER_API_KEY` + the TV split/data). **Reads with:** `METRIC_OPTIONS.md` (Option B),
`LITERATURE_SCAN.md` (the cited evidence behind every choice here).

## Why a second metric
FuDGE measures *geometric* alignment of a conversation to a DAG and, on shallow DAGs, is dominated
by length (the confound this project is fixing). An LLM judge reads a whole multi-turn stretch and
maps it to a stage by *understanding* it, so summary-level DAGs work directly and length never
enters. Its failure modes are independent of FuDGE's (no embeddings, no edit distance), which is the
whole point of carrying two metrics: a conclusion is "strong" only when both agree, and
disagreements are the interesting cases. This is METHODOLOGY TODO 9.

## Grounding: what we borrow from AutoEval-ToD, and what we change
AutoEval-ToD's **Domain Compliance** (Jain et al., NAACL 2025, pp. 10133–10148) scores each chatbot
*response* against predefined domain rules with an LLM (their Prompt H.3), reported as % adherence.

- **Borrow:** the rule-scoring recipe, the fixed strong judge (they used Claude-3-Sonnet), the
  temperature-0 + cache determinism, and the hand-scored human validation.
- **Change (critical):** their rules check response-level *adherence*, **not order**. A dialogue-flow
  DAG's value is *order*, so we add **transition rules** ("X happened before Y", derived from the DAG
  edges). Without them the judge silently degrades into a bag-of-intents content checker and we lose
  the structural signal that justified DAGs at all. See `LITERATURE_SCAN.md` §1.

## Pipeline (implemented in `scripts/llm_judge.py`)
1. **`dag_to_checklist(dag, phase)`** → an LLM converts the DAG (nodes + edges + phase description)
   into 6–15 rules, each tagged `presence` or `transition`; ≥⅓ must be `transition`. Cached per
   (judge, generator, variant, phase). This is the only step runnable today (the CLI does it).
2. **`judge_conversation(conv, rules)`** → the judge reads the rule list + full transcript and returns
   one verdict per rule: **+1 satisfied / 0 N/A / −1 violated** + a one-line justification citing turn
   indices. Strict JSON schema; temperature 0; cached.
3. **`score_session(verdicts)`** → mean of verdicts excluding N/A. Range **[−1, 1], higher = better**
   compliance — *opposite direction to FuDGE* (lower = better). (% adherence as in the paper is just
   `(mean+1)/2`.)

## Locked guards
- **Circularity (enforced in code):** the judge must not be any generator under comparison.
  `JUDGE_REGISTRY` (Claude models) is kept **separate** from the generators' `MODEL_REGISTRY`, and
  `assert_not_circular` raises if they overlap. (This caught a real bug: defaulting the judge to
  `gpt-5.1` would have been circular — `gpt-5.1` is a generator.)
- **Determinism:** temperature 0 + the on-disk `LLMCache` shared with the labelling pipeline.

## The discrimination experiment (to build: `experiments/llm_judge_discrimination.py`)
Mirror `experiments/llm_dag_discrimination.py`, but with judge session-scores and the **flipped
direction**:
- For phase P's DAG → build its checklist once; judge phase-P test convs (positives) and every other
  phase's test convs (negatives).
- A good P DAG scores its own phase **high** and others **low** → discrimination gap = `in_mean −
  out_mean > 0` (sign flipped vs FuDGE). Reuse the existing Mann-Whitney (alternative `greater`),
  bootstrap CI, and Bonferroni machinery; save per-conversation scores + justifications.
- No length normalization needed — the judge is length-agnostic by construction (the headline
  contrast vs FuDGE).

## Validation — this metric's Step-1 gate (to build)
Per `LITERATURE_SCAN.md` §1: hand-score ~20 sessions (a clinician or the team) on a sample of rules,
then compare to the judge. **Report Cohen's / weighted κ (and Krippendorff's α if >2 raters) IN
ADDITION to raw accuracy** — AutoEval-ToD's 94–97% is raw accuracy, which overstates agreement on
imbalanced labels; a chance-corrected statistic is the defensible number. Target the AutoEval-ToD
ballpark. Until this passes, judge discrimination numbers are exploratory, exactly as the prefix-tree
gate governs FuDGE.

## Cost (from the paper, scaled)
≈ $0.0045 per judge call (Claude-Sonnet-class, ~1k in/100 out). One call per (conversation × DAG):
e.g. 150 test + 750 out-of-phase = 900 convs × 3 phases ≈ 2.7k calls ≈ **~$12 per generator**, plus
~one checklist call per cell. Cache makes re-runs free. Budget accordingly; not the free, exact
number FuDGE gives.

## Status / next steps
- ✅ Scaffolded + import-tested: `JudgeClient` (OpenRouter + JSON schema), `dag_to_checklist`,
  `judge_conversation`, `score_session`, circularity guard, checklist CLI.
- ⏳ Needs data/keys to run: build a checklist (`python scripts/llm_judge.py --gen-model gpt-oss-20b
  --variant v3 --phase P6`), eyeball the rules (esp. transition rules), then build the discrimination
  experiment + the hand-scored validation.
- 🔒 Use a Claude judge (verify the exact OpenRouter slug in `JUDGE_REGISTRY` at
  openrouter.ai/models); never a generator from the comparison set.
