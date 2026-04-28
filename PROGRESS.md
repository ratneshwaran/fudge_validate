# Progress Tracker

Working doc — tracks current state and upcoming decisions. The formal write-up
lives in `progress_summary.tex` and `VALIDATION_REPORT.md`; this file is for
day-to-day "what am I doing" reference.

---

## Phase 1 — FuDGE validation on STAR ✅ DONE

- Implemented FuDGE from scratch (naive + efficient algorithms, both verified
  identical across 28 oracle tests).
- Reproduced the Table 1b discrimination experiment on `hotel_book` and
  `bank_fraud_report`, with prefix-trie flows built from observed STAR
  intent sequences.
- Result: **STRONG PASS** on both tasks. No 1σ overlap, score ratios 3.0× and
  3.7× respectively. Full numbers in `VALIDATION_REPORT.md`.

What this means: FuDGE works as a distance metric for dialogue flows when fed
gold-quality `(actor, label)` annotations.

---

## Phase 2 — LLM-based intent labeling 🚧 IN PROGRESS

**Why:** STAR has gold `ActionLabel` on agent turns only; user turns are
labeled with the heuristic `user_before_<next_agent_intent>`. The Thousand
Voices therapy dataset has no labels at all. To run FuDGE there, we need an
LLM-driven labeling pipeline that doesn't depend on either signal.

### What's built

- `scripts/llm_label_star.py` — async OpenAI pipeline:
  - Stage 1: SBERT-cluster utterances → LLM names each cluster → optional merge
  - Stage 2: per-utterance labeling, two methods (`whole` / `window`)
  - On-disk cache, JSONL logs, retry, cost tracking, `--dry-run`,
    `--skip-taxonomy`, `--limit`
- `tests/test_llm_label_smoke.py` — 4 hermetic tests (real `EmbeddingCache`,
  mocked OpenAI), all passing
- `.env` workflow for `OPENAI_API_KEY`
- `pyproject.toml` deps: `openai>=1.50`, `python-dotenv>=1.0`

### Known issue from the smoke run (2026-04-28)

First small run on `hotel_book` (limit 3, cost $0.11):

| | unique utterances | labels produced |
|---|---|---|
| user | 910 | **122** ← way too fine |
| agent | 48 | 17 ← reasonable |

The user-side taxonomy has many near-duplicate labels (e.g. variations of
"request hotel room"). Per-cluster naming has no global view, so synonyms
slip past the post-hoc merge step (currently threshold 0.85).

### Open decision: taxonomy bootstrap method

Considering three approaches as a paper-worthy ablation:

- **(C) per-cluster + post-merge** — what's currently implemented. Reliable,
  but produces synonyms.
- **(A) single-prompt** — sample N utterances, ask the LLM for one unified
  taxonomy. No synonyms by construction; risks missing long-tail intents.
- **(B) hybrid** — cluster cheaply on embeddings, take 1-2 representatives
  per cluster, send all of them in **one** LLM call asking for a unified
  taxonomy. Long-tail coverage + no synonyms. Likely best.

**Proposed evaluation metrics:**

1. **Agreement with gold STAR `ActionLabel` on agent turns** (V-measure / ARI)
   — STAR is a free oracle on the agent side.
2. **Downstream FuDGE discrimination ratio** on `hotel_book` and
   `bank_fraud_report` — does the taxonomy preserve in/out-of-task
   separation? Same setup as `validate_discrimination.py`.

A 3 × 2 results table goes into `VALIDATION_REPORT.md` and feeds the paper.

**Status:** awaiting confirmation to scope.

### Workaround needed in the script

`scripts/llm_label_star.py` currently re-walks `data/STAR/dialogues/` to
recover dialogue IDs because `data_loader.load_star_dialogues` doesn't expose
them. This is captured in `TODO.md` and goes away when Phase 3 lands.

---

## Phase 3 — Wire LLM labels into the FuDGE pipeline ⏳ NEXT

The LLM script writes per-dialogue label files at
`data/STAR_llm_labels/<task>/<method>/<dialogue_id>.json`, but nothing
consumes them yet. Concrete changes (see `TODO.md` for full detail):

1. Expose `dialogue_id` on `Conversation`
2. Add `load_llm_labels(label_dir) -> dict[int, list[str]]`
3. Add `label_source` parameter to `build_flow_from_conversations`, replacing
   the heuristic when provided
4. Add `--label-source` flag to `experiments/validate_discrimination.py`
5. Drop the `load_star_with_ids` workaround in the LLM script

Blocked by Phase 2 method choice (taxonomy method affects what labels look
like, but not the loader plumbing — could land in parallel).

---

## Phase 4 — Re-run Table 1b with LLM labels ⏳ AFTER PHASE 3

- Run validation with each (task × labeling method) combo
- Compare to the heuristic baseline already in `VALIDATION_REPORT.md`
- Key question: do LLM labels preserve the 3-4× discrimination ratio?
  - If yes: the LLM pipeline is a drop-in replacement and we can move to
    Thousand Voices with confidence.
  - If no: investigate which step degrades (taxonomy granularity? labeling
    accuracy?) before scaling.

---

## Phase 5 — Apply to Thousand Voices ⏳ END GOAL

Once Phase 4 confirms the LLM pipeline doesn't break discrimination, run it
on Thousand Voices (mental health) dialogues. There are no gold labels
there, so the metric becomes whatever ablation we settle on in Phase 2.

Dependencies for this phase to start:
- Thousand Voices dataset access / loader (not yet in repo)
- Phase 2 method choice locked in
- Phase 3 loader integration landed

---

## Other paper threads (parallel, unblocked)

These are listed in `progress_summary.tex` as next steps and don't depend
on the labeling pipeline:

- LLM-as-a-judge baseline on STAR (for comparison against FuDGE)
- FF1 (Flow-F1 Score) implementation
- Statistical significance test (Mann–Whitney U) on the existing
  discrimination results — currently we only check 1σ separation

---

## Artifact map

| File | What it is |
|---|---|
| `VALIDATION_REPORT.md` | Phase 1 results, methodology, comparison to paper |
| `progress_summary.tex` | Formal write-up (LaTeX, paper-style) |
| `PROGRESS.md` | This file — working tracker |
| `TODO.md` | Concrete plan for Phase 3 (loader integration) |
| `scripts/llm_label_star.py` | Phase 2 pipeline |
| `tests/test_llm_label_smoke.py` | Phase 2 smoke tests |
| `experiments/validate_discrimination.py` | Phase 1 / 4 experiment runner |
