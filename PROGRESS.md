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

## Phase 2 — LLM-based intent labeling ✅ DONE (single-prompt default)

**Why:** STAR has gold `ActionLabel` on agent turns only; user turns are
labeled with the heuristic `user_before_<next_agent_intent>`. The Thousand
Voices therapy dataset has no labels at all. To run FuDGE there, we need an
LLM-driven labeling pipeline that doesn't depend on either signal.

### What's built

- `scripts/llm_label_star.py` — async OpenAI pipeline:
  - **Stage 1 (default `--taxonomy-method single_prompt`)**: send all unique
    utterances for one actor in one LLM call; the model returns a unified
    taxonomy. Eliminates the synonym problem the cluster-then-name approach
    had.
  - **Stage 1 (`--taxonomy-method cluster`)**: SBERT-cluster + per-cluster
    naming + post-merge. Kept for the planned 3-method ablation.
  - Stage 2: per-utterance labeling, two methods (`whole` / `window`).
  - On-disk cache, JSONL logs, retry, cost tracking, `--dry-run`,
    `--skip-taxonomy`, `--limit`.
- `tests/test_llm_label_smoke.py` — hermetic smoke tests (real
  `EmbeddingCache`, mocked OpenAI), all passing.
- `.env` workflow for `OPENAI_API_KEY`.
- `pyproject.toml` deps: `openai>=1.50`, `python-dotenv>=1.0`.

### History

First small smoke run on `hotel_book` (cluster method, limit 3, cost $0.11):

| | unique utterances | labels produced |
|---|---|---|
| user | 910 | **122** ← way too fine |
| agent | 48 | 17 ← reasonable |

The user-side taxonomy had many near-duplicate labels because per-cluster
naming has no global view. **Switched to single-prompt as the default**;
cluster method retained for the ablation.

### Deferred: 3-method taxonomy ablation 🔜

Compare three approaches and add a section to `VALIDATION_REPORT.md`:

- **(A) single-prompt** — current default. One LLM call returns the unified
  taxonomy.
- **(B) hybrid** — cluster cheaply on embeddings, send 1-2 reps per cluster
  in one LLM call (not yet implemented).
- **(C) cluster-then-name** — historical default; available behind
  `--taxonomy-method cluster`.

**Proposed metrics:**
1. Agreement with gold STAR `ActionLabel` on agent turns (V-measure / ARI).
2. Downstream FuDGE discrimination ratio.

Status: explicitly deferred per user — proceed with single-prompt to unblock
Phase 4.

---

## Phase 3 — Wire LLM labels into the FuDGE pipeline ✅ DONE

- `Conversation.dialogue_id` is now a real dataclass field (set by
  `load_star_dialogues`).
- `data_loader.load_llm_labels(label_dir)` reads per-dialogue label files.
- `build_flow_from_conversations(label_source=...)` overrides the
  `user_before_<next_agent_intent>` heuristic when LLM labels are provided.
  Actor and text still come from the STAR event; only the label changes.
- `experiments/validate_discrimination.py` has `--label-root` and
  `--label-method` CLI flags.
- `tests/test_label_source.py` — 5 unit tests covering the plumbing.
- The `load_star_with_ids` workaround in the LLM script is gone.

---

## Phase 4 — Re-run Table 1b with LLM labels ⏳ NEXT (user action)

To execute (assumes `OPENAI_API_KEY` is in `.env`):

```bash
# 1. Generate labels (single-prompt taxonomy, whole-dialogue labeling)
python scripts/llm_label_star.py --task hotel_book --method whole
python scripts/llm_label_star.py --task bank_fraud_report --method whole

# 2. Re-run Table 1b with the new labels
python experiments/validate_discrimination.py \
    --label-root data/STAR_llm_labels --label-method whole
```

Compare to the heuristic baseline in `VALIDATION_REPORT.md`. Key question:
do LLM labels preserve the 3-4× discrimination ratio?

- If yes: drop-in replacement; Thousand Voices is unblocked.
- If no: investigate which step degrades (taxonomy granularity vs labeling
  accuracy) before scaling.

---

## Phase 5 — Apply to Thousand Voices ⏳ END GOAL

Once Phase 4 confirms LLM labeling doesn't break discrimination, run it on
Thousand Voices (mental health). No gold labels there, so we go directly
with whichever taxonomy method we settled on.

Dependencies:
- Thousand Voices dataset access / loader (not yet in repo)
- Phase 4 results landed

---

## Other paper threads (parallel, unblocked)

Listed in `progress_summary.tex` as next steps; independent of the labeling
pipeline:

- LLM-as-a-judge baseline on STAR (for comparison against FuDGE)
- FF1 (Flow-F1 Score) implementation
- Statistical significance test (Mann–Whitney U) on existing discrimination
  results — currently we only check 1σ separation
- 3-method taxonomy ablation (deferred from Phase 2)

---

## Artifact map

| File | What it is |
|---|---|
| `VALIDATION_REPORT.md` | Phase 1 results, methodology, comparison to paper |
| `progress_summary.tex` | Formal write-up (LaTeX, paper-style) |
| `PROGRESS.md` | This file — working tracker |
| `scripts/llm_label_star.py` | Phase 2 pipeline (single-prompt default) |
| `src/fudge/data_loader.py` | Phase 3 loader (load_llm_labels, label_source) |
| `experiments/validate_discrimination.py` | Phase 1 / 4 experiment runner |
| `tests/test_llm_label_smoke.py` | Phase 2 smoke tests |
| `tests/test_label_source.py` | Phase 3 plumbing tests |
