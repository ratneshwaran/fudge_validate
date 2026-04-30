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

### Stage 2 ablation: chunked labeling ✅ DONE (2026-04-29)

Added `--method chunk` (chunks of 5 utterances, stride 4, last-chunk-wins
on overlap). Run with the single_prompt taxonomy on both tasks:

| Stage 2 method | hotel ratio | bank ratio | calls | total cost |
|---|---|---|---|---|
| whole | 4.60× | 5.08× | 314 | $0.83 |
| chunk (5/4) | 4.50× | 5.14× | 1171 | $1.55 |

Chunk and whole are statistically equivalent on STAR — differences are
within 1σ noise, but chunk costs ~2× more. **As expected**: STAR
dialogues are 8-15 turns, well within one-prompt capacity. The chunk
infrastructure is in place for Thousand Voices, where dialogues run
dozens to hundreds of turns and the whole-method prompt would dilute
the model's attention.

### 3-method taxonomy ablation ✅ DONE (2026-04-29)

Three taxonomy bootstrap methods compared on both tasks:

| method | hotel ratio | bank ratio | hotel labels (u/a) | bank labels (u/a) | total cost |
|---|---|---|---|---|---|
| **single_prompt** | 4.60× | 5.08× | 17 / 19 | 20 / 13 | $0.83 |
| **hybrid** | 4.22× | 4.07× | 15 / 18 | 22 / 12 | $0.85 |
| **cluster** | 5.06× | 7.70× | 122 / 17 | 225 / 12 | $1.62 |

All three STRONG PASS, all three beat the heuristic (3.00× / 3.69×).

**Cluster wins on raw discrimination but produces over-fragmented
taxonomies** (225 user labels for `bank_fraud_report`, many near-synonyms).
**single_prompt is the best parsimony/quality tradeoff** — comparable to
gold `ActionLabel` granularity at the lowest cost. **hybrid is comparable
to single_prompt on hotel but underperforms on bank** because clustering
reduces the input diversity to the naming call.

**Recommendation: use single_prompt for Thousand Voices.** Cluster path
preserved in codebase for future paper experiments. Full analysis in
`VALIDATION_REPORT.md`.

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

## Phase 4 — Re-run Table 1b with LLM labels ✅ DONE — STRONG PASS

### Labeling runs complete (2026-04-29)

Pure-LLM labeling: both user and agent utterances re-labeled by GPT-5 mini
(gold STAR `ActionLabel` is **not** used at any point — the LLM bootstraps
the agent taxonomy from utterance text alone). Single-prompt taxonomy
method, whole-dialogue labeling.

| task | dialogues | user labels | agent labels | API calls | tokens (in/out) | cost |
|---|---|---|---|---|---|---|
| `hotel_book` | 158 | 17 | 19 | 157 (5 hits) | 222k / 172k | $0.40 |
| `bank_fraud_report` | 156 | 20 | 13 | 158 | 224k / 189k | $0.43 |

bank_fraud_report had 3 off-taxonomy warnings (LLM returned a label not in
the actor's taxonomy → embedding fallback to nearest valid label). Logged
in `logs/llm_label_bank_fraud_report_whole_20260429T203015Z.jsonl`.

Taxonomies look semantically clean (e.g. hotel agent side: `greeting`,
`ask_name`, `ask_hotel_choice`, `ask_checkin_date`, `ask_checkout_date`,
`offer_booking_confirmation`, `inform_booking_success`,
`inform_no_availability`, `closing_goodbye`, ...). Comparable granularity
to STAR's gold `ActionLabel` set.

### Prompts used

Two prompts, both constrained by JSON schema:

- **Stage 1 — taxonomy bootstrap**:
  `scripts/llm_label_star.py:_single_prompt_messages` (one call per actor;
  asks for 12-30 distinct intents, snake_case labels, no slot-value
  duplicates).
- **Stage 2 — whole-dialogue labeling**:
  `scripts/llm_label_star.py:_whole_method_messages` (one call per
  dialogue; sends both taxonomies + dialogue indexed `[i] (actor) text`,
  enum-constrained label per utterance).

Literal prompt + response for any call: `logs/llm_label_*.jsonl`.

### Validation results (2026-04-29)

LLM labels **improved** discrimination over the heuristic baseline:

| task | heuristic ratio | LLM ratio | verdict |
|---|---|---|---|
| `hotel_book` | 3.00× | **4.60×** | STRONG PASS |
| `bank_fraud_report` | 3.69× | **5.08×** | STRONG PASS |

In-task means dropped (0.17 → 0.11, 0.16 → 0.12) while out-of-task means
stayed roughly flat. The LLM taxonomy carves in-task dialogues into tighter
`(actor, label)` buckets than the heuristic does. Full numbers and analysis
in `VALIDATION_REPORT.md`.

**Implication: LLM labeling is a drop-in replacement, not a fallback.**
Thousand Voices is unblocked.

---

## Phase 5 — Apply to Thousand Voices ⏳ UNBLOCKED

Phase 4 confirmed LLM labeling not only preserves but improves discrimination,
so the same pipeline transfers directly to Thousand Voices (mental health).
No gold labels needed.

Remaining dependencies:
- Thousand Voices dataset loader (not yet in repo) — needs to mirror
  `load_star_dialogues` so it produces `Conversation` objects with
  `dialogue_id`, `task`, and `utterances`. The labeling script and the
  validation script will both work as-is once that exists.

---

## Other paper threads (parallel, unblocked)

Listed in `progress_summary.tex` as next steps; independent of the labeling
pipeline:

- LLM-as-a-judge baseline on STAR (for comparison against FuDGE)
- FF1 (Flow-F1 Score) implementation
- ~~Statistical significance test (Mann–Whitney U)~~ — DONE 2026-04-30.
- ~~Bootstrap 95% CI on the discrimination ratio~~ — DONE 2026-04-30.
- ~~Held-out flow split~~ — DONE 2026-04-30 (single seed) and HARDENED
  2026-04-30 with 10-split + paired Wilcoxon after a Codex round-3
  review flagged the single-split + CI-overlap issues. **Headline
  finding: ratios collapse from 3-7.7× (in-distribution) to 1.77-2.16×
  (held-out, mean ± 0.07-0.09 across splits)**. Single_prompt LLM and
  heuristic are not significantly different on held-out (p > 0.10
  two-sided, both tasks). Cluster method is significantly *worse* than
  heuristic on held-out (p ≤ 2.8e-15) — the opposite of its
  in-distribution ranking. See VALIDATION_REPORT.md.
- Held-out *taxonomy* bootstrap split (orthogonal to the flow split —
  bootstrap taxonomy on 50% of in-task, label the other 50% with that
  taxonomy) — still deferred

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
| `experiments/significance.py` | Mann-Whitney U test across all (task × regime) cells |
