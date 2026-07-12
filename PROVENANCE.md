# Result provenance — which numbers match which data

**Why this file exists:** `data/` is gitignored, so result JSONs in `experiments/` can silently
outlive the data that produced them. This file records which results correspond to which data
generation. Update it whenever data is regenerated or a result JSON is (re)produced.

## Timeline

- **2026-06 (original):** TV labels P5–P7 (+P8 partial, P10 smoke), STAR labels, split TV_v1, and
  36 LLM DAGs existed on the original (Windows) machine. All June result JSONs were produced
  against that data. That `data/` directory was subsequently **lost** (only the repo survived).
- **2026-06-28 (rebuild, this machine):** re-downloaded the raw TV dataset; **re-labelled P5/P6/P7**
  with `scripts/llm_label_tv.py` (gpt-5-mini, single_prompt, ~$4.25 total, 0 warnings);
  **regenerated the split** `data/splits/TV_v1.json` with the locked recipe (seed 20260530, 70/30,
  same drop-types) — same structure (2082/900, 347/150 per phase), but **labels are LLM-sampled and
  therefore not byte-identical to the lost originals**. LLM DAGs (`data/dags/`) and STAR artifacts
  have **not** been regenerated yet.

## Current result files (`experiments/`)

| File | Provenance | Status |
|---|---|---|
| `tv_prefix_tree_discrimination.json` | **2026-06-28 rebuild** (new labels + split) | ✅ current. Reproduces the original gate almost exactly: P5 1.67× / P6 1.83× / P7 1.47× (original: 1.67/1.82/1.46) — the pipeline is sound. |
| `tv_prefix_tree_discrimination_seg.json` | **2026-06-28 rebuild**, `--segment` run | ⚠️ current but interpret with care: TV labels are agent-only (single `_user_turn` bucket), so v1 of segmentation collapsed the entire client stream; see the caveat in the experiment docstring. |
| `star_v2_validation.json` | original (June) STAR labels | 🕰 historical record; STAR data/labels not present on this machine, not reproducible here. |

## Archived pre-rebuild results (`experiments/archive_pre_relabel/`)

`llm_dag_discrimination_{gpt-oss-20b, gpt-oss-20b_r5, deepseek-v3.2_r5}.json`,
`phase_confusion_{gpt-oss-20b, deepseek-v3.2}_v3_r5.json`, `length_matched_reanalysis.json`.

These were produced from the **old labels + the 36 lost DAGs**. Because DAG generation is
LLM-sampled (non-deterministic), they can **never be regenerated to match**. They remain valid as
the historical record behind the length-confound finding (score↔length ρ 0.89–0.99, the
length-matched re-analysis, and the withdrawal of the pilot rankings) — but they must **not** be
mixed with post-rebuild results in any comparison. `length_matched_reanalysis.py`'s defaults now
point only at current-provenance files.

## 2026-06-28 DAG pilot (post-rebuild generation)

New DAGs generated this day (OpenRouter, `gpt-oss-20b` v3 × P5/P6/P7; P7 flagged
`BAD[2alt]`) live in `data/dags/` with 100%-coverage `aligned_r5.json`. Results produced
against them + the rebuilt labels/split — comparable to each other and to the rebuilt
prefix-tree gate, **not** to anything in `archive_pre_relabel/`:
`llm_dag_discrimination_gpt-oss-20b_r5.json` (baseline), `..._r5_seg.json` (segmented),
`length_matched_pilot_2026-06-28.json` + `length_matched_reanalysis_2026-07.json`
(raw-vs-seg length re-analysis). Segmentation headline, corrected after the 2026-07
adversarial review: out-block ρ **+0.54–0.98** (not 0.92–0.98) → **−0.57…+0.72** —
`--segment` does not zero ρ, it SHIFTS the score to the DAG's granularity (2/15 seg
blocks, both vs the long P10, keep ρ ≈ +0.7 on lengths above the in-phase support, so
the matched ratio never uses them; in-phase ρ vs original length flips, e.g. P5
+0.64→−0.58). The length-matched ratio (binned on ORIGINAL length, so the shift is
controlled by construction) rises: P5 1.14→**1.59×** (95% CI [1.47, 1.73]),
P6 1.17→**1.30×** ([1.25, 1.35]), P7 ~1.06× ([1.04, 1.08], honest null).

## 2026-07-12 DAG grid (deepseek-v3.2, post-rebuild generation)

Second model, generated this day (OpenRouter, `deepseek-v3.2` × v1/v2/v3 × P5/P6/P7,
seed 20260602) in `data/dags/deepseek-v3.2/`. Raw validity was rough (only v1/P7
`ok`; the rest carried cycles/components/alternation violations), but alignment
(`--reassign-passes 5`, root-wires every in-degree-0 node, breaks cycles) yields
**97–100% coverage, ≤1 empty node/cell** — on par with the gpt-oss pilot. One cell to
watch: v2/P7 cut 12 of 53 edges (was `BAD[111cyc]`), so its post-acyclic topology is
somewhat arbitrary. Result files (baseline `--from-aligned --reassign-passes 5`,
label-fallback default), comparable to the gpt-oss pilot and the prefix-tree gate
(same rebuilt labels/split; only the DAGs differ — the intended cross-model contrast):
`llm_dag_discrimination_deepseek-v3.2_r5.json` (baseline),
`..._r5_seg.json` (segmented), `length_matched_deepseek-v3.2_2026-07-12.json`.

Finding — the confound-and-fix pattern GENERALISES to a second DAG generator, but the
discrimination magnitude is weaker than gpt-oss (n=1 generation each):
- raw in-phase ρ(score, original length) is +0.69…+0.97 on all 9 cells (confound
  reproduces); `--segment` reduces it on every cell (mean +0.90 → +0.31, sign-flips on
  both P6 cells, v2/P6 → +0.03);
- length-matched ratio rises under segmentation in most cells but stays modest: P5
  1.09–1.15×, P6 1.12–**1.41×** (v2/P6 the standout, CI [1.37, 1.46]), P7 1.00–1.11×.
  cf. gpt-oss v3 seg lm P5 **1.59×** / P6 **1.30×** — deepseek DAGs discriminate less.
- P7 stays confound-heavy even segmented (seg in-phase ρ +0.60…+0.74) in both models.

Caveats: out-of-phase coverage 42–64% (common-support drops 36–58% of negatives; P5
cells are the most restricted, 42% / 4 bins); n=1 generation/cell (no generation error
bars — next-step 4); direct gpt-oss-vs-deepseek magnitude is single-sample-vs-single-sample.

## Rules of thumb

1. A result JSON is only comparable to another if both are on the same side of the 2026-06-28 line.
2. When DAGs are regenerated (OpenRouter), new `llm_dag_discrimination_*` results start a fresh
   provenance generation — add a dated entry here and never compare across it.
3. Always record per-conversation scores in result JSONs (it's what made the June re-analysis
   possible without re-scoring).
