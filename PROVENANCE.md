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

## Rules of thumb

1. A result JSON is only comparable to another if both are on the same side of the 2026-06-28 line.
2. When DAGs are regenerated (OpenRouter), new `llm_dag_discrimination_*` results start a fresh
   provenance generation — add a dated entry here and never compare across it.
3. Always record per-conversation scores in result JSONs (it's what made the June re-analysis
   possible without re-scoring).
