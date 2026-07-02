# Methodology: LLM-Generated DAG Evaluation for PE Therapy on Thousand Voices of Trauma

**Version:** v0.2 (2026-05-31)
**Supersedes:** v0.1 (TV-only single-track plan, since reframed into a two-step structure by supervisor on 2026-05-17)
**Hard delivery target:** 2026-05-30

## Research Question
Which LLM generates the highest-quality dialogue flow DAG for Prolonged Exposure (PE) therapy, evaluated against real (synthetic) therapy sessions from the Thousand Voices of Trauma (TV) dataset?
Answering this requires first validating that the chosen metric (FuDGE) actually discriminates structure on mental-health dialogue at all — a separate research sub-question that gates everything else.

## Datasets
- **STAR** — task-oriented dialogue corpus with gold agent intent labels; used in original FuDGE paper.
- **Thousand Voices of Trauma (TV)** — 3000 synthetic PE therapy conversations, 500 clients × 6 phases (P5, P6, P7, P8, P10, P11), 13 trauma types, 23 session topics. No labels.

## Two-Step Research Structure (load-bearing decision, supervisor-locked)

**Step 1 — Validate the FuDGE metric.**
Use prefix-tree DAGs as the reference construction. If FuDGE doesn't separate in-class from out-of-class held-out conversations using prefix-tree DAGs, the metric isn't trustworthy and no comparison of LLM DAGs is meaningful.

**Step 2 — Evaluate LLM DAG-generation methods.**
Compare prompt variants × LLM panel. For each LLM DAG, populate intent buckets via cluster-then-recentroid, then run FuDGE. AutoEval-ToD Domain Compliance runs in parallel as a second metric.

**Critical constraint (no circular validation):** prefix-tree is the *validation reference only* — it is NOT a candidate DAG-generation method being compared against LLM DAGs. Using the same construction as both reference and baseline destroys the comparison.

## Design Decisions (locked by supervisor, 2026-05-17)
- **70/30 stratified train/test split** (not 50/50), stratified per condition (STAR task / TV trauma type or phase).
- **Label only agent turns.** User turns are uncontrollable — dialogue flow is agent-driven (call-centre model). Halves labelling cost.
- **Drop the charity grounding scripts entirely.** Prompt 5 examples = ~50 randomly sampled TV training conversations instead.
- **Cross-LLM label-vocabulary canonicalisation:** K-means in label-embedding space (not utterance space — LLM label space is more structured); K chosen by silhouette score or inertia elbow, sweep K = 2..50–60.
- **AutoEval-ToD as second metric** (parallel track). See `[[reference-autoeval-tod]]` memory.
- **Empathy/tone metrics out of scope** for this paper.

---

## Methodology Flowchart

```
                          ┌────────────────────────────────────┐
                          │              STEP 1                │
                          │      Validate FuDGE Metric         │
                          └────────────────┬───────────────────┘
                                           ▼
              ┌───────────────────────┐         ┌───────────────────────┐
              │ TODO 1: STAR re-val   │         │ TODO 2: TV label      │
              │ - defensible prefix   │         │   pipeline (agent     │
              │   tree                │         │   turns only)         │
              │ - 70/30 stratified    │         └─────────┬─────────────┘
              │ - per-task DAGs       │                   ▼
              │ - in/out distribution │         ┌───────────────────────┐
              │ - Mann-Whitney U      │         │ TODO 3: 70/30 split   │
              └─────────┬─────────────┘         │   per TV condition    │
                        │                       └─────────┬─────────────┘
                        │                                 ▼
                        │                       ┌───────────────────────┐
                        │                       │ TODO 4: TV prefix-tree│
                        │                       │   DAG per condition + │
                        │                       │   FuDGE discrimination│
                        │                       └─────────┬─────────────┘
                        └─────────────────┬───────────────┘
                                          ▼
                          ┌────────────────────────────────────┐
                          │   GATE: both pass → Step 2         │
                          │   either fails → diagnose, no go   │
                          └────────────────┬───────────────────┘
                                           ▼
                          ┌────────────────────────────────────┐
                          │              STEP 2                │
                          │   Evaluate LLM DAG generation      │
                          └────────────────┬───────────────────┘
                                           ▼
              ┌───────────────────────────────────────────────────────┐
              │ TODO 5: Prompt variants × LLM panel                   │
              │   - Prompt 1 alone                                    │
              │   - Prompts 1–5 fused                                 │
              │   - Prompts 1–5 sequential                            │
              │   × {GPT-4o, Sonnet 4.6, Gemini 2.5, Llama 3.1 70B}   │
              │     (3–4 OSS + 2 commercial)                          │
              └─────────────────────────┬─────────────────────────────┘
                                        ▼
              ┌───────────────────────────────────────────────────────┐
              │ TODO 6: Canonicalise labels across LLMs               │
              │   K-means in label-embedding space (silhouette/elbow) │
              └─────────────────────────┬─────────────────────────────┘
                                        ▼
              ┌───────────────────────────────────────────────────────┐
              │ TODO 7: Cluster-then-recentroid bucket population     │
              │   for each LLM DAG                                    │
              └─────────────────────────┬─────────────────────────────┘
                                        ▼
              ┌────────────────────────┐         ┌─────────────────────────────┐
              │ TODO 8: FuDGE per DAG  │         │ TODO 9: AutoEval-ToD        │
              │  in-class vs out       │         │   Domain Compliance per DAG │
              └────────────┬───────────┘         └──────────────┬──────────────┘
                           └────────────┬──────────────────────┘
                                        ▼
              ┌───────────────────────────────────────────────────────┐
              │ TODO 10: Cross-LLM comparison table                   │
              └───────────────────────────────────────────────────────┘
              ┌───────────────────────────────────────────────────────┐
              │ TODO 11 (optional, paper polish):                     │
              │   Clinician pairwise comparison on FuDGE / AutoEval   │
              │   disagreement cases                                  │
              └───────────────────────────────────────────────────────┘
```

---

# STEP 1 — Validate FuDGE Metric

## TODO 1 — STAR re-validation with a defensible prefix-tree method

**Why:** The previous STAR validation used a prefix-tree construction the user found online, with no strong methodological justification. Supervisor wants this strengthened so the metric-validation argument is reproducible and citable. Also flips the split from the earlier 50/50 to **70/30 stratified** to match TV.

**How:**
1. Document the prefix-tree construction explicitly: trie over `(actor, label)` sequences, with each node's `IntentBucket.utterances` = the set of observed utterance texts at that trie node.
2. Re-implement or audit `build_flow_from_conversations` in `src/fudge/data_loader.py` against the documented spec; remove any silent fallbacks.
3. 70/30 stratified split per task (hotel_book, bank_fraud_report, etc.). Save split assignment to `data/splits/STAR_v2.json`.
4. Build per-task prefix-tree DAGs from training. Compute centroids from training utterances.
5. Score held-out: each test conversation against (a) its in-task DAG, (b) all out-of-task DAGs. Aggregate scores.
6. Test: Mann-Whitney U on in-task vs out-of-task distributions per task. Pass = significant separation (p < 0.01 after Bonferroni across tasks).
7. Bootstrap 95% CI on the gap (mean(out) − mean(in)) per task.

**Deliverable:** `experiments/star_v2_validation.json` + table + violin plot per task.

---

## TODO 2 — TV agent-turn labelling pipeline

**Why:** TV has no labels. Prefix-tree construction needs labels. Supervisor specified: **label only agent turns** (user turns are uncontrollable; dialogue flow is agent-driven). This halves cost and avoids the noisy user-label problem.

**How:**
1. Reuse `scripts/llm_label_star.py` infrastructure; adapt the prompt to agent-turn-only labelling.
2. Bootstrap a TV-specific taxonomy from a single LLM call over a small sample (~20 conversations across phases), then label all 3000 conversations against that taxonomy.
3. Save per-conversation labels to `data/TV_llm_labels/<conversation_id>.json` with the same hash-versioned schema as STAR labels.
4. Validate on a 20-conversation hand-checked subset — target ≥ 85% agreement on agent turns.

**Cost:** ~$5–10 in API spend.
**Deliverable:** labelled TV corpus + taxonomy + agreement report.

---

## TODO 3 — TV 70/30 stratified split per condition

**Why:** Random sampling from a phase or trauma type can over-represent a single subgroup, conflating DAG quality with subgroup-distribution match. Stratification controls this. Per-condition splits enable per-condition prefix-tree DAGs (which TODO 4 needs).

**How:**
1. Decide stratification key: **phase** (P5..P11 — 500 each, balanced) is the primary axis per v0.1 analysis. Type as secondary stratifier within phase.
2. Drop trauma types with n < 96 (animal attack, imprisonment).
3. 70/30 stratified split: each phase × type cell split proportionally.
4. Save to `data/splits/TV_v1.json`: `{"phase": {"P10": {"train": [...], "test": [...]}, ...}}`.
5. Helper `load_split(name, phase) -> (train_convs, test_convs)`.

**Deliverable:** `data/splits/TV_v1.json` + loader helper.

---

## TODO 4 — TV prefix-tree DAG per condition + FuDGE discrimination

**Why:** The gate test for whether FuDGE works on mental-health dialogue at all. Per-condition prefix-tree DAGs are the reference; if FuDGE doesn't discriminate in-condition from out-of-condition held-out conversations, the metric isn't fit for TV and we cannot proceed to Step 2.

**How:**
1. For each condition (e.g. each phase, or each phase × type cell with sufficient n):
   - Build prefix-tree DAG from training labels (`build_flow_from_conversations` with `label_source=TV_labels`).
   - Compute centroids per node from training utterances.
2. Score held-out: each test conversation against (a) its in-condition DAG, (b) all other-condition DAGs.
3. Mann-Whitney U on in vs out distributions per condition; bootstrap CI on the gap.
4. Pass criterion: significant separation (p < 0.01 after correction) on majority of conditions; ratio mean(out)/mean(in) ≥ 1.3 as effect-size threshold.
5. If fails: diagnose (label quality? trie too deep/shallow? embedding model wrong? phase confound?) before falling back to AutoEval-ToD-only.

**Deliverable:** `experiments/tv_prefix_tree_discrimination.json` + per-condition violin plots + go/no-go note for Step 2.

---

# STEP 2 — Evaluate LLM DAG-Generation Methods

> **Do not start Step 2 until both TODO 1 and TODO 4 pass.**

## TODO 5 — Generate DAGs: prompt variants × LLM panel

**Why:** The research question is which LLM generates the best PE DAG, *and* whether prompt structure matters. Three prompt variants × multiple LLMs lets us decompose those two factors. Open-source + commercial mix improves the paper's generalisability claim.

**How:**
1. **Prompt variants:**
   - **V1:** Prompt 1 only (single shot — "generate a PE DAG").
   - **V2:** Prompts 1–5 fused into a single prompt (all instructions in one call).
   - **V3:** Prompts 1–5 sequential (draft → critique → revise → finalise → merge-with-50-TV-examples).
2. **LLM panel:** GPT-4o, Claude Sonnet 4.6, Gemini 2.5 Pro (commercial); Llama 3.1 70B + Qwen 2.5 72B (open-source). Total 5 LLMs × 3 variants = 15 DAGs per phase.
   > **[2026-06 update — superseded]** The panel actually used (locked 2026-06-02, option C) is
   > **deepseek-v3.2, kimi-k2-0905, gpt-oss-20b, gpt-5.1 via OpenRouter** — see
   > `scripts/generate_llm_dags.py` `MODEL_REGISTRY`. Claude models are reserved for the
   > LLM judge (`scripts/llm_judge.py`), keeping judge ≠ generator.
3. Prompt 5 examples: 50 conversations randomly sampled from the TV training split (TODO 3).
4. Output schema: `{"nodes": [{"id", "actor", "label"}], "edges": [{"from", "to"}]}` saved to `data/dags/<llm>/<variant>/<phase>/dag.json` + Mermaid.

**Cost:** ~$30–80 across the panel.
**Deliverable:** 15 DAGs per chosen phase (start with P10).
> **[2026-06 update — superseded]** Scope is **P5/P6/P7** (the fully labelled phases), not P10.

---

## TODO 6 — Canonicalise labels across LLMs

**Why:** Different LLMs will emit semantically equivalent but textually different labels ("get name", "take name", "ask name"). Without canonicalisation, the cross-LLM comparison conflates label-string variation with structural variation. Supervisor's specific recommendation: cluster in **label-embedding space** (more structured than utterance space because LLM-generated labels tend to be short, intent-shaped phrases).

**How:**
1. Collect all unique labels from all 15 DAGs (TODO 5).
2. Embed each with SBERT.
3. K-means clustering: sweep K = 2..60. Compute silhouette score + inertia for each K.
4. Pick K via (a) silhouette maximum, (b) inertia elbow. Report both; pick the smaller K if they disagree.
5. Canonical label per cluster = most-frequent original label in that cluster.
6. Build `data/dags/label_canonicalisation_v1.json`: `{original_label: canonical_label}`.
7. Apply mapping to all 15 DAGs (rewrite node labels).

**Deliverable:** canonicalisation map + relabelled DAGs.

---

## TODO 7 — Cluster-then-recentroid bucket population

**Why:** LLM DAGs ship with abstract node labels and no real-utterance content. FuDGE needs each node's `IntentBucket.utterances` populated with real text to compute meaningful centroids. The supervisor's prescribed method: embed each training utterance, find closest LLM-label-anchor by cosine, assign to that bucket, then **replace the label embedding with the mean of assigned utterances** as the new centre.

This is novel (not exactly STAR Path 1 nor STAR Path 2) and is a methodological contribution of the project. Faithful to FuDGE's cost-function semantics (bucket centre = real-utterance centroid) while solving the cold-start problem (no utterances yet assigned when the DAG arrives from the LLM).

**How:**
1. For each canonicalised LLM DAG:
   - Embed each (canonical) node label with SBERT — initial anchor.
   - For each training utterance (same actor only):
     - Embed it.
     - Find nearest node anchor by cosine.
     - Append text to that node's `bucket.utterances`.
   - For each node: compute new centroid = mean of `bucket.utterances` embeddings. Replace anchor.
   - Fallback: if a node receives zero utterances → keep label embedding as centroid + log a warning (likely a DAG-redundancy issue).
2. Persist populated DAGs to `data/dags/<llm>/<variant>/<phase>/aligned.pkl`.
3. Coverage report: utterances-per-node distribution, zero-utterance node count, mean-cosine-to-centroid.

**Risk to flag:** mis-assignments stick (one-pass NN, no iteration). The embedding test (v0.1 TODO 1) showed SBERT cannot bridge abstract-label-to-utterance gaps cleanly, which means initial assignments may be noisy. Mitigations: (a) use the canonicalised LLM-output labels rather than re-paraphrasing them, since the K-means canonicalisation step already encodes some structure; (b) consider a second pass that re-assigns based on new centroids if results look poor.

**Deliverable:** populated DAGs + coverage report per LLM × variant × phase.

---

## TODO 8 — FuDGE on each LLM DAG: in-class vs out-of-class

**Why:** The structural metric. Mirrors TODO 4 but with LLM DAGs in place of prefix-tree DAGs. Same discrimination logic: a good LLM DAG for P10 should score P10 held-out lower than P5/P11 held-out.

**How:**
1. Load held-out P10, P5, P11 from the TV split (TODO 3).
2. For each LLM × variant DAG (15 total per phase):
   - Score each held-out conversation: `score = fudge_efficient(conv, flow, costs) / len(conv.utterances)`.
   - Aggregate: mean ± std, in-class (P10) vs out-of-class (P5, P11).
   - Mann-Whitney U; bootstrap gap CI.
3. Per-LLM-per-variant table of in-class mean, out-of-class mean, gap, p-value.

**Deliverable:** `experiments/llm_dag_fudge.json` + comparison plots.

---

## TODO 9 — AutoEval-ToD Domain Compliance (parallel content track)

**Why:** Independent second metric. Doesn't share failure modes with FuDGE (no embedding, no alignment). 94–97% LLM-human agreement reported in the NAACL 2025 paper. Frames the contribution as "two state-of-the-art DAG evaluation methods adapted for mental health."

**How:**
1. Convert each canonicalised DAG node into a natural-language rule (e.g. `elicit_suds_rating` → *"The therapist elicited a SUDS rating from the client during the session."*).
2. For each held-out conversation:
   - Prompt judge LLM (GPT-4 or Sonnet) with rule list + transcript.
   - Output: per-rule score {1, 0, -1} + one-line reason.
3. Session score = `mean(rule_scores excluding -1)`.
4. Aggregate over held-out by in-class vs out-of-class; report mean ± std + gap.
5. Validate on ~20 hand-scored sessions; target ≥ 90% agreement.

**Cost:** ~$15–30 per LLM × phase.
**Deliverable:** `experiments/llm_dag_compliance.json`.

---

## TODO 10 — Cross-LLM comparison table

**Why:** This is the actual research output. Two metrics + multiple LLMs + multiple prompt variants requires a structured comparison so the conclusion ("which LLM × variant is best") is defensible.

**How:**
1. Build the master table:

   | LLM | Variant | FuDGE in-class | FuDGE gap | Compliance in-class | Compliance gap |
   |-----|---------|----------------|-----------|---------------------|----------------|
   | …   | V1      | …              | …         | …                   | …              |
   | …   | V2      | …              | …         | …                   | …              |
   | …   | V3      | …              | …         | …                   | …              |

2. Pairwise statistical tests:
   - Mann-Whitney U on FuDGE distributions across LLMs (Bonferroni-corrected).
   - Wilcoxon signed-rank on per-conversation paired scores (same conv, different DAGs).
3. Best DAG = lowest in-class FuDGE ∩ largest FuDGE gap ∩ highest in-class Compliance ∩ largest Compliance gap. Tie-break: gap size.

**Deliverable:** `experiments/comparison/v1_results.md` with table, plots, narrative.

---

## TODO 11 (optional, paper polish) — Clinician pairwise comparison on metric disagreements

**Why:** Where FuDGE and AutoEval-ToD disagree on a DAG's quality, asking a clinician to choose between the two is more reliable than 0–5 scoring (supervisor's point: 0–5 is hard for humans, pairwise is easy). Correlating clinician picks with each metric tells us which metric tracks clinical judgement better. This strengthens the methodology section without adding much cost.

**How:**
1. Identify N=20 disagreement cases (FuDGE ranks A > B; AutoEval-ToD ranks B > A).
2. Show Francesca pairs of (DAG, transcript): "Which DAG is closer to this conversation's flow?"
3. Compute Cohen's kappa with each metric's preference.
4. Report.

**Deliverable:** clinician-validation paragraph + kappa values.

---

## Versioning Convention
- **v0.x** — single-phase pilot, single track.
- **v1.x** — two-step structure, Step 1 done, Step 2 in progress on one phase.
- **v2.x** — Step 2 across all six phases.
- **v3.x** — long-term hierarchical-clustering DAG generation (Welcome project track; out of scope for this paper).

Bump on design changes. Current = v0.2: two-step structure locked, supervisor decisions integrated.

## Key References
- FuDGE: arXiv:2411.10416
- AutoEval-ToD: NAACL 2025 HLT pp. 10133–10148 — Domain Compliance (H.3), Empathetic Tone (J.5). 94–97% LLM-human agreement.
- Foa, Hembree, Rauch, Rothbaum — PE manual.
- TV: Suhas et al., NeurIPS 2025 (under review).

## Out-of-scope (don't drift in)
- Empathy/tone metrics (parked for next paper).
- Hierarchical-clustering DAG generation (supervisor's Welcome project — bigger publication later).
- Francesca's charity grounding scripts (dropped; TV is the only data source now).
- LLM-judge DAG evaluation that ALSO uses the LLM that generated the DAG (would be circular — judge must be different from generator).
