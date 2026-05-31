# Methodology: LLM-Generated DAG Evaluation for PE Therapy on Thousand Voices of Trauma

**Version:** v0.1 (2026-05-31, TV-only pipeline, phase as primary axis)

## Research Question
Which LLM generates the highest-quality dialogue flow DAG for Prolonged Exposure (PE) therapy, evaluated against real (synthetic) therapy sessions from the Thousand Voices of Trauma (TV) dataset using FuDGE (structural) and AutoEval-ToD Domain Compliance (content) as complementary metrics?

## Dataset
- **Thousand Voices of Trauma (TV)** — 3000 synthetic PE therapy conversations, 500 simulated clients × 6 phases (P5, P6, P7, P8, P10, P11). 13 trauma types, 23 session topics.
- **No external seed scripts.** The LLM drafts the conceptual DAG from its own knowledge of PE, then refines it against TV training conversations.

## Design Decisions (locked)
- **Primary discrimination axis: phase** (perfectly balanced 500/phase; structurally meaningful — PE phases differ by design).
- **Type as secondary control variable**, not primary axis. Drop `animal attack` (n=6) and `imprisonment` (n=12); keep the 11 types with n ≥ 96.
- **Topic dropped** — too imbalanced (smallest = 6) and 21/23 topics cross-contaminate types.
- **Single-phase first** (P10 or P11) before scaling to all six phases — keeps the first end-to-end run cheap and debuggable.

---

## Methodology Flowchart

```
┌──────────────────────────────────────────────────────────────────┐
│  TODO 1: Embedding Test (Component 2)                            │
│  Pass: FuDGE viable → continue.                                  │
│  Fail: skip FuDGE, use AutoEval-ToD only.                        │
└─────────────────────────────┬────────────────────────────────────┘
                              │ pass
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 2: Loader defaults to phase                                │
│  src/fudge/data_loader.py: add task_field="phase"                │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 3: Stratified train/held-out split                         │
│  Within target phase, stratify across 11 usable trauma types.    │
│  50 train / 50 held-out per phase.                               │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 4: Prompts 1–4 (single phase)                              │
│  LLM drafts conceptual PE DAG for chosen phase (P10 first).      │
│  4-stage: draft → critique → revise → finalise.                  │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 5: Prompt 5 (merge with TV training)                       │
│  LLM refines abstract DAG using 50 TV training conversations.    │
│  Output: data/dags/<llm>/P10/dag.{mmd,json}                      │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 6: Alignment                                               │
│  Embed labels → assign training utterances to nearest node       │
│  (same actor) → centroid = mean of assigned utterances.          │
│  Output: (DialogueFlow, list[IntentBucket]).                     │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 7: FuDGE discrimination (in-phase vs out-of-phase)         │
│  Score 50 held-out P10 (positives) and 50 held-out P5/P11        │
│  (negatives) against the P10 DAG. Mann-Whitney U.                │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 8: Type as control variable                                │
│  Within phase, check FuDGE scores don't correlate with type.     │
│  Kruskal-Wallis across the 11 types.                             │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 9: AutoEval-ToD Domain Compliance (parallel content track) │
│  Convert DAG nodes to natural-language rules; LLM-judge per      │
│  session. Add Empathetic Tone (Prompt J.5) as parallel dim.      │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  TODO 10: Repeat 4–9 per candidate LLM, build comparison table   │
│  Best DAG = lowest FuDGE on in-phase + highest compliance        │
│  + largest discrimination gap.                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## TODO 1 — Embedding Test (Component 2)

**Why:** FuDGE's substitution cost compares an utterance embedding to a DAG-node-label embedding. If abstract labels (e.g. *"Therapist orients client to imaginal exposure procedure"*) and real TV utterances live in different regions of embedding space, every distance is large and FuDGE measures "abstract vs concrete language" instead of "right vs wrong intent." The alignment step's *first* assignment uses bare label embeddings before any centroid exists, so failure here corrupts every downstream centroid. This is a kill-switch test: pass → continue; fail → drop FuDGE, use AutoEval-ToD only.

**How:**
1. Pick 3 candidate PE-style node labels, e.g.:
   - *"Therapist orients client to the imaginal exposure procedure"*
   - *"Therapist elicits SUDS rating from client"*
   - *"Therapist processes thoughts and feelings after exposure"*
2. For each label, manually pick ~5 real TV utterances that clearly demonstrate that intent (positives) and ~5 unrelated utterances from different phases (negatives).
3. Embed all of them with SBERT (`sentence-transformers/all-MiniLM-L6-v2`) — same model the FuDGE pipeline uses.
4. Compute cosine distances:
   - label → positives: expect 0.1–0.3
   - label → negatives: expect 0.6–1.0
5. Pass criterion: mean(positive) + 0.2 < mean(negative) consistently across all 3 labels.

**Cost:** ~30 minutes, no API spend (SBERT is local).
**Deliverable:** `experiments/embedding_test/results.json` + 1-paragraph go/no-go note.

---

## TODO 2 — Loader default to phase

**Why:** Current `load_thousand_voices_dialogues(task_field="type")` puts the trauma category into `Conversation.task`, which is what every downstream FuDGE script groups on. Under TV-only with phase as the primary discrimination axis, that default produces the wrong grouping silently. Phase is also balanced (500/each) which removes the imbalance problems that plague type.

**How:**
1. Edit `src/fudge/data_loader.py:108` — add a `"phase"` option to `task_field`. When selected, derive `task` from the filename stem (`'P' + stem.split('_P', 1)[1]`) rather than from metadata.
2. Keep `"type"` and `"session_topic"` as options for control-variable analyses.
3. Update the loader test (`tests/test_thousand_voices_loader.py`) with a `test_task_field_phase` case.
4. Decide on the default: either keep `"type"` for backward compat or flip to `"phase"`. Recommend flipping with a one-line CHANGELOG note in the docstring.

**Deliverable:** loader patch + passing test.

---

## TODO 3 — Stratified 50/50 train/held-out split

**Why:** Random sampling from a single phase can accidentally over-represent one trauma type (e.g. P10 has 500 sessions; if 762 of 3000 are `witnessing violence`, a random draw of 50 could end up 60% witnessing-violence and underrepresent medical/loss/bullying). That would let the LLM "memorise" trauma-specific language during DAG refinement (Prompt 5) instead of learning protocol structure. The held-out test then conflates DAG quality with type-distribution match. Stratification removes this confound.

**How:**
1. Filter TV to chosen phase (e.g. P10 → 500 conversations).
2. Drop conversations whose type is `animal attack` or `imprisonment` (≤12 total in P10 worst case).
3. For each of the 11 usable types, split conversations proportionally — train and held-out each get ~50 sessions while preserving the type distribution of the phase.
4. Save the assignment to `data/splits/P10_v1.json`: `{"train": [dialogue_ids], "held_out": [dialogue_ids]}`.
5. Version the split — every downstream experiment cites it by filename.

**Deliverable:** `data/splits/P10_v1.json` + a helper `load_split(name) -> (train_convs, heldout_convs)`.

---

## TODO 4 — Prompts 1–4 (single phase)

**Why:** PE is a manualised, well-documented protocol; LLMs have substantial training-data coverage of it. A 4-stage prompt chain (draft → critique → revise → finalise) gives the LLM room to self-correct rather than committing to a first draft. Starting with one phase (P10 = Full Imaginal Exposure) caps cost and surfaces pipeline bugs cheaply before scaling to six. P10 is the most structurally distinctive phase (exposure procedure is highly templated), which gives the strongest signal for the discrimination test in TODO 7.

**How:**
1. Write four prompts (`scripts/dag_gen/prompts.py` or similar):
   - **Prompt 1 (draft):** "Generate a Mermaid DAG capturing the dialogue flow of a Prolonged Exposure therapy session for phase P10 (Full Imaginal Exposure). Nodes should be labeled with the therapist/client intent; edges represent allowed transitions."
   - **Prompt 2 (critique):** "Critique the following DAG against the PE manual (Foa et al.). Identify missing steps, incorrect ordering, and ambiguous labels."
   - **Prompt 3 (revise):** "Apply the critique as a structured diff: nodes_to_add, nodes_to_remove, edges_to_change, labels_to_clarify."
   - **Prompt 4 (finalise):** "Clean up: remove hanging nodes, merge redundant ones, verify every node has actor=agent|user, ensure root has outgoing edges."
2. Run the chain on each candidate LLM (start with one: GPT-4 or Claude Sonnet).
3. Save Mermaid + JSON outputs to `data/dags/<llm>/P10/dag.{mmd,json}`.
4. JSON schema: `{"nodes": [{"id": "B1", "actor": "agent", "label": "..."}], "edges": [{"from": "B1", "to": "U1"}]}`.

**Cost:** ~$1–2 per LLM (4 LLM calls × short context).
**Deliverable:** `data/dags/<llm>/P10/dag.json` + Mermaid render.

---

## TODO 5 — Prompt 5 (merge with TV training)

**Why:** The LLM's prior knowledge of PE is generic (textbook). TV is a synthetic dataset with its own dialogue style — phrasings, SUDS conventions, transitional language. Refining the abstract DAG with 50 concrete TV examples grounds the node labels in the dataset's distribution, reducing the abstract-label-vs-real-utterance gap that the embedding test (TODO 1) measures. Without this step the DAG is correct in theory but mis-aligned with the test distribution.

**How:**
1. Load 50 training conversations for the chosen phase via `load_split("P10_v1")["train"]`.
2. Format conversations as short transcripts (truncate to ~30 turns each if needed for context).
3. Prompt: "Given this draft PE P10 DAG and 50 real P10 transcripts, refine node labels to use phrasing that matches the transcripts, add nodes for any common dialogue moves missing from the draft, and remove nodes that never appear in the transcripts."
4. Save refined DAG over the v1 from TODO 4 → `data/dags/<llm>/P10/dag_v2.json` (keep both for ablation).

**Cost:** ~$3–5 per LLM (one call, longer context with 50 transcripts).
**Deliverable:** `data/dags/<llm>/P10/dag_v2.{mmd,json}`.

---

## TODO 6 — Alignment

**Why:** FuDGE needs each DAG node to have a *centroid embedding* in the same space as utterance embeddings. Abstract label embeddings are noisy proxies (TODO 1 gates this). Alignment replaces label embeddings with the mean embedding of real utterances assigned to that node — converting the cost function from "label-vs-utterance" to "utterance-cluster-vs-utterance," which is what FuDGE was designed for.

**How:**
1. Parse `dag_v2.json` → nodes dict + edges list (`parse_mermaid_dag()`).
2. Embed all node labels with SBERT (cached via `EmbeddingCache`).
3. For each training utterance:
   - Embed it.
   - Find the closest node by cosine distance, restricted to nodes where actor matches the utterance's actor.
   - Append the utterance text to that node's assigned list.
4. For each node: `IntentBucket.utterances = list of assigned texts`. Fallback if a node receives zero utterances → use the node label string itself (and log a warning — likely a redundant DAG node).
5. Recompute centroids as `mean(embed(u) for u in bucket.utterances)`.
6. Build `DialogueFlow` from edges; attach buckets.
7. Output: `(DialogueFlow, list[IntentBucket])` saved to `data/dags/<llm>/P10/aligned.pkl`.

**Note:** this is single-pass nearest-neighbour, not K-means. K is fixed by the DAG; no iteration.

**Deliverable:** `aligned.pkl` + a coverage report (utterances/node, % of training utterances assigned, % of nodes with zero assignments).

---

## TODO 7 — FuDGE Discrimination Test (in-phase vs out-of-phase)

**Why:** This is the validation that FuDGE works on TV clinical data — not just on STAR. If a DAG built for P10 doesn't separate held-out P10 from held-out P5/P11, then the metric isn't capturing protocol structure and downstream LLM comparison is meaningless. This mirrors the STAR in-task vs out-of-task discrimination experiment that already worked, transposed to TV's structure.

**How:**
1. Load held-out splits: 50 P10 (positives), 50 P5 (negatives), 50 P11 (negatives).
2. For each conversation: `score = fudge_efficient(conv, flow, costs) / len(conv.utterances)`.
3. Compare distributions:
   - Mean ± std for positives, P5-negatives, P11-negatives separately.
   - Mann-Whitney U: positives vs P5-negatives, positives vs P11-negatives.
   - Bootstrap 95% CI on the difference of means.
4. Pass criterion: positives mean < both negative means with p < 0.05 on both Mann-Whitney tests.
5. If positives don't separate from negatives → the DAG isn't capturing phase-specific structure; investigate the alignment coverage from TODO 6 before blaming FuDGE.

**Deliverable:** `experiments/fudge_tv/P10_discrimination.json` with score arrays, test statistics, and a violin plot.

---

## TODO 8 — Type as Control Variable

**Why:** FuDGE is supposed to measure dialogue *structure*, not trauma *content*. If scores within phase correlate with trauma type, it means the metric is picking up trauma-specific vocabulary (combat sessions use military terms, medical sessions use clinical terms) rather than protocol structure. That confound would invalidate the cross-LLM comparison because the "best" DAG might just be the one whose vocabulary matches the type distribution of the held-out set.

**How:**
1. Take the 50 held-out P10 scores from TODO 7.
2. Group by trauma type (11 usable types).
3. Kruskal-Wallis H-test across the 11 type-groups within P10. Null hypothesis: scores come from the same distribution regardless of type.
4. Pass criterion: p > 0.05 (i.e. no significant type effect within phase).
5. If p < 0.05 → investigate. Check whether one type is dominating the high-score tail; consider whether the DAG over-fit to types overrepresented in training.

**Deliverable:** `experiments/fudge_tv/P10_type_control.json` + a boxplot of scores by type.

---

## TODO 9 — AutoEval-ToD Domain Compliance (parallel track)

**Why:** FuDGE measures structural alignment — "was the sequence right?" Domain Compliance measures content adherence — "were the right things said?" These are complementary; a DAG can have the right structure but wrong content language, or vice versa. AutoEval-ToD reports 94–97% LLM-human agreement on this approach, so it's a credible second metric. It also doesn't need the alignment step (TODO 6), so it works as an independent check that doesn't share failure modes with FuDGE.

**How:**
1. Convert each DAG node to a natural-language rule:
   - Node label *"Therapist elicits SUDS rating"* → Rule *"The therapist elicited a SUDS rating from the client during the session."*
2. For each held-out conversation:
   - Prompt LLM judge (GPT-4 or Claude Sonnet) with: rule list + full transcript.
   - Output: per-rule score in {1, 0, -1} (followed / not followed / not applicable) + one-sentence reason.
3. Session score = `mean(rule_scores excluding -1)`.
4. Aggregate over held-out: in-phase mean vs out-of-phase mean.
5. Validate against ~20 manually scored sessions — target ≥ 90% agreement (paper reports 94–97%).
6. Add Empathetic Tone (Prompt J.5 from the paper) as a parallel dimension — checks therapist language for caring, non-judgmental tone.

**Cost:** ~$10–20 per LLM (150 held-out × ~20 rules each).
**Deliverable:** `experiments/compliance/P10_<llm>.json` with per-session and per-rule scores.

---

## TODO 10 — Cross-LLM Comparison

**Why:** The research question is which LLM generates the best DAG. Steps 4–9 produce per-LLM scores; this step aggregates them into a single comparison table. Two metrics (FuDGE + Compliance) reduce the risk of any single-metric artefact deciding the ranking. The discrimination gap (in-phase vs out-of-phase separation) matters as much as the raw in-phase score because a low score everywhere just means a permissive DAG, not a good one.

**How:**
1. Repeat TODOs 4–9 for each candidate LLM. Suggested initial set: GPT-4o, Claude Sonnet 4.6, Gemini 2.5 Pro, one open-weight (Llama 3.1 70B or Qwen 2.5 72B).
2. Build comparison table:

   | LLM | FuDGE in-phase | FuDGE gap (out – in) | Compliance in-phase | Compliance gap | Type control p |
   |-----|----------------|----------------------|---------------------|----------------|----------------|
   | A   | …              | …                    | …                   | …              | …              |

3. Statistical tests:
   - Pairwise Mann-Whitney U on FuDGE distributions between LLMs (Bonferroni-corrected).
   - Wilcoxon signed-rank on per-conversation paired scores (same held-out conv, different LLM DAGs).
4. Best DAG = (lowest in-phase FuDGE) ∩ (largest FuDGE gap) ∩ (highest in-phase Compliance) ∩ (no significant type effect from TODO 8). Tie-breaker: gap size.

**Deliverable:** `experiments/comparison/v1_results.md` + table + per-LLM violin plots.

---

## Versioning Convention
- **v0.x** — single-phase pilot (P10 only), single-LLM end-to-end shakeout.
- **v1.x** — multi-LLM comparison on a single phase.
- **v2.x** — scale to all six phases.
- **v3.x** — add Component 9 (TV prefix-trie validation) as a parallel track if needed.

Bump the document version when design decisions change (e.g. switching primary axis, dropping a metric, changing the LLM panel).

## Key References
- FuDGE paper: arXiv:2411.10416
- AutoEval-ToD: NAACL 2025 pp. 10133–10148 (Domain Compliance H.3, Empathetic Tone J.5)
- Foa, Hembree, Rauch, Rothbaum — *Prolonged Exposure Therapy for PTSD* (PE manual)
- TV dataset: Suhas et al., *Thousand Voices of Trauma*, NeurIPS 2025 (under review)
