# Full Pipeline: LLM DAG Evaluation for Trauma Grounding

> **⚠ Partly superseded (2026-06-06). Source of truth: `EXPLAINER.md` §13 (live tracker) + `METHODOLOGY.md` v0.2 (locked plan).**
> This doc predates the supervisor's 2026-05-17 two-step methodology. Two framing
> changes since:
> 1. **Discrimination axis is PE *phase* (P5/P6/P7), not "grounding vs non-grounding".**
>    Components 3 + 6 + 7 below describe a grounding/non-grounding split that is no
>    longer the plan — discrimination is now in-class phase vs out-of-class phases
>    (as validated in TODO 4). Treat those component bodies as historical.
> 2. **The "embedding test" (Component 2) is no longer a gate.** It's superseded by
>    the cluster-then-recentroid step (EXPLAINER §11); kept only for reference.
> Checkboxes below are updated to actual state. The detailed step lists remain a
> useful implementation reference for Components 4/5/6/7.

## Overview
Evaluate which LLM best generates a therapeutic dialogue DAG from clinician scripts,
using real Thousand Voices of Trauma (TV) therapy sessions as test data.
Two evaluation metrics: FuDGE (structural) and AutoEval-ToD Domain Compliance (content).

---

## Component 1: STAR Validation (COMPLETE)
Validates FuDGE as a metric on task-oriented data. Does NOT validate the end goal.

- [x] Implement FuDGE naive + efficient algorithms (verified identical, 28 oracle tests)
- [x] Build 2-stage LLM labeling pipeline (GPT) for STAR user utterances
- [x] Build prefix-trie DAG from labeled STAR conversations
- [x] Run discrimination experiment (in-task vs out-of-task)
- [x] Held-out multi-split + paired Wilcoxon (LLM labels ≈ gold labels)

---

## Component 2: Embedding Test (SUPERSEDED — kept for reference)
Status: no longer a gate. The original abstract-label embedding test (in
`experiments/embedding_test/`) is replaced by cluster-then-recentroid
(EXPLAINER §11), which uses short LLM label anchors + real-utterance centroids
rather than abstract action descriptions. Retained only as v0.1 history.

Checks if FuDGE's cost function is valid for script-derived DAG nodes.
If this fails, FuDGE is not viable and Domain Compliance is the only metric.

Steps:
- [ ] Take 3 grounding node label strings (e.g. "Acknowledge body memory")
- [ ] Embed each label using SBERT (all-MiniLM-L6-v2)
- [ ] Embed semantically similar real therapy utterances
- [ ] Embed unrelated utterances (e.g. from STAR bank_fraud task)
- [ ] Compute cosine distances: similar should be 0.1–0.3, unrelated 0.6–1.0
- [ ] If gap is consistent → FuDGE viable. If not → use Domain Compliance only.

---

## Component 3: TV Session Classification (SUPERSEDED — framing changed)
Status: not the current plan. Discrimination is now in-class PE phase vs
out-of-class phases (P5/P6/P7), validated by prefix-tree DAGs in TODO 4 — not a
grounding/non-grounding split. The phase labels come from TODO 2 (per-phase
agent-turn labelling) + the TV_v1 stratified split, not from the keyword/LLM
grounding classifier below. Body retained as history.

Identifies which TV sessions used grounding (positives) vs did not (negatives).
Needed to validate that FuDGE/compliance scores separate the two groups.

Steps:
- [ ] Keyword match pass: search for grounding terms across all TV sessions
      Keywords: "look around", "feel your feet", "body memory", "that was then",
                "press your hands", "name three things", "present moment"
      Non-grounding: "SUDS", "describe the scene", "as if happening now"
- [ ] LLM classification pass: for each session ask LLM "does this contain
      trauma grounding techniques? YES/NO + one sentence evidence"
- [ ] Manual review: read ~20 sessions to validate classifier accuracy
- [ ] Output: session_id → "grounding" | "non-grounding" label file

---

## Component 4: DAG Generation Pipeline ✅ DONE (= TODO 5, P5/P6/P7)
Implemented in `scripts/generate_llm_dags.py` + `prompts.yaml`. 36 DAGs produced
(4 models × v1/v2/v3 × P5/P6/P7) under `data/dags/<model>/<variant>/<phase>/`.
Variants: v1 = prompt 1 alone; v2 = prompts 1–5 fused; v3 = prompts 1–5
sequential. Models routed via OpenRouter: `deepseek-v3.2`, `gpt-5.1`,
`gpt-oss-20b`, `kimi-k2-0905`.

Generates a candidate DAG from domain knowledge + TV training data.
Run once per LLM being compared.

Steps:
- [ ] Select 50 TV training conversations (stratified by trauma type if possible)
- [ ] Run Prompt 1: LLM generates conceptual Mermaid DAG from trauma grounding domain
- [ ] Run Prompt 2: LLM critiques the generated flow
- [ ] Run Prompt 3: LLM revises based on critique (structured diff format)
- [ ] Run Prompt 4: LLM finalises and cleans up DAG (remove hanging nodes, fix structure)
- [ ] Run Prompt 5: LLM merges Prompts 1–4 DAG with 50 TV training conversations
- [ ] Save final Mermaid output as both .mmd and .json per LLM:
      data/dags/<llm_name>/dag.mmd
      data/dags/<llm_name>/dag.json
- [ ] JSON format:
      { "nodes": [{"id": "B1", "actor": "agent", "label": "..."}],
        "edges": [{"from": "B1", "to": "U1"}] }
- [ ] Repeat for each LLM under comparison

---

## Component 5: Alignment Step
Populates DAG nodes with real TV utterances so FuDGE has meaningful embeddings.
Replaces abstract label embeddings with empirically grounded centroids.

How it works:
- Each DAG node label is embedded → initial cluster centre
- Each TV training utterance is embedded
- One-pass nearest-neighbour assignment: each utterance → closest node (same actor only)
- Node centroid = mean embedding of all assigned utterances
- This is NOT K-means — no iteration, K is fixed by DAG node count

Steps:
- [ ] Parse dag.json → nodes dict + edges list (parse_mermaid_dag())
- [ ] Embed all node labels using EmbeddingCache
- [ ] Load 50 TV training conversations via load_thousand_voices_dialogues()
- [ ] For each training utterance:
      - embed utterance text
      - find closest node by cosine distance (actor must match)
      - assign utterance text to that node
- [ ] For each node: IntentBucket.utterances = list of assigned texts
      Fallback: if no utterances assigned → use node label text itself
- [ ] Build DialogueFlow from edges
- [ ] Output: (DialogueFlow, list[IntentBucket]) ready for FuDGE
- [ ] Repeat per LLM DAG

---

## Component 6: FuDGE Scoring
Structural metric — measures sequence alignment between conversation and DAG.
Lower score = better adherence to the therapeutic flow order.

Steps:
- [ ] Load 50 held-out TV conversations (not used in training)
- [ ] For each held-out conversation:
      score = fudge_efficient(conv, flow, costs) / len(conv.utterances)
- [ ] Aggregate: mean ± std across all held-out sessions
- [ ] Aggregate separately: grounding sessions vs non-grounding sessions
- [ ] Expected: grounding sessions score lower than non-grounding sessions
- [ ] Compare across LLMs: best DAG = lowest mean on grounding sessions
      + largest separation (grounding vs non-grounding ratio)
- [ ] Repeat per LLM DAG

---

## Component 7: AutoEval-ToD Domain Compliance Scoring
Content metric — measures whether protocol steps were followed, regardless of order.
Based on AutoEval-ToD (NAACL 2025) Domain Compliance approach (Prompt H.3).
Does NOT require the alignment step. Cleaner for abstract node labels.

Steps:
- [ ] Convert DAG nodes into rule list (one rule per node, natural language)
      e.g. "Therapist acknowledged the somatic/body response before redirecting"
- [ ] For each held-out TV conversation, run LLM compliance prompt:
      - Input: rule list + full conversation
      - Output: score (1/0/-1) + reason per rule
- [ ] Session score = mean of applicable rule scores (exclude -1s)
- [ ] Aggregate across held-out sessions: mean ± std
- [ ] Aggregate separately: grounding vs non-grounding sessions
- [ ] Add empathetic tone check (Prompt J.5) as a parallel dimension
- [ ] Compare across LLMs: best DAG = highest compliance on grounding sessions
- [ ] Validate: LLM compliance scores vs manual review on ~20 sessions
      Target: >90% agreement (paper reports 94–97%)
- [ ] Repeat per LLM DAG

---

## Component 8: LLM Comparison
Ranks candidate LLMs by DAG quality using both metrics.

Steps:
- [ ] Collect FuDGE scores per LLM: mean on grounding, mean on non-grounding, ratio
- [ ] Collect Compliance scores per LLM: mean on grounding, mean on non-grounding
- [ ] Build comparison table:

  | LLM | FuDGE grounding | FuDGE ratio | Compliance grounding | Compliance ratio |
  |-----|-----------------|-------------|----------------------|------------------|
  | A   | ...             | ...         | ...                  | ...              |
  | B   | ...             | ...         | ...                  | ...              |

- [ ] Best DAG = lowest FuDGE on grounding + highest compliance on grounding
      + meaningful separation in both metrics
- [ ] Statistical test: Mann-Whitney U on FuDGE score distributions per LLM

---

## Component 9: TV Prefix-Trie Validation ✅ DONE (= TODO 4) — promoted to Step 1 gate
No longer optional: under the locked two-step methodology this IS Step 1 (the
metric-validation gate that must clear before any LLM-DAG scoring). Implemented in
`experiments/tv_prefix_tree_discrimination.py`. **Result: 3/3 phases pass**
(P5/P6/P7) at Bonferroni α = 3.3×10⁻³, ratios 1.46–1.82×. FuDGE validated on
clinical data.

Validates that FuDGE works on real clinical data at all (not just task-oriented STAR).
Parallel track — does not block Components 4–8 but strengthens the paper.

Steps:
- [ ] Run LLM labeling pipeline on TV conversations (both actor and agent — no gold labels)
- [ ] Build prefix-trie DAG from labeled TV training conversations
- [ ] Score held-out TV conversations against trie DAG
- [ ] Check discrimination by trauma type (accidents vs violence vs disasters)
- [ ] If FuDGE discriminates → metric works on clinical data
- [ ] Estimated cost: ~$3–5 in API calls

---

## Key Files
- src/fudge/           — FuDGE implementation (complete)
- experiments/         — discrimination + significance scripts
- scripts/             — LLM labeling pipeline
- data/STAR/           — STAR dataset
- data/thousand-voices-trauma/ThousandVoicesOfTrauma/ — TV dataset
- data/dags/           — DAG outputs per LLM (to be created)

## Key References
- FuDGE paper: arXiv:2411.10416
- AutoEval-ToD: NAACL 2025, pages 10133–10148
  Domain Compliance (Prompt H.3) — adapt for grounding protocol rules
  Empathetic Tone (Prompt J.5) — apply directly to therapist turns
  Human-LLM agreement: 94–97%
