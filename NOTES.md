# Project Summary

## Goal
Evaluate which LLM best generates a therapeutic dialogue DAG from real therapy conversations (Thousand Voices of Trauma), using FuDGE as the evaluation metric.

---

## What FuDGE Does
FuDGE computes the minimum-cost edit distance between a real conversation and a dialogue flow DAG. Three operations:
- **Substitution** — utterance matches a DAG node imperfectly; cost = cosine distance (0–1)
- **Deletion** — a DAG node has no matching utterance (step was skipped); cost = 1.0
- **Insertion** — an utterance matches no DAG node (off-script); cost = 1.0

Lower score = better alignment with the flow. Score is normalised by conversation length.

---

## What Was Done (STAR Validation)
- Implemented FuDGE (naive + efficient algorithms), verified identical on 28 oracle tests
- Built LLM intent labeling pipeline (GPT, 2-stage) to label **user** utterances in STAR — needed because STAR has gold agent labels but no user labels, and the prefix-trie DAG requires labels for all turns
- Built prefix-trie DAG from labeled conversations; ran discrimination experiment (in-task vs out-of-task)
- Held-out multi-split + paired Wilcoxon showed LLM labels ≈ gold labels on held-out data
- **Key finding**: this validated FuDGE as a metric but does not validate the end goal — the STAR flow is data-derived, the end goal flow is LLM-generated from domain knowledge

---

## Actual Pipeline (End Goal)

```
Prompts 1–4: LLM generates conceptual DAG from trauma grounding domain knowledge
        ↓
Prompt 5: merge LLM DAG + 50 TV training conversations → Mermaid DAG
(the 4 grounding example scripts were a proxy for TV before TV was available)
        ↓
Alignment step: assign TV training utterances to closest DAG node via cosine similarity
→ each node gets a cluster of real utterances (IntentBucket)
→ centroid = mean embedding of assigned utterances
        ↓
Build FuDGE IntentBuckets + DialogueFlow from aligned DAG
        ↓
FuDGE scores 50 held-out TV conversations against the DAG
        ↓
Compare across LLMs: best DAG = lowest mean FuDGE on held-out + interpretable structure
```

---

## Why the Alignment Step Is Needed
Prompt 5 outputs a Mermaid DAG with abstract node labels (e.g. "Acknowledge body memory"). FuDGE needs actual utterance texts per node to compute embedding centroids. Without alignment, the cost function compares real utterances against abstract label embeddings — a representation space mismatch. Alignment fixes this by populating each node with real TV utterances before scoring.

---

## LLM-as-Judge (Alternative/Complement)
Does not need the alignment step. Give the LLM the DAG + a real conversation and ask if it follows the flow. More accurate at matching abstract labels to real utterances, but expensive per call and non-deterministic. FuDGE is better as a fast, scalable metric once the DAG is validated.

### AutoEval-ToD (NAACL 2025) — key reference
Paper proposes automated LLM-based evaluation of task-oriented dialogue across 5 dimensions. Most relevant to this project:

**Domain Compliance (Prompt H.3)**: define protocol steps as rules, feed into LLM with conversation, get 0/1/-1 score per rule. Adapt directly by replacing domain rules with DAG nodes:
```
Rule 1: Therapist acknowledged the body memory before redirecting to present
Rule 2: Client was asked to name objects in current environment
Rule 3: Temporal reorientation was used
Rule 4: Therapist did not skip sensory grounding before affirmation
```
Output: per-rule scores + reasons. Average = session adherence score. No alignment step needed.

**Empathetic Tone (Prompt J.5)**: checks caring, supportive, non-judgmental language — directly applicable to trauma grounding.

**Validation**: 94–97% LLM-human agreement across tasks — strong evidence LLM-as-judge is reliable without constant human re-validation.

### Combined evaluation approach
| Method | What it measures |
|---|---|
| FuDGE | Structural alignment — was the sequence right? |
| Domain Compliance (AutoEval-ToD style) | Content adherence — were the right things said? |
| Empathetic Tone | Clinical appropriateness — was the tone therapeutic? |

---

## What Is Still Needed
1. **Embedding test** — does script/label language embed close enough to real utterance language?
2. **TV prefix-trie validation** — does FuDGE work on real clinical conversation data (not just task-oriented STAR)?
3. **Run the full pipeline** — Prompts 1–5 on 50 TV train, align, score 50 TV held-out
4. **Compare LLMs** — repeat pipeline with different LLMs, compare mean FuDGE on held-out
