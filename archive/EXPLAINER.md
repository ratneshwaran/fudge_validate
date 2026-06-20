# What This Project Is Actually Doing

A plain-language walkthrough of the fudge-validate project. Read this before re-opening `METHODOLOGY.md` if you've been away from the code for a while. The methodology doc is the **plan**; this doc is the **understanding**.

---

## 1. The research question (one sentence)

> **Which LLM generates the highest-quality dialogue flow DAG for Prolonged Exposure (PE) therapy, when evaluated against real PE conversations using FuDGE as the structural metric and AutoEval-ToD as the content metric?**

The output of this project is a ranking of LLMs (GPT-4o, Claude Sonnet, Gemini, etc.) by how well each one can write down the structure of a PE therapy session as a graph.

## 2. What's a "dialogue flow DAG"?

A directed acyclic graph (DAG) where:
- **Each node** = a therapist intent (e.g. *"ask SUDS rating"*, *"prompt continue narrative"*, *"validate and reassure"*)
- **Each edge** = an allowed transition (e.g. after asking SUDS, the therapist either continues the narrative or pauses for grounding)
- **Each node holds an "intent bucket"** = a list of real utterance texts that demonstrate that intent

The DAG is the therapist's protocol made explicit. A good DAG for PE phase P10 ("Full Imaginal Exposure") would capture moves like *start the exposure → ask for sensory details → check SUDS → instruct grounding if too high → continue narrative → repeat → debrief at end*.

## 3. What FuDGE measures

FuDGE = **Fu**zzy **D**ialogue-**G**raph **E**dit distance.

Given a real conversation (sequence of utterances) and a DAG, FuDGE computes the minimum-cost edit distance between them. Three operations, each with a cost:

| Operation | Meaning | Cost |
|---|---|---|
| **Substitution** | Utterance gets matched to a DAG node | cosine distance between the utterance embedding and the node's bucket centroid (0 = perfect fit, ~1 = wrong intent) |
| **Deletion** | DAG node is skipped (therapist didn't do that step) | 1.0 |
| **Insertion** | Utterance has no DAG node (off-script) | 1.0 |

**Lower FuDGE score = conversation aligns better with the DAG.**

### What FuDGE measures

Structural alignment. Did the right things happen in the right order?

### What FuDGE does NOT measure

- Empathy or emotional tone
- Whether the therapist was warm or cold
- Whether the content was clinically appropriate
- Whether the conversation actually helped the client

(For those, you need a second metric. That's what AutoEval-ToD is for — see §10.)

## 4. The critical thing I got wrong at first

I initially thought FuDGE compared *node label strings* (like `"ask_suds"`) against utterance embeddings. **It does not.**

Each DAG node has `IntentBucket.utterances` — a list of real utterance texts. The node's centroid is computed by embedding those texts and averaging. The label string is just a name for humans; it's never embedded into FuDGE's cost function.

This matters because **the LLM-generated DAGs only ship with labels, not utterances**. So we need an extra step ("cluster-then-recentroid", §11) to populate the buckets with real text before FuDGE can score anything.

## 5. Why a two-step research methodology

You can't evaluate LLM DAGs with FuDGE unless you first prove FuDGE works on mental-health dialogue. So:

### Step 1 — Validate the metric

Build "ground-truth" DAGs by a deterministic, defensible construction (a prefix-tree built from labelled training conversations). Run FuDGE on these. If FuDGE separates "this conversation belongs to phase X" from "this conversation belongs to phase Y" cleanly, the metric is trustworthy.

### Step 2 — Use the validated metric

Generate DAGs from prompts (one-shot, fused, sequential) across multiple LLMs. Score each LLM's DAG with FuDGE on the same in-vs-out task. Best LLM = sharpest discrimination.

### Why this order is non-negotiable (the no-circular-validation rule)

If you skipped Step 1 and a prefix-tree DAG outscored an LLM DAG, you wouldn't know if the LLM is bad or if the metric is. Worse: if **prefix-tree is both the validation reference AND a "method" you compare against the LLMs**, the comparison is circular by construction — your measuring stick is defined to favour your baseline. Your supervisor was emphatic about this in the 2026-05-17 meeting and it's now a locked feedback memory.

## 6. What a "prefix-tree DAG" actually looks like

Take labelled conversations. Each one is a sequence of `(actor, label)` pairs:

```
Conversation 1:  greet → ask_suds → instruct_breathing → ask_suds → continue → end
Conversation 2:  greet → ask_suds → continue → ask_suds → continue → end
Conversation 3:  greet → prompt_start → ask_sensory → continue → end
```

Build a tree where shared prefixes are merged:

```
                          ┌─ instruct_breathing ─ ask_suds ─ continue ─ end
            ┌─ ask_suds ──┤
greet ──────┤             └─ continue ─ ask_suds ─ continue ─ end
            │
            └─ prompt_start ─ ask_sensory ─ continue ─ end
```

Each node:
- gets a label from the input sequence
- collects every observed utterance text at that position into its bucket

It's mechanical, fully deterministic, and entirely data-driven. Perfect as a reference DAG for validation — you didn't invent it; you read it off the data.

## 7. What we did in Step 1

### Step 1a — STAR re-validation (TODO 1) ✓ DONE

STAR is a task-oriented dialogue corpus (hotel booking, bank fraud reports, weather, doctor scheduling, etc.) with **gold agent intent labels**. It's where the original FuDGE paper validated the metric.

What we did:
1. Created a versioned **70/30 stratified split per task** (so every task has a fair train/test cut). Saved to `data/splits/STAR_v2.json`.
2. For each task: built a prefix-tree DAG from its 70% training conversations.
3. Scored: that task's 30% test conversations (**positives**) vs every other task's test conversations (**negatives**). Test data is never reused on either side.
4. Mann-Whitney U test per task; Bonferroni correction across 23 tasks; bootstrap CI on the in-vs-out gap.

**Result: 22 of 23 tasks pass at Bonferroni α = 4.3×10⁻⁴ with gap > 0.** Mean ratio ≈ 1.85×. The one fail (`hotel_service_request`, ratio 1.10×) has only 30 training conversations — a tiny DAG with too few intents to discriminate sharply.

**Meaning**: the metric is fit for task-oriented dialogue under the most defensible methodology we can build (proper held-out, stratified split, multiple-comparison correction). The original FuDGE claim re-validated with stricter rigour.

### Step 1b — TV labelling pipeline (TODO 2) ⚠ PARTIAL

Thousand Voices of Trauma (TV) is a corpus of 3000 synthetic PE therapy conversations — 500 simulated clients × 6 PE phases (P5 orientation, P6 SUDS monitoring, P7 reinforcing exposure, P8 eliciting thoughts, P10 full exposure, P11 processing). **No labels.**

To build a prefix-tree, we need labels. So we ran an LLM (gpt-5-mini) over TV with two important design choices:

1. **Per-phase taxonomy.** Each phase gets its own agent-intent vocabulary because PE protocol moves differ across stages. A unified taxonomy would collapse phase-specific moves and destroy the cross-phase signal.
2. **Label only Therapist (agent) turns.** Client (user) turns get a sentinel `_user_turn`. Supervisor's call: dialogue flow is agent-driven (call-centre model). User turns are uncontrollable and shouldn't shape the DAG.

What landed:
- P5, P6, P7: **500/500 dialogues labelled each** ✓
- P8: 266/500 partial
- P10: 3/500 (smoke test only)
- P11: not started

Cause of partial completion: OpenAI quota ran out mid-P8. The `single_prompt` taxonomy bootstrap is expensive (sends all unique agent utterances in one call). Cached calls cost $0 on retry, so finishing the remaining ~1230 dialogues is cheap whenever you top up.

Example P10 taxonomy the pipeline discovered:
```
greet_and_check_in, ask_suds, instruct_breathing_and_grounding,
prompt_start_exposure, prompt_continue_narrative, ask_sensory_details,
ask_body_reaction, validate_encourage_reassure, ask_coping_and_followup,
debrief_and_reflect, offer_pause_or_choice, wrap_up_and_close
```
Clinically coherent — these are textbook PE phase-P10 moves.

### Step 1c — TV 70/30 stratified split (TODO 3) ✓ DONE

Same pattern as STAR. Stratum = phase. Within each phase, also balanced by trauma type. Dropped two tiny trauma types (animal_attack n=6, imprisonment n=12). Result: 6 phases × 347 train + 150 test = 2082 train / 900 test. Saved to `data/splits/TV_v1.json`.

### Step 1d — TV prefix-tree discrimination (TODO 4) ✓ DONE (on labelled phases)

For each labelled phase (P5, P6, P7):
- Built a prefix-tree DAG from the 347 training conversations.
- Scored: phase's 150 test conversations (positives) vs every other phase's test conversations pooled (750 negatives — includes P8/P10/P11 test sets, which don't need labels because they're only being scored, not used to build a DAG).

**Result: 3 of 3 evaluated phases pass at Bonferroni α = 3.3×10⁻³.**

| Phase | DAG nodes | gap | ratio (out/in) | p |
|---|---|---|---|---|
| P5 (Orientation) | 3735 | +0.172 | 1.67× | 9.3×10⁻⁸³ |
| P6 (SUDS) | 6552 | +0.194 | 1.82× | 4.1×10⁻⁸³ |
| P7 (Reinforcing) | 6867 | +0.116 | 1.46× | 2.0×10⁻⁸³ |

**Meaning: FuDGE is validated on mental-health dialogue.** A DAG built for P5 gives lower (better) scores to held-out P5 conversations than to P6/P7/etc. The metric captures real protocol-stage structure.

Effect sizes are lower than STAR (1.46–1.82× vs 1.85× mean) because all PE phases share the same overarching therapy framework, so phases are more similar to each other than e.g. hotel-booking is to weather. The fact that ratios of ~1.5× still emerge with overwhelming significance is the right result.

**Step 1 → Step 2 gate: CLEARED.** The metric is fit for purpose.

## 8. What comes next (Step 2)

Step 1 was the necessary scaffolding. Step 2 is the actual research output: the LLM comparison.

### TODO 5 — Generate LLM DAGs ✅ DONE (P5/P6/P7)
For each LLM × each prompting strategy, ask the LLM: *"Generate a PE phase-X dialogue flow DAG as nodes + edges."* No utterance content — just the structure. Driver: `scripts/generate_llm_dags.py`, prompts in `prompts.yaml` (repo root).

- **LLM panel** (4, all routed via OpenRouter / single `OPENROUTER_API_KEY` — option C, 2026-06-02): `deepseek-v3.2`, `gpt-5.1`, `gpt-oss-20b`, `kimi-k2-0905`. (The earlier 5-model GPT-4o/Sonnet/Gemini/Llama/Qwen panel was superseded by these four; repoint via `MODEL_REGISTRY` in the script.)
- **Prompt variants** (3): v1 = Prompt 1 alone; v2 = Prompts 1–5 fused into one call; v3 = Prompts 1–5 run sequentially (multi-turn). Prompt 5's `{{thousand_voices_data}}` slot is filled with N randomly sampled TV **training** conversations (from `TV_v1.json`, so test data never leaks into DAG construction).
- 4 × 3 × 3 phases = **36 candidate DAGs on disk** at `data/dags/<model>/<variant>/<phase>/{dag.mmd,dag.json,transcript.json}`.
- **Scope note:** generated for P5/P6/P7 only — the three phases validated in TODO 4. Extending to P8/P10/P11 depends on finishing their labelling (TODO 2) and re-running TODO 4 for them first.

### TODO 6 — Canonicalise labels across LLMs
Different LLMs emit different labels for the same intent ("ask_suds" vs "elicit_suds" vs "check_distress"). Cluster all unique labels in label-embedding space using K-means; pick K via silhouette score or inertia elbow. Map each cluster to one canonical label. Now all 15 DAGs share a vocabulary, enabling cross-LLM comparison.

### TODO 7 — Cluster-then-recentroid bucket population
This is the methodological contribution of the project — it solves the bucket-population cold-start problem for LLM-generated DAGs.

For each LLM DAG:
1. Embed each (canonical) node label as the initial anchor.
2. For each training utterance from the same phase, find the nearest LLM-DAG node by cosine; append the utterance text to that node's bucket.
3. Recompute each node's centroid as the mean embedding of its assigned utterances. **Replace** the label-anchor with this centroid.

Now `IntentBucket.utterances` is populated with real text, the centroid is in utterance space, and FuDGE can score against it just like it scored against the prefix-tree DAG.

### TODO 8 — FuDGE on LLM DAGs
For each of the 15 populated DAGs, run the same in-class vs out-of-class discrimination test as TODO 4. A good LLM DAG will discriminate as well as (or better than) the prefix-tree reference. A bad one will discriminate poorly or not at all.

### TODO 9 — AutoEval-ToD Domain Compliance (parallel content track)
Independent second metric. Doesn't use embeddings or alignment, so it can't share failure modes with FuDGE.

For each held-out conversation:
1. Convert each DAG node into a natural-language rule (*"The therapist elicited a SUDS rating"*).
2. Ask an LLM judge to score each rule against the conversation: 1 (followed), 0 (not followed), -1 (not applicable). Plus one-sentence reason.
3. Session score = mean(rule_scores excluding -1).
4. Aggregate as in-class vs out-of-class, same as FuDGE.

Paper claim: 94–97% agreement with human raters. Validates by spot-checking ~20 sessions manually.

### TODO 10 — Cross-LLM comparison table
The actual research output:

| LLM | Variant | FuDGE in-class | FuDGE gap | Compliance in-class | Compliance gap |
|-----|---------|----------------|-----------|---------------------|----------------|
| GPT-4o | V1 | … | … | … | … |
| Claude Sonnet 4.6 | V2 | … | … | … | … |
| (15 rows total) | | | | | |

Best LLM × variant = lowest in-class FuDGE + biggest gap + highest in-class Compliance + biggest Compliance gap. Tie-breaker: gap size.

### TODO 11 — Clinician pairwise validation (optional polish)
Where FuDGE and AutoEval-ToD disagree on a DAG's quality, ask Francesca to do pairwise comparisons: "Which of these two DAGs better captures this conversation's flow?" Compute Cohen's kappa with each metric's preference. Tells you which metric tracks clinical judgement better. Strengthens the paper.

## 9. The big picture, in one paragraph

You're building a **measuring stick** (FuDGE on prefix-tree DAGs, validated on STAR and on three TV phases) and then **using that stick to rank LLMs** at generating PE therapy DAGs. The labelling pipeline turns unlabelled TV conversations into the input prefix-trees need. The split makes everything held-out and reproducible. The cluster-then-recentroid step lets LLM-emitted labels carry real-data centroids so FuDGE's cost function compares utterances to utterances (the only thing it's designed to do well). AutoEval-ToD is a second, independent measuring stick — if both agree, the LLM ranking is solid; if they disagree, that's an interesting subplot for the paper.

## 10. Why two metrics matter

FuDGE alone tells you "structural alignment is correct." That's necessary but not sufficient for a clinically meaningful DAG.

Imagine an LLM DAG that perfectly captures the PE *structure* (right intents in the right order) but the bucket utterances are robotic, cold, or otherwise non-therapeutic. FuDGE would score it well; AutoEval-ToD (which judges content using clinical-style rules) would catch the problem.

Conversely, an LLM DAG with warm, empathetic language but garbled structure would score poorly on FuDGE and well on AutoEval-ToD. The combination gives you a fuller picture.

(Empathetic tone specifically is **out of scope** for this paper — supervisor's call. It's a future-work direction, possibly using AutoEval-ToD's Prompt J.5.)

## 11. The "cluster-then-recentroid" step, in detail

This is the cleverest part of the methodology, so it deserves its own section.

**Problem**: An LLM gives you a DAG with abstract labels and zero utterances per node. FuDGE needs each node to have an utterance-space centroid to compute substitution costs. Where do those utterances come from?

**Three options we considered:**

1. **Prototypes (option A)** — make the LLM emit example utterances per node. Cheap but the prototypes are LLM-imagined, not real-data centroids.
2. **LLM-judge per utterance (option B)** — ask an LLM to assign each training utterance to the best DAG node. Faithful but expensive.
3. **Cluster-then-recentroid (option C)** — embed the label as initial anchor, embed each training utterance, assign by cosine, then replace the anchor with the mean-of-assigned centroid. Cheap *and* uses real data. Novel — not exactly STAR Path 1 or Path 2.

We picked option C (per supervisor's recommendation in the 2026-05-17 meeting), with the explicit understanding that it's a methodological contribution, not a STAR-derived technique.

**Why option C works** when our earlier embedding test seemed to suggest it wouldn't:
- The earlier test embedded *abstract action descriptions* (*"Therapist orients the client to the imaginal exposure procedure"*) against real utterances. SBERT can't bridge that abstract → concrete gap.
- The actual cluster-then-recentroid step uses *LLM-emitted node labels* as initial anchors. Those tend to be short intent-shaped phrases ("ask_suds", "prompt_continue_narrative"), not full descriptions. The label-embedding space is structured enough for an initial nearest-neighbour assignment to be useful.
- Even if some early assignments are noisy, recomputing the centroid as the mean of N>>1 assigned utterances washes out the noise. The centroid converges to wherever the real utterances live.

**Risk to flag**: one-pass NN with no iteration. Mis-assignments stick. Mitigation: optional second pass that re-assigns based on recomputed centroids, if needed. Not implemented yet.

## 12. What you have on disk right now

```
fudge-validate/
├── METHODOLOGY.md                                 the locked plan (v0.2)
├── EXPLAINER.md                                   this doc (the understanding)
├── README.md                                      setup + layout
├── PROGRESS.md                                    day-to-day status
├── VALIDATION_REPORT.md                           formal write-up (older)
│
├── src/fudge/
│   ├── data_loader.py                             STAR + TV loaders
│   ├── splits.py                                  versioned deterministic splits
│   ├── costs.py / fudge_naive.py / fudge_efficient.py   FuDGE implementation
│   ├── embeddings.py                              SBERT cache
│   └── types.py                                   Conversation / IntentBucket / DialogueFlow
│
├── prompts.yaml                                   DAG-generation prompts 1–5 (TODO 5)
│
├── scripts/
│   ├── llm_label_star.py                          STAR labelling pipeline (existed)
│   ├── llm_label_tv.py                            TV labelling pipeline (new)
│   └── generate_llm_dags.py                       Step 2 — TODO 5 DAG generator (OpenRouter)
│
├── experiments/
│   ├── star_v2_validation.py / .json              Step 1a — TODO 1 (22/23 pass)
│   ├── tv_prefix_tree_discrimination.py / .json   Step 1b — TODO 4 (3/3 pass)
│   └── embedding_test.py                          v0.1 gate test (kept for reference)
│
└── data/                                           (gitignored)
    ├── STAR/                                       STAR dataset
    ├── STAR_llm_labels/                            from previous STAR work
    ├── thousand-voices-trauma/                     TV dataset
    ├── TV_llm_labels/{P5..P11}/{taxonomy.json,whole/}  per-phase labels (P5/P6/P7 complete)
    ├── dags/<model>/<variant>/<phase>/             TODO 5 output — 36 DAGs (4×3×P5/P6/P7)
    └── splits/{STAR_v2.json, TV_v1.json}           versioned splits
```

## 13. Status snapshot

- ✅ TODO 1 — STAR re-validation (22/23 pass)
- ⚠ TODO 2 — TV labelling (P5/P6/P7 done; P8 partial; P10 smoke; P11 not started — pending OpenAI top-up)
- ✅ TODO 3 — TV split
- ✅ TODO 4 — TV prefix-tree discrimination (3/3 pass, P5/P6/P7)
- ✅ TODO 5 — LLM DAG generation (36 DAGs: 4 models × 3 variants × P5/P6/P7; `data/dags/`)
- ⬜ TODO 6 — Label canonicalisation across LLMs  ← **NEXT**
- ⬜ TODO 7 — Cluster-then-recentroid bucket population
- ⬜ TODO 8 — FuDGE on LLM DAGs (in-class vs out-of-class, as TODO 4)
- ⬜ TODO 9 — AutoEval-ToD Domain Compliance (parallel content metric)
- ⬜ TODO 10 — Cross-LLM comparison table (the research output)
- ⬜ TODO 11 — Clinician pairwise (optional)

**Step 1 (metric validation) is complete.** Step 2 has started: the 36 candidate DAGs exist (TODO 5). **What remains is TODO 6 → 10:** canonicalise labels across the 4 LLMs, populate each DAG's buckets via cluster-then-recentroid, score with FuDGE (and AutoEval-ToD in parallel), then build the comparison table. TODO 6 is the immediate next action. Everything past TODO 5 currently scopes to P5/P6/P7; widening to P8/P10/P11 is gated on finishing TODO 2 labelling + re-running TODO 4 for those phases.

## 14. Decisions you'll need to make to resume

1. **Finish P8/P10/P11 labelling, or ship the 3-phase paper (P5/P6/P7)?** Everything downstream (TODO 5 DAGs, TODO 6–10) currently scopes to P5/P6/P7. The cleanest paper has all 6 phases but needs TODO 2 finished *and* TODO 4 + TODO 5 re-run for the new phases. ~$8 of OpenAI credit fills the labelling gap; cached calls don't cost anything to retry.
2. **TODO 6 canonicalisation — how to pick K?** The 4 LLMs emit different label vocabularies across the 36 DAGs. Cluster all unique labels in embedding space; choose K by silhouette/elbow, or fix K to the prefix-tree taxonomy size per phase as a principled anchor. This is the immediate next task.
3. **Pilot before scaling TODO 7–8?** Validate the cluster-then-recentroid → FuDGE path end-to-end on one (model × variant × phase) DAG before running all 36, so a plumbing bug doesn't cost a full sweep.

---

If anything in this doc doesn't make sense, ask. The point of writing it is that you (or your supervisor, or a future you) can pick this up cold and know exactly what's going on.
