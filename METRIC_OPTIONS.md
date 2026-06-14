# Design note — fixing the metric/DAG granularity mismatch

**Date:** 8 June 2026. **Companion to:** `SUPERVISOR_UPDATE_2026-06-07.md` (which states the
problem and gives the three-way A/B/C decision). This note goes deeper on the two options that
involve *changing how we score* — modifying FuDGE (Option C) and using an LLM judge (Option B) —
with enough implementation detail to choose between them.

---

## 1. The diagnosis in one line

**A scoring metric and the DAG it scores must share a granularity.** FuDGE aligns **one node ↔ one
turn**; the LLMs drew **summary** DAGs (one node ≈ one *stage*, which spans several turns). That
mismatch is the entire length confound. There are exactly two principled ways to remove it:

- **make the metric stage-aware** (so it fits summary DAGs) — Option C below, or
- **use a judge that reads stages holistically** (LLM judge) — Option B below.

(Option A — prompt for turn-level DAGs so FuDGE's granularity fits — is covered in the main update
and is the cheapest first move. This note is about B and C.)

A useful fact about the data: PE therapy is manualised, so a short session and a long session
follow the **same stages in the same order** — they differ in *how long the client lingers in a
stage*, not in the route taken. The length variation is **dwell time**. Both options below are, at
heart, ways to let the score absorb dwell time.

---

## 2. Option C — make FuDGE stage-aware ("dwell" / DTW-style alignment)

### The idea
FuDGE's edit distance currently allows only: match a turn to a node and **advance** to the next
node. Add one more move — **stay**: match a turn to a node *and remain on that same node* for the
next turn. Now a single `explain_rationale` node can absorb a 6-turn explanation exchange instead
of paying a flat penalty for 5 of those turns.

This is the well-known **Dynamic Time Warping** generalisation of edit distance (one element aligns
to a *run* of elements). The code change is tiny — one extra term in the cost recurrence
(Appendix A).

### The one design knob that matters: the *stay cost*
- If staying is **too cheap**, one vaguely-worded node could greedily swallow an entire
  conversation and *every* DAG would score well — reintroducing the "generic DAG wins" failure.
- The natural protection: each dwell-turn is priced by **fit** — it pays the normal substitution
  cost against the node's centroid. A node only *cheaply* absorbs turns genuinely close to its
  meaning; off-topic turns still cost. So dwelling is cheaper than a flat insertion **only when the
  turn actually belongs to that stage** — exactly the behaviour we want.
- Optional extra knob: a small flat **per-stay penalty λ** to discourage runaway dwelling. λ = 0 is
  the pure DTW case; λ > 0 biases toward advancing. **This is the parameter to tune and report.**

### What it costs us
- Still a clean dynamic program, still O(nodes × turns), no blow-up.
- **It is a new metric, so Step 1 must be re-validated** — re-run the prefix-tree gate under the
  dwell rule. Expected to still pass: prefix-trees are already turn-level, so dwelling rarely
  triggers on them and their scores should barely move.
- **Predicted effect (the experiment that proves it works):** re-run the score-vs-length
  correlation on the *summary* LLM DAGs. Under dwell-FuDGE the leftover turns get absorbed instead
  of paying flat insertions, so the 0.89–0.99 correlation should **drop sharply**. If it doesn't,
  the dwell fix didn't take and we fall back to Option A or B.

### Pros / cons
- ➕ Keeps FuDGE's strengths: free, deterministic, exact, no API.
- ➕ Lets the LLMs draw the *natural* stage-level DAGs they already produce.
- ➖ Requires re-validating Step 1 (one experiment, but real work).
- ➖ Adds a tunable knob (λ) — more defensible but one more thing to justify.

---

## 3. Option B — replace FuDGE with an LLM judge (for summary DAGs)

This is the already-planned **AutoEval-ToD Domain Compliance** second metric (METHODOLOGY TODO 9).
An LLM judge maps a whole multi-turn stretch to a stage by *reading* it, so the one-node-one-turn
limit — and therefore the length confound — simply don't exist.

### The pipeline
1. **DAG → checklist.** Convert each node into a natural-language rule.
   `explain_procedure_and_rationale` → *"The therapist explained the imaginal-exposure procedure
   and gave its rationale."*
2. **Judge each conversation.** Give the judge the rule list + the full transcript. Per rule it
   returns **+1 satisfied / 0 not-applicable / −1 violated**, with a one-line justification.
3. **Score.** Session score = mean of rule scores (excluding N/A). Discrimination is the same logic
   as FuDGE: a P6 DAG should score P6 conversations high, other phases low.

### The design decisions that make or break it
- **🔑 Encode *flow*, not just *presence*.** A naïve checklist only asks "did the therapist explain
  the rationale?" — that measures **content coverage, not dialogue flow**, and a flowchart's whole
  value is *order*. Add **transition rules**: *"elicited a baseline SUDS **before** starting the
  narrative"*, *"offered grounding **only when** distress peaked."* Without this, the LLM judge
  silently degrades into a bag-of-intents checker and we lose the structural signal that justified
  using DAGs at all. **This is the most important design choice in Option B.**
- **🔒 Circularity guard (locked).** The judge must **not** be the model that generated the DAG.
  Use one fixed, strong judge for every DAG.
- **Validation replaces Step 1 here.** The judge itself must be validated: hand-score ~20 sessions
  and target ≥90% agreement (AutoEval-ToD reports 94–97% LLM–human agreement, so this is
  realistic). That hand-scored agreement check is this metric's equivalent of the prefix-tree gate.
- **Cost & determinism.** API calls per (conversation × DAG); temperature 0 and cache to keep it
  repeatable. Not the free, exact number FuDGE gives — budget for it.

### Pros / cons
- ➕ Naturally fits the summary DAGs the LLMs produce — no DAG change, no length confound.
- ➕ Independent failure modes from FuDGE (no embeddings, no edit distance) → a genuine second
  opinion, which is why the plan always wanted two metrics.
- ➖ Needs its own validation; costs money; stochastic.
- ➖ Can drift into measuring content-only unless flow is explicitly encoded (see 🔑 above).

---

## 4. How they compare

| | dwell-FuDGE (Option C) | LLM judge (Option B) |
|---|---|---|
| Fits | summary **and** turn-level DAGs | summary **and** turn-level DAGs |
| Cost | free, deterministic, exact | API cost, stochastic |
| Measures | geometric alignment w/ dwell | rule compliance (+ flow if encoded) |
| Validation needed | re-run Step 1 gate | hand-score ~20 sessions |
| Main risk | tuning the stay-cost λ | drifting to content-only |
| Independent of FuDGE? | no (same family) | **yes** (different failure modes) |

**These are not either/or.** The locked plan always wanted **two metrics** precisely so a
conclusion is only "strong" when both agree, and disagreements become the interesting cases (that's
what the optional clinician comparison, TODO 11, is for). The clean research story: **match the
metric to the DAG granularity, and report both numbers.**

---

## 5. Recommended order of experiments (cheapest first)

1. **Option A (no new metric):** revise prompts for turn-level depth, regenerate the two pilot
   models, re-check the length-corrected ratios. Confirms whether depth alone recovers signal.
2. **Option C (small code change):** add the dwell term (Appendix A), re-run the Step-1 gate, and —
   the key test — re-measure the score-vs-length correlation on the existing summary DAGs. If it
   collapses, dwell-FuDGE works and we get to keep the cheap deterministic metric *and* the natural
   stage-level DAGs.
3. **Option B (parallel track):** stand up the LLM judge with flow-aware rules + the hand-scored
   validation. This proceeds regardless, as the independent second metric.

Each step is informative on its own and they don't conflict, so we can run 1 and 2 quickly and let
3 proceed alongside.

---

## Appendix A — the dwell change, against the current scorer

Today's exact recurrence in `fudge_dag` (`src/fudge/fudge_efficient.py`), per node `v` with parent
row `merged` (= cheapest alignment of any root→v *parent* path) and `new_dist` (= the row being
built for `v`):

```python
# new_dist[j+1] = D[v][j+1]
val = min(
    merged[j + 1] + del_cost,                              # delete node v
    new_dist[j] + costs.insertion_cost(utts[j]),           # insert turn j (matched to nothing, flat 1.0)
    merged[j]   + costs.substitution_cost(bucket, utts[j]) # ADVANCE: v matches turn j, came from parent
)
```

The dwell version adds **one term** — "v matches turn j having *stayed* on v" (note it reads
`new_dist[j]` = D[v][j], not the parent's `merged[j]`):

```python
val = min(
    merged[j + 1] + del_cost,                                          # delete node v
    new_dist[j] + costs.insertion_cost(utts[j]),                       # insert turn j
    merged[j]   + costs.substitution_cost(bucket, utts[j]),            # ADVANCE (from parent)
    new_dist[j] + costs.substitution_cost(bucket, utts[j]) + STAY_PEN, # DWELL (stay on v)  ← new
)
```

- `STAY_PEN` = 0 → pure DTW (a node absorbs a run of turns at pure substitution cost).
- `STAY_PEN` > 0 → the tunable bias toward advancing.
- Complexity unchanged (still one row per node); equivalence to the old metric holds when no dwell
  is ever chosen (set `STAY_PEN = ∞` to recover today's behaviour exactly — a free regression test).

The interpretation that makes it correct: an extra in-stage turn now has a choice — pay a flat
**insertion** (1.0) to be ignored, or **dwell** on the current node at its substitution cost
(~0.5 if it fits). It will dwell exactly when the turn genuinely belongs to that stage, which is
the behaviour we want and the reason the length confound should disappear.
