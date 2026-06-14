# EXPLAINER v2 — what actually happened in Step 2 (plain-language, personal notes)

**Written:** 2026-06-07. **For:** me, future me, and nobody else.
**Relationship to other docs:** `EXPLAINER.md` explains the project fundamentals (what FuDGE is,
what a dialogue-flow DAG is, why the two-step methodology). Read that first if rusty. This file
explains, in simple terms, what was *done* in the two Step-2 sessions (June 6–7) and what we
learned — including the part where our pilot results turned out to be measuring the wrong thing.
`HANDOVER.md` (with its 2026-06-07 addendum) is the operational truth; `SUPERVISOR_UPDATE_2026-06-07.md`
is the formal version of this story.

---

## 0. The 30-second version

We built the whole Step-2 machine and ran it on two LLMs. It works mechanically. Along the way we
hit and fixed a real algorithmic bug (the scorer could take literally forever on certain DAG
shapes). Then, while sanity-checking the results, we discovered that our headline numbers were
mostly measuring **how long conversations are**, not how well DAGs capture therapy flow. We proved
it, quantified it, and re-analysed everything with the confound removed. The good news: Step 1
(the metric validation) survives. The bad news: every interesting-looking Step-2 ranking
("variant 3 is best", "gpt-oss beats deepseek", "P7's DAG is too generic") evaporates. The fix is
clear and cheap, and honestly the project is *stronger* for it.

---

## 1. Quick recap: what Step 2 is supposed to do

One sentence: ask several LLMs to draw a flowchart (DAG) of how a Prolonged-Exposure therapy
session should flow, then score each flowchart by how cheaply real therapy conversations can be
"explained" by walking a path through it (FuDGE = fuzzy edit distance between conversation and
DAG). A good P5 DAG should explain P5 conversations cheaply and other phases' conversations
expensively. That gap (the out/in ratio) is the "discrimination" we measure, and the prefix-tree
DAGs from Step 1 set the bar to beat (raw: 1.46–1.82×).

The pipeline per model: **generate** (phase-conditioned prompts, 3 variants) → **align** (fill
each DAG node's bucket with real training utterances so centroids are real-data, not label text)
→ **score** (FuDGE discrimination + a phase×phase confusion matrix).

## 2. What we built and ran (the machine works)

- **Phase-conditioned prompts** (`prompts.yaml` v2): every prompt now knows which phase it's
  generating for — name, description, expected therapeutic moves.
- **Alignment** (`src/fudge/llm_dag.py`): the cluster-then-recentroid step from EXPLAINER §11,
  plus an optional "second pass" (`reassign_passes=5`) that re-clusters utterances around the
  real centroids instead of the label anchors. Verdict from gpt-oss: mild win, keep it, but it
  doesn't rescue weak DAGs.
- **Scoring harnesses** (`experiments/llm_dag_discrimination.py`, `experiments/phase_confusion.py`):
  same statistics as the validated Step-1 harness (Mann-Whitney, bootstrap, Bonferroni).
- Ran all of it on **gpt-oss-20b** (June 6) and **deepseek-v3.2** (June 7), P5/P6/P7, 3 variants
  each.

## 3. Event one: the scorer blew up (and how it was fixed)

**What happened.** Deepseek likes drawing big DAGs with "diamonds" — a node that splits into
branches which later re-merge. The original scorer (`fudge_efficient`) explores every distinct
root-to-leaf *route* separately. Diamonds multiply routes (each nested diamond roughly doubles
them), and an implementation quirk made it much worse: every time a merge-node was reached again,
it re-processed everything it had already done. One innocent-looking 24-node DAG had 243 real
routes, which the algorithm turned into **~354 million** pieces of work and 18 GB of RAM. It would
never have finished.

**The fix, in one breath.** You don't need to remember every route separately. For each node, keep
a single row: "the cheapest possible cost of reaching this node having consumed the first j
utterances of the conversation, for every j". When branches re-merge, take the elementwise minimum
of the incoming rows — that's still exact, because every step of the cost recurrence respects
minima. Process nodes in topological order, done. This is `fudge_dag()` in
`src/fudge/fudge_efficient.py`.

**Why we can trust it.** It was checked against the old scorer on all 15 DAG cells where the old
one terminates: **identical scores to the last bit** (max difference 0.0). On the exploding cells
it runs in milliseconds. Step-2 scripts now use it; Step-1 results were produced by the old scorer
and are untouched (prefix-trees have no diamonds, so both scorers agree there anyway).

## 4. Event two: the length confound (the important one)

### How it surfaced

Deepseek's results looked *weird* in a specific way: in the confusion matrix, deepseek's P6 flow
was the best fit for *every* phase, and every flow fit P5 conversations suspiciously well. The
innocent explanation ("the DAGs are generic") felt thin — both models, same direction. Then one
number lined up: **P5 conversations are short** (mean 20.6 utterances vs 34.2 for P6, 29.8 for
P7). Everything "fit P5 well". Hmm.

### The smoking gun

Take ONE phase's test conversations and ONE flow — so phase identity can't matter — and correlate
each conversation's normalised score with its length:

| construction | correlation (Spearman ρ) |
|---|---|
| Step-1 prefix-trees | 0.42 – 0.64 |
| LLM DAGs — every cell, both models | **0.89 – 0.99** |

ρ = 0.9–0.99 means: rank the conversations by score and you have essentially ranked them by
length. For LLM DAGs, the score *is* a length measurement with a little flow signal sprinkled in.

### Why this happens (the actual intuition)

FuDGE walks one path through the DAG and pays for each conversation utterance: a *cheap* price if
a path node can absorb it (substitution, usually ~0.5–0.7), and a *flat full price of 1.0* if
nothing can (insertion). Here's the killer: LLM DAG paths are **9–24 nodes long**, but
conversations run **21–34 utterances** (P10: 63!). So a shallow DAG can explain at most the first
~dozen utterances; everything after that pays full price no matter what it says. The normalised
score is (total cost)/(length) — and as conversations get longer, the full-price fraction grows,
so the score creeps up towards 1.0 *for any shallow DAG, regardless of content*.

Prefix-trees never had this problem because their paths ARE real conversations' label sequences —
they're exactly as deep as conversations are long. That's why Step 1 looked fine and the pathology
stayed invisible until LLM DAGs (which are shallow) entered the picture. The metric wasn't lying
in Step 1; it just had a failure mode nothing had triggered yet.

### What it contaminated

The discrimination ratio compares *my phase's conversations* against *other phases' conversations*
— different lengths. So:

- **P5 looked strongly discriminated (1.22–1.33×)** mostly because its out-of-phase pool is
  longer than its in-phase pool. ~75% of that effect was length.
- **P7 looked weak** partly because its out-pool contains short P5 conversations that score well
  on anything.
- **"P7's DAG is promiscuous / too generic"** (last session's confident diagnosis) — artifact.
  P5 conversations are just short.
- **Every variant/model ranking** — artifact-dominated. The differences were 0.01–0.05× in a
  measure that length moves by 0.2×.

### The repair (no re-scoring needed!)

All per-conversation scores were already saved in the result JSONs, so the fix was pure
post-processing: only compare conversations *of the same length* (group into length bins, reweight
the out-of-phase pool to match the in-phase length distribution, permutation test within bins).
Like judging weightlifters within weight classes. Script:
`experiments/length_matched_reanalysis.py`.

**Results after matching:**

| | raw | length-matched | verdict |
|---|---|---|---|
| prefix-tree P5 | 1.67× | 1.29× | survives (at the 1.3 bar) |
| prefix-tree P6 | 1.82× | 1.63× | survives comfortably |
| prefix-tree P7 | 1.46× | 1.38× | survives |
| LLM P5 cells | 1.22–1.30× | 1.07–1.09× | mostly artifact |
| LLM P6 cells | 1.05–1.17× | 1.10–1.17× | the only real LLM signal |
| LLM P7 cells | 1.01–1.07× | 1.01–1.05× | ~nothing |

(Fun detail: P6 cells *improved* after matching — P6 is the longest phase, so the raw comparison
was biased *against* it. The confound doesn't just inflate, it distorts in both directions.)

## 5. So what is actually true now?

1. **The metric is validated** (Step 1 holds after the length control). FuDGE measures flow — on
   DAGs deep enough to cover whole conversations.
2. **The pipeline works** end-to-end, fast, on arbitrary DAG shapes.
3. **Current LLM DAGs are too shallow for the metric to say anything deep about them.** Their
   genuine flow signal beyond length is ≤1.17× vs the reference's 1.29–1.63×. That gap is mostly
   a *depth* gap, plausibly fixable in the prompts.
4. **No variant or model ranking from the pilot is real.** We genuinely do not know yet whether
   gpt-oss or deepseek writes better therapy DAGs.

## 6. What happens next (and what NOT to do)

- **Don't** run kimi-k2 / gpt-5.1 with the current prompts — it would buy more length
  measurements.
- **Do** (after supervisor sign-off): make the prompts demand full-session depth (~30+ alternating
  nodes per path), regenerate the two pilot models, and check the length-matched ratios move.
- Permanent method upgrades: length-matched ratios as the reported statistic; confusion matrices
  read **column-wise only** (same conversations → length cancels); model ranking via
  within-conversation paired tests (each conversation scored under every model's flow — its
  length cancels perfectly); a random-DAG baseline so we know the floor.

## 7. What I'd tell past-me

- A correlation check between your score and the most boring covariate you can think of (length!)
  costs ten minutes and should have been run the day the harness was written.
- When a diagnosis explains one model's quirk ("gpt-oss made a generic P7 DAG"), check whether it
  also "explains" the second model the same way. The same anomaly twice isn't a coincidence —
  it's the metric.
- Saving per-conversation scores in the result JSONs (rather than only the aggregates) is what
  made the entire rescue possible without a single re-run. Always do this.
- Validation passing (Step 1) doesn't mean the metric is safe everywhere — it means it's safe *on
  inputs that look like the validation inputs*. LLM DAGs didn't.

---

*Files that matter from these sessions: `src/fudge/fudge_efficient.py` (`fudge_dag`),
`src/fudge/llm_dag.py`, `experiments/align_llm_dags.py`, `experiments/llm_dag_discrimination.py`,
`experiments/phase_confusion.py`, `experiments/length_matched_reanalysis.py`, results in
`experiments/*_r5.json` + `experiments/length_matched_reanalysis.json`, per-cell artifacts under
`data/dags/<model>/<variant>/<phase>/`.*
