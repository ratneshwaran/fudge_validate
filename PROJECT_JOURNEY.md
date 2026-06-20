# Project journey — what I did, in order

**Purpose:** a plain-language, start-to-finish record of everything done on this project, with the
design choices called out (why k-means this way, why this split, why this metric). For me, to
explain the work to anyone — or to remember it later.

**The one-sentence project:** *Which LLM writes the best "flowchart" (DAG) of how a
Prolonged-Exposure (PE) therapy session should go — judged against real therapy conversations?*

A quick legend used throughout:

> **🔧 Design choice:** a decision I made and why.
> **🔒 Locked:** a decision my supervisor fixed on 2026-05-17 that I must not change.

---

## Phase 0 — Building the FuDGE engine (April 2026)

Before any of the real research, I had to build and trust the measuring instrument.

**What I did**
- Implemented **FuDGE** (Fuzzy Dialogue-Graph Edit distance) from the paper (arXiv:2411.10416):
  the cost functions (Eqs 8, 11, 12), the data types, and *two* versions of the scoring algorithm.
- Added a **sentence embedding layer** so we can compare meanings, not exact words.
- Tested it on **STAR**, a labelled customer-service dialogue dataset (the dataset the FuDGE paper
  itself used), to confirm my implementation reproduced sensible behaviour.

**What FuDGE actually does (plain terms):** it lays a real conversation alongside a DAG and asks
"what's the cheapest way to line these up?" — matching a turn to a node is cheap, skipping or
inserting is expensive. Low score = the conversation fits the flowchart well.

> **🔧 Design choice — embedding model: `all-MiniLM-L6-v2` (SBERT), 384 dimensions.**
> A small, fast, widely-used sentence-transformer. Big enough to capture meaning, small enough to
> run thousands of utterances quickly.
>
> **🔧 Design choice — normalise every embedding to unit length.** This makes "cosine similarity"
> (the standard meaning-closeness measure) equal to a plain dot product, which is faster and lets
> the later clustering use simple matrix maths.
>
> **🔧 Design choice — two FuDGE implementations, verified equal.** A slow obvious one and a fast
> one, checked to give identical scores. (This habit paid off later — see Phase 8.)

**An early lesson (April):** in an experiment I found that an apparent quality gap between methods
**collapsed when tested on held-out data**. Foreshadowing: this project keeps teaching the same
lesson — *test on data the method never saw, and check for hidden explanations.*

---

## Phase 1 — The supervisor locks the methodology (2026-05-17)

The project was reframed into a **two-step structure** that everything else obeys.

> **🔒 Locked — the two-step rule:**
> - **Step 1: validate the metric.** First prove FuDGE can actually tell therapy phases apart,
>   using a *reference* construction (the "prefix-tree", below). If it can't even do that, no
>   comparison of LLMs means anything.
> - **Step 2: evaluate the LLMs.** Only after Step 1 passes, compare LLM-generated DAGs.
>
> **🔒 Locked — no circular validation.** The prefix-tree is *only* the yardstick for Step 1. It
> is never allowed to also be a "contestant" in Step 2. (Using the same thing as both ruler and
> contestant would be meaningless.)
>
> **🔒 Locked — phase is the axis, not trauma type.** Therapy *phase* (P5, P6, …) governs how a
> conversation flows; trauma type is just content. So we test whether DAGs tell *phases* apart.

---

## Phase 2 — Step 1a: re-validating on STAR (2026-05-31)

**What I did**
- Rebuilt the STAR validation cleanly with a documented prefix-tree method and the new split
  rules (below). Result: **22 of 23 tasks showed significant separation** (Bonferroni-corrected).
- This confirmed the metric and the harness work on a labelled dataset before moving to the
  unlabelled therapy data.

> **🔧 Design choice — what a "prefix-tree DAG" is.** Take all the training conversations of one
> type, and merge them into a tree by their step-sequence: conversations that start the same share
> the same early branches, then split where they diverge. Each node collects the real utterances
> seen at that point. It's the most faithful possible flowchart of the data — which is exactly why
> it's the *reference*, not a contestant.

---

## Phase 3 — Step 1b: labelling the therapy data (2026-05-31)

The therapy dataset (**Thousand Voices of Trauma**, "TV": 3000 synthetic PE sessions, 500 clients
× 6 phases) has **no labels**, and the prefix-tree needs them.

**What I did**
- Built an LLM pipeline that reads each conversation and labels its turns with an intent
  (e.g. "explain rationale", "elicit SUDS rating").

> **🔒 Locked — label only the therapist (agent) turns.** The therapist drives the session;
> client turns are unpredictable reactions. Labelling only the therapist halves the cost and
> avoids noisy client labels. (Fully labelled so far: P5, P6, P7.)

---

## Phase 4 — Step 1c: the train/test split (2026-05-31)

**What I did**
- Created one fixed **70% train / 30% test** split, saved as `data/splits/TV_v1.json`
  (~2082 train / 900 test). Everything downstream uses this exact split so results are comparable.

> **🔒 Locked — 70/30, stratified.** "Stratified" means the split keeps the mix of trauma types
> balanced inside each phase, so a phase's test set isn't accidentally all one type. This stops us
> confusing "good DAG" with "got lucky with the subgroup mix".
>
> **🔧 Design choice — train vs. test separation is sacred.** DAGs are always built/aligned from
> *training* conversations and scored on *held-out test* conversations. No conversation is ever in
> both. (This is what makes the numbers trustworthy.)

---

## Phase 5 — Step 1d: the gate test — does FuDGE work on therapy? (2026-05-31)

**What I did**
- Built a prefix-tree DAG for each of P5, P6, P7 from training, then scored held-out test
  conversations: each phase's DAG against its *own* phase (should score low) vs. *other* phases
  (should score high). The gap (ratio = out/in) is the "discrimination".
- **Result — PASS:** P5 **1.67×**, P6 **1.82×**, P7 **1.46×**, all statistically significant.

> **🔧 Design choice — the statistics.** Mann-Whitney U test (does in-phase score significantly
> lower than out-of-phase?), a bootstrap for confidence intervals, and Bonferroni correction
> (a stricter bar because we run several tests at once). Same stats reused everywhere after.

**This was the green light** to start Step 2. The metric demonstrably separates phases — *when fed
a good DAG.* (Hold that caveat; it becomes the whole story later.)

---

## Phase 6 — Step 2: generating LLM DAGs (2026-06-02 → 06-06)

**What I did**
- Wrote prompts asking LLMs to produce a PE dialogue-flow DAG as JSON (nodes with an actor + a
  label, and edges). Later rewrote them to be **phase-conditioned** (the prompt tells the model
  which phase, its description, and the expected therapeutic moves).
- Added a validity checker (is it acyclic? one connected piece? known actors?).

> **🔧 Design choice — three prompt "variants" to test whether prompting *style* matters:**
> - **v1:** one shot — "draw a PE DAG".
> - **v2:** all instructions fused into one big prompt.
> - **v3:** sequential — draft → critique → revise → finalise → merge with example conversations.
>
> **🔧 Design choice — the model panel** (via OpenRouter): gpt-oss-20b, deepseek-v3.2,
> kimi-k2-0905, gpt-5.1. A mix of open and commercial models. **So far only gpt-oss-20b and
> deepseek-v3.2 have been run** (the pilot).
>
> **🔧 Design choice — scope = P5, P6, P7 only**, because those are the phases fully labelled.

---

## Phase 7 — Step 2: alignment / clustering — filling the DAG with real text (2026-06-06)

The LLM's DAG has abstract labels but **no real utterances**. FuDGE compares meanings, so each node
needs real example utterances to define what it "means". This step assigns real *training*
utterances to nodes. It's called **cluster-then-recentroid** and it's a methodological contribution
of the project.

**What I did — the method, step by step:**
1. **Embed** every node's label and every training utterance into the 384-d space.
2. **Assign** each training utterance to the nearest node (cosine similarity = dot product, since
   everything is unit-length).
3. **Recentroid:** recompute each node's centre as the *average of the real utterances it won*,
   then re-assign. Repeat. (This is the "recentroid" part — the node stops meaning "its label" and
   starts meaning "the real utterances that landed there".)

This is essentially **constrained k-means**, with some deliberate choices:

> **🔧 Design choice — K (number of clusters) = number of nodes in the LLM's DAG.** Normally
> k-means *chooses* K (e.g. by silhouette score). Here we don't — the LLM's DAG dictates it.
> *Implication:* if the LLM picked a bad number of nodes, clustering can't fix that. (Noted as a
> limitation.)
>
> **🔧 Design choice — seed the clusters with the node-label embeddings**, not random starts.
> This anchors each cluster to the *meaning the LLM intended* for that node, so the final clusters
> stay interpretable.
>
> **🔧 Design choice — cluster agent and client turns separately.** FuDGE never matches a
> therapist turn to a client node, so they're clustered in separate pools.
>
> **🔧 Design choice — 5 re-assignment passes (`reassign_passes=5`), or stop early if it
> converges.** *Why:* with zero passes (pure nearest-label), one vaguely-worded node sits near
> everything and hoovers up a huge share of utterances — a "hub node" — while its neighbours
> starve. Recomputing centres from real text breaks the hub up. Measured effect: bucket sizes
> became 2.4–3.3× more even. (Honest caveat: it sharpens *significance* a lot but barely moves the
> *effect size* — a mild improvement, not a cure.)
>
> **🔧 Design choice — empty-node fallback.** If a node wins *zero* utterances, its bucket falls
> back to its label text (otherwise its centre would be a divide-by-zero). This is flagged in a
> coverage report. *Caveat:* it technically breaks the "never compare to label strings" principle
> for those nodes (e.g. 16 of 89 nodes in one cell) — flagged for a cleaner fix.

**Inspection point:** every aligned DAG is saved with a **coverage report** (how many real
utterances each node won, with examples) so I can *eyeball* whether utterances landed sensibly
before trusting any score.

---

## Phase 8 — Step 2: scoring, and a bug that had to be fixed first (2026-06-07)

**What I did**
- Ran FuDGE discrimination (same as Step 1) on every aligned LLM DAG, plus a **phase-confusion
  matrix** (every phase's DAG vs. every phase's conversations).

**The bug:** the original fast FuDGE scorer **hung** on some of deepseek's DAGs — one 24-node DAG
made it try to hold ~354 million pieces of work (18 GB of memory) and never finish. Cause: that
scorer explores every distinct route through the DAG separately, and DAGs with lots of
branch-then-rejoin "diamonds" have an explosive number of routes.

> **🔧 Design choice — rewrote the scorer as `fudge_dag` (a topological dynamic program).**
> Instead of one record per *route*, it keeps **one record per node** and merges branches by
> taking the cheaper option where they rejoin. This gives the *identical* score but can never
> explode. **Verified bit-for-bit identical** to the old scorer on all 15 cells where the old one
> finishes; 10–300× faster. (The Phase-0 habit of keeping a reference implementation made this
> check trivial.)

---

## Phase 9 — The big finding: the scores were measuring length (2026-06-07)

The deepseek results looked odd in a tell-tale way (every DAG "fit" the short-conversation phase
P5 well), so I checked the most boring possible explanation: **conversation length**.

**What I found**
- Within a *single* phase (so phase can't be the cause), the score lined up almost perfectly with
  how *long* the conversation was:

  | DAG type | correlation (score ↔ length) |
  |---|---|
  | Step-1 prefix-trees | 0.42 – 0.64 |
  | LLM DAGs (every cell, both models) | **0.89 – 0.99** |

- **Why:** FuDGE matches **one node to one turn.** The LLMs drew short *summary* flowcharts
  (~12 steps); real conversations are 20–34 turns. So only the first ~12 turns can be matched —
  every turn after that pays a flat penalty *regardless of content*. Longer conversation → bigger
  penalty → worse score, automatically. Since phases differ in length (P5 ≈ 21, P6 ≈ 34, P7 ≈ 30
  turns), comparing phases mostly compared *lengths*.
- The prefix-trees escaped this because they're built from real conversations, so their paths are
  naturally as long as the conversations. **That's why Step 1 passed and the flaw stayed hidden
  until shallow LLM DAGs arrived.**

> **🔧 Design choice — the fix was free re-analysis, not re-scoring.** Because every
> per-conversation score had been saved, I re-compared **only conversations of the same length**
> (group into length bins, reweight the comparison pool to match, test within bins with a
> permutation test). Like judging weightlifters within weight classes.

**Result after correcting for length:**

| | raw | length-corrected | verdict |
|---|---|---|---|
| Prefix-tree P5 / P6 / P7 | 1.67 / 1.82 / 1.46× | **1.29 / 1.63 / 1.38×** | **Step 1 still passes** |
| LLM DAGs (P5 cells) | 1.22–1.30× | **1.07–1.09×** | ~75% was length |
| LLM DAGs (P6 cells) | 1.05–1.17× | **1.10–1.17×** | the only real signal |
| LLM DAGs (P7 cells) | 1.01–1.07× | 1.01–1.05× | ~nothing |

**Consequence:** the metric itself is fine (Step 1 survives). But **every Step-2 pilot ranking was
an artifact** — "v3 is best", "gpt-oss beats deepseek", "P7's DAG is too generic" — all withdrawn.
The real gap is that the LLM DAGs are *summaries*, and FuDGE wants *turn-level* maps.

---

## Phase 10 — Where it stands now & the open decision (2026-06-08)

I wrote this up for my supervisor (`archive/SUPERVISOR_UPDATE_2026-06-07.md`). The core question is how to
resolve the **mismatch** (FuDGE wants turn-level DAGs; LLMs drew summary DAGs):

- **Option A — deeper DAGs, keep FuDGE.** Prompt for turn-level DAGs whose paths span real session
  lengths. Cheapest; keeps the validation we have. *(Recommended first step.)*
- **Option B — use an LLM judge instead of FuDGE.** A judge can map a whole multi-turn stretch to
  one summary step, so summary DAGs work directly and length stops mattering. This is our
  already-planned second metric (AutoEval-ToD). Needs its own validation; judge ≠ generator.
- **Option C — change FuDGE so one node can absorb several turns.** Most faithful to real sessions,
  but it's a new metric → Step 1 must be re-validated from scratch.

**Holding** the remaining models (kimi-k2, gpt-5.1) until the direction is chosen — running them
now would just re-measure length.

**Planned method upgrades regardless of A/B/C:** report length-corrected ratios; read confusion
matrices column-wise (lengths cancel); rank models by *within-conversation* paired tests (length
cancels exactly); add a **random-DAG baseline** (align a random same-size DAG to see how much
signal is the LLM vs. the alignment); and generate **≥3 DAGs per cell** for error bars.

---

## The through-line (if I had to say what I learned)

1. **Validate the instrument before trusting the measurement** — and validation only covers inputs
   that look like the validation inputs. Step 1 passed on turn-level DAGs; it said nothing about
   summary DAGs.
2. **Always check the boring explanation** (here, length) before the interesting one.
3. **Save the raw per-item numbers** — it turned a potential full re-run into ten minutes of
   re-analysis.
4. **A metric and the thing it scores have to speak the same language** — FuDGE speaks
   turn-by-turn; the LLMs spoke in stages. The whole current decision is about making them meet.

---

*Key files: `src/fudge/` (engine: `fudge_efficient.py`/`fudge_dag`, `costs.py`, `embeddings.py`,
`llm_dag.py`); `experiments/` (`tv_prefix_tree_discrimination.py`, `align_llm_dags.py`,
`llm_dag_discrimination.py`, `phase_confusion.py`, `length_matched_reanalysis.py`); `prompts.yaml`;
`data/splits/TV_v1.json`; `data/dags/<model>/<variant>/<phase>/`. Fuller versions: `archive/EXPLAINER.md`
(concepts), `EXPLAINER_v2.md` (the two recent sessions), `METHODOLOGY.md` (the locked plan),
`HANDOVER.md` (operational state).*
