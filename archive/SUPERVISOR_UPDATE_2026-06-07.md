# Project update — FuDGE / LLM-DAG pilot (7 June 2026)

The Step-2 pipeline now runs end-to-end on two models (gpt-oss-20b, deepseek-v3.2). While checking
the results I found a measurement problem, traced its cause, and corrected the analysis. This note
lays out what happened and the decision I need from you.

---

## TL;DR

- The pilot scores were **mostly measuring conversation length, not therapy flow.**
- **Cause:** the LLMs drew short *summary* flowcharts (~12 steps); FuDGE matches one step to one
  conversation turn, so a 12-step map can't cover a 30-turn conversation. The leftover turns get a
  flat penalty that just grows with length.
- **Step 1 (the metric validation) still passes** after correcting for length. FuDGE is fine — it
  was fed the wrong kind of DAG.
- **No Step-2 ranking survives** (best variant, best model, the "P7 DAG is generic" claim). All
  length artifacts.
- **Decision needed:** which way to fix it — deeper DAGs, an LLM judge, or a tweaked FuDGE
  (details in §5). I'd hold the remaining models until we choose.

---

## 1. What I did

- Generated, aligned, and scored all 9 deepseek-v3.2 cells (3 prompt variants × P5/P6/P7) — same
  harness as gpt-oss-20b.
- Fixed an engineering bug: the original FuDGE scorer hung on certain DAG shapes (18 GB, never
  finished). Replaced it with an equivalent version — **provably the same score**, 10–300× faster
  (Appendix A1).

## 2. The problem

- FuDGE walks a path through the DAG and lines it up against the conversation, **one step = one
  turn.** A turn that has a matching step is cheap (~0.5); a turn with no step left pays full price
  (1.0). Score = average cost per turn.
- The LLM DAGs are only ~12 steps deep. Conversations are 20–34 turns. So only the first ~12 turns
  can be matched — **every turn after that pays full price regardless of content.**
- Longer conversation → bigger unmatched share → higher score, automatically:

  | conversation | matched | unmatched (full price) | score |
  |---|---|---|---|
  | 20 turns | 12 × 0.5 | 8 × 1.0 | 0.70 |
  | 34 turns | 12 × 0.5 | 22 × 1.0 | 0.82 |

- Our phases differ in length (P5 ≈ 21 turns, P6 ≈ 34, P7 ≈ 30), and the test compares scores
  *across* phases — so for short DAGs it mostly detects length.

## 3. The evidence

- Within a single phase (so phase can't be the cause), score vs. conversation length:

  | DAG type | correlation (score ↔ length) |
  |---|---|
  | Step-1 prefix-trees | 0.42 – 0.64 |
  | LLM DAGs (every cell, both models) | **0.89 – 0.99** |

- For LLM DAGs the score is **almost entirely a length measurement.** Prefix-trees avoid this
  because they're built from full real conversations — always deep enough. That's why Step 1
  passed and the flaw stayed hidden until shallow LLM DAGs arrived.
- Note: the alignment step is working fine — it sets the *quality* of each match. It can't add
  match *slots*; only DAG depth does that.

## 4. Results after correcting for length

Re-analysed the saved scores comparing **only same-length conversations** (no re-scoring needed;
method in Appendix A2).

**Step 1 — prefix-tree validation (the gate):**

| phase | raw | corrected | significant? |
|---|---|---|---|
| P5 | 1.67× | **1.29×** | yes (p ≈ 1e-4) |
| P6 | 1.82× | **1.63×** | yes |
| P7 | 1.46× | **1.38×** | yes |

→ **Gate holds.** P6/P7 clear the 1.3 bar; P5 lands right at it.

**Step 2 — LLM DAGs (both models):**

| phase | raw | corrected | reading |
|---|---|---|---|
| P5 | 1.22–1.30× | **1.07–1.09×** | ~75% of the effect was length |
| P6 | 1.05–1.17× | **1.10–1.17×** | the only real signal |
| P7 | 1.01–1.07× | 1.01–1.05× | essentially nothing |

→ All variant/model means land within ~0.03× of each other. **The pilot rankings don't survive —
I'm withdrawing them.**

## 5. The decision: how to fix it

The root issue is a **mismatch** — FuDGE expects turn-level DAGs; the LLMs drew summary DAGs.
Three ways forward:

- **Option A — deeper DAGs (keep FuDGE).** Tell the prompts to produce turn-level DAGs whose path
  lengths span real session lengths (not one long path — short sessions would then be penalised
  too; we need branch points so sessions can exit at different lengths). *Cheapest; keeps the
  validation we already have.*
- **Option B — LLM judge (replace FuDGE for this).** An LLM judge can map a whole multi-turn
  stretch to one summary step, so summary DAGs work directly and the length problem disappears.
  *This is essentially our already-planned AutoEval-ToD track (TODO 9).* Needs its own validation,
  and the judge must not be the model that generated the DAG (circularity).
- **Option C — tweak FuDGE so one step absorbs several turns.** Most faithful to how sessions
  actually work, but it's a new metric → Step 1 must be re-validated from scratch.

**My recommendation:** do **A first** (cheap, keeps validated metric, directly tests whether the
signal returns) and push **B** in parallel as the independent second metric we always planned.
Keep **C** as the principled fallback if A feels artificial.

**Either way: hold kimi-k2 and gpt-5.1 until we pick** — running them now just re-measures length.

## 6. Other method upgrades (regardless of A/B/C)

- Report length-corrected ratios (with raw alongside).
- Read the phase-confusion matrix column-wise only (same conversations ⇒ length cancels).
- Rank models with within-conversation paired tests (length cancels exactly — already TODO 10).
- Add a random-DAG baseline (same size, shuffled structure) to show how much signal is the LLM vs.
  the alignment.
- ≥3 generations per cell before any ranking claim (currently 1 — no error bars).

## 7. Decisions I need from you

- [ ] Which fix — **A (deeper DAGs)**, **B (LLM judge)**, or both in parallel?
- [ ] OK to hold the remaining models (kimi-k2, gpt-5.1) until then?
- [ ] Report length-corrected ratios as the headline statistic?
- [ ] Add the random-DAG baseline + ≥3 generations per cell? (cost is small — Appendix B Q9)
- [ ] Step-1 P5 sits at 1.29× vs the 1.3 threshold — report as a marginal pass, or tighten?
- [ ] Should P10 stay in the out-of-phase pool? Its conversations are ~2× longer than the rest
      (63 vs 21–34 turns) — the single biggest inflator of the raw numbers.

---

## Appendix A — technical detail

**A1. Scorer fix.** The paper's scorer (Algorithm 2) keeps one distance array per root→leaf
*path*; on DAGs with nested branch-and-remerge "diamonds" the path count explodes (one 24-node
DAG → ~354M arrays, 18 GB). Replacement (`fudge_dag`): topological-order dynamic programming with
one array per *node* — `D[v][j]` = cheapest alignment of any root→v path against the first j
turns; at a merge node the arrays combine by elementwise minimum (exact, because the edit-distance
recurrence distributes over min). O((V+E)·n), no explosion. Verified identical to the original on
all 15 cells where the original terminates. Step-1 numbers untouched.

**A2. Length-correction method.** Per-conversation scores were already saved, so this is
post-processing only. Group conversations into length bins (width 3); keep bins with ≥5
conversations on each side; reweight the out-of-phase pool to match the in-phase length
distribution; significance via within-bin permutation (10k). Coverage 77–94%. Residual within-bin
length variation means the corrected ratios are slight *over*-estimates of true flow signal — i.e.
upper bounds.

**A3. Why alignment doesn't fix it.** Alignment fills each step's bucket with real utterances and
uses their average as the step's "meaning" — this sets the match *price*. But the scoring pairs
each step with *at most one* turn, so a bucket of 600 utterances is still one slot. Match quality,
not coverage.

## Appendix B — likely questions

**Q1. Did the scorer fix change the metric?** No — same score to the last decimal on every cell
both versions can run; only the algorithm changed.

**Q2. If FuDGE is length-biased, how can Step 1 stand?** The bias only bites when the DAG is much
shorter than the conversation. Prefix-trees are depth-matched by construction, so they're only
mildly length-correlated and still discriminate at 1.29–1.63× after correction. FuDGE measures
flow when the DAG is deep enough — which is exactly what Step 1 established.

**Q3. Couldn't alignment compensate for short DAGs?** No — it sets match price, not match count
(A3).

**Q4. Is the length correction sound?** It's non-parametric (binning + permutation, common support
only, coverage reported). I can add a regression-residual check as a sensitivity analysis; I
expect the same conclusion.

**Q5. A DAG can't represent every conversation anyway — isn't that fatal?** No. FuDGE is a
*distance*, built for imperfect fit. All claims are relative (closer to own phase than others;
model A vs. B). The prefix-tree result is the empirical ceiling on how much structure a DAG can
capture here (1.29–1.63×) — and PE therapy being manualised is why that ceiling is usefully above 1.

**Q6. Why did the LLMs draw short DAGs?** We asked for "a dialogue-flow DAG" and they gave
stage-level flowcharts — the natural reading. Nothing in the prompt required turn-level depth.

**Q7. Is the residual P6 signal (1.10–1.17×) real LLM structure or just alignment?** Unknown until
the random-DAG baseline runs — that's exactly what it's for.

**Q8. Does any of this break no-circular-validation?** No. Prefix-tree stays validation-only; the
random-DAG baseline is a null control, not a competitor.

**Q9. Cost of the re-run?** Small. Generation is pennies–a few pounds per model; scoring is now
fast. Main cost is my time on the prompt revision.

**Q10. Why ≥3 generations per cell?** Each cell is currently one stochastic LLM sample; a different
draw could flip a 0.03× gap. Three+ gives error bars at the level the ranking claim is made.
