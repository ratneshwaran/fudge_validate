# Handover — LLM-DAG evaluation pilot (Step 2)

**Written:** 2026-06-07, end of a working session. **For:** the next Claude Code session (or future me).
**Read order:** this file first, then `EXPLAINER.md` (the conceptual walkthrough) and `METHODOLOGY.md`
(the locked v0.2 plan). This handover records work done *after* `EXPLAINER.md` §13 was last written, so
where they disagree, **this file is the current truth** for TODO 5–8 status.

---

## ⚠ ADDENDUM 2026-07 (current state) — read this FIRST

**Where things stand (as of 2026-07-12).** Branch `feat/segmentation-fudge` (pushed to
GitHub, 8 commits ahead of `master`) holds everything below. `master` has only the docs
cleanup. All three `.env` keys are populated (OpenAI, HF, OpenRouter).

**The segmentation method is validated end-to-end.** Pilot (gpt-oss-20b v3 × P5/P6/P7,
fresh DAGs, 100% alignment coverage, commit `28844be`):
- raw FuDGE reproduced the length confound on new DAGs: out-block ρ(score, length)
  = +0.92…+0.98;
- `--segment` killed it (ρ ≈ 0/negative), and length-matched ratios ROSE:
  P5 1.14→**1.59×**, P6 1.17→**1.30×**, P7 flat ~1.06 (that DAG is genuinely weak and
  fails the new alternation validity gate).
- Supervisor deck ready to send: `supervisor_update_2026-06-28.pptx` (untracked).

**Next steps, in order:**
1. Scale the grid: `generate_llm_dags.py` for deepseek-v3.2 (± kimi-k2, gpt-5.1) ×
   v1/v2/v3 → align `--reassign-passes 5` → score baseline + `--segment` →
   `length_matched_reanalysis.py --results ...`. Cheap (~$1–2 OpenRouter).
2. Decide merge of `feat/segmentation-fudge` → `master` (verified; ready).
3. LLM-judge track: `scripts/llm_judge.py` scaffold is tested; needs the
   discrimination experiment + ~20-session hand-scored validation (report kappa,
   not just accuracy — see `LLM_JUDGE_DESIGN.md` / `LITERATURE_SCAN.md`).
4. Later: ≥3 generations/cell for error bars; random-DAG baseline; P8/P10/P11 labels.

**Ground rules that bite:** results are only comparable within a data generation —
check `PROVENANCE.md` before comparing any numbers (June results are quarantined in
`experiments/archive_pre_relabel/`). Run everything with `.venv/bin/python`. Git
commits: no AI/tool attribution. The old June result JSONs can never be reproduced
(lost non-deterministic DAGs).

---

## ⚠ ADDENDUM 2026-06-28 (data rebuild + segmentation) — supersedes the 06-07 addendum where they conflict

**1. The environment moved.** Work now happens on macOS in a repo-local `.venv`
(Python 3.12: `.venv/bin/python`), not the old Windows conda env — every `C:\...` /
`FPY=...` path below is historical. Keys live in `.env`: `OPENAI_API_KEY` and `HF_TOKEN`
are populated; **`OPENROUTER_API_KEY` is NOT** (needed before DAG regeneration / the judge).

**2. The original `data/` was lost and partially rebuilt (see `PROVENANCE.md` — the
authoritative record).** Raw TV re-downloaded; **P5/P6/P7 re-labelled** (2026-06-28,
~$4.25, 0 warnings); split `TV_v1.json` regenerated with the locked recipe (same
2082/900 structure). The prefix-tree gate was re-run and **reproduces**: P5 1.67× /
P6 1.83× / P7 1.47×. STAR artifacts, P8/P10 partial labels, and **all 36 DAGs are gone**
— the June LLM-DAG result JSONs are archived in `experiments/archive_pre_relabel/` and
must never be compared against post-rebuild numbers. DAGs must be regenerated
(OpenRouter) before any Step-2 scoring.

**3. New since 06-07 — the granularity fix (supervisor's whiteboard method):**
`src/fudge/segment.py` collapses consecutive same-bucket utterances into stage-level
segments (mean embedding, min-run smoothing) so conversations match a summary DAG's
granularity *before* FuDGE — the input-side alternative to dwell-FuDGE. Wired into the
scoring experiments via `--segment`. Caveat learned the hard way: TV labels are
agent-only (single `_user_turn` client bucket), so single-bucket streams are left
uncollapsed by design. Also scaffolded: the LLM-judge second metric
(`scripts/llm_judge.py` + `LLM_JUDGE_DESIGN.md` + `LITERATURE_SCAN.md`).

**4. A 2026-06 audit fixed a batch of defects** (fudge_dag now in the test oracle;
variant comparison restricted to common phases; alternation gates DAG validity; empty
completions no longer cached; length-matched p_perm documented as saturated —
read lm_ratio, not p). `git log` has the details.

---

## ⚠ ADDENDUM 2026-06-07 (second session) — supersedes parts of §0/§3/§5/§6

**1. Path explosion was real — fixed with an exact algorithm swap.**
The §6 timing probe hung on deepseek v1/P5 (24 nodes but 5 nested multi-parent diamonds: 243 true
paths → the DFS's re-visit duplication created ~354M distance arrays, 18 GB RAM; v3/P6 same, ~369M).
Fix: `fudge_dag()` in `src/fudge/fudge_efficient.py` — topological-order DP, one distance row per
NODE (elementwise-min merge at reconvergence, exact because the recurrence distributes over min).
**Verified bitwise-identical to `fudge_efficient` on all 15 terminating cells** (max |diff| = 0.0)
and 10–300× faster (explosive cells: 220 ms and 9 ms/conv). Both Step-2 scripts now use it;
Step-1 scripts untouched. The §5.1 risk is closed — gpt-5.1's big DAGs are no longer blocked.

**2. Deepseek-v3.2 scored (discrimination + confusion done)** —
`experiments/llm_dag_discrimination_deepseek-v3.2_r5.json`, `experiments/phase_confusion_deepseek-v3.2_v3_r5.json`.
Raw numbers: all 9 cells pass Bonferroni; v1 1.14× > v2 1.13× > v3 1.12×; confusion: deepseek's P6
flow is column-min for ALL THREE phases. §0's prediction (stronger model fixes P7) was refuted.
**But do not interpret these raw numbers — see point 3.**

**3. MAJOR: the normalized FuDGE score on shallow LLM DAGs is length-dominated.**
Within-phase Spearman corr(score/n, conv length) = **0.89–0.99 for every LLM-DAG cell** (both
models) vs **0.42–0.64 for Step-1 prefix-trees**. Phases differ in length (P5 20.6 / P6 34.2 /
P7 29.8 / P10 63.2 mean utterances) and the out-pool spans all 5 other phases, so the out/in ratio
and all row-wise confusion reads are largely length artifacts. Mechanism: LLM-DAG root→leaf paths
(9–24 nodes) are shorter than the conversations, so uncovered utterances are flat-cost insertions
and score/n is monotone in n. Prefix-trees are depth-matched by construction (full label
sequences), which is why Step 1 passed and the pathology stayed hidden.

**Length-matched re-analysis** (`experiments/length_matched_reanalysis.py`, pure post-processing on
saved scores; results in `experiments/length_matched_reanalysis.json`):
- **Step 1 SURVIVES:** P5 1.67→1.29×, P6 1.82→1.63×, P7 1.46→1.38× (all p≈1e-4). P5 lands at the
  1.3 effect-size bar; majority-pass criterion still met.
- **Step 2 pilot conclusions COLLAPSE:** P5 cells 1.22–1.30× → 1.07–1.09× (~75% artifact); P7
  cells ≈ nothing; only P6 cells keep modest real signal (1.10–1.17×). All variant/model rankings
  end up within ~0.03× of each other → §3's "v3 wins" and any gpt-oss-vs-deepseek ranking are
  withdrawn.

**4. Revised next steps (replaces §6 "after the two-model comparison"):**
1. Supervisor sign-off first — see `SUPERVISOR_UPDATE_2026-06-07.md` (sent 2026-06-08).
2. **Do NOT run kimi-k2 / gpt-5.1 yet** — they would measure length too. Blocking fix: revise
   `prompts.yaml` to require full-session path depth (~30+ alternating actor nodes root→leaf),
   regenerate the two pilot models, confirm length-matched ratios move.
3. Adopt length-matched ratios as the reported discrimination statistic; read confusion matrices
   column-wise only (same convs ⇒ length cancels); rank models by within-conversation paired
   Wilcoxon (already locked as TODO 10.2).
4. Add a random-DAG null baseline (same node count + actor mix, shuffled topology, identical
   alignment) to set the floor LLM DAGs must beat.
5. Generation variance: n≥3 generations per cell before any model-ranking claim (current n=1).
6. Still open from §5: empty-content reroll (§5.4), clinical sign-off on phase descriptions (§5.3),
   label-fallback buckets violate the "never compare to labels" rule on up to 16/89 nodes — prefer
   drop-and-rewire over label fallback.

---

## 0. TL;DR — where we are right now

- **Goal of the project:** rank LLMs by how well each generates a Prolonged-Exposure (PE) therapy
  dialogue-flow DAG, scored against real Thousand Voices (TV) conversations with FuDGE. Two-step
  methodology: Step 1 = validate FuDGE on TV via prefix-tree (DONE, passed); Step 2 = evaluate
  LLM-generated DAGs (IN PROGRESS — this session).
- **This session built and ran the entire Step-2 vertical slice end-to-end on one model (gpt-oss-20b)**
  and is mid-way through a second model (deepseek-v3.2). The pipeline works:
  **generate (phase-conditioned) → align (cluster-then-recentroid) → FuDGE discrimination + confusion matrix.**
- **Headline results (gpt-oss-20b):** the pipeline runs, every cell scores in-phase < out-of-phase,
  **v3 (sequential prompting) is the best variant (3/3 phases pass Bonferroni, mean ratio 1.17×)**, but
  LLM-DAG discrimination is **well below the prefix-tree reference** (1.0–1.33× vs 1.46–1.82×). P5
  (Orientation) separates strongly; P7 (Reinforcing) is weak.
- **Key diagnosis (from the confusion matrix):** P7's weakness is a **DAG-quality problem** (gpt-oss
  produced a *generic* P7 DAG that fits any phase), **not** intrinsic phase overlap. This predicts a
  stronger model should help specifically on P7 — which is exactly what the deepseek arm is testing.

**In flight when this was written:** `align_llm_dags.py --model deepseek-v3.2 --reassign-passes 5` is
running in the background (~1/9 cells done). Next action after it finishes is a **timing probe** on a big
deepseek flow (see §6).

---

## 1. The methodology, in one screen

(Full version in `EXPLAINER.md`. This is the minimum to not break the design.)

- **FuDGE** = fuzzy dialogue-graph edit distance between a real conversation and a DAG. Lower = better
  alignment. Each DAG node holds an `IntentBucket` of real utterance texts; the node's centroid is the
  mean embedding of those texts. **FuDGE compares utterance embeddings to bucket centroids — never to
  label strings.** That's why alignment (populating buckets with real text) is required.
- **Discrimination test** = a DAG built/aligned for phase X should score phase-X held-out test convs
  *lower* than other phases' test convs. Ratio = out_mean / in_mean; >1 with significance = it discriminates.
- **Two non-negotiable rules (locked feedback memories):**
  1. **No circular validation.** The prefix-tree is the *validation reference only*; it must NEVER also be
     a candidate "method" in the LLM comparison.
  2. **Phase is the discrimination axis, not trauma type.** Phase governs dialogue *flow*; type is a
     content attribute used only as a *stratification control* within phase. (See the long answer in the
     session log / `EXPLAINER.md` — a previous session explicitly asked "why phase not type".)
- **Step-1 result (prefix-tree reference, the bar to beat):** P5 1.67×, P6 1.82×, P7 1.46× — all pass
  Bonferroni. LLM DAGs are being compared against this.
- **Scope:** everything is **P5/P6/P7 only**. P8/P10/P11 need labelling finished (TODO 2) before they can
  enter. Models in scope: gpt-oss-20b (done), deepseek-v3.2 (in progress); kimi-k2-0905 and gpt-5.1 not
  yet run with phase-conditioning.

---

## 2. What was built this session (code artifacts)

All committed-ready, all compile. The phase-conditioning + the whole align/score/confusion chain is new.

### Modified
- **`prompts.yaml`** — rewritten to **v2, phase-conditioned**. Added a `phases:` block (P5–P11 with
  `name` / `description` / `moves`) and `{{phase_id}}` / `{{phase_name}}` / `{{phase_description}}` /
  `{{phase_moves}}` slots in prompts 1–5. ⚠ The phase descriptions are drafted from project notes and
  **still need Francesca's clinical sign-off before the final full run** (fine for the pilot).
- **`scripts/generate_llm_dags.py`** — now fills the phase slots per phase (`render_phase_slots`), and
  added `check_dag_validity()` (acyclic / single-component / unknown-actor / alternation checks) written
  to `validity.json` per cell and surfaced in the run log.

### New
- **`src/fudge/llm_dag.py`** — the heart of Step 2:
  - `build_flow_from_llm_dag(dag, train_convs, emb, reassign_passes=0)` → `(DialogueFlow, all_buckets, stats)`.
    Cluster-then-recentroid: embed each node label as an anchor, assign each same-actor TRAINING utterance
    to its nearest node, bucket = assigned texts (label fallback if a node wins none), force acyclic, wire
    root → start nodes. `reassign_passes>0` = the §11 second pass (recompute centroids from assigned
    utterances, re-assign; constrained k-means seeded by labels, K = node count; iterates to convergence).
  - `serialize_flow` / `deserialize_flow` — persist/reload an aligned flow exactly (round-trip verified).
  - `coverage_report` — per-node assignment report for inspecting alignment quality.
- **`experiments/align_llm_dags.py`** — TODO 7 standalone. Builds + persists `aligned.json` (or
  `aligned_r<N>.json`) + `coverage.json`/`coverage_r<N>.json` per cell, prints a per-node coverage summary.
  This is the **inspection point** — eyeball where utterances landed before trusting any score.
- **`experiments/llm_dag_discrimination.py`** — TODO 8 (+ per-variant comparison = TODO 10 for one model).
  `--from-aligned` scores the persisted flows; `--reassign-passes N` selects the `aligned_r<N>` artifacts.
  Mirrors the validated `tv_prefix_tree_discrimination.py` harness (Mann-Whitney + bootstrap, Bonferroni).
- **`experiments/phase_confusion.py`** — phase×phase FuDGE matrix for one model/variant. Reveals *which*
  phase pairs are confusable instead of one pooled ratio.

### Result files produced
- `experiments/llm_dag_discrimination_gpt-oss-20b.json` (one-pass)
- `experiments/llm_dag_discrimination_gpt-oss-20b_r5.json` (second-pass)
- `experiments/phase_confusion_gpt-oss-20b_v3_r5.json`
- `data/dags/<model>/<variant>/<phase>/{dag.json,dag.mmd,validity.json,transcript.json,aligned*.json,coverage*.json}`

---

## 3. Results so far (gpt-oss-20b, P5/P6/P7)

### Discrimination, one-pass vs second-pass (r5)
| cell | one-pass ratio | r5 ratio | notes |
|---|---|---|---|
| v3/P5 | 1.33× | 1.28× | strong; r5 slightly hurts the already-good phase |
| v3/P6 | 1.12× | 1.17× | r5 helps; p 2e-14 → 3e-46 |
| v3/P7 | 1.06× | 1.07× | weak in both |
| v2/P6 | 1.02× (ns) | 1.05× | r5 makes it cross into significance |

**Per-variant mean ratio (Bonferroni pass count):** v1 1.16× (1/2; v1/P6 DAG was empty), v2 1.12→1.13×
(1/3 → 2/3 with r5), **v3 1.17× (3/3)** — v3 wins in both regimes.

**Second-pass verdict:** breaks up "hub" nodes structurally (coverage std dropped 2.4–3.3× on P6/P7) and
sharpens *significance* a lot, but only nudges effect size (+0.01–0.05× on P6/P7) and slightly hurts P5.
**Keep it as default (mild net win), but it is NOT a fix for weak phases.**

### Confusion matrix (gpt-oss v3, r5) — the important diagnostic
```
flow\test     P5       P6       P7
P5          0.602    0.790    0.741     P5 flow: fits P5, rejects P6/P7 (specific)
P6          0.618    0.548*   0.590*    P6 flow: best on P6 AND on P7 (clean)
P7          0.566*   0.699    0.634     P7 flow: fits P5 better than own P7 (promiscuous!)
```
(`*` = column min; lower = better fit). **P7's DAG is too generic** — full of "you're doing well" filler
that loosely matches any phase, so it fails to claim its own convs (flagged `CONFUSED w/ P5`). P6's DAG is
the best discriminator. **Conclusion: the weak P7 number is DAG quality, not intrinsic phase similarity.**

---

## 4. In-flight state (deepseek-v3.2)

- **Generated:** all 9 phase-conditioned DAGs (`logs/gen_deepseek.log`). Deepseek is messier than gpt-oss:
  more cycles (v1/P7 had 13), some fragments, and **bigger DAGs** — v2/P7=103 nodes, v3/P5=89, v3/P7=72.
  The validity guard flagged most as BAD; alignment auto-breaks cycles, drops unknown-actor nodes, and
  root-wires fragments, so they still process — but see the path-explosion risk in §5.
- **Aligning now:** `align_llm_dags.py --model deepseek-v3.2 --reassign-passes 5` running in background
  (~1/9 cells done when this was written; log buffers, so check for `data/dags/deepseek-v3.2/*/*/aligned_r5.json`).

---

## 5. Open risks / things to watch (READ before running deepseek scoring)

1. **FuDGE path explosion on big reconvergent DAGs.** `fudge_efficient` (`src/fudge/fudge_efficient.py`)
   is a tree-oriented DFS with **no visited-set**: cycles → infinite loop (we prevent this by forcing
   acyclic in the flow builder), but **reconvergence (a node with multiple parents) re-expands the
   subtree per path**, which is exponential in the number of nested "diamonds". gpt-oss DAGs (11–28 nodes,
   mostly linear) were fine at ~269 ms/conv. **deepseek's 72–103-node DAGs are the real test.** The
   immediate next action is a **timing probe** (score ~3 convs on deepseek v3/P5, 89 nodes); if it's slow
   (> ~2 s/conv) or hangs, add a guard before any full run. Options for a guard: cap the number of paths
   accumulated per node, or rewrite scoring as a topological-order DP (correct for DAGs, no re-expansion).
2. **Second-pass alignment is only a mild win** — don't oversell it. The real lever for weak phases is DAG
   quality (better model / better prompts), not alignment iteration.
3. **Phase descriptions in `prompts.yaml` are unreviewed** (clinical sign-off pending). OK for pilots,
   must be checked before the final full-panel run.
4. **gpt-oss empty-content quirk.** `gpt-oss-20b v1/P6` returned empty content (all tokens went to the
   reasoning channel) and the **LLM cache persisted the empty result** — a re-run replays the empty. Fix:
   detect empty `content` and retry with a cache-bust + higher `max_tokens`. Not yet implemented.
5. **Scope is P5/P6/P7.** Widening to 6 phases needs TODO 2 labelling (P8 partial, P10 smoke, P11 none)
   finished + TODO 4 re-run for those phases.
6. **Docs are intentionally NOT updated.** The user asked to leave `EXPLAINER.md` / `PROGRESS.md`
   untouched for this session's TODO 7–8 work. So `EXPLAINER.md` §13 still lists TODO 7–8 as not-done —
   that is stale; **this handover is the source of truth.** Don't "fix" the docs unless asked.

---

## 6. Exact next steps (with commands)

**Environment (every command assumes this):**
- Run from repo root: `C:\Users\ratne\fudge_validate`
- Python: `C:\Users\ratne\anaconda3\envs\fudge\python.exe` (the `fudge` conda env — has openai,
  sentence-transformers, torch, scipy, networkx, pyyaml, dotenv). The base python does NOT have openai.
- Experiments import `fudge.*` directly → prefix with `PYTHONPATH=src`. (The generate script inserts
  `src` itself, so it doesn't need it.)
- `OPENROUTER_API_KEY` is in `.env` (loaded via dotenv in the generate script). It's populated.

```bash
FPY="/c/Users/ratne/anaconda3/envs/fudge/python.exe"

# 1. (after deepseek align finishes) TIMING PROBE on the biggest flow before any full run:
PYTHONPATH=src "$FPY" - <<'PY'
import json,time,sys; sys.modules["tensorflow"]=None
from fudge.embeddings import EmbeddingCache
from fudge.llm_dag import deserialize_flow
from fudge.costs import FudgeCosts
from fudge.fudge_efficient import fudge_efficient
from fudge.data_loader import load_thousand_voices_dialogues
from fudge.splits import load_split, split_conversations
emb=EmbeddingCache(); split=load_split("data/splits/TV_v1.json")
a=json.load(open("data/dags/deepseek-v3.2/v3/P5/aligned_r5.json",encoding="utf-8"))
flow,bk=deserialize_flow(a); costs=FudgeCosts(emb,bk)
cv=load_thousand_voices_dialogues("data/thousand-voices-trauma/ThousandVoicesOfTrauma",task_field="type",require_phases=("P5",))
inS=set(split["splits"]["P5"]["train"])|set(split["splits"]["P5"]["test"])
_,test=split_conversations([c for c in cv if c.dialogue_id in inS],split,"P5")
t0=time.time()
for c in test[:3]: fudge_efficient(c,flow,costs)
print("ms/conv ~", round((time.time()-t0)/3*1000))
PY

# 2. If timing is OK (< ~2s/conv): run deepseek discrimination + confusion (background recommended)
PYTHONPATH=src "$FPY" experiments/llm_dag_discrimination.py --model deepseek-v3.2 --from-aligned --reassign-passes 5 > logs/disc_deepseek_r5.log 2>&1
PYTHONPATH=src "$FPY" experiments/phase_confusion.py --model deepseek-v3.2 --variant v3 --reassign-passes 5 > logs/confusion_deepseek_v3_r5.log 2>&1

# 3. Compare models: read the two *_r5.json discrimination files + confusion JSONs.
#    Key question: does deepseek's P7 flow become self-best (specific) vs gpt-oss's promiscuous P7?
```

**After the two-model comparison:**
- If deepseek's bigger DAGs blow up FuDGE → implement the path-explosion guard (§5.1) — this is likely
  needed before gpt-5.1 (its DAGs were 66–109 nodes when phase-blind).
- Run the remaining models (kimi-k2-0905, gpt-5.1) through generate → align r5 → discriminate → confusion.
- Patch the empty-content reroll (§5.4) so no cell silently produces a 0-node DAG.
- THEN the real TODO 6 (cross-model label canonicalisation — only meaningful with ≥2 models), TODO 9
  (AutoEval-ToD Domain Compliance, the independent content metric), TODO 10 (full cross-model table).

---

## 7. Gotchas (Windows / this repo specifically)

- **cp1252 stdout:** printing non-ASCII (Δ, ellipsis, smart quotes) to a redirected stdout raises
  `UnicodeEncodeError`. Keep `print()` ASCII-only. Always `open(..., encoding="utf-8")` — the base python
  defaults to cp1252 and will crash on the unicode in dag.json/coverage.json.
- **Buffered logs:** Python buffers stdout when redirected to a file, so a background run's log can look
  empty while it's actually working — check for output *files* (e.g. `aligned_r5.json`) to confirm progress.
- **grep "binary file matches":** the logs contain tqdm carriage-returns + unicode, so `grep` treats them
  as binary. Filter with `grep -vE "it/s|Loading weights|HF_TOKEN"`, or just read the result/coverage JSON
  with python instead of grepping the log.
- **Model load cost:** every fresh process pays ~100 s to load sentence-transformers + the first embedding
  pass. The `EmbeddingCache` is in-memory per process (no disk cache), so embeddings are re-computed each
  run. Long scoring runs go in the background.
- **`.llm_cache/`** persists LLM responses keyed on (slug, stage, message payload). Changing prompt text
  busts the cache (fresh calls); identical re-runs are free. ~4400 cached entries exist.

---

## 8. File map (what to read / run)

```
HANDOVER.md                  <- this file (current Step-2 truth)
EXPLAINER.md                 conceptual walkthrough (§13 status is STALE re: TODO 7-8)
METHODOLOGY.md               locked v0.2 plan
PROGRESS.md / PIPELINE.md / NOTES.md   older trackers (banners point to EXPLAINER)
prompts.yaml                 v2 phase-conditioned prompts + phases: block
prompts_phase_conditioned_DRAFT.md     the draft these prompts came from (clinical-review notes)

scripts/generate_llm_dags.py            TODO 5 — generate (phase-conditioned + validity guard)
src/fudge/llm_dag.py                     TODO 7 — alignment, serialize/deserialize, coverage, reassign
experiments/align_llm_dags.py            TODO 7 standalone — persist aligned + coverage
experiments/llm_dag_discrimination.py    TODO 8 — FuDGE discrimination (--from-aligned, --reassign-passes)
experiments/phase_confusion.py           phase x phase confusion matrix
experiments/tv_prefix_tree_discrimination.py   Step-1 reference harness (the bar: 1.46-1.82x)

data/dags/<model>/<variant>/<phase>/     dag.json dag.mmd validity.json transcript.json
                                         aligned[_r5].json coverage[_r5].json
data/splits/TV_v1.json                   locked 70/30 phase-stratified split
logs/                                    run logs (filter with grep -vE "it/s|Loading weights")
experiments/llm_dag_discrimination_*.json + phase_confusion_*.json   results
```

**Models / variants:** models routed via OpenRouter (`OPENROUTER_API_KEY`); registry in
`generate_llm_dags.py` (`MODEL_REGISTRY`): deepseek-v3.2, kimi-k2-0905, gpt-oss-20b, gpt-5.1.
Variants: v1 = prompt 1 alone; v2 = prompts 1–5 fused; v3 = prompts 1–5 sequential (v3 has won so far).
```
