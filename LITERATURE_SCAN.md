# Literature scan — LLM-judge reliability + sequence segmentation

**Date:** 2026-06-20. **Purpose:** ground two design decisions — (a) how to validate the
LLM-judge second metric (Option B), and (b) whether the conversation-segmentation smoothing
(Track 1) needs a named algorithm or the simple run-collapse suffices.

**Provenance note.** Section 1 facts are quoted from the AutoEval-ToD PDF read in full this
session. Section 2/3 facts come from live web search (June 2026). Classic references (Hearst,
Choi, Zheng et al., etc.) are cited from established knowledge and flagged "verify exact cite" —
do not quote their page numbers without checking.

---

## 1. AutoEval-ToD and LLM-judge reliability (PRIMARY)

### AutoEval-ToD — what it actually does
Jain, Aggarwal, Sahay, Dong, Saladi. *AutoEval-ToD: Automated Evaluation of Task-oriented Dialog
Systems.* NAACL 2025, pp. **10133–10148**. <https://aclanthology.org/2025.naacl-long.508/>
(Amazon; judge model = **Claude-3-Sonnet**.)

- It evaluates ToD systems across 5 dimensions; the one we borrow is **Domain Compliance** (§3.4).
- **Domain Compliance mechanism:** feed the LLM (i) predefined domain rules and (ii) a chatbot
  response; the LLM scores each rule for adherence (**Prompt H.3**, scale **1 = compliant /
  0 = non-compliant / −1 = not applicable**); non-compliance triggers an explanation. Reported as
  **% adherence** across rules (their Table 4, e.g. restaurant 99/97/100/100% over 4 rules).
- **Critical for us:** their rules check *response-level adherence*, **not dialogue order/flow**.
  A flow DAG's entire value is order, so we must add **transition rules** ourselves — this is the
  single most important deviation from the paper (see `LLM_JUDGE_DESIGN.md`).

### How AutoEval-ToD validates the judge (the load-bearing fact)
- §7.6: they compute the **raw accuracy** between LLM verdicts and human verdicts per task:
  **97% solution-checker, 96% domain-compliance, 94% response-quality.**
- §E: **inter-annotator agreement 96%** via dual annotation on a **10%** sample; 97% LLM–human
  alignment on solution-checker.
- **The 94–97% figure is raw accuracy / percent-agreement, NOT Cohen's κ or Krippendorff's α.**
  This matters: our docs previously cited "94–97% agreement" loosely; it is accuracy.
- Cost (§A): ≈ **$0.0045 per LLM call** (Claude-3-Sonnet, ~1k in / 100 out), ~**25× cheaper** than
  human (~$0.125, ~3 min/item); Haiku ~3× cheaper still.

### LLM-as-judge reliability — general (web, June 2026)
- Known biases: **position bias** (up to ~75% preference for the first-presented option),
  **verbosity bias** (longer answers preferred regardless of quality), **self-preference /
  self-enhancement** (~10–25% boost for a judge's own outputs; correlated with self-recognition).
  Foundational: Zheng et al., *Judging LLM-as-a-Judge with MT-Bench* (NeurIPS 2023, *verify cite*);
  Liu et al., *G-Eval* (EMNLP 2023, *verify cite*); "Self-Preference Bias in LLM-as-a-Judge"
  (2024/25). Sources: <https://galileo.ai/blog/llm-as-a-judge-vs-human-evaluation>,
  <https://www.researchgate.net/publication/385353198_Self-Preference_Bias_in_LLM-as-a-Judge>.
- Agreement statistics used in the field: Cohen's κ / weighted κ, Fleiss' κ (multi-rater),
  Krippendorff's α, plus Spearman/Kendall correlation with human scores. Caution: reported
  judge–human Fleiss' κ is often modest (~0.3 in some settings), so **raw accuracy alone
  overstates reliability** on imbalanced label distributions.
- Mitigations: option/position shuffling, per-criterion atomic evaluation (one rule at a time)
  with natural-language justifications, length normalization, and **judge ≠ generator**.

**→ Design takeaway (judge validation).** Mirror AutoEval-ToD's hand-scored study (~20 sessions),
but report **Cohen's / weighted κ** (and Krippendorff's α if >2 annotators) **in addition to** raw
accuracy — a chance-corrected statistic is the defensible number and pre-empts the "you're just
reporting accuracy on a skewed label set" critique. Keep the **judge ≠ generator** guard (the code
enforces it) and use a fixed strong Claude judge as the paper did.

---

## 2. Sequence / text segmentation (for the run-collapse smoothing)

Our Track-1 step assigns each turn to its nearest DAG bucket and collapses consecutive same-bucket
runs, with min-run smoothing to absorb an isolated mislabeled turn (the `D G D → D` case). The
question: is the simple heuristic defensible, or should we adopt a named algorithm?

Survey anchor: *Recent Trends in Linear Text Segmentation: A Survey*, Findings of EMNLP 2024.
<https://aclanthology.org/2024.findings-emnlp.174.pdf>

- **TextTiling** (Hearst, 1997, *Computational Linguistics*; *verify cite*) — lexical-cohesion dips
  mark subtopic boundaries. Simple, unsupervised, the canonical baseline a reviewer expects.
- **C99** (Choi, 2000, NAACL; *verify cite*) — divisive clustering on a rank matrix of inter-sentence
  similarity.
- **BayesSeg** (Eisenstein & Barzilay, 2008, EMNLP; *verify cite*) — Bayesian topic-shift model.
- **Change-point detection** — PELT (Killick, Fearnhead & Eckley, 2012, JASA; *verify cite*) and
  Bayesian online CPD (Adams & MacKay, 2007); recently applied as **kernel change-point on
  sentence embeddings** for unsupervised segmentation (arXiv 2026, e.g.
  <https://arxiv.org/html/2601.18788>).
- **HMM / Viterbi smoothing** — model the bucket-label sequence with a transition prior that favours
  staying, decode the smoothed path. A principled probabilistic version of our min-run heuristic.

**→ Design takeaway (segmentation).** Our run-collapse + min-run smoothing is a **median-filter /
morphological smoothing** of a discrete label sequence — defensible and fully deterministic. Keep it
as the default and the headline result, but: (i) cite the survey + TextTiling as the lineage; (ii)
add a **change-point-on-embeddings** (or HMM/Viterbi) variant as a robustness comparison so a
reviewer sees the result isn't an artifact of one ad-hoc smoother. The `--min-run` sweep is our
analogue of a segmentation sensitivity analysis.

---

## 3. Conversation→flow matching and dwell/DTW alignment (brief)

- Dialogue-flow / workflow extraction from conversations is an active area (intent-graph and
  dialogue-flow induction); our prefix-tree and LLM-DAG constructions sit here.
- **DTW** (Dynamic Time Warping) is the standard way to let one template element align to a *run* of
  sequence elements — exactly the "dwell" generalization in `METRIC_OPTIONS.md` Option C. The
  segmentation approach (Track 1) and dwell-FuDGE (Option C) are two routes to the same end
  (many turns ↔ one node); segmentation does it as preprocessing, dwell as a metric change.
- FuDGE itself (arXiv:2411.10416) is graph-edit-distance for dialogue flows — the metric we are
  validating; little prior work applies GED to therapy-phase discrimination, which is the novelty.

**→ Takeaway.** Frame Track 1 (segmentation) and Option C (dwell-FuDGE) as the preprocessing-side
and metric-side instances of the same DTW idea; reporting both, plus the independent LLM judge,
is the "two-to-three metrics agree" story the project already wanted.
