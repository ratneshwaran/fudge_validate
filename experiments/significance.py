"""Significance, multi-split held-out, and between-regime paired tests for
the FuDGE discrimination experiment.

Three modes (combined into one run):

  In-distribution (one fixed flow built from all in-task; bootstrap CI on
  pos/neg score arrays). Mirrors the original validate_discrimination
  setup. Per-cell Mann-Whitney U + 95% bootstrap CI on the ratio.

  Held-out, multi-split. For each of N seeded random 50/50 splits per
  task, build the flow from the train half (for each regime
  independently) and score positives (test half) + negatives (out-of-
  task). Report per-regime ratio mean ± std *across splits*. This
  replaces the previous single-split held-out that was correctly flagged
  as conditioned on one arbitrary partition.

  Between-regime paired tests, on held-out scores. For each conversation
  we average its score across the splits where it was a positive, giving
  one paired score per regime per conversation. We then run a paired
  Wilcoxon signed-rank test between regimes (e.g. LLM vs heuristic) and
  report the median per-conversation Δ + p. This is the right tool for
  the "is regime A equivalent to regime B" question — overlapping
  marginal CIs do not imply equivalence.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.modules["tensorflow"] = None

import numpy as np
from scipy import stats
from tqdm import tqdm

from fudge.data_loader import (
    build_flow_from_conversations,
    group_by_task,
    load_llm_labels,
    load_star_dialogues,
)
from fudge.embeddings import EmbeddingCache
from fudge.costs import FudgeCosts
from fudge.fudge_efficient import fudge_efficient


REGIMES = [
    ("heuristic", None, None),
    ("single_prompt + whole", "single_prompt", "whole"),
    ("single_prompt + chunk", "single_prompt", "chunk"),
    ("hybrid + whole", "hybrid", "whole"),
    ("cluster + whole", "cluster", "whole"),
]


def _length_normalize(score: float, conv) -> float:
    return score / len(conv.utterances) if len(conv.utterances) > 0 else score


# ---------------------------------------------------------------------------
# Per-split scoring
# ---------------------------------------------------------------------------

def split_scores(
    task_convs: list,
    other_convs: list,
    label_source: dict | None,
    emb: EmbeddingCache,
    seed: int,
    held_out: bool,
) -> dict:
    """Sample a 50/50 split (or use all in-task) and score positives + negatives.

    Returns dict with: pos_ids, pos_scores, neg_ids, neg_scores.
    pos_ids is the list of dialogue_ids for positives, in the same order as
    pos_scores. Used by the paired test to align across regimes.
    """
    n_in = len(task_convs)
    n_sample = n_in // 2
    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(n_in, n_sample, replace=False)
    neg_idx = rng.choice(
        len(other_convs), min(n_sample, len(other_convs)), replace=False
    )

    if held_out:
        flow_train = [c for i, c in enumerate(task_convs) if i not in set(pos_idx)]
    else:
        flow_train = task_convs

    if label_source is not None:
        flow_train = [c for c in flow_train if c.dialogue_id in label_source]

    flow, all_buckets = build_flow_from_conversations(
        flow_train, label_source=label_source
    )
    costs = FudgeCosts(emb, all_buckets)

    positives = [task_convs[i] for i in pos_idx]
    negatives = [other_convs[i] for i in neg_idx]

    def score(c):
        return _length_normalize(fudge_efficient(c, flow, costs), c)

    pos_scores = np.array([score(c) for c in positives])
    neg_scores = np.array([score(c) for c in negatives])
    pos_ids = [c.dialogue_id for c in positives]
    neg_ids = [c.dialogue_id for c in negatives]
    return {
        "pos_ids": pos_ids,
        "pos_scores": pos_scores,
        "neg_ids": neg_ids,
        "neg_scores": neg_scores,
    }


def mannwhitney(pos: np.ndarray, neg: np.ndarray) -> dict:
    res = stats.mannwhitneyu(pos, neg, alternative="less")
    n1, n2 = len(pos), len(neg)
    rbc = 1.0 - (2.0 * res.statistic) / (n1 * n2) if n1 * n2 else 0.0
    return {"U": float(res.statistic), "p": float(res.pvalue), "rbc": float(rbc),
            "n_pos": n1, "n_neg": n2}


def bootstrap_ratio_ci(pos, neg, n_boot=2000, alpha=0.05, seed=0) -> dict:
    """Score-array percentile bootstrap (in-distribution only — explicitly
    *not* a CI for held-out generalization, which needs flow re-training)."""
    rng = np.random.default_rng(seed)
    n_p, n_n = len(pos), len(neg)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        p = pos[rng.integers(0, n_p, n_p)].mean()
        n = neg[rng.integers(0, n_n, n_n)].mean()
        ratios[b] = n / p if p > 0 else float("inf")
    return {
        "ratio": float(neg.mean() / pos.mean()) if pos.mean() > 0 else float("inf"),
        "ratio_ci_lo": float(np.quantile(ratios, alpha / 2)),
        "ratio_ci_hi": float(np.quantile(ratios, 1 - alpha / 2)),
    }


# ---------------------------------------------------------------------------
# Multi-split held-out aggregation
# ---------------------------------------------------------------------------

def run_multisplit(
    task: str,
    by_task: dict,
    all_convs: list,
    emb: EmbeddingCache,
    regimes: list,
    label_sources: dict,
    n_splits: int,
) -> dict:
    """For one task, run all regimes across n_splits seeded held-out splits.

    Returns:
      regime_split_ratios: dict regime -> list[float] of length n_splits
      regime_pos_scores:   dict regime -> dict[conv_id -> list[float]]
      regime_neg_scores:   dict regime -> dict[conv_id -> list[float]]
      pos_ids_per_split:   list[list[int]] (for paired test ordering)
    """
    task_convs = by_task[task]
    other_convs = [c for c in all_convs if c.task != task]

    regime_split_ratios = {r[0]: [] for r in regimes}
    regime_pos_scores: dict[str, dict[int, list[float]]] = {r[0]: {} for r in regimes}
    regime_neg_scores: dict[str, dict[int, list[float]]] = {r[0]: {} for r in regimes}
    regime_mw_per_split: dict[str, list[dict]] = {r[0]: [] for r in regimes}

    for split_i in tqdm(range(n_splits), desc=f"{task} splits", leave=False):
        seed = split_i  # one seed per split, shared across regimes
        for label, _, _ in regimes:
            ls = label_sources.get(label)
            if ls == "skip":
                continue
            res = split_scores(
                task_convs, other_convs, ls if ls is not None else None,
                emb, seed=seed, held_out=True,
            )
            pos = res["pos_scores"]
            neg = res["neg_scores"]
            ratio = float(neg.mean() / pos.mean()) if pos.mean() > 0 else float("inf")
            regime_split_ratios[label].append(ratio)
            regime_mw_per_split[label].append(mannwhitney(pos, neg))
            for cid, s in zip(res["pos_ids"], pos):
                regime_pos_scores[label].setdefault(cid, []).append(float(s))
            for cid, s in zip(res["neg_ids"], neg):
                regime_neg_scores[label].setdefault(cid, []).append(float(s))

    return {
        "regime_split_ratios": regime_split_ratios,
        "regime_pos_scores": regime_pos_scores,
        "regime_neg_scores": regime_neg_scores,
        "regime_mw_per_split": regime_mw_per_split,
    }


def aggregate_regime(label: str, mr: dict) -> dict:
    ratios = mr["regime_split_ratios"][label]
    pos_avg = {cid: float(np.mean(scores))
               for cid, scores in mr["regime_pos_scores"][label].items()}
    neg_avg = {cid: float(np.mean(scores))
               for cid, scores in mr["regime_neg_scores"][label].items()}
    pos_arr = np.array(list(pos_avg.values()))
    neg_arr = np.array(list(neg_avg.values()))
    if not ratios or not len(pos_arr) or not len(neg_arr):
        return {"regime": label, "n_splits": 0}
    mw = mannwhitney(pos_arr, neg_arr)
    return {
        "regime": label,
        "n_splits": len(ratios),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "n_pos_unique": len(pos_avg),
        "n_neg_unique": len(neg_avg),
        **mw,
    }


def paired_compare(
    label_a: str, label_b: str, mr: dict
) -> dict:
    """Paired Wilcoxon signed-rank between two regimes on per-conversation
    averaged held-out scores. Conversations must have appeared as a
    positive under both regimes; we intersect on dialogue_id."""
    pa = mr["regime_pos_scores"][label_a]
    pb = mr["regime_pos_scores"][label_b]
    common = sorted(set(pa.keys()) & set(pb.keys()))
    if len(common) < 5:
        return {"a": label_a, "b": label_b, "n": len(common), "skipped": True}
    avg_a = np.array([np.mean(pa[c]) for c in common])
    avg_b = np.array([np.mean(pb[c]) for c in common])
    diffs = avg_a - avg_b  # negative means a < b (a scores lower / matches better)
    if np.allclose(diffs, 0):
        return {"a": label_a, "b": label_b, "n": len(common),
                "median_diff": 0.0, "p_two_sided": 1.0,
                "p_a_less": 1.0, "p_a_greater": 1.0}
    res_two = stats.wilcoxon(avg_a, avg_b, alternative="two-sided", zero_method="wilcox")
    res_lt = stats.wilcoxon(avg_a, avg_b, alternative="less", zero_method="wilcox")
    res_gt = stats.wilcoxon(avg_a, avg_b, alternative="greater", zero_method="wilcox")
    return {
        "a": label_a,
        "b": label_b,
        "n": len(common),
        "median_diff": float(np.median(diffs)),
        "p_two_sided": float(res_two.pvalue),
        "p_a_less": float(res_lt.pvalue),
        "p_a_greater": float(res_gt.pvalue),
    }


# ---------------------------------------------------------------------------
# In-distribution single-split (still useful as an ablation against held-out)
# ---------------------------------------------------------------------------

def run_in_distribution(task, by_task, all_convs, emb, regimes, label_sources):
    rows = []
    for label, _, _ in regimes:
        ls = label_sources.get(label)
        if ls == "skip":
            continue
        res = split_scores(
            by_task[task], [c for c in all_convs if c.task != task],
            ls if ls is not None else None, emb, seed=42, held_out=False,
        )
        mw = mannwhitney(res["pos_scores"], res["neg_scores"])
        boot = bootstrap_ratio_ci(res["pos_scores"], res["neg_scores"])
        rows.append({"task": task, "regime": label, **mw, **boot})
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_in_distribution(rows):
    print("\n=== In-distribution (single seeded split, score-array bootstrap CI) ===\n")
    print(f"{'task':<22} {'regime':<24} {'ratio':>7}  {'95% CI':>20}  {'p':>10}  {'r':>8}")
    print("-" * 100)
    for r in rows:
        ci = f"[{r['ratio_ci_lo']:.2f}, {r['ratio_ci_hi']:.2f}]"
        print(f"{r['task']:<22} {r['regime']:<24} {r['ratio']:>7.2f}  "
              f"{ci:>20}  {r['p']:>10.2e}  {r['rbc']:>8.4f}")


def _print_held_out_aggregate(rows, n_splits):
    print(f"\n=== Held-out, {n_splits} random 50/50 splits per task ===\n")
    print(f"{'task':<22} {'regime':<24} "
          f"{'ratio mean':>11}  {'± std':>7}  {'min':>5}  {'max':>5}  "
          f"{'pooled p':>10}  {'r':>8}")
    print("-" * 110)
    for r in rows:
        if r.get("n_splits", 0) == 0:
            continue
        print(f"{r['task']:<22} {r['regime']:<24} "
              f"{r['ratio_mean']:>11.2f}  ±{r['ratio_std']:>5.2f}  "
              f"{r['ratio_min']:>5.2f}  {r['ratio_max']:>5.2f}  "
              f"{r['p']:>10.2e}  {r['rbc']:>8.4f}")


def _print_paired(rows, n_splits):
    print(f"\n=== Paired comparisons on held-out scores "
          f"(per-conv averaged across {n_splits} splits, Wilcoxon signed-rank) ===\n")
    print(f"{'task':<22} {'A':<24} {'B':<24} "
          f"{'n':>4}  {'median delta(A-B)':>15}  {'p (two-sided)':>14}")
    print("-" * 110)
    for r in rows:
        if r.get("skipped"):
            continue
        p_disp = f"{r['p_two_sided']:.2e}" if r['p_two_sided'] < 1e-3 else f"{r['p_two_sided']:.4f}"
        print(f"{r['task']:<22} {r['a']:<24} {r['b']:<24} "
              f"{r['n']:>4}  {r['median_diff']:>+15.4f}  {p_disp:>14}")


def _markdown_in_dist(rows):
    print("\n### In-distribution (one fixed flow per regime; bootstrap CI on score arrays)\n")
    print("| task | regime | ratio | 95% CI | U | p (1-sided) | r |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        ci = f"[{r['ratio_ci_lo']:.2f}, {r['ratio_ci_hi']:.2f}]"
        p_disp = f"{r['p']:.2e}" if r['p'] < 1e-3 else f"{r['p']:.4f}"
        print(f"| `{r['task']}` | {r['regime']} | "
              f"{r['ratio']:.2f}x | {ci} | {r['U']:.0f} | {p_disp} | {r['rbc']:.4f} |")


def _markdown_held_out(rows, n_splits):
    print(f"\n### Held-out across {n_splits} random 50/50 splits "
          f"(flow rebuilt each split)\n")
    print("| task | regime | ratio mean ± std | range | pooled p (1-sided) | r |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r.get("n_splits", 0) == 0:
            continue
        rng_str = f"[{r['ratio_min']:.2f}, {r['ratio_max']:.2f}]"
        p_disp = f"{r['p']:.2e}" if r['p'] < 1e-3 else f"{r['p']:.4f}"
        print(f"| `{r['task']}` | {r['regime']} | "
              f"{r['ratio_mean']:.2f}x ± {r['ratio_std']:.2f} | "
              f"{rng_str} | {p_disp} | {r['rbc']:.4f} |")


def _markdown_paired(rows):
    print("\n### Between-regime paired comparisons (held-out per-conv averaged scores)\n")
    print("| task | A | B | n | median delta (A - B) | p two-sided | p (A < B) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("skipped"):
            continue
        p2 = f"{r['p_two_sided']:.2e}" if r['p_two_sided'] < 1e-3 else f"{r['p_two_sided']:.4f}"
        plt = f"{r['p_a_less']:.2e}" if r['p_a_less'] < 1e-3 else f"{r['p_a_less']:.4f}"
        print(f"| `{r['task']}` | {r['a']} | {r['b']} | {r['n']} | "
              f"{r['median_diff']:+.4f} | {p2} | {plt} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star-dir", default="data/STAR")
    ap.add_argument("--label-root", default="data/STAR_llm_labels")
    ap.add_argument("--tasks", nargs="+", default=["hotel_book", "bank_fraud_report"])
    ap.add_argument("--n-splits", type=int, default=10,
                    help="Number of random 50/50 splits for held-out evaluation")
    ap.add_argument(
        "--paired-against",
        default="heuristic",
        help="Regime to use as the reference (B) in paired comparisons. "
        "All other regimes (A) are compared against it.",
    )
    args = ap.parse_args()

    print("Loading STAR...")
    convs = load_star_dialogues(args.star_dir)
    by_task = group_by_task(convs)
    print("Initializing embeddings...")
    emb = EmbeddingCache()

    def make_label_sources(task: str) -> dict:
        d: dict = {}
        for label, tax_method, lab_method in REGIMES:
            if tax_method is None:
                d[label] = None
                continue
            ld = Path(args.label_root) / task / tax_method / lab_method
            if not ld.exists():
                d[label] = "skip"
                continue
            d[label] = load_llm_labels(ld)
        return d

    in_dist_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    paired_rows: list[dict] = []

    for task in args.tasks:
        ls = make_label_sources(task)
        regimes_avail = [r for r in REGIMES if ls.get(r[0]) != "skip"]

        # In-distribution (single fixed split, the original report numbers).
        print(f"\n>>> {task} | in-distribution (single fixed flow)")
        in_dist_rows.extend(run_in_distribution(task, by_task, convs, emb, regimes_avail, ls))

        # Held-out, multiple seeded splits.
        print(f">>> {task} | held-out, {args.n_splits} splits")
        mr = run_multisplit(task, by_task, convs, emb, regimes_avail, ls, args.n_splits)

        for label, _, _ in regimes_avail:
            agg = aggregate_regime(label, mr)
            agg["task"] = task
            aggregate_rows.append(agg)

        # Paired comparisons against the reference regime.
        ref = args.paired_against
        if ref in [r[0] for r in regimes_avail]:
            for label, _, _ in regimes_avail:
                if label == ref:
                    continue
                pc = paired_compare(label, ref, mr)
                pc["task"] = task
                paired_rows.append(pc)

    _print_in_distribution(in_dist_rows)
    _print_held_out_aggregate(aggregate_rows, args.n_splits)
    _print_paired(paired_rows, args.n_splits)

    print("\n\n==== Markdown ====")
    _markdown_in_dist(in_dist_rows)
    _markdown_held_out(aggregate_rows, args.n_splits)
    _markdown_paired(paired_rows)


if __name__ == "__main__":
    main()
