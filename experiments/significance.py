"""Significance, bootstrap CI, and held-out robustness checks for the FuDGE
discrimination experiment.

For each (task, regime) cell:

  1. Mann-Whitney U test (one-sided, H1: in-task scores < out-of-task scores).
  2. Bootstrap 95% CI on the discrimination ratio (mean(neg)/mean(pos)) by
     resampling the seeded pos/neg score arrays.
  3. Optional --held-out: build the prefix-trie flow from 50% of in-task,
     test on the OTHER 50%. The original validate_discrimination samples
     positives from the same set the flow was built from (in-distribution
     evaluation). Held-out mode quantifies how much of the in-task score
     reduction is fitting vs genuine generalization.

This is the data behind the post-hoc statistical paragraphs in
VALIDATION_REPORT.md.
"""
from __future__ import annotations

import argparse
import sys
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


def compute_scores(
    task: str,
    by_task: dict,
    all_convs: list,
    emb: EmbeddingCache,
    label_source: dict | None,
    held_out: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_scores, neg_scores) for one (task, regime) cell.

    held_out=False (default): flow built from ALL in-task; positives sampled
    from in-task. Same setup as validate_discrimination.

    held_out=True: flow built from 50% in-task (flow_train); positives are
    the OTHER 50% (flow_test). No positive ever appears in the trie.
    """
    task_convs = by_task[task]
    other_convs = [c for c in all_convs if c.task != task]

    n_in = len(task_convs)
    n_sample = n_in // 2

    np.random.seed(42)
    pos_idx = np.random.choice(n_in, n_sample, replace=False)
    neg_idx = np.random.choice(
        len(other_convs), min(n_sample, len(other_convs)), replace=False
    )

    if held_out:
        # flow_train = the half NOT sampled as positives. flow_test = positives.
        flow_train = [c for i, c in enumerate(task_convs) if i not in set(pos_idx)]
    else:
        flow_train = task_convs

    if label_source is not None:
        # When using LLM labels, only conversations with matching labels can
        # contribute to the trie. Filter both pools accordingly.
        flow_train = [c for c in flow_train if c.dialogue_id in label_source]

    flow, all_buckets = build_flow_from_conversations(
        flow_train, label_source=label_source
    )
    costs = FudgeCosts(emb, all_buckets)

    positives = [task_convs[i] for i in pos_idx]
    negatives = [other_convs[i] for i in neg_idx]

    def score(c):
        return _length_normalize(fudge_efficient(c, flow, costs), c)

    pos_scores = np.array([score(c) for c in tqdm(positives, desc=f"{task} pos", leave=False)])
    neg_scores = np.array([score(c) for c in tqdm(negatives, desc=f"{task} neg", leave=False)])
    return pos_scores, neg_scores


def mannwhitney(pos: np.ndarray, neg: np.ndarray) -> dict:
    """One-sided Mann-Whitney U: H1 is pos stochastically less than neg."""
    res = stats.mannwhitneyu(pos, neg, alternative="less")
    n1, n2 = len(pos), len(neg)
    rbc = 1.0 - (2.0 * res.statistic) / (n1 * n2) if n1 * n2 else 0.0
    return {
        "U": float(res.statistic),
        "p": float(res.pvalue),
        "rbc": float(rbc),
        "n_pos": n1,
        "n_neg": n2,
    }


def bootstrap_ratio_ci(
    pos: np.ndarray,
    neg: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI on the ratio mean(neg) / mean(pos).

    Resamples pos and neg independently with replacement n_boot times.
    Returns the point estimate plus the percentile interval.
    """
    rng = np.random.default_rng(seed)
    n_p, n_n = len(pos), len(neg)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        p_idx = rng.integers(0, n_p, n_p)
        n_idx = rng.integers(0, n_n, n_n)
        p_mean = float(pos[p_idx].mean())
        n_mean = float(neg[n_idx].mean())
        ratios[b] = n_mean / p_mean if p_mean > 0 else float("inf")
    point = float(neg.mean() / pos.mean()) if pos.mean() > 0 else float("inf")
    lo = float(np.quantile(ratios, alpha / 2))
    hi = float(np.quantile(ratios, 1 - alpha / 2))
    return {
        "ratio": point,
        "ratio_ci_lo": lo,
        "ratio_ci_hi": hi,
        "n_boot": n_boot,
    }


def run_one_cell(
    task: str,
    by_task: dict,
    all_convs: list,
    emb: EmbeddingCache,
    label_source: dict | None,
    held_out: bool,
    n_boot: int,
) -> dict:
    pos, neg = compute_scores(task, by_task, all_convs, emb, label_source, held_out=held_out)
    mw = mannwhitney(pos, neg)
    boot = bootstrap_ratio_ci(pos, neg, n_boot=n_boot)
    return {
        "task": task,
        "held_out": held_out,
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std()),
        "neg_mean": float(neg.mean()),
        "neg_std": float(neg.std()),
        **mw,
        **boot,
    }


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===\n")
    print(
        f"{'task':<22} {'regime':<24} {'n':>7}  "
        f"{'ratio':>7}  {'95% CI':>20}  {'U':>5}  {'p':>10}  {'r':>8}"
    )
    print("-" * 110)
    for r in rows:
        n_str = f"{r['n_pos']}/{r['n_neg']}"
        ci = f"[{r['ratio_ci_lo']:.2f}, {r['ratio_ci_hi']:.2f}]"
        print(
            f"{r['task']:<22} {r['regime']:<24} {n_str:>7}  "
            f"{r['ratio']:>7.2f}  {ci:>20}  "
            f"{r['U']:>5.0f}  {r['p']:>10.2e}  {r['rbc']:>8.4f}"
        )


def _print_markdown(rows: list[dict], title: str) -> None:
    print(f"\n### {title}\n")
    print("| task | regime | ratio | 95% CI | U | p-value | r |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        ci = f"[{r['ratio_ci_lo']:.2f}, {r['ratio_ci_hi']:.2f}]"
        p_disp = f"{r['p']:.2e}" if r["p"] < 1e-3 else f"{r['p']:.4f}"
        print(
            f"| `{r['task']}` | {r['regime']} | "
            f"{r['ratio']:.2f}× | {ci} | {r['U']:.0f} | {p_disp} | {r['rbc']:.4f} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star-dir", default="data/STAR")
    ap.add_argument("--label-root", default="data/STAR_llm_labels")
    ap.add_argument("--tasks", nargs="+", default=["hotel_book", "bank_fraud_report"])
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap iterations per cell")
    ap.add_argument(
        "--also-held-out",
        action="store_true",
        help="Also run the held-out flow split (build flow from 50% in-task, "
        "test on the other 50%). This is the methodologically stricter regime.",
    )
    args = ap.parse_args()

    print("Loading STAR...")
    convs = load_star_dialogues(args.star_dir)
    by_task = group_by_task(convs)
    print("Initializing embeddings...")
    emb = EmbeddingCache()

    def regime_label_source(task: str, tax_method: str | None, lab_method: str | None):
        if tax_method is None:
            return None
        ld = Path(args.label_root) / task / tax_method / lab_method
        if not ld.exists():
            return "skip-no-dir"
        label_source = load_llm_labels(ld)
        covered = sum(1 for c in by_task[task] if c.dialogue_id in label_source)
        return label_source if covered > 0 else "skip-empty"

    in_dist_rows: list[dict] = []
    held_out_rows: list[dict] = []

    for task in args.tasks:
        for label, tax_method, lab_method in REGIMES:
            print(f"\n>>> {task} | {label}")
            ls = regime_label_source(task, tax_method, lab_method)
            if isinstance(ls, str):  # skip sentinel
                print(f"  [skip] {ls}")
                continue

            r = run_one_cell(task, by_task, convs, emb, ls, held_out=False, n_boot=args.n_boot)
            r["regime"] = label
            in_dist_rows.append(r)
            print(
                f"  in-dist  ratio={r['ratio']:.2f}× CI=[{r['ratio_ci_lo']:.2f}, {r['ratio_ci_hi']:.2f}]  "
                f"U={r['U']:.0f} p={r['p']:.2e}"
            )

            if args.also_held_out:
                r_ho = run_one_cell(task, by_task, convs, emb, ls, held_out=True, n_boot=args.n_boot)
                r_ho["regime"] = label
                held_out_rows.append(r_ho)
                print(
                    f"  held-out ratio={r_ho['ratio']:.2f}× CI=[{r_ho['ratio_ci_lo']:.2f}, {r_ho['ratio_ci_hi']:.2f}]  "
                    f"U={r_ho['U']:.0f} p={r_ho['p']:.2e}"
                )

    _print_table(in_dist_rows, "In-distribution (flow built from all in-task)")
    if held_out_rows:
        _print_table(held_out_rows, "Held-out (flow built from 50% in-task, tested on other 50%)")

    _print_markdown(in_dist_rows, "In-distribution")
    if held_out_rows:
        _print_markdown(held_out_rows, "Held-out flow split")


if __name__ == "__main__":
    main()
