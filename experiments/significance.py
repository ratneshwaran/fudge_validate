"""Mann-Whitney U significance test for the FuDGE discrimination experiment.

For each (task, label-regime) combination, computes FuDGE scores for the same
positives / negatives that validate_discrimination.py uses (seeded sample),
then runs a one-sided Mann-Whitney U test (alternative='less': in-task scores
stochastically less than out-of-task scores). Reports U, p-value, and
rank-biserial effect size.

Why this exists: VALIDATION_REPORT.md previously claimed STRONG PASS based
only on a 1-sigma overlap check. That is a weak claim for a paper. This
script gives a quantitative significance + effect-size statement per cell.
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


# Same regimes that appear in VALIDATION_REPORT.md's comparison table.
REGIMES = [
    # (display name, taxonomy_method or None for heuristic, label_method)
    ("heuristic", None, None),
    ("single_prompt + whole", "single_prompt", "whole"),
    ("single_prompt + chunk", "single_prompt", "chunk"),
    ("hybrid + whole", "hybrid", "whole"),
    ("cluster + whole", "cluster", "whole"),
]


def compute_scores(
    task: str,
    by_task: dict,
    all_convs: list,
    emb: EmbeddingCache,
    label_source: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Same sampling as validate_discrimination.run_discrimination_experiment (seed 42)."""
    task_convs = by_task[task]
    other_convs = [c for c in all_convs if c.task != task]

    flow, all_buckets = build_flow_from_conversations(task_convs, label_source=label_source)
    costs = FudgeCosts(emb, all_buckets)

    n_sample = int(len(task_convs) * 0.5)
    np.random.seed(42)
    pos_idx = np.random.choice(len(task_convs), n_sample, replace=False)
    positives = [task_convs[i] for i in pos_idx]
    neg_idx = np.random.choice(
        len(other_convs), min(n_sample, len(other_convs)), replace=False
    )
    negatives = [other_convs[i] for i in neg_idx]

    def score(c):
        s = fudge_efficient(c, flow, costs)
        if len(c.utterances) > 0:
            s /= len(c.utterances)
        return s

    pos_scores = np.array([score(c) for c in tqdm(positives, desc=f"{task} pos", leave=False)])
    neg_scores = np.array([score(c) for c in tqdm(negatives, desc=f"{task} neg", leave=False)])
    return pos_scores, neg_scores


def mannwhitney(pos: np.ndarray, neg: np.ndarray) -> dict:
    """One-sided Mann-Whitney U: H1 is pos stochastically less than neg."""
    res = stats.mannwhitneyu(pos, neg, alternative="less")
    n1, n2 = len(pos), len(neg)
    # Rank-biserial correlation = 1 - 2U/(n1*n2). Range [-1, 1].
    # +1.0 = perfect separation in the expected direction (every pos < every neg).
    rbc = 1.0 - (2.0 * res.statistic) / (n1 * n2) if n1 * n2 else 0.0
    return {
        "U": float(res.statistic),
        "p": float(res.pvalue),
        "rbc": float(rbc),
        "n_pos": n1,
        "n_neg": n2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star-dir", default="data/STAR")
    ap.add_argument("--label-root", default="data/STAR_llm_labels")
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["hotel_book", "bank_fraud_report"],
    )
    args = ap.parse_args()

    print("Loading STAR...")
    convs = load_star_dialogues(args.star_dir)
    by_task = group_by_task(convs)
    print("Initializing embeddings...")
    emb = EmbeddingCache()

    rows: list[dict] = []
    for task in args.tasks:
        for label, tax_method, lab_method in REGIMES:
            print(f"\n>>> {task} | {label}")
            label_source = None
            if tax_method is not None:
                ld = Path(args.label_root) / task / tax_method / lab_method
                if not ld.exists():
                    print(f"  [skip] no labels at {ld}")
                    continue
                label_source = load_llm_labels(ld)
                covered = sum(1 for c in by_task[task] if c.dialogue_id in label_source)
                if covered == 0:
                    print(f"  [skip] no labeled in-task convs at {ld}")
                    continue

            try:
                pos, neg = compute_scores(task, by_task, convs, emb, label_source)
            except Exception as e:
                print(f"  [error] {e}")
                continue

            test = mannwhitney(pos, neg)
            row = {
                "task": task,
                "regime": label,
                "pos_mean": float(pos.mean()),
                "pos_std": float(pos.std()),
                "neg_mean": float(neg.mean()),
                "neg_std": float(neg.std()),
                **test,
            }
            rows.append(row)
            print(
                f"  pos {row['pos_mean']:.4f}±{row['pos_std']:.4f}  "
                f"neg {row['neg_mean']:.4f}±{row['neg_std']:.4f}  "
                f"U={row['U']:.0f}  p={row['p']:.2e}  r={row['rbc']:.3f}"
            )

    print("\n\n=== Mann-Whitney U Results ===\n")
    print(
        f"{'task':<22} {'regime':<24} {'n':>7}  "
        f"{'U':>9}  {'p-value':>12}  {'r (effect)':>10}"
    )
    print("-" * 100)
    for r in rows:
        n_str = f"{r['n_pos']}/{r['n_neg']}"
        print(
            f"{r['task']:<22} {r['regime']:<24} {n_str:>7}  "
            f"{r['U']:>9.0f}  {r['p']:>12.2e}  {r['rbc']:>10.4f}"
        )

    print("\n\n=== Markdown table ===\n")
    print("| task | regime | n (pos/neg) | U | p-value | rank-biserial r |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        n_str = f"{r['n_pos']}/{r['n_neg']}"
        # Use scientific for very small p; otherwise fixed.
        p_disp = f"{r['p']:.2e}" if r["p"] < 1e-3 else f"{r['p']:.4f}"
        print(
            f"| `{r['task']}` | {r['regime']} | {n_str} | "
            f"{r['U']:.0f} | {p_disp} | {r['rbc']:.4f} |"
        )


if __name__ == "__main__":
    main()
