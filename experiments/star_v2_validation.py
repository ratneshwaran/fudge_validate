"""METHODOLOGY.md v0.2 TODO 1 — STAR re-validation with defensible 70/30 split.

Re-runs the FuDGE in-task vs out-of-task discrimination test on STAR using:
  - A versioned, deterministic 70/30 stratified split (one stratum per task).
  - Per-task prefix-tree DAGs built ONLY from each task's training half.
  - Test-only positives (each task's test half) vs out-of-task negatives
    (other tasks' test halves only; never sees test data twice).
  - Mann-Whitney U on the in/out distributions per task; Bonferroni across
    tasks. Bootstrap 95% CI on the in-out gap.

Run:
  python experiments/star_v2_validation.py                 # creates split if missing
  python experiments/star_v2_validation.py --tasks hotel_book bank_fraud_report

This is the Step-1 STAR validation referenced in METHODOLOGY.md v0.2. It is
NOT a method comparison — prefix-tree is the reference construction, not a
candidate to evaluate. See feedback_no_circular_validation memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.modules["tensorflow"] = None

import numpy as np
from scipy import stats
from tqdm import tqdm

from fudge.costs import FudgeCosts
from fudge.data_loader import (
    build_flow_from_conversations,
    group_by_task,
    load_llm_labels,
    load_star_dialogues,
)
from fudge.embeddings import EmbeddingCache
from fudge.fudge_efficient import fudge_efficient
from fudge.splits import (
    create_star_split,
    load_split,
    split_conversations,
)


SPLIT_PATH = Path("data/splits/STAR_v2.json")


def _normalize(score: float, conv) -> float:
    n = len(conv.utterances)
    return score / n if n > 0 else score


def _score_all(convs, flow, costs, desc: str) -> np.ndarray:
    return np.array(
        [_normalize(fudge_efficient(c, flow, costs), c) for c in tqdm(convs, desc=desc, leave=False)]
    )


def _bootstrap_gap_ci(pos: np.ndarray, neg: np.ndarray, n_boot=2000, alpha=0.05, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n_p, n_n = len(pos), len(neg)
    gaps = np.empty(n_boot)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        p = pos[rng.integers(0, n_p, n_p)].mean()
        n = neg[rng.integers(0, n_n, n_n)].mean()
        gaps[b] = n - p
        ratios[b] = n / p if p > 0 else float("inf")
    return {
        "gap_mean": float(neg.mean() - pos.mean()),
        "gap_ci_lo": float(np.quantile(gaps, alpha / 2)),
        "gap_ci_hi": float(np.quantile(gaps, 1 - alpha / 2)),
        "ratio_mean": float(neg.mean() / pos.mean()) if pos.mean() > 0 else float("inf"),
        "ratio_ci_lo": float(np.quantile(ratios, alpha / 2)),
        "ratio_ci_hi": float(np.quantile(ratios, 1 - alpha / 2)),
    }


def _mannwhitney(pos: np.ndarray, neg: np.ndarray) -> dict:
    res = stats.mannwhitneyu(pos, neg, alternative="less")
    n1, n2 = len(pos), len(neg)
    rbc = 1.0 - (2.0 * res.statistic) / (n1 * n2) if n1 * n2 else 0.0
    return {"U": float(res.statistic), "p": float(res.pvalue), "rbc": float(rbc)}


def evaluate_task(
    task: str,
    by_task: dict,
    split_meta: dict,
    emb: EmbeddingCache,
    label_root: Path | None,
    taxonomy_method: str,
    label_method: str,
) -> dict | None:
    """Run discrimination for one task. Returns per-task result dict, or
    None if the task can't be evaluated (no labels, empty test, etc.)."""
    if task not in split_meta["splits"]:
        return None
    if task not in by_task:
        return None

    train, test = split_conversations(by_task[task], split_meta, task)
    if len(train) < 5 or len(test) < 3:
        return {"task": task, "skipped": "train<5 or test<3", "n_train": len(train), "n_test": len(test)}

    # Negatives: every OTHER task's TEST conversations only. Never reuses
    # training data on either side.
    negatives = []
    for other_task, s in split_meta["splits"].items():
        if other_task == task or other_task not in by_task:
            continue
        _, other_test = split_conversations(by_task[other_task], split_meta, other_task)
        negatives.extend(other_test)

    label_source = None
    if label_root is not None:
        ld = label_root / task / taxonomy_method / label_method
        if ld.exists():
            label_source = load_llm_labels(ld)
            train_with_labels = [c for c in train if c.dialogue_id in label_source]
            if len(train_with_labels) < 5:
                return {"task": task, "skipped": f"only {len(train_with_labels)} train convs have LLM labels",
                        "n_train": len(train), "n_test": len(test)}
            train = train_with_labels
        else:
            label_source = None  # fall back to heuristic

    flow, all_buckets = build_flow_from_conversations(train, label_source=label_source)
    costs = FudgeCosts(emb, all_buckets)

    pos_scores = _score_all(test, flow, costs, desc=f"{task}/in")
    neg_scores = _score_all(negatives, flow, costs, desc=f"{task}/out")

    mw = _mannwhitney(pos_scores, neg_scores)
    boot = _bootstrap_gap_ci(pos_scores, neg_scores)

    return {
        "task": task,
        "label_source": "llm" if label_source is not None else "heuristic",
        "taxonomy_method": taxonomy_method if label_source is not None else None,
        "label_method": label_method if label_source is not None else None,
        "n_train": len(train),
        "n_test_in": len(test),
        "n_test_out": len(negatives),
        "n_dag_nodes": flow.num_nodes,
        "in_mean": float(pos_scores.mean()),
        "in_std": float(pos_scores.std(ddof=1)) if len(pos_scores) > 1 else 0.0,
        "out_mean": float(neg_scores.mean()),
        "out_std": float(neg_scores.std(ddof=1)) if len(neg_scores) > 1 else 0.0,
        **mw,
        **boot,
        "in_scores": pos_scores.tolist(),
        "out_scores": neg_scores.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star-dir", default="data/STAR")
    ap.add_argument("--label-root", default="data/STAR_llm_labels",
                    help="Pass empty string to force heuristic labels.")
    ap.add_argument("--taxonomy-method", default="single_prompt",
                    choices=["single_prompt", "hybrid", "cluster"])
    ap.add_argument("--label-method", default="whole",
                    choices=["whole", "window", "chunk"])
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Restrict to a subset of tasks; default = all in split")
    ap.add_argument("--split-path", default=str(SPLIT_PATH))
    ap.add_argument("--out", default="experiments/star_v2_validation.json")
    args = ap.parse_args()

    split_path = Path(args.split_path)
    if not split_path.exists():
        print(f"Creating STAR v2 70/30 stratified split at {split_path} ...")
        create_star_split(args.star_dir, split_path)
    split_meta = load_split(split_path)
    print(f"Loaded split: {split_meta['n_strata']} strata, "
          f"{split_meta['n_train_total']} train / {split_meta['n_test_total']} test "
          f"(seed={split_meta['seed']}, train_frac={split_meta['train_frac']})")
    if split_meta["dropped_tasks"]:
        print(f"  dropped (n < {split_meta['min_per_task']}): "
              f"{[(d['task'], d['n']) for d in split_meta['dropped_tasks']]}")

    print("Loading STAR ...")
    convs = load_star_dialogues(args.star_dir)
    by_task = group_by_task(convs)
    print(f"  {len(convs)} convs across {len(by_task)} tasks")

    print("Initialising embeddings ...")
    emb = EmbeddingCache()

    label_root = Path(args.label_root) if args.label_root else None
    if label_root is None:
        print("Using heuristic labels.")
    else:
        print(f"Using LLM labels from {label_root} "
              f"(taxonomy={args.taxonomy_method}, label={args.label_method}); "
              "falling back to heuristic where labels missing.")

    tasks = args.tasks or list(split_meta["splits"].keys())

    results: list[dict] = []
    for task in tasks:
        r = evaluate_task(task, by_task, split_meta, emb, label_root,
                          args.taxonomy_method, args.label_method)
        if r is None:
            print(f"  [skip] {task}: not in split or no convs")
            continue
        if r.get("skipped"):
            print(f"  [skip] {task}: {r['skipped']}")
            results.append(r)
            continue
        results.append(r)
        sig_flag = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else "ns")
        print(f"  {task:<22} {r['label_source']:<10} "
              f"in={r['in_mean']:.3f}+-{r['in_std']:.3f}  "
              f"out={r['out_mean']:.3f}+-{r['out_std']:.3f}  "
              f"gap={r['gap_mean']:+.3f}[{r['gap_ci_lo']:+.3f},{r['gap_ci_hi']:+.3f}]  "
              f"p={r['p']:.2e} {sig_flag}")

    # Bonferroni across evaluated (non-skipped) tasks
    evaluated = [r for r in results if "p" in r]
    n_tests = len(evaluated)
    bonferroni = 0.01 / n_tests if n_tests else 1.0
    n_pass = sum(1 for r in evaluated if r["p"] < bonferroni and r["gap_mean"] > 0)

    summary = {
        "split_version": split_meta["version"],
        "split_path": str(split_path),
        "n_tasks_evaluated": n_tests,
        "bonferroni_alpha": bonferroni,
        "n_tasks_passing": n_pass,
        "pass_criterion": (
            f"p < {bonferroni:.4f} (Bonferroni 0.01/{n_tests}) AND gap > 0"
        ),
        "verdict": (
            "PASS — FuDGE discriminates on STAR under defensible 70/30 split"
            if n_pass == n_tests and n_tests > 0
            else f"PARTIAL — {n_pass}/{n_tests} tasks pass Bonferroni"
        ),
    }
    print(f"\n=== Summary ===")
    print(f"{n_pass}/{n_tests} tasks pass at Bonferroni alpha={bonferroni:.4f}")
    print(summary["verdict"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results, "split": split_meta}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
