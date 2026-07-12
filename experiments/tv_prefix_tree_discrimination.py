"""METHODOLOGY.md v0.2 TODO 4 — TV prefix-tree discrimination.

Mirrors experiments/star_v2_validation.py but on TV:
  - Stratum = PE phase (P5, P6, P7, P8, P10, P11).
  - Per-phase prefix-tree DAG built from training labels
    (data/TV_llm_labels/<phase>/<taxonomy_method>/whole/<id>.json).
  - Positives = each phase's test conversations. Negatives = OTHER phases'
    test conversations only.
  - Mann-Whitney U + bootstrap gap CI per phase; Bonferroni across phases.

This is the Step-1 TV validation. If it passes, FuDGE works on mental-health
dialogue and Step-2 LLM-DAG evaluation is unblocked. If it fails, we diagnose
(label quality, trie shape, embedding model) before any LLM-DAG work.

Run:
  python experiments/tv_prefix_tree_discrimination.py
  python experiments/tv_prefix_tree_discrimination.py --taxonomy-method hybrid
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
    load_llm_labels,
    load_thousand_voices_dialogues,
)
from fudge.embeddings import EmbeddingCache
from fudge.fudge_efficient import fudge_efficient
from fudge.segment import segment_conversation
from fudge.splits import load_split, split_conversations


SPLIT_PATH = Path("data/splits/TV_v1.json")
DEFAULT_TV_DIR = "data/thousand-voices-trauma/ThousandVoicesOfTrauma"
DEFAULT_LABEL_ROOT = Path("data/TV_llm_labels")


def _normalize(score: float, conv) -> float:
    n = len(conv.utterances)
    return score / n if n > 0 else score


def _score(convs, flow, costs, desc):
    return np.array(
        [_normalize(fudge_efficient(c, flow, costs), c) for c in tqdm(convs, desc=desc, leave=False)]
    )


def _bootstrap_gap_ci(pos, neg, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_p, n_n = len(pos), len(neg)
    gaps = np.empty(n_boot); ratios = np.empty(n_boot)
    for b in range(n_boot):
        p = pos[rng.integers(0, n_p, n_p)].mean()
        n = neg[rng.integers(0, n_n, n_n)].mean()
        gaps[b] = n - p
        ratios[b] = n / p if p > 0 else float("inf")
    return {
        "gap_mean": float(neg.mean() - pos.mean()),
        "gap_ci_lo": float(np.quantile(gaps, alpha/2)),
        "gap_ci_hi": float(np.quantile(gaps, 1 - alpha/2)),
        "ratio_mean": float(neg.mean() / pos.mean()) if pos.mean() > 0 else float("inf"),
        "ratio_ci_lo": float(np.quantile(ratios, alpha/2)),
        "ratio_ci_hi": float(np.quantile(ratios, 1 - alpha/2)),
    }


def _mannwhitney(pos, neg):
    res = stats.mannwhitneyu(pos, neg, alternative="less")
    n1, n2 = len(pos), len(neg)
    rbc = 1.0 - (2.0 * res.statistic) / (n1 * n2) if n1 * n2 else 0.0
    return {"U": float(res.statistic), "p": float(res.pvalue), "rbc": float(rbc)}


def evaluate_phase(
    phase: str,
    tv_dir: str,
    split_meta: dict,
    emb: EmbeddingCache,
    label_root: Path,
    taxonomy_method: str,
    segment: bool = False,
    min_run: int = 2,
) -> dict | None:
    convs = load_thousand_voices_dialogues(tv_dir, require_phases=(phase,))
    label_dir = label_root / phase / taxonomy_method / "whole"
    if not label_dir.exists():
        return {"phase": phase, "skipped": f"no labels at {label_dir}"}
    labels = load_llm_labels(label_dir)

    # Drop convs not in split (small dropped types) and convs without labels.
    in_split = set(split_meta["splits"][phase]["train"]) | set(split_meta["splits"][phase]["test"])
    convs = [c for c in convs if c.dialogue_id in in_split and c.dialogue_id in labels]

    train, test = split_conversations(convs, split_meta, phase)
    train_labelled = [c for c in train if c.dialogue_id in labels]
    if len(train_labelled) < 5 or len(test) < 5:
        return {"phase": phase, "skipped": f"train_labelled={len(train_labelled)} test={len(test)}"}

    # Negatives: every OTHER phase's TEST conversations only.
    negatives = []
    for other in split_meta["splits"]:
        if other == phase:
            continue
        other_convs = load_thousand_voices_dialogues(tv_dir, require_phases=(other,))
        other_in = set(split_meta["splits"][other]["test"])
        negatives.extend(c for c in other_convs if c.dialogue_id in other_in)

    flow, all_buckets = build_flow_from_conversations(train_labelled, label_source=labels)
    costs = FudgeCosts(emb, all_buckets)

    # Sanity re-validation under --segment. NOTE: TV labels are agent-only, so
    # the user side is a single `_user_turn` bucket — segment_conversation leaves
    # single-bucket streams uncollapsed, but the agent stream still collapses
    # across the ~8-20 label taxonomy, so ratios WILL move. The check here is
    # that discrimination significance survives segmentation, not invariance.
    if segment:
        test = [segment_conversation(c, all_buckets, emb, min_run=min_run) for c in test]
        negatives = [segment_conversation(c, all_buckets, emb, min_run=min_run)
                     for c in negatives]

    pos_scores = _score(test, flow, costs, desc=f"{phase}/in")
    neg_scores = _score(negatives, flow, costs, desc=f"{phase}/out")

    mw = _mannwhitney(pos_scores, neg_scores)
    boot = _bootstrap_gap_ci(pos_scores, neg_scores)
    return {
        "phase": phase,
        "taxonomy_method": taxonomy_method,
        "n_train": len(train_labelled),
        "n_test_in": len(test),
        "n_test_out": len(negatives),
        "n_dag_nodes": flow.num_nodes,
        "segmented": segment, "min_run": min_run if segment else None,
        "in_mean": float(pos_scores.mean()),
        "in_std": float(pos_scores.std(ddof=1)) if len(pos_scores) > 1 else 0.0,
        "out_mean": float(neg_scores.mean()),
        "out_std": float(neg_scores.std(ddof=1)) if len(neg_scores) > 1 else 0.0,
        **mw, **boot,
        "in_scores": pos_scores.tolist(),
        "out_scores": neg_scores.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tv-dir", default=DEFAULT_TV_DIR)
    ap.add_argument("--label-root", default=str(DEFAULT_LABEL_ROOT))
    ap.add_argument("--taxonomy-method", default="single_prompt",
                    choices=["single_prompt", "hybrid"])
    ap.add_argument("--phases", nargs="+", default=None)
    ap.add_argument("--split-path", default=str(SPLIT_PATH))
    ap.add_argument("--segment", action="store_true",
                    help="Re-validate under granularity-normalised FuDGE (segment test convs "
                         "against the prefix-tree buckets). Agent-side collapse shifts the "
                         "ratios; the check is that significance survives.")
    ap.add_argument("--min-run", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    split_meta = load_split(args.split_path)
    print(f"Loaded split {split_meta['version']}: "
          f"{split_meta['n_train_total']} train / {split_meta['n_test_total']} test "
          f"(seed={split_meta['seed']}, train_frac={split_meta['train_frac']})")

    emb = EmbeddingCache()
    label_root = Path(args.label_root)
    phases = args.phases or list(split_meta["splits"].keys())

    results = []
    for phase in phases:
        r = evaluate_phase(phase, args.tv_dir, split_meta, emb, label_root, args.taxonomy_method,
                           segment=args.segment, min_run=args.min_run)
        if r is None:
            print(f"  [skip] {phase}: no convs"); continue
        if r.get("skipped"):
            print(f"  [skip] {phase}: {r['skipped']}"); results.append(r); continue
        results.append(r)
        sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else "ns")
        print(f"  {phase:<5}  nodes={r['n_dag_nodes']:>4}  "
              f"in={r['in_mean']:.3f}+-{r['in_std']:.3f}  "
              f"out={r['out_mean']:.3f}+-{r['out_std']:.3f}  "
              f"gap={r['gap_mean']:+.3f}[{r['gap_ci_lo']:+.3f},{r['gap_ci_hi']:+.3f}]  "
              f"ratio={r['ratio_mean']:.2f}x  p={r['p']:.2e} {sig}")

    evaluated = [r for r in results if "p" in r]
    n = len(evaluated)
    bonf = 0.01 / n if n else 1.0
    n_pass = sum(1 for r in evaluated if r["p"] < bonf and r["gap_mean"] > 0)
    summary = {
        "split_version": split_meta["version"],
        "taxonomy_method": args.taxonomy_method,
        "n_phases_evaluated": n,
        "bonferroni_alpha": bonf,
        "n_phases_passing": n_pass,
        "pass_criterion": f"p < {bonf:.4f} (Bonferroni 0.01/{n}) AND gap > 0",
        "verdict": (
            "PASS — FuDGE discriminates on TV; Step-2 LLM-DAG evaluation unblocked"
            if n_pass == n and n > 0
            else f"PARTIAL — {n_pass}/{n} phases pass Bonferroni; diagnose before Step 2"
        ),
    }
    print(f"\n=== Summary ===")
    print(f"{n_pass}/{n} phases pass at Bonferroni alpha={bonf:.4f}")
    print(summary["verdict"])

    out = Path(args.out) if args.out else Path(
        f"experiments/tv_prefix_tree_discrimination{'_seg' if args.segment else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results, "split": split_meta}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
