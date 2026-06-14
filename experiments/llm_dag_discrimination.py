"""TODO 8 (+ TODO 10 for a single model) — FuDGE discrimination on LLM DAGs.

Mirrors experiments/tv_prefix_tree_discrimination.py, but the per-phase flow is
built from an LLM-generated DAG (data/dags/<model>/<variant>/<phase>/dag.json)
via cluster-then-recentroid (fudge.llm_dag.build_flow_from_llm_dag) instead of
the prefix-tree reference.

Same discrimination logic: a good P5 DAG should score P5 held-out test lower
(better) than the other phases' test conversations. Reports per (variant, phase)
plus a per-variant comparison so you can see which prompting strategy wins.

Run (pilot):
  python experiments/llm_dag_discrimination.py --model gpt-oss-20b
  python experiments/llm_dag_discrimination.py --model gpt-oss-20b --variants v2
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
from fudge.data_loader import load_thousand_voices_dialogues
from fudge.embeddings import EmbeddingCache
from fudge.fudge_efficient import fudge_dag
from fudge.llm_dag import build_flow_from_llm_dag, deserialize_flow
from fudge.splits import load_split, split_conversations

DEFAULT_TV_DIR = "data/thousand-voices-trauma/ThousandVoicesOfTrauma"
SPLIT_PATH = "data/splits/TV_v1.json"
DAGS_ROOT = "data/dags"


def _normalize(score: float, conv) -> float:
    n = len(conv.utterances)
    return score / n if n > 0 else score


def _score(convs, flow, costs, desc):
    # fudge_dag = topological DP, exact same score as fudge_efficient but
    # immune to path explosion on reconvergent DAGs (deepseek v1/P5, v3/P6).
    return np.array([_normalize(fudge_dag(c, flow, costs), c)
                     for c in tqdm(convs, desc=desc, leave=False)])


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
        "gap_ci_lo": float(np.quantile(gaps, alpha / 2)),
        "gap_ci_hi": float(np.quantile(gaps, 1 - alpha / 2)),
        "ratio_mean": float(neg.mean() / pos.mean()) if pos.mean() > 0 else float("inf"),
        "ratio_ci_lo": float(np.quantile(ratios, alpha / 2)),
        "ratio_ci_hi": float(np.quantile(ratios, 1 - alpha / 2)),
    }


def _mannwhitney(pos, neg):
    res = stats.mannwhitneyu(pos, neg, alternative="less")
    n1, n2 = len(pos), len(neg)
    rbc = 1.0 - (2.0 * res.statistic) / (n1 * n2) if n1 * n2 else 0.0
    return {"U": float(res.statistic), "p": float(res.pvalue), "rbc": float(rbc)}


def _load_phase_convs(tv_dir, split_meta, cache):
    """Memoised: phase -> {"train": [...], "test": [...]} Conversation lists."""
    if cache.get("_done"):
        return cache
    for phase in split_meta["splits"]:
        convs = load_thousand_voices_dialogues(tv_dir, task_field="type",
                                               require_phases=(phase,))
        in_split = (set(split_meta["splits"][phase]["train"])
                    | set(split_meta["splits"][phase]["test"]))
        convs = [c for c in convs if c.dialogue_id in in_split]
        train, test = split_conversations(convs, split_meta, phase)
        cache[phase] = {"train": train, "test": test}
    cache["_done"] = True
    return cache


def evaluate(model, variant, phase, dags_root, phase_convs, emb, from_aligned,
             suffix, reassign_passes):
    cell_dir = Path(dags_root) / model / variant / phase
    test = phase_convs[phase]["test"]
    negatives = []
    for other in phase_convs:
        if other in ("_done", phase):
            continue
        negatives.extend(phase_convs[other]["test"])

    if from_aligned:
        # Step 2 proper: score the persisted, inspected flow from align_llm_dags.py.
        apath = cell_dir / f"aligned{suffix}.json"
        if not apath.exists():
            return {"variant": variant, "phase": phase,
                    "skipped": f"no {apath.name} — run align_llm_dags.py "
                               f"--reassign-passes {reassign_passes} first"}
        aligned = json.load(open(apath, encoding="utf-8"))
        flow, all_buckets = deserialize_flow(aligned)
        bstats = aligned.get("stats", {})
        n_train = bstats.get("n_train_align", len(phase_convs[phase]["train"]))
    else:
        dag_path = cell_dir / "dag.json"
        if not dag_path.exists():
            return {"variant": variant, "phase": phase, "skipped": f"no dag at {dag_path}"}
        dag = json.load(open(dag_path, encoding="utf-8"))
        if not dag.get("nodes"):
            return {"variant": variant, "phase": phase, "skipped": "empty dag (0 nodes)"}
        flow, all_buckets, bstats = build_flow_from_llm_dag(
            dag, phase_convs[phase]["train"], emb, reassign_passes=reassign_passes)
        n_train = len(phase_convs[phase]["train"])
    costs = FudgeCosts(emb, all_buckets)

    pos = _score(test, flow, costs, desc=f"{variant}/{phase}/in")
    neg = _score(negatives, flow, costs, desc=f"{variant}/{phase}/out")

    mw = _mannwhitney(pos, neg)
    boot = _bootstrap_gap_ci(pos, neg)
    return {
        "variant": variant, "phase": phase,
        "n_train_align": n_train, "n_test_in": len(test), "n_test_out": len(negatives),
        **{f"dag_{k}": v for k, v in bstats.items()},
        "in_mean": float(pos.mean()), "in_std": float(pos.std(ddof=1)) if len(pos) > 1 else 0.0,
        "out_mean": float(neg.mean()), "out_std": float(neg.std(ddof=1)) if len(neg) > 1 else 0.0,
        **mw, **boot,
        "in_scores": pos.tolist(), "out_scores": neg.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", nargs="+", default=["v1", "v2", "v3"])
    ap.add_argument("--phases", nargs="+", default=["P5", "P6", "P7"])
    ap.add_argument("--tv-dir", default=DEFAULT_TV_DIR)
    ap.add_argument("--split-path", default=SPLIT_PATH)
    ap.add_argument("--dags-root", default=DAGS_ROOT)
    ap.add_argument("--from-aligned", action="store_true",
                    help="Score the persisted aligned.json flows (run align_llm_dags.py first). "
                         "Two-step mode; without it the flow is rebuilt inline.")
    ap.add_argument("--reassign-passes", type=int, default=0,
                    help="Selects which aligned_r<N>.json to score (--from-aligned) or how "
                         "many re-assignment passes to run inline. 0 = one-pass.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = "" if args.reassign_passes == 0 else f"_r{args.reassign_passes}"

    split_meta = load_split(args.split_path)
    print(f"Loaded split {split_meta['version']}: "
          f"{split_meta['n_train_total']} train / {split_meta['n_test_total']} test")
    emb = EmbeddingCache()
    phase_convs = _load_phase_convs(args.tv_dir, split_meta, {})

    results = []
    for variant in args.variants:
        print(f"\n=== {args.model} / {variant} ===")
        for phase in args.phases:
            r = evaluate(args.model, variant, phase, args.dags_root, phase_convs, emb,
                         args.from_aligned, suffix, args.reassign_passes)
            results.append(r)
            if r.get("skipped"):
                print(f"  [skip] {phase}: {r['skipped']}"); continue
            sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else "ns")
            print(f"  {phase:<4} nodes={r['dag_n_nodes']:>3}(a{r['dag_n_agent_nodes']}/u{r['dag_n_user_nodes']}) "
                  f"empty={r['dag_n_empty_buckets']:>2} cut={r['dag_n_backedges_removed']} | "
                  f"in={r['in_mean']:.3f} out={r['out_mean']:.3f} "
                  f"gap={r['gap_mean']:+.3f} ratio={r['ratio_mean']:.2f}x p={r['p']:.1e} {sig}")

    # per-variant comparison (TODO 10, single model)
    print("\n=== Per-variant summary (mean over phases) ===")
    evald = [r for r in results if "p" in r]
    bonf = 0.01 / len(evald) if evald else 1.0
    variant_summ = {}
    for variant in args.variants:
        rs = [r for r in evald if r["variant"] == variant]
        if not rs:
            continue
        n_pass = sum(1 for r in rs if r["p"] < bonf and r["gap_mean"] > 0)
        variant_summ[variant] = {
            "n_phases": len(rs),
            "n_pass_bonf": n_pass,
            "mean_ratio": float(np.mean([r["ratio_mean"] for r in rs])),
            "mean_gap": float(np.mean([r["gap_mean"] for r in rs])),
            "mean_in": float(np.mean([r["in_mean"] for r in rs])),
        }
        s = variant_summ[variant]
        print(f"  {variant}: pass {s['n_pass_bonf']}/{s['n_phases']} "
              f"| mean ratio {s['mean_ratio']:.2f}x | mean gap {s['mean_gap']:+.3f} "
              f"| mean in {s['mean_in']:.3f}")
    if variant_summ:
        best = max(variant_summ, key=lambda v: variant_summ[v]["mean_ratio"])
        print(f"\nBest variant for {args.model}: {best} "
              f"(mean ratio {variant_summ[best]['mean_ratio']:.2f}x) "
              f"[Bonferroni alpha={bonf:.4f}]")

    out = Path(args.out) if args.out else Path(f"experiments/llm_dag_discrimination_{args.model}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "bonferroni_alpha": bonf,
                   "variant_summary": variant_summ, "results": results}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
