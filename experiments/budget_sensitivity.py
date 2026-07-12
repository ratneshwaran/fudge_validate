"""Budget-sensitivity check for the cluster-then-recentroid alignment (2026-07 review).

The persisted aligned_r5 flows carry converged=False: the constrained k-means was
still reassigning ~3-4% of utterances at the 5-pass cutoff, so a single ABSOLUTE
ratio could in principle depend on the arbitrary pass budget. This rebuilds each
flow INLINE at several budgets in ONE process (one embedding load; EmbeddingCache
memoises across budgets) and reports the length-matched ratio per budget, so the
reader can see whether the headline ratio is stable across the budget rather than
an artifact of it. Reuses the exact scoring harness (llm_dag_discrimination.evaluate)
and standardiser (length_matched_reanalysis.length_matched) — no logic is duplicated.

Note: inline build at reassign_passes=5 reproduces the persisted aligned_r5 flow
bit-for-bit (the alignment is deterministic), so the pass=5 column double-checks the
published numbers.

Run:
  PYTHONPATH=src python experiments/budget_sensitivity.py --model gpt-oss-20b --variant v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.modules["tensorflow"] = None
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from fudge.embeddings import EmbeddingCache
from fudge.splits import load_split

import llm_dag_discrimination as disc
import length_matched_reanalysis as lmr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--variant", default="v3")
    ap.add_argument("--phases", nargs="+", default=["P5", "P6", "P7"])
    ap.add_argument("--passes", nargs="+", type=int, default=[4, 5, 6])
    ap.add_argument("--tv-dir", default=disc.DEFAULT_TV_DIR)
    ap.add_argument("--split-path", default=disc.SPLIT_PATH)
    ap.add_argument("--dags-root", default=disc.DAGS_ROOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    split_meta = load_split(args.split_path)
    split_order = list(split_meta["splits"].keys())
    emb = EmbeddingCache()
    phase_convs = disc._load_phase_convs(args.tv_dir, split_meta, {})
    lens = lmr.load_test_lengths(args.tv_dir, split_meta)

    rows = []
    for phase in args.phases:
        in_l = lens[phase]
        out_l = [x for other in split_order if other != phase for x in lens[other]]
        for p in args.passes:
            for seg in (False, True):
                r = disc.evaluate(args.model, args.variant, phase, args.dags_root,
                                  phase_convs, emb, from_aligned=False, suffix="",
                                  reassign_passes=p, segment=seg)
                if r.get("skipped"):
                    print(f"[skip] {phase} pass={p} seg={seg}: {r['skipped']}")
                    continue
                lm = lmr.length_matched(r["in_scores"], in_l, r["out_scores"], out_l)
                rows.append({
                    "phase": phase, "passes": p, "segment": seg,
                    "raw_ratio": float(np.mean(r["out_scores"]) / np.mean(r["in_scores"])),
                    "lm_ratio": lm["lm_ratio"], "lm_ci_lo": lm["lm_ci_lo"],
                    "lm_ci_hi": lm["lm_ci_hi"], "coverage": lm["coverage"],
                    "converged": r.get("dag_converged"),
                    "n_passes_run": r.get("dag_n_passes_run"),
                })

    # Stability of the segmented length-matched ratio across pass budgets, per phase.
    print(f"\n=== Budget sensitivity: {args.model}/{args.variant} "
          f"(segmented length-matched ratio per pass budget) ===")
    header = "  " + f"{'phase':<6} " + "  ".join(f"r{p:>2}" for p in args.passes) \
             + "   spread  converged"
    print(header)
    stability = {}
    for phase in args.phases:
        segrows = {r["passes"]: r for r in rows if r["phase"] == phase and r["segment"]}
        if not segrows:
            continue
        vals = [segrows[p]["lm_ratio"] for p in args.passes if p in segrows]
        spread = float(max(vals) - min(vals))
        conv = all(bool(segrows[p]["converged"]) for p in segrows)
        stability[phase] = {"values": vals, "spread": spread, "converged": conv}
        cells = "  ".join(f"{segrows[p]['lm_ratio']:.2f}" if p in segrows else "  -  "
                          for p in args.passes)
        print(f"  {phase:<6} {cells}   {spread:+.2f}   {conv}")

    out = Path(args.out) if args.out else Path(
        f"experiments/budget_sensitivity_{args.model}_{args.variant}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "variant": args.variant, "passes": args.passes,
                   "stability": stability, "rows": rows}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
