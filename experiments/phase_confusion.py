"""Phase x phase FuDGE confusion matrix for one model/variant.

The pooled in-vs-out ratio in llm_dag_discrimination.py hides *which* phases are
confusable. Here we score every phase's TEST conversations against every phase's
DAG flow, giving a matrix M where M[i][j] = mean normalised FuDGE of phase-j test
convs under phase-i's flow.

Reading it: for a well-separated set of DAGs, each COLUMN j should be minimised on
the diagonal (phase j's own flow fits phase j's convs best). Off-diagonal cells
near the diagonal value = those two phases are confusable.

Scores the persisted aligned flows (run align_llm_dags.py first). Match
--reassign-passes to the alignment you want to read.

Run:
  python experiments/phase_confusion.py --model gpt-oss-20b --variant v3 --reassign-passes 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.modules["tensorflow"] = None

import numpy as np
from tqdm import tqdm

from fudge.costs import FudgeCosts
from fudge.data_loader import load_thousand_voices_dialogues
from fudge.embeddings import EmbeddingCache
from fudge.fudge_efficient import fudge_dag
from fudge.llm_dag import deserialize_flow
from fudge.splits import load_split, split_conversations

DEFAULT_TV_DIR = "data/thousand-voices-trauma/ThousandVoicesOfTrauma"
SPLIT_PATH = "data/splits/TV_v1.json"
DAGS_ROOT = "data/dags"


def _norm_scores(convs, flow, costs, desc):
    # fudge_dag = topological DP, exact same score as fudge_efficient but
    # immune to path explosion on reconvergent DAGs (deepseek v1/P5, v3/P6).
    return np.array([fudge_dag(c, flow, costs) / max(1, len(c.utterances))
                     for c in tqdm(convs, desc=desc, leave=False)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="v3")
    ap.add_argument("--phases", nargs="+", default=["P5", "P6", "P7"])
    ap.add_argument("--reassign-passes", type=int, default=0)
    ap.add_argument("--tv-dir", default=DEFAULT_TV_DIR)
    ap.add_argument("--split-path", default=SPLIT_PATH)
    ap.add_argument("--dags-root", default=DAGS_ROOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = "" if args.reassign_passes == 0 else f"_r{args.reassign_passes}"

    split_meta = load_split(args.split_path)
    emb = EmbeddingCache()

    # test sets per phase
    test_by_phase = {}
    for phase in args.phases:
        convs = load_thousand_voices_dialogues(args.tv_dir, task_field="type",
                                               require_phases=(phase,))
        in_split = (set(split_meta["splits"][phase]["train"])
                    | set(split_meta["splits"][phase]["test"]))
        convs = [c for c in convs if c.dialogue_id in in_split]
        test_by_phase[phase] = split_conversations(convs, split_meta, phase)[1]

    # flow per phase (from aligned artifacts)
    flows = {}
    for phase in args.phases:
        apath = Path(args.dags_root) / args.model / args.variant / phase / f"aligned{suffix}.json"
        if not apath.exists():
            print(f"[skip] {phase}: no {apath.name} — run align_llm_dags.py "
                  f"--reassign-passes {args.reassign_passes} for {args.model}/{args.variant}")
            continue
        aligned = json.load(open(apath, encoding="utf-8"))
        flow, buckets = deserialize_flow(aligned)
        flows[phase] = (flow, FudgeCosts(emb, buckets))
    flow_phases = [p for p in args.phases if p in flows]

    # M[flow_phase][test_phase]
    M = {fp: {} for fp in flow_phases}
    for fp in flow_phases:
        flow, costs = flows[fp]
        for tp in args.phases:
            M[fp][tp] = float(_norm_scores(test_by_phase[tp], flow, costs,
                                           desc=f"flow {fp} / test {tp}").mean())

    # ---- print matrix (rows = flow/DAG, cols = test convs) ----
    print(f"\n=== {args.model}/{args.variant}{suffix} — mean normalised FuDGE ===")
    print("rows = DAG flow, cols = test convs. Lower = better fit. "
          "Diagonal should be the column min.\n")
    hdr = "flow\\test  " + "".join(f"{tp:>9}" for tp in args.phases)
    print(hdr)
    col_min = {tp: min(M[fp][tp] for fp in flow_phases) for tp in args.phases}
    for fp in flow_phases:
        cells = []
        for tp in args.phases:
            v = M[fp][tp]
            mark = "*" if abs(v - col_min[tp]) < 1e-9 else " "
            cells.append(f"{v:8.3f}{mark}")
        print(f"{fp:<10}" + "".join(cells))

    print("\nPer-DAG separation (own test vs nearest other-phase test):")
    for fp in flow_phases:
        own = M[fp][fp]
        others = {tp: M[fp][tp] for tp in args.phases if tp != fp and tp in M[fp]}
        if not others:
            continue
        nearest_tp = min(others, key=others.get)
        gap = others[nearest_tp] - own
        diag_is_min = all(own <= M[fp][tp] + 1e-9 for tp in others)
        print(f"  {fp}: own={own:.3f}  nearest={nearest_tp}({others[nearest_tp]:.3f})  "
              f"gap={gap:+.3f}  {'self-best' if diag_is_min else 'CONFUSED w/ '+nearest_tp}")

    out = Path(args.out) if args.out else Path(
        f"experiments/phase_confusion_{args.model}_{args.variant}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "variant": args.variant,
                   "reassign_passes": args.reassign_passes,
                   "phases": args.phases, "matrix": M}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
