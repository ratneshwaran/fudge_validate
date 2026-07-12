"""TODO 7 (standalone) — align LLM DAGs and persist them for inspection.

Step 1 of the two-step LLM-DAG evaluation: build the cluster-then-recentroid
flow for each <model>/<variant>/<phase> DAG and WRITE it to disk, alongside a
per-node coverage report, BEFORE any scoring. This is the inspection point the
methodology (TODO 7) calls for — eyeball which real utterances landed in each
node before trusting the discrimination numbers (TODO 8).

Outputs, per cell:
  data/dags/<model>/<variant>/<phase>/aligned.json    (reloadable flow + buckets)
  data/dags/<model>/<variant>/<phase>/coverage.json   (per-node assignment report)

Run:
  python experiments/align_llm_dags.py --model gpt-oss-20b
  python experiments/align_llm_dags.py --model gpt-oss-20b --variants v2 --show 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.modules["tensorflow"] = None

from fudge.data_loader import load_thousand_voices_dialogues
from fudge.embeddings import EmbeddingCache
from fudge.llm_dag import (
    build_flow_from_llm_dag,
    coverage_report,
    serialize_flow,
)
from fudge.splits import load_split, split_conversations

DEFAULT_TV_DIR = "data/thousand-voices-trauma/ThousandVoicesOfTrauma"
SPLIT_PATH = "data/splits/TV_v1.json"
DAGS_ROOT = "data/dags"


def _phase_train(tv_dir, split_meta, phase, cache):
    if phase not in cache:
        convs = load_thousand_voices_dialogues(tv_dir, task_field="type",
                                               require_phases=(phase,))
        in_split = (set(split_meta["splits"][phase]["train"])
                    | set(split_meta["splits"][phase]["test"]))
        convs = [c for c in convs if c.dialogue_id in in_split]
        cache[phase] = split_conversations(convs, split_meta, phase)[0]
    return cache[phase]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", nargs="+", default=["v1", "v2", "v3"])
    ap.add_argument("--phases", nargs="+", default=["P5", "P6", "P7"])
    ap.add_argument("--tv-dir", default=DEFAULT_TV_DIR)
    ap.add_argument("--split-path", default=SPLIT_PATH)
    ap.add_argument("--dags-root", default=DAGS_ROOT)
    ap.add_argument("--show", type=int, default=5,
                    help="Top-N nodes (by #utterances) to print per cell.")
    ap.add_argument("--reassign-passes", type=int, default=0,
                    help="0 = one-pass label-anchor NN. >0 = recompute centroids "
                         "from assigned utterances and re-assign (§11 hub fix). "
                         "Artifacts get an _r<N> suffix so one-pass stays intact.")
    ap.add_argument("--drop-empty-nodes", action="store_true",
                    help="Rule-2 guard: drop nodes that win no training utterance "
                         "(rewire parents->children) instead of the label-string "
                         "fallback. Artifacts get a _nofb suffix so the default "
                         "label-fallback aligned files are never overwritten.")
    args = ap.parse_args()
    suffix = "" if args.reassign_passes == 0 else f"_r{args.reassign_passes}"
    suffix += "_nofb" if args.drop_empty_nodes else ""

    split_meta = load_split(args.split_path)
    emb = EmbeddingCache()
    train_cache: dict = {}

    for variant in args.variants:
        for phase in args.phases:
            cell = f"{args.model}/{variant}/{phase}"
            dag_path = Path(args.dags_root) / args.model / variant / phase / "dag.json"
            if not dag_path.exists():
                print(f"[skip] {cell}: no dag.json"); continue
            dag = json.load(open(dag_path, encoding="utf-8"))
            if not dag.get("nodes"):
                print(f"[skip] {cell}: empty dag (0 nodes)"); continue

            train = _phase_train(args.tv_dir, split_meta, phase, train_cache)
            flow, _buckets, stats = build_flow_from_llm_dag(
                dag, train, emb, reassign_passes=args.reassign_passes,
                label_fallback=not args.drop_empty_nodes)
            stats["n_train_align"] = len(train)

            out_dir = dag_path.parent
            aligned = serialize_flow(flow, stats,
                                     {"model": args.model, "variant": variant, "phase": phase})
            cov = coverage_report(flow, stats)
            with open(out_dir / f"aligned{suffix}.json", "w", encoding="utf-8") as f:
                json.dump(aligned, f, ensure_ascii=False, indent=2)
            with open(out_dir / f"coverage{suffix}.json", "w", encoding="utf-8") as f:
                json.dump(cov, f, ensure_ascii=False, indent=2)

            print(f"\n=== {cell} ===  nodes={cov['n_nodes']} "
                  f"(a{stats['n_agent_nodes']}/u{stats['n_user_nodes']}) | "
                  f"coverage={cov['coverage_frac']*100:.0f}% "
                  f"({cov['n_empty_fallback']} empty) | "
                  f"passes={stats['n_passes_run']}{'*' if stats['converged'] else ''} | "
                  f"cut={stats['n_backedges_removed']} dropped={stats['n_dropped_unknown']} | "
                  f"train_convs={len(train)}")
            for r in cov["nodes"][:args.show]:
                tag = "agent" if r["actor"] == "agent" else "user "
                sample = (r["samples"][0][:70] + "…") if r["samples"] else "<empty: fallback to label>"
                print(f"   [{tag}] {r['label'][:34]:<34} n={r['n_utterances']:>4}  e.g. {sample}")

    print(f"\nWrote aligned{suffix}.json + coverage{suffix}.json per cell. Inspect coverage, then run "
          f"experiments/llm_dag_discrimination.py --from-aligned --reassign-passes {args.reassign_passes}")


if __name__ == "__main__":
    main()
