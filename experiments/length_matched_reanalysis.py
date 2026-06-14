"""Length-matched re-analysis of saved FuDGE discrimination results.

Motivation (2026-06-07 methodology review): within-phase Spearman corr between
the length-normalised FuDGE score and conversation length is 0.89-0.99 for LLM
DAG flows (vs 0.42-0.64 for the Step-1 prefix-trees), and TV phases differ
systematically in conversation length (P5 ~20.6, P6 ~34.2, P7 ~29.8 mean
utterances). The published out/in discrimination ratio compares scores across
DIFFERENT phases' test conversations, so for shallow LLM DAGs it is largely a
length artifact. Mechanism: LLM-DAG root->leaf paths (9-24 nodes) are shorter
than the conversations, so each uncovered utterance costs a flat insertion and
score/n rises monotonically with n.

This script re-analyses the SAVED per-conversation scores (in_scores /
out_scores in the result JSONs) with the out-of-phase pool length-matched to
the in-phase length distribution — no re-embedding, no re-scoring.

Method: direct standardisation over length bins on the common support.
  - bin all conversations by length (default bin width 3);
  - keep bins with >= MIN_PER_BIN in-phase AND out-of-phase convs;
  - lm_ratio = sum_b w_b * mean_out_b / sum_b w_b * mean_in_b, w_b = n_in_b
    (i.e. out-of-phase means reweighted to the in-phase length distribution);
  - one-sided permutation test: shuffle in/out labels WITHIN each bin,
    p = P(lm_ratio_perm >= lm_ratio_observed);
  - coverage = fraction of in-phase conversations inside matched bins.

Score-to-length alignment is reconstructed from the split order (the scoring
scripts iterate split_meta["splits"] insertion order) and VALIDATED per
out-phase block via the within-block score-length Spearman rho: a misaligned
block would show rho ~ 0 where the in-phase rho is high.

Usage:
  PYTHONPATH=src python experiments/length_matched_reanalysis.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from fudge.data_loader import load_thousand_voices_dialogues
from fudge.splits import load_split, split_conversations

DEFAULT_TV_DIR = "data/thousand-voices-trauma/ThousandVoicesOfTrauma"
SPLIT_PATH = "data/splits/TV_v1.json"

RESULT_FILES = [
    ("prefix-tree", "experiments/tv_prefix_tree_discrimination.json"),
    ("gpt-oss-20b", "experiments/llm_dag_discrimination_gpt-oss-20b_r5.json"),
    ("deepseek-v3.2", "experiments/llm_dag_discrimination_deepseek-v3.2_r5.json"),
]


def load_test_lengths(tv_dir: str, split_meta: dict) -> dict[str, list[int]]:
    """Per-phase test-conversation lengths, in split_conversations order."""
    lens: dict[str, list[int]] = {}
    for phase in split_meta["splits"]:
        convs = load_thousand_voices_dialogues(tv_dir, task_field="type",
                                               require_phases=(phase,))
        in_split = (set(split_meta["splits"][phase]["train"])
                    | set(split_meta["splits"][phase]["test"]))
        convs = [c for c in convs if c.dialogue_id in in_split]
        _, test = split_conversations(convs, split_meta, phase)
        lens[phase] = [len(c.utterances) for c in test]
    return lens


def length_matched(in_s, in_l, out_s, out_l, bin_w=3, min_per_bin=5,
                   n_perm=10000, seed=0):
    """Direct standardisation of out-of-phase scores to the in-phase length
    distribution + within-bin permutation test. Returns a result dict."""
    in_s, in_l = np.asarray(in_s), np.asarray(in_l)
    out_s, out_l = np.asarray(out_s), np.asarray(out_l)
    in_bin, out_bin = in_l // bin_w, out_l // bin_w
    bins = [b for b in np.unique(in_bin)
            if (in_bin == b).sum() >= min_per_bin and (out_bin == b).sum() >= min_per_bin]
    if not bins:
        return {"lm_ratio": float("nan"), "coverage": 0.0, "p_perm": float("nan"),
                "n_bins": 0}

    def standardized_ratio(in_scores_by_bin, out_scores_by_bin, weights):
        num = sum(w * o.mean() for w, o in zip(weights, out_scores_by_bin))
        den = sum(w * i.mean() for w, i in zip(weights, in_scores_by_bin))
        return num / den if den > 0 else float("inf")

    in_by = [in_s[in_bin == b] for b in bins]
    out_by = [out_s[out_bin == b] for b in bins]
    w = [len(x) for x in in_by]
    obs = standardized_ratio(in_by, out_by, w)
    coverage = sum(w) / len(in_s)

    # Permutation: within each matched bin, shuffle in/out labels.
    rng = np.random.default_rng(seed)
    pooled = [np.concatenate([i, o]) for i, o in zip(in_by, out_by)]
    n_in = [len(i) for i in in_by]
    count = 0
    for _ in range(n_perm):
        pi, po = [], []
        for pool, k in zip(pooled, n_in):
            perm = rng.permutation(len(pool))
            pi.append(pool[perm[:k]])
            po.append(pool[perm[k:]])
        if standardized_ratio(pi, po, w) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"lm_ratio": float(obs), "coverage": float(coverage),
            "p_perm": float(p), "n_bins": len(bins)}


def block_rhos(out_s, lens, split_order, phase):
    """Per-out-phase-block Spearman rho — alignment sanity check."""
    rhos, i = {}, 0
    for other in split_order:
        if other == phase:
            continue
        n = len(lens[other])
        block = np.asarray(out_s[i:i + n])
        rho, _ = stats.spearmanr(block, lens[other])
        rhos[other] = float(rho)
        i += n
    return rhos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tv-dir", default=DEFAULT_TV_DIR)
    ap.add_argument("--split-path", default=SPLIT_PATH)
    ap.add_argument("--bin-width", type=int, default=3)
    ap.add_argument("--min-per-bin", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out", default="experiments/length_matched_reanalysis.json")
    args = ap.parse_args()

    split_meta = load_split(args.split_path)
    split_order = list(split_meta["splits"].keys())
    print(f"Split phases (out-pool order): {split_order}")
    lens = load_test_lengths(args.tv_dir, split_meta)
    for ph, L in lens.items():
        print(f"  {ph}: n={len(L)} mean_len={np.mean(L):.1f}")

    all_results = []
    for name, fn in RESULT_FILES:
        if not Path(fn).exists():
            print(f"\n[skip] {name}: {fn} not found")
            continue
        d = json.load(open(fn, encoding="utf-8"))
        rows = d["results"] if isinstance(d, dict) and "results" in d else d
        if not isinstance(rows, list):
            rows = [rows]
        print(f"\n=== {name} ===")
        print(f"  {'cell':<10} {'raw':>6} {'lm':>6} {'cover':>6} {'p_perm':>9} "
              f"{'bins':>4}  out-block rho range")
        for r in rows:
            if not isinstance(r, dict) or "in_scores" not in r:
                continue
            phase = r["phase"]
            cell = f"{r.get('variant', 'ptree')}/{phase}"
            in_l = lens[phase]
            out_l = [x for other in split_order if other != phase
                     for x in lens[other]]
            if len(r["in_scores"]) != len(in_l) or len(r["out_scores"]) != len(out_l):
                print(f"  {cell:<10} LENGTH MISMATCH "
                      f"in {len(r['in_scores'])}/{len(in_l)} "
                      f"out {len(r['out_scores'])}/{len(out_l)}")
                continue
            raw = float(np.mean(r["out_scores"]) / np.mean(r["in_scores"]))
            lm = length_matched(r["in_scores"], in_l, r["out_scores"], out_l,
                                bin_w=args.bin_width, min_per_bin=args.min_per_bin,
                                n_perm=args.n_perm)
            rhos = block_rhos(r["out_scores"], lens, split_order, phase)
            rho_lo, rho_hi = min(rhos.values()), max(rhos.values())
            print(f"  {cell:<10} {raw:>5.2f}x {lm['lm_ratio']:>5.2f}x "
                  f"{lm['coverage']:>5.0%} {lm['p_perm']:>9.2e} {lm['n_bins']:>4}  "
                  f"[{rho_lo:+.2f}, {rho_hi:+.2f}]")
            all_results.append({"source": name, "cell": cell, "raw_ratio": raw,
                                **lm, "out_block_rhos": rhos})

    json.dump({"bin_width": args.bin_width, "min_per_bin": args.min_per_bin,
               "n_perm": args.n_perm, "results": all_results},
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
