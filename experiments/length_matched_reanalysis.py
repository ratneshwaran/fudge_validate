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
  - lm_ratio 95% CI: percentile bootstrap resampling conversations within each
    matched bin (n_boot, default 2000) — the point estimate now ships with an
    error bar (2026-07 review);
  - one-sided permutation test: shuffle in/out labels WITHIN each bin,
    p = P(lm_ratio_perm >= lm_ratio_observed);
  - coverage = fraction of in-phase conversations inside matched bins;
  - out_coverage = fraction of OUT-of-phase (negative) convs inside matched bins;
    1 - out_coverage is the share of negatives the common-support restriction drops;
  - in_phase_rho = Spearman(in_score, original length) — the within-phase axis the
    length confound was DEFINED on (block_rhos below only inspects OUT-of-phase blocks).

CAVEAT (2026-06 audit): at these sample sizes (150 in / 750 out) the permutation
p saturates at its floor 1/(n_perm+1) for essentially EVERY cell — including
lm_ratio ~ 1.01 cells with no meaningful effect. p_perm confirms direction only;
the informative statistic is the lm_ratio magnitude (with its CI) vs the 1.3
effect-size bar.

NOTE (2026-07 review): "--segment removes the confound (rho -> ~0)" is imprecise.
Segmentation SHIFTS the score to the DAG's granularity: in-phase rho(score, original
length) flips sign (e.g. P5 +0.64 -> -0.58) and a new dependence on the SEGMENT count
appears (rho up to -0.90); a few out-blocks against the longest phase (P10) keep
rho ~ +0.7 on lengths ABOVE the in-phase support. The length-matched ratio stays valid
because it BINS ON ORIGINAL length, so this residual/shifted dependence is controlled
by construction — read in_phase_rho + the CI, not the "rho -> 0" gloss.

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

# Defaults cover CURRENT-provenance results only (see PROVENANCE.md). The
# pre-relabel LLM-DAG results live in experiments/archive_pre_relabel/ and were
# produced under the old labels/split — mixing them with the current split's
# lengths would silently pair old scores with new orderings. Analyse them via
# --results only, and only alongside their own-era split.
RESULT_FILES = [
    ("prefix-tree", "experiments/tv_prefix_tree_discrimination.json"),
]


def parse_result_specs(specs: list[str]) -> list[tuple[str, str]]:
    """Parse --results specs: 'label=path' or a bare path (label = file stem)."""
    out = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:  # bare path: derive the label from the filename
            label, path = Path(spec).stem, spec
        out.append((label, path))
    return out


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
                   n_perm=10000, n_boot=2000, alpha=0.05, seed=0):
    """Direct standardisation of out-of-phase scores to the in-phase length
    distribution, with a within-bin bootstrap CI on the ratio and a within-bin
    permutation test. Returns a result dict.

    Reported alongside the point estimate (2026-07 review): a percentile bootstrap
    CI (resample conversations within each matched bin), the in-phase `coverage`
    AND `out_coverage` (1 - out_coverage = fraction of negatives dropped by the
    common-support restriction). p_perm saturates at these sample sizes — read
    lm_ratio and its CI, not p."""
    in_s, in_l = np.asarray(in_s), np.asarray(in_l)
    out_s, out_l = np.asarray(out_s), np.asarray(out_l)
    in_bin, out_bin = in_l // bin_w, out_l // bin_w
    bins = [b for b in np.unique(in_bin)
            if (in_bin == b).sum() >= min_per_bin and (out_bin == b).sum() >= min_per_bin]
    if not bins:
        return {"lm_ratio": float("nan"), "coverage": 0.0, "out_coverage": 0.0,
                "lm_ci_lo": float("nan"), "lm_ci_hi": float("nan"),
                "p_perm": float("nan"), "n_bins": 0}

    def standardized_ratio(in_scores_by_bin, out_scores_by_bin, weights):
        num = sum(w * o.mean() for w, o in zip(weights, out_scores_by_bin))
        den = sum(w * i.mean() for w, i in zip(weights, in_scores_by_bin))
        return num / den if den > 0 else float("inf")

    in_by = [in_s[in_bin == b] for b in bins]
    out_by = [out_s[out_bin == b] for b in bins]
    w = [len(x) for x in in_by]
    obs = standardized_ratio(in_by, out_by, w)
    coverage = sum(w) / len(in_s)
    out_coverage = sum(len(x) for x in out_by) / len(out_s)

    # Bootstrap CI: resample conversations WITHIN each matched bin (weights fixed
    # at the observed in-phase bin counts). Own rng, seeded independently, so the
    # permutation stream below stays bit-identical to pre-2026-07 runs.
    if n_boot and n_boot > 0:
        boot_rng = np.random.default_rng(seed + 12345)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            ins = [i[boot_rng.integers(0, len(i), len(i))] for i in in_by]
            outs = [o[boot_rng.integers(0, len(o), len(o))] for o in out_by]
            boot[b] = standardized_ratio(ins, outs, w)
        lm_ci_lo = float(np.nanquantile(boot, alpha / 2))
        lm_ci_hi = float(np.nanquantile(boot, 1 - alpha / 2))
    else:
        lm_ci_lo = lm_ci_hi = float("nan")

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
            "out_coverage": float(out_coverage),
            "lm_ci_lo": lm_ci_lo, "lm_ci_hi": lm_ci_hi,
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
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="Bootstrap resamples for the lm_ratio 95%% CI (0 disables).")
    ap.add_argument("--results", nargs="+", default=None,
                    help="Override result files as label=path. Use this to point at the "
                         "--segment outputs, e.g. "
                         "'gpt-oss-seg=experiments/llm_dag_discrimination_gpt-oss-20b_r5_seg.json'. "
                         "Lengths are still the ORIGINAL conversation lengths, so this measures "
                         "whether the segmented score still tracks length (it should not).")
    ap.add_argument("--out", default="experiments/length_matched_reanalysis.json")
    args = ap.parse_args()

    result_files = parse_result_specs(args.results) if args.results else RESULT_FILES

    split_meta = load_split(args.split_path)
    split_order = list(split_meta["splits"].keys())
    print(f"Split phases (out-pool order): {split_order}")
    lens = load_test_lengths(args.tv_dir, split_meta)
    for ph, L in lens.items():
        print(f"  {ph}: n={len(L)} mean_len={np.mean(L):.1f}")
    print("NOTE: p_perm saturates at 1/(n_perm+1) at these sample sizes — "
          "read lm_ratio, not p.")

    all_results = []
    for name, fn in result_files:
        if not Path(fn).exists():
            print(f"\n[skip] {name}: {fn} not found")
            continue
        d = json.load(open(fn, encoding="utf-8"))
        rows = d["results"] if isinstance(d, dict) and "results" in d else d
        if not isinstance(rows, list):
            rows = [rows]
        print(f"\n=== {name} ===")
        print(f"  {'cell':<10} {'raw':>6} {'lm':>6} {'lm 95% CI':>15} "
              f"{'in_cov':>6} {'out_cov':>7} {'in_rho':>7} {'bins':>4}  out-block rho")
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
                                n_perm=args.n_perm, n_boot=args.n_boot)
            rhos = block_rhos(r["out_scores"], lens, split_order, phase)
            rho_lo, rho_hi = min(rhos.values()), max(rhos.values())
            # In-phase rho: the axis the confound was DEFINED on. block_rhos only
            # inspects OUT-of-phase blocks, so it never sees this (2026-07 review).
            in_rho_val, _ = stats.spearmanr(r["in_scores"], in_l)
            in_rho = float(in_rho_val)
            ci = f"[{lm['lm_ci_lo']:.2f},{lm['lm_ci_hi']:.2f}]"
            print(f"  {cell:<10} {raw:>5.2f}x {lm['lm_ratio']:>5.2f}x {ci:>15} "
                  f"{lm['coverage']:>6.0%} {lm['out_coverage']:>7.0%} {in_rho:>+7.2f} "
                  f"{lm['n_bins']:>4}  [{rho_lo:+.2f}, {rho_hi:+.2f}]")
            all_results.append({"source": name, "cell": cell, "raw_ratio": raw,
                                "in_phase_rho": in_rho, **lm, "out_block_rhos": rhos})

    json.dump({"bin_width": args.bin_width, "min_per_bin": args.min_per_bin,
               "n_perm": args.n_perm, "results": all_results},
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
