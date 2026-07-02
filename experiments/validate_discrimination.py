"""
Reproduce Table 1b: FuDGE separates in-task vs out-of-task conversations.

SUPERSEDED (kept as the original Table-1b reproducer): this script builds the
flow from ALL in-task conversations (no held-out split, no significance test,
legacy global np.random.seed). Use experiments/star_v2_validation.py (proper
70/30 split + Mann-Whitney + Bonferroni) or experiments/significance.py for
any defensible number.

For each task:
1. Build a supervised flow from in-task conversations
2. Sample in-task conversations as "positives"
3. Sample equal number of out-of-task conversations as "negatives"
4. Compute FuDGE score for each conversation against the flow
5. Report: mean +/- std for positives vs negatives
6. Assert: mean(positives) < mean(negatives) with clear gap

Expected (Table 1b, ALG1-Centroid):
  Hotel Book:        positives 0.08 +/- 0.03, negatives 0.59 +/- 0.18
  Bank Fraud Report: positives 0.09 +/- 0.04, negatives 0.63 +/- 0.19

CLI:
  # default: heuristic labels (user_before_<next_agent_intent>)
  python experiments/validate_discrimination.py

  # use LLM-generated labels from scripts/llm_label_star.py
  python experiments/validate_discrimination.py \
      --label-root data/STAR_llm_labels --label-method whole
"""
import argparse
import sys
from pathlib import Path

sys.modules["tensorflow"] = None

import numpy as np
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


def run_discrimination_experiment(
    task_name: str,
    task_conversations: list,
    other_conversations: list,
    flow,
    costs: FudgeCosts,
    sample_ratio: float = 0.5,
    normalize: bool = True,
):
    n_sample = int(len(task_conversations) * sample_ratio)
    np.random.seed(42)
    pos_indices = np.random.choice(len(task_conversations), n_sample, replace=False)
    positives = [task_conversations[i] for i in pos_indices]
    neg_indices = np.random.choice(len(other_conversations), min(n_sample, len(other_conversations)), replace=False)
    negatives = [other_conversations[i] for i in neg_indices]

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Positives: {len(positives)}, Negatives: {len(negatives)}")
    print(f"{'='*60}")

    pos_scores = []
    for conv in tqdm(positives, desc="In-task"):
        score = fudge_efficient(conv, flow, costs)
        if normalize and len(conv.utterances) > 0:
            score /= len(conv.utterances)
        pos_scores.append(score)

    neg_scores = []
    for conv in tqdm(negatives, desc="Out-of-task"):
        score = fudge_efficient(conv, flow, costs)
        if normalize and len(conv.utterances) > 0:
            score /= len(conv.utterances)
        neg_scores.append(score)

    pos_mean, pos_std = np.mean(pos_scores), np.std(pos_scores)
    neg_mean, neg_std = np.mean(neg_scores), np.std(neg_scores)
    separation = neg_mean - pos_mean

    print(f"\nResults ({'normalized' if normalize else 'raw'}):")
    print(f"  In-task:     {pos_mean:.4f} +/- {pos_std:.4f}")
    print(f"  Out-of-task: {neg_mean:.4f} +/- {neg_std:.4f}")
    print(f"  Separation:  {separation:.4f}")
    print(f"  Ratio:       {neg_mean/pos_mean:.2f}x")

    # Validation
    if pos_mean < neg_mean:
        # Check for meaningful separation (means don't overlap within 1 std)
        if pos_mean + pos_std < neg_mean - neg_std:
            print("  STRONG PASS: clear separation, no overlap at 1-sigma")
        else:
            print("  PASS: positives lower than negatives (some overlap)")
    else:
        print("  FAIL: positives NOT lower than negatives")

    return pos_scores, neg_scores


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["hotel_book", "bank_fraud_report"],
        help="STAR tasks to evaluate (default: hotel_book bank_fraud_report)",
    )
    p.add_argument(
        "--label-root",
        default=None,
        help="Root dir of LLM-generated labels (e.g. data/STAR_llm_labels). "
        "When omitted, uses the heuristic user_before_<next_agent_intent> labels.",
    )
    p.add_argument(
        "--taxonomy-method",
        default="single_prompt",
        choices=["single_prompt", "hybrid", "cluster"],
        help="Taxonomy method subdir under <label-root>/<task>/. Only used "
        "when --label-root is set (default: single_prompt).",
    )
    p.add_argument(
        "--label-method",
        default="whole",
        choices=["whole", "window", "chunk"],
        help="Labeling method subdir under <label-root>/<task>/<taxonomy-method>/ "
        "(default: whole). Only used when --label-root is set.",
    )
    p.add_argument("--star-dir", default="data/STAR")
    return p.parse_args()


def main():
    args = _parse_args()
    using_llm_labels = args.label_root is not None

    print("Loading STAR dataset...")
    convs = load_star_dialogues(args.star_dir)
    by_task = group_by_task(convs)

    print("Initializing embeddings...")
    emb = EmbeddingCache()

    if using_llm_labels:
        print(
            f"Using LLM labels from {args.label_root} "
            f"(taxonomy={args.taxonomy_method}, label={args.label_method})"
        )
    else:
        print("Using heuristic labels (user_before_<next_agent_intent>)")

    for task_name in args.tasks:
        task_convs = by_task[task_name]
        other_convs = [c for c in convs if c.task != task_name]

        label_source = None
        if using_llm_labels:
            label_dir = Path(args.label_root) / task_name / args.taxonomy_method / args.label_method
            if not label_dir.exists():
                print(f"  [skip] {task_name}: no labels at {label_dir}")
                continue
            label_source = load_llm_labels(label_dir)
            covered = sum(1 for c in task_convs if c.dialogue_id in label_source)
            print(
                f"  {task_name}: {covered}/{len(task_convs)} in-task conversations "
                f"have LLM labels (other-task convs use heuristic for the flow only "
                f"— flow is built from in-task)"
            )
            if covered == 0:
                print(f"  [skip] {task_name}: no labeled conversations")
                continue

        print(f"\nBuilding flow for {task_name} from {len(task_convs)} conversations...")
        flow, all_buckets = build_flow_from_conversations(task_convs, label_source=label_source)
        print(f"  Flow: {flow.num_nodes} nodes, {len(flow.get_all_paths())} paths")
        print(f"  Buckets: {len(all_buckets)} ({len([b for b in all_buckets if b.actor=='user'])} user, {len([b for b in all_buckets if b.actor=='agent'])} agent)")

        costs = FudgeCosts(emb, all_buckets)

        run_discrimination_experiment(
            task_name=task_name,
            task_conversations=task_convs,
            other_conversations=other_convs,
            flow=flow,
            costs=costs,
        )

    print("\n\nDone.")


if __name__ == "__main__":
    main()
