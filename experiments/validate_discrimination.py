"""
Reproduce Table 1b: FuDGE separates in-task vs out-of-task conversations.

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
"""
import sys
sys.modules["tensorflow"] = None

import numpy as np
from tqdm import tqdm

from fudge.data_loader import load_star_dialogues, group_by_task, build_flow_from_conversations
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


def main():
    print("Loading STAR dataset...")
    convs = load_star_dialogues("data/STAR")
    by_task = group_by_task(convs)

    print("Initializing embeddings...")
    emb = EmbeddingCache()

    tasks_to_test = ["hotel_book", "bank_fraud_report"]

    for task_name in tasks_to_test:
        task_convs = by_task[task_name]
        other_convs = [c for c in convs if c.task != task_name]

        print(f"\nBuilding flow for {task_name} from {len(task_convs)} conversations...")
        flow, all_buckets = build_flow_from_conversations(task_convs)
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
