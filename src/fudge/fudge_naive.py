import numpy as np

from .types import Conversation, IntentBucket, DialogueFlow
from .costs import FudgeCosts


def path_edit_distance(conversation: Conversation, path_buckets: list[IntentBucket],
                       costs: FudgeCosts) -> float:
    """
    Standard Levenshtein DP between conversation utterances and a sequence of intent buckets.

    DP table: d[r][s] = edit distance between path_buckets[0:r] and conversation.utterances[0:s]

    Recurrence (Eq 1):
    d[r][s] = min(
        d[r-1][s] + deletion_cost(path_buckets[r-1]),
        d[r][s-1] + insertion_cost(utterances[s-1]),
        d[r-1][s-1] + substitution_cost(path_buckets[r-1], utterances[s-1])
    )
    """
    m = len(conversation.utterances)
    n = len(path_buckets)

    dp = np.zeros((n + 1, m + 1))

    # Base cases
    for s in range(1, m + 1):
        dp[0][s] = dp[0][s - 1] + costs.insertion_cost(conversation.utterances[s - 1])
    for r in range(1, n + 1):
        dp[r][0] = dp[r - 1][0] + costs.deletion_cost(path_buckets[r - 1])

    # Fill DP table
    for r in range(1, n + 1):
        for s in range(1, m + 1):
            dp[r][s] = min(
                dp[r - 1][s] + costs.deletion_cost(path_buckets[r - 1]),
                dp[r][s - 1] + costs.insertion_cost(conversation.utterances[s - 1]),
                dp[r - 1][s - 1] + costs.substitution_cost(path_buckets[r - 1],
                                                            conversation.utterances[s - 1]),
            )

    return float(dp[n][m])


def fudge_naive(conversation: Conversation, flow: DialogueFlow,
                costs: FudgeCosts) -> float:
    """
    Algorithm 1: FuDGE(C_i, G) = min over all paths P_k of dist(C_i, P_k)
    """
    all_paths = flow.get_all_paths()
    if not all_paths:
        # Empty flow — distance = number of utterances (all insertions)
        return float(len(conversation.utterances))

    min_dist = float('inf')
    for path_node_ids in all_paths:
        path_buckets = [flow.get_bucket(nid) for nid in path_node_ids]
        dist = path_edit_distance(conversation, path_buckets, costs)
        min_dist = min(min_dist, dist)

    return min_dist
