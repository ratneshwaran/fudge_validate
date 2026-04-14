from .types import Conversation, DialogueFlow
from .costs import FudgeCosts


def fudge_efficient(conversation: Conversation, flow: DialogueFlow,
                    costs: FudgeCosts) -> float:
    """
    Algorithm 2: DFS traversal with memoized distance arrays.

    For each node, store list of (path_length, distance_array) tuples.
    distance_array[j] = edit distance between path root->...->node and conversation[0:j]

    At leaf nodes, distance_array[-1] gives the full path-conversation edit distance.
    Return min across all leaves.

    Complexity: O((|V| + |E|) * n) where n = len(conversation)
    """
    utts = conversation.utterances
    n = len(utts)
    node2dist: dict[str, list[tuple[int, list[float]]]] = {}

    def dfs(node: str, parent: str | None) -> None:
        if parent is None:
            # Root: dist array = [0, 1, 2, ..., n] (insert all utterances)
            dist = [float(i) for i in range(n + 1)]
            node2dist[node] = [(0, dist)]
        else:
            if node not in node2dist:
                node2dist[node] = []
            bucket = flow.get_bucket(node)
            for path_len, parent_dist in node2dist[parent]:
                # Extend parent's distance array by one more node
                new_dist = [parent_dist[0] + costs.deletion_cost(bucket)]
                for j in range(n):
                    val = min(
                        parent_dist[j + 1] + costs.deletion_cost(bucket),        # delete node
                        new_dist[j] + costs.insertion_cost(utts[j]),              # insert utterance
                        parent_dist[j] + costs.substitution_cost(bucket, utts[j]) # substitute
                    )
                    new_dist.append(val)
                node2dist[node].append((path_len + 1, new_dist))

        for child in flow.get_children(node):
            dfs(child, node)

    dfs(flow.root, None)

    # Find minimum across all leaf nodes
    min_dist = float('inf')
    for leaf in flow.get_leaf_nodes():
        if leaf in node2dist:
            for _, dist_array in node2dist[leaf]:
                min_dist = min(min_dist, dist_array[-1])

    if min_dist == float('inf'):
        # Empty flow
        return float(len(utts))

    return min_dist
