from dataclasses import dataclass, field
import numpy as np
import networkx as nx


@dataclass
class Utterance:
    actor: str  # 'user' or 'agent'
    text: str
    # Optional precomputed embedding. Set by segment_conversation for collapsed
    # segments (a segment's vector is the mean of its members, not the encoding
    # of any single text). When present, FudgeCosts uses it instead of encoding
    # `text`. compare/repr off so it never breaks dataclass equality or logging.
    embedding: "np.ndarray | None" = field(default=None, compare=False, repr=False)


@dataclass
class Conversation:
    utterances: list[Utterance]
    task: str = ""
    dialogue_id: int = -1  # set by load_star_dialogues; -1 = unknown


@dataclass
class IntentBucket:
    """Paper: B_r = (actor, utterances). A cluster of semantically similar utterances."""
    actor: str               # 'user' or 'agent'
    utterances: list[str]    # the actual utterance texts in this cluster
    label: str = ""          # human-readable name


class DialogueFlow:
    """Paper: G = (V, E), a DAG with a dummy root node.

    Each non-root node is associated with an IntentBucket.
    Internally uses networkx.DiGraph.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.root = "__ROOT__"
        self.graph.add_node(self.root, bucket=None)

    def add_node(self, node_id: str, bucket: IntentBucket):
        self.graph.add_node(node_id, bucket=bucket)

    def add_edge(self, parent: str, child: str):
        self.graph.add_edge(parent, child)

    def get_bucket(self, node_id: str) -> IntentBucket:
        return self.graph.nodes[node_id]['bucket']

    def get_children(self, node_id: str) -> list[str]:
        return list(self.graph.successors(node_id))

    def get_leaf_nodes(self) -> list[str]:
        return [n for n in self.graph.nodes
                if n != self.root and self.graph.out_degree(n) == 0]

    def get_all_paths(self) -> list[list[str]]:
        """All root-to-leaf paths, excluding dummy root node."""
        leaves = self.get_leaf_nodes()
        all_paths = []
        for leaf in leaves:
            for path in nx.all_simple_paths(self.graph, self.root, leaf):
                all_paths.append(path[1:])  # exclude dummy root
        return all_paths

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes() - 1  # exclude root
