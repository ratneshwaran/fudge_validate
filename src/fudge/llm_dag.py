"""TODO 7 — build a FuDGE DialogueFlow from an LLM-generated DAG via
cluster-then-recentroid bucket population (archive/EXPLAINER.md §11).

Counterpart to data_loader.build_flow_from_conversations (which builds the
prefix-tree *reference* flow). Here the graph topology comes from the LLM's DAG
(`dag.json`), and each node's IntentBucket is populated by assigning the phase's
real TRAINING utterances to their nearest node-label anchor — so FuDGE compares
utterances against real-data centroids, not abstract label embeddings.

Returns (DialogueFlow, all_buckets) — the same shape FudgeCosts expects.

The flow is forced acyclic (fudge_efficient's DFS recurses without a visited-set,
so a cycle would loop forever) and every in-degree-0 node is wired to the dummy
root so all nodes are reachable.
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from .types import Conversation, DialogueFlow, IntentBucket
from .embeddings import EmbeddingCache


def _break_cycles(g: nx.DiGraph) -> int:
    """Remove back-edges until acyclic. Returns the number removed."""
    removed = 0
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g, orientation="original")
        u, v = cycle[-1][0], cycle[-1][1]
        g.remove_edge(u, v)
        removed += 1
    return removed


def build_flow_from_llm_dag(
    dag: dict,
    train_convs: list[Conversation],
    emb: EmbeddingCache,
    drop_unknown: bool = True,
    reassign_passes: int = 0,
    label_fallback: bool = True,
) -> tuple[DialogueFlow, list[IntentBucket], dict]:
    """Build (flow, all_buckets, stats) from a parsed LLM DAG + training convs.

    dag: {"nodes": [{id, actor, label}], "edges": [{from, to}]}.
    train_convs: phase TRAINING conversations (never test — no leakage).

    Cluster-then-recentroid (archive/EXPLAINER.md §11). Each same-actor training
    utterance is assigned to its nearest node centroid (embeddings are normalised,
    so cosine = dot product). With `reassign_passes=0` the centroids are the node
    LABEL anchors (one-pass NN — the original behaviour). With reassign_passes>0
    we then recompute each node's centroid from its assigned utterances and
    re-assign against those real centroids (constrained k-means seeded by the
    labels, K fixed by node count) — the §11 mitigation for the one-pass
    "hub node" problem. Iterates until convergence or the pass budget.

    `label_fallback` controls what happens to a node that wins NO training
    utterance (an empty bucket):
      True  (default): seed its bucket with the node LABEL string so
        intent_centroid doesn't average over nothing. This makes FuDGE compare
        utterances to a label-string embedding for that node — a deviation from
        the "never compare to labels" rule. Kept as the default only so existing
        aligned artifacts reproduce bit-for-bit.
      False (Rule-2 guard): DROP the empty node and rewire its parents to its
        children, so no label-string centroid ever enters scoring. Prefer this
        for new/scaled runs (recorded in stats as n_empty_dropped).
    """
    raw_nodes = dag["nodes"]
    nodes = [n for n in raw_nodes
             if (not drop_unknown) or n["actor"] in ("agent", "user")]
    ids = [n["id"] for n in nodes]
    id_set = set(ids)
    actor = {n["id"]: n["actor"] for n in nodes}
    label = {n["id"]: (n["label"].strip() or n["id"]) for n in nodes}

    by_actor = {a: [nid for nid in ids if actor[nid] == a] for a in ("agent", "user")}

    # Collect training utterance texts per actor (deduped per actor for the
    # assignment maths; bucket-level dedup happens again below).
    utt_texts = {"agent": [], "user": []}
    for conv in train_convs:
        for u in conv.utterances:
            if u.actor in utt_texts:
                utt_texts[u.actor].append(u.text)

    assigned: dict[str, list[str]] = {nid: [] for nid in ids}
    n_passes_run = 0
    converged = False
    for a in ("agent", "user"):
        nodes_a = by_actor[a]
        texts_a = utt_texts[a]
        if not nodes_a or not texts_a:
            continue
        label_anchors = emb.encode_batch([label[nid] for nid in nodes_a])  # (K, d)
        U = emb.encode_batch(texts_a)                                      # (N, d)
        centroids = label_anchors.copy()
        assign = None
        for p in range(reassign_passes + 1):
            new_assign = (U @ centroids.T).argmax(axis=1)
            if assign is not None and np.array_equal(new_assign, assign):
                converged = True
                break
            assign = new_assign
            n_passes_run = max(n_passes_run, p + 1)
            if p < reassign_passes:  # recompute centroids for the next pass
                new_c = centroids.copy()
                for k in range(len(nodes_a)):
                    mask = assign == k
                    if mask.any():
                        v = U[mask].mean(axis=0)
                        nrm = np.linalg.norm(v)
                        if nrm > 0:
                            new_c[k] = v / nrm
                    # else: keep the label anchor so the node can still win later
                centroids = new_c
        for i, text in enumerate(texts_a):
            assigned[nodes_a[int(assign[i])]].append(text)

    # buckets: dedup assigned texts. A node that won nothing is EMPTY -> either
    # seed with the label string (label_fallback=True, default) or drop it below.
    # An empty bucket would make intent_centroid average over nothing -> NaN.
    buckets: dict[str, IntentBucket] = {}
    empty_ids: list[str] = []
    for nid in ids:
        texts = list(dict.fromkeys(assigned[nid]))
        if not texts:
            empty_ids.append(nid)
            if not label_fallback:
                continue  # dropped-and-rewired below; no bucket created
            texts = [label[nid]]
        buckets[nid] = IntentBucket(actor=actor[nid], utterances=texts, label=label[nid])
    n_empty = len(empty_ids)
    empty_set = set() if label_fallback else set(empty_ids)
    surviving_ids = [nid for nid in ids if nid not in empty_set]
    all_buckets = list(buckets.values())

    # --- topology: keep only edges between surviving nodes, force acyclic, wire root ---
    g = nx.DiGraph()
    g.add_nodes_from(ids)
    for e in dag["edges"]:
        if e["from"] in id_set and e["to"] in id_set and e["from"] != e["to"]:
            g.add_edge(e["from"], e["to"])
    n_backedges = _break_cycles(g)

    # Rule-2 guard (label_fallback=False): excise each empty node, bridging its
    # parents to its children so every root->leaf path stays connected. Iterate
    # over the live graph so chains of empty nodes collapse correctly.
    for nid in empty_set:
        preds = list(g.predecessors(nid))
        succs = list(g.successors(nid))
        for p in preds:
            for s in succs:
                if p != s:
                    g.add_edge(p, s)
        g.remove_node(nid)

    flow = DialogueFlow()
    for nid in surviving_ids:
        flow.add_node(nid, buckets[nid])
    for u, v in g.edges():
        flow.add_edge(u, v)
    for nid in surviving_ids:
        if g.in_degree(nid) == 0:
            flow.add_edge(flow.root, nid)

    stats = {
        "n_nodes": len(surviving_ids),
        "n_edges": g.number_of_edges(),
        "n_agent_nodes": sum(1 for nid in surviving_ids if actor[nid] == "agent"),
        "n_user_nodes": sum(1 for nid in surviving_ids if actor[nid] == "user"),
        "n_empty_buckets": n_empty if label_fallback else 0,
        "n_empty_dropped": 0 if label_fallback else n_empty,
        "label_fallback": label_fallback,
        "n_backedges_removed": n_backedges,
        "n_dropped_unknown": len(raw_nodes) - len(ids),
        "reassign_passes": reassign_passes,
        "n_passes_run": n_passes_run,
        "converged": converged,
    }
    return flow, all_buckets, stats


def serialize_flow(flow: DialogueFlow, stats: dict, meta: dict) -> dict:
    """Flatten an aligned flow to a JSON-able dict (persist after TODO 7).

    Captures the FINAL topology (post cycle-break + root-wiring) and each node's
    populated bucket, so deserialize_flow reproduces exactly what was scored.
    """
    nodes = []
    for nid in flow.graph.nodes:
        if nid == flow.root:
            continue
        b = flow.get_bucket(nid)
        nodes.append({"id": nid, "actor": b.actor, "label": b.label,
                      "utterances": b.utterances})
    edges = [[u, v] for u, v in flow.graph.edges]
    return {**meta, "stats": stats, "nodes": nodes, "edges": edges}


def deserialize_flow(d: dict) -> tuple[DialogueFlow, list[IntentBucket]]:
    """Rebuild (flow, all_buckets) from serialize_flow output."""
    flow = DialogueFlow()
    buckets: dict[str, IntentBucket] = {}
    for n in d["nodes"]:
        b = IntentBucket(actor=n["actor"], utterances=list(n["utterances"]),
                         label=n.get("label", ""))
        buckets[n["id"]] = b
        flow.add_node(n["id"], b)
    for u, v in d["edges"]:
        flow.add_edge(u, v)
    return flow, list(buckets.values())


def coverage_report(flow: DialogueFlow, stats: dict, n_samples: int = 3) -> dict:
    """Per-node assignment summary for inspecting alignment quality (TODO 7).

    A node whose only "utterance" equals its label is an empty (fallback) bucket
    — it won no real training utterance and is flagged.
    """
    per_node = []
    n_real = 0
    for nid in flow.graph.nodes:
        if nid == flow.root:
            continue
        b = flow.get_bucket(nid)
        is_fallback = (len(b.utterances) == 1 and b.utterances[0] == b.label)
        n_real += 0 if is_fallback else 1
        per_node.append({
            "id": nid, "actor": b.actor, "label": b.label,
            "n_utterances": 0 if is_fallback else len(b.utterances),
            "empty_fallback": is_fallback,
            "samples": [] if is_fallback else b.utterances[:n_samples],
        })
    per_node.sort(key=lambda r: r["n_utterances"], reverse=True)
    n_nodes = stats["n_nodes"]
    return {
        "n_nodes": n_nodes,
        "n_nodes_with_real_utterances": n_real,
        "coverage_frac": (n_real / n_nodes) if n_nodes else 0.0,
        "n_empty_fallback": n_nodes - n_real,
        "nodes": per_node,
    }
