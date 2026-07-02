"""Granularity-normalised conversation segmentation (the supervisor's 2026-06
whiteboard fix for the length confound).

FuDGE aligns one DAG node to one conversation turn. The LLM DAGs are coarse,
stage-level summaries, so on a long conversation most turns pay a flat insertion
penalty and the normalised score becomes a length measurement (within-phase
rho(score, length) ~= 0.9-0.99 for LLM DAGs vs 0.4-0.6 for prefix-trees).

This module removes the confound from the *input* side, leaving FuDGE unchanged:
assign each utterance to its nearest same-actor bucket, collapse maximal runs of
the same bucket into one segment (with a min-run smoothing pass that absorbs the
occasional noise turn in the middle of a run, e.g. D G D -> D), and represent a
segment by the mean (re-normalised) embedding of its members. A 40-turn session
with ~10 stage transitions becomes a ~10-segment sequence at the DAG's
granularity, so FuDGE can align them without the length penalty.

The counterpart metric-side fix is dwell-FuDGE (METRIC_OPTIONS.md, Option C);
this is the preprocessing-side alternative that keeps the scorer untouched.
"""
from __future__ import annotations

import numpy as np

from .types import Conversation, Utterance, IntentBucket
from .embeddings import EmbeddingCache

_ACTORS = ("agent", "user")


def _rle_with_smoothing(labels: list[int], min_run: int) -> list[list[int]]:
    """Run-length-encode `labels` into runs (lists of indices), then absorb a
    short interior run (len < min_run) whose two neighbours share a label.

    Example with min_run=2: labels [0,0,1,0] -> the lone `1` is flanked by `0`
    on both sides, so it is absorbed -> a single run [0,1,2,3].
    """
    if not labels:
        return []
    runs: list[list[int]] = [[0]]
    for i in range(1, len(labels)):
        if labels[i] == labels[runs[-1][0]]:
            runs[-1].append(i)
        else:
            runs.append([i])

    # Absorb short interior runs flanked by the same label. Restart after each
    # merge so newly-adjacent runs are re-examined; terminates because every
    # merge strictly reduces the run count.
    changed = True
    while changed and len(runs) >= 3:
        changed = False
        for k in range(1, len(runs) - 1):
            mid = runs[k]
            if len(mid) < min_run and labels[runs[k - 1][0]] == labels[runs[k + 1][0]]:
                runs[k - 1: k + 2] = [runs[k - 1] + mid + runs[k + 1]]
                changed = True
                break
    return runs


def segment_conversation(
    conv: Conversation,
    all_buckets: list[IntentBucket],
    emb: EmbeddingCache,
    min_run: int = 2,
) -> Conversation:
    """Collapse `conv` to stage-level segments against `all_buckets`.

    Each segment is an `Utterance` whose `embedding` is the unit-normalised mean
    of its members' embeddings (the `text` is a synthetic, position-tagged join
    used only as a unique cache key). Collapsing is done per actor stream so the
    alternating agent/user turns of a stage each yield one segment, then segments
    are re-interleaved by the original position of each run's first member.
    FuDGE's actor constraint is preserved (assignment is within-actor).

    Known limitation: when a run merges non-contiguous turns, no linear order can
    preserve every original pairwise ordering — first-member position is a
    deterministic tie-break, not a faithful ordering. Inherent to collapsing.

    Returns a new Conversation; `len(result.utterances) <= len(conv.utterances)`.
    """
    # A single-bucket actor stream has no granularity to normalise against —
    # every turn would map to that one bucket and the whole stream would collapse
    # to ONE segment per conversation (e.g. the TV prefix-tree, whose agent-only
    # labelling leaves exactly one `_user_turn` client bucket). Leave streams
    # with fewer than 2 buckets uncollapsed: each turn stays its own segment.
    centroids: dict[str, np.ndarray | None] = {}
    for a in _ACTORS:
        bs = [b for b in all_buckets if b.actor == a]
        centroids[a] = (np.array([emb.intent_centroid(b) for b in bs])
                        if len(bs) >= 2 else None)

    segments: list[tuple[int, Utterance]] = []
    for a in _ACTORS:
        idx = [i for i, u in enumerate(conv.utterances) if u.actor == a]
        if not idx:
            continue
        texts = [conv.utterances[i].text for i in idx]
        U = emb.encode_batch(texts)  # (N, d), unit-normalised
        if centroids[a] is None:
            labels = list(range(len(idx)))  # <2 buckets for this actor -> no collapse
        else:
            labels = [int(x) for x in (U @ centroids[a].T).argmax(axis=1)]

        for run in _rle_with_smoothing(labels, min_run):
            vec = U[run].mean(axis=0)
            nrm = float(np.linalg.norm(vec))
            if nrm > 0:
                vec = vec / nrm
            first_pos = idx[run[0]]
            # Key includes the member count so a literal " | " inside an
            # utterance can't make two different segments share a cost-cache key.
            text = f"[seg@{first_pos}x{len(run)}] " + " | ".join(texts[r] for r in run)
            segments.append((first_pos, Utterance(actor=a, text=text, embedding=vec)))

    segments.sort(key=lambda t: t[0])
    return Conversation(
        utterances=[u for _, u in segments],
        task=conv.task,
        dialogue_id=conv.dialogue_id,
    )
