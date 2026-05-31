"""Versioned, deterministic train/test splits for STAR and TV.

Splits are stored as JSON: a mapping from stratum key to {"train": [ids],
"test": [ids]}. For STAR the stratum key is the task name; for TV it's the
phase (or phase+type, configurable).

A split is locked once written. To revise, bump the version (STAR_v3 etc.)
rather than mutating an existing file — every experiment that cites a
split is reproducible from the version string.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _stratified_split(
    items_by_stratum: dict[str, list[int]],
    train_frac: float,
    seed: int,
) -> dict[str, dict[str, list[int]]]:
    """For each stratum independently, shuffle and take the first
    train_frac as train, the rest as test.

    Strata with fewer than 2 items are placed entirely in train (no test
    sample) and a warning is printed — they cannot contribute to the
    held-out evaluation.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, list[int]]] = {}
    for stratum, ids in items_by_stratum.items():
        ids_sorted = sorted(ids)  # deterministic input order before shuffle
        idx = np.arange(len(ids_sorted))
        rng.shuffle(idx)
        n_train = max(1, int(round(train_frac * len(ids_sorted))))
        if n_train >= len(ids_sorted):
            print(f"  [warn] stratum '{stratum}' has only {len(ids_sorted)} items; "
                  "all assigned to train, no test held out")
            train_ids = [ids_sorted[i] for i in idx]
            test_ids: list[int] = []
        else:
            train_ids = sorted(ids_sorted[i] for i in idx[:n_train])
            test_ids = sorted(ids_sorted[i] for i in idx[n_train:])
        out[stratum] = {"train": train_ids, "test": test_ids}
    return out


def create_star_split(
    star_dir: str,
    out_path: str | Path,
    train_frac: float = 0.7,
    seed: int = 20260530,
    min_per_task: int = 10,
) -> dict:
    """Create a per-task 70/30 stratified STAR split.

    Tasks with fewer than `min_per_task` conversations are dropped (they
    cannot support train+test meaningfully).
    """
    from .data_loader import load_star_dialogues, group_by_task

    convs = load_star_dialogues(star_dir)
    by_task = group_by_task(convs)

    items_by_task: dict[str, list[int]] = {}
    dropped: list[tuple[str, int]] = []
    for task, ts in by_task.items():
        if len(ts) < min_per_task:
            dropped.append((task, len(ts)))
            continue
        items_by_task[task] = [c.dialogue_id for c in ts]

    splits = _stratified_split(items_by_task, train_frac=train_frac, seed=seed)

    meta = {
        "dataset": "STAR",
        "version": Path(out_path).stem,
        "train_frac": train_frac,
        "seed": seed,
        "min_per_task": min_per_task,
        "dropped_tasks": [{"task": t, "n": n} for t, n in sorted(dropped)],
        "n_strata": len(splits),
        "n_train_total": sum(len(v["train"]) for v in splits.values()),
        "n_test_total": sum(len(v["test"]) for v in splits.values()),
        "splits": splits,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_split(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_conversations(
    convs: list, split_meta: dict, stratum: str
) -> tuple[list, list]:
    """Partition a list of Conversation objects according to the stratum
    entry in a saved split. Conversations whose dialogue_id is in train/test
    are returned in those buckets; ids in the stratum but not in `convs`
    are silently dropped (caller is expected to have filtered).
    """
    s = split_meta["splits"][stratum]
    train_ids = set(s["train"])
    test_ids = set(s["test"])
    train = [c for c in convs if c.dialogue_id in train_ids]
    test = [c for c in convs if c.dialogue_id in test_ids]
    return train, test
