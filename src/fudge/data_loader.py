"""Load STAR dataset and build supervised dialogue flows from task definitions."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict

from .types import Utterance, Conversation, IntentBucket, DialogueFlow


def _hash_taxonomy(taxonomy: dict) -> str:
    """Same hash the LLM-labeling script writes into per-dialogue files
    (`scripts/llm_label_star.py:run_pipeline`)."""
    blob = json.dumps(taxonomy, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_star_dialogues(star_dir: str, filter_unlabeled: bool = True) -> list[Conversation]:
    """
    Load dialogues from STAR.

    Wizard 'pick_suggestion' events are the agent utterances (with ActionLabel = intent).
    User 'utter' events are the user utterances.
    Task is identified from APIName in query events.

    If filter_unlabeled is True (default), conversations containing Wizard 'utter'
    events (free-typed agent responses with no intent label) are excluded.  The paper
    states: "We processed the dialogues in the STAR dataset and removed those with
    unlabeled agent utterances."
    """
    conversations = []
    dialogue_dir = Path(star_dir) / "dialogues"

    for filename in sorted(dialogue_dir.glob("*.json"), key=lambda f: int(f.stem)):
        with open(filename) as f:
            data = json.load(f)

        utterances = []
        task = ""
        intent_sequence = []  # (intent_label, actor, text) for flow building
        has_unlabeled_agent = False

        # First pass: collect raw events
        raw_events = []
        for event in data.get("Events", []):
            if event.get("APIName") and not task:
                task = event["APIName"]

            if event.get("Agent") == "User" and event.get("Action") == "utter":
                text = event.get("Text", "")
                if text:
                    raw_events.append(("user", None, text))

            elif event.get("Agent") == "Wizard" and event.get("Action") == "pick_suggestion":
                text = event.get("Text", "")
                label = event.get("ActionLabel", "")
                if text:
                    raw_events.append(("agent", label, text))

            elif event.get("Agent") == "Wizard" and event.get("Action") == "utter":
                if event.get("Text", ""):
                    has_unlabeled_agent = True

        if filter_unlabeled and has_unlabeled_agent:
            continue

        # Second pass: label user utterances by the next agent intent
        for i, (actor, label, text) in enumerate(raw_events):
            utterances.append(Utterance(actor=actor, text=text))
            if actor == "user":
                # Find next agent intent to create a contextual label
                next_label = "unknown"
                for j in range(i + 1, len(raw_events)):
                    if raw_events[j][0] == "agent" and raw_events[j][1]:
                        next_label = raw_events[j][1]
                        break
                intent_sequence.append((f"user_before_{next_label}", "user", text))
            else:
                intent_sequence.append((label, "agent", text))

        if utterances and task:
            conv = Conversation(
                utterances=utterances,
                task=task,
                dialogue_id=int(data.get("DialogueID", -1)),
            )
            conv._intent_sequence = intent_sequence  # stash for flow building
            conversations.append(conv)

    return conversations


def load_llm_labels(label_dir: str | Path) -> dict[int, list[str]]:
    """Load per-dialogue LLM-generated label files.

    Reads every <dialogue_id>.json under `label_dir` produced by
    scripts/llm_label_star.py. Returns dialogue_id -> utterance_labels (one
    label per Conversation.utterances entry).

    Validates `taxonomy_version` against the sibling `taxonomy.json`
    (one level up from `label_dir`, e.g. `<task>/<taxonomy_method>/taxonomy.json`).
    Raises if any per-dialogue file references a different taxonomy than the
    one currently on disk — this catches stale outputs from a prior
    bootstrap that would otherwise silently mix taxonomies and invalidate
    flow-construction.

    If no sibling `taxonomy.json` exists, version checking is skipped (legacy
    layouts).
    """
    label_dir = Path(label_dir)
    taxonomy_path = label_dir.parent / "taxonomy.json"
    expected_version: str | None = None
    if taxonomy_path.exists():
        with open(taxonomy_path, encoding="utf-8") as f:
            expected_version = _hash_taxonomy(json.load(f))

    out: dict[int, list[str]] = {}
    mismatched: list[tuple[int, str | None]] = []
    for f in label_dir.glob("*.json"):
        try:
            did = int(f.stem)
        except ValueError:
            continue
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        version = data.get("taxonomy_version")
        if expected_version is not None and version != expected_version:
            mismatched.append((did, version))
            continue
        out[did] = list(data["utterance_labels"])

    if mismatched:
        sample = mismatched[:5]
        more = "..." if len(mismatched) > 5 else ""
        raise ValueError(
            f"Stale taxonomy_version on label files in {label_dir} (expected "
            f"{expected_version}). Mismatched dialogues: "
            f"{[(d, v) for d, v in sample]}{more} "
            f"(total {len(mismatched)} of {len(mismatched) + len(out)}). "
            "Re-run `python scripts/llm_label_star.py ...` to refresh, or "
            "clear the directory before re-running with --limit."
        )
    return out


def group_by_task(conversations: list[Conversation]) -> dict[str, list[Conversation]]:
    """Group conversations by task name."""
    result: dict[str, list[Conversation]] = defaultdict(list)
    for conv in conversations:
        result[conv.task].append(conv)
    return dict(result)


def build_flow_from_task_definition(star_dir: str, task_name: str) -> tuple[DialogueFlow, list[IntentBucket]]:
    """
    Build a DialogueFlow from the official STAR task definition JSON.

    The task JSON has:
    - 'replies': dict mapping intent_label -> template text
    - 'graph': dict mapping intent_label -> next_intent_label (linear transitions)

    This gives us the official task flow structure.
    """
    task_dir = Path(star_dir) / "tasks" / task_name
    task_file = task_dir / f"{task_name}.json"

    with open(task_file) as f:
        task_data = json.load(f)

    replies = task_data.get("replies", {})
    graph = task_data.get("graph", {})

    # Build intent buckets from template replies
    # All replies are agent utterances in the task definition
    buckets: dict[str, IntentBucket] = {}
    for label, text in replies.items():
        buckets[label] = IntentBucket(actor="agent", utterances=[text], label=label)

    # Build flow from graph
    flow = DialogueFlow()
    for label in buckets:
        flow.add_node(label, buckets[label])

    # Find root(s): nodes that appear as source but not as target
    sources = set(graph.keys())
    targets = set(graph.values())
    roots = sources - targets
    if not roots:
        roots = {list(graph.keys())[0]} if graph else set()

    for root_label in roots:
        flow.add_edge(flow.root, root_label)

    for src, dst in graph.items():
        if src in buckets and dst in buckets:
            flow.add_edge(src, dst)

    return flow, list(buckets.values())


def build_flow_from_conversations(conversations: list[Conversation],
                                  star_dir: str = "",
                                  task_name: str = "",
                                  label_source: dict[int, list[str]] | None = None) -> tuple[DialogueFlow, list[IntentBucket]]:
    """
    Build a supervised flow as a prefix-trie DAG from observed intent sequences.

    Each conversation has an _intent_sequence: list of (label, actor, text).
    We merge common prefixes into a trie, then attach intent buckets to each node.

    This avoids parsing the task definition's conditional graph structure.

    If `label_source` is provided (mapping dialogue_id -> [label, label, ...]
    aligned with Conversation.utterances), the existing _intent_sequence labels
    are replaced. Actor and text still come from the STAR event; only the label
    changes. Use this to consume LLM-generated labels from
    scripts/llm_label_star.py.
    """
    # Step 1: Extract (actor, label) sequences and collect utterances per (actor, label)
    utterances_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    sequences: list[list[tuple[str, str]]] = []

    for conv in conversations:
        if label_source is not None:
            if conv.dialogue_id < 0 or conv.dialogue_id not in label_source:
                continue
            llm_labels = label_source[conv.dialogue_id]
            if len(llm_labels) != len(conv.utterances):
                raise ValueError(
                    f"Label count mismatch for dialogue {conv.dialogue_id}: "
                    f"{len(llm_labels)} labels vs {len(conv.utterances)} utterances"
                )
            triples = [
                (lbl, u.actor, u.text)
                for lbl, u in zip(llm_labels, conv.utterances)
            ]
        else:
            if not hasattr(conv, '_intent_sequence'):
                continue
            triples = list(conv._intent_sequence)

        seq = []
        for label, actor, text in triples:
            key = (actor, label)
            utterances_by_key[key].append(text)
            seq.append(key)
        if seq:
            sequences.append(seq)

    # Step 2: Build intent buckets per (actor, label)
    buckets: dict[tuple[str, str], IntentBucket] = {}
    for key, texts in utterances_by_key.items():
        actor, label = key
        unique_texts = list(dict.fromkeys(texts))
        buckets[key] = IntentBucket(actor=actor, utterances=unique_texts, label=label)

    all_buckets = list(buckets.values())

    # Step 3: Build prefix-trie DAG
    flow = DialogueFlow()
    node_counter = 0

    # Trie node: dict mapping (actor,label) -> child_trie_node
    # Each trie node also stores its flow node_id
    class TrieNode:
        def __init__(self, node_id: str):
            self.node_id = node_id
            self.children: dict[tuple[str, str], 'TrieNode'] = {}

    root_trie = TrieNode(flow.root)

    for seq in sequences:
        current = root_trie
        for key in seq:
            if key not in current.children:
                node_counter += 1
                nid = f"n{node_counter}_{key[0]}_{key[1]}"
                flow.add_node(nid, buckets[key])
                flow.add_edge(current.node_id, nid)
                current.children[key] = TrieNode(nid)
            current = current.children[key]

    return flow, all_buckets
