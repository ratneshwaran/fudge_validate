# TODO

## Consume LLM-generated labels in `build_flow_from_conversations`

`scripts/llm_label_star.py` writes per-dialogue label files at:

    data/STAR_llm_labels/<task>/<method>/<dialogue_id>.json

with format:

    {
      "utterance_labels": ["label1", "label2", ...],   # one per Conversation.utterances entry
      "taxonomy_version": "<sha256-prefix>"
    }

A separate PR needs to make `src/fudge/data_loader.py` consume these. Concrete
changes:

1. **Expose dialogue ID on `Conversation`.** `load_star_dialogues` currently
   drops `DialogueID`. Add it (e.g., `Conversation.dialogue_id: int`) so the
   label loader can match files. The current LLM script works around this by
   re-walking the dialogues directory in `load_star_with_ids`; that workaround
   should go away when this lands.

2. **Add a `load_llm_labels(label_dir: Path) -> dict[int, list[str]]` helper.**
   Reads all `<dialogue_id>.json` files in a method directory and returns
   `dialogue_id -> utterance_labels`. Optionally validate `taxonomy_version`
   against `taxonomy.json` and warn on mismatch.

3. **Modify `build_flow_from_conversations` to accept an optional label source.**
   New signature:

       build_flow_from_conversations(
           conversations,
           star_dir="",
           task_name="",
           label_source: dict[int, list[str]] | None = None,
       )

   When `label_source` is provided:
   - For each conversation, replace its `_intent_sequence` labels with
     `label_source[conv.dialogue_id]`. Keep `actor` from the STAR event and
     `text` from the utterance — only the label changes.
   - Skip the `user_before_<next_agent_intent>` heuristic entirely.
   - Buckets are still grouped by `(actor, label)` exactly as today, so
     downstream FuDGE code is unaffected.

4. **Wire it into `experiments/validate_discrimination.py`.** Add a CLI flag
   like `--label-source data/STAR_llm_labels/hotel_book/whole` that, when set,
   calls `load_llm_labels` and forwards to `build_flow_from_conversations`.
   Then we can rerun Table 1b with LLM labels and compare to the heuristic
   baseline.

5. **Drop the `load_star_with_ids` workaround in `scripts/llm_label_star.py`**
   once step 1 lands — replace with a direct call to `load_star_dialogues`.

The point of all this: the labeling pipeline must work the same way on
Thousand Voices, where there are no gold ActionLabels at all. Steps 1–4 make
labels a pluggable input; step 5 keeps the script honest.
