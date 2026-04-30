import json
from pathlib import Path

import pytest

from fudge.data_loader import (
    _tv_stem_to_dialogue_id,
    load_thousand_voices_dialogues,
)


def _write_pair(tv_root: Path, stem: str, lines: list[str], trauma_type: str,
                session_topic: str = "topic") -> None:
    (tv_root / "conversations" / f"{stem}_conversation.json").write_text(
        json.dumps({"full_conversation": lines}), encoding="utf-8"
    )
    (tv_root / "metadata" / f"{stem}_metadata.json").write_text(
        json.dumps({
            "client_profile": {},
            "therapist_profile": {},
            "trauma_info": {"type": trauma_type, "session_topic": session_topic},
        }),
        encoding="utf-8",
    )


@pytest.fixture
def tv_root(tmp_path: Path) -> Path:
    root = tmp_path / "ThousandVoicesOfTrauma"
    (root / "conversations").mkdir(parents=True)
    (root / "metadata").mkdir(parents=True)
    _write_pair(
        root, "1_P10",
        [
            "Therapist: How are you feeling today?",
            "Client: Anxious. The flashbacks have been intense.",
            "Therapist: Let's start with a grounding exercise.",
        ],
        trauma_type="accidents",
        session_topic="airplane crash",
    )
    _write_pair(
        root, "2_P7",
        [
            "Therapist: Welcome back.",
            "Client: Thanks for seeing me.",
        ],
        trauma_type="combat or war experiences",
    )
    return root


def test_loads_conversations_with_actor_mapping(tv_root: Path) -> None:
    convs = load_thousand_voices_dialogues(tv_root)
    assert len(convs) == 2
    by_id = {c.dialogue_id: c for c in convs}

    c1 = by_id[_tv_stem_to_dialogue_id("1_P10")]
    assert c1.task == "accidents"
    assert [u.actor for u in c1.utterances] == ["agent", "user", "agent"]
    assert c1.utterances[0].text == "How are you feeling today?"
    assert c1.utterances[1].text.startswith("Anxious")


def test_dialogue_ids_are_unique_ints(tv_root: Path) -> None:
    convs = load_thousand_voices_dialogues(tv_root)
    ids = [c.dialogue_id for c in convs]
    assert all(isinstance(d, int) for d in ids)
    assert len(set(ids)) == len(ids)


def test_task_field_session_topic(tv_root: Path) -> None:
    convs = load_thousand_voices_dialogues(tv_root, task_field="session_topic")
    by_id = {c.dialogue_id: c for c in convs}
    assert by_id[_tv_stem_to_dialogue_id("1_P10")].task == "airplane crash"


def test_require_phases_filter(tv_root: Path) -> None:
    convs = load_thousand_voices_dialogues(tv_root, require_phases=("P10",))
    assert len(convs) == 1
    assert convs[0].dialogue_id == _tv_stem_to_dialogue_id("1_P10")


def test_skips_unknown_speaker(tv_root: Path) -> None:
    _write_pair(
        tv_root, "3_P5",
        [
            "Therapist: Hi.",
            "Narrator: (the patient hesitated)",
            "Client: Hello.",
        ],
        trauma_type="bullying",
    )
    convs = load_thousand_voices_dialogues(tv_root)
    by_id = {c.dialogue_id: c for c in convs}
    c3 = by_id[_tv_stem_to_dialogue_id("3_P5")]
    assert [u.actor for u in c3.utterances] == ["agent", "user"]


def test_missing_dirs_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_thousand_voices_dialogues(tmp_path / "does_not_exist")


def test_skips_pair_with_missing_metadata(tv_root: Path) -> None:
    (tv_root / "conversations" / "9_P11_conversation.json").write_text(
        json.dumps({"full_conversation": ["Therapist: Hi.", "Client: Bye."]}),
        encoding="utf-8",
    )
    convs = load_thousand_voices_dialogues(tv_root)
    assert all(c.dialogue_id != _tv_stem_to_dialogue_id("9_P11") for c in convs)
