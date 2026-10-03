import json
import re

import pytest

from mafia_lie_detector.annotate import llm_assist
from mafia_lie_detector.schema import NARRATOR, Alignment, Game, Player, Turn


@pytest.fixture
def draft() -> Game:
    players = [
        Player(player_id="P1", model="llm_a", role="mafia", alignment=Alignment.DECEIVER),
        Player(player_id="P2", model="llm_b", role="villager", alignment=Alignment.TRUTHFUL),
    ]
    turns = [Turn(turn_id=i, text=t) for i, t in enumerate(
        ["Welcome to the game.", "I am a villager, I swear.", "I think P1 is lying."])]
    return Game(game_id="d", players=players, turns=turns)


def suggestion(turn_id, speaker, phase="day", claims=()):
    return {"turn_id": turn_id, "speaker_id": speaker, "phase": phase, "claims": list(claims)}


LIE = {"text": "I am a villager", "kind": "role_claim", "truthful": False}


def test_parse_handles_fences_and_chatter():
    raw = 'Sure!\n```json\n[{"turn_id": 1, "speaker_id": "P1"}, {"no_id": 2}, "junk"]\n```'
    assert llm_assist.parse_suggestions(raw) == [{"turn_id": 1, "speaker_id": "P1"}]
    with pytest.raises(ValueError):
        llm_assist.parse_suggestions("I cannot do that")


def test_apply_fills_blanks_and_derives_deceptive(draft):
    counts = llm_assist.apply_suggestions(draft, [
        suggestion(0, NARRATOR, "intro"), suggestion(1, "P1", claims=[LIE]), suggestion(2, "P2"),
    ])
    assert counts == {"speaker": 3, "phase": 3, "claims": 1}
    t = draft.turns[1]
    assert (t.speaker_id, t.phase, t.deceptive) == ("P1", "day", True)
    assert t.claims[0].kind.value == "role_claim"


def test_apply_never_overwrites_human_values(draft):
    draft.turns[1].speaker_id = "P2"
    draft.turns[1].phase = "vote"
    draft.turns[1].deceptive = False
    counts = llm_assist.apply_suggestions(draft, [suggestion(1, "P1", "day", claims=[LIE])])
    t = draft.turns[1]
    assert (t.speaker_id, t.phase, t.deceptive, t.claims) == ("P2", "vote", False, [])
    assert counts == {"speaker": 0, "phase": 0, "claims": 0}


def test_apply_ignores_invalid_suggestions(draft):
    counts = llm_assist.apply_suggestions(draft, [
        suggestion(1, "ghost", "nonsense", claims=[{"text": "x", "truthful": "maybe"}, {"truthful": True}]),
        suggestion(99, "P1"),
    ])
    assert counts == {"speaker": 0, "phase": 0, "claims": 0}
    assert draft.turns[1].speaker_id is None and draft.turns[1].claims == []


def test_unknown_claim_kind_falls_back_to_other(draft):
    llm_assist.apply_suggestions(draft, [suggestion(1, "P1", claims=[{"text": "x", "kind": "weird", "truthful": True}])])
    assert draft.turns[1].claims[0].kind.value == "other" and draft.turns[1].deceptive is False


def test_prompt_contains_roster_and_only_requested_turns(draft):
    prompt = llm_assist.build_prompt(draft, [1])
    assert "P1 | model=llm_a | role=mafia | alignment=deceiver" in prompt
    assert "[1] I am a villager, I swear." in prompt and "[2]" not in prompt
    assert NARRATOR in prompt


def test_annotate_game_chunks_and_stays_draft(draft):
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        ids = [int(x) for x in re.findall(r"^\[(\d+)\]", prompt, flags=re.M)]
        return "```json\n" + json.dumps([suggestion(i, "P1" if i else NARRATOR) for i in ids]) + "\n```"

    totals = llm_assist.annotate_game(draft, fake, chunk_size=2)
    assert len(prompts) == 2 and totals["speaker"] == 3
    assert draft.annotation_status == "draft"


def test_annotate_game_requires_a_roster():
    with pytest.raises(ValueError, match="players"):
        llm_assist.annotate_game(Game(game_id="x", turns=[Turn(turn_id=0, text="a")]), lambda p: "[]")
