"""Sanity checks on the committed answer keys in ground_truth/."""
from pathlib import Path

import pytest

from mafia_lie_detector import cli
from mafia_lie_detector.schema import Alignment, load_game

KEYS = sorted((Path(__file__).parent.parent / "ground_truth").glob("*.json"))


def test_there_is_at_least_one_key():
    assert KEYS


@pytest.mark.parametrize("path", KEYS, ids=lambda p: p.stem)
def test_answer_key_is_consistent(path):
    game = load_game(path)
    assert game.turns == []  # answer keys hold facts only, never dialogue
    assert game.winner in ("mafia", "town")
    for p in game.players:
        assert p.family and p.model and p.evidence
    alive = [p for p in game.players if p.eliminated_round is None]
    deceivers = sum(p.alignment is Alignment.DECEIVER for p in alive)
    town = len(alive) - deceivers
    if game.winner == "mafia":
        assert deceivers >= town > -1 and deceivers > 0
    else:
        assert deceivers == 0


def test_ten_ais_play_mafia_roster():
    game = load_game(next(k for k in KEYS if k.stem == "10-ais-play-mafia"))
    roles = {p.player_id: (p.role, p.alignment.value) for p in game.players}
    assert len(roles) == 10
    assert {pid for pid, (_, a) in roles.items() if a == "deceiver"} == {"chatgpt-5.1", "llama-4", "gemini-2.5-pro"}
    assert roles["claude-sonnet-4.5"] == ("doctor", "truthful") and roles["grok-4.1"] == ("sheriff", "truthful")
    assert {p.player_id for p in game.players if p.eliminated_round is None} == {"chatgpt-5.1", "chatgpt-4o"}


def test_draft_with_roster_prefills_players(tmp_path):
    key = next(k for k in KEYS if k.stem == "10-ais-play-mafia")
    cap = tmp_path / "abcDEF12345.json"
    cap.write_text('{"video_id": "abcDEF12345", "source": "srt", "snippets": [{"text": "hi", "start": 0, "duration": 1}]}')
    out = tmp_path / "g.json"
    assert cli.main(["draft", str(cap), "-o", str(out), "--roster", str(key)]) == 0
    game = load_game(out)
    assert game.game_id == "10-ais-play-mafia" and game.source == "youtube:abcDEF12345"
    assert len(game.players) == 10 and game.winner == "mafia" and game.events
    assert game.annotation_status == "draft" and game.turns[0].speaker_id is None
