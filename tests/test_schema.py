import pytest
from pydantic import ValidationError

from mafia_lie_detector.schema import (
    NARRATOR, Alignment, Claim, Game, Player, Turn, find_game_files, load_game, save_game, suggest_alignment,
)


def _player(pid="P1", **kw):
    return Player(player_id=pid, model="llm_a", role=kw.pop("role", "villager"),
                  alignment=kw.pop("alignment", Alignment.TRUTHFUL), **kw)


def test_deceptive_is_derived_from_claims():
    t = Turn(turn_id=0, text="x", claims=[Claim(text="a", truthful=True), Claim(text="b", truthful=False)])
    assert t.deceptive is True
    assert Turn(turn_id=1, text="x", claims=[Claim(text="a", truthful=True)]).deceptive is False


def test_deceptive_contradicting_claims_is_rejected():
    with pytest.raises(ValidationError, match="contradicts"):
        Turn(turn_id=0, text="x", deceptive=False, claims=[Claim(text="a", truthful=False)])


def test_end_before_start_is_rejected():
    with pytest.raises(ValidationError):
        Turn(turn_id=0, text="x", start=5.0, end=4.0)


def test_role_is_normalised():
    assert _player(role="  Mafia ").role == "mafia"


def test_unknown_speaker_rejected():
    with pytest.raises(ValidationError, match="unknown speaker"):
        Game(game_id="g", players=[_player()], turns=[Turn(turn_id=0, speaker_id="ghost", text="hi")])


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate turn_id"):
        Game(game_id="g", turns=[Turn(turn_id=0, text="a"), Turn(turn_id=0, text="b")])
    with pytest.raises(ValidationError, match="duplicate player_id"):
        Game(game_id="g", players=[_player(), _player()], turns=[])


def test_narrator_id_is_reserved():
    with pytest.raises(ValidationError, match="reserved"):
        Game(game_id="g", players=[_player(NARRATOR)], turns=[])


def test_narrator_is_a_valid_speaker():
    Game(game_id="g", turns=[Turn(turn_id=0, speaker_id=NARRATOR, text="Night falls.")])


def test_completeness_problems(small_game):
    assert small_game.completeness_problems() == []
    draft = Game(game_id="d", turns=[Turn(turn_id=0, text="a"), Turn(turn_id=1, text="b")])
    problems = draft.completeness_problems()
    assert "no players listed" in problems
    assert any("without a speaker" in p for p in problems)


def test_a_game_needs_a_deceiver():
    g = Game(game_id="g", players=[_player()], turns=[Turn(turn_id=0, speaker_id="P1", text="hi")])
    assert "no deceiver among players" in g.completeness_problems()


def test_roundtrip_keeps_draft_nulls(tmp_path, small_game):
    draft = Game(game_id="d", turns=[Turn(turn_id=0, text="a")])
    path = tmp_path / "sub" / "d.json"
    save_game(draft, path)
    assert '"speaker_id": null' in path.read_text()
    assert load_game(path) == draft
    save_game(small_game, tmp_path / "g1.json")
    assert load_game(tmp_path / "g1.json") == small_game


def test_find_game_files_expands_directories(tmp_path, small_game):
    save_game(small_game, tmp_path / "a.json")
    save_game(small_game, tmp_path / "nested" / "b.json")
    (tmp_path / "notes.txt").write_text("ignored")
    assert [p.name for p in find_game_files([tmp_path])] == ["a.json", "b.json"]


def test_suggest_alignment():
    assert suggest_alignment("Werewolf") is Alignment.DECEIVER
    assert suggest_alignment("doctor") is Alignment.TRUTHFUL
