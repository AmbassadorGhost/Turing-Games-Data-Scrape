import pytest

from mafia_lie_detector.dataset import (
    DEFAULT_EXCLUDED_PHASES, build_examples, read_examples, scrub_names, write_examples,
)
from mafia_lie_detector.schema import Game, Turn


def by_turn(result):
    return {e.turn_id: e for e in result.examples}


def test_excludes_narrator_reveal_postgame_and_unlabelled(small_game):
    got = by_turn(build_examples([small_game]))
    assert set(got) == {1, 2, 3}  # not 0 (intro/narrator), 4 (unlabelled), 5/6 (reveal/postgame)
    assert got[1].label == 1 and got[2].label == 0


def test_alignment_target_uses_player_side_and_keeps_unlabelled_turns(small_game):
    got = by_turn(build_examples([small_game], "alignment"))
    assert set(got) == {1, 2, 3, 4}
    assert [got[i].label for i in (1, 2, 3, 4)] == [1, 0, 0, 0]
    assert got[1].speaker_model == "llm_a" and got[1].role == "mafia"


def test_reveal_text_never_enters_context(small_game):
    ex = by_turn(build_examples([small_game], context_turns=5))
    assert ex[1].context == ""  # the intro line is narrator-only
    assert "was the mafia" not in " ".join(e.context for e in ex.values())
    assert ex[3].context.splitlines()[0].startswith("P1: ")


def test_context_window_is_bounded(small_game):
    assert len(by_turn(build_examples([small_game], context_turns=1))[3].context.splitlines()) == 1


def test_scrub_replaces_ids_and_aliases_in_text_and_context(small_game):
    ex = by_turn(build_examples([small_game], scrub=True, context_turns=2))
    assert ex[1].text == "I am just a villager, trust me. <PLAYER> here has nothing to hide."
    assert ex[2].text == "I think <PLAYER> is acting strange."
    assert "P1" not in ex[3].context and "<PLAYER>:" in ex[3].context


def test_scrub_respects_word_boundaries():
    assert scrub_names("P2 and p22 and xP2", ["P2"]) == "<PLAYER> and p22 and xP2"
    assert scrub_names("Player 3 spoke", ["Player 3", "P"]) == "<PLAYER> spoke"  # 1-char names ignored
    assert scrub_names("nothing", []) == "nothing"


def test_incomplete_or_unreviewed_games_are_skipped_with_reasons(small_game):
    draft = Game(game_id="draft", turns=[Turn(turn_id=0, text="hi")])
    unreviewed = small_game.model_copy(update={"game_id": "u", "annotation_status": "draft"})
    result = build_examples([small_game, draft, unreviewed])
    assert {e.game_id for e in result.examples} == {"g1"}
    assert "no players listed" in result.skipped_games["draft"]
    assert result.skipped_games["u"] == ["annotation_status is not 'reviewed'"]
    assert build_examples([unreviewed], require_reviewed=False).examples


def test_custom_exclusions(small_game):
    got = by_turn(build_examples([small_game], "alignment", exclude_phases=["day"]))
    assert set(got) == {6}  # postgame now allowed, day dropped
    assert "reveal" in DEFAULT_EXCLUDED_PHASES


def test_bad_target_rejected(small_game):
    with pytest.raises(ValueError):
        build_examples([small_game], "nope")


def test_jsonl_roundtrip(tmp_path, small_game):
    examples = build_examples([small_game], context_turns=1).examples
    path = tmp_path / "out" / "turns.jsonl"
    assert write_examples(examples, path) == len(examples)
    assert read_examples(path) == examples
