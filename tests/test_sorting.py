import json

import pytest

from mafia_lie_detector import cli, sorting
from mafia_lie_detector.schema import (
    HUMAN, NARRATOR, UNKNOWN_MODEL, UNKNOWN_SPEAKER, Alignment, Game, Player, Turn, save_game,
)


def make_game(game_id="g1", winner="town", unknown_turns=0) -> Game:
    players = [
        Player(player_id="P1", model="model-one", family="Vendor A", role="mafia", alignment=Alignment.DECEIVER,
               evidence="name tag, frame f_00004"),
        Player(player_id="P2", model="model-two", family="vendor-b", role="jester", alignment=Alignment.DECEIVER),
        Player(player_id="P3", model="model-one", family="Vendor A", role="villager", alignment=Alignment.TRUTHFUL),
        Player(player_id="P4", model=HUMAN, role="villager", alignment=Alignment.TRUTHFUL),
        Player(player_id="P5", model=UNKNOWN_MODEL, family="vendor-c", role="doctor", alignment=Alignment.TRUTHFUL),
        Player(player_id="P6", model="model-three", role="villager", alignment=Alignment.TRUTHFUL),  # no family
    ]
    turns = [Turn(turn_id=0, speaker_id=NARRATOR, text="Welcome.", phase="intro")]
    for p in players:
        turns.append(Turn(turn_id=len(turns), speaker_id=p.player_id, text=f"{p.player_id} speaking", phase="day"))
    turns.append(Turn(turn_id=len(turns), speaker_id="P1", text="Well played.", phase="postgame"))
    for _ in range(unknown_turns):
        turns.append(Turn(turn_id=len(turns), speaker_id=UNKNOWN_SPEAKER, text="who said this?", phase="day"))
    return Game(game_id=game_id, players=players, turns=turns, winner=winner, annotation_status="reviewed")


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_folder_layout_and_lying_truth_by_alignment(tmp_path):
    report = sorting.sort_games([make_game()], tmp_path)
    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.jsonl"))
    assert files == [
        "vendor-a/model-one/lying/g1__p1.jsonl",   # mafia
        "vendor-a/model-one/truth/g1__p3.jsonl",   # villager
        "vendor-b/model-two/lying/g1__p2.jsonl",   # jester counts as lying
    ]
    assert report.files == 3


def test_rows_carry_metadata_and_exclude_postgame_text(tmp_path):
    sorting.sort_games([make_game(winner="mafia")], tmp_path)
    rows = read(tmp_path / "vendor-a/model-one/lying/g1__p1.jsonl")
    assert [r["text"] for r in rows] == ["P1 speaking"]  # "Well played." was postgame
    r = rows[0]
    assert (r["winner"], r["role"], r["alignment"], r["model"], r["family"]) == ("mafia", "mafia", "deceiver", "model-one", "Vendor A")
    assert r["addressees"] == []


def test_addressees_are_carried_into_rows(tmp_path):
    game = make_game()
    game.turns.append(Turn(turn_id=200, speaker_id="P1", text="P3, you are lying.", phase="day", addressees=["P3", "all"]))
    sorting.sort_games([game], tmp_path)
    rows = read(tmp_path / "vendor-a/model-one/lying/g1__p1.jsonl")
    assert rows[-1]["addressees"] == ["P3", "all"]


def test_discards_humans_unverified_models_and_missing_family_with_reasons(tmp_path):
    report = sorting.sort_games([make_game()], tmp_path)
    assert report.discarded_players == {
        "g1/P4": "human player",
        "g1/P5": "exact model not verified",
        "g1/P6": "model family not recorded",
    }
    written = [str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.jsonl")]
    assert written and not any("human" in w or "unknown" in w for w in written)


def test_game_with_too_many_unattributed_turns_is_discarded(tmp_path):
    ok = make_game("ok")
    bad = make_game("bad", unknown_turns=12)
    report = sorting.sort_games([ok, bad], tmp_path, max_unknown=0.4)
    assert "bad" in report.discarded_games and "ok" not in report.discarded_games
    assert "no established speaker" in report.discarded_games["bad"][0]
    assert not list(tmp_path.rglob("g1*bad*")) and (tmp_path / "vendor-a/model-one/lying/ok__p1.jsonl").exists()
    # A few unattributed turns are tolerated, and are never written out.
    tolerated = sorting.sort_games([make_game("few", unknown_turns=2)], tmp_path / "t")
    assert not tolerated.discarded_games
    assert all("who said" not in p.read_text() for p in (tmp_path / "t").rglob("*.jsonl"))


def test_drafts_and_unknown_winner(tmp_path):
    draft = make_game("d").model_copy(update={"annotation_status": "draft"})
    report = sorting.sort_games([draft, make_game("w", winner=None)], tmp_path)
    assert report.discarded_games["d"] == ["annotation_status is not 'reviewed'"]
    assert report.unknown_winner == ["w"]
    assert json.loads((tmp_path / sorting.REPORT_NAME).read_text())["games_with_unknown_winner"] == ["w"]


def test_safe_segment_blocks_traversal_and_rejects_empty():
    assert sorting.safe_segment("../../Evil Model/1.5") == "evil-model-1.5"
    assert sorting.safe_segment("GPT 4o (mini)") == "gpt-4o-mini"
    with pytest.raises(ValueError):
        sorting.safe_segment("///")


def test_clean_only_deletes_directories_this_tool_wrote(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="refusing"):
        sorting.clean_output(foreign)
    assert (foreign / "precious.txt").exists()
    mine = tmp_path / "mine"
    sorting.sort_games([make_game()], mine)
    sorting.clean_output(mine)
    assert not mine.exists()
    sorting.clean_output(tmp_path / "never-existed")  # no-op


def test_cli_sort(tmp_path, capsys):
    ann = tmp_path / "ann"
    save_game(make_game(), ann / "g1.json")
    out = tmp_path / "dataset"
    assert cli.main(["sort", str(ann), "--out", str(out)]) == 0
    text = capsys.readouterr()
    assert "vendor-a/model-one/lying: 1 turns" in text.out and "discarded player g1/P4: human player" in text.err
    assert cli.main(["sort", str(ann), "--out", str(out), "--clean"]) == 0
    other = tmp_path / "other"
    other.mkdir()
    assert cli.main(["sort", str(ann), "--out", str(other), "--clean"]) == 1  # not ours: refuse, exit 1
    assert cli.main(["sort", str(tmp_path / "missing"), "--out", str(tmp_path / "o2")]) == 2
    assert "not found" in capsys.readouterr().err


def test_custom_ais_go_to_their_own_box(tmp_path):
    game = make_game()
    game.players.append(Player(player_id="z2", model="z2", family="custom", role="mafia", alignment=Alignment.DECEIVER))
    game.players.append(Player(player_id="bot9", model=UNKNOWN_MODEL, family="Custom", role="villager",
                               alignment=Alignment.TRUTHFUL))
    game.turns.append(Turn(turn_id=100, speaker_id="z2", text="I am totally town.", phase="day"))
    game.turns.append(Turn(turn_id=101, speaker_id="bot9", text="Beep, I trust Z2.", phase="day"))
    report = sorting.sort_games([game], tmp_path)
    assert (tmp_path / "_unidentified/z2/lying/g1__z2.jsonl").exists()
    assert (tmp_path / "_unidentified/bot9/truth/g1__bot9.jsonl").exists()  # unknown model: named by player
    assert report.written["_unidentified/z2/lying"] == 1
    assert "g1/z2" not in report.discarded_players


def test_unknown_speaker_is_a_valid_but_reserved_id():
    Game(game_id="g", turns=[Turn(turn_id=0, speaker_id=UNKNOWN_SPEAKER, text="?")])
    with pytest.raises(ValueError, match="reserved"):
        Game(game_id="g", players=[Player(player_id=UNKNOWN_SPEAKER, model="m", role="r", alignment=Alignment.TRUTHFUL)], turns=[])
