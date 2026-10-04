import json

from mafia_lie_detector import cli
from mafia_lie_detector.dataset import build_examples
from mafia_lie_detector.export import assign_splits, export_clean, normalise_text
from mafia_lie_detector.schema import NARRATOR, Alignment, Game, Player, Turn, save_game


def game(game_id="g1") -> Game:
    players = [
        Player(player_id="ai1", model="model-a", family="vendor", role="mafia", alignment=Alignment.DECEIVER),
        Player(player_id="ai2", model="model-b", family="vendor", role="villager", alignment=Alignment.TRUTHFUL),
        Player(player_id="host", model="human", role="villager", alignment=Alignment.TRUTHFUL, aliases=["Hosty"]),
        Player(player_id="bot", model="unknown", family="vendor", role="villager", alignment=Alignment.TRUTHFUL),
        Player(player_id="z2", model="z2", family="custom", role="villager", alignment=Alignment.TRUTHFUL),
    ]
    turns = [
        Turn(turn_id=0, speaker_id=NARRATOR, text="Welcome.", phase="intro"),
        Turn(turn_id=1, speaker_id="ai1", text='"I am a simple villager, [laughter] trust me."', phase="day",
             addressees=["all", "host"]),
        Turn(turn_id=2, speaker_id="ai2", text="I think ai1 is lying to us.", phase="day"),
        Turn(turn_id=3, speaker_id="host", text="Human words that must never leak.", phase="day"),
        Turn(turn_id=4, speaker_id="bot", text="Unverified model speaking here.", phase="day"),
        Turn(turn_id=5, speaker_id="z2", text="Custom AI says something useful.", phase="day"),
        Turn(turn_id=6, speaker_id="ai1", text="We kill ai2 tonight, agreed.", phase="night_chat"),
        Turn(turn_id=7, speaker_id="ai2", text="Too short", phase="day"),
        Turn(turn_id=8, speaker_id="ai2", text="I think AI1 is lying to us!", phase="day"),  # duplicate
        Turn(turn_id=9, speaker_id="ai1", text="Good game everyone, well played.", phase="postgame"),
    ]
    return Game(game_id=game_id, players=players, turns=turns, winner="mafia",
                annotation_status="reviewed", text_provenance="captions")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_export_is_clean_and_human_free(tmp_path):
    report = export_clean([game()], tmp_path)
    public = rows(tmp_path / "turns.jsonl")
    assert [r["id"] for r in public] == ["g1:1", "g1:2", "g1:5"]
    first = public[0]
    assert first["text"] == "I am a simple villager, trust me."  # cue and quotes stripped
    assert first["addressees"] == ["all", "human"] and first["label"] == "lying"
    assert first["text_provenance"] == "captions" and first["split"] in ("train", "test")
    assert public[2]["model"] == "z2" and public[2]["model_verified"] is False
    assert [r["id"] for r in rows(tmp_path / "private.jsonl")] == ["g1:6"]
    assert report.dropped["human player"] == 1 and report.dropped["exact model not verified"] == 1
    assert report.dropped["duplicate text"] == 1 and report.dropped["fewer than 3 words"] == 1
    everything = (tmp_path / "turns.jsonl").read_text() + (tmp_path / "private.jsonl").read_text()
    assert "Human words" not in everything


def test_redacted_game_files_contain_no_human_text(tmp_path):
    export_clean([game()], tmp_path)
    text = (tmp_path / "games" / "g1.json").read_text()
    assert "Human words" not in text and "Hosty" not in text
    g = Game.model_validate_json(text)
    assert all(t.speaker_id != "host" for t in g.turns)
    assert g.turns[1].addressees == ["all", "human"]


def test_split_is_exact_stratified_and_stable():
    groups = {f"r{i}": "llm_transcription" for i in range(18)} | {"c1": "captions", "c2": "captions"}
    split = assign_splits(groups, 0.2)
    assert split == assign_splits(dict(reversed(list(groups.items()))), 0.2)  # order-independent
    test = {g for g, s in split.items() if s == "test"}
    assert len(test & {f"r{i}" for i in range(18)}) == 4  # round(18 * 0.2)
    assert len(test & {"c1", "c2"}) == 1  # small groups still get one test game
    assert assign_splits({"only": "captions"}) == {"only": "train"}  # a lone game is never held out


def test_normalise_text():
    assert normalise_text('  "Hello [music]  there"  ') == "Hello there"


def test_build_examples_never_uses_human_speech():
    ex = build_examples([game()], "alignment")
    assert all(e.speaker_id != "host" for e in ex.examples)
    ctx = build_examples([game()], "alignment", context_turns=5).examples
    assert all("Human words" not in e.context for e in ctx)


def test_cli_export(tmp_path, capsys):
    save_game(game(), tmp_path / "ann" / "g1.json")
    assert cli.main(["export", str(tmp_path / "ann"), "--out", str(tmp_path / "clean")]) == 0
    assert "public rows: 3" in capsys.readouterr().out
    assert json.loads((tmp_path / "clean" / "report.json").read_text())["labels"] == {"lying": 1, "truth": 2}
