import json

from mafia_lie_detector import cli
from mafia_lie_detector.ingest import youtube
from mafia_lie_detector.schema import Game, Turn, load_game, save_game
from mafia_lie_detector.synthetic import make_games


def run(*argv):
    return cli.main([str(a) for a in argv])


def test_end_to_end_validate_build_evaluate(tmp_path, capsys):
    ann = tmp_path / "annotations"
    for g in make_games(30, seed=2):
        save_game(g, ann / f"{g.game_id}.json")
    assert run("validate", ann) == 0
    out = tmp_path / "turns.jsonl"
    assert run("build-dataset", ann, "-o", out, "--scrub") == 0
    assert "30 games" in capsys.readouterr().out
    assert run("evaluate", out, "--json", "--folds", 3) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["split"] == "group_kfold" and report["pooled"]["roc_auc"] > 0.6
    assert run("evaluate", out, "--split", "leave_model_out", "--min-test", 50) == 0


def test_validate_flags_bad_and_incomplete_reviewed_files(tmp_path, capsys, small_game):
    (tmp_path / "broken.json").write_text('{"game_id": "x", "turns": [{"turn_id": 0, "text": "a", "speaker_id": "nobody"}]}')
    assert run("validate", tmp_path) == 1
    assert "INVALID" in capsys.readouterr().err
    (tmp_path / "broken.json").unlink()
    assert run("validate", tmp_path) == 2  # no files left to check
    save_game(small_game, tmp_path / "ok.json")
    assert run("validate", tmp_path) == 0
    # A game marked reviewed while still missing players/speakers is an error, not a draft.
    save_game(Game(game_id="inc", turns=[Turn(turn_id=0, text="a")], annotation_status="reviewed"), tmp_path / "inc.json")
    assert run("validate", tmp_path) == 1
    assert "no players listed" in capsys.readouterr().out


def test_build_dataset_reports_skipped_drafts(tmp_path, capsys):
    save_game(Game(game_id="draft", turns=[Turn(turn_id=0, text="hi")]), tmp_path / "draft.json")
    assert run("build-dataset", tmp_path, "-o", tmp_path / "out.jsonl") == 1  # nothing usable
    assert "skipped draft" in capsys.readouterr().err


def test_draft_and_reveals_from_captions(tmp_path, capsys):
    captions = {
        "video_id": "vid9", "source": "youtube_captions",
        "snippets": [
            {"text": "Alice: I am just a villager", "start": 0.0, "duration": 2.0},
            {"text": "the mafia were Alice and Bob", "start": 60.0, "duration": 3.0},
        ],
    }
    cap = tmp_path / "vid9.json"
    cap.write_text(json.dumps(captions))
    out = tmp_path / "ann" / "yt-vid9.json"
    assert run("draft", cap, "-o", out, "--names", "Alice", "--title", "Test") == 0
    game = load_game(out)
    assert game.annotation_status == "draft" and game.turns[0].speaker_hint == "Alice"
    # A second run must not clobber (possibly hand-edited) work.
    assert run("draft", cap, "-o", out) == 1
    assert run("draft", cap, "-o", out, "--force") == 0
    capsys.readouterr()
    assert run("reveals", cap) == 0
    assert "https://youtu.be/vid9?t=60" in capsys.readouterr().out


def test_fetch_captions_needs_ids(capsys):
    assert run("fetch-captions") == 2


def test_fetch_captions_exit_codes(tmp_path, monkeypatch, capsys):
    def fake_fetch_many(ids, out, languages, **_):
        if ids == ["down"]:
            return youtube.FetchReport(failed={"down": "ProxyError: refused"}, stopped="network")
        return youtube.FetchReport(saved=["ok"], failed={"nocap": "TranscriptsDisabled: off"})

    monkeypatch.setattr(youtube, "fetch_many", fake_fetch_many)
    assert run("fetch-captions", "--id", "down", "--out", tmp_path) == 1
    assert "cannot connect to YouTube" in capsys.readouterr().err
    assert run("fetch-captions", "--id", "ok", "--out", tmp_path) == 0  # one video without captions is normal
    assert "mld transcribe --id nocap" in capsys.readouterr().out


def test_suggest_needs_a_model(tmp_path, monkeypatch, capsys, small_game):
    monkeypatch.delenv("MLD_ANNOTATION_MODEL", raising=False)
    save_game(small_game, tmp_path / "g.json")
    assert run("suggest", tmp_path / "g.json") == 2
    assert "--model" in capsys.readouterr().err


def test_demo_runs(capsys):
    assert run("demo", "--games", 24) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out and "POOLED" in out and "majority baseline" in out
