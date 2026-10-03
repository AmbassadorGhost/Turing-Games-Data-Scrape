import json
from types import SimpleNamespace

import pytest

from mafia_lie_detector.ingest import audio, youtube


# ------------------------------------------------------------------ channel / listing

def test_channel_tab_urls():
    expect = [("videos", "https://www.youtube.com/@turing_games/videos"),
              ("streams", "https://www.youtube.com/@turing_games/streams")]
    assert youtube.channel_tab_urls("https://www.youtube.com/@turing_games") == expect
    assert youtube.channel_tab_urls("@turing_games/") == expect
    assert youtube.channel_tab_urls("turing_games") == expect
    # A URL that already names a tab is used as given.
    assert youtube.channel_tab_urls("https://www.youtube.com/@turing_games/videos") == expect[:1]


def _proc(code=0, out="", err=""):
    return SimpleNamespace(returncode=code, stdout=out, stderr=err)


def test_list_videos_dedupes_and_downgrades_missing_tab_to_warning():
    def runner(cmd, **_):
        url = cmd[-1]
        if url.endswith("/videos"):
            return _proc(out="aaa\tFirst game\nbbb\tSecond\tgame\n")
        return _proc(code=1, err="ERROR: This channel does not have a streams tab")

    videos, warnings = youtube.list_videos("@turing_games", runner=runner)
    assert [(v.id, v.tab) for v in videos] == [("aaa", "videos"), ("bbb", "videos")]
    assert videos[0].title == "First game"
    assert videos[1].title == "Second\tgame"  # only the first tab separates id from title
    assert len(warnings) == 1 and warnings[0].startswith("streams:")


def test_video_list_roundtrip(tmp_path):
    vids = [youtube.VideoRef("aaa", "A\ttitle", "videos"), youtube.VideoRef("bbb", "B", "streams")]
    path = tmp_path / "ids.tsv"
    youtube.write_video_list(vids, path)
    path.write_text("# comment\n\n" + path.read_text())
    assert youtube.read_video_ids(path) == ["aaa", "bbb"]


# ------------------------------------------------------------------------ captions

class FakeFetched:
    language_code = "en"
    is_generated = True

    def __init__(self, snippets):
        self._snippets = snippets

    def to_raw_data(self):
        return self._snippets


class IpBlocked(Exception):  # matched by class name, like the real youtube-transcript-api error
    pass


class TranscriptsDisabled(Exception):
    pass


class FakeApi:
    def __init__(self, script):
        self.script, self.calls = script, []

    def fetch(self, video_id, languages):
        self.calls.append((video_id, tuple(languages)))
        outcome = self.script[video_id]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeFetched(outcome)


SNIPPETS = [{"text": "hello", "start": 0.0, "duration": 1.0}]


def test_fetch_captions_uses_v1_fetch_interface():
    api = FakeApi({"v1": SNIPPETS})
    data = youtube.fetch_captions("v1", ["en", "en-US"], api=api)
    assert api.calls == [("v1", ("en", "en-US"))]
    assert data["snippets"] == SNIPPETS and data["is_generated"] is True and data["source"] == "youtube_captions"


def test_fetch_many_saves_skips_and_records_failures(tmp_path):
    api = FakeApi({"ok": SNIPPETS, "off": TranscriptsDisabled("Subtitles are disabled"), "old": SNIPPETS})
    (tmp_path / "old.json").write_text("{}")
    sleeps = []
    report = youtube.fetch_many(["ok", "off", "old", "ok"], tmp_path, api=api, sleeper=sleeps.append, sleep=0.5)
    assert report.saved == ["ok"] and report.skipped == ["old"]
    assert report.failed == {"off": "TranscriptsDisabled: Subtitles are disabled"}
    assert json.loads((tmp_path / "ok.json").read_text())["snippets"] == SNIPPETS
    assert [c[0] for c in api.calls] == ["ok", "off"]  # duplicate id fetched once, existing file untouched
    assert sleeps == [0.5, 0.5]
    youtube.write_failures(report, tmp_path / "failures.tsv")
    assert (tmp_path / "failures.tsv").read_text().startswith("off\tTranscriptsDisabled")


def test_fetch_many_stops_when_blocked(tmp_path):
    api = FakeApi({"a": SNIPPETS, "b": IpBlocked("blocked"), "c": SNIPPETS, "d": SNIPPETS})
    report = youtube.fetch_many(["a", "b", "c", "d"], tmp_path, api=api, sleeper=lambda _: None)
    assert report.stopped == "blocked" and report.saved == ["a"]
    assert report.not_attempted == ["c", "d"] and [c[0] for c in api.calls] == ["a", "b"]


def test_fetch_many_stops_on_connectivity_errors_and_does_not_suggest_whisper(tmp_path):
    class ProxyError(Exception):
        pass

    api = FakeApi({"a": ProxyError("Tunnel connection failed: 403"), "b": SNIPPETS})
    report = youtube.fetch_many(["a", "b"], tmp_path, api=api, sleeper=lambda _: None)
    assert report.stopped == "network" and report.not_attempted == ["b"]
    assert report.no_captions == []  # a network failure says nothing about captions


def test_no_captions_failures_are_flagged_for_transcription(tmp_path):
    api = FakeApi({"a": TranscriptsDisabled("off"), "b": SNIPPETS})
    report = youtube.fetch_many(["a", "b"], tmp_path, api=api, sleeper=lambda _: None)
    assert report.stopped is None and report.no_captions == ["a"] and report.saved == ["b"]


# --------------------------------------------------------------------------- audio

def test_download_audio(tmp_path):
    def runner(cmd, **_):
        (tmp_path / "vid.wav").write_bytes(b"RIFF")
        return _proc()

    assert audio.download_audio("vid", tmp_path, runner=runner) == tmp_path / "vid.wav"
    # Second call reuses the file without invoking yt-dlp.
    assert audio.download_audio("vid", tmp_path, runner=lambda *a, **k: pytest.fail("re-downloaded")).exists()


def test_download_audio_failure_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg missing"):
        audio.download_audio("vid", tmp_path, runner=lambda *a, **k: _proc(1, err="ffmpeg missing"))


def test_transcribe_matches_caption_shape():
    seg = lambda t, s, e: SimpleNamespace(text=t, start=s, end=e)  # noqa: E731

    class FakeWhisper:
        def transcribe(self, path, language, vad_filter):
            assert language == "en" and vad_filter
            return iter([seg(" hi there ", 0.0, 1.5), seg("   ", 2.0, 2.5), seg("bye", 3.0, 4.0)]), None

    data = audio.transcribe("x.wav", video_id="v", model=FakeWhisper())
    assert data["source"] == "whisper"
    assert data["snippets"] == [
        {"text": "hi there", "start": 0.0, "duration": 1.5},
        {"text": "bye", "start": 3.0, "duration": 1.0},
    ]
