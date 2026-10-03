import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mafia_lie_detector import cli
from mafia_lie_detector.ingest import extract, srt, youtube

ROLLING = """1
00:00:01,000 --> 00:00:03,000
hello there

2
00:00:03,000 --> 00:00:03,010


3
00:00:03,010 --> 00:00:05,000
hello there
how are <c>you</c> today

4
00:00:05,000 --> 00:00:07,500
how are you today
I am <00:00:06.000>fine
"""


def test_parse_srt_drops_rolling_duplicates_tags_and_empty_cues():
    got = srt.parse_srt(ROLLING)
    assert [g["text"] for g in got] == ["hello there", "how are you today", "I am fine"]
    assert got[0] == {"text": "hello there", "start": 1.0, "duration": 2.0}
    assert got[2]["start"] == 5.0


def test_parse_srt_handles_crlf_hours_and_junk():
    text = "1\r\n01:02:03,500 --> 01:02:05,000\r\nlate line\r\n\r\nnot a cue\r\n"
    assert srt.parse_srt(text) == [{"text": "late line", "start": 3723.5, "duration": 1.5}]


def test_load_captions_reads_srt_and_names_video_by_folder(tmp_path):
    d = tmp_path / "abcDEF12345"
    d.mkdir()
    (d / "captions.en.srt").write_text(ROLLING)
    data = youtube.load_captions(d / "captions.en.srt")
    assert data["video_id"] == "abcDEF12345" and data["source"] == "srt" and len(data["snippets"]) == 3
    (tmp_path / "xyz98765432.en.srt").write_text(ROLLING)
    assert youtube.load_captions(tmp_path / "xyz98765432.en.srt")["video_id"] == "xyz98765432"


@pytest.mark.parametrize("ref", [
    "x61Dcbl1SFI",
    "https://www.youtube.com/watch?v=x61Dcbl1SFI",
    "https://www.youtube.com/watch?v=x61Dcbl1SFI&t=90s&list=PL1",
    "https://youtu.be/x61Dcbl1SFI?si=abc",
    "https://www.youtube.com/live/x61Dcbl1SFI",
])
def test_video_id_from_accepts_common_forms(ref):
    assert extract.video_id_from(ref) == "x61Dcbl1SFI"


def test_video_id_from_rejects_garbage():
    with pytest.raises(ValueError):
        extract.video_id_from("not a video")


class FakeTools:
    """Stands in for yt-dlp / ffprobe / ffmpeg and creates the files they would."""

    def __init__(self, out_root: Path, *, fail_video=False, duration="300.0", captions=True):
        self.root, self.fail_video, self.duration, self.captions, self.calls = out_root, fail_video, duration, captions, []

    def __call__(self, cmd, **_):
        self.calls.append(cmd)
        ok = SimpleNamespace(returncode=0, stdout="", stderr="")
        tool = cmd[0]
        if tool == "yt-dlp" and "--print" in cmd:
            return SimpleNamespace(returncode=0, stdout="Can 1 Human Trick 9 AIs?\n", stderr="")
        if tool == "yt-dlp" and "--write-subs" in cmd:
            if self.captions:
                out = Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "en.srt"))
                out.write_text(ROLLING)
            return ok
        if tool == "yt-dlp":
            if self.fail_video:
                return SimpleNamespace(returncode=1, stdout="", stderr="ERROR: blocked")
            Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "mp4")).write_bytes(b"video")
            return ok
        if tool == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=self.duration + "\n", stderr="")
        if tool == "ffmpeg":
            start = float(cmd[cmd.index("-ss") + 1])
            n = 3 if start == 0 else 2
            pattern = cmd[-1]
            for i in range(1, n + 1):
                Path(pattern % i).write_bytes(b"jpg")
            return ok
        raise AssertionError(f"unexpected command {cmd}")


def test_extract_video_writes_layout_index_and_cleans_up(tmp_path):
    tools = FakeTools(tmp_path)
    meta = extract.extract_video("https://youtu.be/x61Dcbl1SFI", tmp_path, frame_every=10.0, tail_seconds=120.0,
                                 tail_every=4.0, runner=tools)
    d = tmp_path / "x61Dcbl1SFI"
    assert meta["title"] == "Can 1 Human Trick 9 AIs?" and meta["captions"] == ["captions.en.srt"]
    assert not (d / "video.mp4").exists()  # the large file is not kept by default
    frames = json.loads((d / "frames.json").read_text())
    assert [(f["file"], f["t"]) for f in frames] == [
        ("frames/f_00001.jpg", 0.0), ("frames/f_00002.jpg", 10.0), ("frames/f_00003.jpg", 20.0),
        ("reveal_frames/r_0001.jpg", 180.0), ("reveal_frames/r_0002.jpg", 184.0),  # 300s video, last 120s
    ]
    assert json.loads((d / "meta.json").read_text())["n_reveal_frames"] == 2
    ffmpeg_calls = [c for c in tools.calls if c[0] == "ffmpeg"]
    assert "fps=1/10.0" in ffmpeg_calls[0] and "fps=1/4.0" in ffmpeg_calls[1]


def test_extract_video_keep_video_and_short_video(tmp_path):
    tools = FakeTools(tmp_path, duration="90.0")
    extract.extract_video("x61Dcbl1SFI", tmp_path, keep_video=True, tail_seconds=600.0, runner=tools)
    assert (tmp_path / "x61Dcbl1SFI" / "video.mp4").exists()
    tail_call = [c for c in tools.calls if c[0] == "ffmpeg"][1]
    assert tail_call[tail_call.index("-ss") + 1] == "0.00"  # a tail longer than the video never seeks negative


def test_extract_video_reports_download_failure(tmp_path):
    with pytest.raises(RuntimeError, match="video download failed"):
        extract.extract_video("x61Dcbl1SFI", tmp_path, runner=FakeTools(tmp_path, fail_video=True))


def test_cli_extract_prints_summary_and_flags_missing_captions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(extract.subprocess, "run", FakeTools(tmp_path, captions=False))
    assert cli.main(["extract", "x61Dcbl1SFI", "--out", str(tmp_path)]) == 0
    assert "NO CAPTIONS" in capsys.readouterr().out
    assert cli.main(["extract", "bogus", "--out", str(tmp_path)]) == 1
