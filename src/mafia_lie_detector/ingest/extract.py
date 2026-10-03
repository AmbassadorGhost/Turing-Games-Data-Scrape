"""Local media extraction: captions plus sampled frames, for annotating who is who.

Run this on a machine that can reach YouTube. Needs ``yt-dlp`` and ``ffmpeg`` on PATH. Captions
give the words; frames give what captions cannot: on-screen name tags, avatars, model labels and
the role reveal. Frames are sampled across the whole video (speaker/model identity is shown
throughout, not only at the end) plus a denser pass over the final minutes (the reveal).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

_ID_IN_URL = re.compile(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([\w-]{11})")
_BARE_ID = re.compile(r"[\w-]{11}")


def video_id_from(ref: str) -> str:
    """Accept a bare id or any common YouTube URL form."""
    ref = ref.strip()
    m = _ID_IN_URL.search(ref)
    if m:
        return m.group(1)
    if _BARE_ID.fullmatch(ref):
        return ref
    raise ValueError(f"cannot find a YouTube video id in {ref!r}")


def _sample_frames(
    runner: Callable[..., Any], video: Path, pattern: Path, *, every: float, start: float
) -> list[dict]:
    pattern.parent.mkdir(parents=True, exist_ok=True)
    proc = runner(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(video),
         "-vf", f"fps=1/{every}", "-q:v", "3", str(pattern)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {(proc.stderr or '').strip()[-200:]}")
    files = sorted(pattern.parent.glob(pattern.name.replace("%05d", "*").replace("%04d", "*")))
    # fps=1/N emits the first frame at the start offset, then one every N seconds.
    return [{"file": f"{f.parent.name}/{f.name}", "t": round(start + i * every, 2)} for i, f in enumerate(files)]


def extract_video(
    ref: str,
    out_root: str | Path = "data/raw",
    *,
    frame_every: float = 10.0,
    tail_seconds: float = 180.0,
    tail_every: float = 3.0,
    height: int = 480,
    keep_video: bool = False,
    runner: Optional[Callable[..., Any]] = None,
) -> dict:
    """Write ``<out_root>/<id>/`` with ``captions.en.srt``, ``frames/``, ``reveal_frames/``,
    ``frames.json`` (file -> seconds) and ``meta.json``; returns the metadata."""
    runner = runner or subprocess.run  # resolved per call so tests can patch subprocess.run
    vid = video_id_from(ref)
    d = Path(out_root) / vid
    d.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={vid}"

    title_proc = runner(["yt-dlp", "--skip-download", "--no-warnings", "--print", "%(title)s", url],
                        capture_output=True, text=True)
    title = (title_proc.stdout or "").strip().splitlines()[0] if title_proc.returncode == 0 and (title_proc.stdout or "").strip() else ""

    # Manual captions are used when they exist; auto-captions only for languages without them.
    runner(["yt-dlp", "--skip-download", "--no-warnings", "--write-subs", "--write-auto-subs",
            "--sub-langs", "en", "--convert-subs", "srt", "-o", str(d / "captions.%(ext)s"), url],
           capture_output=True, text=True)
    captions = sorted(d.glob("captions*.srt"))

    video = d / "video.mp4"
    proc = runner(
        ["yt-dlp", "--no-warnings", "-f", f"bv*[height<={height}]+ba/b[height<={height}]/worst",
         "--merge-output-format", "mp4", "-o", str(d / "video.%(ext)s"), url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not video.exists():
        raise RuntimeError(f"video download failed for {vid}: {(proc.stderr or '').strip()[-200:]}")
    probe = runner(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                   capture_output=True, text=True)
    try:
        duration = float((probe.stdout or "").strip())
    except ValueError:
        raise RuntimeError(f"could not read the duration of {video}") from None

    frames = _sample_frames(runner, video, d / "frames" / "f_%05d.jpg", every=frame_every, start=0.0)
    tail = _sample_frames(runner, video, d / "reveal_frames" / "r_%04d.jpg",
                          every=tail_every, start=max(0.0, duration - tail_seconds))
    (d / "frames.json").write_text(json.dumps(frames + tail, indent=1) + "\n", encoding="utf-8")
    if not keep_video:
        video.unlink()

    meta = {
        "video_id": vid, "url": url, "title": title, "duration": duration,
        "captions": [c.name for c in captions], "n_frames": len(frames), "n_reveal_frames": len(tail),
        "frame_every": frame_every, "tail_seconds": tail_seconds, "tail_every": tail_every,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta
