"""Speech-to-text fallback for videos without usable captions.

Needs ``yt-dlp`` + ``ffmpeg`` (download) and the ``whisper`` extra (faster-whisper). The output
has the same shape as fetched captions, so the rest of the pipeline treats both identically.
"""
from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any, Callable


def download_audio(
    video_id: str,
    out_dir: str | Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{video_id}.wav"
    if target.exists():
        return target
    proc = runner(
        [
            "yt-dlp", "-x", "--audio-format", "wav", "--no-warnings",
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not target.exists():
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"audio download failed for {video_id}: {tail[0]}")
    return target


def transcribe(
    audio_path: str | Path,
    *,
    video_id: str,
    model: Any = None,
    model_size: str = "small",
    language: str = "en",
) -> dict:
    """Transcribe with faster-whisper (or any object exposing the same ``transcribe`` method)."""
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(str(audio_path), language=language, vad_filter=True)
    snippets = [
        {"text": s.text.strip(), "start": float(s.start), "duration": float(s.end - s.start)}
        for s in segments
        if s.text.strip()
    ]
    return {
        "video_id": video_id,
        "source": "whisper",
        "language_code": language,
        "is_generated": True,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "snippets": snippets,
    }
