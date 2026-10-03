"""SRT parsing, including yt-dlp's converted auto-captions."""
from __future__ import annotations

import re

_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{1,3})")
_TAG = re.compile(r"<[^>]+>")


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[dict]:
    """Return ``[{"text", "start", "duration"}]``, the same shape as fetched captions.

    Auto-captions are "rolling": each cue repeats the previous cue's last line before adding
    new words. Lines already shown in the previous cue are dropped so text is not duplicated.
    """
    snippets: list[dict] = []
    previous: set[str] = set()
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.split("\n")
        at = next((i for i, line in enumerate(lines) if _TIME.search(line)), None)
        if at is None:
            continue
        m = _TIME.search(lines[at])
        start, end = _seconds(*m.group(1, 2, 3, 4)), _seconds(*m.group(5, 6, 7, 8))
        body = [b for b in (_TAG.sub("", line).strip() for line in lines[at + 1 :]) if b]
        if not body:
            continue  # yt-dlp inserts near-empty cues between real ones; they must not reset `previous`
        fresh = [line for line in body if line not in previous]
        previous = set(body)
        if fresh:
            snippets.append({"text": " ".join(fresh), "start": start, "duration": max(end - start, 0.0)})
    return snippets
