"""Caption fragments -> utterances (candidate turns)."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

_SOUND_CUE = re.compile(r"\[[^\]]{1,40}\]|♪+")
_SPACES = re.compile(r"\s+")


@dataclass
class Utterance:
    text: str
    start: float
    end: float
    speaker_hint: Optional[str] = None


def clean_caption_text(text: str) -> str:
    """Drop ``[Music]``-style cues and note glyphs, unescape entities, collapse whitespace."""
    return _SPACES.sub(" ", _SOUND_CUE.sub(" ", html.unescape(text))).strip()


def _continues(cur: Utterance, start: float, max_gap: float, max_chars: int, min_sentence: int) -> bool:
    if start - cur.end > max_gap or len(cur.text) >= max_chars:
        return False
    # Hand-made captions are punctuated: close a sentence once it is long enough to be a turn.
    return not (cur.text.endswith((".", "?", "!")) and len(cur.text) >= min_sentence)


def snippets_to_utterances(
    snippets: Sequence[dict],
    *,
    max_gap: float = 1.2,
    max_chars: int = 400,
    min_sentence_chars: int = 80,
) -> list[Utterance]:
    """Merge caption snippets (``text``/``start``/``duration``) into utterances.

    A new utterance starts at a ``>>`` speaker-change marker, after a pause longer than
    ``max_gap`` seconds, once ``max_chars`` is reached, or after a sentence end in punctuated
    captions. Caption durations overlap the next snippet, so each end is clipped to it.
    """
    ordered = sorted(snippets, key=lambda s: float(s["start"]))
    out: list[Utterance] = []
    cur: Optional[Utterance] = None
    for i, snip in enumerate(ordered):
        start = float(snip["start"])
        end = start + float(snip.get("duration", 0.0))
        if i + 1 < len(ordered):
            end = min(end, float(ordered[i + 1]["start"]))
        for j, piece in enumerate(html.unescape(snip["text"]).split(">>")):
            force_new = j > 0  # text after a ">>" belongs to a new speaker
            text = clean_caption_text(piece)
            if not text:
                if force_new:
                    cur = None
                continue
            if cur is not None and not force_new and _continues(cur, start, max_gap, max_chars, min_sentence_chars):
                cur.text += " " + text
                cur.end = max(cur.end, end)
            else:
                cur = Utterance(text, start, end)
                out.append(cur)
    return out


def assign_speakers_by_prefix(utterances: Iterable[Utterance], names: Sequence[str]) -> list[Utterance]:
    """Handle ``Name: text`` lines when the captions (or a script) label speakers."""
    cleaned = sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True)
    if not cleaned:
        return list(utterances)
    pattern = re.compile(r"^\s*(" + "|".join(re.escape(n) for n in cleaned) + r")\s*[:\-–]\s*", re.IGNORECASE)
    lookup = {n.lower(): n for n in cleaned}
    result = []
    for u in utterances:
        m = pattern.match(u.text)
        if m:
            u.speaker_hint = lookup[m.group(1).lower()]
            u.text = u.text[m.end():].strip()
        result.append(u)
    return result
