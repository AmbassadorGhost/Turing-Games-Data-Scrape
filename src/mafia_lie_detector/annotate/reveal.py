"""Find the end-of-game role reveal in a transcript.

Captions never say who held which role, so the ground truth has to come from the reveal (or an
on-screen overlay). This only ranks utterances worth a human look; it assigns nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .segment import Utterance

_ROLES = (
    r"(?:mafia|mafioso|godfather|werewolf|werewolves|wolf|wolves|impostors?|imposters?|traitors?|"
    r"killers?|villagers?|townies?|town|crewmates?|detective|doctor|sheriff|seer)"
)
_DECEIVER_NOUNS = (
    r"(?:mafia|mafioso|godfather|werewolf|werewolves|wolf|wolves|impostors?|imposters?|traitors?|killers?)"
)
_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(rf"\b(?:was|were|is|are)\s+(?:the\s+|a\s+|an\s+)?{_ROLES}\b", re.I), 2),
    # "the mafia were Alice and Bob": past tense only, since "the mafia is probably P3" is ordinary play.
    (re.compile(rf"\b{_DECEIVER_NOUNS}\s+(?:was|were)\b", re.I), 2),
    (re.compile(r"\broles?\s+(?:were|are|reveal(?:ed)?)\b", re.I), 3),
    (re.compile(r"\b(?:reveal(?:ed|s)?|flipped|flips)\b", re.I), 1),
    (re.compile(r"\b(?:game over|wins?|won|victory)\b", re.I), 2),
    (re.compile(r"\b(?:voted out|eliminated|executed)\b", re.I), 1),
]


@dataclass(frozen=True)
class RevealCandidate:
    index: int
    start: float
    score: int
    text: str


def find_reveal_candidates(utterances: Sequence[Utterance], min_score: int = 2) -> list[RevealCandidate]:
    found = []
    for i, u in enumerate(utterances):
        score = sum(weight for pat, weight in _PATTERNS if pat.search(u.text))
        if score >= min_score:
            found.append(RevealCandidate(i, u.start, score, u.text))
    return found


def youtube_link(video_id: str, seconds: float) -> str:
    return f"https://youtu.be/{video_id}?t={int(seconds)}"
