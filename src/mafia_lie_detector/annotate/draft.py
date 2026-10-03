"""Build a draft Game from captions for a human to complete."""
from __future__ import annotations

from typing import Optional, Sequence

from ..schema import Game, Turn
from .segment import Utterance, assign_speakers_by_prefix, snippets_to_utterances


def utterances_from_captions(captions: dict, names: Optional[Sequence[str]] = None) -> list[Utterance]:
    utterances = snippets_to_utterances(captions["snippets"])
    return assign_speakers_by_prefix(utterances, names) if names else utterances


def apply_roster(draft: Game, roster: Game) -> Game:
    """Copy an answer key (players, winner, events) onto a caption draft. Turns stay as drafted."""
    return Game(
        game_id=roster.game_id,
        source=draft.source,
        variant=roster.variant,
        players=[p.model_copy(deep=True) for p in roster.players],
        turns=draft.turns,
        winner=roster.winner,
        annotation_status="draft",
        notes=" | ".join(n for n in (roster.notes, draft.notes) if n),
        events=list(roster.events),
    )


def make_draft(
    video_id: str,
    utterances: Sequence[Utterance],
    *,
    title: str = "",
    source_kind: str = "youtube_captions",
) -> Game:
    """Players stay empty (roles come from the reveal) and speakers null; any ``Name:`` label
    found in the captions is kept as ``speaker_hint`` for the annotator to resolve."""
    turns = [
        Turn(
            turn_id=i,
            speaker_id=None,
            speaker_hint=u.speaker_hint,
            text=u.text,
            start=round(u.start, 2),
            end=round(max(u.end, u.start), 2),
        )
        for i, u in enumerate(utterances)
    ]
    return Game(
        game_id=f"yt-{video_id}",
        source=f"youtube:{video_id}",
        players=[],
        turns=turns,
        annotation_status="draft",
        notes=f"{title} [{source_kind}]".strip(),
    )
