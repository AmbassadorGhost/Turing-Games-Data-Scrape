"""Turn annotated games into flat, leakage-aware training examples."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from .schema import HUMAN, NARRATOR, UNKNOWN_SPEAKER, Alignment, Game

TARGETS = ("turn_deceptive", "alignment")

# Phases that talk about the outcome of the game. Their text names roles ("X was the mafia"),
# so training on them teaches the model to read the answer key, not to detect lies.
OUTCOME_PHASES = frozenset({"intro", "reveal", "postgame", "outro", "preview"})
# Private channels: internal thoughts, evil-team night chat, night actions, dead players' chat.
# A mafia player planning with teammates is not lying to anyone, so these are kept in game files
# but left out of training data unless asked for.
PRIVATE_PHASES = frozenset({"thought", "night_chat", "night_action", "ghost"})
DEFAULT_EXCLUDED_PHASES = OUTCOME_PHASES | PRIVATE_PHASES

PLAYER_TOKEN = "<PLAYER>"


@dataclass(frozen=True)
class Example:
    game_id: str
    turn_id: int
    speaker_id: str
    speaker_model: str
    role: str
    alignment: str
    text: str
    context: str
    label: int  # 1 = deceptive turn / deceiver, depending on the target

    @property
    def group(self) -> str:
        return self.game_id


@dataclass
class BuildResult:
    examples: list[Example]
    skipped_games: dict[str, list[str]]  # game_id -> reasons


def scrub_names(text: str, names: Iterable[str]) -> str:
    """Replace player ids / aliases with a neutral token.

    Who is accused or defended is game-specific, and a name can also encode which LLM is behind a
    seat, so leaving names in lets a model memorise seats instead of learning deception cues.
    """
    cleaned = sorted({n.strip() for n in names if len(n.strip()) >= 2}, key=len, reverse=True)
    if not cleaned:
        return text
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(n) for n in cleaned) + r")(?!\w)", re.IGNORECASE
    )
    return pattern.sub(PLAYER_TOKEN, text)


def build_examples(
    games: Iterable[Game],
    target: str = "turn_deceptive",
    *,
    exclude_phases: Iterable[str] = DEFAULT_EXCLUDED_PHASES,
    context_turns: int = 0,
    scrub: bool = False,
    require_reviewed: bool = True,
) -> BuildResult:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    excluded = {p.lower() for p in exclude_phases}
    examples: list[Example] = []
    skipped: dict[str, list[str]] = {}

    for game in games:
        reasons = list(game.completeness_problems())
        if require_reviewed and game.annotation_status != "reviewed":
            reasons.append("annotation_status is not 'reviewed'")
        if reasons:
            skipped[game.game_id] = reasons
            continue

        names: list[str] = []
        for p in game.players:
            names += [p.player_id, *p.aliases]

        def clean(s: str) -> str:
            return scrub_names(s, names) if scrub else s

        # Human players are never examples, and their words never appear as context either.
        humans = {p.player_id for p in game.players if p.model.strip().lower() == HUMAN}
        # Context is drawn only from turns that are themselves allowed in the dataset, so a
        # reveal-phase line can never leak in through the context window.
        visible = [
            t
            for t in game.turns
            if t.speaker_id not in (None, NARRATOR, UNKNOWN_SPEAKER)
            and t.speaker_id not in humans
            and (t.phase or "").lower() not in excluded
            and t.text.strip()
        ]
        for i, turn in enumerate(visible):
            player = game.player(turn.speaker_id)
            if player is None:
                continue
            if target == "turn_deceptive":
                if turn.deceptive is None:
                    continue
                label = int(turn.deceptive)
            else:
                label = int(player.alignment is Alignment.DECEIVER)
            ctx = "\n".join(
                f"{clean(t.speaker_id)}: {clean(t.text)}"
                for t in visible[max(0, i - context_turns) : i]
            ) if context_turns else ""
            examples.append(
                Example(
                    game_id=game.game_id,
                    turn_id=turn.turn_id,
                    speaker_id=turn.speaker_id,
                    speaker_model=player.model,
                    role=player.role,
                    alignment=player.alignment.value,
                    text=clean(turn.text),
                    context=ctx,
                    label=label,
                )
            )
    return BuildResult(examples, skipped)


def write_examples(examples: Iterable[Example], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_examples(path: str | Path) -> list[Example]:
    with Path(path).open(encoding="utf-8") as f:
        return [Example(**json.loads(line)) for line in f if line.strip()]


def class_balance(examples: list[Example]) -> Optional[float]:
    return sum(e.label for e in examples) / len(examples) if examples else None
