"""Data model for annotated social-deduction games.

Three label levels, cheapest to most useful:

* ``Player.alignment`` - which side the speaker is on (known from the role reveal). A weak
  proxy for deception: deceivers also say true things, and town players sometimes say false ones.
* ``Turn.deceptive``   - whether the turn contains at least one claim the speaker knew was false.
* ``Turn.claims``      - the individual claims, each tagged truthful / untruthful relative to the
  speaker's hidden knowledge. ``Turn.deceptive`` is derived from these when they are present.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

NARRATOR = "__narrator__"
"""Speaker id for host / voice-over lines. Never a player and never a training example."""

UNKNOWN_SPEAKER = "__unknown__"
"""Speaker id for turns whose speaker could not be established. Kept for audit, never trained on."""

HUMAN = "human"  # Player.model for a human seat; never sorted into a model folder
UNKNOWN_MODEL = "unknown"  # Player.model when only the family (or nothing) could be verified

# Roles that usually belong to the lying side. Only a hint for tooling: variants differ, so
# alignment is always stated explicitly on each Player.
DECEIVER_ROLE_HINTS = frozenset(
    {"mafia", "mafioso", "godfather", "werewolf", "wolf", "impostor", "imposter", "traitor", "killer"}
)


class Alignment(str, Enum):
    DECEIVER = "deceiver"  # mafia / werewolf / impostor: must lie to survive
    TRUTHFUL = "truthful"  # town / villager / crewmate: no strategic reason to lie


class ClaimKind(str, Enum):
    ROLE_CLAIM = "role_claim"  # "I'm the detective"
    KNOWLEDGE_CLAIM = "knowledge_claim"  # "I checked B last night and they're clean"
    ALIBI = "alibi"  # "I was asleep", "I was with C"
    ACCUSATION = "accusation"  # "A is mafia"
    VOTE_INTENT = "vote_intent"  # "I'm voting B"
    OTHER = "other"


class Claim(BaseModel):
    text: str
    kind: ClaimKind = ClaimKind.OTHER
    truthful: bool  # relative to the speaker's own hidden knowledge, not to the audience's


class Player(BaseModel):
    player_id: str
    model: str  # exact underlying LLM as shown on screen, "human", or "unknown"
    role: str
    alignment: Alignment
    family: Optional[str] = None  # vendor / model family, e.g. "anthropic"
    evidence: str = ""  # how model and role were identified (frame, timestamp), for spot-checking
    aliases: list[str] = Field(default_factory=list)  # other names used in speech ("Bot 3", "Alice")
    eliminated_round: Optional[int] = None

    @field_validator("role")
    @classmethod
    def _normalise_role(cls, v: str) -> str:
        return v.strip().lower()


class Turn(BaseModel):
    turn_id: int
    speaker_id: Optional[str] = None  # None only while a game is still a draft
    speaker_hint: Optional[str] = None  # raw "Name:" label from the captions, to resolve by hand
    text: str
    start: Optional[float] = None  # seconds into the video
    end: Optional[float] = None
    phase: Optional[str] = None  # intro | day | vote | night | reveal | postgame | ...
    round: Optional[int] = None
    addressees: list[str] = Field(default_factory=list)  # player ids spoken to, or "all" / "self"
    claims: list[Claim] = Field(default_factory=list)
    deceptive: Optional[bool] = None

    @model_validator(mode="after")
    def _derive_deceptive(self) -> "Turn":
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError(f"turn {self.turn_id}: end < start")
        if self.claims:
            derived = any(not c.truthful for c in self.claims)
            if self.deceptive is None:
                self.deceptive = derived
            elif self.deceptive != derived:
                raise ValueError(
                    f"turn {self.turn_id}: deceptive={self.deceptive} contradicts its claims"
                )
        return self


class Game(BaseModel):
    game_id: str
    source: str = ""  # e.g. "youtube:<video_id>"
    variant: str = "mafia"
    players: list[Player] = Field(default_factory=list)
    turns: list[Turn]
    winner: Optional[str] = None
    annotation_status: Literal["draft", "reviewed"] = "draft"
    notes: str = ""
    events: list[str] = Field(default_factory=list)  # timeline, e.g. "N1: mafia kill X; doctor saves Y"
    # Where the dialogue text came from, e.g. "captions" (speech-to-text of the video) or
    # "llm_reconstruction" (another model rewrote it). Wording-level features are only trustworthy
    # for caption-grade text.
    text_provenance: str = "unknown"

    @model_validator(mode="after")
    def _check_integrity(self) -> "Game":
        ids = [p.player_id for p in self.players]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate player_id")
        if reserved := {NARRATOR, UNKNOWN_SPEAKER} & set(ids):
            raise ValueError(f"{sorted(reserved)[0]!r} is reserved")
        turn_ids = [t.turn_id for t in self.turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("duplicate turn_id")
        known = set(ids) | {NARRATOR, UNKNOWN_SPEAKER}
        for t in self.turns:
            if t.speaker_id is not None and t.speaker_id not in known:
                raise ValueError(f"turn {t.turn_id}: unknown speaker {t.speaker_id!r}")
        return self

    def player(self, player_id: str) -> Optional[Player]:
        return next((p for p in self.players if p.player_id == player_id), None)

    def unattributed_fraction(self) -> float:
        """Share of non-narrator turns whose speaker could not be established."""
        spoken = [t for t in self.turns if t.speaker_id != NARRATOR]
        return sum(t.speaker_id == UNKNOWN_SPEAKER for t in spoken) / len(spoken) if spoken else 0.0

    def completeness_problems(self) -> list[str]:
        """Things that must be fixed before this game can feed a training set."""
        problems: list[str] = []
        if not self.players:
            problems.append("no players listed")
        if not any(p.alignment is Alignment.DECEIVER for p in self.players):
            problems.append("no deceiver among players")
        missing = [t.turn_id for t in self.turns if t.speaker_id is None]
        if missing:
            shown = ", ".join(map(str, missing[:8])) + (" ..." if len(missing) > 8 else "")
            problems.append(f"{len(missing)} turn(s) without a speaker (turn_id {shown})")
        return problems


def suggest_alignment(role: str) -> Alignment:
    """Best-guess alignment for a role name; the annotator confirms it."""
    return Alignment.DECEIVER if role.strip().lower() in DECEIVER_ROLE_HINTS else Alignment.TRUTHFUL


def save_game(game: Game, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Nulls are kept on purpose: in a draft they show the annotator which fields are left to fill.
    path.write_text(game.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_game(path: str | Path) -> Game:
    return Game.model_validate_json(Path(path).read_text(encoding="utf-8"))


def find_game_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories (searched for ``*.json``) into a sorted list of files."""
    found: list[Path] = []
    for p in map(Path, paths):
        found.extend(sorted(p.rglob("*.json")) if p.is_dir() else [p])
    return found


def iter_games(paths: Iterable[str | Path]) -> Iterator[Game]:
    for f in find_game_files(paths):
        yield load_game(f)
