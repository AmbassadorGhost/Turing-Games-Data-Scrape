"""Sort annotated games into ``dataset/<family>/<exact-model>/<lying|truth>/``.

One file per player per game holds that player's turns. Folders depend only on who the speaker
is and which side their role lies on; who won is recorded in each line's metadata, not in the path.

Discarded (never written): human seats, players whose exact model was not verified, games that
are drafts or incomplete, and games where too many turns have no established speaker.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .dataset import DEFAULT_EXCLUDED_PHASES
from .schema import HUMAN, NARRATOR, UNKNOWN_MODEL, UNKNOWN_SPEAKER, Alignment, Game

REPORT_NAME = "_report.json"
# Custom / homemade AIs whose underlying model is not known (e.g. Z2) are kept, but in their own box.
CUSTOM_FAMILY = "custom"
UNIDENTIFIED_DIR = "_unidentified"


def safe_segment(name: str) -> str:
    """Lower-case, filesystem-safe folder name (no separators, no dots-only names)."""
    seg = re.sub(r"[^a-z0-9._-]+", "-", name.strip().lower()).strip("-.")
    if not seg:
        raise ValueError(f"cannot make a folder name from {name!r}")
    return seg


@dataclass
class SortReport:
    written: Counter = field(default_factory=Counter)  # "family/model/side" -> turns written
    files: int = 0
    discarded_games: dict[str, list[str]] = field(default_factory=dict)
    discarded_players: dict[str, str] = field(default_factory=dict)  # "game/player" -> reason
    unknown_winner: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "files": self.files,
            "turns_by_folder": dict(sorted(self.written.items())),
            "discarded_games": self.discarded_games,
            "discarded_players": self.discarded_players,
            "games_with_unknown_winner": self.unknown_winner,
        }


def _is_custom(player) -> bool:
    return (player.family or "").strip().lower() == CUSTOM_FAMILY


def _folder(out: Path, player, side: str) -> tuple[Path, str]:
    """Verified models go to <family>/<model>/<side>; custom AIs to _unidentified/<name>/<side>."""
    if _is_custom(player):
        name = player.model if player.model.strip().lower() not in ("", UNKNOWN_MODEL) else player.player_id
        rel = f"{UNIDENTIFIED_DIR}/{safe_segment(name)}/{side}"
    else:
        rel = f"{safe_segment(player.family)}/{safe_segment(player.model)}/{side}"
    return out / rel, rel


def _player_skip_reason(player) -> str | None:
    if player.model.strip().lower() == HUMAN:
        return "human player"
    if _is_custom(player):
        return None  # kept apart in _unidentified/ until the underlying model is known
    if player.model.strip().lower() in ("", UNKNOWN_MODEL):
        return "exact model not verified"
    if not (player.family or "").strip():
        return "model family not recorded"
    return None


def sort_games(
    games: Iterable[Game],
    out_dir: str | Path = "dataset",
    *,
    require_reviewed: bool = True,
    max_unknown: float = 0.4,
    exclude_phases: Iterable[str] = DEFAULT_EXCLUDED_PHASES,
) -> SortReport:
    out = Path(out_dir)
    excluded = {p.lower() for p in exclude_phases}
    report = SortReport()
    for game in games:
        reasons = list(game.completeness_problems())
        if require_reviewed and game.annotation_status != "reviewed":
            reasons.append("annotation_status is not 'reviewed'")
        if game.unattributed_fraction() > max_unknown:
            reasons.append(
                f"{game.unattributed_fraction():.0%} of turns have no established speaker (limit {max_unknown:.0%})"
            )
        if reasons:
            report.discarded_games[game.game_id] = reasons
            continue
        if not (game.winner or "").strip():
            report.unknown_winner.append(game.game_id)
        for player in game.players:
            why = _player_skip_reason(player)
            if why:
                report.discarded_players[f"{game.game_id}/{player.player_id}"] = why
                continue
            turns = [
                t for t in game.turns
                if t.speaker_id == player.player_id and t.text.strip() and (t.phase or "").lower() not in excluded
            ]
            if not turns:
                report.discarded_players[f"{game.game_id}/{player.player_id}"] = "no usable turns"
                continue
            side = "lying" if player.alignment is Alignment.DECEIVER else "truth"
            folder, rel = _folder(out, player, side)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{safe_segment(game.game_id)}__{safe_segment(player.player_id)}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for t in turns:
                    f.write(json.dumps({
                        "game_id": game.game_id, "source": game.source, "winner": game.winner,
                        "player_id": player.player_id, "model": player.model, "family": player.family,
                        "role": player.role, "alignment": player.alignment.value,
                        "turn_id": t.turn_id, "start": t.start, "end": t.end, "phase": t.phase,
                        "deceptive": t.deceptive, "text": t.text,
                        "claims": [c.model_dump(mode="json") for c in t.claims],
                    }, ensure_ascii=False) + "\n")
            report.files += 1
            report.written[rel] += len(turns)
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    return report


def clean_output(out_dir: str | Path) -> None:
    """Delete a previous sort. Refuses unless the directory carries this tool's report file."""
    out = Path(out_dir)
    if not out.exists():
        return
    if not (out / REPORT_NAME).exists():
        raise ValueError(f"{out} has no {REPORT_NAME}; refusing to delete a directory this tool did not write")
    shutil.rmtree(out)
