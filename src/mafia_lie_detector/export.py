"""Clean, human-free export of annotated games for training.

Writes:
- ``turns.jsonl``: public AI speech, one row per turn, with label (lying/truth), provenance and split.
- ``private.jsonl``: AI internal thoughts / evil-team chat / night actions (same cleaning), for
  research that wants them; not deception examples.
- ``games/``: the game files with every human turn removed and human addressees anonymised.
- ``report.json``: counts and every drop by reason.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .dataset import OUTCOME_PHASES, PRIVATE_PHASES
from .schema import HUMAN, NARRATOR, UNKNOWN_MODEL, UNKNOWN_SPEAKER, Alignment, Game

CUSTOM_FAMILY = "custom"
_CUE = re.compile(r"\[[^\]]{0,40}\]")  # [laughter], [music], [clears throat]
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9']+")


def normalise_text(text: str) -> str:
    text = _CUE.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    return text.strip("\"“”'‘’ ").strip()


def _dedupe_key(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _rank(game_id: str) -> int:
    return int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)


def assign_splits(groups: dict[str, str], test_fraction: float = 0.2) -> dict[str, str]:
    """Game-level train/test split, stratified by group (e.g. text provenance).

    Within each group, games are ranked by a hash of their id and the lowest-ranked
    round(n * test_fraction) go to test (at least one when the group has two or more games).
    Deterministic, and with few games it still yields a real test set, unlike a per-game coin flip.
    """
    by_group: dict[str, list[str]] = {}
    for gid, grp in groups.items():
        by_group.setdefault(grp, []).append(gid)
    split = {}
    for ids in by_group.values():
        ids.sort(key=_rank)
        n_test = round(len(ids) * test_fraction)
        if len(ids) >= 2:
            n_test = max(1, n_test)
        for i, gid in enumerate(ids):
            split[gid] = "test" if i < n_test else "train"
    return split


@dataclass
class ExportReport:
    rows: Counter = field(default_factory=Counter)  # "public"/"private" -> rows
    labels: Counter = field(default_factory=Counter)
    by_model: Counter = field(default_factory=Counter)
    by_provenance: Counter = field(default_factory=Counter)
    by_split: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)  # reason -> turns
    dropped_games: dict[str, list[str]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {k: (dict(sorted(v.items())) if isinstance(v, Counter) else v) for k, v in self.__dict__.items()}


def _player_status(p) -> str | None:
    """None if the player's speech is usable, else the reason it is not."""
    model = p.model.strip().lower()
    if model == HUMAN:
        return "human player"
    if (p.family or "").strip().lower() == CUSTOM_FAMILY:
        return None
    if model in ("", UNKNOWN_MODEL):
        return "exact model not verified"
    if not (p.family or "").strip():
        return "model family not recorded"
    return None


def export_clean(
    games: Iterable[Game],
    out_dir: str | Path,
    *,
    min_words: int = 3,
    test_fraction: float = 0.2,
    require_reviewed: bool = True,
    max_unknown: float = 0.4,
) -> ExportReport:
    out = Path(out_dir)
    (out / "games").mkdir(parents=True, exist_ok=True)
    report = ExportReport()
    seen: set[str] = set()
    public_f = (out / "turns.jsonl").open("w", encoding="utf-8")
    private_f = (out / "private.jsonl").open("w", encoding="utf-8")
    games = list(games)
    splits = assign_splits({g.game_id: g.text_provenance for g in games}, test_fraction)
    try:
        for game in games:
            reasons = list(game.completeness_problems())
            if require_reviewed and game.annotation_status != "reviewed":
                reasons.append("annotation_status is not 'reviewed'")
            if game.unattributed_fraction() > max_unknown:
                reasons.append("too many turns without an established speaker")
            if reasons:
                report.dropped_games[game.game_id] = reasons
                continue
            humans = {p.player_id for p in game.players if p.model.strip().lower() == HUMAN}
            status = {p.player_id: _player_status(p) for p in game.players}
            split = splits[game.game_id]

            def anon(ids: list[str]) -> list[str]:
                return ["human" if a in humans else a for a in ids]

            # Redacted game file: no human words anywhere.
            redacted = game.model_copy(deep=True)
            redacted.turns = [t for t in redacted.turns if t.speaker_id not in humans]
            for t in redacted.turns:
                t.addressees = anon(t.addressees)
            for p in redacted.players:
                if p.player_id in humans:
                    p.aliases = []
            (out / "games" / f"{game.game_id}.json").write_text(redacted.model_dump_json(indent=2) + "\n", encoding="utf-8")

            for t in game.turns:
                if t.speaker_id in (None, NARRATOR, UNKNOWN_SPEAKER):
                    continue
                why = status.get(t.speaker_id)
                if why:
                    report.dropped[why] += 1
                    continue
                phase = (t.phase or "day").lower()
                if phase in OUTCOME_PHASES:
                    report.dropped[f"outcome phase ({phase})"] += 1
                    continue
                text = normalise_text(t.text)
                if len(_WORD.findall(text.lower())) < min_words:
                    report.dropped[f"fewer than {min_words} words"] += 1
                    continue
                key = _dedupe_key(text)
                if key in seen:
                    report.dropped["duplicate text"] += 1
                    continue
                seen.add(key)
                p = game.player(t.speaker_id)
                private = phase in PRIVATE_PHASES
                label = "lying" if p.alignment is Alignment.DECEIVER else "truth"
                row = {
                    "id": f"{game.game_id}:{t.turn_id}", "game_id": game.game_id, "split": split,
                    "text_provenance": game.text_provenance, "family": p.family, "model": p.model,
                    "model_verified": (p.family or "").lower() != CUSTOM_FAMILY,
                    "player_id": p.player_id, "role": p.role, "label": label, "phase": phase,
                    "addressees": anon(t.addressees), "winner": game.winner, "text": text,
                }
                (private_f if private else public_f).write(json.dumps(row, ensure_ascii=False) + "\n")
                kind = "private" if private else "public"
                report.rows[kind] += 1
                if not private:
                    report.labels[label] += 1
                    report.by_model[f"{p.family}/{p.model}"] += 1
                    report.by_provenance[game.text_provenance] += 1
                    report.by_split[split] += 1
    finally:
        public_f.close()
        private_f.close()
    (out / "report.json").write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    return report
