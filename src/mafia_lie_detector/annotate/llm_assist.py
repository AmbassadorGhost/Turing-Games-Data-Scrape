"""Optional LLM pre-annotation of speakers, phases and claims.

Workflow: a human first fills in ``players`` (roles come from the reveal), then this proposes
speakers / phases / claim truth values for the turns that are still empty. It only fills blanks,
never overwrites a human value, and never marks a game ``reviewed``.

Claim truth is judged only from the roster's true roles and facts stated in the transcript. Hidden
game state (who was killed, who the detective checked) is not visible, so many claims cannot be
judged and are left out; treat the output as a head start for review, not as ground truth.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from ..schema import NARRATOR, Claim, ClaimKind, Game

PHASES = ("intro", "day", "vote", "night", "reveal", "postgame", "other")

SYSTEM = (
    "You help annotate transcripts of social-deduction games (Mafia / Werewolf style) for "
    "research on deception detection. You are given the true roster, including hidden roles, and a "
    "numbered chunk of transcript. Reply with JSON only."
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def build_prompt(game: Game, turn_ids: list[int]) -> str:
    wanted = set(turn_ids)
    roster = "\n".join(
        f"- {p.player_id} | model={p.model} | role={p.role} | alignment={p.alignment.value}"
        for p in game.players
    )
    allowed = ", ".join([p.player_id for p in game.players] + [NARRATOR])
    lines = "\n".join(f"[{t.turn_id}] {t.text}" for t in game.turns if t.turn_id in wanted)
    return (
        f"Roster (true hidden roles):\n{roster}\n\n"
        f"Allowed speaker ids: {allowed}\n\n"
        f"Transcript chunk, one line per turn as [turn_id] text:\n{lines}\n\n"
        "Return a JSON array with one object per turn:\n"
        '{"turn_id": <int>, "speaker_id": <allowed id, or null if unclear>, '
        f'"phase": <one of {", ".join(PHASES)}>, '
        '"claims": [{"text": <the claim>, "kind": <role_claim|knowledge_claim|alibi|accusation|'
        'vote_intent|other>, "truthful": <true|false>}]}\n\n'
        "Rules:\n"
        "- Do not guess speakers. Use null when the text does not make the speaker clear.\n"
        "- Host or voice-over lines use the speaker id " + NARRATOR + ".\n"
        "- Judge `truthful` from the speaker's TRUE role in the roster and from facts stated in the "
        "transcript. A claim the speaker would know to be false is truthful=false. Leave out any claim "
        "whose truth cannot be determined from those sources.\n"
        "- Output the JSON array and nothing else."
    )


def parse_suggestions(raw: str) -> list[dict]:
    text = _FENCE.sub("", raw.strip())
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array found in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("expected a JSON array")
    return [d for d in data if isinstance(d, dict) and isinstance(d.get("turn_id"), int)]


def apply_suggestions(game: Game, suggestions: list[dict]) -> dict[str, int]:
    """Fill empty fields from ``suggestions``; returns counts of what was filled."""
    by_id = {t.turn_id: t for t in game.turns}
    valid_speakers = {p.player_id for p in game.players} | {NARRATOR}
    counts = {"speaker": 0, "phase": 0, "claims": 0}
    for s in suggestions:
        turn = by_id.get(s["turn_id"])
        if turn is None:
            continue
        speaker = s.get("speaker_id")
        if turn.speaker_id is None and speaker in valid_speakers:
            turn.speaker_id = speaker
            counts["speaker"] += 1
        phase = s.get("phase")
        if turn.phase is None and phase in PHASES:
            turn.phase = phase
            counts["phase"] += 1
        if turn.claims:
            continue
        claims = []
        for c in s.get("claims") or []:
            if not isinstance(c, dict) or not isinstance(c.get("truthful"), bool) or not c.get("text"):
                continue
            kind = c.get("kind")
            claims.append(
                Claim(
                    text=str(c["text"]),
                    kind=ClaimKind(kind) if kind in {k.value for k in ClaimKind} else ClaimKind.OTHER,
                    truthful=c["truthful"],
                )
            )
        if not claims:
            continue
        derived = any(not c.truthful for c in claims)
        if turn.deceptive is not None and turn.deceptive != derived:
            continue  # a human label disagrees; keep it and skip the proposal
        turn.claims = claims
        turn.deceptive = derived
        counts["claims"] += 1
    return counts


def annotate_game(
    game: Game, complete: Callable[[str], str], *, chunk_size: int = 40
) -> dict[str, int]:
    if not game.players:
        raise ValueError("fill in `players` (roles from the reveal) before asking for suggestions")
    totals = {"speaker": 0, "phase": 0, "claims": 0}
    todo = [
        t.turn_id for t in game.turns if t.speaker_id is None or t.phase is None or not t.claims
    ]
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i : i + chunk_size]
        counts = apply_suggestions(game, parse_suggestions(complete(build_prompt(game, chunk))))
        for k, v in counts.items():
            totals[k] += v
    return totals


def anthropic_completer(
    model: str, *, max_tokens: int = 16000, api_key: Optional[str] = None
) -> Callable[[str], str]:
    """Build a ``complete(prompt) -> str`` backed by the Anthropic API (the ``llm`` extra).

    ``model`` is deliberately required: pick it per run (see ``MLD_ANNOTATION_MODEL`` in the CLI).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def complete(prompt: str) -> str:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if message.stop_reason == "refusal":
            raise RuntimeError("the model declined this chunk; review it by hand")
        if message.stop_reason == "max_tokens":
            raise RuntimeError("model output was cut off; retry with a smaller --chunk-size")
        return "".join(b.text for b in message.content if b.type == "text")

    return complete
