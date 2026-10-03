"""Synthetic games for exercising the pipeline offline.

The text here is template filler with a planted signal. Scores obtained on it say nothing
about real LLM deception; it exists so the code paths (schema -> dataset -> splits -> model)
can be tested and demonstrated without any real transcripts.
"""
from __future__ import annotations

import random

from .schema import NARRATOR, Alignment, Claim, ClaimKind, Game, Player, Turn

LLMS = ("llm_a", "llm_b", "llm_c", "human")
FAMILY = {"llm_a": "family-x", "llm_b": "family-x", "llm_c": "family-y"}  # humans have none

_LIE = [
    "Honestly I was asleep all night, trust me.",
    "Why would I lie? I am clearly just a villager.",
    "I swear I am town, you can check anything.",
    "I did not do anything last night, I promise.",
]
_NEUTRAL = [
    "I think {x} is acting strange today.",
    "I do not have much information yet, who has a read?",
    "I am going to vote for {x} because of the earlier comment.",
    "Let us slow down and think about who voted for whom.",
    "Something about {x} does not add up for me.",
    "Can someone explain why {x} stayed quiet?",
]


def make_games(n_games: int = 60, *, seed: int = 0, turns_per_player: int = 4) -> list[Game]:
    rng = random.Random(seed)
    games: list[Game] = []
    for g in range(n_games):
        seats = [f"P{i}" for i in range(1, 7)]
        deceivers = set(rng.sample(seats, 2))
        models = [rng.choice(LLMS) for _ in seats]
        players = [
            Player(
                player_id=pid,
                model=model,
                family=FAMILY.get(model),
                role="mafia" if pid in deceivers else "villager",
                alignment=Alignment.DECEIVER if pid in deceivers else Alignment.TRUTHFUL,
            )
            for pid, model in zip(seats, models)
        ]
        turns = [Turn(turn_id=0, speaker_id=NARRATOR, text="Welcome to the game.", phase="intro")]
        for _ in range(turns_per_player):
            for p in rng.sample(players, len(players)):
                other = rng.choice([s for s in seats if s != p.player_id])
                lying = p.alignment is Alignment.DECEIVER and rng.random() < 0.5
                lying = lying or (p.alignment is Alignment.TRUTHFUL and rng.random() < 0.03)
                # The planted signal is imperfect: liars use the telltale phrases 70% of the time
                # and a few honest turns use them too.
                if (lying and rng.random() < 0.7) or (not lying and rng.random() < 0.08):
                    text = rng.choice(_LIE)
                else:
                    text = rng.choice(_NEUTRAL).format(x=other)
                claims = (
                    [Claim(text=text, kind=ClaimKind.ALIBI, truthful=False)] if lying else []
                )
                turns.append(
                    Turn(
                        turn_id=len(turns),
                        speaker_id=p.player_id,
                        text=text,
                        phase="day",
                        claims=claims,
                        deceptive=lying,
                    )
                )
        # Post-game turns name the roles outright; the dataset builder must keep them out.
        for pid in sorted(deceivers):
            turns.append(
                Turn(turn_id=len(turns), speaker_id=NARRATOR, text=f"{pid} was the mafia.", phase="reveal")
            )
        games.append(
            Game(
                game_id=f"synthetic-{g:03d}",
                source="synthetic",
                players=players,
                turns=turns,
                annotation_status="reviewed",
            )
        )
    return games
