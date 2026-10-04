import pytest

from mafia_lie_detector.schema import NARRATOR, Alignment, Claim, ClaimKind, Game, Player, Turn


@pytest.fixture
def small_game() -> Game:
    """Three players, one mafia, with an intro line, discussion, and a revealing postgame line."""
    players = [
        Player(player_id="P1", model="llm_a", role="mafia", alignment=Alignment.DECEIVER, aliases=["Alice"]),
        Player(player_id="P2", model="llm_b", role="villager", alignment=Alignment.TRUTHFUL),
        Player(player_id="P3", model="llm_c", role="detective", alignment=Alignment.TRUTHFUL),
    ]
    turns = [
        Turn(turn_id=0, speaker_id=NARRATOR, text="Welcome to the game.", phase="intro"),
        Turn(
            turn_id=1, speaker_id="P1", phase="day",
            text="I am just a villager, trust me. Alice here has nothing to hide.",
            claims=[Claim(text="I am a villager", kind=ClaimKind.ROLE_CLAIM, truthful=False)],
        ),
        Turn(turn_id=2, speaker_id="P2", phase="day", text="I think P1 is acting strange.", deceptive=False),
        Turn(turn_id=3, speaker_id="P3", phase="day", text="I checked P1 and they are mafia.", deceptive=False),
        Turn(turn_id=4, speaker_id="P2", phase="day", text="Unlabelled remark with no verdict."),
        Turn(turn_id=5, speaker_id=NARRATOR, phase="reveal", text="P1 was the mafia."),
        Turn(turn_id=6, speaker_id="P1", phase="postgame", text="Well played, you caught me."),
    ]
    return Game(game_id="g1", players=players, turns=turns, annotation_status="reviewed")
