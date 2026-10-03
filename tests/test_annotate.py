from mafia_lie_detector.annotate.draft import make_draft, utterances_from_captions
from mafia_lie_detector.annotate.reveal import find_reveal_candidates, youtube_link
from mafia_lie_detector.annotate.segment import (
    Utterance, assign_speakers_by_prefix, clean_caption_text, snippets_to_utterances,
)


def snip(text, start, dur=1.0):
    return {"text": text, "start": start, "duration": dur}


def test_clean_caption_text():
    assert clean_caption_text("[Music] hello   &amp; welcome ♪") == "hello & welcome"
    assert clean_caption_text("[Applause]") == ""


def test_merges_close_snippets_and_splits_on_pause():
    out = snippets_to_utterances([snip("I think", 0.0), snip("it was him", 1.0), snip("next point", 10.0)])
    assert [u.text for u in out] == ["I think it was him", "next point"]


def test_overlapping_durations_do_not_create_false_pauses():
    # Auto-captions show each line for ~4s although the next starts after 1.5s.
    out = snippets_to_utterances([snip("so I was", 0.0, 4.0), snip("asleep all night", 1.5, 4.0)])
    assert [u.text for u in out] == ["so I was asleep all night"]


def test_speaker_change_marker_forces_new_utterance():
    out = snippets_to_utterances([snip("I voted for him. >> No you did not", 0.0, 2.0)])
    assert [u.text for u in out] == ["I voted for him.", "No you did not"]


def test_marker_at_snippet_start():
    out = snippets_to_utterances([snip("first speaker", 0.0), snip(">> second speaker", 1.0)])
    assert [u.text for u in out] == ["first speaker", "second speaker"]


def test_punctuated_long_sentence_closes_the_turn():
    long_sentence = "I have been watching everyone closely this whole round and honestly nothing adds up at all."
    assert len(long_sentence) >= 80
    out = snippets_to_utterances([snip(long_sentence, 0.0), snip("Then another.", 1.0)])
    assert len(out) == 2


def test_unsorted_input_and_sound_cues():
    out = snippets_to_utterances([snip("world", 1.0), snip("[Music]", 5.0), snip("hello", 0.0)])
    assert [u.text for u in out] == ["hello world"]


def test_assign_speakers_by_prefix():
    utts = [Utterance("Alice: I am town", 0, 1), Utterance("bob - no way", 1, 2), Utterance("plain", 2, 3)]
    out = assign_speakers_by_prefix(utts, ["Alice", "Bob"])
    assert [(u.speaker_hint, u.text) for u in out] == [("Alice", "I am town"), ("Bob", "no way"), (None, "plain")]


def test_reveal_candidates_rank_role_statements():
    utts = [
        Utterance("I think we should vote now", 10, 12),
        Utterance("And the mafia were Alice and Bob", 100, 104),
        Utterance("Bob was the werewolf all along", 110, 112),
        Utterance("game over, the town wins", 120, 123),
        Utterance("the mafia is probably P3 right now", 30, 33),  # mid-game speculation, not a reveal
    ]
    found = find_reveal_candidates(utts)
    assert sorted(c.index for c in found) == [1, 2, 3]
    assert youtube_link("abc123", 110.9) == "https://youtu.be/abc123?t=110"


def test_make_draft_from_captions_keeps_hints_and_nulls_speakers():
    captions = {"video_id": "vid1", "source": "youtube_captions",
                "snippets": [snip("Alice: hello there", 0.0), snip("a pause then more", 9.0)]}
    game = make_draft("vid1", utterances_from_captions(captions, ["Alice"]), title="Test game")
    assert game.game_id == "yt-vid1" and game.annotation_status == "draft"
    assert game.players == [] and all(t.speaker_id is None for t in game.turns)
    assert game.turns[0].speaker_hint == "Alice" and game.turns[0].text == "hello there"
    assert game.turns[1].start == 9.0
