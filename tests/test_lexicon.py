import math
from collections import Counter

import pytest

from mafia_lie_detector.lexicon import (
    Lexicon, Unit, benjamini_hochberg, build_units, fightin_words, fit_lexicon, mantel_haenszel, name_pattern,
    permutation_test, tokenize,
)


def test_tokenize_and_name_scrubbing():
    pat = name_pattern(["Alice"])
    assert tokenize("Why did Grock vote? Gemini and Alice agree!", pat) == [
        "why", "did", "<p>", "vote", "?", "<p>", "and", "<p>", "agree", "!"]
    assert tokenize("ChatGPT-4o left", pat) == ["<p>", "<p>", "left"]  # brand and seat nickname, both scrubbed
    assert tokenize("I'm 100% sure at 4:22") == ["i'm", "100", "sure", "at", "4:22"]


def test_fightin_words_direction_and_shrinkage():
    a = Counter({"if": 30, "the": 100, "rare": 1})
    b = Counter({"if": 5, "the": 100, "other": 1})
    fw = fightin_words(a, b)
    assert fw["if"][0] > 0 and fw["if"][1] > 2
    assert abs(fw["rare"][1]) < 1.5  # a single occurrence cannot be a strong signal


def rows_with_signal(n_games=6):
    rows = []
    for g in range(n_games):
        for p in range(4):
            lying = p < 2
            for k in range(5):
                tell = " why if" if lying and k % 2 == 0 else ""
                rows.append({"game_id": f"g{g}", "player_id": f"p{p}", "model": f"m{p % 3}",
                             "label": "lying" if lying else "truth",
                             "text": f"I think the vote should go to player {k} today{tell}."})
    return rows


def test_permutation_test_finds_planted_signal():
    units = build_units(rows_with_signal(), None)
    res = permutation_test(units, ["why", "if", "vote", "today"], n_perm=300)
    assert res["why"]["p"] < 0.05 and res["if"]["delta"] > 0
    assert abs(res["vote"]["delta"]) < 0.3 * res["why"]["delta"]  # present in every utterance: no real signal
    assert benjamini_hochberg({w: d["p"] for w, d in res.items()}, 0.1) >= {"why", "if"}


def test_mantel_haenszel_pools_within_strata():
    units = build_units(rows_with_signal(), None)
    mh = mantel_haenszel(units, ["why", "vote"], lambda u: u.model)
    assert mh["why"]["mh_log_or"] > 0 and mh["why"]["agree"] == mh["why"]["strata"]
    assert abs(mh["vote"]["mh_log_or"]) < 0.2


def test_unit_total_override():
    u = Unit("g", "p", "m", True, Counter({"cat": 3}), total=50)
    assert u.n_tokens == 50 and Unit("g", "p", "m", True, Counter({"a": 2})).n_tokens == 2


def test_fit_score_roundtrip(tmp_path):
    rows = rows_with_signal()
    lex = fit_lexicon(rows, None, min_count=3, max_words=10)
    assert lex.weights["why"] > 0 and lex.weights["if"] > 0
    p_lie = lex.probability("why would you vote if that is true")
    p_truth = lex.probability("I was in storage with the doctor")
    assert p_lie > p_truth
    assert math.isclose(lex.intercept, math.log((60 + 1) / (60 + 1)), abs_tol=1e-9)
    lex.save(tmp_path / "lex.json")
    assert Lexicon.load(tmp_path / "lex.json").score("why if") == pytest.approx(lex.score("why if"))
