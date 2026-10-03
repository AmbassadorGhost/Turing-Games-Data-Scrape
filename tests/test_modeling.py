import math

import numpy as np
import pytest

from mafia_lie_detector.dataset import Example, build_examples
from mafia_lie_detector.modeling import (
    evaluate, group_kfold, leave_model_out, make_model, score_metrics, style_features,
)
from mafia_lie_detector.synthetic import make_games


@pytest.fixture(scope="module")
def examples():
    return build_examples(make_games(60, seed=1), scrub=True).examples


def test_synthetic_games_exercise_the_exclusions():
    games = make_games(3, seed=0)
    assert any(t.phase == "reveal" for g in games for t in g.turns)
    assert not any(e.text.endswith("was the mafia.") for e in build_examples(games).examples)


def test_group_kfold_never_splits_a_group():
    groups = [f"g{i % 12}" for i in range(120)]
    seen_test = []
    for _, train, test in group_kfold(groups, 4, seed=3):
        assert not {groups[i] for i in train} & {groups[i] for i in test}
        assert len(train) + len(test) == len(groups)
        seen_test += test
    assert sorted(seen_test) == list(range(len(groups)))  # every example tested exactly once


def test_group_kfold_is_deterministic_and_balanced():
    groups = [f"g{i % 10}" for i in range(100)]
    a = [t for _, _, t in group_kfold(groups, 5, seed=7)]
    assert a == [t for _, _, t in group_kfold(groups, 5, seed=7)]
    assert {len(t) for t in a} == {20}


def test_group_kfold_rejects_too_many_splits():
    with pytest.raises(ValueError):
        list(group_kfold(["a", "a", "b"], 3))


def test_leave_model_out_holds_out_one_model():
    models = ["x"] * 30 + ["y"] * 30 + ["z"] * 5
    games = [f"g{i % 6}" for i in range(65)]
    folds = list(leave_model_out(models, games, min_test=20))
    assert [f[0] for f in folds] == ["x", "y"]  # z is too small to test
    for held, train, test in folds:
        assert {models[i] for i in test} == {held}
        assert held not in {models[i] for i in train}


def test_leave_model_out_strict_drops_shared_games():
    # g0 seats only x, g1 seats both, g2 seats only y.
    games = ["g0"] * 10 + ["g1"] * 20 + ["g2"] * 10
    models = ["x"] * 10 + ["x", "y"] * 10 + ["y"] * 10
    folds = list(leave_model_out(models, games, min_test=5, drop_shared_games=True))
    assert [f[0] for f in folds] == ["x", "y"]
    for _, train, test in folds:
        assert train and not {games[i] for i in test} & {games[i] for i in train}
    loose = {m: len(tr) for m, tr, _ in leave_model_out(models, games, min_test=5)}
    assert all(len(tr) < loose[m] for m, tr, _ in folds)


def test_metrics_handle_single_class():
    m = score_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert math.isnan(m["roc_auc"]) and m["f1"] == 1.0
    m = score_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m["roc_auc"] == 1.0 and m["balanced_acc"] == 1.0


def test_style_features_shape_and_values():
    X = style_features(["I think maybe not?!", ""])
    assert X.shape == (2, 9)
    assert X[0, 6] == 1 and X[0, 7] == 1  # one "?" and one "!"
    assert np.isfinite(X).all()


@pytest.mark.parametrize("kind", ["style", "tfidf", "tfidf_style", "majority"])
def test_every_model_kind_fits_and_scores(examples, kind):
    model = make_model(kind).fit([e.text for e in examples], [e.label for e in examples])
    p = model.predict_proba([e.text for e in examples[:5]])[:, 1]
    assert p.shape == (5,) and ((0 <= p) & (p <= 1)).all()


def test_group_kfold_detects_the_planted_signal(examples):
    r = evaluate(examples, split="group_kfold", model_kind="tfidf_style", n_splits=5)
    assert len(r["folds"]) == 5
    assert r["pooled"]["roc_auc"] > 0.7
    assert r["pooled_majority_baseline"]["roc_auc"] == pytest.approx(0.5)
    assert all(f["majority_baseline"]["roc_auc"] == pytest.approx(0.5) for f in r["folds"])
    assert r["n_examples"] == len(examples)


def test_leave_model_out_runs_per_model_and_reports_context_flag(examples):
    r = evaluate(examples, split="leave_model_out", use_context=False, min_test=50)
    assert {f["held_out"] for f in r["folds"]} <= {"llm_a", "llm_b", "llm_c", "human"}
    assert r["pooled"]["roc_auc"] > 0.65 and r["use_context"] is False


def test_strict_leave_model_out_trains_on_fewer_examples(examples):
    loose = {f["held_out"]: f["n_train"] for f in evaluate(examples, split="leave_model_out", min_test=50)["folds"]}
    strict = {
        f["held_out"]: f["n_train"]
        for f in evaluate(examples, split="leave_model_out", drop_shared_games=True, min_test=50)["folds"]
    }
    assert strict and all(strict[m] < loose[m] for m in strict)


def test_evaluate_rejects_bad_inputs(examples):
    one_class = [Example(**{**examples[0].__dict__, "label": 0}) for _ in range(10)]
    with pytest.raises(ValueError, match="single class"):
        evaluate(one_class)
    with pytest.raises(ValueError):
        evaluate(examples, split="nope")
    with pytest.raises(ValueError, match="no LLM can be held out"):
        evaluate(examples, split="leave_model_out", min_test=10_000)
