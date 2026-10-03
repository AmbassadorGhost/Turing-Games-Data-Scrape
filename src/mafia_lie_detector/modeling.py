"""Baseline deception classifiers plus leakage-aware evaluation splits.

Two splits matter for this problem:

* ``group_kfold``     - whole games are held out together. Turns from one game share names,
  events and phrasing, so a random turn-level split would flatter the model.
* ``leave_model_out`` - every turn spoken by one underlying LLM is held out. This asks the
  question that motivates the project: does the detector work on an LLM it has never seen, or
  did it only learn that model's style?
"""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .dataset import Example

MODEL_KINDS = ("majority", "style", "tfidf", "tfidf_style")
SPLITS = ("group_kfold", "leave_model_out")

_WORD = re.compile(r"[a-z']+")
_HEDGES = frozenset(
    "maybe perhaps possibly probably might seems seem think guess honestly kinda somewhat suppose".split()
)
_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"})
_NEGATIONS = frozenset({"no", "not", "never", "nothing", "none", "nobody"})
STYLE_FEATURES = (
    "log_chars", "log_words", "mean_word_len", "first_person", "hedges",
    "negations", "question_marks", "exclamations", "upper_ratio",
)


def style_features(texts: Sequence[str]) -> np.ndarray:
    """Cheap stylometric cues. Deliberately content-free so they can be ablated against TF-IDF."""
    rows = []
    for text in texts:
        words = _WORD.findall(text.lower())
        n = max(len(words), 1)
        letters = [c for c in text if c.isalpha()]
        rows.append(
            [
                math.log1p(len(text)),
                math.log1p(len(words)),
                sum(len(w) for w in words) / n,
                sum(w in _FIRST_PERSON for w in words) / n,
                sum(w in _HEDGES for w in words) / n,
                sum(w in _NEGATIONS or w.endswith("n't") for w in words) / n,
                text.count("?"),
                text.count("!"),
                sum(c.isupper() for c in letters) / max(len(letters), 1),
            ]
        )
    return np.asarray(rows, dtype=float)


def make_model(kind: str = "tfidf_style", *, C: float = 1.0, min_df: int = 2, seed: int = 0) -> Pipeline:
    if kind not in MODEL_KINDS:
        raise ValueError(f"model kind must be one of {MODEL_KINDS}, got {kind!r}")
    if kind == "majority":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))])

    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, max_features=50_000)

    def style() -> Pipeline:
        return Pipeline(
            [("f", FunctionTransformer(style_features)), ("scale", StandardScaler(with_mean=False))]
        )

    if kind == "tfidf":
        features = tfidf()
    elif kind == "style":
        features = style()
    else:
        features = FeatureUnion([("tfidf", tfidf()), ("style", style())])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def example_text(ex: Example, use_context: bool) -> str:
    return f"{ex.context}\n>> {ex.text}" if use_context and ex.context else ex.text


# ---------------------------------------------------------------------------- splits

Fold = tuple[str, list[int], list[int]]  # (held-out label, train indices, test indices)


def group_kfold(groups: Sequence[str], n_splits: int = 5, seed: int = 0) -> Iterator[Fold]:
    """Deterministic, size-balanced group k-fold: a group never straddles train and test."""
    sizes: dict[str, int] = defaultdict(int)
    for g in groups:
        sizes[g] += 1
    if n_splits < 2 or n_splits > len(sizes):
        raise ValueError(f"need 2 <= n_splits <= number of groups ({len(sizes)}), got {n_splits}")
    order = list(sizes)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -sizes[g])  # stable: the shuffle breaks ties
    load = [0] * n_splits
    assignment: dict[str, int] = {}
    for g in order:
        k = load.index(min(load))
        assignment[g] = k
        load[k] += sizes[g]
    for k in range(n_splits):
        test = [i for i, g in enumerate(groups) if assignment[g] == k]
        train = [i for i, g in enumerate(groups) if assignment[g] != k]
        yield f"fold{k}", train, test


def leave_model_out(
    models: Sequence[str],
    games: Sequence[str],
    *,
    min_test: int = 20,
    drop_shared_games: bool = False,
) -> Iterator[Fold]:
    """Hold out all turns of one underlying LLM per fold.

    By default the other players in the same games stay in the training set. With
    ``drop_shared_games`` every game the held-out LLM appeared in is removed from training,
    which is stricter but, when each game seats several LLMs, may leave little to train on.
    """
    for m in sorted(set(models)):
        test = [i for i, x in enumerate(models) if x == m]
        if len(test) < min_test:
            continue
        banned = {games[i] for i in test} if drop_shared_games else set()
        train = [i for i, x in enumerate(models) if x != m and games[i] not in banned]
        if train:
            yield m, train, test


# ------------------------------------------------------------------------- evaluation

def score_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    both = len(set(y.tolist())) == 2
    pred = (s >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "avg_precision": float(average_precision_score(y, s)) if both else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)) if both else float("nan"),
    }


def _positive_scores(model: Pipeline, texts: list[str]) -> np.ndarray:
    return model.predict_proba(texts)[:, 1]


def evaluate(
    examples: Sequence[Example],
    *,
    split: str = "group_kfold",
    model_kind: str = "tfidf_style",
    n_splits: int = 5,
    use_context: bool = False,
    drop_shared_games: bool = False,
    min_test: int = 20,
    seed: int = 0,
) -> dict:
    """Cross-validate a baseline and report per-fold and pooled out-of-fold metrics."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if len(examples) < 2:
        raise ValueError("need at least two examples")
    labels = np.asarray([e.label for e in examples])
    if len(set(labels.tolist())) < 2:
        raise ValueError("labels contain a single class; cannot train a detector")
    texts = [example_text(e, use_context) for e in examples]

    if split == "group_kfold":
        folds = list(group_kfold([e.group for e in examples], n_splits, seed))
    else:
        folds = list(
            leave_model_out(
                [e.speaker_model for e in examples],
                [e.group for e in examples],
                min_test=min_test,
                drop_shared_games=drop_shared_games,
            )
        )
        if not folds:
            raise ValueError(
                f"no LLM can be held out: need >= {min_test} examples for it and some training data left"
            )

    report_folds = []
    pooled_idx: list[int] = []
    pooled_score: list[float] = []
    for held_out, train, test in folds:
        y_train = labels[train]
        if len(set(y_train.tolist())) < 2:
            continue  # nothing to learn from in this fold
        model = make_model(model_kind, seed=seed)
        model.fit([texts[i] for i in train], y_train)
        score = _positive_scores(model, [texts[i] for i in test])
        base = make_model("majority").fit([texts[i] for i in train], y_train)
        base_score = _positive_scores(base, [texts[i] for i in test])
        report_folds.append(
            {
                "held_out": held_out,
                "n_train": len(train),
                "n_test": len(test),
                "metrics": score_metrics(labels[test], score),
                "majority_baseline": score_metrics(labels[test], base_score),
            }
        )
        pooled_idx += test
        pooled_score += score.tolist()
    if not report_folds:
        raise ValueError("no usable folds (every training set had a single class)")
    return {
        "split": split,
        "model": model_kind,
        "use_context": use_context,
        "n_examples": len(examples),
        "positive_rate": float(labels.mean()),
        "folds": report_folds,
        "pooled": score_metrics(labels[pooled_idx], pooled_score),
        # A majority baseline is a constant score; per-fold priors would only add noise when pooled.
        "pooled_majority_baseline": score_metrics(
            labels[pooled_idx], np.full(len(pooled_idx), labels.mean())
        ),
    }
