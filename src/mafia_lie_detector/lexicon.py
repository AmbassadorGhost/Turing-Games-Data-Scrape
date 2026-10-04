"""Keyword lie detector: word statistics, confound controls, and a scoring lexicon.

Core statistic: weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi & Quinn
2008, "Fightin' Words"), which is far more stable for rare words than raw frequency ratios.
Significance comes from a permutation test at the (game, player) level: labels are shuffled among
the players *within* each game, so game-specific vocabulary and the per-game liar count are held
fixed. Stratified (Mantel-Haenszel) estimates control for which model is speaking.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

TOKEN = re.compile(r"[a-z][a-z']*|\d+(?:[:.]\d+)?|[?!]")
NAME_TOKEN = "<p>"

# Names of seats/models as they are spoken in the games. Any token matching is replaced, so the
# detector cannot learn "who is talking about whom" instead of "how liars talk".
GENERIC_NAMES = [
    "chatgpt", "chat gpt", "gpt", "4o", "40", "o4", "5.1", "5.2", "5.4", "5.5", "5.6", "gemini", "flash", "pro",
    "claude", "opus", "sonnet", "fable", "grok", "grock", "groc", "kimi", "kimmy", "k2", "llama", "yama",
    "deepseek", "deep seek", "deepse", "glm", "astra", "z2", "morpheus", "5up", "five up", "fiveup", "twitch",
    "youtube", "unyx", "clone",
]


# Vocabulary that only exists because these are Mafia / Among Us / Werewolf games. A lexicon meant
# to transfer to other settings (e.g. agents talking in AI Village) must not lean on any of it.
GAME_TERMS = frozenset("""
mafia mafioso mafias godfather impostor impostors imposter imposters crewmate crewmates crew villager villagers
village town townie townies werewolf werewolves wolf wolves minion tanner jester seer robber troublemaker insomniac
mason masons hunter drunk doctor sheriff detective vigilante vig jailor jail jailed cop medic bodyguard escort mayor
investigator investigate investigated investigation investigations check checks checked red green clear cleared
clears hardclear hard-clear innocent guilty flip flipped flips lynch lynched lynching mislynch mislynched wagon
wagons bus bussing bussed hammer hammered eject ejected ejection execute executed execution vote voted votes voting
voter unvote abstain abstained abstaining skip skipped skipping night nights nightly day days daytime kill killed
kills killer killing murder murdered dead death deaths die died dies body bodies corpse protect protected protection
save saved saves heal healed target targeted targets role roles claim claims claimed claiming counterclaim alibi
alibis task tasks vent vented vents sabotage sabotaged lights blackout cams camera cameras electrical medbay
cafeteria cafe storage navigation nav weapons shields admin security reactor o2 comms communications engine engines
upper lower hallway hall spawn lobby meeting emergency button report reported reporting self-report parity majority
scum scumread townread powerrole mechanic mechanics round rounds endgame
confirmed confirm confirms slip slipped yesterday tonight today
""".split())
NUMERIC = re.compile(r"^\d")  # bare numbers and timestamps ("2", "4:22") are game-log artefacts


def name_pattern(extra_names: Iterable[str] = ()) -> re.Pattern:
    names = sorted({n.lower() for n in [*GENERIC_NAMES, *extra_names] if len(n) >= 2}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(n) for n in names) + r")(?![a-z])(?:\s*\d+(?:\.\d+)*)?", re.I)


def tokenize(text: str, scrub: re.Pattern | None = None) -> list[str]:
    text = text.lower()
    if scrub is not None:
        text = scrub.sub(f" {NAME_TOKEN} ", text)
    toks = TOKEN.findall(text) if scrub is None else re.findall(r"<p>|" + TOKEN.pattern, text)
    return toks


def ngrams(tokens: Sequence[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# --------------------------------------------------------------------------- log-odds

def fightin_words(counts_a: Counter, counts_b: Counter, prior_scale: float = 0.1) -> dict[str, tuple[float, float]]:
    """Return word -> (delta, z): log-odds of group A over B with an informative Dirichlet prior.

    The prior is the pooled distribution scaled by ``prior_scale`` of the total token count, so
    rare words are shrunk towards zero instead of producing huge spurious ratios.
    """
    vocab = set(counts_a) | set(counts_b)
    n_a, n_b = sum(counts_a.values()), sum(counts_b.values())
    pooled = counts_a + counts_b
    n_pool = n_a + n_b
    a0 = prior_scale * n_pool
    out = {}
    for w in vocab:
        aw = a0 * pooled[w] / n_pool
        ya, yb = counts_a[w], counts_b[w]
        la = math.log((ya + aw) / (n_a + a0 - ya - aw))
        lb = math.log((yb + aw) / (n_b + a0 - yb - aw))
        delta = la - lb
        var = 1 / (ya + aw) + 1 / (yb + aw)
        out[w] = (delta, delta / math.sqrt(var))
    return out


# ----------------------------------------------------------------------- permutation

@dataclass
class Unit:
    """All tokens one player produced in one game: the natural unit for shuffling labels."""
    game: str
    player: str
    model: str
    lying: bool
    counts: Counter
    total: int | None = None  # token total; defaults to the sum of counts (differs for category counts)

    @property
    def n_tokens(self) -> int:
        return self.total if self.total is not None else sum(self.counts.values())


def build_units(rows: Iterable[dict], scrub: re.Pattern | None, n: int = 1) -> list[Unit]:
    by_key: dict[tuple, Unit] = {}
    for r in rows:
        key = (r["game_id"], r["player_id"])
        u = by_key.get(key)
        if u is None:
            u = by_key[key] = Unit(r["game_id"], r["player_id"], r["model"], r["label"] == "lying", Counter())
        toks = tokenize(r["text"], scrub)
        u.counts.update(toks if n == 1 else ngrams(toks, n))
    return list(by_key.values())


def permutation_test(
    units: list[Unit], vocab: list[str], n_perm: int = 2000, seed: int = 0, prior_scale: float = 0.1
) -> dict[str, dict]:
    """Within-game label shuffles. Returns word -> {delta, z, p} for the observed split."""
    rng = np.random.default_rng(seed)
    idx = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(units), len(vocab)))
    for i, u in enumerate(units):
        for w, c in u.counts.items():
            j = idx.get(w)
            if j is not None:
                M[i, j] = c
    totals = np.array([u.n_tokens for u in units], dtype=float)
    labels = np.array([u.lying for u in units])
    games = defaultdict(list)
    for i, u in enumerate(units):
        games[u.game].append(i)
    pooled = M.sum(axis=0)
    n_pool = totals.sum()
    a0 = prior_scale * n_pool
    alpha = a0 * pooled / n_pool

    def zscores(lab: np.ndarray) -> np.ndarray:
        ya, yb = M[lab].sum(axis=0), M[~lab].sum(axis=0)
        na, nb = totals[lab].sum(), totals[~lab].sum()
        delta = np.log((ya + alpha) / (na + a0 - ya - alpha)) - np.log((yb + alpha) / (nb + a0 - yb - alpha))
        return delta, delta / np.sqrt(1 / (ya + alpha) + 1 / (yb + alpha))

    d_obs, z_obs = zscores(labels)
    exceed = np.zeros(len(vocab))
    for _ in range(n_perm):
        lab = labels.copy()
        for members in games.values():
            members = np.array(members)
            lab[members] = labels[rng.permutation(members)]
        _, z = zscores(lab)
        exceed += np.abs(z) >= np.abs(z_obs)
    p = (exceed + 1) / (n_perm + 1)
    return {w: {"delta": float(d_obs[i]), "z": float(z_obs[i]), "p": float(p[i])} for w, i in idx.items()}


def benjamini_hochberg(pvals: dict[str, float], q: float = 0.1) -> set[str]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    keep, threshold = set(), 0.0
    for rank, (w, p) in enumerate(items, start=1):
        if p <= q * rank / m:
            threshold = p
    return {w for w, p in items if p <= threshold} if threshold else keep


# ------------------------------------------------------------------------- stratified

def mantel_haenszel(units: list[Unit], vocab: list[str], stratum_of) -> dict[str, dict]:
    """Pooled log odds ratio of each word across strata (e.g. model), plus how many strata agree in sign.

    Each stratum contributes a 2x2 table (word vs other tokens, lying vs truthful); strata where one
    side is empty are skipped. Only words are compared *within* a stratum, so a word that is just
    'how GPT talks' cannot look like a lie signal because GPT happened to be mafia more often.
    """
    strata: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        strata[stratum_of(u)].append(u)
    out = {}
    for w in vocab:
        num = den = 0.0
        signs = []
        for members in strata.values():
            ly = [u for u in members if u.lying]
            tr = [u for u in members if not u.lying]
            if not ly or not tr:
                continue
            a = sum(u.counts[w] for u in ly)
            c = sum(u.counts[w] for u in tr)
            if a == 0 and c == 0:
                continue
            b = sum(u.n_tokens for u in ly) - a
            d = sum(u.n_tokens for u in tr) - c
            n = a + b + c + d
            num += (a + 0.5) * (d + 0.5) / n
            den += (b + 0.5) * (c + 0.5) / n
            signs.append(np.sign(math.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))))
        if den > 0 and signs:
            out[w] = {"mh_log_or": math.log(num / den), "strata": len(signs),
                      "agree": int(sum(1 for s in signs if s == np.sign(math.log(num / den))))}
    return out


# --------------------------------------------------------------------------- lexicon

@dataclass
class Lexicon:
    weights: dict[str, float]
    intercept: float
    scrub_names: bool = True

    def score(self, text: str, scrub: re.Pattern | None = None) -> float:
        """Log-odds-style score: positive leans deceiving. Uses word presence, not counts."""
        toks = set(tokenize(text, scrub if self.scrub_names else None))
        return self.intercept + sum(self.weights.get(t, 0.0) for t in toks)

    def probability(self, text: str, scrub: re.Pattern | None = None) -> float:
        return 1 / (1 + math.exp(-self.score(text, scrub)))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"intercept": self.intercept, "scrub_names": self.scrub_names,
                                          "weights": dict(sorted(self.weights.items(), key=lambda kv: -abs(kv[1])))},
                                         indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["weights"], d["intercept"], d.get("scrub_names", True))


def fit_lexicon(
    rows: Iterable[dict], scrub: re.Pattern | None, *, min_count: int = 30, max_words: int = 40,
    prior_scale: float = 0.1, shrink: float = 0.5, exclude: frozenset[str] = frozenset()
) -> Lexicon:
    """Weights = shrunk Fightin'-Words log-odds of the top |z| words; intercept = base-rate log-odds.

    Deliberately simple so it can be read, audited and ported: a word list with signed weights.
    Defaults (words seen >= 30 times, 40 words) were chosen on held-out games: a short list of
    well-attested words generalises better than a long tail of rare ones.
    """
    rows = list(rows)
    ca, cb = Counter(), Counter()
    for r in rows:
        (ca if r["label"] == "lying" else cb).update(set(tokenize(r["text"], scrub)))
    stats = fightin_words(ca, cb, prior_scale)
    eligible = [w for w in stats if ca[w] + cb[w] >= min_count and w != NAME_TOKEN and w not in exclude
                and not (exclude and NUMERIC.match(w))]
    top = sorted(eligible, key=lambda w: -abs(stats[w][1]))[:max_words]
    weights = {w: shrink * stats[w][0] for w in top}
    n_ly = sum(r["label"] == "lying" for r in rows)
    intercept = math.log((n_ly + 1) / (len(rows) - n_ly + 1))
    return Lexicon(weights, intercept)
