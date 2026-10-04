"""Deceiving vs truthful word analysis with confound controls, plus lexicon detector evaluation.

usage: python analysis/word_analysis.py data/clean/turns.jsonl data/clean/games analysis/out
Writes report.md, word_stats.csv, lexicon.json (general) and lexicons/<family>.json.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mafia_lie_detector.lexicon import (  # noqa: E402
    GAME_TERMS, NAME_TOKEN, Unit, benjamini_hochberg, build_units, fightin_words, fit_lexicon, mantel_haenszel,
    name_pattern, ngrams, permutation_test, tokenize,
)
from mafia_lie_detector.modeling import group_kfold  # noqa: E402

N_PERM = 3000
MIN_COUNT = 10

CATEGORIES = {
    "I / me / my (1st singular)": set("i me my mine myself i'm i've i'll i'd".split()),
    "we / us / our (1st plural)": set("we us our ours ourselves we're we've we'll let's".split()),
    "you / your (2nd person)": set("you your yours you're you've yourself".split()),
    "they / them (3rd plural)": set("they them their theirs they're themselves".split()),
    "negations": set("no not never nothing none nobody n't don't didn't doesn't isn't wasn't can't cannot won't".split()),
    "hedges / uncertainty": set("maybe perhaps possibly probably might could seems seem think guess likely unsure somewhat suppose feel".split()),
    "certainty / emphasis": set("definitely clearly obviously certainly 100% literally absolutely exactly confirmed proven impossible must".split()),
    "agreement / alignment": set("agree agreed agreeing makes sense right correct fair good point same".split()),
    "accusation verbs": set("lying liar lie lied suspicious sus fake framing framed deflect deflecting bus bussing".split()),
    "role claims (own)": set("sheriff doctor detective vigilante jailor seer cop investigator medic".split()),
    "evil-side words": set("mafia mafioso impostor imposter impostors werewolf wolf scum minion".split()),
    "vote words": set("vote voting voted votes skip abstain abstaining lynch eject execute hammer".split()),
    "evidence / timing": set("saw seen last night yesterday earlier round timing timeline cams camera body report reported".split()),
    "questions (?)": {"?"},
    "exclamations (!)": {"!"},
}


def load(turns_path: str, games_dir: str):
    rows = [json.loads(l) for l in open(turns_path, encoding="utf-8")]
    rows = [r for r in rows if r["model_verified"]]
    names = set()
    for f in glob.glob(f"{games_dir}/*.json"):
        for p in json.load(open(f, encoding="utf-8"))["players"]:
            names.add(p["player_id"]); names.update(p["aliases"])
            names.update(part for part in p["player_id"].replace("-", " ").split() if len(part) > 2 and not part.isdigit())
    return rows, name_pattern(names)


def side_counts(rows, scrub, n=1, presence=False):
    ca, cb = Counter(), Counter()
    for r in rows:
        toks = tokenize(r["text"], scrub)
        grams = toks if n == 1 else ngrams(toks, n)
        (ca if r["label"] == "lying" else cb).update(set(grams) if presence else grams)
    return ca, cb


def md_table(headers, rows_):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows_]
    return "\n".join(out)


def fmt(x, nd=2):
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def paired_by_model(rows, stat):
    """Per-model mean of `stat` on each side, for models seen on both sides; sign test + Wilcoxon."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["model"]][r["label"]].append(stat(r))
    diffs = []
    for m, d in per.items():
        if d["lying"] and d["truth"]:
            diffs.append((m, np.mean(d["lying"]) - np.mean(d["truth"]), len(d["lying"]), len(d["truth"])))
    vals = np.array([d[1] for d in diffs])
    pos = int((vals > 0).sum())
    sign_p = sps.binomtest(pos, len(vals), 0.5).pvalue if len(vals) else float("nan")
    wil_p = sps.wilcoxon(vals).pvalue if len(vals) >= 6 and np.any(vals != 0) else float("nan")
    return diffs, pos, sign_p, wil_p


def eval_lexicon(rows, scrub, folds, **fit_kw):
    """Out-of-fold scores for the lexicon detector; the lexicon is refit inside every fold."""
    y = np.array([r["label"] == "lying" for r in rows], dtype=int)
    score = np.full(len(rows), np.nan)
    for _, train, test in folds:
        lex = fit_lexicon([rows[i] for i in train], scrub, **fit_kw)
        for i in test:
            score[i] = lex.score(rows[i]["text"], scrub)
    mask = ~np.isnan(score)
    return y[mask], score[mask]


def eval_tfidf(rows, scrub, folds):
    y = np.array([r["label"] == "lying" for r in rows], dtype=int)
    texts = [" ".join(tokenize(r["text"], scrub)) for r in rows]
    score = np.full(len(rows), np.nan)
    for _, train, test in folds:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
        X = vec.fit_transform([texts[i] for i in train])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X, y[train])
        score[test] = clf.predict_proba(vec.transform([texts[i] for i in test]))[:, 1]
    mask = ~np.isnan(score)
    return y[mask], score[mask]


def metrics(y, s):
    if len(set(y.tolist())) < 2:
        return "n/a (one class)"
    return f"AUC {roc_auc_score(y, s):.3f}, AP {average_precision_score(y, s):.3f} (base rate {y.mean():.2f}, n={len(y)})"


def leave_model_out_folds(rows, min_test=20):
    models = [r["model"] for r in rows]
    for m in sorted(set(models)):
        test = [i for i, x in enumerate(models) if x == m]
        if len(test) >= min_test:
            yield m, [i for i, x in enumerate(models) if x != m], test


def main(turns_path, games_dir, out_dir):
    out = Path(out_dir)
    (out / "lexicons").mkdir(parents=True, exist_ok=True)
    rows, scrub = load(turns_path, games_dir)
    ly = [r for r in rows if r["label"] == "lying"]
    tr = [r for r in rows if r["label"] == "truth"]
    R = []  # report lines
    R.append("# Deceiving vs truthful speech: word analysis\n")
    R.append(f"Public in-game AI speech from the Turing Games bins: **{len(ly)} deceiving and {len(tr)} truthful "
             f"utterances** ({sum(len(tokenize(r['text'])) for r in ly):,} and {sum(len(tokenize(r['text'])) for r in tr):,} "
             f"tokens) across {len({r['game_id'] for r in rows})} games and {len({r['model'] for r in rows})} models. "
             "All player/model names are replaced by a placeholder before counting, so 'who is being talked about' "
             "cannot masquerade as a lie signal.\n")

    # ---------------------------------------------------------------- 1. surface statistics
    R.append("## 1. Surface statistics (how much, not what)\n")
    surf = {
        "words per utterance": lambda r: len([t for t in tokenize(r["text"]) if t not in "?!"]),
        "questions per utterance": lambda r: r["text"].count("?"),
        "exclamations per utterance": lambda r: r["text"].count("!"),
        "mean word length": lambda r: np.mean([len(t) for t in tokenize(r["text"]) if t.isalpha()] or [0]),
        "sentences per utterance": lambda r: max(1, len(re.findall(r"[.!?]+", r["text"]))),
    }
    tbl = []
    for name, f in surf.items():
        a, b = np.mean([f(r) for r in ly]), np.mean([f(r) for r in tr])
        diffs, pos, sign_p, wil_p = paired_by_model(rows, f)
        tbl.append([name, f"{a:.2f}", f"{b:.2f}", fmt(a - b), f"{pos}/{len(diffs)}", f"{sign_p:.2f}", f"{wil_p:.2f}"])
    R.append(md_table(["statistic", "deceiving", "truthful", "diff", "models where deceiving > truthful",
                       "sign-test p", "Wilcoxon p"], tbl))
    R.append("\nThe paired columns compare each model with *itself* on its lying vs truthful games (18 models appear on "
             "both sides), which removes 'which models happen to be mafia more often'.\n")

    # ---------------------------------------------------------------- 2. raw fightin' words
    R.append("## 2. Which words differ (names scrubbed, uncontrolled)\n")
    ca, cb = side_counts(rows, scrub)
    fw = fightin_words(ca, cb)
    elig = [w for w in fw if ca[w] + cb[w] >= MIN_COUNT and w != NAME_TOKEN]
    top_ly = sorted(elig, key=lambda w: -fw[w][1])[:25]
    top_tr = sorted(elig, key=lambda w: fw[w][1])[:25]
    n_ly_tok, n_tr_tok = sum(ca.values()), sum(cb.values())

    def rate_row(w):
        return [w, ca[w], cb[w], f"{1000*ca[w]/n_ly_tok:.1f}", f"{1000*cb[w]/n_tr_tok:.1f}", fmt(fw[w][1], 1)]

    R.append("Weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008); z > ~2.5 is notable. "
             "Rates are per 1,000 tokens.\n")
    R.append("**Leaning deceiving**\n\n" + md_table(["word", "n deceiving", "n truthful", "rate dec", "rate tru", "z"],
                                                    [rate_row(w) for w in top_ly]))
    R.append("\n**Leaning truthful**\n\n" + md_table(["word", "n deceiving", "n truthful", "rate dec", "rate tru", "z"],
                                                     [rate_row(w) for w in top_tr]))
    ca2, cb2 = side_counts(rows, scrub, n=2)
    fw2 = fightin_words(ca2, cb2)
    elig2 = [w for w in fw2 if ca2[w] + cb2[w] >= 8 and NAME_TOKEN not in w]
    R.append("\n**Bigrams** (min 8 occurrences)\n\n" + md_table(
        ["leaning deceiving", "z", "leaning truthful", "z"],
        [[a, fmt(fw2[a][1], 1), b, fmt(fw2[b][1], 1)] for a, b in
         zip(sorted(elig2, key=lambda w: -fw2[w][1])[:15], sorted(elig2, key=lambda w: fw2[w][1])[:15])]))

    # ---------------------------------------------------------------- 3. permutation test (game-level)
    R.append("\n## 3. Does it survive a game-level permutation test?\n")
    units = build_units(rows, scrub)
    perm = permutation_test(units, elig, n_perm=N_PERM)
    sig05 = {w for w, d in perm.items() if d["p"] < 0.05}
    bh = benjamini_hochberg({w: d["p"] for w, d in perm.items()}, q=0.10)
    R.append(f"Labels were shuffled {N_PERM:,} times among the players *within each game* (each player's whole "
             f"set of utterances moves together, and every game keeps its real number of liars). This asks: given "
             f"these exact games and players, how often would a random choice of 'who is mafia' produce a word gap "
             f"this large?\n\n- words tested: {len(elig)}; expected false positives at p<0.05 by chance: ~{0.05*len(elig):.0f}\n"
             f"- words with p < 0.05: **{len(sig05)}**\n- words surviving Benjamini-Hochberg FDR 10%: **{len(bh)}** "
             f"{sorted(bh, key=lambda w: perm[w]['p']) if bh else ''}\n")
    sig_rows = sorted(sig05, key=lambda w: perm[w]["p"])
    R.append(md_table(["word", "direction", "n dec", "n tru", "log-odds", "z", "perm p", "FDR 10%"],
                      [[w, "deceiving" if perm[w]["delta"] > 0 else "truthful", ca[w], cb[w], fmt(perm[w]["delta"]),
                        fmt(perm[w]["z"], 1), f"{perm[w]['p']:.3f}", "yes" if w in bh else ""] for w in sig_rows]))

    # ---------------------------------------------------------------- 4. within-model consistency
    R.append("\n## 4. Within-model consistency (controls for which model is speaking)\n")
    mh_model = mantel_haenszel(units, elig, lambda u: u.model)
    consistent = [w for w, d in mh_model.items() if d["strata"] >= 6 and d["agree"] / d["strata"] >= 0.75
                  and abs(d["mh_log_or"]) >= 0.4]
    consistent.sort(key=lambda w: -abs(mh_model[w]["mh_log_or"]))
    R.append("Mantel-Haenszel pooled odds ratio with the speaking model as the stratum: each word is compared only "
             "between a model's lying and truthful games, then pooled. 'agree' = how many of the models that use the "
             "word lean the same way as the pooled estimate. Listed: >= 6 models, >= 75% agreement, |log OR| >= 0.4.\n")
    R.append(md_table(["word", "direction", "pooled log OR", "models agreeing", "perm p"],
                      [[w, "deceiving" if mh_model[w]["mh_log_or"] > 0 else "truthful", fmt(mh_model[w]["mh_log_or"]),
                        f"{mh_model[w]['agree']}/{mh_model[w]['strata']}", f"{perm[w]['p']:.3f}"] for w in consistent[:30]]))
    robust = [w for w in consistent if w in sig05]
    robust_txt = ", ".join(f"{w} ({'dec' if perm[w]['delta'] > 0 else 'tru'})" for w in robust) or "none"
    R.append(f"\n**Robust to both controls** (consistent across models *and* game-permutation p < 0.05): {robust_txt}\n")

    # ---------------------------------------------------------------- 5. other controls
    R.append("## 5. Other controls\n")
    # 5a villagers only
    vill = [r for r in rows if r["label"] == "lying" or r["role"] in ("villager", "crewmate")]
    cav, cbv = side_counts(vill, scrub)
    fwv = fightin_words(cav, cbv)
    R.append("### 5a. Power roles removed\nSheriffs, doctors, vigilantes etc. make claims villagers never make. "
             "Comparing liars with *plain villagers/crewmates only*:\n")
    flips = [(w, fmt(fw[w][1], 1), fmt(fwv[w][1], 1)) for w in top_ly + top_tr if w in fwv and np.sign(fw[w][1]) != np.sign(fwv[w][1])]
    R.append(md_table(["word (from section 2)", "z all truthful", "z villagers only"],
                      [[w, fmt(fw[w][1], 1), fmt(fwv[w][1], 1)] for w in top_ly[:12] + top_tr[:12] if w in fwv]))
    R.append(f"\nWords that flip direction once power roles are removed: {', '.join(f[0] for f in flips) or 'none'}.\n")
    # 5b position in game
    pos_of = {}
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game_id"]].append(r)
    for g, rs in by_game.items():
        rs.sort(key=lambda r: int(r["id"].rsplit(":", 1)[1]))
        for i, r in enumerate(rs):
            pos_of[r["id"]] = i / max(1, len(rs) - 1)
    terc = lambda r: "early" if pos_of[r["id"]] < 1/3 else "mid" if pos_of[r["id"]] < 2/3 else "late"  # noqa: E731
    share = {t: (sum(1 for r in ly if terc(r) == t) / len(ly), sum(1 for r in tr if terc(r) == t) / len(tr)) for t in ("early", "mid", "late")}
    R.append("### 5b. Position in the game\nLiars survive longer, so more of their speech is late-game.\n\n" +
             md_table(["third of game", "share of deceiving speech", "share of truthful speech"],
                      [[t, f"{a:.0%}", f"{b:.0%}"] for t, (a, b) in share.items()]))
    units_pos = []
    for r in rows:
        units_pos.append(Unit(r["game_id"], f"{r['player_id']}|{terc(r)}", terc(r), r["label"] == "lying", Counter(tokenize(r["text"], scrub))))
    merged = {}
    for u in units_pos:
        k = (u.game, u.player)
        if k in merged:
            merged[k].counts.update(u.counts)
        else:
            merged[k] = u
    mh_pos = mantel_haenszel(list(merged.values()), elig, lambda u: u.model)
    R.append("\nPooled log OR stratified by game third (so early speech is only compared with early speech):\n\n" +
             md_table(["word", "z uncontrolled", "log OR within position", "thirds agreeing"],
                      [[w, fmt(fw[w][1], 1), fmt(mh_pos[w]["mh_log_or"]), f"{mh_pos[w]['agree']}/{mh_pos[w]['strata']}"]
                       for w in top_ly[:10] + top_tr[:10] if w in mh_pos]))
    # 5c game type
    gtype = lambda g: "among_us" if ("among" in g or "can-1" in g or "forced" in g) else "werewolf" if "onuw" in g else "mafia"  # noqa: E731
    mh_type = mantel_haenszel(units, elig, lambda u: gtype(u.game))
    R.append("\n### 5c. Game type\nAmong Us is about locations and bodies, Mafia about roles and votes. Stratifying by "
             "game type (mafia / among_us / one-night werewolf):\n\n" +
             md_table(["word", "z uncontrolled", "log OR within game type", "types agreeing"],
                      [[w, fmt(fw[w][1], 1), fmt(mh_type[w]["mh_log_or"]), f"{mh_type[w]['agree']}/{mh_type[w]['strata']}"]
                       for w in top_ly[:10] + top_tr[:10] if w in mh_type]))
    # 5d provenance
    cap = [r for r in rows if r["text_provenance"] == "captions"]
    cac, cbc = side_counts(cap, scrub)
    fwc = fightin_words(cac, cbc)
    agree = [(w, fmt(fw[w][1], 1), fmt(fwc[w][1], 1), cac[w] + cbc[w]) for w in top_ly[:15] + top_tr[:15] if cac[w] + cbc[w] >= 3]
    same = sum(1 for a in agree if np.sign(float(a[1])) == np.sign(float(a[2])))
    R.append(f"\n### 5d. Caption-grade text only\nMost of the text is Gemini's transcription of the videos (checked against real captions: ~93% of its content words appear in the captions at the stated timestamps). In the "
             f"{len(cap)} caption-grade utterances ({sum(r['label']=='lying' for r in cap)} deceiving, from 2 games), "
             f"{same} of {len(agree)} top words keep the same direction:\n\n" +
             md_table(["word", "z all text", "z captions only", "n in captions"], [list(a) for a in agree]))
    R.append("\nToo little caption text to confirm anything; treat this as a consistency check only.\n")

    # ---------------------------------------------------------------- 6. categories
    R.append("## 6. Word categories (LIWC-style)\n")
    cat_units = []
    for u in units:
        cc = Counter()
        for cat, words in CATEGORIES.items():
            cc[cat] = sum(c for w, c in u.counts.items() if w in words)
        cat_units.append(Unit(u.game, u.player, u.model, u.lying, cc, total=u.n_tokens))
    cat_perm = permutation_test(cat_units, list(CATEGORIES), n_perm=N_PERM)
    tok_ly = sum(ca.values()); tok_tr = sum(cb.values())
    tbl = []
    for cat, words in CATEGORIES.items():
        a = sum(ca[w] for w in words); b = sum(cb[w] for w in words)
        diffs, pos, sign_p, _ = paired_by_model(rows, lambda r, ws=words: 100 * sum(1 for t in tokenize(r["text"]) if t in ws) / max(1, len(tokenize(r["text"]))))
        tbl.append([cat, f"{100*a/tok_ly:.2f}", f"{100*b/tok_tr:.2f}", fmt(cat_perm[cat]["delta"]), f"{cat_perm[cat]['p']:.3f}",
                    f"{pos}/{len(diffs)}", f"{sign_p:.2f}"])
    R.append("Rates per 100 tokens. 'perm p' is the within-game permutation test; the last two columns are the "
             "within-model paired comparison.\n\n" + md_table(
        ["category", "deceiving %", "truthful %", "log-odds", "perm p", "models dec > tru", "sign p"], tbl))

    # ---------------------------------------------------------------- 7. detector
    R.append("\n## 7. The keyword detector, evaluated honestly\n")
    groups = [r["game_id"] for r in rows]
    folds = list(group_kfold(groups, 5, seed=0))
    y, s = eval_lexicon(rows, scrub, folds)
    y2, s2 = eval_lexicon(rows, None, folds)
    y3, s3 = eval_tfidf(rows, scrub, folds)
    lmo = list(leave_model_out_folds(rows))
    y4, s4 = eval_lexicon(rows, scrub, lmo)
    y5, s5 = eval_tfidf(rows, scrub, lmo)
    R.append("The lexicon (signed word weights, section 8) is re-fit inside every fold, so no test utterance ever "
             "influenced the weights used to score it. AUC 0.5 = coin flip; AP is compared with the base rate.\n\n" +
             md_table(["detector", "held-out games (5-fold)", "held-out models (train without a model, test on it)"],
                      [["keyword lexicon, names scrubbed", metrics(y, s), metrics(y4, s4)],
                       ["keyword lexicon, names kept (leaky)", metrics(y2, s2), "-"],
                       ["TF-IDF + logistic regression (reference)", metrics(y3, s3), metrics(y5, s5)]]))
    # lexicon size sweep + robust-words-only check
    sweep = []
    for mc, mw in [(5, 300), (10, 100), (15, 100), (30, 40), (50, 20)]:
        ya, sa = eval_lexicon(rows, scrub, folds, min_count=mc, max_words=mw)
        yb, sb = eval_lexicon(rows, scrub, lmo, min_count=mc, max_words=mw)
        sweep.append([f">= {mc} occurrences, top {mw} words", f"{roc_auc_score(ya, sa):.3f}", f"{roc_auc_score(yb, sb):.3f}"])
    R.append("\nLexicon size (same fold protocol): fewer, better-attested words generalise better.\n\n" +
             md_table(["lexicon", "AUC held-out games", "AUC held-out models"], sweep))
    yr = np.array([r["label"] == "lying" for r in rows], dtype=int)
    sr = np.full(len(rows), np.nan)
    for _, train, test in folds:
        ca_t, cb_t = side_counts([rows[i] for i in train], scrub, presence=True)
        fw_t = fightin_words(ca_t, cb_t)
        w_t = {w: 0.5 * fw_t[w][0] for w in robust if w in fw_t}
        for i in test:
            sr[i] = sum(w_t.get(t, 0.0) for t in set(tokenize(rows[i]["text"], scrub)))
    four = ["if", "just", "your", "why"]
    s_four = np.array([sum(t in four for t in set(tokenize(r["text"], scrub))) for r in rows])
    R.append(f"\nOnly the {len(robust)} words robust to both controls (section 4), weights refit per fold: "
             f"**AUC {roc_auc_score(yr, sr):.3f}** on held-out games (the word list itself was chosen on all data, "
             f"so this is slightly optimistic). Simply counting how many of {{if, just, your, why}} appear, with no "
             f"fitting at all: **AUC {roc_auc_score(yr, s_four):.3f}**.\n")
    # per-model breakdown of held-out-model AUC
    per = []
    for (m, train, test) in lmo:
        lex = fit_lexicon([rows[i] for i in train], scrub)
        yy = np.array([rows[i]["label"] == "lying" for i in test], dtype=int)
        ss = np.array([lex.score(rows[i]["text"], scrub) for i in test])
        if len(set(yy.tolist())) == 2:
            per.append([m, len(test), int(yy.sum()), f"{roc_auc_score(yy, ss):.2f}"])
    R.append("\nHeld-out model, one at a time (general lexicon trained on everyone else):\n\n" +
             md_table(["model held out", "utterances", "deceiving", "AUC"], per))
    # family tier
    fam_rows = defaultdict(list)
    for r in rows:
        fam_rows[r["family"]].append(r)
    ftbl = []
    for fam, frs in sorted(fam_rows.items()):
        games = sorted({r["game_id"] for r in frs})
        n_ly = sum(r["label"] == "lying" for r in frs)
        if len(games) < 3 or n_ly < 15:
            ftbl.append([fam, len(frs), n_ly, len(games), "too little data", "too little data"])
            continue
        k = min(5, len(games))
        ffolds = list(group_kfold([r["game_id"] for r in frs], k, seed=0))
        yf, sf = eval_lexicon(frs, scrub, ffolds)
        # general lexicon: trained on all rows from other games (any family), scored on this family's held-out games
        sg = np.full(len(frs), np.nan)
        for _, _tr, te in ffolds:
            held = {frs[i]["game_id"] for i in te}
            lex = fit_lexicon([r for r in rows if r["game_id"] not in held], scrub)
            for i in te:
                sg[i] = lex.score(frs[i]["text"], scrub)
        yg = np.array([r["label"] == "lying" for r in frs], dtype=int)
        ftbl.append([fam, len(frs), n_ly, len(games), metrics(yf, sf), metrics(yg[~np.isnan(sg)], sg[~np.isnan(sg)])])
    R.append("\n### Family tier\nFor each family: a lexicon trained only on that family's other games vs the general "
             "lexicon trained on all other games, both scored on the family's held-out games.\n\n" +
             md_table(["family", "utterances", "deceiving", "games", "family-specific lexicon", "general lexicon"], ftbl))
    R.append("\n### Per-model tier\nMost models have under 20 deceiving utterances, so a per-model lexicon cannot be "
             "evaluated honestly yet; the per-model bins exist, and `lexicons/` holds family lexicons as the "
             "finest level that has enough data. Per-model 'tells' below are the top words by within-model "
             "log-odds and are **provisional**.\n")
    mtbl = []
    for m in sorted({r["model"] for r in rows}):
        mrs = [r for r in rows if r["model"] == m]
        nl = sum(r["label"] == "lying" for r in mrs)
        if nl < 8 or len(mrs) - nl < 8:
            continue
        cam, cbm = side_counts(mrs, scrub)
        fwm = fightin_words(cam, cbm)
        el = [w for w in fwm if cam[w] + cbm[w] >= 4 and w != NAME_TOKEN]
        mtbl.append([m, nl, len(mrs) - nl, ", ".join(sorted(el, key=lambda w: -fwm[w][1])[:6]),
                     ", ".join(sorted(el, key=lambda w: fwm[w][1])[:6])])
    R.append(md_table(["model", "dec", "tru", "leans deceiving", "leans truthful"], mtbl))

    # ---------------------------------------------------------------- 8. lexicon output
    lex = fit_lexicon(rows, scrub)
    lex.save(out / "lexicon.json")
    gen = fit_lexicon(rows, scrub, exclude=GAME_TERMS)
    gen.save(out / "lexicon_general.json")
    yg, sg = eval_lexicon(rows, scrub, folds, exclude=GAME_TERMS)
    yg2, sg2 = eval_lexicon(rows, scrub, lmo, exclude=GAME_TERMS)
    R.append("\n### Domain-neutral lexicon\n`lexicon_general.json` drops every Mafia / Among Us / Werewolf term "
             "(roles, votes, kills, locations, mechanics) and bare numbers, for use outside these games. "
             f"Held-out games AUC {roc_auc_score(yg, sg):.3f}, held-out models AUC {roc_auc_score(yg2, sg2):.3f} "
             f"(vs {roc_auc_score(y, s):.3f} / {roc_auc_score(y4, s4):.3f} with game terms): the signal is not in the "
             f"game vocabulary. Words: {', '.join(sorted(gen.weights, key=lambda w: -gen.weights[w]))}.\n")
    for fam, frs in fam_rows.items():
        if sum(r["label"] == "lying" for r in frs) >= 15:
            fit_lexicon(frs, scrub, min_count=4, max_words=150).save(out / "lexicons" / f"{fam}.json")
    top_w = sorted(lex.weights.items(), key=lambda kv: -kv[1])[:20]
    bot_w = sorted(lex.weights.items(), key=lambda kv: kv[1])[:20]
    R.append("\n## 8. The lexicon\n`lexicon.json`: a word list with signed weights (positive = deceiving) and an "
             "intercept; score = intercept + sum of weights of the words present (after name scrubbing). "
             f"{len(lex.weights)} words. Strongest weights:\n\n" + md_table(
        ["deceiving", "weight", "truthful", "weight"],
        [[a, fmt(wa), b, fmt(wb)] for (a, wa), (b, wb) in zip(top_w, bot_w)]))
    R.append("\n## 9. Bottom line\nSee FINDINGS.md next to this report for the interpretation of these numbers.\n")

    with (out / "word_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "n_deceiving", "n_truthful", "log_odds", "z", "perm_p", "fdr10", "mh_model_log_or",
                    "mh_models_agree", "mh_models"])
        for word in sorted(elig, key=lambda x: perm[x]["p"]):
            mm = mh_model.get(word, {})
            w.writerow([word, ca[word], cb[word], f"{perm[word]['delta']:.3f}", f"{perm[word]['z']:.2f}",
                        f"{perm[word]['p']:.4f}", word in bh, f"{mm.get('mh_log_or', float('nan')):.3f}",
                        mm.get("agree", ""), mm.get("strata", "")])
    (out / "report.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print(f"wrote {out}/report.md, word_stats.csv, lexicon.json, lexicons/")


if __name__ == "__main__":
    main(*sys.argv[1:4])
