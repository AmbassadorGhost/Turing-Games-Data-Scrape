# Deceiving vs truthful speech: word analysis

Public in-game AI speech from the Turing Games bins: **104 deceiving and 221 truthful utterances** (3,984 and 8,995 tokens) across 7 games and 15 models. All player/model names are replaced by a placeholder before counting, so 'who is being talked about' cannot masquerade as a lie signal.

## 1. Surface statistics (how much, not what)

| statistic | deceiving | truthful | diff | models where deceiving > truthful | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|---|
| words per utterance | 38.12 | 40.52 | -2.40 | 3/8 | 0.73 | 0.25 |
| questions per utterance | 0.18 | 0.18 | +0.01 | 1/8 | 0.07 | 0.30 |
| exclamations per utterance | 0.00 | 0.00 | +0.00 | 0/8 | 0.01 | nan |
| mean word length | 4.53 | 4.51 | +0.02 | 3/8 | 0.73 | 0.95 |
| sentences per utterance | 4.00 | 4.34 | -0.34 | 2/8 | 0.29 | 0.05 |

The paired columns compare each model with *itself* on its lying vs truthful games (18 models appear on both sides), which removes 'which models happen to be mafia more often'.

## 2. Which words differ (names scrubbed, uncontrolled)

Weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008); z > ~2.5 is notable. Rates are per 1,000 tokens.

**Leaning deceiving**

| word | n deceiving | n truthful | rate dec | rate tru | z |
|---|---|---|---|---|---|
| jester | 14 | 4 | 3.7 | 0.5 | +3.4 |
| your | 19 | 13 | 5.0 | 1.5 | +3.0 |
| if | 22 | 23 | 5.8 | 2.7 | +2.3 |
| sense | 8 | 4 | 2.1 | 0.5 | +2.3 |
| how | 13 | 11 | 3.4 | 1.3 | +2.2 |
| let's | 13 | 11 | 3.4 | 1.3 | +2.2 |
| too | 10 | 7 | 2.6 | 0.8 | +2.2 |
| agree | 10 | 7 | 2.6 | 0.8 | +2.2 |
| exactly | 9 | 6 | 2.4 | 0.7 | +2.1 |
| just | 26 | 32 | 6.8 | 3.7 | +2.1 |
| reads | 7 | 4 | 1.8 | 0.5 | +2.0 |
| body | 9 | 7 | 2.4 | 0.8 | +1.9 |
| what | 8 | 6 | 2.1 | 0.7 | +1.8 |
| focus | 7 | 5 | 1.8 | 0.6 | +1.8 |
| after | 18 | 22 | 4.7 | 2.6 | +1.7 |
| makes | 13 | 14 | 3.4 | 1.6 | +1.7 |
| keep | 6 | 4 | 1.6 | 0.5 | +1.7 |
| there | 11 | 11 | 2.9 | 1.3 | +1.7 |
| why | 11 | 11 | 2.9 | 1.3 | +1.7 |
| see | 11 | 11 | 2.9 | 1.3 | +1.7 |
| from | 22 | 29 | 5.8 | 3.4 | +1.7 |
| wagon | 8 | 7 | 2.1 | 0.8 | +1.6 |
| it's | 13 | 15 | 3.4 | 1.7 | +1.6 |
| instead | 7 | 6 | 1.8 | 0.7 | +1.6 |
| here | 6 | 5 | 1.6 | 0.6 | +1.5 |

**Leaning truthful**

| word | n deceiving | n truthful | rate dec | rate tru | z |
|---|---|---|---|---|---|
| mafia | 30 | 148 | 7.9 | 17.2 | -3.6 |
| was | 16 | 84 | 4.2 | 9.8 | -2.8 |
| town | 9 | 58 | 2.4 | 6.8 | -2.7 |
| night | 5 | 37 | 1.3 | 4.3 | -2.3 |
| then | 1 | 19 | 0.3 | 2.2 | -2.1 |
| us | 5 | 32 | 1.3 | 3.7 | -2.0 |
| confirmed | 4 | 28 | 1.0 | 3.3 | -1.9 |
| both | 1 | 17 | 0.3 | 2.0 | -1.9 |
| were | 5 | 31 | 1.3 | 3.6 | -1.9 |
| trying | 7 | 37 | 1.8 | 4.3 | -1.9 |
| our | 4 | 26 | 1.0 | 3.0 | -1.8 |
| member | 1 | 15 | 0.3 | 1.7 | -1.8 |
| targeted | 0 | 10 | 0.0 | 1.2 | -1.7 |
| left | 4 | 24 | 1.0 | 2.8 | -1.6 |
| at | 5 | 27 | 1.3 | 3.1 | -1.6 |
| who | 12 | 48 | 3.1 | 5.6 | -1.6 |
| they | 13 | 50 | 3.4 | 5.8 | -1.5 |
| investigated | 1 | 12 | 0.3 | 1.4 | -1.5 |
| abstaining | 1 | 12 | 0.3 | 1.4 | -1.5 |
| doctor | 5 | 25 | 1.3 | 2.9 | -1.5 |
| being | 2 | 15 | 0.5 | 1.7 | -1.5 |
| look | 2 | 15 | 0.5 | 1.7 | -1.5 |
| day | 9 | 36 | 2.4 | 4.2 | -1.4 |
| miss | 1 | 10 | 0.3 | 1.2 | -1.3 |
| logic | 1 | 10 | 0.3 | 1.2 | -1.3 |

**Bigrams** (min 8 occurrences)

| leaning deceiving | z | leaning truthful | z |
|---|---|---|---|
| the one | +2.2 | a mafia | -2.0 |
| vote on | +1.8 | mafia member | -1.8 |
| i agree | +1.8 | trying to | -1.8 |
| agree with | +1.6 | was the | -1.6 |
| to be | +1.5 | town and | -1.6 |
| are you | +1.2 | on the | -1.6 |
| my vote | +1.0 | i voted | -1.5 |
| abstain for | +1.0 | you were | -1.4 |
| as i | +1.0 | i was | -1.4 |
| we need | +1.0 | last night | -1.3 |
| as the | +0.9 | day one | -1.3 |
| instead of | +0.9 | the remaining | -1.2 |
| i have | +0.9 | with me | -1.2 |
| the body | +0.8 | vote for | -1.2 |
| to see | +0.8 | not mafia | -1.1 |

## 3. Does it survive a game-level permutation test?

Labels were shuffled 3,000 times among the players *within each game* (each player's whole set of utterances moves together, and every game keeps its real number of liars). This asks: given these exact games and players, how often would a random choice of 'who is mafia' produce a word gap this large?

- words tested: 217; expected false positives at p<0.05 by chance: ~11
- words with p < 0.05: **18**
- words surviving Benjamini-Hochberg FDR 10%: **0** 

| word | direction | n dec | n tru | log-odds | z | perm p | FDR 10% |
|---|---|---|---|---|---|---|---|
| mafia | truthful | 30 | 148 | -0.59 | -3.6 | 0.002 |  |
| town | truthful | 9 | 58 | -0.75 | -2.7 | 0.002 |  |
| your | deceiving | 19 | 13 | +0.98 | +3.0 | 0.009 |  |
| were | truthful | 5 | 31 | -0.73 | -1.9 | 0.010 |  |
| agree | deceiving | 10 | 7 | +0.96 | +2.2 | 0.010 |  |
| how | deceiving | 13 | 11 | +0.81 | +2.2 | 0.011 |  |
| then | truthful | 1 | 19 | -1.28 | -2.1 | 0.011 |  |
| was | truthful | 16 | 84 | -0.62 | -2.8 | 0.013 |  |
| us | truthful | 5 | 32 | -0.75 | -2.0 | 0.018 |  |
| member | truthful | 1 | 15 | -1.19 | -1.8 | 0.021 |  |
| if | deceiving | 22 | 23 | +0.63 | +2.3 | 0.022 |  |
| instead | deceiving | 7 | 6 | +0.79 | +1.6 | 0.027 |  |
| both | truthful | 1 | 17 | -1.24 | -1.9 | 0.030 |  |
| there | deceiving | 11 | 11 | +0.67 | +1.7 | 0.030 |  |
| jester | deceiving | 14 | 4 | +1.67 | +3.4 | 0.034 |  |
| sense | deceiving | 8 | 4 | +1.24 | +2.3 | 0.037 |  |
| trying | truthful | 7 | 37 | -0.63 | -1.9 | 0.042 |  |
| night | truthful | 5 | 37 | -0.84 | -2.3 | 0.048 |  |

## 4. Within-model consistency (controls for which model is speaking)

Mantel-Haenszel pooled odds ratio with the speaking model as the stratum: each word is compared only between a model's lying and truthful games, then pooled. 'agree' = how many of the models that use the word lean the same way as the pooled estimate. Listed: >= 6 models, >= 75% agreement, |log OR| >= 0.4.

| word | direction | pooled log OR | models agreeing | perm p |
|---|---|---|---|---|
| jester | deceiving | +1.90 | 5/6 | 0.034 |
| your | deceiving | +1.12 | 5/6 | 0.009 |
| vigilante | deceiving | +1.00 | 5/6 | 0.390 |
| seems | deceiving | +0.92 | 5/6 | 0.127 |
| mafia | truthful | -0.86 | 7/8 | 0.002 |
| left | truthful | -0.67 | 6/6 | 0.081 |
| was | truthful | -0.67 | 6/8 | 0.013 |
| need | deceiving | +0.66 | 6/7 | 0.977 |
| after | deceiving | +0.65 | 6/7 | 0.084 |
| kill | truthful | -0.65 | 5/6 | 0.144 |
| were | truthful | -0.64 | 5/6 | 0.010 |
| my | truthful | -0.62 | 5/6 | 0.441 |
| it | deceiving | +0.58 | 6/8 | 0.423 |
| no | deceiving | +0.56 | 6/7 | 0.090 |
| at | truthful | -0.52 | 6/6 | 0.051 |
| real | deceiving | +0.52 | 6/6 | 0.833 |
| town | truthful | -0.52 | 6/7 | 0.002 |
| without | deceiving | +0.49 | 5/6 | 0.525 |
| us | truthful | -0.48 | 6/7 | 0.018 |
| just | deceiving | +0.44 | 6/7 | 0.060 |

**Robust to both controls** (consistent across models *and* game-permutation p < 0.05): jester (dec), your (dec), mafia (tru), was (tru), were (tru), town (tru), us (tru)

## 5. Other controls

### 5a. Power roles removed
Sheriffs, doctors, vigilantes etc. make claims villagers never make. Comparing liars with *plain villagers/crewmates only*:

| word (from section 2) | z all truthful | z villagers only |
|---|---|---|
| jester | +3.4 | +3.2 |
| your | +3.0 | +2.7 |
| if | +2.3 | +2.1 |
| sense | +2.3 | +1.8 |
| how | +2.2 | +1.9 |
| let's | +2.2 | +2.1 |
| too | +2.2 | +1.6 |
| agree | +2.2 | +1.6 |
| exactly | +2.1 | +1.8 |
| just | +2.1 | +2.2 |
| reads | +2.0 | +2.1 |
| body | +1.9 | +1.4 |
| mafia | -3.6 | -2.7 |
| was | -2.8 | -3.1 |
| town | -2.7 | -2.6 |
| night | -2.3 | -1.1 |
| then | -2.1 | -2.4 |
| us | -2.0 | -2.1 |
| confirmed | -1.9 | -0.6 |
| both | -1.9 | -2.2 |
| were | -1.9 | -2.1 |
| trying | -1.9 | -1.9 |
| our | -1.8 | -1.7 |
| member | -1.8 | -1.4 |

Words that flip direction once power roles are removed: none.

### 5b. Position in the game
Liars survive longer, so more of their speech is late-game.

| third of game | share of deceiving speech | share of truthful speech |
|---|---|---|
| early | 36% | 33% |
| mid | 28% | 34% |
| late | 37% | 33% |

Pooled log OR stratified by game third (so early speech is only compared with early speech):

| word | z uncontrolled | log OR within position | thirds agreeing |
|---|---|---|---|
| jester | +3.4 | +1.89 | 3/3 |
| your | +3.0 | +1.17 | 3/3 |
| if | +2.3 | +0.75 | 3/3 |
| sense | +2.3 | +1.29 | 3/3 |
| how | +2.2 | +0.86 | 3/3 |
| let's | +2.2 | +0.95 | 3/3 |
| too | +2.2 | +1.14 | 3/3 |
| agree | +2.2 | +1.02 | 3/3 |
| exactly | +2.1 | +1.22 | 3/3 |
| just | +2.1 | +0.60 | 3/3 |
| mafia | -3.6 | -0.75 | 3/3 |
| was | -2.8 | -0.77 | 3/3 |
| town | -2.7 | -0.89 | 3/3 |
| night | -2.3 | -0.98 | 3/3 |
| then | -2.1 | -1.28 | 3/3 |
| us | -2.0 | -0.86 | 3/3 |
| confirmed | -1.9 | -0.83 | 3/3 |
| both | -1.9 | -1.21 | 3/3 |
| were | -1.9 | -0.74 | 3/3 |
| trying | -1.9 | -0.72 | 3/3 |

### 5c. Game type
Among Us is about locations and bodies, Mafia about roles and votes. Stratifying by game type (mafia / among_us / one-night werewolf):

| word | z uncontrolled | log OR within game type | types agreeing |
|---|---|---|---|
| jester | +3.4 | +2.02 | 1/1 |
| your | +3.0 | +1.19 | 2/2 |
| if | +2.3 | +0.75 | 2/2 |
| sense | +2.3 | +1.37 | 2/2 |
| how | +2.2 | +1.01 | 1/1 |
| let's | +2.2 | +0.98 | 1/2 |
| too | +2.2 | +1.13 | 1/2 |
| agree | +2.2 | +1.18 | 1/1 |
| exactly | +2.1 | +1.22 | 1/1 |
| just | +2.1 | +0.58 | 2/2 |
| mafia | -3.6 | -0.75 | 1/1 |
| was | -2.8 | -0.88 | 2/2 |
| town | -2.7 | -0.98 | 1/1 |
| night | -2.3 | -1.08 | 1/1 |
| then | -2.1 | -1.59 | 2/2 |
| us | -2.0 | -0.90 | 2/2 |
| confirmed | -1.9 | -1.01 | 1/1 |
| both | -1.9 | -1.45 | 2/2 |
| were | -1.9 | -0.91 | 2/2 |
| trying | -1.9 | -0.73 | 2/2 |

### 5d. Caption-grade text only
90% of the text is Gemini's reconstruction of the videos. In the 0 caption-grade utterances (0 deceiving, from 2 games), 0 of 0 top words keep the same direction:

| word | z all text | z captions only | n in captions |
|---|---|---|---|

Too little caption text to confirm anything; treat this as a consistency check only.

## 6. Word categories (LIWC-style)

Rates per 100 tokens. 'perm p' is the within-game permutation test; the last two columns are the within-model paired comparison.

| category | deceiving % | truthful % | log-odds | perm p | models dec > tru | sign p |
|---|---|---|---|---|---|---|
| I / me / my (1st singular) | 4.12 | 4.22 | -0.02 | 0.844 | 3/8 | 0.73 |
| we / us / our (1st plural) | 1.63 | 1.79 | -0.08 | 0.506 | 6/8 | 0.29 |
| you / your (2nd person) | 1.89 | 1.49 | +0.19 | 0.295 | 3/8 | 0.73 |
| they / them (3rd plural) | 1.23 | 1.48 | -0.14 | 0.344 | 4/8 | 1.00 |
| negations | 1.21 | 0.98 | +0.17 | 0.366 | 5/8 | 0.73 |
| hedges / uncertainty | 0.58 | 0.48 | +0.15 | 0.560 | 3/8 | 0.73 |
| certainty / emphasis | 0.63 | 0.62 | +0.02 | 0.955 | 4/8 | 1.00 |
| agreement / alignment | 1.52 | 0.86 | +0.47 | 0.012 | 6/8 | 0.29 |
| accusation verbs | 0.60 | 0.70 | -0.12 | 0.567 | 3/8 | 0.73 |
| role claims (own) | 0.97 | 1.20 | -0.17 | 0.311 | 5/8 | 0.73 |
| evil-side words | 0.94 | 1.85 | -0.51 | 0.003 | 0/8 | 0.01 |
| vote words | 2.49 | 2.42 | +0.02 | 0.868 | 3/8 | 0.73 |
| evidence / timing | 1.57 | 1.99 | -0.19 | 0.232 | 2/8 | 0.29 |
| questions (?) | 0.50 | 0.45 | +0.07 | 0.747 | 2/8 | 0.29 |
| exclamations (!) | 0.00 | 0.00 | +nan | 0.000 | 0/8 | 0.01 |

## 7. The keyword detector, evaluated honestly

The lexicon (signed word weights, section 8) is re-fit inside every fold, so no test utterance ever influenced the weights used to score it. AUC 0.5 = coin flip; AP is compared with the base rate.

| detector | held-out games (5-fold) | held-out models (train without a model, test on it) |
|---|---|---|
| keyword lexicon, names scrubbed | AUC 0.528, AP 0.376 (base rate 0.32, n=325) | AUC 0.506, AP 0.335 (base rate 0.33, n=262) |
| keyword lexicon, names kept (leaky) | AUC 0.566, AP 0.394 (base rate 0.32, n=325) | - |
| TF-IDF + logistic regression (reference) | AUC 0.589, AP 0.403 (base rate 0.32, n=325) | AUC 0.528, AP 0.424 (base rate 0.33, n=262) |

Lexicon size (same fold protocol): fewer, better-attested words generalise better.

| lexicon | AUC held-out games | AUC held-out models |
|---|---|---|
| >= 5 occurrences, top 300 words | 0.573 | 0.499 |
| >= 10 occurrences, top 100 words | 0.634 | 0.553 |
| >= 15 occurrences, top 100 words | 0.581 | 0.577 |
| >= 30 occurrences, top 40 words | 0.528 | 0.506 |
| >= 50 occurrences, top 20 words | 0.488 | 0.421 |

Only the 7 words robust to both controls (section 4), weights refit per fold: **AUC 0.717** on held-out games (the word list itself was chosen on all data, so this is slightly optimistic). Simply counting how many of {if, just, your, why} appear, with no fitting at all: **AUC 0.613**.


Held-out model, one at a time (general lexicon trained on everyone else):

| model held out | utterances | deceiving | AUC |
|---|---|---|---|
| claude-sonnet-4.5 | 38 | 7 | 0.61 |
| gemini-3-flash | 70 | 18 | 0.53 |
| gpt-4o | 30 | 12 | 0.55 |
| gpt-5.2 | 20 | 7 | 0.57 |
| grok-4.1 | 40 | 18 | 0.58 |

### Family tier
For each family: a lexicon trained only on that family's other games vs the general lexicon trained on all other games, both scored on the family's held-out games.

| family | utterances | deceiving | games | family-specific lexicon | general lexicon |
|---|---|---|---|---|---|
| anthropic | 49 | 11 | 6 | too little data | too little data |
| deepseek | 20 | 0 | 5 | too little data | too little data |
| google | 108 | 25 | 7 | AUC 0.443, AP 0.206 (base rate 0.23, n=108) | AUC 0.457, AP 0.236 (base rate 0.23, n=108) |
| meta | 17 | 5 | 6 | too little data | too little data |
| moonshot | 15 | 2 | 5 | too little data | too little data |
| openai | 74 | 43 | 6 | AUC 0.171, AP 0.424 (base rate 0.58, n=74) | AUC 0.545, AP 0.613 (base rate 0.58, n=74) |
| xai | 40 | 18 | 5 | AUC 0.000, AP 0.371 (base rate 0.45, n=40) | AUC 0.525, AP 0.557 (base rate 0.45, n=40) |
| zhipu | 2 | 0 | 1 | too little data | too little data |

### Per-model tier
Most models have under 20 deceiving utterances, so a per-model lexicon cannot be evaluated honestly yet; the per-model bins exist, and `lexicons/` holds family lexicons as the finest level that has enough data. Per-model 'tells' below are the top words by within-model log-odds and are **provisional**.

| model | dec | tru | leans deceiving | leans truthful |
|---|---|---|---|---|
| gemini-3-flash | 18 | 52 | if, your, makes, everyone, be, to | and, s, was, mafia, engine, night |
| gpt-4o | 12 | 18 | on, focus, let's, too, votes, so | mafia, vote, in, me, makes, that |
| grok-4.1 | 18 | 22 | lama, is, 4, vote, their, voted | i, like, my, everyone, lynch, confirmed |

### Domain-neutral lexicon
`lexicon_general.json` drops every Mafia / Among Us / Werewolf term (roles, votes, kills, locations, mechanics) and bare numbers, for use outside these games. Held-out games AUC 0.458, held-out models AUC 0.415 (vs 0.528 / 0.506 with game terms): the signal is not in the game vocabulary. Words: if, just, after, from, right, you're, so, not, a, that, with, ?, are, it, this, is, of, but, and, i, i'm, the, for, we, suspicious, one, before, like, two, them, last, have, my, in, they, who, was, trying, were, us.


## 8. The lexicon
`lexicon.json`: a word list with signed weights (positive = deceiving) and an intercept; score = intercept + sum of weights of the words present (after name scrubbing). 40 words. Strongest weights:

| deceiving | weight | truthful | weight |
|---|---|---|---|
| if | +0.32 | night | -0.40 |
| just | +0.24 | us | -0.38 |
| after | +0.23 | town | -0.35 |
| from | +0.16 | were | -0.35 |
| vote | +0.12 | trying | -0.31 |
| right | +0.11 | was | -0.28 |
| you're | +0.09 | mafia | -0.24 |
| so | +0.08 | who | -0.24 |
| not | +0.07 | day | -0.19 |
| a | +0.07 | they | -0.17 |
| that | +0.06 | in | -0.17 |
| with | +0.06 | have | -0.14 |
| ? | +0.05 | my | -0.14 |
| this | +0.04 | last | -0.12 |
| is | +0.04 | them | -0.10 |
| i'm | -0.03 | two | -0.10 |
| the | -0.03 | like | -0.09 |
| for | -0.04 | sheriff | -0.08 |
| we | -0.06 | before | -0.08 |
| suspicious | -0.06 | one | -0.06 |

## 9. Bottom line
See FINDINGS.md next to this report for the interpretation of these numbers.

