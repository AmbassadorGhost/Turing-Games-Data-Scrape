# Deceiving vs truthful speech: word analysis

Public in-game AI speech from the Turing Games bins: **104 deceiving and 221 truthful utterances** (3,709 and 8,265 tokens) across 7 games and 15 models. All player/model names are replaced by a placeholder before counting, so 'who is being talked about' cannot masquerade as a lie signal.

## 1. Surface statistics (how much, not what)

| statistic | deceiving | truthful | diff | models where deceiving > truthful | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|---|
| words per utterance | 35.48 | 37.19 | -1.70 | 4/8 | 1.00 | 0.38 |
| questions per utterance | 0.18 | 0.18 | +0.01 | 1/8 | 0.07 | 0.44 |
| exclamations per utterance | 0.00 | 0.04 | -0.04 | 0/8 | 0.01 | 0.50 |
| mean word length | 4.56 | 4.58 | -0.01 | 4/8 | 1.00 | 1.00 |
| sentences per utterance | 3.26 | 3.63 | -0.37 | 3/8 | 0.73 | 0.30 |

The paired columns compare each model with *itself* on its lying vs truthful games (18 models appear on both sides), which removes 'which models happen to be mafia more often'.

## 2. Which words differ (names scrubbed, uncontrolled)

Weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008); z > ~2.5 is notable. Rates are per 1,000 tokens.

**Leaning deceiving**

| word | n deceiving | n truthful | rate dec | rate tru | z |
|---|---|---|---|---|---|
| jester | 14 | 5 | 4.0 | 0.6 | +3.3 |
| your | 19 | 14 | 5.4 | 1.8 | +2.8 |
| just | 26 | 25 | 7.4 | 3.2 | +2.7 |
| if | 21 | 19 | 5.9 | 2.4 | +2.6 |
| sense | 8 | 3 | 2.3 | 0.4 | +2.4 |
| let's | 14 | 11 | 4.0 | 1.4 | +2.3 |
| exactly | 9 | 5 | 2.5 | 0.6 | +2.3 |
| body | 9 | 5 | 2.5 | 0.6 | +2.3 |
| reads | 7 | 4 | 2.0 | 0.5 | +2.0 |
| too | 10 | 8 | 2.8 | 1.0 | +1.9 |
| from | 22 | 27 | 6.2 | 3.4 | +1.9 |
| how | 11 | 10 | 3.1 | 1.3 | +1.8 |
| makes | 13 | 13 | 3.7 | 1.7 | +1.8 |
| it's | 13 | 13 | 3.7 | 1.7 | +1.8 |
| what | 8 | 6 | 2.3 | 0.8 | +1.8 |
| wagon | 8 | 6 | 2.3 | 0.8 | +1.8 |
| why | 10 | 9 | 2.8 | 1.1 | +1.8 |
| focus | 7 | 5 | 2.0 | 0.6 | +1.8 |
| agree | 9 | 8 | 2.5 | 1.0 | +1.7 |
| after | 18 | 22 | 5.1 | 2.8 | +1.7 |
| no | 13 | 15 | 3.7 | 1.9 | +1.6 |
| role | 6 | 5 | 1.7 | 0.6 | +1.5 |
| here | 6 | 5 | 1.7 | 0.6 | +1.5 |
| abstain | 6 | 5 | 1.7 | 0.6 | +1.5 |
| path | 6 | 5 | 1.7 | 0.6 | +1.5 |

**Leaning truthful**

| word | n deceiving | n truthful | rate dec | rate tru | z |
|---|---|---|---|---|---|
| mafia | 29 | 138 | 8.2 | 17.6 | -3.4 |
| was | 15 | 82 | 4.2 | 10.4 | -2.9 |
| town | 10 | 54 | 2.8 | 6.9 | -2.3 |
| night | 5 | 36 | 1.4 | 4.6 | -2.3 |
| confirmed | 3 | 26 | 0.8 | 3.3 | -2.1 |
| both | 1 | 18 | 0.3 | 2.3 | -2.0 |
| us | 5 | 30 | 1.4 | 3.8 | -1.9 |
| member | 1 | 15 | 0.3 | 1.9 | -1.8 |
| being | 1 | 15 | 0.3 | 1.9 | -1.8 |
| were | 5 | 29 | 1.4 | 3.7 | -1.8 |
| at | 4 | 25 | 1.1 | 3.2 | -1.7 |
| our | 4 | 25 | 1.1 | 3.2 | -1.7 |
| then | 2 | 18 | 0.6 | 2.3 | -1.7 |
| trying | 7 | 34 | 2.0 | 4.3 | -1.7 |
| targeted | 0 | 10 | 0.0 | 1.3 | -1.7 |
| who | 12 | 46 | 3.4 | 5.9 | -1.5 |
| they | 12 | 46 | 3.4 | 5.9 | -1.5 |
| left | 4 | 22 | 1.1 | 2.8 | -1.5 |
| look | 2 | 15 | 0.6 | 1.9 | -1.5 |
| abstaining | 1 | 11 | 0.3 | 1.4 | -1.4 |
| day | 9 | 35 | 2.5 | 4.5 | -1.3 |
| win | 1 | 10 | 0.3 | 1.3 | -1.3 |
| am | 1 | 10 | 0.3 | 1.3 | -1.3 |
| doctor | 5 | 23 | 1.4 | 2.9 | -1.3 |
| sheriff | 18 | 59 | 5.1 | 7.5 | -1.3 |

**Bigrams** (min 8 occurrences)

| leaning deceiving | z | leaning truthful | z |
|---|---|---|---|
| the one | +2.2 | a mafia | -2.0 |
| jester bait | +1.9 | mafia member | -1.8 |
| vote on | +1.5 | and i | -1.6 |
| as the | +1.4 | trying to | -1.6 |
| to be | +1.4 | town and | -1.6 |
| the body | +1.4 | i was | -1.5 |
| i agree | +1.3 | with me | -1.5 |
| agree with | +1.0 | was with | -1.5 |
| as i | +1.0 | was the | -1.4 |
| abstain for | +1.0 | on the | -1.4 |
| have to | +0.8 | you were | -1.3 |
| i have | +0.8 | i am | -1.3 |
| classic mafia | +0.7 | last night | -1.3 |
| on me | +0.7 | confirmed mafia | -1.2 |
| we need | +0.6 | vote for | -1.2 |

## 3. Does it survive a game-level permutation test?

Labels were shuffled 3,000 times among the players *within each game* (each player's whole set of utterances moves together, and every game keeps its real number of liars). This asks: given these exact games and players, how often would a random choice of 'who is mafia' produce a word gap this large?

- words tested: 200; expected false positives at p<0.05 by chance: ~10
- words with p < 0.05: **22**
- words surviving Benjamini-Hochberg FDR 10%: **0** 

| word | direction | n dec | n tru | log-odds | z | perm p | FDR 10% |
|---|---|---|---|---|---|---|---|
| mafia | truthful | 29 | 138 | -0.57 | -3.4 | 0.003 |  |
| was | truthful | 15 | 82 | -0.66 | -2.9 | 0.006 |  |
| town | truthful | 10 | 54 | -0.65 | -2.3 | 0.006 |  |
| just | deceiving | 26 | 25 | +0.69 | +2.7 | 0.010 |  |
| your | deceiving | 19 | 14 | +0.91 | +2.8 | 0.012 |  |
| if | deceiving | 21 | 19 | +0.74 | +2.6 | 0.013 |  |
| member | truthful | 1 | 15 | -1.20 | -1.8 | 0.016 |  |
| sense | deceiving | 8 | 3 | +1.45 | +2.4 | 0.017 |  |
| how | deceiving | 11 | 10 | +0.74 | +1.8 | 0.018 |  |
| abstain | deceiving | 6 | 5 | +0.81 | +1.5 | 0.018 |  |
| were | truthful | 5 | 29 | -0.70 | -1.8 | 0.020 |  |
| being | truthful | 1 | 15 | -1.20 | -1.8 | 0.024 |  |
| both | truthful | 1 | 18 | -1.27 | -2.0 | 0.027 |  |
| us | truthful | 5 | 30 | -0.72 | -1.9 | 0.033 |  |
| too | deceiving | 10 | 8 | +0.84 | +1.9 | 0.035 |  |
| then | truthful | 2 | 18 | -0.96 | -1.7 | 0.038 |  |
| body | deceiving | 9 | 5 | +1.14 | +2.3 | 0.038 |  |
| night | truthful | 5 | 36 | -0.83 | -2.3 | 0.041 |  |
| our | truthful | 4 | 25 | -0.74 | -1.7 | 0.043 |  |
| let's | deceiving | 14 | 11 | +0.86 | +2.3 | 0.046 |  |
| jester | deceiving | 14 | 5 | +1.49 | +3.3 | 0.049 |  |
| agree | deceiving | 9 | 8 | +0.75 | +1.7 | 0.049 |  |

## 4. Within-model consistency (controls for which model is speaking)

Mantel-Haenszel pooled odds ratio with the speaking model as the stratum: each word is compared only between a model's lying and truthful games, then pooled. 'agree' = how many of the models that use the word lean the same way as the pooled estimate. Listed: >= 6 models, >= 75% agreement, |log OR| >= 0.4.

| word | direction | pooled log OR | models agreeing | perm p |
|---|---|---|---|---|
| your | deceiving | +1.17 | 5/6 | 0.012 |
| seems | deceiving | +0.94 | 5/6 | 0.130 |
| vigilante | deceiving | +0.90 | 5/6 | 0.548 |
| mafia | truthful | -0.83 | 7/8 | 0.003 |
| no | deceiving | +0.68 | 6/7 | 0.055 |
| were | truthful | -0.66 | 5/6 | 0.020 |
| was | truthful | -0.64 | 6/8 | 0.006 |
| after | deceiving | +0.62 | 6/7 | 0.090 |
| left | truthful | -0.60 | 6/6 | 0.117 |
| just | deceiving | +0.55 | 6/7 | 0.010 |
| kill | truthful | -0.54 | 5/6 | 0.449 |
| check | truthful | -0.54 | 5/6 | 0.169 |
| it | deceiving | +0.48 | 7/8 | 0.503 |
| real | deceiving | +0.47 | 5/6 | 0.998 |
| at | truthful | -0.46 | 5/6 | 0.064 |
| us | truthful | -0.44 | 6/7 | 0.033 |
| you | truthful | -0.44 | 6/7 | 0.951 |

**Robust to both controls** (consistent across models *and* game-permutation p < 0.05): your (dec), mafia (tru), were (tru), was (tru), just (dec), us (tru)

## 5. Other controls

### 5a. Power roles removed
Sheriffs, doctors, vigilantes etc. make claims villagers never make. Comparing liars with *plain villagers/crewmates only*:

| word (from section 2) | z all truthful | z villagers only |
|---|---|---|
| jester | +3.3 | +3.1 |
| your | +2.8 | +2.4 |
| just | +2.7 | +2.9 |
| if | +2.6 | +2.3 |
| sense | +2.4 | +2.0 |
| let's | +2.3 | +2.2 |
| exactly | +2.3 | +2.0 |
| body | +2.3 | +1.8 |
| reads | +2.0 | +2.1 |
| too | +1.9 | +1.4 |
| from | +1.9 | +1.8 |
| how | +1.8 | +1.6 |
| mafia | -3.4 | -2.5 |
| was | -2.9 | -3.2 |
| town | -2.3 | -2.4 |
| night | -2.3 | -1.3 |
| confirmed | -2.1 | -0.8 |
| both | -2.0 | -2.3 |
| us | -1.9 | -1.9 |
| member | -1.8 | -1.4 |
| being | -1.8 | -2.0 |
| were | -1.8 | -1.9 |
| at | -1.7 | -2.1 |
| our | -1.7 | -1.6 |

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
| jester | +3.3 | +1.70 | 3/3 |
| your | +2.8 | +1.07 | 3/3 |
| just | +2.7 | +0.82 | 3/3 |
| if | +2.6 | +0.87 | 3/3 |
| sense | +2.4 | +1.47 | 3/3 |
| let's | +2.3 | +1.00 | 3/3 |
| exactly | +2.3 | +1.38 | 3/3 |
| body | +2.3 | +1.29 | 3/3 |
| reads | +2.0 | +1.26 | 3/3 |
| too | +1.9 | +1.00 | 3/3 |
| mafia | -3.4 | -0.73 | 3/3 |
| was | -2.9 | -0.82 | 3/3 |
| town | -2.3 | -0.75 | 3/3 |
| night | -2.3 | -0.96 | 3/3 |
| confirmed | -2.1 | -0.99 | 3/3 |
| both | -2.0 | -1.28 | 3/3 |
| us | -1.9 | -0.81 | 3/3 |
| member | -1.8 | -1.00 | 3/3 |
| being | -1.8 | -1.05 | 3/3 |
| were | -1.8 | -0.69 | 3/3 |

### 5c. Game type
Among Us is about locations and bodies, Mafia about roles and votes. Stratifying by game type (mafia / among_us / one-night werewolf):

| word | z uncontrolled | log OR within game type | types agreeing |
|---|---|---|---|
| jester | +3.3 | +1.80 | 1/1 |
| your | +2.8 | +1.10 | 2/2 |
| just | +2.7 | +0.81 | 2/2 |
| if | +2.6 | +0.89 | 2/2 |
| sense | +2.4 | +1.58 | 2/2 |
| let's | +2.3 | +1.03 | 1/2 |
| exactly | +2.3 | +1.37 | 1/1 |
| body | +2.3 | +1.21 | 1/2 |
| reads | +2.0 | +1.33 | 1/1 |
| too | +1.9 | +1.00 | 2/2 |
| mafia | -3.4 | -0.74 | 1/1 |
| was | -2.9 | -0.91 | 2/2 |
| town | -2.3 | -0.83 | 1/1 |
| night | -2.3 | -1.07 | 1/1 |
| confirmed | -2.1 | -1.21 | 1/1 |
| both | -2.0 | -1.50 | 2/2 |
| us | -1.9 | -0.85 | 2/2 |
| member | -1.8 | -1.52 | 1/1 |
| being | -1.8 | -1.52 | 1/1 |
| were | -1.8 | -0.84 | 2/2 |

### 5d. Caption-grade text only
90% of the text is Gemini's reconstruction of the videos. In the 0 caption-grade utterances (0 deceiving, from 2 games), 0 of 0 top words keep the same direction:

| word | z all text | z captions only | n in captions |
|---|---|---|---|

Too little caption text to confirm anything; treat this as a consistency check only.

## 6. Word categories (LIWC-style)

Rates per 100 tokens. 'perm p' is the within-game permutation test; the last two columns are the within-model paired comparison.

| category | deceiving % | truthful % | log-odds | perm p | models dec > tru | sign p |
|---|---|---|---|---|---|---|
| I / me / my (1st singular) | 3.93 | 4.13 | -0.04 | 0.709 | 4/8 | 1.00 |
| we / us / our (1st plural) | 1.73 | 1.86 | -0.06 | 0.633 | 6/8 | 0.29 |
| you / your (2nd person) | 1.90 | 1.50 | +0.19 | 0.330 | 2/8 | 0.29 |
| they / them (3rd plural) | 1.30 | 1.55 | -0.14 | 0.370 | 4/8 | 1.00 |
| negations | 1.27 | 0.92 | +0.27 | 0.156 | 6/8 | 0.29 |
| hedges / uncertainty | 0.62 | 0.50 | +0.18 | 0.493 | 4/8 | 1.00 |
| certainty / emphasis | 0.59 | 0.62 | -0.04 | 0.886 | 3/8 | 0.73 |
| agreement / alignment | 1.53 | 0.89 | +0.44 | 0.013 | 6/8 | 0.29 |
| accusation verbs | 0.62 | 0.76 | -0.16 | 0.385 | 3/8 | 0.73 |
| role claims (own) | 0.93 | 1.26 | -0.23 | 0.135 | 4/8 | 1.00 |
| evil-side words | 0.99 | 1.89 | -0.49 | 0.003 | 1/8 | 0.07 |
| vote words | 2.52 | 2.38 | +0.05 | 0.730 | 2/8 | 0.29 |
| evidence / timing | 1.67 | 2.07 | -0.17 | 0.273 | 2/8 | 0.29 |
| questions (?) | 0.54 | 0.50 | +0.06 | 0.797 | 1/8 | 0.07 |
| exclamations (!) | 0.00 | 0.10 | -1.74 | 0.299 | 0/8 | 0.01 |

## 7. The keyword detector, evaluated honestly

The lexicon (signed word weights, section 8) is re-fit inside every fold, so no test utterance ever influenced the weights used to score it. AUC 0.5 = coin flip; AP is compared with the base rate.

| detector | held-out games (5-fold) | held-out models (train without a model, test on it) |
|---|---|---|
| keyword lexicon, names scrubbed | AUC 0.516, AP 0.358 (base rate 0.32, n=325) | AUC 0.469, AP 0.318 (base rate 0.33, n=262) |
| keyword lexicon, names kept (leaky) | AUC 0.577, AP 0.397 (base rate 0.32, n=325) | - |
| TF-IDF + logistic regression (reference) | AUC 0.575, AP 0.391 (base rate 0.32, n=325) | AUC 0.524, AP 0.409 (base rate 0.33, n=262) |

Lexicon size (same fold protocol): fewer, better-attested words generalise better.

| lexicon | AUC held-out games | AUC held-out models |
|---|---|---|
| >= 5 occurrences, top 300 words | 0.575 | 0.487 |
| >= 10 occurrences, top 100 words | 0.615 | 0.548 |
| >= 15 occurrences, top 100 words | 0.567 | 0.534 |
| >= 30 occurrences, top 40 words | 0.516 | 0.469 |
| >= 50 occurrences, top 20 words | 0.458 | 0.379 |

Only the 6 words robust to both controls (section 4), weights refit per fold: **AUC 0.697** on held-out games (the word list itself was chosen on all data, so this is slightly optimistic). Simply counting how many of {if, just, your, why} appear, with no fitting at all: **AUC 0.626**.


Held-out model, one at a time (general lexicon trained on everyone else):

| model held out | utterances | deceiving | AUC |
|---|---|---|---|
| claude-sonnet-4.5 | 38 | 7 | 0.59 |
| gemini-3-flash | 70 | 18 | 0.52 |
| gpt-4o | 30 | 12 | 0.53 |
| gpt-5.2 | 20 | 7 | 0.45 |
| grok-4.1 | 40 | 18 | 0.51 |

### Family tier
For each family: a lexicon trained only on that family's other games vs the general lexicon trained on all other games, both scored on the family's held-out games.

| family | utterances | deceiving | games | family-specific lexicon | general lexicon |
|---|---|---|---|---|---|
| anthropic | 49 | 11 | 6 | too little data | too little data |
| deepseek | 20 | 0 | 5 | too little data | too little data |
| google | 108 | 25 | 7 | AUC 0.447, AP 0.207 (base rate 0.23, n=108) | AUC 0.388, AP 0.196 (base rate 0.23, n=108) |
| meta | 17 | 5 | 6 | too little data | too little data |
| moonshot | 15 | 2 | 5 | too little data | too little data |
| openai | 74 | 43 | 6 | AUC 0.178, AP 0.428 (base rate 0.58, n=74) | AUC 0.485, AP 0.609 (base rate 0.58, n=74) |
| xai | 40 | 18 | 5 | AUC 0.000, AP 0.371 (base rate 0.45, n=40) | AUC 0.437, AP 0.497 (base rate 0.45, n=40) |
| zhipu | 2 | 0 | 1 | too little data | too little data |

### Per-model tier
Most models have under 20 deceiving utterances, so a per-model lexicon cannot be evaluated honestly yet; the per-model bins exist, and `lexicons/` holds family lexicons as the finest level that has enough data. Per-model 'tells' below are the top words by within-model log-odds and are **provisional**.

| model | dec | tru | leans deceiving | leans truthful |
|---|---|---|---|---|
| gemini-3-flash | 18 | 52 | if, your, to, be, need, makes | and, s, mafia, was, engine, upper |
| gpt-4o | 12 | 18 | on, focus, let's, votes, too, so | vote, in, mafia, me, point, town |
| grok-4.1 | 18 | 22 | vote, is, their, execute, voted, s | like, everyone, confirmed, hammer, 1, my |

### Domain-neutral lexicon
`lexicon_general.json` drops every Mafia / Among Us / Werewolf term (roles, votes, kills, locations, mechanics) and bare numbers, for use outside these games. Held-out games AUC 0.448, held-out models AUC 0.392 (vs 0.516 / 0.469 with game terms): the signal is not in the game vocabulary. Words: if, just, after, from, right, so, not, of, with, a, as, that's, is, that, s, this, ?, it, and, on, now, the, i'm, my, one, suspicious, we, for, them, in, like, have, last, before, they, who, was, trying, were, us.


## 8. The lexicon
`lexicon.json`: a word list with signed weights (positive = deceiving) and an intercept; score = intercept + sum of weights of the words present (after name scrubbing). 40 words. Strongest weights:

| deceiving | weight | truthful | weight |
|---|---|---|---|
| if | +0.39 | night | -0.39 |
| just | +0.31 | us | -0.36 |
| after | +0.23 | were | -0.33 |
| from | +0.21 | trying | -0.30 |
| right | +0.14 | town | -0.30 |
| so | +0.13 | was | -0.27 |
| vote | +0.11 | mafia | -0.24 |
| not | +0.09 | who | -0.23 |
| of | +0.07 | day | -0.19 |
| with | +0.06 | they | -0.19 |
| a | +0.05 | before | -0.12 |
| as | +0.05 | last | -0.11 |
| is | +0.04 | have | -0.11 |
| that | +0.03 | sheriff | -0.10 |
| s | +0.03 | like | -0.08 |
| and | +0.02 | in | -0.07 |
| the | -0.04 | them | -0.07 |
| i'm | -0.04 | for | -0.06 |
| voting | -0.04 | we | -0.06 |
| one | -0.05 | suspicious | -0.06 |

## 9. Bottom line
See FINDINGS.md next to this report for the interpretation of these numbers.

