# Deceiving vs truthful speech: word analysis

Public in-game AI speech from the Turing Games bins: **254 deceiving and 648 truthful utterances** (9,623 and 23,492 tokens) across 20 games and 36 models. All player/model names are replaced by a placeholder before counting, so 'who is being talked about' cannot masquerade as a lie signal.

## 1. Surface statistics (how much, not what)

| statistic | deceiving | truthful | diff | models where deceiving > truthful | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|---|
| words per utterance | 37.63 | 35.98 | +1.65 | 10/18 | 0.81 | 0.93 |
| questions per utterance | 0.23 | 0.21 | +0.02 | 11/18 | 0.48 | 0.80 |
| exclamations per utterance | 0.02 | 0.06 | -0.04 | 3/18 | 0.01 | 0.14 |
| mean word length | 4.58 | 4.56 | +0.02 | 10/18 | 0.81 | 0.97 |
| sentences per utterance | 3.43 | 3.78 | -0.35 | 6/18 | 0.24 | 0.11 |

The paired columns compare each model with *itself* on its lying vs truthful games (18 models appear on both sides), which removes 'which models happen to be mafia more often'.

## 2. Which words differ (names scrubbed, uncontrolled)

Weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008); z > ~2.5 is notable. Rates are per 1,000 tokens.

**Leaning deceiving**

| word | n deceiving | n truthful | rate dec | rate tru | z |
|---|---|---|---|---|---|
| if | 67 | 63 | 7.2 | 2.8 | +4.9 |
| just | 54 | 60 | 5.8 | 2.6 | +3.7 |
| your | 48 | 53 | 5.1 | 2.3 | +3.6 |
| why | 33 | 33 | 3.5 | 1.5 | +3.2 |
| you're | 44 | 52 | 4.7 | 2.3 | +3.2 |
| makes | 22 | 20 | 2.4 | 0.9 | +2.9 |
| sense | 11 | 5 | 1.2 | 0.2 | +2.9 |
| jester | 14 | 9 | 1.5 | 0.4 | +2.8 |
| exactly | 23 | 22 | 2.5 | 1.0 | +2.8 |
| fast | 8 | 2 | 0.9 | 0.1 | +2.7 |
| agree | 20 | 19 | 2.1 | 0.8 | +2.6 |
| killer | 14 | 11 | 1.5 | 0.5 | +2.5 |
| how | 21 | 22 | 2.3 | 1.0 | +2.5 |
| wagon | 15 | 13 | 1.6 | 0.6 | +2.5 |
| serial | 12 | 9 | 1.3 | 0.4 | +2.4 |
| convenient | 11 | 8 | 1.2 | 0.4 | +2.3 |
| keep | 13 | 11 | 1.4 | 0.5 | +2.3 |
| really | 12 | 10 | 1.3 | 0.4 | +2.3 |
| concrete | 13 | 12 | 1.4 | 0.5 | +2.2 |
| don't | 17 | 19 | 1.8 | 0.8 | +2.1 |
| quiet | 9 | 7 | 1.0 | 0.3 | +2.0 |
| abstain | 13 | 13 | 1.4 | 0.6 | +2.0 |
| reads | 11 | 10 | 1.2 | 0.4 | +2.0 |
| here | 14 | 15 | 1.5 | 0.7 | +2.0 |
| own | 8 | 6 | 0.9 | 0.3 | +2.0 |

**Leaning truthful**

| word | n deceiving | n truthful | rate dec | rate tru | z |
|---|---|---|---|---|---|
| was | 56 | 253 | 6.0 | 11.1 | -3.8 |
| lights | 0 | 43 | 0.0 | 1.9 | -3.3 |
| i | 154 | 518 | 16.5 | 22.8 | -3.2 |
| 2 | 5 | 45 | 0.5 | 2.0 | -2.6 |
| werewolf | 0 | 24 | 0.0 | 1.1 | -2.5 |
| at | 23 | 103 | 2.5 | 4.5 | -2.4 |
| both | 5 | 40 | 0.5 | 1.8 | -2.3 |
| did | 7 | 47 | 0.8 | 2.1 | -2.3 |
| east | 1 | 23 | 0.1 | 1.0 | -2.2 |
| ! | 6 | 42 | 0.6 | 1.8 | -2.2 |
| security | 11 | 59 | 1.2 | 2.6 | -2.2 |
| in | 55 | 192 | 5.9 | 8.4 | -2.1 |
| claims | 3 | 29 | 0.3 | 1.3 | -2.1 |
| electrical | 7 | 43 | 0.8 | 1.9 | -2.0 |
| member | 1 | 20 | 0.1 | 0.9 | -2.0 |
| them | 17 | 76 | 1.8 | 3.3 | -2.0 |
| mafia | 80 | 260 | 8.6 | 11.4 | -2.0 |
| blackout | 0 | 15 | 0.0 | 0.7 | -2.0 |
| saw | 25 | 99 | 2.7 | 4.4 | -1.9 |
| night | 28 | 108 | 3.0 | 4.8 | -1.9 |
| and | 192 | 559 | 20.6 | 24.6 | -1.9 |
| my | 21 | 86 | 2.3 | 3.8 | -1.9 |
| they | 29 | 109 | 3.1 | 4.8 | -1.8 |
| weapons | 2 | 21 | 0.2 | 0.9 | -1.8 |
| alone | 2 | 21 | 0.2 | 0.9 | -1.8 |

**Bigrams** (min 8 occurrences)

| leaning deceiving | z | leaning truthful | z |
|---|---|---|---|
| if you're | +2.9 | with me | -2.5 |
| serial killer | +2.6 | i saw | -2.4 |
| i agree | +2.3 | i found | -2.1 |
| saw me | +2.3 | confirmed mafia | -2.1 |
| to keep | +2.3 | was with | -2.0 |
| was doing | +2.3 | mafia member | -1.9 |
| only concrete | +2.3 | i was | -1.8 |
| vote on | +2.1 | o 2 | -1.8 |
| if we | +2.0 | saw you | -1.8 |
| going to | +2.0 | night and | -1.8 |
| claim to | +2.0 | was on | -1.8 |
| to be | +2.0 | vote for | -1.7 |
| are you | +2.0 | i investigated | -1.7 |
| if you | +2.0 | came back | -1.7 |
| i'm not | +2.0 | night one | -1.6 |

## 3. Does it survive a game-level permutation test?

Labels were shuffled 3,000 times among the players *within each game* (each player's whole set of utterances moves together, and every game keeps its real number of liars). This asks: given these exact games and players, how often would a random choice of 'who is mafia' produce a word gap this large?

- words tested: 461; expected false positives at p<0.05 by chance: ~23
- words with p < 0.05: **35**
- words surviving Benjamini-Hochberg FDR 10%: **4** ['if', 'your', 'why', 'just']

| word | direction | n dec | n tru | log-odds | z | perm p | FDR 10% |
|---|---|---|---|---|---|---|---|
| your | deceiving | 48 | 53 | +0.65 | +3.6 | 0.000 | yes |
| if | deceiving | 67 | 63 | +0.78 | +4.9 | 0.000 | yes |
| just | deceiving | 54 | 60 | +0.64 | +3.7 | 0.001 | yes |
| why | deceiving | 33 | 33 | +0.73 | +3.2 | 0.001 | yes |
| was | truthful | 56 | 253 | -0.46 | -3.8 | 0.002 |  |
| sense | deceiving | 11 | 5 | +1.37 | +2.9 | 0.004 |  |
| fast | deceiving | 8 | 2 | +1.83 | +2.7 | 0.004 |  |
| abstain | deceiving | 13 | 13 | +0.73 | +2.0 | 0.006 |  |
| both | truthful | 5 | 40 | -0.82 | -2.3 | 0.007 |  |
| member | truthful | 1 | 20 | -1.24 | -2.0 | 0.008 |  |
| how | deceiving | 21 | 22 | +0.69 | +2.5 | 0.009 |  |
| really | deceiving | 12 | 10 | +0.88 | +2.3 | 0.010 |  |
| robber | truthful | 1 | 15 | -1.13 | -1.7 | 0.011 |  |
| claims | truthful | 3 | 29 | -0.92 | -2.1 | 0.011 |  |
| you're | deceiving | 44 | 52 | +0.59 | +3.2 | 0.012 |  |
| agree | deceiving | 20 | 19 | +0.77 | +2.6 | 0.016 |  |
| werewolf | truthful | 0 | 24 | -1.67 | -2.5 | 0.018 |  |
| correct | truthful | 0 | 10 | -1.67 | -1.6 | 0.020 |  |
| makes | deceiving | 22 | 20 | +0.81 | +2.9 | 0.023 |  |
| quiet | deceiving | 9 | 7 | +0.94 | +2.0 | 0.024 |  |
| don't | deceiving | 17 | 19 | +0.64 | +2.1 | 0.027 |  |
| exactly | deceiving | 23 | 22 | +0.77 | +2.8 | 0.028 |  |
| start | deceiving | 7 | 5 | +1.01 | +1.9 | 0.032 |  |
| own | deceiving | 8 | 6 | +0.97 | +2.0 | 0.032 |  |
| here | deceiving | 14 | 15 | +0.67 | +2.0 | 0.035 |  |
| keep | deceiving | 13 | 11 | +0.87 | +2.3 | 0.035 |  |
| concrete | deceiving | 13 | 12 | +0.79 | +2.2 | 0.038 |  |
| arrived | truthful | 0 | 10 | -1.67 | -1.6 | 0.041 |  |
| 2 | truthful | 5 | 45 | -0.88 | -2.6 | 0.041 |  |
| i | truthful | 154 | 518 | -0.25 | -3.2 | 0.042 |  |
| let | deceiving | 9 | 8 | +0.83 | +1.9 | 0.043 |  |
| navigation | truthful | 0 | 12 | -1.67 | -1.8 | 0.043 |  |
| and | truthful | 192 | 559 | -0.14 | -1.9 | 0.048 |  |
| we're | deceiving | 8 | 8 | +0.73 | +1.6 | 0.049 |  |
| instead | deceiving | 13 | 14 | +0.67 | +1.9 | 0.049 |  |

## 4. Within-model consistency (controls for which model is speaking)

Mantel-Haenszel pooled odds ratio with the speaking model as the stratum: each word is compared only between a model's lying and truthful games, then pooled. 'agree' = how many of the models that use the word lean the same way as the pooled estimate. Listed: >= 6 models, >= 75% agreement, |log OR| >= 0.4.

| word | direction | pooled log OR | models agreeing | perm p |
|---|---|---|---|---|
| fast | deceiving | +1.66 | 6/6 | 0.004 |
| killer | deceiving | +1.58 | 6/7 | 0.255 |
| maybe | deceiving | +1.45 | 6/6 | 0.061 |
| start | deceiving | +1.41 | 5/6 | 0.032 |
| keep | deceiving | +1.39 | 10/10 | 0.035 |
| phrasing | deceiving | +1.26 | 7/7 | 0.109 |
| going | deceiving | +1.25 | 7/8 | 0.073 |
| convenient | deceiving | +1.13 | 7/9 | 0.087 |
| exactly | deceiving | +1.08 | 11/13 | 0.028 |
| wagon | deceiving | +1.08 | 8/9 | 0.135 |
| robber | truthful | -1.06 | 5/6 | 0.011 |
| own | deceiving | +1.06 | 7/8 | 0.032 |
| why | deceiving | +1.06 | 13/14 | 0.001 |
| saying | deceiving | +1.03 | 5/6 | 0.282 |
| really | deceiving | +1.00 | 9/9 | 0.010 |
| abstain | deceiving | +1.00 | 9/10 | 0.006 |
| solid | deceiving | +0.99 | 6/7 | 0.137 |
| skip | deceiving | +0.99 | 5/6 | 0.347 |
| we're | deceiving | +0.98 | 7/9 | 0.049 |
| question | deceiving | +0.96 | 8/9 | 0.713 |
| immediate | deceiving | +0.95 | 7/8 | 0.161 |
| making | deceiving | +0.95 | 8/8 | 0.101 |
| losing | deceiving | +0.94 | 8/9 | 0.385 |
| through | deceiving | +0.93 | 9/11 | 0.124 |
| how | deceiving | +0.93 | 13/15 | 0.009 |
| concrete | deceiving | +0.93 | 9/12 | 0.038 |
| line | deceiving | +0.91 | 5/6 | 0.231 |
| focus | deceiving | +0.90 | 7/9 | 0.141 |
| clean | deceiving | +0.89 | 6/6 | 0.376 |
| consensus | deceiving | +0.88 | 7/8 | 0.111 |

**Robust to both controls** (consistent across models *and* game-permutation p < 0.05): fast (dec), start (dec), keep (dec), exactly (dec), robber (tru), own (dec), why (dec), really (dec), abstain (dec), we're (dec), how (dec), concrete (dec), claims (tru), makes (dec), let (dec), if (dec), quiet (dec), just (dec), don't (dec), agree (dec)

## 5. Other controls

### 5a. Power roles removed
Sheriffs, doctors, vigilantes etc. make claims villagers never make. Comparing liars with *plain villagers/crewmates only*:

| word (from section 2) | z all truthful | z villagers only |
|---|---|---|
| if | +4.9 | +4.9 |
| just | +3.7 | +4.3 |
| your | +3.6 | +3.3 |
| why | +3.2 | +3.3 |
| you're | +3.2 | +2.9 |
| makes | +2.9 | +2.2 |
| sense | +2.9 | +2.2 |
| jester | +2.8 | +2.4 |
| exactly | +2.8 | +2.4 |
| fast | +2.7 | +2.5 |
| agree | +2.6 | +2.3 |
| killer | +2.5 | +2.0 |
| was | -3.8 | -4.6 |
| lights | -3.3 | -3.9 |
| i | -3.2 | -3.4 |
| 2 | -2.6 | -2.6 |
| werewolf | -2.5 | -1.9 |
| at | -2.4 | -3.3 |
| both | -2.3 | -2.7 |
| did | -2.3 | -2.7 |
| east | -2.2 | -2.8 |
| ! | -2.2 | -1.1 |
| security | -2.2 | -3.3 |
| in | -2.1 | -3.7 |

Words that flip direction once power roles are removed: night.

### 5b. Position in the game
Liars survive longer, so more of their speech is late-game.

| third of game | share of deceiving speech | share of truthful speech |
|---|---|---|
| early | 33% | 33% |
| mid | 30% | 34% |
| late | 37% | 33% |

Pooled log OR stratified by game third (so early speech is only compared with early speech):

| word | z uncontrolled | log OR within position | thirds agreeing |
|---|---|---|---|
| if | +4.9 | +0.96 | 3/3 |
| just | +3.7 | +0.78 | 3/3 |
| your | +3.6 | +0.79 | 3/3 |
| why | +3.2 | +0.88 | 3/3 |
| you're | +3.2 | +0.72 | 3/3 |
| makes | +2.9 | +0.97 | 3/3 |
| sense | +2.9 | +1.52 | 3/3 |
| jester | +2.8 | +1.29 | 3/3 |
| exactly | +2.8 | +0.93 | 3/3 |
| fast | +2.7 | +1.86 | 3/3 |
| was | -3.8 | -0.60 | 3/3 |
| lights | -3.3 | -2.50 | 3/3 |
| i | -3.2 | -0.32 | 3/3 |
| 2 | -2.6 | -1.09 | 3/3 |
| werewolf | -2.5 | -1.94 | 3/3 |
| at | -2.4 | -0.57 | 3/3 |
| both | -2.3 | -1.00 | 3/3 |
| did | -2.3 | -0.80 | 3/3 |
| east | -2.2 | -1.33 | 3/3 |
| ! | -2.2 | -0.97 | 3/3 |

### 5c. Game type
Among Us is about locations and bodies, Mafia about roles and votes. Stratifying by game type (mafia / among_us / one-night werewolf):

| word | z uncontrolled | log OR within game type | types agreeing |
|---|---|---|---|
| if | +4.9 | +0.99 | 3/3 |
| just | +3.7 | +0.80 | 3/3 |
| your | +3.6 | +0.84 | 2/3 |
| why | +3.2 | +0.95 | 3/3 |
| you're | +3.2 | +0.70 | 3/3 |
| makes | +2.9 | +0.88 | 2/2 |
| sense | +2.9 | +1.52 | 2/2 |
| jester | +2.8 | +1.10 | 1/1 |
| exactly | +2.8 | +0.97 | 3/3 |
| fast | +2.7 | +2.07 | 2/2 |
| was | -3.8 | -0.42 | 3/3 |
| lights | -3.3 | -3.04 | 1/1 |
| i | -3.2 | -0.15 | 2/3 |
| 2 | -2.6 | -0.96 | 2/2 |
| werewolf | -2.5 | -2.15 | 1/1 |
| at | -2.4 | -0.26 | 3/3 |
| both | -2.3 | -0.86 | 3/3 |
| did | -2.3 | -0.68 | 3/3 |
| east | -2.2 | -1.32 | 1/1 |
| ! | -2.2 | -0.98 | 2/2 |

### 5d. Caption-grade text only
90% of the text is Gemini's reconstruction of the videos. In the 107 caption-grade utterances (14 deceiving, from 2 games), 12 of 17 top words keep the same direction:

| word | z all text | z captions only | n in captions |
|---|---|---|---|
| if | +4.9 | +0.7 | 8 |
| just | +3.7 | +1.3 | 5 |
| your | +3.6 | +1.0 | 11 |
| why | +3.2 | +2.2 | 8 |
| you're | +3.2 | -0.7 | 6 |
| killer | +2.5 | -0.5 | 3 |
| was | -3.8 | -0.7 | 61 |
| lights | -3.3 | -1.9 | 40 |
| i | -3.2 | +0.5 | 121 |
| 2 | -2.6 | +0.0 | 7 |
| at | -2.4 | -1.2 | 38 |
| both | -2.3 | -0.7 | 6 |
| did | -2.3 | -1.0 | 10 |
| east | -2.2 | -0.9 | 8 |
| security | -2.2 | -0.3 | 27 |
| in | -2.1 | +0.3 | 71 |
| electrical | -2.0 | -1.2 | 27 |

Too little caption text to confirm anything; treat this as a consistency check only.

## 6. Word categories (LIWC-style)

Rates per 100 tokens. 'perm p' is the within-game permutation test; the last two columns are the within-model paired comparison.

| category | deceiving % | truthful % | log-odds | perm p | models dec > tru | sign p |
|---|---|---|---|---|---|---|
| I / me / my (1st singular) | 4.19 | 4.56 | -0.07 | 0.234 | 9/18 | 1.00 |
| we / us / our (1st plural) | 1.83 | 1.52 | +0.15 | 0.123 | 12/18 | 0.24 |
| you / your (2nd person) | 2.26 | 1.76 | +0.20 | 0.100 | 8/18 | 0.81 |
| they / them (3rd plural) | 0.89 | 1.22 | -0.25 | 0.094 | 5/18 | 0.10 |
| negations | 1.61 | 1.32 | +0.16 | 0.110 | 11/18 | 0.48 |
| hedges / uncertainty | 0.59 | 0.51 | +0.11 | 0.536 | 9/18 | 1.00 |
| certainty / emphasis | 0.62 | 0.59 | +0.04 | 0.822 | 9/18 | 1.00 |
| agreement / alignment | 1.21 | 0.82 | +0.31 | 0.012 | 11/18 | 0.48 |
| accusation verbs | 0.61 | 0.76 | -0.17 | 0.209 | 5/18 | 0.10 |
| role claims (own) | 0.85 | 0.89 | -0.04 | 0.737 | 5/18 | 0.10 |
| evil-side words | 0.95 | 1.40 | -0.29 | 0.022 | 2/18 | 0.00 |
| vote words | 2.45 | 2.09 | +0.13 | 0.178 | 8/18 | 0.81 |
| evidence / timing | 1.85 | 2.48 | -0.23 | 0.124 | 8/18 | 0.81 |
| questions (?) | 0.63 | 0.61 | +0.03 | 0.866 | 8/18 | 0.81 |
| exclamations (!) | 0.06 | 0.18 | -0.74 | 0.194 | 3/18 | 0.01 |

## 7. The keyword detector, evaluated honestly

The lexicon (signed word weights, section 8) is re-fit inside every fold, so no test utterance ever influenced the weights used to score it. AUC 0.5 = coin flip; AP is compared with the base rate.

| detector | held-out games (5-fold) | held-out models (train without a model, test on it) |
|---|---|---|
| keyword lexicon, names scrubbed | AUC 0.629, AP 0.401 (base rate 0.28, n=902) | AUC 0.603, AP 0.427 (base rate 0.32, n=715) |
| keyword lexicon, names kept (leaky) | AUC 0.608, AP 0.393 (base rate 0.28, n=902) | - |
| TF-IDF + logistic regression (reference) | AUC 0.608, AP 0.371 (base rate 0.28, n=902) | AUC 0.569, AP 0.412 (base rate 0.32, n=715) |

Lexicon size (same fold protocol): fewer, better-attested words generalise better.

| lexicon | AUC held-out games | AUC held-out models |
|---|---|---|
| >= 5 occurrences, top 300 words | 0.580 | 0.546 |
| >= 10 occurrences, top 100 words | 0.594 | 0.551 |
| >= 15 occurrences, top 100 words | 0.605 | 0.567 |
| >= 30 occurrences, top 40 words | 0.629 | 0.603 |
| >= 50 occurrences, top 20 words | 0.624 | 0.581 |

Only the 20 words robust to both controls (section 4), weights refit per fold: **AUC 0.696** on held-out games (the word list itself was chosen on all data, so this is slightly optimistic). Simply counting how many of {if, just, your, why} appear, with no fitting at all: **AUC 0.633**.


Held-out model, one at a time (general lexicon trained on everyone else):

| model held out | utterances | deceiving | AUC |
|---|---|---|---|
| claude-opus-4.5 | 40 | 11 | 0.58 |
| claude-opus-4.8 | 28 | 8 | 0.75 |
| claude-sonnet-4.5 | 75 | 14 | 0.75 |
| deepseek-v3.2 | 46 | 15 | 0.64 |
| gemini-3-flash | 107 | 18 | 0.81 |
| gemini-3-pro | 47 | 7 | 0.29 |
| gemini-3.1-pro | 35 | 11 | 0.83 |
| gemini-3.5-flash | 21 | 10 | 0.81 |
| gpt-4o | 69 | 18 | 0.58 |
| gpt-5.2 | 55 | 32 | 0.83 |
| gpt-5.5 | 20 | 3 | 0.96 |
| grok-4.1 | 72 | 25 | 0.39 |
| kimi-k2.5 | 27 | 13 | 0.72 |
| llama-4 | 49 | 20 | 0.55 |

### Family tier
For each family: a lexicon trained only on that family's other games vs the general lexicon trained on all other games, both scored on the family's held-out games.

| family | utterances | deceiving | games | family-specific lexicon | general lexicon |
|---|---|---|---|---|---|
| anthropic | 177 | 41 | 19 | AUC 0.394, AP 0.211 (base rate 0.23, n=177) | AUC 0.529, AP 0.297 (base rate 0.23, n=177) |
| deepseek | 63 | 18 | 16 | AUC 0.093, AP 0.183 (base rate 0.29, n=63) | AUC 0.660, AP 0.524 (base rate 0.29, n=63) |
| google | 242 | 53 | 20 | AUC 0.516, AP 0.222 (base rate 0.22, n=242) | AUC 0.677, AP 0.334 (base rate 0.22, n=242) |
| meta | 49 | 20 | 17 | AUC 0.165, AP 0.308 (base rate 0.41, n=49) | AUC 0.555, AP 0.438 (base rate 0.41, n=49) |
| moonshot | 50 | 20 | 15 | AUC 0.145, AP 0.301 (base rate 0.40, n=50) | AUC 0.565, AP 0.453 (base rate 0.40, n=50) |
| openai | 189 | 77 | 17 | AUC 0.362, AP 0.342 (base rate 0.41, n=189) | AUC 0.632, AP 0.557 (base rate 0.41, n=189) |
| xai | 103 | 25 | 16 | AUC 0.156, AP 0.155 (base rate 0.24, n=103) | AUC 0.366, AP 0.192 (base rate 0.24, n=103) |
| zhipu | 29 | 0 | 4 | too little data | too little data |

### Per-model tier
Most models have under 20 deceiving utterances, so a per-model lexicon cannot be evaluated honestly yet; the per-model bins exist, and `lexicons/` holds family lexicons as the finest level that has enough data. Per-model 'tells' below are the top words by within-model log-odds and are **provisional**.

| model | dec | tru | leans deceiving | leans truthful |
|---|---|---|---|---|
| claude-opus-4.5 | 11 | 29 | in, admin, that, see, doing, anyone | is, my, two, of, yesterday, mafia |
| claude-opus-4.8 | 8 | 20 | we, killer, to, mafia, have, serial | security, in, was, and, body, ? |
| claude-sonnet-4.5 | 14 | 61 | if, does, just, that, really, night | s, voting, voted, only, been, mafia |
| deepseek-v3.2 | 15 | 31 | you, have, you're, slip, by, last | i, was, to, need, clear, their |
| gemini-3-flash | 18 | 89 | if, to, be, need, vigilante, it's | and, s, was, you, were, saw |
| gemini-3.1-pro | 11 | 24 | we, your, structural, for, is, data | you, i, and, in, them, villager |
| gemini-3.5-flash | 10 | 11 | that, it, killer, serial, we, am | o, 2, was, ?, of, from |
| gpt-4o | 18 | 51 | focus, votes, on, let's, too, push | in, vote, that, mafia, for, we |
| gpt-5.2 | 32 | 23 | day, vote, your, exactly, a, of | left, info, tonight, i, want, my |
| grok-4.1 | 25 | 47 | vote, their, saw, execute, is, tasks | 1, hammer, confirmed, you, everyone, 2 |
| kimi-k2.5 | 13 | 14 | you're, if, three, lynch, you, ? | was, need, about, to, not, claim |
| llama-4 | 20 | 29 | need, or, player, from, evidence, no | you, my, that, like, mason, were |

## 8. The lexicon
`lexicon.json`: a word list with signed weights (positive = deceiving) and an intercept; score = intercept + sum of weights of the words present (after name scrubbing). 40 words. Strongest weights:

| deceiving | weight | truthful | weight |
|---|---|---|---|
| if | +0.40 | lights | -0.85 |
| makes | +0.40 | 2 | -0.41 |
| agree | +0.37 | both | -0.40 |
| exactly | +0.37 | ! | -0.37 |
| why | +0.35 | did | -0.33 |
| how | +0.33 | electrical | -0.32 |
| don't | +0.30 | security | -0.29 |
| just | +0.30 | found | -0.25 |
| your | +0.26 | hallway | -0.23 |
| role | +0.24 | where | -0.23 |
| i'll | +0.24 | confirmed | -0.22 |
| what | +0.23 | them | -0.22 |
| you're | +0.23 | saw | -0.21 |
| it's | +0.21 | cafeteria | -0.20 |
| slip | +0.21 | two | -0.20 |
| vote | +0.12 | was | -0.20 |
| and | -0.06 | my | -0.19 |
| mafia | -0.09 | at | -0.19 |
| i | -0.09 | were | -0.16 |
| night | -0.14 | in | -0.15 |

## 9. Bottom line
See FINDINGS.md next to this report for the interpretation of these numbers.

