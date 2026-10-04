# Findings: how lying AIs talk differently (Turing Games, 902 utterances)

Companion to `report.md` (all tables) and `word_stats.csv` (every word). Data: 254 deceiving and
648 truthful public utterances, 20 games, 36 models, names scrubbed before counting.

## 1. There is a real but small signal, and it is about *stance*, not vocabulary

After controlling for game, speaker model, role, position in the game and game type, the words
that separate liars from truthful players are function words and discourse moves, not content words:

| liars say more | truthful players say more |
|---|---|
| **if**, **why**, **how** (hypotheticals and rhetorical questions: "if you're honest, why would...") | **I was**, **I saw**, **I found**, **with me** (first-person past-tense reports) |
| **your**, **you're** (talking about the listener) | **both**, **did**, **at**, **2** (specific, checkable detail) |
| **just**, **really**, **exactly** (minimising and emphasis) | **lights, security, electrical, east** (places; Among Us) |
| **agree**, **makes sense** (aligning with the group) | **claims**, **member**, **mafia** (naming the enemy role) |
| **abstain**, **keep**, **concrete**, **quiet**, **convenient**, **fast** | **!** (exclamation) |

Reading: liars steer the conversation (conditionals, questions back at the accuser, second person,
agreeing with whoever is winning the argument); truthful players report what they did and saw. Liars
also avoid naming the evil role: 16 of 18 models say "mafia/impostor" *less* when they are mafia
(sign test p < 0.01), and they use fewer exclamation marks (15 of 18 models, p = 0.01). This matches
the deception literature (less first-person detail, more other-focus), which is reassuring given the
text source caveat below.

What did **not** differ: utterance length, word length, question rate, hedging ("maybe", "probably"),
certainty words, explicit accusation verbs. Liars do not hedge more or accuse more.

## 2. How sure are we?

- **Permutation test** (labels shuffled among players within each game, 3,000 times): 35 of 461 words
  reach p < 0.05 where ~23 would by chance. Four survive a 10% false-discovery correction:
  **your, if, why, just**. So most individual words are noise; a handful are not.
- **Within-model consistency**: "why" leans deceiving in 13 of 14 models that use it, "keep" 10/10,
  "really" 9/9, "exactly" 11/13, "how" 13/15. Twenty words are consistent across models *and* pass
  the permutation test; they are the lexicon's backbone.
- **Other controls changed nothing**: removing sheriffs/doctors, stratifying by early/mid/late game,
  and stratifying by game type all leave the top words pointing the same way.
- **Caption-grade text** (the two re-done games, 107 utterances): 12 of 17 top words keep direction;
  "why" is +2.2 even there. Too small to confirm, consistent enough not to alarm.

## 3. The detector

`lexicon.json`: 40 signed word weights plus an intercept; `mld score lexicon.json "text"` gives
P(deceiving). Evaluated with the lexicon refit inside every fold:

| | held-out games | held-out models |
|---|---|---|
| keyword lexicon (40 words) | **AUC 0.63** | **AUC 0.60** |
| 20 robust words only | 0.70 (word list chosen on all data; optimistic) | 0.64 |
| count of {if, just, your, why}, no fitting | 0.63 | |
| TF-IDF + logistic regression (reference) | 0.61 | 0.57 |
| majority / chance | 0.50 | 0.50 |

A long keyword list is *worse* (300 words: 0.58): rare words overfit. Fewer, better-attested words
generalise, and the honest number for "does this transfer to a model we have never seen" is about
0.60: real, weak. Per-model AUCs when held out range from 0.29 (gemini-3-pro, 7 lying lines) to
0.96 (gpt-5.5, 3 lying lines); those extremes are sample noise.

## 4. The three tiers, with this data

1. **General detector**: viable at the "weak signal" level. Use the short lexicon.
2. **Family detector**: **not yet**. A lexicon trained only on a family's own games is *worse* than
   the general one for every family (e.g. xai 0.17 vs 0.42, anthropic 0.43 vs 0.47): 20 to 77 lying
   utterances per family is not enough to learn family tells, only to overfit. Until there is roughly
   5x more data per family, the family tier should fall back to the general lexicon.
3. **Per-model detector**: not evaluable; most models have under 20 lying utterances. The per-model
   "tells" table in the report is provisional and should not be used.

## 5. Caveats that bound everything above

- **90% of the text is Gemini's reconstruction of the videos**, not caption text. The signal could be
  partly how Gemini writes liars. Against that: the words that emerged are generic function words,
  they hold within each model, and they match the human deception literature. For that, real captions
  via `mld extract` are the fix.
- Labels are by role. A mafia player's honest sentence counts as deceiving. The detector therefore
  learns "talks like someone on the lying side", which is what the use case needs, but it caps the
  achievable accuracy.
- 902 utterances is small. Expect the specific word list to move as data grows; expect the stance
  pattern (conditionals, second person, agreement, no first-person reports) to stay.
