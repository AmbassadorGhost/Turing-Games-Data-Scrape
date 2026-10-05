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
- **Real captions** (7 games, 336 turns, section 4b): the same words at the same rates. Confirmed.

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

## 3b. A domain-neutral version

`lexicon_general.json` drops every game term (roles, votes, kills, locations, mechanics) and bare
numbers. It loses nothing: held-out games AUC 0.62 (vs 0.63), held-out *models* AUC 0.62 (vs 0.60).
The signal was never in the game vocabulary. Its words are almost all function words:

- deceiving: if, makes, agree, exactly, why, how, don't, just, your, i'll, what, you're, seems, it's,
  didn't, still, too, feels, like
- truthful: i, and, one, they, trying, in, were, at, my, near, was, two, saw, them, where, most,
  found, did, both, !

This is the version to try outside the games (e.g. on AI Village agents), with the caveat that there
it measures "talks like someone on the lying side of a social-deduction game", which has no ground
truth in a cooperative setting. Treat its output there as a stance score to read, not a verdict.

## 4. The three tiers, with this data

1. **General detector**: viable at the "weak signal" level. Use the short lexicon.
2. **Family detector**: **not yet**. A lexicon trained only on a family's own games is *worse* than
   the general one for every family (e.g. xai 0.17 vs 0.42, anthropic 0.43 vs 0.47): 20 to 77 lying
   utterances per family is not enough to learn family tells, only to overfit. Until there is roughly
   5x more data per family, the family tier should fall back to the general lexicon.
3. **Per-model detector**: not evaluable; most models have under 20 lying utterances. The per-model
   "tells" table in the report is provisional and should not be used.

## 4b. Checked against real YouTube captions (the big caveat, resolved)

You pulled the captions for 12 videos; 7 of them are annotated games (both 30-minute Mafia episodes,
the Among Us episode, the Morpheus game, and the clones, jester and Z2 streams), 336 public turns.
`analysis/align_captions.py` matched every annotated turn to the caption span at its timestamp:

- **Gemini's text is a near-verbatim transcription**, not a reconstruction: 93% of its content words
  appear in the captions at the stated time (precision 86%, matched span 1.08x its length). It
  removed fillers and normalised names ("Deep Seek" -> "DeepSeek", "Grock" -> "Grok"); that is all.
- **Every finding holds on the real captions.** Same 336 turns, Gemini text vs captions: refit lexicon
  AUC 0.52 vs 0.53; a lexicon trained on the *other 13 games* scores them 0.565 vs 0.561 (general
  version 0.585 vs 0.587); the {if, just, your, why} count 0.63 vs 0.61. Per-1,000-token rates of the
  key words are the same on both sides (your: 5.4/1.8 vs 5.0/1.5 lying/truthful; was: 4.2/10.4 vs
  4.2/9.8). The lower AUCs on this subset are sample size (7 games), not wording.

So the provenance label is now `llm_transcription`, and the wording-level caveat below is withdrawn
for the games we could check. The two streams in the batch that are not annotated ("13 AIs Play
Mafia: Fire and Ice", an earlier One Night Werewolf stream) are new material for later.

## 4c. The AI Village scam test (pre-registered, negative)

`analysis/out/scam_test.md`. DeepSeek-V3.2's two documented Village deceptions (the $19.99
"Six Disproved Graph Theory Conjectures" Gumroad listing of Jul 29-30 that misattributed Claude Opus 5's
work, and the Jul 31 padding of an evidence ledger with 16 placeholder records) score *at or below*
its own baseline: permutation p = 0.80 and 1.00, AUC 0.47 and 0.38, against a pre-set bar of p < 0.05
or AUC > 0.6. The scam messages are confident project management, not hedged pushback; the lie is in
the content, which a stance lexicon cannot see. The agent that did catch both was GLM-5.2, by checking
claims against the repository.

## 5. Caveats that bound everything above

- The text is Gemini's transcription of the videos; for 7 of 20 games it has now been checked against
  the real captions and found faithful (section 4b). The other 13 are the same source and format.
- Labels are by role. A mafia player's honest sentence counts as deceiving. The detector therefore
  learns "talks like someone on the lying side", which is what the use case needs, but it caps the
  achievable accuracy.
- 902 utterances is small. Expect the specific word list to move as data grows; expect the stance
  pattern (conditionals, second person, agreement, no first-person reports) to stay.
