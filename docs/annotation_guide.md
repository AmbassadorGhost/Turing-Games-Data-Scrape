# Annotation guide

One JSON file per game in `data/annotations/`. `mld draft` creates it from captions; you finish it.
Run `mld validate` as you go.

## 1. Players

From the role reveal (`mld reveals` prints timestamped links), list every player:

```json
{"player_id": "P3", "model": "llm_a", "role": "detective", "alignment": "truthful",
 "aliases": ["Three"]}
```

- `model` is the underlying LLM, or `human`. This is what `leave_model_out` groups on, so keep the
  spelling consistent across games.
- `alignment` is `deceiver` for the side that must lie to win (mafia / werewolf / impostor) and
  `truthful` otherwise. Say so explicitly; variants with odd roles (e.g. a third faction) need a
  judgement call, and the tools never guess.
- `aliases` are the names used in speech; `--scrub` removes them from the training text.

## 2. Turns

Each turn needs a `speaker_id` (a `player_id`, or `__narrator__` for host/voice-over lines). If the
captions had `Name:` prefixes the name is in `speaker_hint`; resolve it into `speaker_id`.

Set `phase` where it matters. These phases are **excluded** from training because they describe
outcomes: `intro`, `reveal`, `postgame`, `outro`. Use `day`, `vote`, `night` for real play.

## 3. Deception labels

- `deceptive`: `true` if the turn contains at least one claim the speaker **knew** was false.
  `false` if you judged it and found none. Leave it out (null) if you did not judge the turn;
  unjudged turns are skipped for the `turn_deceptive` target.
- `claims` (optional, recommended): itemise the checkable statements and mark each `truthful`.
  `deceptive` is then derived (and a contradicting value is rejected). Kinds: `role_claim`,
  `knowledge_claim`, `alibi`, `accusation`, `vote_intent`, `other`.

Judge truth relative to **the speaker's own knowledge**. A villager wrongly saying "P2 is mafia" is
mistaken, not lying; a mafia member saying "P2 is mafia" when they know P2 is town is lying. An
honest-but-wrong statement is `truthful: true`.

Only label what the transcript supports. Hidden game state (who was killed, who was checked) is
often not visible; skip those claims instead of guessing.

## 4. Finish

Set `annotation_status` to `reviewed`. `mld build-dataset` skips drafts and incomplete games and
tells you why. If an LLM proposed labels with `mld suggest`, read every proposal before reviewing.
