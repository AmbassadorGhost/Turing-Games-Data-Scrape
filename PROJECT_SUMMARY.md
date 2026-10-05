# Mafia lie detector: project summary

Goal: learn how LLMs talk when they are lying, from Turing Games social-deduction videos (Mafia,
Among Us, One Night Werewolf), and build keyword detectors at three levels: general, per model
family, per model. Then try the general one on AI Village agents.

## What exists

| piece | where | state |
|---|---|---|
| Annotated games (20; speakers, exact models, roles, winners, timestamps) | `data/annotations/` (local), built from `scratchpad/annot/configs` | done, verified against game logic |
| Clean human-free export, train/test split by game | `mld export` -> `data/clean/` | done: 922 public turns, 265 lying / 657 truthful, 36 models |
| Plain-text bins: all / family / model x deceiving / truthful | `mld bins` -> `data/bins/` | done |
| Word analysis with confound controls | `analysis/word_analysis.py` -> `analysis/out/` | done; `FINDINGS.md` is the write-up |
| Keyword detector (general + domain-neutral) | `analysis/out/lexicon.json`, `lexicon_general.json`; `mld score` | done |
| Check of transcript fidelity against real YouTube captions | `analysis/align_captions.py`; FINDINGS 4b | done: transcripts are near-verbatim |
| AI Village scorer with per-agent baselines | `mld village` | done; `analysis/out/village_check.md` |

Data (`data/`, `dataset/`) is third-party transcript material and is git-ignored on purpose.
Deliverable zips were handed over in the session (dataset, bins, analysis, village check).

## Results in one paragraph

Liars and truthful players differ in *stance*, not vocabulary: liars use conditionals and questions
back at the accuser (if, why, how), second person (your, you're), minimisers and emphasis (just,
really, exactly) and agreement (agree, makes sense); truthful players give first-person reports with
specifics (I was, I saw, I found, with me). Liars also avoid naming the enemy role and use fewer
exclamation marks (both in 15+ of 18 models). The effect is real (permutation test within games,
consistent within models, unchanged by removing power roles, game position, game type) but small:
a 40-word lexicon reaches AUC 0.63 on held-out games and 0.60-0.63 on never-seen models (chance
0.50); counting {if, just, your, why} alone gives 0.63. Removing every game term costs nothing. Family-
and model-specific lexicons are *worse* than the general one with current data (too few lying turns
per family); they need ~5x more data. On AI Village, all agents score in the truthful band; agent
differences are real but reflect communication register (coordination vs reporting), with DeepSeek-V3.2
second behind GPT-5.2; the top-scoring messages are not deceptive on reading.

## How to run everything

```bash
pip install -e ".[ingest,dev]" && pytest
mld export && mld bins                                   # from data/annotations/
python analysis/word_analysis.py data/clean/turns.jsonl data/clean/games analysis/out
mld score analysis/out/lexicon_general.json "Why would I lie? If you're honest, your vote makes sense."
mld village analysis/out/lexicon_general.json --days 7 --agent DeepSeek   # needs theaidigest.org access
```

Adding a game: transcript in the quote-only `Speaker | Role | To` format (or captions + frames via
`mld extract`), a config in the annotation format (see `docs/annotation_guide.md`), rebuild, re-export.

## Limits to keep in mind

- Labels are by role: a mafia player's true sentence is still "deceiving". The detector learns
  "talks like someone on the lying side", not statement truth. Claim-level labels (`Turn.claims`) are
  the upgrade path and are mostly unfilled.
- 902 public turns is small. The stance pattern is stable; the exact word list will move.
- Outside the games the lexicon is a register meter. Use per-agent baselines (`z_vs_self` in
  `mld village`) rather than cross-agent levels, and treat "still" and "I'll" as context-dependent.
- Two caption streams in hand are unannotated: "13 AIs Play Mafia: Fire and Ice" (Claude Haiku, Qwen,
  Mistral; a roster not in the compilation) and an earlier One Night Werewolf stream. Gemini's
  transcription proved faithful, so it is the right tool to transcribe them.

## Open item for the owner

Commit `b1fad34` in this public repository still contains an exported transcript file that was pushed
by mistake and then removed. Options: rewrite history to drop it (force-push; irreversible) or make the
repository private. Not done, because it needs the owner's decision.
