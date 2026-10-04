# Mafia lie detector

Tools for building a role-labelled transcript dataset from Mafia / Werewolf-style games played by
LLMs, and for training and honestly evaluating detectors that spot when a particular LLM is
lying. Inspired by the [Turing Games](https://www.youtube.com/@turing_games/videos) videos and
livestreams.

**Status (v0.1):** the full pipeline and baseline models exist and are tested. There is no real
data in this repo, and the YouTube-facing code has only been tested against fakes (the
environment it was built in could not reach youtube.com). Run the ingest step on your own machine.

## Why this is mostly an annotation problem

YouTube captions give you words and timestamps. They do **not** say who is speaking, which role
each player held, or which statements were lies. Those labels have to come from somewhere:

| Label | Where it comes from |
|---|---|
| Who spoke | `>>` markers / `Name:` prefixes if present, otherwise you (or on-screen overlays, or voice diarization later) |
| Each player's role | The end-of-game reveal. `mld reveals` finds the likely moments and prints timestamped links |
| Which turns were lies | Compare each claim to the speaker's true role and the facts of the game |

So the repo is a pipeline: fetch -> draft -> annotate (human, optionally LLM-assisted) -> build dataset -> evaluate.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[ingest,dev]'      # add ',whisper' and/or ',llm' as needed
pytest                              # runs offline
mld demo                            # whole pipeline on synthetic data (plumbing check only)
```

### 1. Collect transcripts (run locally; needs YouTube access)

```bash
mld list-videos                                  # defaults to @turing_games: /videos and /streams tabs
mld fetch-captions --ids data/raw/video_ids.tsv  # -> data/raw/captions/<id>.json, failures.tsv
mld transcribe --id VIDEO_ID                     # Whisper fallback (needs ffmpeg + '.[whisper]')
```

`fetch-captions` skips videos it already has, records per-video failures, and stops cleanly if
YouTube starts blocking your IP (common from cloud hosts; use a home connection or a proxy).

### 1b. Captions *and frames* (recommended for annotating)

Captions cannot tell you who is speaking, which model that is, or who won. Those are on screen, so
`mld extract` also samples frames:

```bash
mld extract VIDEO_ID_OR_URL [...]     # needs yt-dlp + ffmpeg on PATH; run where YouTube is reachable
```

For each video it writes `data/raw/<id>/` with `captions.en.srt`, `frames/` (one every 10 s over
the whole video, since name tags appear throughout), `reveal_frames/` (one every 3 s over the last
3 minutes), `frames.json` (file -> seconds) and `meta.json`. The video itself is deleted unless you
pass `--keep-video`. `mld draft` and `mld reveals` accept the `.srt` directly. Hand the folder to
whoever annotates; `data/raw/` is git-ignored.

### 2. Annotate

```bash
mld draft data/raw/captions/VIDEO_ID.json --title "..."   # -> data/annotations/yt-VIDEO_ID.json
mld reveals data/raw/captions/VIDEO_ID.json               # where do they reveal roles?
mld validate                                              # structural + completeness checks
```

Fill in `players` (with `model`, `role`, `alignment`), each turn's `speaker_id`/`phase`, and where
you can, `claims` with `truthful` flags. Set `annotation_status` to `reviewed` when done. See
[docs/annotation_guide.md](docs/annotation_guide.md). Once `players` is filled in,
`mld suggest GAME.json --model <model-id>` (needs `.[llm]` and `ANTHROPIC_API_KEY`) can propose
speakers, phases and claims for the blanks; it never overwrites your values or marks a game reviewed.

### 2b. Sort by model

```bash
mld sort --out dataset          # -> dataset/<family>/<exact-model>/<lying|truth>/<game>__<player>.jsonl
```

`lying` = a deceiver alignment (mafia, jester, any other lying role); `truth` = everything else.
The winner is stored in each row, not in the path. Discarded, with the reason printed and saved to
`dataset/_report.json`: human seats, players whose exact model was not verified (`model` is
`human` / `unknown`, or no `family`), drafts and incomplete games, and games where more than 40%
of turns have the speaker `__unknown__` (turns you could not attribute; tune with `--max-unknown`).
Custom or homemade AIs whose underlying model is unknown (set `family` to `custom`, e.g. Z2) are
kept apart in `dataset/_unidentified/<name>/<lying|truth>/` until they can be identified.
Each player's `evidence` field records how they were identified, for spot-checking. `dataset/` is
git-ignored. Re-sorting into an existing directory needs `--clean` (it only deletes directories it wrote).

### 2c. Clean training export

```bash
mld export --out data/clean   # turns.jsonl (public AI speech), private.jsonl, redacted games/, report.json
```

Removes every human line (including from the game files; human addressees become `human`), drops
unverified models, teasers/post-game, lines under 3 words and duplicates, strips stream cues, and
adds a deterministic train/test split by game, stratified by `text_provenance` (`captions` vs
`llm_reconstruction`). Only caption-grade text is trustworthy for wording-level features.

### 3. Build a dataset and evaluate

```bash
mld build-dataset --scrub -o data/dataset/turns.jsonl
mld evaluate data/dataset/turns.jsonl                              # held-out games
mld evaluate data/dataset/turns.jsonl --split leave_model_out      # unseen LLMs
mld evaluate data/dataset/turns.jsonl --model style                # style-only ablation
```

## What the evaluation guards against

- **Game leakage.** Turns from one game share names, events and phrasing, so a random turn split
  flatters the model. `group_kfold` holds out whole games.
- **Memorising an LLM's style.** Your goal is detecting lies from *particular* LLMs, so
  `leave_model_out` trains without one model and tests on it. If scores collapse there, the
  detector learned that model's voice, not deception. `--strict` also drops every game the
  held-out model played in from training.
- **Reading the answer key.** Intro, reveal and post-game phases (and narrator lines) are excluded,
  and the context window is built only from allowed turns, so "X was the mafia" cannot leak in.
- **Seat memorisation.** `--scrub` replaces player ids and aliases with `<PLAYER>`.
- **Weak labels.** `--target alignment` (mafia vs town) is cheap but noisy: mafia players mostly say
  true things. `turn_deceptive` needs claim-level annotation and is the label worth the effort.
- **Models that beat nothing.** Every report includes a majority-class baseline and
  `--model style|tfidf|tfidf_style` ablations (writing style versus content).

Known data caveats: Turing Games videos are edited (selection bias toward dramatic moments),
captions of synthesized speech are noisy, and humans seated among the LLMs are a different
population. Expect these to matter more than model choice.

## About the Gemini snippet

Three things in that conversation need correcting:

1. `YouTubeTranscriptApi.get_transcript(...)` **no longer exists** in `youtube-transcript-api` 1.x.
   The current interface is `YouTubeTranscriptApi().fetch(video_id, languages=[...])`, which is what
   `mld fetch-captions` uses. The old snippet would raise `AttributeError` on a fresh install.
2. Captions alone cannot give role labels; see above.
3. Its list of ready-made alternatives (AI-Werewolf, CICERO / Diplomacy, Hoodwinked, Avalon sets)
   is unverified: check each one's actual contents and license before relying on it. Nothing here
   converts them yet; mapping any of them onto `schema.Game` is the intended route.

## Layout

```
src/mafia_lie_detector/
  schema.py        Game / Player / Turn / Claim (pydantic) + JSON IO
  ingest/          youtube.py (list + captions), audio.py (Whisper fallback)
  annotate/        segment.py, reveal.py, draft.py, llm_assist.py
  dataset.py       games -> leakage-aware examples (JSONL)
  modeling.py      baselines, group / leave-model-out splits, metrics
  synthetic.py     fake games for tests and `mld demo` only
  cli.py           the `mld` command
tests/             offline tests (network and LLM calls are faked)
docs/              annotation guide
```

## Data and licensing

Transcripts of other people's videos are third-party content. `data/raw`, `data/audio`,
`data/dataset` and `data/annotations/*` are git-ignored on purpose. Check YouTube's terms and the
creators' wishes before publishing a dataset or model derived from them.
