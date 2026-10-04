"""Command-line entry point: ``mld <command>``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from pydantic import ValidationError

from . import dataset, modeling, schema
from .ingest import youtube

RAW_DIR = Path("data/raw")
CAPTIONS_DIR = RAW_DIR / "captions"
ANNOTATIONS_DIR = Path("data/annotations")


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


# ----------------------------------------------------------------------------- ingest

def cmd_list_videos(args: argparse.Namespace) -> int:
    videos, warnings = youtube.list_videos(args.channel, args.tabs)
    for w in warnings:
        _err(f"warning: {w}")
    if not videos:
        _err("no videos found (is yt-dlp installed, and can this machine reach YouTube?)")
        return 1
    youtube.write_video_list(videos, args.output)
    print(f"{len(videos)} videos -> {args.output}")
    return 0


def cmd_fetch_captions(args: argparse.Namespace) -> int:
    ids = list(args.id or [])
    if args.ids:
        ids += youtube.read_video_ids(args.ids)
    if not ids:
        _err("give --ids FILE and/or --id VIDEO_ID")
        return 2
    report = youtube.fetch_many(ids, args.out, args.lang, sleep=args.sleep, overwrite=args.overwrite)
    print(f"saved {len(report.saved)}, already had {len(report.skipped)}, failed {len(report.failed)}")
    if report.failed:
        failures = Path(args.out) / "failures.tsv"
        youtube.write_failures(report, failures)
        print(f"failures -> {failures}")
    if report.no_captions:
        print(f"{len(report.no_captions)} video(s) have no captions; try: mld transcribe "
              + " ".join(f"--id {v}" for v in report.no_captions[:3]) + (" ..." if len(report.no_captions) > 3 else ""))
    if report.stopped:
        why = (
            "YouTube is blocking caption requests from this IP (common on cloud hosts)"
            if report.stopped == "blocked"
            else "cannot connect to YouTube"
        )
        _err(
            f"{why}; stopped with {len(report.not_attempted)} video(s) not attempted. Fix the "
            f"connection (or use a proxy) and re-run; finished videos are skipped automatically."
        )
        return 1
    return 1 if report.failed and not report.saved and not report.skipped else 0


def cmd_extract(args: argparse.Namespace) -> int:
    from .ingest import extract

    rc = 0
    for ref in args.refs:
        try:
            meta = extract.extract_video(
                ref, args.out, frame_every=args.frame_every, tail_seconds=args.tail_seconds,
                tail_every=args.tail_every, height=args.height, keep_video=args.keep_video,
            )
        except (RuntimeError, ValueError) as exc:
            _err(f"{ref}: {exc}")
            rc = 1
            continue
        caps = ", ".join(meta["captions"]) or "NO CAPTIONS (use `mld transcribe`)"
        print(f"{meta['video_id']}: {meta['n_frames']} frames + {meta['n_reveal_frames']} reveal frames, captions: {caps}")
    return rc


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .ingest import audio

    rc = 0
    model = None
    for vid in args.id:
        try:
            wav = audio.download_audio(vid, args.audio_dir)
            if model is None:
                from faster_whisper import WhisperModel

                model = WhisperModel(args.model_size, device="auto", compute_type="auto")
            data = audio.transcribe(wav, video_id=vid, model=model, language=args.lang)
            path = youtube.save_captions(data, args.out)
            print(f"{vid}: {len(data['snippets'])} segments -> {path}")
        except Exception as exc:  # noqa: BLE001
            _err(f"{vid}: {exc}")
            rc = 1
    return rc


# -------------------------------------------------------------------------- annotation

def cmd_draft(args: argparse.Namespace) -> int:
    from .annotate import draft

    captions = youtube.load_captions(args.captions)
    utterances = draft.utterances_from_captions(captions, args.names)
    game = draft.make_draft(captions["video_id"], utterances, title=args.title, source_kind=captions.get("source", ""))
    if args.roster:
        game = draft.apply_roster(game, schema.load_game(args.roster))
    out = Path(args.output) if args.output else ANNOTATIONS_DIR / f"{game.game_id}.json"
    if out.exists() and not args.force:
        _err(f"{out} exists; refusing to overwrite hand-labelled work (use --force)")
        return 1
    schema.save_game(game, out)
    print(f"{len(game.turns)} draft turns -> {out}")
    return 0


def cmd_reveals(args: argparse.Namespace) -> int:
    from .annotate import draft, reveal

    captions = youtube.load_captions(args.captions)
    utterances = draft.utterances_from_captions(captions)
    found = reveal.find_reveal_candidates(utterances, args.min_score)
    for c in found:
        print(f"score {c.score}  {reveal.youtube_link(captions['video_id'], c.start)}  {c.text[:100]}")
    if not found:
        print("no reveal-like lines found; check the end of the video by hand")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    files = schema.find_game_files(args.paths)
    if not files:
        _err("no game files found")
        return 2
    bad = 0
    for f in files:
        try:
            game = schema.load_game(f)
        except (ValidationError, ValueError) as exc:
            _err(f"{f}: INVALID\n  " + str(exc).replace("\n", "\n  "))
            bad += 1
            continue
        problems = game.completeness_problems()
        labelled = sum(t.deceptive is not None for t in game.turns)
        state = "ok" if not problems else "incomplete"
        print(f"{f}: {game.annotation_status}, {state}, {len(game.turns)} turns, {labelled} labelled")
        for p in problems:
            print(f"  - {p}")
        if problems and game.annotation_status == "reviewed":
            bad += 1
    return 1 if bad else 0


def cmd_suggest(args: argparse.Namespace) -> int:
    from .annotate import llm_assist

    model = args.model or os.environ.get("MLD_ANNOTATION_MODEL")
    if not model:
        _err("pass --model or set MLD_ANNOTATION_MODEL (and ANTHROPIC_API_KEY)")
        return 2
    game = schema.load_game(args.game)
    try:
        counts = llm_assist.annotate_game(game, llm_assist.anthropic_completer(model), chunk_size=args.chunk_size)
    except (ValueError, RuntimeError) as exc:
        _err(str(exc))
        return 1
    schema.save_game(game, args.output or args.game)
    print(f"filled {counts['speaker']} speakers, {counts['phase']} phases, {counts['claims']} claim sets; status stays draft")
    return 0


# ------------------------------------------------------------------- dataset / modelling

def cmd_build_dataset(args: argparse.Namespace) -> int:
    result = dataset.build_examples(
        schema.iter_games(args.paths),
        args.target,
        exclude_phases=args.exclude_phase or dataset.DEFAULT_EXCLUDED_PHASES,
        context_turns=args.context,
        scrub=args.scrub,
        require_reviewed=not args.include_drafts,
    )
    for gid, reasons in result.skipped_games.items():
        _err(f"skipped {gid}: {'; '.join(reasons)}")
    n = dataset.write_examples(result.examples, args.output)
    rate = dataset.class_balance(result.examples)
    print(f"{n} examples from {len({e.game_id for e in result.examples})} games -> {args.output}")
    if rate is not None:
        print(f"positive rate {rate:.1%}")
    return 0 if n else 1


def cmd_sort(args: argparse.Namespace) -> int:
    from . import sorting

    try:
        if args.clean:
            sorting.clean_output(args.out)
        report = sorting.sort_games(
            schema.iter_games(args.paths), args.out,
            require_reviewed=not args.include_drafts, max_unknown=args.max_unknown,
        )
    except ValueError as exc:
        _err(str(exc))
        return 1
    for gid, reasons in report.discarded_games.items():
        _err(f"discarded game {gid}: {'; '.join(reasons)}")
    for key, why in report.discarded_players.items():
        _err(f"discarded player {key}: {why}")
    if report.unknown_winner:
        _err(f"winner not recorded for: {', '.join(report.unknown_winner)}")
    for folder, n in sorted(report.written.items()):
        print(f"{folder}: {n} turns")
    print(f"{report.files} files -> {args.out} (summary: {args.out}/{sorting.REPORT_NAME})")
    return 0 if report.files else 1


def cmd_export(args: argparse.Namespace) -> int:
    from . import export

    report = export.export_clean(
        schema.iter_games(args.paths), args.out,
        min_words=args.min_words, test_fraction=args.test_fraction, require_reviewed=not args.include_drafts,
    )
    for gid, reasons in report.dropped_games.items():
        _err(f"dropped game {gid}: {'; '.join(reasons)}")
    print(f"public rows: {report.rows['public']} ({dict(report.labels)}), private rows: {report.rows['private']}")
    print(f"splits: {dict(report.by_split)} | provenance: {dict(report.by_provenance)}")
    print("dropped turns: " + ", ".join(f"{k}={v}" for k, v in report.dropped.most_common()))
    print(f"-> {args.out}/turns.jsonl, private.jsonl, games/, report.json")
    return 0 if report.rows["public"] else 1


def cmd_bins(args: argparse.Namespace) -> int:
    from . import bins

    manifest = bins.write_bins(bins.read_rows(args.turns), args.out)
    top = {k: v for k, v in manifest["lines_per_file"].items() if k.startswith("all/")}
    print(f"{top} -> {args.out} (full counts in manifest.json)")
    return 0 if top else 1


def cmd_score(args: argparse.Namespace) -> int:
    from .lexicon import Lexicon, name_pattern

    lex = Lexicon.load(args.lexicon)
    scrub = name_pattern()
    texts = args.text or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    for t in texts:
        print(f"{lex.probability(t, scrub):.3f}\t{t}")
    return 0


def _print_report(report: dict) -> None:
    def row(name: str, m: dict) -> str:
        return (
            f"{name:<18} n={m['n']:<6} auc={m['roc_auc']:.3f}  ap={m['avg_precision']:.3f}  "
            f"f1={m['f1']:.3f}  bal_acc={m['balanced_acc']:.3f}"
        )

    print(f"split={report['split']} model={report['model']} context={report['use_context']} "
          f"examples={report['n_examples']} positive_rate={report['positive_rate']:.1%}")
    for fold in report["folds"]:
        print(row(str(fold["held_out"]), fold["metrics"]))
    print("-" * 80)
    print(row("POOLED", report["pooled"]))
    print(row("majority baseline", report["pooled_majority_baseline"]))


def cmd_evaluate(args: argparse.Namespace) -> int:
    examples = dataset.read_examples(args.dataset)
    try:
        report = modeling.evaluate(
            examples,
            split=args.split,
            model_kind=args.model,
            n_splits=args.folds,
            use_context=args.use_context,
            drop_shared_games=args.strict,
            min_test=args.min_test,
            seed=args.seed,
        )
    except ValueError as exc:
        _err(str(exc))
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .synthetic import make_games

    print("SYNTHETIC DATA: template text with a planted signal. This checks the plumbing only;")
    print("the numbers say nothing about real LLM deception.\n")
    games = make_games(args.games, seed=args.seed)
    built = dataset.build_examples(games, "turn_deceptive", scrub=True)
    print(f"{len(built.examples)} examples from {len(games)} games "
          f"(reveal/intro/narrator turns excluded)\n")
    for split in modeling.SPLITS:
        _print_report(modeling.evaluate(built.examples, split=split, seed=args.seed))
        print()
    return 0


# ----------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mld", description="Mafia lie detector toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list-videos", help="list a channel's videos and past livestreams (yt-dlp)")
    s.add_argument("channel", nargs="?", default=youtube.DEFAULT_CHANNEL)
    s.add_argument("--tabs", nargs="+", default=list(youtube.CHANNEL_TABS))
    s.add_argument("-o", "--output", default=str(RAW_DIR / "video_ids.tsv"))
    s.set_defaults(func=cmd_list_videos)

    s = sub.add_parser("fetch-captions", help="download captions for video ids")
    s.add_argument("--ids", help="file from list-videos (first column = id)")
    s.add_argument("--id", action="append", help="a single video id (repeatable)")
    s.add_argument("--out", default=str(CAPTIONS_DIR))
    s.add_argument("--lang", nargs="+", default=["en"])
    s.add_argument("--sleep", type=float, default=1.5, help="seconds between requests")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_fetch_captions)

    s = sub.add_parser("extract", help="captions + sampled frames for annotating speakers/roles (needs yt-dlp, ffmpeg)")
    s.add_argument("refs", nargs="+", help="video ids or URLs")
    s.add_argument("--out", default=str(RAW_DIR))
    s.add_argument("--frame-every", type=float, default=10.0, help="seconds between frames over the whole video")
    s.add_argument("--tail-seconds", type=float, default=180.0, help="length of the denser end-of-game pass")
    s.add_argument("--tail-every", type=float, default=3.0, help="seconds between frames in the end pass")
    s.add_argument("--height", type=int, default=480, help="max video height to download")
    s.add_argument("--keep-video", action="store_true", help="keep the downloaded video file")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("transcribe", help="speech-to-text fallback for videos without captions")
    s.add_argument("--id", action="append", required=True)
    s.add_argument("--audio-dir", default="data/audio")
    s.add_argument("--out", default=str(CAPTIONS_DIR))
    s.add_argument("--model-size", default="small")
    s.add_argument("--lang", default="en")
    s.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("draft", help="turn a captions file into a draft game to annotate")
    s.add_argument("captions")
    s.add_argument("-o", "--output")
    s.add_argument("--title", default="")
    s.add_argument("--names", nargs="*", help="speaker names to detect as 'Name: text' prefixes")
    s.add_argument("--roster", help="answer-key game file (e.g. ground_truth/<episode>.json) to pre-fill players/winner")
    s.add_argument("--force", action="store_true", help="overwrite an existing annotation file")
    s.set_defaults(func=cmd_draft)

    s = sub.add_parser("reveals", help="print likely role-reveal moments with timestamped links")
    s.add_argument("captions")
    s.add_argument("--min-score", type=int, default=2)
    s.set_defaults(func=cmd_reveals)

    s = sub.add_parser("validate", help="check annotation files")
    s.add_argument("paths", nargs="*", default=[str(ANNOTATIONS_DIR)])
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("suggest", help="LLM-assisted speaker/claim suggestions (needs the llm extra)")
    s.add_argument("game")
    s.add_argument("--model", help="model id (or set MLD_ANNOTATION_MODEL)")
    s.add_argument("--chunk-size", type=int, default=40)
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_suggest)

    s = sub.add_parser("build-dataset", help="flatten reviewed games into JSONL examples")
    s.add_argument("paths", nargs="*", default=[str(ANNOTATIONS_DIR)])
    s.add_argument("-o", "--output", default="data/dataset/turns.jsonl")
    s.add_argument("--target", choices=dataset.TARGETS, default="turn_deceptive")
    s.add_argument("--context", type=int, default=0, help="previous turns to include as context")
    s.add_argument("--scrub", action="store_true", help="replace player names with <PLAYER>")
    s.add_argument("--exclude-phase", action="append", help="phase to drop (repeatable; replaces the defaults)")
    s.add_argument("--include-drafts", action="store_true", help="also use games not marked reviewed")
    s.set_defaults(func=cmd_build_dataset)

    s = sub.add_parser("sort", help="sort reviewed games into dataset/<family>/<model>/<lying|truth>/")
    s.add_argument("paths", nargs="*", default=[str(ANNOTATIONS_DIR)])
    s.add_argument("--out", default="dataset")
    s.add_argument("--max-unknown", type=float, default=0.4, help="discard games with a larger share of unattributed turns")
    s.add_argument("--include-drafts", action="store_true")
    s.add_argument("--clean", action="store_true", help="first delete a previous sort output (only if it has _report.json)")
    s.set_defaults(func=cmd_sort)

    s = sub.add_parser("export", help="clean, human-free training export (turns.jsonl + redacted games)")
    s.add_argument("paths", nargs="*", default=[str(ANNOTATIONS_DIR)])
    s.add_argument("--out", default="data/clean")
    s.add_argument("--min-words", type=int, default=3)
    s.add_argument("--test-fraction", type=float, default=0.2, help="share of games held out as test")
    s.add_argument("--include-drafts", action="store_true")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("bins", help="plain-text deceiving/truthful bins: all, per family, per model")
    s.add_argument("turns", nargs="?", default="data/clean/turns.jsonl", help="turns.jsonl from `mld export`")
    s.add_argument("--out", default="data/bins")
    s.set_defaults(func=cmd_bins)

    s = sub.add_parser("score", help="score text with a lexicon: P(deceiving) per line")
    s.add_argument("lexicon", help="lexicon.json from analysis/word_analysis.py")
    s.add_argument("text", nargs="*", help="utterances (or pipe one per line on stdin)")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("evaluate", help="cross-validate a baseline detector")
    s.add_argument("dataset")
    s.add_argument("--split", choices=modeling.SPLITS, default="group_kfold")
    s.add_argument("--model", choices=modeling.MODEL_KINDS, default="tfidf_style")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--use-context", action="store_true")
    s.add_argument("--strict", action="store_true", help="leave_model_out: also drop shared games from training")
    s.add_argument("--min-test", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("demo", help="run the pipeline on synthetic data (plumbing check)")
    s.add_argument("--games", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        _err(f"not found: {exc.filename or exc}")
        return 2
    except ModuleNotFoundError as exc:
        _err(f"missing dependency: {exc.name}. Install the matching extra, e.g. pip install -e '.[ingest]'")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
