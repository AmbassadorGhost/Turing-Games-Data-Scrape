"""Align annotated turns (LLM-reconstructed text with timestamps) to real YouTube captions.

For every public turn with a timestamp, search the caption cues within a time window for the
contiguous span that best covers the turn's content words. Report fidelity (what share of the
turn's content words appear in the matched span) and write a captions-text copy of each game.

usage: python analysis/align_captions.py MAPPING.json GAMES_DIR CAPTIONS_DIR OUT_DIR
MAPPING.json: {"<game_id>": {"video": "<video id>", "offset_seconds": 0}, ...}
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mafia_lie_detector.ingest.srt import parse_srt  # noqa: E402

WORD = re.compile(r"[a-z0-9']+")
STOP = set("the a an and or but to of in on at for with is are was were be been it its this that these those i you he she "
           "we they me him her us them my your his our their so if as not no do did does have has had will would can could "
           "just too very there here what who why how when then than also about into out up down over".split())
# caption-side normalisation of common speech-to-text mishearings of seat names
ALIASES = {"grock": "grok", "groc": "grok", "kimmy": "kimi", "yama": "llama", "chad": "chat", "40": "4o", "deepsea": "deepseek",
           "imposter": "impostor", "imposters": "impostors"}


def content_words(text: str) -> list[str]:
    toks = [ALIASES.get(t, t) for t in WORD.findall(text.lower())]
    return [t for t in toks if t not in STOP and len(t) > 1]


def best_span(cues: list[dict], turn_words: list[str], t0: float, before: float, after: float):
    """Best contiguous cue span inside [t0-before, t0+after] by content-word coverage.

    Coverage = share of the turn's distinct content words present in the span; a length penalty keeps
    the span from swallowing neighbours' speech.
    """
    window = [c for c in cues if t0 - before <= c["start"] <= t0 + after]
    if not window or not turn_words:
        return None
    need = Counter(turn_words)
    n_need = sum(need.values())
    best = None
    for i in range(len(window)):
        have: Counter = Counter()
        span_len = 0
        for j in range(i, min(len(window), i + 40)):
            cw = content_words(window[j]["text"])
            have.update(cw)
            span_len += len(cw)
            matched = sum(min(have[w], need[w]) for w in need)
            coverage = matched / n_need
            bloat = max(0, span_len - 1.15 * n_need) / max(n_need, 1)
            score = coverage - 0.35 * bloat
            if best is None or score > best[0]:
                best = (score, coverage, i, j)
    score, coverage, i, j = best
    # drop edge cues that share no content word with the turn: neighbours' speech at the window edges
    while i < j and not any(w in need for w in content_words(window[i]["text"])):
        i += 1
    while j > i and not any(w in need for w in content_words(window[j]["text"])):
        j -= 1
    span = window[i : j + 1]
    text = trim_to_speaker(" ".join(c["text"] for c in span), need)
    return {"coverage": round(coverage, 3), "start": span[0]["start"], "end": span[-1]["start"] + span[-1].get("duration", 0),
            "text": text}


def trim_to_speaker(text: str, need: Counter) -> str:
    """Captions mark speaker changes with '>>'. Keep only the contiguous run of segments that carry
    the turn's content words, so neighbours' speech at the window edges is cut away."""
    segs = [s.strip() for s in text.split(">>") if s.strip()]
    if len(segs) <= 1:
        return text.replace(">>", " ").strip()
    hits = [sum(1 for w in content_words(s) if w in need) for s in segs]
    if max(hits) == 0:
        return text.replace(">>", " ").strip()
    first = next(k for k, h in enumerate(hits) if h > 0)
    last = max(k for k, h in enumerate(hits) if h > 0)
    # drop edge segments that only brush the turn (1 shared word) when a strong core exists
    core = max(hits)
    while last > first and hits[last] <= 1 and core >= 3:
        last -= 1
    while first < last and hits[first] <= 1 and core >= 3:
        first += 1
    return " ".join(segs[first : last + 1])


def align_game(game: dict, cues: list[dict], offset: float, before: float = 45.0, after: float = 90.0) -> tuple[dict, dict]:
    out = json.loads(json.dumps(game))
    stats = Counter()
    coverages = []
    for t in out["turns"]:
        if t["speaker_id"] in ("__narrator__", "__unknown__") or t["phase"] not in ("day", "vote"):
            continue
        if t.get("start") is None:
            stats["no timestamp"] += 1
            continue
        words = content_words(t["text"])
        m = best_span(cues, words, t["start"] + offset, before, after)
        if m is None:
            stats["no captions in window"] += 1
            continue
        coverages.append(m["coverage"])
        t["caption_text"] = m["text"]
        t["caption_coverage"] = m["coverage"]
        t["caption_start"] = round(m["start"], 1)
        stats["aligned"] += 1
    out["text_provenance"] = "captions_aligned"
    report = {"game_id": game["game_id"], "turns": dict(stats),
              "coverage_mean": round(sum(coverages) / len(coverages), 3) if coverages else None,
              "coverage_quartiles": [round(sorted(coverages)[int(len(coverages) * q)], 2) for q in (0.25, 0.5, 0.75)] if coverages else None,
              "share_coverage_ge_0_6": round(sum(c >= 0.6 for c in coverages) / len(coverages), 3) if coverages else None}
    return out, report


def main(mapping_path, games_dir, captions_dir, out_dir):
    mapping = json.loads(Path(mapping_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for game_id, m in mapping.items():
        game = json.loads((Path(games_dir) / f"{game_id}.json").read_text(encoding="utf-8"))
        cues = parse_srt((Path(captions_dir) / m["video"] / "captions.en.srt").read_text(encoding="utf-8"))
        aligned, rep = align_game(game, cues, float(m.get("offset_seconds", 0)),
                                  before=float(m.get("before", 45)), after=float(m.get("after", 90)))
        rep["video"] = m["video"]
        (out / f"{game_id}.json").write_text(json.dumps(aligned, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        reports.append(rep)
        print(f"{game_id:<44} {m['video']} aligned={rep['turns'].get('aligned', 0):<4} mean coverage={rep['coverage_mean']} "
              f"quartiles={rep['coverage_quartiles']} >=0.6: {rep['share_coverage_ge_0_6']}")
    (out / "_alignment_report.json").write_text(json.dumps(reports, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:5])
