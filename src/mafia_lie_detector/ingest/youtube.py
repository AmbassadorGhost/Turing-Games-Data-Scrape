"""List a channel's videos and download their captions.

Needs the ``ingest`` extra and network access to youtube.com. YouTube commonly blocks
cloud/datacenter IPs for caption requests, so run this from a normal home connection (or
configure a proxy for ``youtube-transcript-api``).
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

DEFAULT_CHANNEL = "https://www.youtube.com/@turing_games"
CHANNEL_TABS = ("videos", "streams")  # streams = past livestreams
_TAB_NAMES = frozenset({"videos", "streams", "shorts", "playlists", "featured"})
# Once YouTube starts refusing us, every further request will fail too; stop instead of hammering.
_BLOCKED = frozenset({"IpBlocked", "RequestBlocked"})
# Connectivity problems (no route, proxy refusal, TLS) also affect every remaining video.
_NETWORK = frozenset(
    {"ProxyError", "ConnectionError", "ConnectTimeout", "ReadTimeout", "SSLError", "NameResolutionError"}
)
# Failures that really mean "this video has no captions", where speech-to-text is the remedy.
NO_CAPTIONS = frozenset({"TranscriptsDisabled", "NoTranscriptFound"})


@dataclass(frozen=True)
class VideoRef:
    id: str
    title: str
    tab: str


def channel_tab_urls(channel: str, tabs: Iterable[str] = CHANNEL_TABS) -> list[tuple[str, str]]:
    """Expand ``@handle`` / channel URL into one ``(tab, url)`` pair per tab.

    A URL that already ends in a tab name is used as given.
    """
    channel = channel.strip().rstrip("/")
    if channel.startswith("@"):
        channel = f"https://www.youtube.com/{channel}"
    elif not channel.startswith("http"):
        channel = f"https://www.youtube.com/@{channel}"
    last = channel.rsplit("/", 1)[-1]
    if last in _TAB_NAMES:
        return [(last, channel)]
    return [(tab, f"{channel}/{tab}") for tab in tabs]


def list_videos(
    channel: str = DEFAULT_CHANNEL,
    tabs: Iterable[str] = CHANNEL_TABS,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[list[VideoRef], list[str]]:
    """Return ``(videos, warnings)``. A missing tab (e.g. no livestreams) is a warning, not an error."""
    videos: dict[str, VideoRef] = {}
    warnings: list[str] = []
    for tab, url in channel_tab_urls(channel, tabs):
        proc = runner(
            ["yt-dlp", "--flat-playlist", "--no-warnings", "--print", "%(id)s\t%(title)s", url],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            last = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
            warnings.append(f"{tab}: yt-dlp failed ({last[0][:200]})")
            continue
        for line in proc.stdout.splitlines():
            vid, _, title = line.partition("\t")
            vid = vid.strip()
            if vid and vid not in videos:
                videos[vid] = VideoRef(vid, title.strip(), tab)
    return list(videos.values()), warnings


def write_video_list(videos: Iterable[VideoRef], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in videos:
            f.write(f"{v.id}\t{v.tab}\t{v.title.replace(chr(9), ' ')}\n")


def read_video_ids(path: str | Path) -> list[str]:
    """First column of a TSV / one-id-per-line file; blank lines and ``#`` comments ignored."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line.split("\t")[0])
    return ids


def fetch_captions(video_id: str, languages: Iterable[str] = ("en",), *, api: Any = None) -> dict:
    """Fetch one video's captions as ``{"snippets": [{"text", "start", "duration"}, ...], ...}``.

    Uses the ``youtube-transcript-api`` 1.x interface (``YouTubeTranscriptApi().fetch``); the
    older ``get_transcript`` classmethod no longer exists.
    """
    if api is None:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=list(languages))
    return {
        "video_id": video_id,
        "source": "youtube_captions",
        "language_code": fetched.language_code,
        "is_generated": fetched.is_generated,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "snippets": fetched.to_raw_data(),
    }


def save_captions(data: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir) / f"{data['video_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_captions(path: str | Path) -> dict:
    """Load a captions JSON (from fetch-captions / transcribe) or an ``.srt`` (from ``mld extract``)."""
    path = Path(path)
    if path.suffix.lower() == ".srt":
        from .srt import parse_srt

        # data/raw/<id>/captions.en.srt names the video by its folder; <id>.en.srt by its file.
        vid = path.parent.name if path.name.startswith("captions") else path.name.split(".")[0]
        return {"video_id": vid, "source": "srt", "snippets": parse_srt(path.read_text(encoding="utf-8"))}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class FetchReport:
    saved: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already on disk
    failed: dict[str, str] = field(default_factory=dict)  # video_id -> "ErrorName: message"
    not_attempted: list[str] = field(default_factory=list)  # left over after the batch stopped early
    stopped: Optional[str] = None  # "blocked" (YouTube refused us) or "network" (cannot connect)

    @property
    def no_captions(self) -> list[str]:
        return [v for v, reason in self.failed.items() if reason.split(":", 1)[0] in NO_CAPTIONS]


def fetch_many(
    video_ids: Iterable[str],
    out_dir: str | Path,
    languages: Iterable[str] = ("en",),
    *,
    sleep: float = 1.5,
    overwrite: bool = False,
    api: Any = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> FetchReport:
    out_dir = Path(out_dir)
    ids = list(dict.fromkeys(video_ids))
    report = FetchReport()
    if api is None:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
    for n, vid in enumerate(ids):
        if (out_dir / f"{vid}.json").exists() and not overwrite:
            report.skipped.append(vid)
            continue
        try:
            save_captions(fetch_captions(vid, languages, api=api), out_dir)
            report.saved.append(vid)
        except Exception as exc:  # noqa: BLE001 - one bad video must not end the batch
            name = type(exc).__name__
            report.failed[vid] = f"{name}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}"
            if name in _BLOCKED or name in _NETWORK:
                report.stopped = "blocked" if name in _BLOCKED else "network"
                report.not_attempted = ids[n + 1 :]
                break
        sleeper(sleep)
    return report


def write_failures(report: FetchReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for vid, reason in report.failed.items():
            f.write(f"{vid}\t{reason}\n")
