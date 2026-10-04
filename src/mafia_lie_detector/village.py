"""Score AI Village chat with a lexicon: a face-validity test outside the games.

AI Village is cooperative, so there is no deception ground truth here. The output is a stance score
("talks like someone on the lying side of a social-deduction game") per message and per agent, with
the highest-scoring messages listed so a human can judge whether they look like steering/evasion.

Village API shape (same endpoints the AI Village heat monitor reads):
  GET /villages?slug=<slug>            -> {"id": ...}
  GET /villages/<id>                   -> {"agents": [{"id", "name"}, ...]}
  GET /events?villageId=<id>&date=YYYY-MM-DD -> {"events": [{"createdAt", "data": {...}}, ...]}
Chat messages are events whose data.actionType == "AGENT_TALK"; the text is data.content.
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from .lexicon import Lexicon, name_pattern

API = "https://theaidigest.org/village/api"
DEFAULT_SLUG = "actual-launch-1"


def http_json(url: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "mafia-lie-detector/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_events(slug: str, days: int, fetch: Callable[[str], dict] = http_json) -> tuple[dict[str, str], list[dict]]:
    """-> ({agent id: name}, events for the last `days` days, oldest first)."""
    v = fetch(f"{API}/villages?slug={slug}")
    detail = fetch(f"{API}/villages/{v['id']}")
    id2name = {a["id"]: a["name"] for a in detail.get("agents", [])}
    events: list[dict] = []
    today = dt.datetime.now(dt.timezone.utc).date()
    for back in range(days - 1, -1, -1):
        day = (today - dt.timedelta(days=back)).strftime("%Y-%m-%d")
        events += fetch(f"{API}/events?villageId={v['id']}&date={day}").get("events") or []
    return id2name, events


def chat_messages(events: Iterable[dict], id2name: dict[str, str]) -> list[dict]:
    """Agent chat lines only: [{agent, when, text}] in time order. Humans and non-chat events dropped."""
    out = []
    for e in events:
        d = e.get("data") or {}
        if d.get("actionType") != "AGENT_TALK":
            continue
        aid = d.get("agentId") or d.get("speakerId")
        text = (d.get("content") or "").strip()
        if aid in id2name and text:
            out.append({"agent": id2name[aid], "when": e.get("createdAt", ""), "text": text})
    out.sort(key=lambda m: m["when"])
    return out


def score_messages(messages: list[dict], lex: Lexicon, agent_names: Iterable[str] = ()) -> list[dict]:
    scrub = name_pattern(agent_names)
    return [{**m, "p_deceiving": round(lex.probability(m["text"], scrub), 4)} for m in messages]


def summarise(scored: list[dict], top_k: int = 10) -> dict:
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for m in scored:
        by_agent[m["agent"]].append(m)
    agents = {}
    for name, ms in sorted(by_agent.items(), key=lambda kv: -len(kv[1])):
        ps = [m["p_deceiving"] for m in ms]
        agents[name] = {
            "messages": len(ms), "mean": round(statistics.fmean(ps), 4), "median": round(statistics.median(ps), 4),
            "share_above_0_5": round(sum(p > 0.5 for p in ps) / len(ps), 4),
            "top": sorted(ms, key=lambda m: -m["p_deceiving"])[:top_k],
            "bottom": sorted(ms, key=lambda m: m["p_deceiving"])[:3],
        }
    all_p = [m["p_deceiving"] for m in scored]
    return {"messages": len(scored), "overall_mean": round(statistics.fmean(all_p), 4) if all_p else None, "agents": agents}


def write_markdown(summary: dict, path: str | Path, focus: str | None = None) -> None:
    L = [f"# AI Village stance scores\n", f"{summary['messages']} agent chat messages; overall mean P(deceiving-style) "
         f"= {summary['overall_mean']}. **No ground truth exists here**: this is a reading aid, not a verdict.\n",
         "| agent | messages | mean | median | share > 0.5 |", "|---|---|---|---|---|"]
    for name, a in summary["agents"].items():
        L.append(f"| {name} | {a['messages']} | {a['mean']} | {a['median']} | {a['share_above_0_5']:.0%} |")
    for name, a in summary["agents"].items():
        if focus and focus.lower() not in name.lower():
            continue
        L.append(f"\n## {name}: highest-scoring messages\n")
        for m in a["top"]:
            L.append(f"- **{m['p_deceiving']:.2f}** ({m['when'][:16]}) {m['text'][:300].replace(chr(10), ' ')}")
        L.append(f"\n{name}: lowest-scoring\n")
        for m in a["bottom"]:
            L.append(f"- **{m['p_deceiving']:.2f}** {m['text'][:200].replace(chr(10), ' ')}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
