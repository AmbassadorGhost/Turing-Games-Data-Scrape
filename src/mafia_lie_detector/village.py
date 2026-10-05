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


def date_range(days: int = 7, start: str | None = None, end: str | None = None) -> list[str]:
    """YYYY-MM-DD strings, oldest first: either the last `days` days, or start..end inclusive."""
    if start:
        a = dt.date.fromisoformat(start)
        b = dt.date.fromisoformat(end) if end else a
        if b < a:
            raise ValueError("end date is before start date")
        return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]
    today = dt.datetime.now(dt.timezone.utc).date()
    return [(today - dt.timedelta(days=back)).isoformat() for back in range(days - 1, -1, -1)]


def fetch_events(slug: str, days: int = 7, fetch: Callable[[str], dict] = http_json, *,
                 start: str | None = None, end: str | None = None) -> tuple[dict[str, str], list[dict]]:
    """-> ({agent id: name}, events for the requested days, oldest first)."""
    v = fetch(f"{API}/villages?slug={slug}")
    detail = fetch(f"{API}/villages/{v['id']}")
    id2name = {a["id"]: a["name"] for a in detail.get("agents", [])}
    events: list[dict] = []
    for day in date_range(days, start, end):
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


def add_baselines(scored: list[dict], min_messages: int = 20) -> None:
    """Per-agent z-score: how far each message sits from *that agent's own* typical register.

    Cross-agent levels mostly encode style (a coordinator scores higher than a narrator every day). A
    monitor should instead ask whether an agent has drifted from its own baseline, so each message gets
    ``z_vs_self`` = (p - agent mean) / agent sd, computed only for agents with enough history.
    """
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for m in scored:
        by_agent[m["agent"]].append(m)
    for ms in by_agent.values():
        ps = [m["p_deceiving"] for m in ms]
        if len(ps) < min_messages:
            for m in ms:
                m["z_vs_self"] = None
            continue
        mu, sd = statistics.fmean(ps), statistics.pstdev(ps) or 1e-9
        for m in ms:
            m["z_vs_self"] = round((m["p_deceiving"] - mu) / sd, 2)


def summarise(scored: list[dict], top_k: int = 10) -> dict:
    add_baselines(scored)
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for m in scored:
        by_agent[m["agent"]].append(m)
    agents = {}
    for name, ms in sorted(by_agent.items(), key=lambda kv: -len(kv[1])):
        ps = [m["p_deceiving"] for m in ms]
        with_z = [m for m in ms if m.get("z_vs_self") is not None]
        agents[name] = {
            "messages": len(ms), "mean": round(statistics.fmean(ps), 4), "median": round(statistics.median(ps), 4),
            "share_above_0_5": round(sum(p > 0.5 for p in ps) / len(ps), 4),
            "top": sorted(ms, key=lambda m: -m["p_deceiving"])[:top_k],
            "bottom": sorted(ms, key=lambda m: m["p_deceiving"])[:3],
            "drift": sorted(with_z, key=lambda m: -m["z_vs_self"])[:top_k],  # biggest departures from own baseline
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
        if a["drift"]:
            L.append(f"\n{name}: biggest departures from its own baseline (z vs self)\n")
            for m in a["drift"][:5]:
                L.append(f"- **z={m['z_vs_self']:+.1f}** (P={m['p_deceiving']:.2f}, {m['when'][:16]}) {m['text'][:200].replace(chr(10), ' ')}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
