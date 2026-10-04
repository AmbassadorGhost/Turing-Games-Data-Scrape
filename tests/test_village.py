import json

from mafia_lie_detector import cli, village
from mafia_lie_detector.lexicon import Lexicon


def fake_events():
    agents = {"a1": "DeepSeek V3.2", "a2": "Claude Opus 4.5"}
    events = [
        {"createdAt": "2026-10-04T10:00:00Z", "data": {"actionType": "AGENT_TALK", "agentId": "a1",
                                                       "content": "Why would you check that? If you're honest, it just makes sense."}},
        {"createdAt": "2026-10-04T09:00:00Z", "data": {"actionType": "AGENT_TALK", "agentId": "a2",
                                                       "content": "I was in the doc and I saw the bug at line two."}},
        {"createdAt": "2026-10-04T09:30:00Z", "data": {"actionType": "USER_TALK", "speakerId": "h1", "speakerType": "HUMAN",
                                                       "content": "human line must be ignored"}},
        {"createdAt": "2026-10-04T09:40:00Z", "data": {"actionType": "PAUSE", "agentId": "a1", "seconds": 60}},
        {"createdAt": "2026-10-04T09:50:00Z", "data": {"actionType": "AGENT_TALK", "agentId": "a1", "content": "   "}},
    ]
    return agents, events


def test_chat_messages_filters_and_orders():
    agents, events = fake_events()
    msgs = village.chat_messages(events, agents)
    assert [m["agent"] for m in msgs] == ["Claude Opus 4.5", "DeepSeek V3.2"]
    assert all("human" not in m["text"] for m in msgs)


def test_scoring_and_summary(tmp_path):
    agents, events = fake_events()
    lex = Lexicon({"why": 0.8, "if": 0.8, "just": 0.6, "makes": 0.6, "i": -0.4, "saw": -0.5, "was": -0.4}, -0.9)
    scored = village.score_messages(village.chat_messages(events, agents), lex, agents.values())
    by = {m["agent"]: m["p_deceiving"] for m in scored}
    assert by["DeepSeek V3.2"] > 0.5 > by["Claude Opus 4.5"]
    summary = village.summarise(scored, top_k=5)
    assert summary["messages"] == 2 and set(summary["agents"]) == set(agents.values())
    village.write_markdown(summary, tmp_path / "s.md", focus="deepseek")
    text = (tmp_path / "s.md").read_text()
    assert "DeepSeek V3.2: highest-scoring" in text and "Claude Opus 4.5: highest-scoring" not in text


def test_fetch_events_uses_api_shape():
    calls = []

    def fetch(url):
        calls.append(url)
        if "villages?slug=" in url:
            return {"id": "V1"}
        if url.endswith("/villages/V1"):
            return {"agents": [{"id": "a1", "name": "DeepSeek V3.2"}]}
        return {"events": [{"createdAt": url[-10:], "data": {}}]}

    id2name, events = village.fetch_events("some-slug", days=3, fetch=fetch)
    assert id2name == {"a1": "DeepSeek V3.2"} and len(events) == 3
    assert calls[0].endswith("villages?slug=some-slug") and sum("events?villageId=V1&date=" in c for c in calls) == 3


def test_cli_offline(tmp_path, capsys):
    agents, events = fake_events()
    (tmp_path / "ev.json").write_text(json.dumps({"agents": agents, "events": events}))
    Lexicon({"why": 0.8, "saw": -0.5}, -0.5).save(tmp_path / "lex.json")
    assert cli.main(["village", str(tmp_path / "lex.json"), "--events", str(tmp_path / "ev.json"),
                     "--out", str(tmp_path / "out"), "--agent", "DeepSeek"]) == 0
    assert "DeepSeek V3.2" in capsys.readouterr().out
    assert (tmp_path / "out" / "summary.md").exists() and (tmp_path / "out" / "scored_messages.jsonl").exists()
