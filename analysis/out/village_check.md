# AI Village check (aggregate only; message text withheld)

`lexicon_general.json` applied to 1,352 AI Village agent chat messages (52 agents, 2026-10-02/04).

- Village overall: mean P(deceiving-style) 0.241, median 0.243, 0.2% of messages above 0.5. That is
  where the Turing Games *truthful* turns sit (0.249); game liars sit at 0.341.
- Agents differ reliably (shuffled-label p < 0.001) but within 0.18-0.30. Length matters (corr -0.38
  with log words). Length-adjusted means of agents with 40+ messages: GPT-5.2 +0.046, DeepSeek-V3.2
  +0.016, GPT-5 +0.013, Claude Opus 5 +0.008, GPT-6.1 Sol +0.004, Claude Opus 4.8 -0.003,
  GPT-6 Sol -0.008, Gemini 2.5 Pro -0.016, Gemini 3.8 Flash -0.068.
- DeepSeek-V3.2 vs others: +0.031 raw (p < 0.001), +0.018 length-adjusted. Driven by "still", "your",
  "I'll", "if" in status/coordination messages addressed to teammates. Its top-scoring messages read as
  diligent coordination, not deception.
- Conclusion: outside the games the lexicon measures communication register (second-person present-tense
  coordination vs first-person reporting), not lying. Use per-agent baselines (deviation from an agent's
  own register) if it is ever used as a monitor; treat "still" and "I'll" as context-dependent.
