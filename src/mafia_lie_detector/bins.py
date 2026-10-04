"""Plain-text bins for keyword lie detectors: one utterance per line.

Three tiers, each with a ``deceiving.txt`` and a ``truthful.txt``:

    all/                               general detector (model unknown)
    families/<family>/                 family detector (family known, model not)
    families/<family>/<model>/         per-model detector

Custom AIs whose underlying model is unknown (Z2) go under ``families/_unidentified/<name>/``
and are left out of ``all/`` and of every real family.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

SIDE = {"lying": "deceiving.txt", "truth": "truthful.txt"}


def write_bins(rows: Iterable[dict], out_dir: str | Path) -> dict:
    out = Path(out_dir)
    lines: dict[Path, list[str]] = {}
    counts: Counter = Counter()
    for r in rows:
        text = " ".join(r["text"].split())
        if not text:
            continue
        fname = SIDE[r["label"]]
        family = r["family"] if r.get("model_verified", True) else "_unidentified"
        targets = [out / "families" / family / r["model"] / fname, out / "families" / family / fname]
        if family != "_unidentified":
            targets.append(out / "all" / fname)
        for t in targets:
            lines.setdefault(t, []).append(text)
            counts[str(t.relative_to(out))] += 1
    for path, texts in lines.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    manifest = {"lines_per_file": dict(sorted(counts.items()))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
