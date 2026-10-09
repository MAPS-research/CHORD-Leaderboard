"""Recall of a dry-run discovery pass against the seeded registry.

  python -m discovery run --since 2025-01-01 --dry-run --no-dedupe --max-judge-calls 1000 --out backtest.jsonl
  python tests/backtest.py backtest.jsonl --since 2025-01-01

Targets are the registry papers with an arXiv id from the --since month onward (sampler-only rows
excluded). A target counts as found when a reported candidate shares any key with it.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from discovery.keys import arxiv_key
from discovery.models import Candidate
from discovery.registry import REGISTRY_DIR, load_models

TARGET_RECALL = 0.90
MAX_PER_WEEK = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--since", type=date.fromisoformat, required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    reported = [Candidate.model_validate(r["candidate"]) for r in rows if r["reported"]]
    found = set().union(*(c.keys() for c in reported))

    cutoff = args.since.strftime("%y%m")
    targets: dict[str, tuple[str, set[str]]] = {}
    unmeasured = []
    for m in load_models(REGISTRY_DIR / "models.yaml"):
        if m.checkpoint.kind == "sampler":
            continue
        if not (m.paper and m.paper.arxiv):
            unmeasured.append(m.id)
            continue
        if m.paper.arxiv[:4] >= cutoff:
            targets.setdefault(arxiv_key(m.paper.arxiv), (m.name, set()))[1].update(m.keys())

    missed = sorted(f"{name} ({key})" for key, (name, keys) in targets.items() if not keys & found)
    recall = 1 - len(missed) / len(targets) if targets else 0.0
    weeks = max((date.today() - args.since).days / 7, 1)
    per_week = len(reported) / weeks
    print(f"targets: {len(targets)}  recall: {recall:.2%}  (goal >= {TARGET_RECALL:.0%})")
    print(f"reported: {len(reported)} over {weeks:.1f} weeks = {per_week:.1f}/week  (goal <= {MAX_PER_WEEK})")
    print("missed:", *missed, sep="\n  ") if missed else print("missed: none")
    print(f"not measurable (no arXiv id in registry): {len(unmeasured)} rows")
    return 0 if recall >= TARGET_RECALL and per_week <= MAX_PER_WEEK else 1


if __name__ == "__main__":
    sys.exit(main())
