"""Build site/data/leaderboard.json from registry/models.yaml and results/*.json.

  python -m leaderboard_site.build [--registry registry/models.yaml] [--results results] [--out site/data/leaderboard.json]
"""

import argparse
import json
import re
import statistics
import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from discovery.models import RegistryEntry
from discovery.registry import REPO_ROOT, load_models

HUMAN_ID = "human-heldout"
HUMAN_NAME = "Held-out human (packed)"
TOLERANCE = 1e-3
_PARAMS = re.compile(r"^(\d+(?:\.\d+)?)([MB])$")


class BuildError(RuntimeError):
    pass


class Result(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    setting: str
    encoder: str
    metric: str
    folds: list[float]
    samples_per_fold: int
    mean: float
    std: float
    scored_on: date


def parse_params(text: str) -> int:
    m = _PARAMS.match(text.strip())
    if not m:
        raise BuildError(f"cannot parse params {text!r}; use a number followed by M or B, e.g. 355M or 1.5B")
    return round(float(m.group(1)) * (1e6 if m.group(2) == "M" else 1e9))


def load_results(directory: Path) -> dict[str, Result]:
    out = {}
    for path in sorted(directory.glob("*.json")):
        try:
            r = Result.model_validate_json(path.read_text())
        except ValidationError as exc:
            raise BuildError(f"{path.name}: invalid results file: {exc}") from exc
        if r.id != path.stem:
            raise BuildError(f"{path.name}: id is {r.id!r}; it must match the file name")
        if len(r.folds) < 2:
            raise BuildError(f"{path.name}: needs at least two folds")
        mean, std = statistics.mean(r.folds), statistics.stdev(r.folds)
        if abs(mean - r.mean) > TOLERANCE or abs(std - r.std) > TOLERANCE:
            raise BuildError(f"{path.name}: mean/std ({r.mean}, {r.std}) do not match its folds ({mean:.4f}, {std:.4f})")
        out[r.id] = r
    return out


def _paper_link(e: RegistryEntry) -> str:
    return f"https://arxiv.org/abs/{e.paper.arxiv}" if e.paper.arxiv else e.paper.url


def build(models: list[RegistryEntry], results: dict[str, Result]) -> dict:
    scored = {m.id: m for m in models if m.status == "scored"}
    if HUMAN_ID not in results:
        raise BuildError(f"results/{HUMAN_ID}.json (the held-out human reference) is missing")
    orphans = sorted(set(results) - set(scored) - {HUMAN_ID})
    if orphans:
        raise BuildError(f"results without a scored registry entry: {', '.join(orphans)}")
    missing = sorted(set(scored) - set(results))
    if missing:
        raise BuildError(f"scored registry entries without results: {', '.join(missing)}")
    generators = [{
        "id": e.id, "name": e.name, "group": e.group, "params": e.params, "params_value": parse_params(e.params),
        "released": e.paper.published.isoformat(), "paper": _paper_link(e), "code": e.github,
        "mean": round(results[e.id].mean, 4), "std": round(results[e.id].std, 4),
    } for e in scored.values()]
    generators.sort(key=lambda g: g["mean"])
    human = results[HUMAN_ID]
    last = max([r.scored_on for r in results.values()] + [e.added for e in scored.values()])
    return {"last_modified": last.isoformat(),
            "human": {"name": HUMAN_NAME, "mean": round(human.mean, 4), "std": round(human.std, 4)},
            "generators": generators}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, default=REPO_ROOT / "registry" / "models.yaml")
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "site" / "data" / "leaderboard.json")
    args = ap.parse_args(argv)
    try:
        data = build(load_models(args.registry), load_results(args.results))
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}: {len(data['generators'])} generators, last modified {data['last_modified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
