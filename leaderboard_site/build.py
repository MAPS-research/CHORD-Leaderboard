"""Build site/data/leaderboard.json: the unconditional board (registry/models.yaml, results/*.json) and the
prefix-continuation board (registry/continuation.yaml, results/continuation-dclm/*.json, see continuation.py).

  python -m leaderboard_site.build [--registry registry/models.yaml] [--results results] [--out site/data/leaderboard.json]
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from discovery.models import RegistryEntry
from discovery.registry import REPO_ROOT, load_models
from leaderboard_site.common import HUMAN_ID, BuildError, check_folds, checkpoint_link, paper_link, parse_params
from leaderboard_site.continuation import build_continuation, load_entries, load_reported

HUMAN_NAME = "Held-out human (packed)"


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


def load_results(directory: Path) -> dict[str, Result]:
    out = {}
    for path in sorted(directory.glob("*.json")):
        try:
            r = Result.model_validate_json(path.read_text())
        except ValidationError as exc:
            raise BuildError(f"{path.name}: invalid results file: {exc}") from exc
        if r.id != path.stem:
            raise BuildError(f"{path.name}: id is {r.id!r}; it must match the file name")
        check_folds(path.name, r.folds, r.mean, r.std)
        out[r.id] = r
    return out


FAMILY_GROUP = {"ar": "ar", "continuous": "continuous", "flow": "continuous"}  # every other family is discrete diffusion/flow


def _group(e: RegistryEntry) -> str:
    return e.group or FAMILY_GROUP.get(e.family, "discrete")


def _pending(models: list[RegistryEntry]) -> list[dict]:
    """Registry entries found and accepted but not scored yet (sampler-only rows are not models of their own)."""
    rows = [{
        "id": e.id, "name": e.name, "group": _group(e), "params": e.params,
        "released": e.paper.published.isoformat() if e.paper and e.paper.published else None,
        "paper": paper_link(e.paper), "code": e.github, "checkpoint": checkpoint_link(e.checkpoint),
    } for e in models if e.status == "queued" and e.checkpoint.kind != "sampler"]
    rows.sort(key=lambda r: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", r["name"].lower())])  # "block size 4" before "16"
    rows.sort(key=lambda r: r["released"] or "", reverse=True)  # newest first, unknown dates last
    return rows


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
        "released": e.paper.published.isoformat(), "paper": paper_link(e.paper), "code": e.github,
        "mean": round(results[e.id].mean, 4), "std": round(results[e.id].std, 4),
        "checkpoint": checkpoint_link(e.checkpoint),
    } for e in scored.values()]
    generators.sort(key=lambda g: g["mean"])
    human = results[HUMAN_ID]
    last = max([r.scored_on for r in results.values()] + [e.added for e in scored.values()])
    return {"last_modified": last.isoformat(),
            "human": {"name": HUMAN_NAME, "mean": round(human.mean, 4), "std": round(human.std, 4)},
            "generators": generators,
            "pending": _pending(models)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, default=REPO_ROOT / "registry" / "models.yaml")
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--continuation-registry", type=Path, default=REPO_ROOT / "registry" / "continuation.yaml")
    ap.add_argument("--continuation-results", type=Path, default=REPO_ROOT / "results" / "continuation-dclm")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "site" / "data" / "leaderboard.json")
    args = ap.parse_args(argv)
    try:
        data = build(load_models(args.registry), load_results(args.results))
        cont = build_continuation(load_entries(args.continuation_registry), load_reported(args.continuation_results))
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    data = {**data, "last_modified": max(data["last_modified"], cont["last_modified"].isoformat()),
            "continuation": {"human": cont["human"], "generators": cont["generators"]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}: {len(data['generators'])} generators, {len(data['pending'])} pending, "
          f"{len(cont['generators'])} on the continuation board, last modified {data['last_modified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
