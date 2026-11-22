"""The prefix-continuation board: generators listed in registry/continuation.yaml, scores in results/continuation-dclm/.

Unlike the unconditional board, these generators are not limited to OpenWebText models, and a score may be
reported without its folds (mean and std taken from the CHORD paper's table) as long as the file names its source.
When folds are given, mean and std are checked against them like on the unconditional board.
"""

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from discovery.models import Checkpoint, Paper, SiteGroup
from discovery.registry import _load_list
from leaderboard_site.common import ENCODER, HUMAN_ID, METRIC, BuildError, check_folds, checkpoint_link, paper_link, parse_params

SETTING = "continuation-dclm"
HUMAN_NAME = "Held-out human"


class ContinuationEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    group: SiteGroup
    params: str
    paper: Paper
    github: str
    checkpoint: Checkpoint
    added: date

    @model_validator(mode="after")
    def _paper_has_link_and_date(self):
        if not (self.paper.arxiv or self.paper.url):
            raise ValueError(f"{self.id}: paper needs arxiv or url")
        if not self.paper.published:
            raise ValueError(f"{self.id}: paper needs published (first arXiv version, or official release)")
        return self


class ReportedResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    setting: str
    encoder: str
    metric: str
    n_folds: int
    samples_per_fold: int
    folds: list[float] | None = None
    mean: float
    std: float
    scored_on: date
    source: str | None = None  # where mean and std come from when there are no folds


def load_entries(path: Path) -> list[ContinuationEntry]:
    return _load_list(path, ContinuationEntry)


def _check(name: str, r: ReportedResult) -> None:
    for field, expected in (("setting", SETTING), ("encoder", ENCODER), ("metric", METRIC)):
        if getattr(r, field) != expected:
            raise BuildError(f"{name}: {field} is {getattr(r, field)!r}; this board needs {expected!r}")
    if r.n_folds < 2:
        raise BuildError(f"{name}: n_folds is {r.n_folds}; a score needs at least two folds")
    if r.folds is None:
        if not r.source:
            raise BuildError(f"{name}: a score without folds must name its source")
        return
    if len(r.folds) != r.n_folds:
        raise BuildError(f"{name}: has {len(r.folds)} folds but n_folds is {r.n_folds}")
    check_folds(name, r.folds, r.mean, r.std)


def load_reported(directory: Path) -> dict[str, ReportedResult]:
    out = {}
    for path in sorted(directory.glob("*.json")):
        try:
            r = ReportedResult.model_validate_json(path.read_text())
        except ValidationError as exc:
            raise BuildError(f"{path.name}: invalid results file: {exc}") from exc
        if r.id != path.stem:
            raise BuildError(f"{path.name}: id is {r.id!r}; it must match the file name")
        _check(path.name, r)
        out[r.id] = r
    return out


def build_continuation(entries: list[ContinuationEntry], results: dict[str, ReportedResult]) -> dict:
    """Rows of the continuation board, sorted by mean; `last_modified` is a date for the caller to merge."""
    ids = [e.id for e in entries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise BuildError(f"duplicate ids in registry/continuation.yaml: {', '.join(duplicates)}")
    if HUMAN_ID not in results:
        raise BuildError(f"results/{SETTING}/{HUMAN_ID}.json (the held-out human row) is missing")
    orphans = sorted(set(results) - set(ids) - {HUMAN_ID})
    if orphans:
        raise BuildError(f"{SETTING} results without a registry/continuation.yaml entry: {', '.join(orphans)}")
    missing = sorted(set(ids) - set(results))
    if missing:
        raise BuildError(f"registry/continuation.yaml entries without {SETTING} results: {', '.join(missing)}")
    generators = sorted(({
        "id": e.id, "name": e.name, "group": e.group, "params": e.params, "params_value": parse_params(e.params),
        "released": e.paper.published.isoformat(), "paper": paper_link(e.paper), "code": e.github,
        "mean": round(results[e.id].mean, 4), "std": round(results[e.id].std, 4),
        "checkpoint": checkpoint_link(e.checkpoint),
    } for e in entries), key=lambda g: g["mean"])
    human = results[HUMAN_ID]
    last = max([r.scored_on for r in results.values()] + [e.added for e in entries])
    return {"last_modified": last,
            "human": {"name": HUMAN_NAME, "mean": round(human.mean, 4), "std": round(human.std, 4)},
            "generators": generators}
