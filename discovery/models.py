"""Data models shared by every pipeline stage. All models are immutable."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from discovery.keys import arxiv_key, github_key, hf_key

Tri = Literal["yes", "no", "unclear"]
WeightKind = Literal["hf", "gdrive", "dropbox", "zenodo", "box", "github-release"]
CheckpointKind = Literal["hf", "gdrive", "dropbox", "zenodo", "box", "github-release", "sampler"]
CheckResult = Literal["ok", "missing", "gated", "unverifiable", "unchecked"]
SiteGroup = Literal["ar", "discrete", "continuous"]  # leaderboard type tag: AR, discrete or continuous diffusion/flow
Family = Literal["ar", "masked-dlm", "uniform-dlm", "hybrid-dlm", "block-hybrid", "continuous", "flow", "distilled"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Evidence(_Frozen):
    quote: str
    url: str


class Judgement(_Frozen):
    value: Tri = "unclear"
    evidence: list[Evidence] = []


class TypeJudgement(_Frozen):
    """Generation type for the site's three tags, judged like the other criteria (quotes required)."""

    value: Literal["ar", "discrete", "continuous", "unclear"] = "unclear"
    evidence: list[Evidence] = []


class CheckpointRef(_Frozen):
    kind: WeightKind
    ref: str
    evidence: list[Evidence] = []


class Verdict(_Frozen):
    owt_trained: Judgement = Judgement()
    unconditional: Judgement = Judgement()
    official_repo: str | None = None
    official_checkpoints: list[CheckpointRef] = []
    family_guess: Family | None = None
    generation_type: TypeJudgement = TypeJudgement()
    notes: str = ""


class Weight(_Frozen):
    kind: WeightKind
    ref: str
    check: CheckResult = "unchecked"


def _union(a: list, b: list, ident) -> list:
    seen, out = set(), []
    for item in [*a, *b]:
        k = ident(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


class Candidate(_Frozen):
    arxiv_id: str | None = None
    title: str = ""
    abstract: str = ""
    comments: str = ""
    published: date | None = None
    sources: list[str]
    repos: list[str] = []
    readme: str = ""
    readme_repo: str | None = None
    weights: list[Weight] = []
    verdict: Verdict | None = None
    judge_error: str | None = None
    repo_stars: dict[str, int] = {}  # github key -> stargazers, for live repos
    paper_upvotes: int | None = None  # HF Papers upvotes

    @model_validator(mode="after")
    def _needs_identity(self):
        if not self.arxiv_id and not any(w.kind == "hf" for w in self.weights):
            raise ValueError("candidate needs an arxiv_id or an HF weight")
        return self

    @property
    def key(self) -> str:
        if self.arxiv_id:
            return arxiv_key(self.arxiv_id)
        return hf_key(next(w.ref for w in self.weights if w.kind == "hf"))

    def keys(self) -> set[str]:
        """Primary key plus the *official* repo/checkpoints from the verdict.

        Repos and weights merely mentioned in the abstract or README are excluded on purpose:
        new papers often cite the MDLM repo or checkpoint they start from.
        """
        out = {self.key}
        if self.verdict is None:
            return out
        if self.verdict.official_repo and (gk := github_key(self.verdict.official_repo)):
            out.add(gk)
        out |= {hf_key(c.ref) for c in self.verdict.official_checkpoints if c.kind == "hf"}
        return out

    def merge(self, other: "Candidate") -> "Candidate":
        return self.model_copy(
            update={
                "title": self.title or other.title,
                "abstract": self.abstract or other.abstract,
                "comments": self.comments or other.comments,
                "published": self.published or other.published,
                "arxiv_id": self.arxiv_id or other.arxiv_id,
                "sources": _union(self.sources, other.sources, lambda s: s),
                "repos": _union(self.repos, other.repos, lambda r: github_key(r) or r),
                "weights": _union(self.weights, other.weights, lambda w: (w.kind, w.ref.lower())),
                "repo_stars": {**other.repo_stars, **self.repo_stars},
                "paper_upvotes": self.paper_upvotes if self.paper_upvotes is not None else other.paper_upvotes,
            }
        )


class Paper(_Frozen):
    arxiv: str | None = None
    title: str | None = None
    venue: str | None = None
    published: date | None = None  # first arXiv version (or official release when there is no arXiv paper)
    url: str | None = None  # paper link when there is no arXiv id


class Checkpoint(_Frozen):
    kind: CheckpointKind
    ref: str
    path: str | None = None  # file or folder inside ref when the repo holds several checkpoints


class RegistryEntry(_Frozen):
    id: str
    name: str
    paper: Paper | None = None
    github: str
    checkpoint: Checkpoint
    family: Family
    group: SiteGroup | None = None
    params: str | None = None
    train_data: Literal["owt", "owt2", "webtext"]
    tokenizer: str | None = None
    in_paper: bool = False
    status: Literal["queued", "scored", "excluded"]
    sampler: dict | None = None
    added: date
    source: str

    @model_validator(mode="after")
    def _scored_entries_have_site_fields(self):
        if self.status != "scored":
            return self
        missing = [f for f, ok in (("group", self.group), ("params", self.params),
                                   ("paper.published", self.paper and self.paper.published),
                                   ("paper.arxiv or paper.url", self.paper and (self.paper.arxiv or self.paper.url))) if not ok]
        if missing:
            raise ValueError(f"scored entry {self.id} is missing {', '.join(missing)}")
        return self

    def keys(self) -> set[str]:
        out = set()
        if self.paper and self.paper.arxiv:
            out.add(arxiv_key(self.paper.arxiv))
        if gk := github_key(self.github):
            out.add(gk)
        if self.checkpoint.kind == "hf":
            out.add(hf_key(self.checkpoint.ref))
        return out


class RejectedEntry(_Frozen):
    key: str
    name: str
    reason: str
    decided: date
    source: str
