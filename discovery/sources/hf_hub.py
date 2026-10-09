"""HF Hub: recently created models whose name or card mentions OpenWebText."""

from datetime import date

from discovery.keys import normalize_arxiv_id
from discovery.models import Candidate, Weight

LIST_LIMIT = 500


def _own_paper(tags: list[str]) -> str | None:
    """A card often tags the papers it builds on as well; the model's own paper is the newest id."""
    ids = [normalize_arxiv_id(t.split(":", 1)[1]) for t in tags if t.startswith("arxiv:")]
    return max(ids) if ids else None


def search(api, queries: list[str], since: date) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for q in queries:
        for info in api.list_models(search=q, sort="created_at", limit=LIST_LIMIT):  # newest first
            if info.created_at.date() < since:
                break
            cand = Candidate(arxiv_id=_own_paper(info.tags or []), sources=[f"hf-search:{q}"],
                             weights=[Weight(kind="hf", ref=info.id)])
            merged[cand.key] = merged[cand.key].merge(cand) if cand.key in merged else cand
    return list(merged.values())
