"""Drop candidates that are already listed, rejected or reported."""

from discovery.models import Candidate


def filter_new(cands: list[Candidate], known: frozenset[str], *, judged: bool,
               paper_exempt: frozenset[str] = frozenset()) -> list[Candidate]:
    """`paper_exempt`: keys (rejected HF-only models) that must not block a candidate that has a paper."""
    if not judged:
        return [c for c in cands if c.key not in known]
    return [c for c in cands if not (c.keys() & (known - paper_exempt if c.arxiv_id else known))]


def dedupe_within_run(cands: list[Candidate]) -> list[Candidate]:
    """Drop candidates whose keys overlap one already kept; paper candidates win over HF-only ones."""
    kept, seen = [], set()
    for c in sorted(cands, key=lambda c: c.arxiv_id is None):
        if c.keys() & seen:
            continue
        kept.append(c)
        seen |= c.keys()
    return kept
