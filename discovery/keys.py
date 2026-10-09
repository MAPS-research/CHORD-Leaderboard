"""Normalise arXiv ids, GitHub URLs and HF repo ids into dedupe keys."""

import re

_ARXIV_PREFIX = re.compile(r"^(arxiv:|https?://(www\.)?arxiv\.org/(abs|pdf)/)", re.IGNORECASE)
_VERSION = re.compile(r"v\d+$")
_GITHUB = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.IGNORECASE)
GITHUB_RESERVED = frozenset(
    {"features", "sponsors", "orgs", "topics", "about", "marketplace", "settings", "login", "apps", "collections"}
)


def normalize_arxiv_id(raw: str) -> str:
    s = _ARXIV_PREFIX.sub("", raw.strip())
    s = s.removesuffix(".pdf")
    return _VERSION.sub("", s)


def arxiv_key(raw: str) -> str:
    return f"arxiv:{normalize_arxiv_id(raw)}"


def normalize_github(url: str) -> str | None:
    m = _GITHUB.search(url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).removesuffix(".git").rstrip(".")
    if owner.lower() in GITHUB_RESERVED or not repo:
        return None
    return f"{owner}/{repo}".lower()


def github_key(url: str) -> str | None:
    norm = normalize_github(url)
    return f"github:{norm}" if norm else None


def hf_key(ref: str) -> str:
    return f"hf:{ref.strip().lower()}"
