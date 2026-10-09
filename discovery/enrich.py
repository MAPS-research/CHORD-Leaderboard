"""Find a candidate's GitHub repos, README and weight links."""

import re
import time

from discovery.http import github_headers, raise_if_rate_limited, request_with_retry
from discovery.keys import github_key, normalize_github
from discovery.models import Candidate, Weight

HF_API = "https://huggingface.co/api"
GITHUB_API = "https://api.github.com"

_TAIL = r"[^\s)\"'<>\]]+"
_GITHUB_ANY = re.compile(r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.IGNORECASE)
_HF_URL = re.compile(
    r"huggingface\.co/(?!datasets/|spaces/|papers/|docs/|blog/|collections/|api/)([A-Za-z0-9][\w.-]*/[\w.-]+)")
_HF_COLLECTION = re.compile(r"huggingface\.co/collections/([A-Za-z0-9][\w.-]*/[\w.-]+)")
_FROM_PRETRAINED = re.compile(r"""from_pretrained\(\s*["']([A-Za-z0-9][\w.-]*/[\w.-]+)["']""")
_EXTERNAL = [
    ("gdrive", re.compile(rf"https?://drive\.google\.com/{_TAIL}")),
    ("dropbox", re.compile(rf"https?://(?:www\.)?dropbox\.com/{_TAIL}")),
    ("zenodo", re.compile(rf"https?://(?:www\.)?zenodo\.org/records?/\d+(?:{_TAIL})?")),
    ("box", re.compile(rf"https?://[\w-]+\.box\.com/{_TAIL}")),
    ("github-release", re.compile(rf"https?://github\.com/[\w.-]+/[\w.-]+/releases(?:{_TAIL})?")),
]


def extract_github_repos(text: str) -> list[str]:
    out: dict[str, str] = {}
    for m in _GITHUB_ANY.finditer(text):
        norm = normalize_github(m.group(0))
        if norm and norm not in out:
            out[norm] = f"https://github.com/{norm}"
    return list(out.values())


def extract_weight_links(text: str) -> list[Weight]:
    found: dict[tuple[str, str], Weight] = {}

    def add(kind: str, ref: str) -> None:
        ref = ref.rstrip(".,;:")
        found.setdefault((kind, ref.lower()), Weight(kind=kind, ref=ref))

    for pattern in (_HF_URL, _FROM_PRETRAINED):
        for m in pattern.finditer(text):
            add("hf", m.group(1))
    for kind, pattern in _EXTERNAL:
        for m in pattern.finditer(text):
            add(kind, m.group(0))
    return list(found.values())


def _hf_paper_links(client, arxiv_id: str, sleep) -> tuple[list[str], list[Weight]]:
    repos: list[str] = []
    resp = request_with_retry(client, "GET", f"{HF_API}/papers/{arxiv_id}", sleep=sleep)
    if resp.status_code == 200 and (repo := resp.json().get("githubRepo")):
        repos.append(repo)
    resp = request_with_retry(client, "GET", f"{HF_API}/models", params={"filter": f"arxiv:{arxiv_id}", "limit": 100}, sleep=sleep)
    models = [Weight(kind="hf", ref=m["id"]) for m in resp.json()] if resp.status_code == 200 else []
    return repos, models


def _collection_models(client, text: str, sleep) -> list[Weight]:
    """Models in the HF collections a README links; authors often publish checkpoints only that way."""
    out = []
    for slug in dict.fromkeys(m.group(1).rstrip(".,;:") for m in _HF_COLLECTION.finditer(text)):
        resp = request_with_retry(client, "GET", f"{HF_API}/collections/{slug}", sleep=sleep)
        if resp.status_code == 200:
            out += [Weight(kind="hf", ref=i["id"]) for i in resp.json().get("items", []) if i.get("type") == "model"]
    return out


def _fetch_readme(client, repo_url: str, sleep) -> str | None:
    norm = normalize_github(repo_url)
    if not norm:
        return None
    headers = {**github_headers(), "Accept": "application/vnd.github.raw"}
    resp = request_with_retry(client, "GET", f"{GITHUB_API}/repos/{norm}/readme", headers=headers, sleep=sleep)
    raise_if_rate_limited(resp)
    return resp.text if resp.status_code == 200 else None


def _unique_repos(repos: list[str]) -> list[str]:
    out: dict[str, str] = {}
    for r in repos:
        if (k := github_key(r)) and k not in out:
            out[k] = f"https://github.com/{k.removeprefix('github:')}"
    return list(out.values())


def enrich(client, cand: Candidate, max_readme_chars: int, sleep=time.sleep) -> Candidate:
    repos, weights = list(cand.repos), list(cand.weights)
    if cand.arxiv_id:
        paper_repos, paper_models = _hf_paper_links(client, cand.arxiv_id, sleep)
        repos += paper_repos
        weights += paper_models
    repos = _unique_repos(repos + extract_github_repos(f"{cand.comments}\n{cand.abstract}"))
    readme, readme_repo = "", None
    for repo in repos:
        if (text := _fetch_readme(client, repo, sleep)) is not None:
            readme, readme_repo = text[:max_readme_chars], repo
            break
    if readme:
        weights += extract_weight_links(readme) + _collection_models(client, readme, sleep)
    merged = cand.model_copy(update={"repos": repos, "readme": readme, "readme_repo": readme_repo, "weights": []})
    return merged.merge(cand.model_copy(update={"weights": weights}))
