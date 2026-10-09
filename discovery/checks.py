"""Deterministic checks: HF weights are public with weight files; GitHub repos exist and are live."""

import time

from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from discovery.enrich import GITHUB_API
from discovery.http import github_headers, raise_if_rate_limited, request_with_retry
from discovery.keys import github_key, normalize_github
from discovery.models import Candidate, CheckResult

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".pth")


def check_hf(api, ref: str) -> CheckResult:
    try:
        info = api.model_info(ref)
    except GatedRepoError:  # subclass of RepositoryNotFoundError, so catch it first
        return "gated"
    except RepositoryNotFoundError:
        return "missing"
    except HfHubHTTPError:
        return "unverifiable"
    if getattr(info, "private", False):
        return "missing"
    if getattr(info, "gated", False):
        return "gated"
    files = [s.rfilename for s in (getattr(info, "siblings", None) or [])]
    return "ok" if any(f.endswith(WEIGHT_SUFFIXES) for f in files) else "missing"


def _repo_info(client, repo_url: str, sleep) -> dict | None:
    """The repo's API record when it exists and is not archived, else None."""
    norm = normalize_github(repo_url)
    if not norm:
        return None
    resp = request_with_retry(client, "GET", f"{GITHUB_API}/repos/{norm}", headers=github_headers(), sleep=sleep)
    raise_if_rate_limited(resp)
    if resp.status_code != 200 or resp.json().get("archived", False):
        return None
    return resp.json()


def repo_exists(client, repo_url: str, sleep=time.sleep) -> bool:
    return _repo_info(client, repo_url, sleep) is not None


def run_checks(api, client, cand: Candidate, sleep=time.sleep) -> Candidate:
    weights = [w.model_copy(update={"check": check_hf(api, w.ref) if w.kind == "hf" else "unverifiable"})
               for w in cand.weights]
    repos, stars = [], {}
    for r in cand.repos:
        if (info := _repo_info(client, r, sleep)) is not None:
            repos.append(r)
            stars[github_key(r)] = int(info.get("stargazers_count") or 0)
    return cand.model_copy(update={"weights": weights, "repos": repos, "repo_stars": stars})
