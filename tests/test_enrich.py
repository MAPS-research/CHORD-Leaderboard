import httpx
import respx

from discovery.enrich import GITHUB_API, HF_API, enrich, extract_github_repos, extract_weight_links
from discovery.http import make_client
from discovery.models import Candidate

README = """
# ProSeCo
Weights: [here](https://huggingface.co/kuleshov-group/proseco-owt).
Also `AutoModel.from_pretrained("kuleshov-group/proseco-owt-large")`.
Data: https://huggingface.co/datasets/Skylion007/openwebtext and https://huggingface.co/spaces/x/y
Backup: https://drive.google.com/drive/folders/abc123?usp=sharing).
Box: https://ibm.box.com/v/soft-masked-dlm-checkpoints
Zenodo: https://zenodo.org/records/15124163
Release: https://github.com/igul222/plaid/releases/tag/v1.0.0
Based on https://github.com/kuleshov-group/mdlm.
"""
NOSLEEP = dict(sleep=lambda _: None)


def test_extract_github_repos_canonical_unique():
    text = "see https://GitHub.com/Org/Repo.git and github.com/org/repo/tree/main, also (https://github.com/a/b)."
    assert extract_github_repos(text) == ["https://github.com/org/repo", "https://github.com/a/b"]


def test_extract_weight_links_kinds_and_exclusions():
    got = {(w.kind, w.ref) for w in extract_weight_links(README)}
    assert ("hf", "kuleshov-group/proseco-owt") in got
    assert ("hf", "kuleshov-group/proseco-owt-large") in got
    assert ("gdrive", "https://drive.google.com/drive/folders/abc123?usp=sharing") in got
    assert ("box", "https://ibm.box.com/v/soft-masked-dlm-checkpoints") in got
    assert ("zenodo", "https://zenodo.org/records/15124163") in got
    assert ("github-release", "https://github.com/igul222/plaid/releases/tag/v1.0.0") in got
    assert not any("datasets" in ref or "spaces" in ref or ref.startswith("Skylion") for _, ref in got)


@respx.mock
def test_enrich_uses_hf_papers_comments_and_readme():
    respx.get(f"{HF_API}/papers/2602.11590").mock(
        return_value=httpx.Response(200, json={"githubRepo": "https://github.com/kuleshov-group/proseco"}))
    respx.get(url__startswith=f"{HF_API}/models").mock(return_value=httpx.Response(200, json=[{"id": "kuleshov-group/proseco-owt"}]))
    respx.get(f"{GITHUB_API}/repos/kuleshov-group/proseco/readme").mock(return_value=httpx.Response(200, text=README))
    cand = Candidate(arxiv_id="2602.11590", sources=["hf-search:owt"], comments="Code at github.com/kuleshov-group/proseco.")
    out = enrich(make_client(), cand, max_readme_chars=50, **NOSLEEP)
    assert out.repos == ["https://github.com/kuleshov-group/proseco"]
    assert out.readme == README[:50] and out.readme_repo == "https://github.com/kuleshov-group/proseco"
    assert ("hf", "kuleshov-group/proseco-owt") in {(w.kind, w.ref) for w in out.weights}
    assert cand.repos == []  # input untouched


@respx.mock
def test_enrich_survives_missing_paper_page_and_readme():
    respx.get(f"{HF_API}/papers/2601.00002").mock(return_value=httpx.Response(404))
    respx.get(url__startswith=f"{HF_API}/models").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{GITHUB_API}/repos/a/b/readme").mock(return_value=httpx.Response(404))
    cand = Candidate(arxiv_id="2601.00002", sources=["arxiv-kw"], abstract="Code: https://github.com/a/b")
    out = enrich(make_client(), cand, max_readme_chars=30000, **NOSLEEP)
    assert out.repos == ["https://github.com/a/b"] and out.readme == "" and out.readme_repo is None


@respx.mock
def test_github_rate_limit_is_an_error_not_a_missing_readme():
    import pytest

    from discovery.http import HttpError

    respx.get(f"{HF_API}/papers/2601.00002").mock(return_value=httpx.Response(404))
    respx.get(url__startswith=f"{HF_API}/models").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{GITHUB_API}/repos/a/b/readme").mock(
        return_value=httpx.Response(403, headers={"x-ratelimit-remaining": "0"}, json={"message": "API rate limit exceeded"}))
    cand = Candidate(arxiv_id="2601.00002", sources=["arxiv-kw"], abstract="Code: https://github.com/a/b")
    with pytest.raises(HttpError, match="rate limit"):
        enrich(make_client(), cand, max_readme_chars=30000, **NOSLEEP)


@respx.mock
def test_enrich_expands_linked_hf_collections_into_model_weights():
    readme = ("Checkpoints: [HF collection](https://huggingface.co/collections/sahoo-diffusion/eso-lms-6838e86cb2c49f45302f0092) "
              "and https://huggingface.co/collections/gone/old-123.")
    respx.get(f"{HF_API}/papers/2506.01928").mock(return_value=httpx.Response(404))
    respx.get(url__startswith=f"{HF_API}/models").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{GITHUB_API}/repos/s-sahoo/eso-lms/readme").mock(return_value=httpx.Response(200, text=readme))
    respx.get(f"{HF_API}/collections/sahoo-diffusion/eso-lms-6838e86cb2c49f45302f0092").mock(return_value=httpx.Response(200, json={
        "items": [{"type": "model", "id": "sahoo-diffusion/Eso-LM-B-alpha-1"}, {"type": "paper", "id": "2506.01928"},
                  {"type": "model", "id": "sahoo-diffusion/Eso-LM-B-alpha-0_25"}]}))
    respx.get(f"{HF_API}/collections/gone/old-123").mock(return_value=httpx.Response(404))
    cand = Candidate(arxiv_id="2506.01928", sources=["arxiv-kw"], comments="https://github.com/s-sahoo/Eso-LMs")
    out = enrich(make_client(), cand, max_readme_chars=30000, **NOSLEEP)
    refs = [w.ref for w in out.weights if w.kind == "hf"]
    assert refs == ["sahoo-diffusion/Eso-LM-B-alpha-1", "sahoo-diffusion/Eso-LM-B-alpha-0_25"]
