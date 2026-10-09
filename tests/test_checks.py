from types import SimpleNamespace

import httpx
import respx
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from discovery.checks import check_hf, repo_exists, run_checks
from discovery.enrich import GITHUB_API
from discovery.http import make_client
from discovery.models import Candidate, Weight

NOSLEEP = dict(sleep=lambda _: None)


def _err(cls, status):
    return cls(f"HTTP {status}", response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co")))


def _info(files, private=False, gated=False):
    return SimpleNamespace(private=private, gated=gated, siblings=[SimpleNamespace(rfilename=f) for f in files])


class FakeApi:
    def __init__(self, table):
        self.table = table

    def model_info(self, ref):
        result = self.table[ref]
        if isinstance(result, Exception):
            raise result
        return result


API = FakeApi({
    "ok/model": _info(["config.json", "model.safetensors"]),
    "ok/ckpt": _info(["README.md", "checkpoints/last.ckpt"]),
    "no/weights": _info(["README.md", "config.json"]),
    "gated/flag": _info(["model.safetensors"], gated="manual"),
    "gated/error": _err(GatedRepoError, 403),
    "missing/model": _err(RepositoryNotFoundError, 404),
    "flaky/model": _err(HfHubHTTPError, 500),
})


def test_check_hf_branches():
    assert check_hf(API, "ok/model") == "ok"
    assert check_hf(API, "ok/ckpt") == "ok"
    assert check_hf(API, "no/weights") == "missing"
    assert check_hf(API, "gated/flag") == "gated"
    assert check_hf(API, "gated/error") == "gated"
    assert check_hf(API, "missing/model") == "missing"
    assert check_hf(API, "flaky/model") == "unverifiable"


@respx.mock
def test_repo_exists():
    respx.get(f"{GITHUB_API}/repos/a/live").mock(return_value=httpx.Response(200, json={"archived": False}))
    respx.get(f"{GITHUB_API}/repos/a/archived").mock(return_value=httpx.Response(200, json={"archived": True}))
    respx.get(f"{GITHUB_API}/repos/a/gone").mock(return_value=httpx.Response(404))
    client = make_client()
    assert repo_exists(client, "https://github.com/a/live", **NOSLEEP)
    assert not repo_exists(client, "https://github.com/a/archived", **NOSLEEP)
    assert not repo_exists(client, "https://github.com/a/gone", **NOSLEEP)


@respx.mock
def test_run_checks_marks_weights_and_filters_repos():
    respx.get(f"{GITHUB_API}/repos/a/live").mock(return_value=httpx.Response(200, json={"archived": False}))
    respx.get(f"{GITHUB_API}/repos/a/gone").mock(return_value=httpx.Response(404))
    cand = Candidate(arxiv_id="2602.11590", sources=["x"], repos=["https://github.com/a/live", "https://github.com/a/gone"],
                     weights=[Weight(kind="hf", ref="ok/model"), Weight(kind="gdrive", ref="https://drive.google.com/x")])
    out = run_checks(API, make_client(), cand, **NOSLEEP)
    assert [(w.ref, w.check) for w in out.weights] == [("ok/model", "ok"), ("https://drive.google.com/x", "unverifiable")]
    assert out.repos == ["https://github.com/a/live"]


@respx.mock
def test_repo_exists_raises_on_rate_limit():
    import pytest

    from discovery.http import HttpError

    respx.get(f"{GITHUB_API}/repos/a/limited").mock(return_value=httpx.Response(429, headers={"retry-after": "60"}))
    respx.get(f"{GITHUB_API}/repos/a/secondary").mock(return_value=httpx.Response(403, headers={"retry-after": "60"}))
    with pytest.raises(HttpError):
        repo_exists(make_client(), "https://github.com/a/limited", **NOSLEEP)
    with pytest.raises(HttpError, match="rate limit"):
        repo_exists(make_client(), "https://github.com/a/secondary", **NOSLEEP)
