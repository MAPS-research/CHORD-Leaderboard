import json

import httpx
import pytest
import respx

from discovery.enrich import GITHUB_API
from discovery.github import GitHubClient
from discovery.http import HttpError, make_client

REPO = "MAPS-research/CHORD-Leaderboard"
ISSUES = f"{GITHUB_API}/repos/{REPO}/issues"


def _gh():
    return GitHubClient(make_client(), REPO, sleep=lambda _: None)


@respx.mock
def test_candidate_issue_keys_paginates_and_skips_prs():
    page1 = [{"number": i, "body": f"<!-- cand: arxiv:2601.{i:05d} -->"} for i in range(100)]
    page2 = [{"number": 200, "body": None}, {"number": 201, "body": "<!-- cand: hf:x/y -->", "pull_request": {}}]
    route = respx.get(url__startswith=ISSUES).mock(side_effect=[httpx.Response(200, json=page1), httpx.Response(200, json=page2)])
    keys = _gh().candidate_issue_keys()
    assert len(keys) == 100 and "hf:x/y" not in keys
    params = route.calls[0].request.url.params
    assert params["state"] == "all" and params["labels"] == "candidate" and params["per_page"] == "100"


@respx.mock
def test_create_issue_returns_number_and_raises_on_error():
    route = respx.post(ISSUES).mock(side_effect=[httpx.Response(201, json={"number": 7}), httpx.Response(422, json={})])
    assert _gh().create_issue("t", "b", ["candidate"]) == 7
    assert json.loads(route.calls[0].request.content) == {"title": "t", "body": "b", "labels": ["candidate"]}
    with pytest.raises(HttpError, match="422"):
        _gh().create_issue("t", "b", [])


@respx.mock
def test_upsert_failure_issue_comments_when_open():
    respx.get(url__startswith=ISSUES).mock(return_value=httpx.Response(200, json=[{"number": 3, "body": ""}]))
    comment = respx.post(f"{ISSUES}/3/comments").mock(return_value=httpx.Response(201, json={}))
    assert _gh().upsert_failure_issue("still failing") == 3 and comment.called


@respx.mock
def test_upsert_failure_issue_opens_when_none():
    respx.get(url__startswith=ISSUES).mock(return_value=httpx.Response(200, json=[]))
    create = respx.post(ISSUES).mock(return_value=httpx.Response(201, json={"number": 9}))
    assert _gh().upsert_failure_issue("x") == 9
    assert json.loads(create.calls[0].request.content)["labels"] == ["pipeline-failure"]
