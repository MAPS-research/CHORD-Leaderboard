"""Minimal GitHub REST client for candidate and failure issues."""

import time

from discovery.enrich import GITHUB_API
from discovery.http import HttpError, request_with_retry
from discovery.report import issue_keys

PER_PAGE = 100
FAILURE_TITLE = "[pipeline-failure] discovery source failing in consecutive runs"


class GitHubClient:
    def __init__(self, client, repo: str, sleep=time.sleep):
        self.client, self.repo, self.sleep = client, repo, sleep

    def _url(self, path: str) -> str:
        return f"{GITHUB_API}/repos/{self.repo}/{path}"

    def _list_issues(self, **params) -> list[dict]:
        out, page = [], 1
        while True:
            resp = request_with_retry(self.client, "GET", self._url("issues"),
                                      params={**params, "per_page": PER_PAGE, "page": page}, sleep=self.sleep)
            if resp.status_code != 200:
                raise HttpError(f"list issues: HTTP {resp.status_code}")
            batch = resp.json()
            out += [i for i in batch if "pull_request" not in i]
            if len(batch) < PER_PAGE:
                return out
            page += 1

    def candidate_issue_keys(self) -> set[str]:
        keys: set[str] = set()
        for issue in self._list_issues(state="all", labels="candidate"):
            keys |= issue_keys(issue.get("body") or "")
        return keys

    def create_issue(self, title: str, body: str, labels: list[str]) -> int:
        resp = request_with_retry(self.client, "POST", self._url("issues"),
                                  json={"title": title, "body": body, "labels": labels}, sleep=self.sleep)
        if resp.status_code != 201:
            raise HttpError(f"create issue: HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()["number"]

    def upsert_failure_issue(self, body: str) -> int:
        open_issues = self._list_issues(state="open", labels="pipeline-failure")
        if not open_issues:
            return self.create_issue(FAILURE_TITLE, body, ["pipeline-failure"])
        number = open_issues[0]["number"]
        resp = request_with_retry(self.client, "POST", self._url(f"issues/{number}/comments"), json={"body": body}, sleep=self.sleep)
        if resp.status_code != 201:
            raise HttpError(f"comment on #{number}: HTTP {resp.status_code}")
        return number
