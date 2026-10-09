"""Shared httpx client with retry on rate limits, server errors and transport failures."""

import os
import time

import httpx

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
TIMEOUT_S = 30.0
USER_AGENT = "chord-leaderboard-discovery (+https://github.com/MAPS-research/CHORD-Leaderboard)"


class HttpError(RuntimeError):
    pass


def raise_if_rate_limited(resp: httpx.Response) -> None:
    """GitHub signals primary and secondary rate limits with 403/429; never read those as "absent"."""
    limited = resp.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in resp.headers
    if resp.status_code in (403, 429) and limited:
        raise HttpError(f"rate limit: {resp.request.method} {resp.request.url} -> {resp.status_code}")


def make_client(headers: dict[str, str] | None = None) -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT_S, follow_redirects=True, headers={"User-Agent": USER_AGENT, **(headers or {})})


def request_with_retry(client, method, url, *, params=None, headers=None, json=None, timeout=None,
                       attempts: int = 5, base_delay: float = 2.0, sleep=time.sleep) -> httpx.Response:
    last: Exception | None = None
    for i in range(attempts):
        try:
            kwargs = {"params": params, "headers": headers, "json": json}
            if timeout is not None:
                kwargs["timeout"] = timeout
            resp = client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            last = exc
        else:
            if resp.status_code not in RETRY_STATUS:
                return resp
            last = HttpError(f"{method} {url} -> {resp.status_code}")
        if i < attempts - 1:
            sleep(base_delay * 2**i)
    raise HttpError(f"{method} {url} failed after {attempts} attempts: {last}") from last


def github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return headers
