import httpx
import pytest
import respx

from discovery.http import HttpError, make_client, request_with_retry

URL = "https://api.example.com/x"


@respx.mock
def test_retries_then_succeeds():
    route = respx.get(URL).mock(side_effect=[httpx.Response(429), httpx.Response(503), httpx.Response(200, json={"ok": 1})])
    delays = []
    resp = request_with_retry(make_client(), "GET", URL, sleep=delays.append)
    assert resp.json() == {"ok": 1} and route.call_count == 3
    assert delays == [2.0, 4.0]


@respx.mock
def test_gives_up_after_five_attempts():
    respx.get(URL).mock(return_value=httpx.Response(500))
    with pytest.raises(HttpError, match="500"):
        request_with_retry(make_client(), "GET", URL, sleep=lambda _: None)
    assert respx.calls.call_count == 5


@respx.mock
def test_404_is_returned_not_retried():
    respx.get(URL).mock(return_value=httpx.Response(404))
    assert request_with_retry(make_client(), "GET", URL, sleep=lambda _: None).status_code == 404
    assert respx.calls.call_count == 1


@respx.mock
def test_transport_errors_are_retried():
    respx.get(URL).mock(side_effect=[httpx.ConnectTimeout("t"), httpx.Response(200)])
    assert request_with_retry(make_client(), "GET", URL, sleep=lambda _: None).status_code == 200


def test_client_timeout_is_30s():
    assert make_client().timeout.read == 30
