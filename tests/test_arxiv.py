from datetime import date
from pathlib import Path

import httpx
import respx

from discovery.http import make_client
from discovery.sources.arxiv import ARXIV_API, build_query, fetch_by_ids, parse_feed, search

FEED = (Path(__file__).parent / "fixtures" / "arxiv_feed.xml").read_text()
EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
NOSLEEP = dict(sleep=lambda _: None)


def test_build_query():
    q = build_query(["cs.CL", "cs.LG"], ['abs:"OpenWebText"', 'abs:"gen-PPL"'])
    assert q == '(cat:cs.CL OR cat:cs.LG) AND (abs:"OpenWebText" OR abs:"gen-PPL")'


def test_parse_feed_normalises_whitespace_and_ids():
    first, second = parse_feed(FEED, source="arxiv-kw")
    assert first.arxiv_id == "2602.11590" and second.arxiv_id == "2601.00002"
    assert first.title == "Self-Correcting Masked Diffusion"
    assert first.abstract == "We train on OpenWebText and release checkpoints."
    assert first.comments == "Code: https://github.com/kuleshov-group/proseco"
    assert first.published == date(2026, 2, 12) and first.sources == ["arxiv-kw"]
    assert second.comments == ""


def test_parse_feed_skips_error_entries():
    xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#bad_id</id>'
           '<title>Error</title><summary>bad id</summary></entry></feed>')
    assert parse_feed(xml, source=None) == []


@respx.mock
def test_search_stops_at_since():
    route = respx.get(url__startswith=ARXIV_API).mock(return_value=httpx.Response(200, text=FEED))
    got = search(make_client(), ["cs.CL"], ['abs:"OpenWebText"'], since=date(2026, 1, 20), **NOSLEEP)
    assert [c.arxiv_id for c in got] == ["2602.11590"]
    assert route.call_count == 1


@respx.mock
def test_fetch_by_ids_batches_and_keys_by_id():
    respx.get(url__startswith=ARXIV_API).mock(return_value=httpx.Response(200, text=FEED))
    got = fetch_by_ids(make_client(), ["2602.11590v2", "2601.00002"], **NOSLEEP)
    assert set(got) == {"2602.11590", "2601.00002"} and got["2602.11590"].sources == []


@respx.mock
def test_fetch_by_ids_empty_input_makes_no_request():
    assert fetch_by_ids(make_client(), [], **NOSLEEP) == {}
    assert respx.calls.call_count == 0
