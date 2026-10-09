"""arXiv export API: keyword search over recent submissions and metadata lookup by id."""

import time
import xml.etree.ElementTree as ET
from datetime import date

from discovery.http import HttpError, request_with_retry
from discovery.keys import normalize_arxiv_id
from discovery.models import Candidate

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
PAGE_SIZE = 100
MAX_RESULTS = 2000
POLITE_DELAY_S = 3.0  # arXiv API terms of use: one request every 3 seconds


def build_query(categories: list[str], terms: list[str]) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    return f"({cats}) AND ({' OR '.join(terms)})"


def _text(entry, path: str) -> str:
    return " ".join((entry.findtext(path, default="", namespaces=NS) or "").split())


def parse_feed(xml_text: str, source: str | None) -> list[Candidate]:
    out = []
    for entry in ET.fromstring(xml_text).findall("a:entry", NS):
        raw_id = _text(entry, "a:id")
        if "arxiv.org/abs/" not in raw_id:
            continue
        out.append(Candidate(
            arxiv_id=normalize_arxiv_id(raw_id.split("arxiv.org/abs/")[1]),
            title=_text(entry, "a:title"),
            abstract=_text(entry, "a:summary"),
            comments=_text(entry, "arxiv:comment"),
            published=date.fromisoformat(_text(entry, "a:published")[:10]),
            sources=[source] if source else [],
        ))
    return out


def _get(client, params: dict, sleep) -> str:
    resp = request_with_retry(client, "GET", ARXIV_API, params=params, sleep=sleep)
    if resp.status_code != 200:
        raise HttpError(f"arXiv API: HTTP {resp.status_code}")
    return resp.text


def search(client, categories: list[str], terms: list[str], since: date, sleep=time.sleep) -> list[Candidate]:
    query, out, start = build_query(categories, terms), [], 0
    while start < MAX_RESULTS:
        page = parse_feed(_get(client, {"search_query": query, "sortBy": "submittedDate", "sortOrder": "descending",
                                        "start": start, "max_results": PAGE_SIZE}, sleep), source="arxiv-kw")
        out.extend(c for c in page if c.published and c.published >= since)
        if len(page) < PAGE_SIZE or (page[-1].published and page[-1].published < since):
            break
        start += PAGE_SIZE
        sleep(POLITE_DELAY_S)
    return out


def fetch_by_ids(client, ids: list[str], sleep=time.sleep) -> dict[str, Candidate]:
    norm = list(dict.fromkeys(normalize_arxiv_id(i) for i in ids))
    out: dict[str, Candidate] = {}
    for i in range(0, len(norm), PAGE_SIZE):
        if i:
            sleep(POLITE_DELAY_S)
        batch = norm[i:i + PAGE_SIZE]
        for cand in parse_feed(_get(client, {"id_list": ",".join(batch), "max_results": len(batch)}, sleep), source=None):
            out[cand.arxiv_id] = cand
    return out
