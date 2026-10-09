from datetime import date, datetime, timezone
from types import SimpleNamespace

from discovery.sources.hf_hub import search


def _m(mid, day, tags=()):
    return SimpleNamespace(id=mid, created_at=datetime(2026, 2, day, tzinfo=timezone.utc), tags=list(tags))


class FakeApi:
    def __init__(self, by_query):
        self.by_query, self.calls = by_query, []

    def list_models(self, **kw):
        self.calls.append(kw)
        return iter(self.by_query[kw["search"]])


def test_search_maps_newest_arxiv_tag_and_stops_at_since():
    api = FakeApi({
        "owt": [_m("kuleshov-group/proseco-owt", 20, ["arxiv:2406.07524", "arxiv:2602.11590"]),
                _m("someone/ermine-owt", 15),
                _m("old/model-owt", 1)],
        "openwebtext": [_m("kuleshov-group/proseco-owt", 20, ["arxiv:2602.11590"])],
    })
    got = search(api, ["owt", "openwebtext"], since=date(2026, 2, 10))
    by_key = {c.key: c for c in got}
    assert set(by_key) == {"arxiv:2602.11590", "hf:someone/ermine-owt"}
    assert by_key["arxiv:2602.11590"].sources == ["hf-search:owt", "hf-search:openwebtext"]
    assert [w.ref for w in by_key["arxiv:2602.11590"].weights] == ["kuleshov-group/proseco-owt"]
    assert api.calls[0]["sort"] == "created_at"


def test_list_models_kwargs_exist_in_real_api():
    import inspect

    from huggingface_hub import HfApi

    api = FakeApi({"owt": []})
    search(api, ["owt"], since=date(2026, 2, 10))
    assert set(api.calls[0]) <= set(inspect.signature(HfApi.list_models).parameters)
