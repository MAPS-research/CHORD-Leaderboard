from datetime import date

import pytest
from pydantic import ValidationError

from discovery.models import Candidate, Checkpoint, CheckpointRef, Paper, RegistryEntry, Verdict, Weight


def _cand(**kw):
    base = dict(arxiv_id="2602.11590", title="ProSeCo", sources=["hf-search:owt"])
    base.update(kw)
    return Candidate(**base)


def test_candidate_key_prefers_arxiv():
    c = _cand(weights=[Weight(kind="hf", ref="kuleshov-group/proseco-owt")])
    assert c.key == "arxiv:2602.11590"


def test_hf_only_candidate_key():
    c = Candidate(title="", sources=["hf-search:owt"], weights=[Weight(kind="hf", ref="Org/Model-OWT")])
    assert c.key == "hf:org/model-owt"


def test_candidate_without_arxiv_or_hf_weight_is_invalid():
    with pytest.raises(ValidationError):
        Candidate(title="x", sources=["arxiv-kw:x"])


def test_candidate_keys_use_only_official_repo_and_checkpoints():
    c = _cand(repos=["https://github.com/kuleshov-group/proseco", "https://github.com/kuleshov-group/mdlm"],
              weights=[Weight(kind="hf", ref="kuleshov-group/proseco-owt"), Weight(kind="hf", ref="kuleshov-group/mdlm-owt")])
    assert c.keys() == {"arxiv:2602.11590"}
    judged = c.model_copy(update={"verdict": Verdict(
        official_repo="https://github.com/kuleshov-group/proseco",
        official_checkpoints=[CheckpointRef(kind="hf", ref="kuleshov-group/proseco-owt")])})
    assert judged.keys() == {"arxiv:2602.11590", "github:kuleshov-group/proseco", "hf:kuleshov-group/proseco-owt"}


def test_merge_unions_and_returns_new_object():
    a = _cand(sources=["hf-search:owt"], abstract="")
    b = _cand(sources=["arxiv-kw:OpenWebText"], abstract="We train on OWT.", comments="code: github.com/a/b",
              weights=[Weight(kind="hf", ref="A/B")])
    m = a.merge(b)
    assert m is not a and a.sources == ["hf-search:owt"]
    assert m.sources == ["hf-search:owt", "arxiv-kw:OpenWebText"]
    assert m.abstract == "We train on OWT." and m.comments == "code: github.com/a/b"
    assert [w.ref for w in m.weights] == ["A/B"]


def test_merge_dedupes_weights_case_insensitively():
    a = _cand(weights=[Weight(kind="hf", ref="A/B")])
    b = _cand(weights=[Weight(kind="hf", ref="a/b")])
    assert len(a.merge(b).weights) == 1


def test_candidate_json_round_trip():
    c = _cand(published=date(2026, 2, 12), repos=["https://github.com/a/b"])
    assert Candidate.model_validate_json(c.model_dump_json()) == c


def test_registry_entry_keys_skip_sampler_ref():
    e = RegistryEntry(id="remdm", name="ReMDM", paper=Paper(arxiv="2503.00307"), github="https://github.com/kuleshov-group/remdm",
                      checkpoint=Checkpoint(kind="sampler", ref="mdlm-owt"), family="masked-dlm", train_data="owt",
                      status="queued", added=date(2026, 10, 8), source="seed")
    assert e.keys() == {"arxiv:2503.00307", "github:kuleshov-group/remdm"}


def test_registry_entry_rejects_unknown_family():
    with pytest.raises(ValidationError):
        RegistryEntry(id="x", name="x", github="https://github.com/a/b", checkpoint=Checkpoint(kind="hf", ref="a/b"),
                      family="transformer", train_data="owt", status="queued", added=date(2026, 10, 8), source="seed")


def _scored(**over):
    base = dict(id="x", name="X", github="https://github.com/a/b", checkpoint=Checkpoint(kind="hf", ref="a/b"),
                family="masked-dlm", train_data="owt", status="scored", added=date(2026, 10, 9), source="seed",
                group="discrete", params="170M", paper=Paper(arxiv="2406.07524", published=date(2024, 6, 11)))
    base.update(over)
    return RegistryEntry(**base)


def test_scored_entry_with_site_fields_is_valid():
    assert _scored().group == "discrete"
    assert _scored(paper=Paper(url="https://cdn.openai.com/x.pdf", published=date(2019, 2, 14))).paper.url


@pytest.mark.parametrize("over", [
    {"group": None}, {"params": None}, {"paper": None},
    {"paper": Paper(arxiv="2406.07524")},           # no published date
    {"paper": Paper(published=date(2024, 6, 11))},  # neither arxiv nor url
])
def test_scored_entry_missing_site_fields_is_rejected(over):
    with pytest.raises(ValidationError):
        _scored(**over)


def test_group_must_be_one_of_three():
    with pytest.raises(ValidationError):
        _scored(group="flow")


def test_queued_entry_needs_no_site_fields():
    assert _scored(status="queued", group=None, params=None, paper=None).status == "queued"
