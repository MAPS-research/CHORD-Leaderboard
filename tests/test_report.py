from datetime import date

import pytest

from discovery.models import Candidate, CheckpointRef, Evidence, Judgement, Verdict, Weight
from discovery.report import classify, issue_keys, needs_manual_check, parse_candidate, render_digest, render_issue

YES = Judgement(value="yes", evidence=[Evidence(quote="trained on OpenWebText", url="https://arxiv.org/abs/2602.11590")])


def _c(verdict=None, weights=None, **kw):
    return Candidate(arxiv_id="2602.11590", title="ProSeCo", abstract="abs", sources=["hf-search:owt"],
                     weights=weights if weights is not None else [Weight(kind="hf", ref="a/b", check="ok")],
                     verdict=verdict, readme="x" * 30000, **kw)


def test_classify_rules():
    assert classify(_c(Verdict(owt_trained=YES, unconditional=YES))) == "issue"
    assert classify(_c(Verdict(owt_trained=Judgement(value="no")))) is None
    assert classify(_c(Verdict(owt_trained=Judgement(value="unclear")), weights=[Weight(kind="gdrive", ref="u", check="unverifiable")])) == "digest"
    assert classify(_c(Verdict(owt_trained=YES), weights=[Weight(kind="hf", ref="a/b", check="missing")])) is None
    assert classify(_c(Verdict(owt_trained=YES, official_checkpoints=[CheckpointRef(kind="hf", ref="a/b")]),
                       weights=[Weight(kind="hf", ref="a/b", check="gated")])) == "issue"
    assert classify(_c(None, judge_error="JudgeError: down")) == "digest"


def test_digest_lists_every_candidate_and_stays_small():
    cands = [_c(Verdict(owt_trained=Judgement(), notes="n" * 2000)).model_copy(update={"arxiv_id": f"2610.{i:05d}"}) for i in range(150)]
    cands.append(Candidate(sources=["hf-search:owt"], weights=[Weight(kind="hf", ref="x/m-owt", check="ok")], judge_error="JudgeError: x"))
    title, body, labels = render_digest(cands, date(2026, 10, 12))
    assert title == "[digest] 151 unclear candidates (run of 2026-10-12)" and labels == ["candidate-digest"]
    assert "https://arxiv.org/abs/2610.00000" in body and "x/m-owt" in body and "JudgeError: x" in body
    assert len(body) < 65536


def test_needs_manual_check():
    assert not needs_manual_check(_c(Verdict(owt_trained=YES, unconditional=YES)))
    assert needs_manual_check(_c(Verdict(owt_trained=YES, unconditional=Judgement())))
    assert needs_manual_check(_c(Verdict(owt_trained=YES, unconditional=YES), weights=[Weight(kind="box", ref="u", check="unverifiable")]))
    assert needs_manual_check(_c(None, judge_error="x"))


def test_render_and_parse_round_trip_without_readme():
    c = _c(Verdict(owt_trained=YES, unconditional=YES, official_repo="https://github.com/a/b", notes="n"),
           repos=["https://github.com/a/b"], published=None)
    title, body, labels = render_issue(c)
    assert title == "[candidate] ProSeCo (arXiv:2602.11590)"
    assert labels == ["candidate"]
    assert "<!-- cand: arxiv:2602.11590 -->" in body and "trained on OpenWebText" in body
    assert len(body) < 65536
    parsed = parse_candidate(body)
    assert parsed == c.model_copy(update={"readme": ""})
    assert issue_keys(body) == {"arxiv:2602.11590", "github:a/b"}


def test_render_hf_only_and_judge_error():
    c = Candidate(sources=["hf-search:owt"], weights=[Weight(kind="hf", ref="x/ermine-owt", check="ok")], judge_error="JudgeError: down")
    title, body, labels = render_issue(c)
    assert title == "[candidate] x/ermine-owt (HF Hub, no paper)"
    assert "Automatic verdict failed: JudgeError: down" in body
    assert labels == ["candidate", "needs-manual-check"]


def test_parse_candidate_reports_broken_block():
    with pytest.raises(ValueError, match="json candidate"):
        parse_candidate("someone deleted the block")
    with pytest.raises(ValueError, match="could not be parsed"):
        parse_candidate("```json candidate\n{not json\n```")


def test_parse_candidate_accepts_crlf_bodies_from_web_edits():
    c = _c(Verdict(owt_trained=YES, unconditional=YES))
    body = render_issue(c)[1].replace("\n", "\r\n")
    assert parse_candidate(body).key == c.key
    assert issue_keys(body) == {"arxiv:2602.11590"}


def test_markers_cover_every_dedupe_key():
    c = _c(Verdict(owt_trained=YES, official_repo="https://github.com/a/b",
                   official_checkpoints=[CheckpointRef(kind="hf", ref="A/B")]), repos=["https://github.com/a/b"])
    assert issue_keys(render_issue(c)[1]) == {"arxiv:2602.11590", "github:a/b", "hf:a/b"}


def test_huge_candidates_stay_under_the_issue_body_limit():
    long_quote = Evidence(quote="q" * 5000, url="https://arxiv.org/abs/2602.11590")
    v = Verdict(owt_trained=Judgement(value="yes", evidence=[long_quote] * 5), unconditional=YES)
    c = _c(v, weights=[Weight(kind="hf", ref=f"org/model-{i:04d}-" + "x" * 60, check="ok") for i in range(800)])
    c = c.model_copy(update={"abstract": "a" * 20000})
    body = render_issue(c)[1]
    assert len(body) < 65536
    assert parse_candidate(body).key == c.key


def test_issue_shows_generation_type_with_quote():
    from discovery.models import TypeJudgement
    gt = TypeJudgement(value="continuous", evidence=[Evidence(quote="embedding flow matching", url="https://arxiv.org/abs/2602.11590")])
    body = render_issue(_c(Verdict(owt_trained=YES, unconditional=YES, generation_type=gt)))[1]
    assert "**Generation type:** Continuous diffusion/flow" in body and "embedding flow matching" in body
