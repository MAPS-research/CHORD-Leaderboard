"""Real-API checks of the verdict on known papers. Run with: pytest -m live tests/test_live_verdicts.py"""

import pytest
from huggingface_hub import HfApi

from discovery import checks, enrich, judge
from discovery.config import load_config
from discovery.http import make_client
from discovery.models import Candidate, Weight
from discovery.report import classify
from discovery.sources import arxiv

pytestmark = pytest.mark.live


def _screen(cand: Candidate) -> Candidate:
    http = make_client()
    if cand.arxiv_id:
        cand = cand.merge(arxiv.fetch_by_ids(http, [cand.arxiv_id])[cand.arxiv_id])
    cand = checks.run_checks(HfApi(), http, enrich.enrich(http, cand, 30000))
    out = judge.judge(judge.run_claude, load_config().judge_model, cand)
    assert out.verdict is not None, out.judge_error
    return out


def test_proseco_qualifies():
    c = _screen(Candidate(arxiv_id="2602.11590", sources=["live"]))
    assert c.verdict.owt_trained.value == "yes" and c.verdict.unconditional.value == "yes"
    assert any(w.ref.lower() == "kuleshov-group/proseco-owt" and w.check == "ok" for w in c.weights)
    assert classify(c) == "issue"


def test_eso_lm_official_checkpoints_are_the_public_copies():
    c = _screen(Candidate(sources=["live"], repos=["https://github.com/s-sahoo/Eso-LMs"],
                          weights=[Weight(kind="hf", ref="sahoo-diffusion/Eso-LM-B-alpha-1")]))
    assert c.verdict.owt_trained.value == "yes"
    assert any(k.ref.lower().startswith("sahoo-diffusion/") for k in c.verdict.official_checkpoints)


def test_rdlm_is_not_judged_owt():
    c = _screen(Candidate(arxiv_id="2502.11564", sources=["live"]))
    assert "diffusion" in c.title.lower()
    assert c.verdict.owt_trained.value != "yes"


def test_md4_has_no_official_checkpoint():
    c = _screen(Candidate(arxiv_id="2406.04329", sources=["live"]))
    assert "masked diffusion" in c.title.lower(), f"wrong arXiv id for MD4: {c.title}"
    assert c.verdict.official_checkpoints == []
