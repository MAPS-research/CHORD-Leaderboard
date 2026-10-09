from datetime import date
from pathlib import Path

import pytest

from discovery.apply_decision import accept_entries, apply, reject_entry
from discovery.models import Candidate, CheckpointRef, Judgement, Verdict, Weight
from discovery.registry import load_models, load_rejected
from discovery.report import render_issue

TODAY = date(2026, 10, 12)
CAND = Candidate(
    arxiv_id="2602.11590", title="Self-Correcting Masked Diffusion", sources=["hf-search:owt"],
    repos=["https://github.com/kuleshov-group/proseco", "https://github.com/kuleshov-group/mdlm"],
    weights=[Weight(kind="hf", ref="kuleshov-group/proseco-owt", check="ok"), Weight(kind="hf", ref="kuleshov-group/mdlm-owt", check="ok")],
    verdict=Verdict(owt_trained=Judgement(value="yes"), official_repo="https://github.com/kuleshov-group/proseco",
                    official_checkpoints=[CheckpointRef(kind="hf", ref="kuleshov-group/proseco-owt")], family_guess="masked-dlm"))


def _event(cand=CAND, labels=("candidate", "accept"), body=None):
    return {"issue": {"number": 42, "labels": [{"name": n} for n in labels], "body": body if body is not None else render_issue(cand)[1]}}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "models.yaml").write_text("[]\n")
    (tmp_path / "registry" / "rejected.yaml").write_text("# Rejected candidates\n[]\n")
    return tmp_path


def test_accept_uses_official_checkpoints_only():
    entries = accept_entries(CAND, 42, TODAY, taken=set())
    assert [(e.id, e.checkpoint.ref) for e in entries] == [("proseco-owt", "kuleshov-group/proseco-owt")]
    e = entries[0]
    assert e.github == "https://github.com/kuleshov-group/proseco" and e.paper.arxiv == "2602.11590"
    assert e.status == "queued" and e.source == "discovery#42" and e.added == TODAY and e.name == "proseco"


def test_accept_falls_back_to_ok_weights_and_suffixes_collisions():
    cand = CAND.model_copy(update={"verdict": CAND.verdict.model_copy(update={"official_checkpoints": []})})
    ids = [e.id for e in accept_entries(cand, 42, TODAY, taken={"proseco-owt"})]
    assert ids == ["proseco-owt-2", "mdlm-owt"]


def test_accept_without_any_checkpoint_fails():
    cand = CAND.model_copy(update={"weights": [], "verdict": CAND.verdict.model_copy(update={"official_checkpoints": []})})
    with pytest.raises(ValueError, match="no checkpoint"):
        accept_entries(cand, 42, TODAY, taken=set())


def test_apply_accept_writes_valid_yaml_with_todo_comments(root):
    cand = CAND.model_copy(update={"verdict": CAND.verdict.model_copy(update={"family_guess": None})})
    body = apply("accept", _event(cand), [], root, TODAY)
    text = (root / "registry" / "models.yaml").read_text()
    assert "# TODO(sampler): fill from paper" in text and "# TODO(review): family unknown" in text
    assert [m.id for m in load_models(root / "registry" / "models.yaml")] == ["proseco-owt"]
    assert body.startswith("Closes #42")


def test_apply_reject_uses_last_reason_comment(root):
    comments = [{"body": "reason: first"}, {"body": "looks fine"}, {"body": "Reason: trained on C4, not OWT"}]
    apply("reject", _event(labels=("candidate", "reject")), comments, root, TODAY)
    text = (root / "registry" / "rejected.yaml").read_text()
    assert text.startswith("# Rejected candidates")
    [r] = load_rejected(root / "registry" / "rejected.yaml")
    assert r.key == "arxiv:2602.11590" and r.reason == "trained on C4, not OWT" and r.source == "discovery#42"


def test_reject_default_reason():
    assert reject_entry(CAND, 42, TODAY, comments=[]).reason == "rejected by maintainer"


def test_broken_issue_body_fails_without_touching_files(root):
    before = (root / "registry" / "models.yaml").read_text()
    with pytest.raises(ValueError, match="json candidate"):
        apply("accept", _event(body="I edited this issue and removed the block"), [], root, TODAY)
    assert (root / "registry" / "models.yaml").read_text() == before


def test_non_candidate_issue_is_refused(root):
    with pytest.raises(ValueError, match="candidate"):
        apply("accept", _event(labels=("accept",)), [], root, TODAY)


def test_accept_fills_group_from_the_generation_type():
    from discovery.models import TypeJudgement
    v = CAND.verdict.model_copy(update={"generation_type": TypeJudgement(value="discrete")})
    [e] = accept_entries(CAND.model_copy(update={"verdict": v}), 42, TODAY, taken=set())
    assert e.group == "discrete"


def test_accept_with_unclear_type_leaves_group_for_the_reviewer(root):
    body = apply("accept", _event(), [], root, TODAY)
    text = (root / "registry" / "models.yaml").read_text()
    [e] = load_models(root / "registry" / "models.yaml")
    assert e.group is None and "# TODO(review): generation type unclear" in text
