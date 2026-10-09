from datetime import date
from pathlib import Path

import pytest

from discovery import pipeline
from discovery.config import load_config
from discovery.http import HttpError
from discovery.models import Candidate, CheckpointRef, Judgement, Verdict, Weight
from discovery.pipeline import Deps, RunSummary, passes_prefilter, render_summary, repeated_failures, run

CFG = load_config().model_copy(update={"max_judge_calls": 2, "judge_workers": 1})
SINCE = date(2026, 9, 17)


def _c(arxiv, title="A diffusion language model", sources=("arxiv-kw",), published=date(2026, 10, 1), **kw):
    return Candidate(arxiv_id=arxiv, title=title, abstract="abs", sources=list(sources), published=published, **kw)


class FakeGitHub:
    def __init__(self, keys=frozenset(), fail=False):
        self.keys, self.fail, self.created = set(keys), fail, []

    def candidate_issue_keys(self):
        if self.fail:
            raise HttpError("boom")
        return self.keys

    def create_issue(self, title, body, labels):
        self.created.append(title)
        return 100 + len(self.created)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "models.yaml").write_text(
        "- id: mdlm-owt\n  name: MDLM\n  paper: {arxiv: '2406.07524'}\n  github: https://github.com/kuleshov-group/mdlm\n"
        "  checkpoint: {kind: hf, ref: kuleshov-group/mdlm-owt}\n  family: masked-dlm\n  train_data: owt\n"
        "  status: queued\n  added: 2026-10-08\n  source: seed\n")
    (reg / "rejected.yaml").write_text("- {key: 'arxiv:2609.00009', name: R, reason: r, decided: 2026-10-08, source: seed}\n")
    return tmp_path


@pytest.fixture
def stubs(monkeypatch):
    """Replace every network-facing stage with deterministic fakes; tests override single stages."""
    calls = {"judged": []}
    monkeypatch.setattr(pipeline.arxiv, "search", lambda *a, **k: [
        _c("2609.00001"), _c("2406.07524"), _c("2609.00009"), _c("2609.00003", title="A survey of protein folding"),
        _c("2609.00002", published=date(2026, 9, 20))])
    monkeypatch.setattr(pipeline.arxiv, "fetch_by_ids", lambda http, ids, sleep: {})
    monkeypatch.setattr(pipeline.hf_hub, "search", lambda api, queries, since: [
        Candidate(arxiv_id="2609.00001", sources=["hf-search:owt"], weights=[Weight(kind="hf", ref="x/m-owt")])])
    monkeypatch.setattr(pipeline.enrich, "enrich", lambda http, c, n, sleep: c.model_copy(update={"weights": [Weight(kind="hf", ref=f"x/{c.arxiv_id}")]}))
    monkeypatch.setattr(pipeline.checks, "run_checks", lambda hf, http, c, sleep: c.model_copy(
        update={"weights": [w.model_copy(update={"check": "ok"}) for w in c.weights]}))

    def fake_judge(llm, model, c, sleep):
        calls["judged"].append(c.arxiv_id)
        return c.model_copy(update={"verdict": Verdict(owt_trained=Judgement(value="yes"))})

    monkeypatch.setattr(pipeline.judge, "judge", fake_judge)
    return calls


def _deps(github=None):
    return Deps(http=None, hf=None, llm=None, github=github, sleep=lambda _: None)


def test_full_run_merges_dedupes_prefilters_and_reports(root, stubs):
    gh = FakeGitHub()
    summary, judged = run(CFG, _deps(gh), root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert summary.harvested == {"arxiv": 5, "hf_hub": 1}
    assert summary.new == 3 and summary.after_prefilter == 2      # 2406.07524 known, 2609.00009 rejected; protein paper filtered
    assert stubs["judged"] == ["2609.00001", "2609.00002"]       # newest first
    assert summary.reported == ["arxiv:2609.00001", "arxiv:2609.00002"] and summary.opened == [101, 102]
    assert {c.key for c in judged} == set(summary.reported)
    merged = next(c for c in judged if c.arxiv_id == "2609.00001")
    assert merged.sources == ["arxiv-kw", "hf-search:owt"]


def test_issue_markers_dedupe(root, stubs):
    gh = FakeGitHub(keys={"arxiv:2609.00001"})
    summary, _ = run(CFG, _deps(gh), root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert summary.reported == ["arxiv:2609.00002"]


def test_dry_run_and_no_dedupe(root, stubs):
    summary, _ = run(CFG.model_copy(update={"max_judge_calls": 10}), _deps(None), root=root, since=SINCE, dry_run=True, no_dedupe=True)
    assert summary.opened == [] and "arxiv:2406.07524" in summary.reported


def test_source_failure_is_isolated(root, stubs, monkeypatch):
    def boom(*a, **k):
        raise HttpError("arXiv API: HTTP 503")
    monkeypatch.setattr(pipeline.arxiv, "search", boom)
    summary, _ = run(CFG, _deps(FakeGitHub()), root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert "arxiv" in summary.errors and "503" in summary.errors["arxiv"]
    assert summary.reported == ["arxiv:2609.00001"] and not summary.all_sources_failed  # still found via the HF Hub


def test_all_sources_failing(root, stubs, monkeypatch):
    def boom(*a, **k):
        raise HttpError("down")
    for mod, name in ((pipeline.arxiv, "search"), (pipeline.hf_hub, "search")):
        monkeypatch.setattr(mod, name, boom)
    summary, _ = run(CFG, _deps(FakeGitHub()), root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert summary.all_sources_failed and summary.opened == []


def test_issue_listing_failure_blocks_reporting(root, stubs):
    gh = FakeGitHub(fail=True)
    summary, _ = run(CFG, _deps(gh), root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert "github_issues" in summary.errors and gh.created == []


def test_judge_outage_is_a_source_error(root, stubs, monkeypatch):
    monkeypatch.setattr(pipeline.judge, "judge", lambda llm, m, c, sleep: c.model_copy(update={"judge_error": "JudgeError: 529"}))
    gh = FakeGitHub()
    summary, _ = run(CFG.model_copy(update={"max_judge_calls": 10}), _deps(gh), root=root, since=SINCE, dry_run=False, no_dedupe=True)
    assert summary.errors["judge"] == "JudgeError: 529" and gh.created == []


def test_cap_defers_oldest(root, stubs):
    summary, _ = run(CFG.model_copy(update={"max_judge_calls": 1}), _deps(FakeGitHub()), root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert summary.deferred == 1 and stubs["judged"] == ["2609.00001"]


def test_prefilter_skips_hf_and_matches_case_insensitively():
    assert passes_prefilter(Candidate(sources=["hf-search:owt"], weights=[Weight(kind="hf", ref="a/b")]), ["diffusion"])
    assert passes_prefilter(_c("1", title="Masked DIFFUSION"), ["diffusion"])
    assert not passes_prefilter(_c("1", title="Protein folding"), ["diffusion"])


def test_repeated_failures_and_summary():
    prev = RunSummary(since=SINCE, errors={"arxiv": "x", "hf_hub": "y"})
    cur = RunSummary(since=SINCE, errors={"arxiv": "z"}, harvested={"hf_hub": 3})
    assert repeated_failures(None, cur) == [] and repeated_failures(prev, cur) == ["arxiv"]
    md = render_summary(cur, ["arxiv"])
    assert "arxiv" in md and "hf_hub | 3" in md and "consecutive" in md


def test_same_model_found_as_paper_and_as_hf_only_is_reported_once(root, stubs, monkeypatch):
    monkeypatch.setattr(pipeline.hf_hub, "search", lambda api, queries, since: [
        Candidate(sources=["hf-search:owt"], weights=[Weight(kind="hf", ref="x/2609.00001")], published=date(2026, 10, 2))])
    monkeypatch.setattr(pipeline.enrich, "enrich", lambda http, c, n, sleep: c if not c.arxiv_id else c.model_copy(
        update={"weights": [Weight(kind="hf", ref=f"x/{c.arxiv_id}")]}))
    monkeypatch.setattr(pipeline.judge, "judge", lambda llm, m, c, sleep: c.model_copy(update={"verdict": Verdict(
        owt_trained=Judgement(value="yes"), official_checkpoints=[CheckpointRef(kind="hf", ref=c.weights[0].ref)])}))
    summary, _ = run(CFG.model_copy(update={"max_judge_calls": 10}), _deps(FakeGitHub()), root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert summary.reported == ["arxiv:2609.00001", "arxiv:2609.00002"]  # the HF-only copy of 2609.00001 is dropped


def test_one_failed_issue_creation_does_not_stop_the_run(root, stubs):
    class FlakyGitHub(FakeGitHub):
        def create_issue(self, title, body, labels):
            if not self.created:
                self.created.append("failed")
                raise HttpError("create issue: HTTP 422")
            return super().create_issue(title, body, labels)

    gh = FlakyGitHub()
    summary, _ = run(CFG, _deps(gh), root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert "422" in summary.errors["github_create"] and summary.opened == [102]


def test_notify_repeated_failures_never_raises():
    class Broken:
        def upsert_failure_issue(self, body):
            raise HttpError("403")

    assert "403" in pipeline.notify_repeated(Broken(), "md")
    assert pipeline.notify_repeated(None, "md") is None


def test_semantic_scholar_is_never_contacted(root, stubs):
    (root / "registry" / "seeds.yaml").write_text('- {name: MDLM, arxiv: "2406.07524", title: "Simple"}\n')

    class NoNetwork:
        def request(self, method, url, **kw):
            raise AssertionError(f"unexpected HTTP call: {method} {url}")

    deps = Deps(http=NoNetwork(), hf=None, llm=None, github=FakeGitHub(), sleep=lambda _: None)
    summary, _ = run(CFG, deps, root=root, since=SINCE, dry_run=False, no_dedupe=False)
    assert set(summary.harvested) == {"arxiv", "hf_hub"} and summary.errors == {}


def test_judge_calls_run_concurrently_and_keep_order(root, stubs, monkeypatch):
    import time as _time

    def slow_judge(llm, model, c, sleep):
        _time.sleep(0.3)
        return c.model_copy(update={"verdict": Verdict(owt_trained=Judgement(value="yes"))})

    monkeypatch.setattr(pipeline.judge, "judge", slow_judge)
    cfg = CFG.model_copy(update={"max_judge_calls": 10, "judge_workers": 4})
    start = _time.monotonic()
    summary, _ = run(cfg, _deps(FakeGitHub()), root=root, since=SINCE, dry_run=True, no_dedupe=True)
    assert _time.monotonic() - start < 0.8  # 4 candidates x 0.3 s would take 1.2 s sequentially
    assert summary.reported == ["arxiv:2609.00001", "arxiv:2406.07524", "arxiv:2609.00009", "arxiv:2609.00002"]


TODAY = date(2026, 10, 8)


def test_seen_candidates_are_skipped_and_remembered(root, stubs):
    seen = {"arxiv:2609.00001": date(2026, 10, 1), "arxiv:2501.00001": date(2025, 1, 5)}
    summary, _ = run(CFG, _deps(FakeGitHub()), root=root, since=SINCE, dry_run=False, no_dedupe=False, seen=seen, today=TODAY)
    assert stubs["judged"] == ["2609.00002"] and summary.skipped_seen == 1
    assert summary.judged_on == {"arxiv:2609.00001": date(2026, 10, 1), "arxiv:2609.00002": TODAY}


def test_judge_failures_are_not_remembered(root, stubs, monkeypatch):
    monkeypatch.setattr(pipeline.judge, "judge", lambda llm, m, c, sleep: c.model_copy(update={"judge_error": "JudgeError: x"}))
    summary, _ = run(CFG, _deps(FakeGitHub()), root=root, since=SINCE, dry_run=False, no_dedupe=False, today=TODAY)
    assert summary.judged_on == {}


def test_unclear_candidates_go_into_one_digest_issue(root, stubs, monkeypatch):
    def judge_one_unclear(llm, model, c, sleep):
        value = "yes" if c.arxiv_id == "2609.00001" else "unclear"
        return c.model_copy(update={"verdict": Verdict(owt_trained=Judgement(value=value))})

    monkeypatch.setattr(pipeline.judge, "judge", judge_one_unclear)
    gh = FakeGitHub()
    summary, _ = run(CFG, _deps(gh), root=root, since=SINCE, dry_run=False, no_dedupe=False, today=TODAY)
    assert summary.reported == ["arxiv:2609.00001"] and summary.digested == ["arxiv:2609.00002"]
    assert gh.created == ["[candidate] A diffusion language model (arXiv:2609.00001)",
                          "[digest] 1 unclear candidates (run of 2026-10-08)"]
    assert summary.opened == [101] and summary.digest_issue == 102


def test_dry_run_opens_no_digest(root, stubs, monkeypatch):
    monkeypatch.setattr(pipeline.judge, "judge", lambda llm, m, c, sleep: c.model_copy(
        update={"verdict": Verdict(owt_trained=Judgement(value="unclear"))}))
    gh = FakeGitHub()
    summary, _ = run(CFG, _deps(gh), root=root, since=SINCE, dry_run=True, no_dedupe=False, today=TODAY)
    assert gh.created == [] and summary.digest_issue is None and len(summary.digested) == 2
