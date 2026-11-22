import json
from datetime import date
from pathlib import Path

import pytest

from discovery.models import Checkpoint, Paper
from leaderboard_site.build import BuildError
from leaderboard_site.continuation import ContinuationEntry, build_continuation, load_entries, load_reported

SOURCE = "CHORD paper (rebuttal revision), Table tab:dclm-continuation"


def _entry(id_, name, group="ar", added=date(2026, 11, 22), **over):
    data = dict(id=id_, name=name, group=group, params="7B", github=f"https://github.com/o/{id_}",
                checkpoint=Checkpoint(kind="hf", ref=f"o/{id_}"), added=added,
                paper=Paper(arxiv="2406.11794", published=date(2024, 6, 17)))
    data.update(over)
    return ContinuationEntry(**data)


def _reported(dir_: Path, id_, mean, std, folds=None, scored_on="2026-11-14", **over):
    data = {"id": id_, "setting": "continuation-dclm", "encoder": "qwen35-27b-prompteol-coherence-l62",
            "metric": "rbf_mmd2_x100_biased", "n_folds": 10, "samples_per_fold": 500, "folds": folds,
            "mean": mean, "std": std, "scored_on": scored_on, "source": SOURCE}
    data.update(over)
    (dir_ / f"{id_}.json").write_text(json.dumps(data))


@pytest.fixture
def reported(tmp_path: Path) -> Path:
    _reported(tmp_path, "dclm-7b", 7.15, 0.95)
    _reported(tmp_path, "gemma3-27b-pt", 2.49, 0.35)
    _reported(tmp_path, "human-heldout", 0.17, 0.07)
    return tmp_path


ENTRIES = [_entry("dclm-7b", "DCLM-7B"),
           _entry("gemma3-27b-pt", "Gemma-3-27B-pt", paper=Paper(url="https://example.org/gemma", published=date(2025, 3, 25)))]


def test_build_joins_entries_with_reported_scores(reported):
    data = build_continuation(ENTRIES, load_reported(reported))
    assert [g["id"] for g in data["generators"]] == ["gemma3-27b-pt", "dclm-7b"]  # sorted by mean
    assert data["generators"][1] == {
        "id": "dclm-7b", "name": "DCLM-7B", "group": "ar", "params": "7B", "params_value": 7_000_000_000,
        "released": "2024-06-17", "paper": "https://arxiv.org/abs/2406.11794", "code": "https://github.com/o/dclm-7b",
        "mean": 7.15, "std": 0.95, "checkpoint": {"label": "o/dclm-7b", "url": "https://huggingface.co/o/dclm-7b"}}
    assert data["generators"][0]["paper"] == "https://example.org/gemma"
    assert data["human"] == {"name": "Held-out human", "mean": 0.17, "std": 0.07}


def test_last_modified_is_the_latest_score_or_addition(reported):
    assert build_continuation(ENTRIES, load_reported(reported))["last_modified"] == date(2026, 11, 22)
    early = [_entry("dclm-7b", "DCLM-7B", added=date(2026, 11, 1)), ENTRIES[1].model_copy(update={"added": date(2026, 11, 1)})]
    assert build_continuation(early, load_reported(reported))["last_modified"] == date(2026, 11, 14)


def test_reported_scores_without_folds_need_a_source(tmp_path):
    _reported(tmp_path, "dclm-7b", 7.15, 0.95, source=None)
    with pytest.raises(BuildError, match="dclm-7b.json"):
        load_reported(tmp_path)


def test_reported_folds_must_match_mean_and_std(tmp_path):
    _reported(tmp_path, "dclm-7b", 7.0, 1.4142, folds=[6.0, 8.0], n_folds=2)
    assert load_reported(tmp_path)["dclm-7b"].folds == [6.0, 8.0]
    _reported(tmp_path, "dclm-7b", 9.0, 1.4142, folds=[6.0, 8.0], n_folds=2)
    with pytest.raises(BuildError, match="dclm-7b.json"):
        load_reported(tmp_path)


def test_reported_folds_must_match_n_folds(tmp_path):
    _reported(tmp_path, "dclm-7b", 7.0, 1.4142, folds=[6.0, 8.0], n_folds=10)
    with pytest.raises(BuildError, match="n_folds"):
        load_reported(tmp_path)


def test_reported_file_name_must_match_id(tmp_path):
    _reported(tmp_path, "dclm-7b", 7.15, 0.95, id="other")
    with pytest.raises(BuildError, match="dclm-7b.json"):
        load_reported(tmp_path)


def test_orphan_reported_scores_fail(reported):
    _reported(reported, "tess2", 19.01, 1.03)
    with pytest.raises(BuildError, match="tess2"):
        build_continuation(ENTRIES, load_reported(reported))


def test_entry_without_reported_scores_fails(reported):
    (reported / "dclm-7b.json").unlink()
    with pytest.raises(BuildError, match="dclm-7b"):
        build_continuation(ENTRIES, load_reported(reported))


def test_missing_human_row_fails(reported):
    (reported / "human-heldout.json").unlink()
    with pytest.raises(BuildError, match="human-heldout"):
        build_continuation(ENTRIES, load_reported(reported))


def test_duplicate_registry_ids_fail(reported):
    with pytest.raises(BuildError, match="dclm-7b"):
        build_continuation(ENTRIES + [ENTRIES[0]], load_reported(reported))


def test_entries_need_a_paper_link_and_release_date():
    with pytest.raises(ValueError, match="paper"):
        _entry("x", "X", paper=Paper(published=date(2024, 1, 1)))
    with pytest.raises(ValueError, match="published"):
        _entry("x", "X", paper=Paper(arxiv="2401.00001"))


def test_load_entries_reads_the_yaml_list(tmp_path):
    (tmp_path / "continuation.yaml").write_text(
        "- {id: dclm-7b, name: DCLM-7B, group: ar, params: 7B, paper: {arxiv: '2406.11794', published: 2024-06-17},\n"
        "   github: 'https://github.com/mlfoundations/dclm', checkpoint: {kind: hf, ref: apple/DCLM-7B}, added: 2026-11-22}\n")
    [entry] = load_entries(tmp_path / "continuation.yaml")
    assert entry.id == "dclm-7b" and entry.paper.published == date(2024, 6, 17)


def test_repository_data_builds():
    from discovery.registry import REPO_ROOT
    data = build_continuation(load_entries(REPO_ROOT / "registry" / "continuation.yaml"),
                              load_reported(REPO_ROOT / "results" / "continuation-dclm"))
    assert len(data["generators"]) == 21
    assert data["generators"][0]["name"] == "Gemma-3-27B-pt"


def test_main_writes_both_boards(tmp_path):
    from leaderboard_site.build import main
    out = tmp_path / "leaderboard.json"
    assert main(["--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["continuation"]["generators"]) == 21
    assert data["continuation"]["human"]["name"] == "Held-out human"
    assert data["generators"] and data["last_modified"] >= "2026-11-22"


def test_module_entry_point_reports_continuation_errors_cleanly(tmp_path):
    import subprocess
    import sys

    from discovery.registry import REPO_ROOT
    run = subprocess.run([sys.executable, "-m", "leaderboard_site.build", "--continuation-results", str(tmp_path),
                          "--out", str(tmp_path / "out.json")], check=False, cwd=REPO_ROOT, capture_output=True, text=True)
    assert run.returncode == 1
    assert "build failed:" in run.stderr and "human-heldout" in run.stderr and "Traceback" not in run.stderr


@pytest.mark.parametrize("field, value", [("setting", "unconditional-owt"), ("encoder", "other-encoder"),
                                          ("metric", "other-metric")])
def test_reported_scores_must_be_for_this_board(tmp_path, field, value):
    _reported(tmp_path, "dclm-7b", 7.15, 0.95, **{field: value})
    with pytest.raises(BuildError, match=field):
        load_reported(tmp_path)


def test_reported_scores_need_at_least_two_folds(tmp_path):
    _reported(tmp_path, "dclm-7b", 6.0, 0.0, folds=[6.0], n_folds=1)
    with pytest.raises(BuildError, match="n_folds"):
        load_reported(tmp_path)
