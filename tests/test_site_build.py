import json
from datetime import date
from pathlib import Path

import pytest

from discovery.models import Checkpoint, Paper, RegistryEntry
from leaderboard_site.build import BuildError, build, checkpoint_link, load_results, parse_params


def _entry(id_, name, group="discrete", status="scored", paper=None, added=date(2026, 10, 8), family=None):
    family = family or ("flow" if id_.startswith("elf") else "masked-dlm")
    return RegistryEntry(id=id_, name=name, github=f"https://github.com/o/{id_}", checkpoint=Checkpoint(kind="hf", ref=f"o/{id_}"),
                         family=family, group=group, params="170M", train_data="owt", status=status, added=added, source="seed",
                         paper=paper or Paper(arxiv="2406.07524", published=date(2024, 6, 11)))


def _result(dir_: Path, id_, folds, scored_on="2026-09-23", **over):
    mean = sum(folds) / len(folds)
    std = (sum((f - mean) ** 2 for f in folds) / (len(folds) - 1)) ** 0.5
    data = {"id": id_, "setting": "unconditional-owt", "encoder": "qwen35-27b-prompteol-coherence-l62",
            "metric": "rbf_mmd2_x100_biased", "folds": folds, "samples_per_fold": 500,
            "mean": round(mean, 4), "std": round(std, 4), "scored_on": scored_on}
    data.update(over)
    (dir_ / f"{id_}.json").write_text(json.dumps(data))


@pytest.fixture
def results(tmp_path: Path) -> Path:
    _result(tmp_path, "mdlm-owt", [57.0, 59.0])
    _result(tmp_path, "gpt2-large", [19.0, 21.0], scored_on="2026-10-09")
    _result(tmp_path, "human-heldout", [0.1, 0.2])
    return tmp_path


GPT2 = Paper(url="https://cdn.openai.com/gpt2.pdf", published=date(2019, 2, 14))
MODELS = [_entry("mdlm-owt", "MDLM"), _entry("gpt2-large", "GPT-2-large", group="ar", paper=GPT2),
          _entry("sedd-medium", "SEDD-medium", status="queued")]


def test_parse_params():
    assert parse_params("355M") == 355_000_000 and parse_params("1.5B") == 1_500_000_000
    with pytest.raises(BuildError):
        parse_params("big")


def test_build_joins_scored_entries_with_results(results):
    data = build(MODELS, load_results(results))
    assert [g["id"] for g in data["generators"]] == ["gpt2-large", "mdlm-owt"]  # sorted by mean; queued entry absent
    gpt2 = data["generators"][0]
    assert gpt2 == {"id": "gpt2-large", "name": "GPT-2-large", "group": "ar", "params": "170M", "params_value": 170_000_000,
                    "released": "2019-02-14", "paper": "https://cdn.openai.com/gpt2.pdf", "code": "https://github.com/o/gpt2-large",
                    "mean": 20.0, "std": 1.4142, "checkpoint": {"label": "o/gpt2-large", "url": "https://huggingface.co/o/gpt2-large"}}
    assert data["generators"][1]["paper"] == "https://arxiv.org/abs/2406.07524"
    assert data["human"] == {"name": "Held-out human (packed)", "mean": 0.15, "std": 0.0707}
    assert data["last_modified"] == "2026-10-09"


def test_last_modified_includes_registry_additions(results):
    later = MODELS[:1] + [_entry("gpt2-large", "GPT-2-large", group="ar", paper=GPT2, added=date(2026, 10, 12))]
    assert build(later, load_results(results))["last_modified"] == "2026-10-12"


def test_orphan_results_file_fails(results):
    _result(results, "elf-b-owt", [50.0, 52.0])
    with pytest.raises(BuildError, match="elf-b-owt"):
        build(MODELS, load_results(results))


def test_scored_entry_without_results_fails(results):
    (results / "mdlm-owt.json").unlink()
    with pytest.raises(BuildError, match="mdlm-owt"):
        build(MODELS, load_results(results))


def test_results_must_match_their_folds(tmp_path):
    _result(tmp_path, "mdlm-owt", [57.0, 59.0], mean=12.0)
    with pytest.raises(BuildError, match="mdlm-owt"):
        load_results(tmp_path)


def test_results_file_name_must_match_id(tmp_path):
    _result(tmp_path, "mdlm-owt", [57.0, 59.0], id="other")
    with pytest.raises(BuildError, match="mdlm-owt.json"):
        load_results(tmp_path)


def test_missing_human_reference_fails(results):
    (results / "human-heldout.json").unlink()
    with pytest.raises(BuildError, match="human-heldout"):
        build(MODELS, load_results(results))


def test_queued_entries_are_listed_as_pending(results):
    pending_models = MODELS + [
        _entry("elf-b-owt", "ELF-B", group=None, status="queued", paper=Paper(arxiv="2605.10938", published=date(2026, 5, 11))),
        RegistryEntry(id="gpt2-xl", name="GPT-2 XL", github="https://github.com/openai/gpt-2", checkpoint=Checkpoint(kind="hf", ref="gpt2-xl"),
                      family="ar", train_data="webtext", status="queued", added=date(2026, 10, 8), source="seed"),
        RegistryEntry(id="remdm", name="ReMDM", github="https://github.com/k/remdm", checkpoint=Checkpoint(kind="sampler", ref="mdlm-owt"),
                      family="masked-dlm", train_data="owt", status="queued", added=date(2026, 10, 8), source="seed"),
    ]
    pending = build(pending_models, load_results(results))["pending"]
    assert [p["id"] for p in pending] == ["elf-b-owt", "sedd-medium", "gpt2-xl"]  # newest first, unknown dates last; samplers skipped
    elf, sedd, gpt2 = pending
    assert elf == {"id": "elf-b-owt", "name": "ELF-B", "group": "continuous", "params": "170M",
                   "released": "2026-05-11", "paper": "https://arxiv.org/abs/2605.10938", "code": "https://github.com/o/elf-b-owt",
                   "checkpoint": {"label": "o/elf-b-owt", "url": "https://huggingface.co/o/elf-b-owt"}}
    assert sedd["group"] == "discrete"            # explicit group wins
    assert gpt2["group"] == "ar" and gpt2["released"] is None and gpt2["paper"] is None and gpt2["params"] is None


def test_pending_entries_do_not_affect_last_modified(results):
    later = MODELS + [_entry("new", "New", status="queued", added=date(2027, 1, 1))]
    assert build(later, load_results(results))["last_modified"] == "2026-10-09"


def test_pending_names_sort_numbers_by_value(results):
    same_day = Paper(arxiv="2503.09573", published=date(2025, 3, 12))
    variants = [_entry(f"bd3lm-{b}", f"BD3-LM (block size {b})", status="queued", paper=same_day) for b in (16, 4, 8)]
    names = [p["name"] for p in build(MODELS[:2] + variants, load_results(results))["pending"]]
    assert names == ["BD3-LM (block size 4)", "BD3-LM (block size 8)", "BD3-LM (block size 16)"]


def test_checkpoint_link_points_to_the_hf_repo():
    assert checkpoint_link(Checkpoint(kind="hf", ref="openai-community/gpt2")) == {
        "label": "openai-community/gpt2", "url": "https://huggingface.co/openai-community/gpt2"}


def test_checkpoint_link_points_to_a_file_inside_an_hf_repo():
    assert checkpoint_link(Checkpoint(kind="hf", ref="jdeschena/duo2-owt", path="duo/61-1000000.safetensors")) == {
        "label": "jdeschena/duo2-owt/duo/61-1000000.safetensors",
        "url": "https://huggingface.co/jdeschena/duo2-owt/blob/main/duo/61-1000000.safetensors"}


def test_checkpoint_link_for_files_hosted_elsewhere_uses_the_ref_url():
    release = checkpoint_link(Checkpoint(kind="github-release", ref="https://github.com/igul222/plaid/releases/tag/v1.0.0"))
    assert release == {"label": "GitHub release v1.0.0", "url": "https://github.com/igul222/plaid/releases/tag/v1.0.0"}
    dropbox = checkpoint_link(Checkpoint(kind="dropbox", ref="https://github.com/LituRout/ADLM", path="adlm-large.ckpt"))
    assert dropbox == {"label": "adlm-large.ckpt (Dropbox, linked from the README)", "url": "https://github.com/LituRout/ADLM"}


def test_scored_generators_carry_their_checkpoint_link(results):
    gens = build(MODELS, load_results(results))["generators"]
    assert gens[0]["checkpoint"] == {"label": "o/gpt2-large", "url": "https://huggingface.co/o/gpt2-large"}
