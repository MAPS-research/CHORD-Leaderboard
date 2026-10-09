"""Sanity checks for the Table 2 scoring configs used on the cluster."""

import json
from collections import Counter
from pathlib import Path

import yaml

SCORING = Path(__file__).resolve().parents[1] / "scoring"
PAPER_MODELS = {"GPT2-medium-AR", "GPT2-large-AR", "MDLM-OWT", "SEDD-small", "LangFlow-OWT", "ELF-L-OWT"}
NEW_MODELS = {"GPT2-small-AR", "GPT2-XL-AR", "SEDD-medium", "ELF-B-OWT", "ELF-M-OWT"}


def _runs():
    return [json.loads(line) for line in (SCORING / "configs" / "runs.jsonl").read_text().splitlines() if line.strip()]


def test_runs_cover_paper_and_new_models_with_ten_paper_seeds():
    runs = _runs()
    assert len({r["run_id"] for r in runs}) == len(runs)
    per_model = Counter(r["model"] for r in runs)
    assert per_model.pop("Human-packed") == 1
    assert set(per_model) == PAPER_MODELS | NEW_MODELS and set(per_model.values()) == {10}
    for r in runs:
        if r["model"] != "Human-packed":
            k = int(r["run_id"].rsplit("-s", 1)[1])
            assert r["seed"] == 20260800 + k and r["samples_path"].startswith("/scratch/jz5770/")


def test_featurize_config_matches_the_paper_protocol():
    cfg = yaml.safe_load((SCORING / "configs" / "featurize.yaml").read_text())
    assert cfg["seed"] == 20260920 and cfg["reference"]["limit"] == 5150
    [enc] = cfg["encoders"]
    assert enc["name"] == "qwen35-27b-prompteol-coherence-l62"
    p = enc["protocol"]
    assert (p["model"], p["revision"], p["layer"], p["max_length"], p["prompteol_template"]) == (
        "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", -3, 532, "coherence")


def test_elf_sampling_configs_follow_the_elf_readme():
    b = yaml.safe_load((SCORING / "configs" / "elf_b_sde32_sccfg3.yaml").read_text())
    m = yaml.safe_load((SCORING / "configs" / "elf_m_sde64_sccfg3.yaml").read_text())
    assert (b[0]["num_sampling_steps"], b[0]["sde_gamma"], b[0]["self_cond_cfg_scales"]) == ([32], 1.5, [3])
    assert (m[0]["num_sampling_steps"], m[0]["sde_gamma"], m[0]["self_cond_cfg_scales"]) == ([64], 1.0, [3])
