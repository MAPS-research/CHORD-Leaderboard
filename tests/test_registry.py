from datetime import date
from pathlib import Path

import yaml

from discovery.config import load_config
from discovery.models import Checkpoint, Paper, RegistryEntry, RejectedEntry
from discovery.registry import known_keys, load_models, load_rejected, render_yaml_block


def test_default_config_loads():
    cfg = load_config()
    assert cfg.lookback_days == 21 and cfg.judge_model == "claude-opus-5-5" and cfg.max_readme_chars == 30000
    assert cfg.assignees == ["JimmmmmL", "JunhaoZhu0220"]


def test_empty_and_missing_files_load_as_empty(tmp_path: Path):
    (tmp_path / "m.yaml").write_text("[]\n")
    (tmp_path / "blank.yaml").write_text("")
    assert load_models(tmp_path / "m.yaml") == []
    assert load_models(tmp_path / "blank.yaml") == []
    assert load_rejected(tmp_path / "absent.yaml") == []


def test_known_keys_union():
    m = RegistryEntry(id="mdlm-owt", name="MDLM", paper=Paper(arxiv="2406.07524"), github="https://github.com/kuleshov-group/mdlm",
                      checkpoint=Checkpoint(kind="hf", ref="kuleshov-group/mdlm-owt"), family="masked-dlm", train_data="owt",
                      status="queued", added=date(2026, 10, 8), source="seed")
    r = RejectedEntry(key="arxiv:2502.11564", name="RDLM", reason="LM1B only", decided=date(2026, 10, 8), source="seed")
    assert known_keys([m], [r]) == {"arxiv:2406.07524", "github:kuleshov-group/mdlm", "hf:kuleshov-group/mdlm-owt", "arxiv:2502.11564"}


def test_render_yaml_block_appends_cleanly(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text("# header comment\n- id: a\n  name: A\n")
    block = render_yaml_block([{"id": "b", "name": "B", "added": date(2026, 10, 8)}])
    path.write_text(path.read_text() + block)
    text = path.read_text()
    assert text.startswith("# header comment")
    assert yaml.safe_load(text) == [{"id": "a", "name": "A"}, {"id": "b", "name": "B", "added": date(2026, 10, 8)}]
