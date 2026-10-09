import re
from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"
UNTRUSTED = re.compile(r"\$\{\{\s*(inputs\.|github\.event\.(issue|comment|label)\.(title|body))")


def _load(name):
    return yaml.safe_load((WF / name).read_text())


def _run_blocks(wf):
    for job in wf["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                yield step["run"]


def test_discover_schedule_and_permissions():
    wf = _load("discover.yml")
    triggers = wf[True]  # PyYAML parses the `on` key as boolean True
    assert triggers["schedule"] == [{"cron": "0 14 * * 1"}]
    assert set(triggers["workflow_dispatch"]["inputs"]) == {"since", "dry_run"}
    assert wf["permissions"] == {"contents": "read", "issues": "write", "actions": "read"}


def test_apply_decision_trigger_and_permissions():
    wf = _load("apply-decision.yml")
    assert wf[True] == {"issues": {"types": ["labeled"]}}
    assert wf["permissions"] == {"contents": "write", "issues": "write", "pull-requests": "write"}


def test_no_untrusted_expressions_inside_run_scripts():
    for name in ("discover.yml", "apply-decision.yml"):
        for script in _run_blocks(_load(name)):
            assert not UNTRUSTED.search(script), f"{name}: pass untrusted values through env, not ${{{{ }}}} in run"
