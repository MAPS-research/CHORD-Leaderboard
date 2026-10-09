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


def test_apply_decision_commits_to_main_and_closes_the_issue():
    wf = _load("apply-decision.yml")
    assert wf[True] == {"issues": {"types": ["labeled"]}}
    assert wf["permissions"] == {"contents": "write", "issues": "write"}  # the organization forbids Actions from opening PRs
    assert wf["concurrency"] == {"group": "apply-decision", "cancel-in-progress": False}
    scripts = "\n".join(_run_blocks(wf))
    assert "gh pr create" not in scripts
    assert 'git push origin "HEAD:$BASE"' in scripts
    assert "gh issue close" in scripts and "--summary-out summary.md" in scripts


def test_no_untrusted_expressions_inside_run_scripts():
    for name in ("discover.yml", "apply-decision.yml", "pages.yml"):
        for script in _run_blocks(_load(name)):
            assert not UNTRUSTED.search(script), f"{name}: pass untrusted values through env, not ${{{{ }}}} in run"


def test_pages_builds_and_deploys_on_data_or_site_changes():
    wf = _load("pages.yml")
    push = wf[True]["push"]
    assert push["branches"] == ["main"]
    assert {"registry/**", "results/**", "site/**", "leaderboard_site/**"} <= set(push["paths"])
    assert "workflow_dispatch" in wf[True]
    assert wf["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    steps = wf["jobs"]["deploy"]["steps"]
    runs = [s.get("run", "") for s in steps]
    uses = [s.get("uses", "") for s in steps]
    assert any("python -m leaderboard_site.build" in r for r in runs)
    assert any(u.startswith("actions/upload-pages-artifact@") for u in uses)
    assert any(u.startswith("actions/deploy-pages@") for u in uses)
