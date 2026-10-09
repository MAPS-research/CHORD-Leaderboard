from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_model_request_form_asks_for_what_listing_needs():
    form = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / "model-request.yml").read_text())
    assert form["labels"] == ["model-request"]
    fields = {b["id"]: b for b in form["body"] if "id" in b}
    assert {"model", "paper", "code", "checkpoint", "owt", "sampling"} <= set(fields)
    assert all(fields[k]["validations"]["required"] for k in ("model", "code", "checkpoint"))


def test_site_links_to_the_request_form():
    html = (ROOT / "site" / "index.html").read_text()
    assert "https://github.com/MAPS-research/CHORD-Leaderboard/issues/new?template=model-request.yml" in html
