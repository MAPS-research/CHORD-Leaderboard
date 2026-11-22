import html
import re

import yaml

from discovery.registry import REPO_ROOT, load_models

SITE = REPO_ROOT / "site"
CHANGELOG = SITE / "changelog" / "index.html"


def _text(path) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", path.read_text()))


def test_main_page_links_to_the_changelog():
    assert 'href="changelog/"' in (SITE / "index.html").read_text()


def test_changelog_links_back_to_the_leaderboard():
    assert 'href="../"' in CHANGELOG.read_text()


def test_changelog_mentions_every_scored_generator():
    text = _text(CHANGELOG)
    scored = [e.name for e in load_models(REPO_ROOT / "registry" / "models.yaml") if e.status == "scored"]
    continuation = [e["name"] for e in yaml.safe_load((REPO_ROOT / "registry" / "continuation.yaml").read_text())]
    missing = [name for name in scored + continuation if name not in text]
    assert not missing, f"add these generators to site/changelog/index.html: {missing}"


def test_changelog_entries_are_newest_first_and_listed_in_the_sidebar():
    page = CHANGELOG.read_text()
    entries = re.findall(r'<h2 id="(\d{4}-\d{2}-\d{2})">', page)
    assert entries and entries == sorted(entries, reverse=True)
    sidebar = re.findall(r'<a href="#(\d{4}-\d{2}-\d{2})"', page)
    assert sidebar == entries


def test_pages_stamps_the_changelog_assets_too():
    workflow = (REPO_ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "site/changelog/index.html" in workflow and r"changelog\.css" in workflow


def _nav(path) -> list[tuple[str, str]]:
    header = re.search(r'<header class="site-header">(.*?)</header>', path.read_text(), re.S)
    assert header, f"{path.name} has no site header"
    items = re.findall(r'<(a|span)\b([^>]*)>(?:<svg.*?</svg>)?([^<]+)</\1>', header.group(1), re.S)
    return [("current" if "aria-current" in attrs else "link", text.strip()) for _, attrs, text in items]


def test_both_pages_share_the_site_header():
    main, changelog = _nav(SITE / "index.html"), _nav(CHANGELOG)
    labels = ["CHORD Leaderboard", "Leaderboard", "Changelog", "Code"]
    assert [text for _, text in main] == labels and [text for _, text in changelog] == labels
    assert ("current", "Leaderboard") in main and ("link", "Changelog") in main
    assert ("link", "Leaderboard") in changelog and ("current", "Changelog") in changelog


def test_main_page_has_no_checkpoints_section():
    assert "ckpt" not in (SITE / "index.html").read_text() and "ckpt" not in (SITE / "app.js").read_text()
