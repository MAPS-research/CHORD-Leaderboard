"""One discovery pass: harvest -> dedupe -> metadata -> prefilter -> enrich/check -> judge -> dedupe -> report."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from discovery import checks, enrich, judge, report
from discovery.config import Config
from discovery.dedupe import dedupe_within_run, filter_new
from discovery.models import Candidate
from discovery.registry import known_keys, load_models, load_rejected
from discovery.sources import arxiv, hf_hub

MIN_JUDGED_FOR_OUTAGE = 3


@dataclass(frozen=True)
class Deps:
    http: Any
    hf: Any
    llm: Any
    github: Any | None
    sleep: Callable[[float], None] = time.sleep


class RunSummary(BaseModel):
    since: date
    harvested: dict[str, int] = {}
    errors: dict[str, str] = {}
    warnings: list[str] = []
    new: int = 0
    after_prefilter: int = 0
    judged: int = 0
    deferred: int = 0
    reported: list[str] = []
    opened: list[int] = []
    digested: list[str] = []
    digest_issue: int | None = None
    skipped_seen: int = 0
    judged_on: dict[str, date] = {}  # key -> date judged with a verdict; later runs skip these keys
    all_sources_failed: bool = False


def _err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _harvest(cfg: Config, deps: Deps, since: date):
    jobs = {
        "arxiv": lambda: arxiv.search(deps.http, cfg.arxiv_categories, cfg.arxiv_query_terms, since, sleep=deps.sleep),
        "hf_hub": lambda: hf_hub.search(deps.hf, cfg.hf_queries, since),
    }
    merged: dict[str, Candidate] = {}
    counts, errors = {}, {}
    for name, job in jobs.items():
        try:
            found = job()
        except Exception as exc:  # source isolation: record and continue
            errors[name] = _err(exc)
            continue
        counts[name] = len(found)
        for c in found:
            merged[c.key] = merged[c.key].merge(c) if c.key in merged else c
    return list(merged.values()), counts, errors


def _fill_metadata(deps: Deps, cands: list[Candidate]) -> list[Candidate]:
    need = [c.arxiv_id for c in cands if c.arxiv_id and "arxiv-kw" not in c.sources]
    meta = arxiv.fetch_by_ids(deps.http, need, sleep=deps.sleep) if need else {}
    return [c.merge(meta[c.arxiv_id]) if c.arxiv_id in meta else c for c in cands]


def passes_prefilter(c: Candidate, terms: list[str]) -> bool:
    if any(s.startswith("hf-search:") for s in c.sources):
        return True
    text = f"{c.title} {c.abstract}".lower()
    return any(t.lower() in text for t in terms)


def _enrich_and_check(cfg: Config, deps: Deps, cands: list[Candidate]):
    out, warnings = [], []
    for c in cands:
        try:
            c = checks.run_checks(deps.hf, deps.http, enrich.enrich(deps.http, c, cfg.max_readme_chars, sleep=deps.sleep), sleep=deps.sleep)
        except Exception as exc:  # keep the candidate with what we have; the reviewer sees the warning
            warnings.append(f"enrich/check {c.key}: {_err(exc)}")
        out.append(c)
    return out, warnings


def _judge_all(cfg: Config, deps: Deps, cands: list[Candidate]):
    ordered = sorted(cands, key=lambda c: c.published or date.min, reverse=True)
    batch, deferred = ordered[: cfg.max_judge_calls], ordered[cfg.max_judge_calls :]
    def one(c: Candidate) -> Candidate:  # judge.judge never raises; failures come back as judge_error
        return judge.judge(deps.llm, cfg.judge_model, c, sleep=deps.sleep)

    if cfg.judge_workers <= 1:
        judged = [one(c) for c in batch]
    else:
        with ThreadPoolExecutor(max_workers=cfg.judge_workers) as pool:
            judged = list(pool.map(one, batch))  # map keeps the newest-first order
    outage = len(judged) >= MIN_JUDGED_FOR_OUTAGE and all(c.verdict is None for c in judged)
    return judged, len(deferred), outage


def run(cfg: Config, deps: Deps, *, root: Path, since: date, dry_run: bool, no_dedupe: bool,
        seen: dict[str, date] | None = None, today: date | None = None):
    reg = root / "registry"
    today = today or date.today()
    seen = {k: d for k, d in (seen or {}).items() if d >= since}  # entries older than the window cannot recur
    cands, counts, errors = _harvest(cfg, deps, since)

    known, exempt, can_report = frozenset(), frozenset(), not dry_run
    if not no_dedupe:
        rejected = load_rejected(reg / "rejected.yaml")
        known = known_keys(load_models(reg / "models.yaml"), rejected)
        exempt = frozenset(r.key for r in rejected if r.key.startswith("hf:"))  # HF-only rejections wait for a paper
        if deps.github is not None:
            try:
                known = known | deps.github.candidate_issue_keys()
            except Exception as exc:  # without issue markers we could open duplicates, so report nothing
                errors["github_issues"] = _err(exc)
                can_report = False

    fresh = filter_new(cands, known, judged=False)
    skipped_seen = sum(c.key in seen for c in fresh)
    fresh = [c for c in fresh if c.key not in seen]
    try:
        fresh = _fill_metadata(deps, fresh)
    except Exception as exc:
        errors["arxiv_metadata"] = _err(exc)
    relevant = [c for c in fresh if passes_prefilter(c, cfg.prefilter_terms)]
    checked, warnings = _enrich_and_check(cfg, deps, relevant)
    judged, deferred, outage = _judge_all(cfg, deps, checked)
    judged_on = {**seen, **{c.key: today for c in judged if c.verdict is not None}}  # failures are retried next run
    if outage:
        errors["judge"] = judged[0].judge_error or "all judge calls failed"
        judged = [c for c in judged if c.verdict is not None]
    judged = dedupe_within_run(filter_new(judged, known, judged=True, paper_exempt=exempt))
    to_report = [c for c in judged if report.classify(c) == "issue"]
    to_digest = [c for c in judged if report.classify(c) == "digest"]

    opened, create_errors = [], []
    if can_report:
        for c in to_report:
            try:
                opened.append(deps.github.create_issue(*report.render_issue(c), assignees=cfg.assignees))
            except Exception as exc:  # one bad issue must not cost the rest of the run
                create_errors.append(f"{c.key}: {_err(exc)}")
    if create_errors:
        errors["github_create"] = "; ".join(create_errors)
    digest_issue = None
    if can_report and to_digest:
        try:
            digest_issue = deps.github.create_issue(*report.render_digest(to_digest, today), assignees=cfg.assignees)
        except Exception as exc:
            errors["github_digest"] = _err(exc)
    summary = RunSummary(since=since, harvested=counts, errors=errors, warnings=warnings, new=len(fresh),
                         after_prefilter=len(relevant), judged=len(judged), deferred=deferred,
                         reported=[c.key for c in to_report], opened=opened, all_sources_failed=not counts,
                         digested=[c.key for c in to_digest], digest_issue=digest_issue,
                         skipped_seen=skipped_seen, judged_on=judged_on)
    return summary, judged


def notify_repeated(github, md: str) -> str | None:
    """Open or update the pipeline-failure issue; return an error message instead of raising."""
    if github is None:
        return None
    try:
        github.upsert_failure_issue(md)
    except Exception as exc:
        return _err(exc)
    return None


def repeated_failures(prev: RunSummary | None, cur: RunSummary) -> list[str]:
    return sorted(set(prev.errors) & set(cur.errors)) if prev else []


def render_summary(s: RunSummary, repeated: list[str]) -> str:
    lines = [f"## Discovery run (since {s.since})", "", "| source | harvested |", "|---|---|"]
    lines += [f"| {name} | {n} |" for name, n in s.harvested.items()]
    lines += ["", f"- new (not yet known): {s.new}", f"- after prefilter: {s.after_prefilter}",
              f"- judged: {s.judged} (deferred by cap: {s.deferred})", f"- reported: {len(s.reported)}",
              f"- issues opened: {', '.join(f'#{n}' for n in s.opened) or 'none'}",
              f"- digest: {len(s.digested)} unclear candidates" + (f" in #{s.digest_issue}" if s.digest_issue else ""),
              f"- skipped (judged in an earlier run): {s.skipped_seen}"]
    if s.errors:
        lines += ["", "### Errors"] + [f"- **{k}**: {v}" for k, v in s.errors.items()]
    if repeated:
        lines += ["", f"Failing in consecutive runs: {', '.join(repeated)}"]
    if s.warnings:
        lines += ["", "### Warnings"] + [f"- {w}" for w in s.warnings]
    return "\n".join(lines)
