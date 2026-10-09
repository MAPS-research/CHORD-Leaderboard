"""Decide what to report and render/parse candidate issues."""

import re

from pydantic import ValidationError

from discovery.models import Candidate, Judgement

MARKER_RE = re.compile(r"<!-- cand: (\S+) -->")
BLOCK_RE = re.compile(r"```json candidate\n(.*?)\n```", re.DOTALL)
TITLE_MAX = 250
QUOTE_MAX = 500
ABSTRACT_MAX = 4000
NOTES_MAX = 1000
TABLE_WEIGHTS_MAX = 50
JSON_WEIGHTS_MAX = 100  # keeps the body far below GitHub's 65,536-character limit
_PLAUSIBLE = ("ok", "unverifiable")
TYPE_LABEL = {"ar": "AR", "discrete": "Discrete diffusion/flow", "continuous": "Continuous diffusion/flow", "unclear": "unclear"}


DIGEST_ROWS_MAX = 300
DIGEST_NOTE_MAX = 300
DIGEST_BUDGET = 50000  # characters for all rows; notes shrink so every row fits


def classify(c: Candidate) -> str | None:
    """"issue" for a confirmed OWT candidate, "digest" for one a human should glance at, None to drop."""
    if c.verdict is None:
        return "digest"  # judge failed: listed for a human
    if c.verdict.owt_trained.value == "no":
        return None
    if not (any(w.check in _PLAUSIBLE for w in c.weights) or c.verdict.official_checkpoints):
        return None
    return "issue" if c.verdict.owt_trained.value == "yes" else "digest"


def _cell(text: str, limit: int) -> str:
    return " ".join(text.split()).replace("|", "\\|")[:limit]


def render_digest(cands: list[Candidate], run_date) -> tuple[str, str, list[str]]:
    rows = []
    note_max = max(40, min(DIGEST_NOTE_MAX, DIGEST_BUDGET // max(len(cands), 1) - 160))
    for c in cands[:DIGEST_ROWS_MAX]:
        if c.arxiv_id:
            link = f"[{_cell(c.title or c.arxiv_id, 90)}](https://arxiv.org/abs/{c.arxiv_id})"
        else:
            ref = next(w.ref for w in c.weights if w.kind == "hf")
            link = f"[{ref}](https://huggingface.co/{ref})"
        ok = sum(w.check == "ok" for w in c.weights)
        note = c.judge_error if c.verdict is None else c.verdict.notes
        rows.append(f"| {link} | {ok}/{len(c.weights)} | {_cell(note or '-', note_max)} |")
    if len(cands) > DIGEST_ROWS_MAX:
        rows.append(f"| ... | | {len(cands) - DIGEST_ROWS_MAX} more not shown |")
    body = "\n".join([
        "Candidates found this run whose OpenWebText training could not be confirmed from the abstract, README or model card.",
        "Skim for anything that belongs on the leaderboard; add it with a PR to `registry/models.yaml`.",
        "",
        "| candidate | HF weights ok / links | judge notes |",
        "|---|---|---|",
        *rows,
    ])
    return f"[digest] {len(cands)} unclear candidates (run of {run_date})", body, ["candidate-digest"]


def needs_manual_check(c: Candidate) -> bool:
    if c.verdict is None:
        return True
    unclear = "unclear" in (c.verdict.owt_trained.value, c.verdict.unconditional.value)
    return unclear or any(w.check == "unverifiable" for w in c.weights)


def _normalise(body: str | None) -> str:
    return (body or "").replace("\r\n", "\n")  # bodies edited in the GitHub web UI come back with CRLF


def _short(j: Judgement) -> Judgement:
    return j.model_copy(update={"evidence": [e.model_copy(update={"quote": e.quote[:QUOTE_MAX]}) for e in j.evidence]})


def _compact(c: Candidate) -> Candidate:
    """The candidate as stored in the issue: no README, bounded abstract, quotes and weight list."""
    weights = sorted(c.weights, key=lambda w: w.check != "ok")[:JSON_WEIGHTS_MAX]
    v = c.verdict
    if v is not None:
        ckpts = [k.model_copy(update={"evidence": [e.model_copy(update={"quote": e.quote[:QUOTE_MAX]}) for e in k.evidence]})
                 for k in v.official_checkpoints]
        v = v.model_copy(update={"owt_trained": _short(v.owt_trained), "unconditional": _short(v.unconditional),
                                 "official_checkpoints": ckpts, "notes": v.notes[:NOTES_MAX]})
    return c.model_copy(update={"readme": "", "abstract": c.abstract[:ABSTRACT_MAX], "weights": weights, "verdict": v})


def _title(c: Candidate) -> str:
    if c.arxiv_id:
        text = f"[candidate] {c.title or 'Untitled'} (arXiv:{c.arxiv_id})"
    else:
        text = f"[candidate] {next(w.ref for w in c.weights if w.kind == 'hf')} (HF Hub, no paper)"
    return text[:TITLE_MAX]


def _judgement_md(label: str, j, value: str | None = None) -> str:
    lines = [f"- **{label}:** {value or j.value}"]
    lines += [f'  > "{e.quote[:QUOTE_MAX]}" ([source]({e.url}))' for e in j.evidence]
    return "\n".join(lines)


def _verdict_md(c: Candidate) -> str:
    if c.verdict is None:
        return f"> **Automatic verdict failed: {c.judge_error}**. Please check the paper and README manually."
    v = c.verdict
    ckpts = "\n".join(f"  - {k.kind} `{k.ref}`" for k in v.official_checkpoints) or "  - none identified"
    return "\n".join([
        _judgement_md("Trained on OpenWebText", v.owt_trained),
        _judgement_md("Unconditional generation", v.unconditional),
        _judgement_md("Generation type", v.generation_type, TYPE_LABEL[v.generation_type.value]),
        f"- **Official checkpoints:**\n{ckpts}",
        f"- **Family guess:** {v.family_guess or 'unknown'}",
        f"- **Notes:** {v.notes or '-'}",
    ])


def render_issue(c: Candidate) -> tuple[str, str, list[str]]:
    paper = (f"[{c.title}](https://arxiv.org/abs/{c.arxiv_id})" + (f", published {c.published}" if c.published else "")
             if c.arxiv_id else "none (found on the HF Hub)")
    official = c.verdict.official_repo if c.verdict and c.verdict.official_repo else "not identified"
    repos = ", ".join(c.repos) or "none found"
    shown = sorted(c.weights, key=lambda w: w.check != "ok")[:TABLE_WEIGHTS_MAX]
    weights = "\n".join(f"| {w.kind} | `{w.ref}` | {w.check} |" for w in shown) or "| - | none found | - |"
    if len(c.weights) > len(shown):
        weights += f"\n| ... | {len(c.weights) - len(shown)} more not shown | |"
    body = "\n".join([
        f"**Paper:** {paper}",
        f"**Found via:** {', '.join(c.sources)}",
        f"**Official repository:** {official}",
        f"**All repositories found:** {repos}",
        "",
        "### Weights",
        "| kind | ref | check |",
        "|---|---|---|",
        weights,
        "",
        "### Verdict",
        _verdict_md(c),
        "",
        "### Review",
        "Label `accept` to open a registry PR, or comment `reason: <why>` and label `reject`.",
        "",
        *(f"<!-- cand: {k} -->" for k in sorted(c.keys())),
        "```json candidate",
        _compact(c).model_dump_json(exclude={"readme"}),
        "```",
    ])
    labels = ["candidate"] + (["needs-manual-check"] if needs_manual_check(c) else [])
    return _title(c), body, labels


def parse_candidate(body: str) -> Candidate:
    m = BLOCK_RE.search(_normalise(body))
    if not m:
        raise ValueError("The issue body has no ```json candidate block. Restore it from the issue's edit history, then re-apply the label.")
    try:
        return Candidate.model_validate_json(m.group(1))
    except ValidationError as exc:
        raise ValueError(f"The ```json candidate block could not be parsed: {exc}") from exc


def issue_keys(body: str) -> set[str]:
    return set(MARKER_RE.findall(_normalise(body)))
