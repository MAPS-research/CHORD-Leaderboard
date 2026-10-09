"""Turn an accept/reject label on a candidate issue into registry edits and a PR body."""

import re
from datetime import date
from pathlib import Path

from discovery.keys import normalize_github
from discovery.models import Candidate, Checkpoint, Paper, RegistryEntry, RejectedEntry
from discovery.registry import load_models, load_rejected, render_yaml_block
from discovery.report import parse_candidate

SAMPLER_TODO = "sampler: null  # TODO(sampler): fill from paper"
FAMILY_TODO = "  # TODO(review): family unknown, set before merging"
DEFAULT_FAMILY = "masked-dlm"
_REASON = re.compile(r"^\s*reason:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _unique(base: str, taken: set[str]) -> str:
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def _checkpoint_refs(cand: Candidate) -> list[tuple[str, str]]:
    if cand.verdict and cand.verdict.official_checkpoints:
        return [(c.kind, c.ref) for c in cand.verdict.official_checkpoints]
    return [(w.kind, w.ref) for w in cand.weights if w.check == "ok"]


def accept_entries(cand: Candidate, issue_number: int, today: date, taken: set[str]) -> list[RegistryEntry]:
    refs = _checkpoint_refs(cand)
    if not refs:
        raise ValueError("The verdict lists no checkpoint and no weight passed the checks. Add the checkpoint to the issue's candidate block or to the registry by hand.")
    repo = (cand.verdict.official_repo if cand.verdict and cand.verdict.official_repo else None) or (cand.repos[0] if cand.repos else None)
    if not repo:
        raise ValueError("No GitHub repository was identified for this candidate.")
    repo_name = (normalize_github(repo) or repo).split("/")[-1]
    family = (cand.verdict.family_guess if cand.verdict else None) or DEFAULT_FAMILY
    paper = Paper(arxiv=cand.arxiv_id, title=cand.title or None) if cand.arxiv_id else None
    taken = set(taken)
    out = []
    for kind, ref in refs:
        base = _slug(ref.rstrip("/").split("/")[-1]) if kind == "hf" else _slug(f"{repo_name}-{kind}")
        entry_id = _unique(base, taken)
        taken.add(entry_id)
        out.append(RegistryEntry(id=entry_id, name=repo_name, paper=paper, github=repo, checkpoint=Checkpoint(kind=kind, ref=ref),
                                 family=family, train_data="owt", status="queued", added=today, source=f"discovery#{issue_number}"))
    return out


def reject_entry(cand: Candidate, issue_number: int, today: date, comments: list[dict]) -> RejectedEntry:
    reasons = [m.group(1).strip() for c in comments if (m := _REASON.match(c.get("body") or ""))]
    return RejectedEntry(key=cand.key, name=cand.title or cand.key, reason=reasons[-1] if reasons else "rejected by maintainer",
                         decided=today, source=f"discovery#{issue_number}")


def _append(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.rstrip("\n").splitlines()
    if lines and lines[-1].strip() == "[]":  # empty list placeholder
        lines = lines[:-1]
    head = "\n".join(lines).rstrip("\n")
    path.write_text((head + "\n" if head else "") + block.lstrip("\n"), encoding="utf-8")


def _entry_yaml(entry: RegistryEntry, family_known: bool) -> str:
    text = render_yaml_block([entry.model_dump(mode="json")]).replace("sampler: null", SAMPLER_TODO)
    if not family_known:
        text = text.replace(f"family: {entry.family}", f"family: {entry.family}{FAMILY_TODO}", 1)
    return text


def apply(action: str, event: dict, comments: list[dict], root: Path, today: date) -> str:
    issue = event["issue"]
    number = issue["number"]
    if "candidate" not in {label["name"] for label in issue.get("labels", [])}:
        raise ValueError(f"Issue #{number} is not a candidate issue.")
    cand = parse_candidate(issue.get("body") or "")
    reg = root / "registry"
    if action == "accept":
        path = reg / "models.yaml"
        entries = accept_entries(cand, number, today, taken={m.id for m in load_models(path)})
        family_known = bool(cand.verdict and cand.verdict.family_guess)
        backup = path.read_text(encoding="utf-8")
        _append(path, "".join(_entry_yaml(e, family_known) for e in entries))
        try:
            load_models(path)
        except Exception:
            path.write_text(backup, encoding="utf-8")
            raise
        listed = "\n".join(f"- `{e.id}`: {e.checkpoint.kind} `{e.checkpoint.ref}`" for e in entries)
        return (f"Closes #{number}\n\nAdds {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} to `registry/models.yaml`:\n{listed}\n\n"
                "Before merging:\n- [ ] check `id`, `name`, `family`, `params`, `train_data`, `tokenizer`\n"
                "- [ ] keep `status: queued`; `sampler` is filled by the sampling sub-project")
    if action == "reject":
        path = reg / "rejected.yaml"
        entry = reject_entry(cand, number, today, comments)
        backup = path.read_text(encoding="utf-8") if path.exists() else ""
        _append(path, render_yaml_block([entry.model_dump(mode="json")]))
        try:
            load_rejected(path)
        except Exception:
            path.write_text(backup, encoding="utf-8")
            raise
        return f"Closes #{number}\n\nRecords `{entry.key}` in `registry/rejected.yaml`.\n\nReason: {entry.reason}"
    raise ValueError(f"Unknown action {action!r}; expected accept or reject.")
