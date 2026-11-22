"""Pieces shared by both leaderboard boards (unconditional in build.py, prefix continuation in continuation.py)."""

import re
import statistics

from discovery.models import Checkpoint, Paper

HUMAN_ID = "human-heldout"
ENCODER = "qwen35-27b-prompteol-coherence-l62"
METRIC = "rbf_mmd2_x100_biased"
TOLERANCE = 1e-3
_PARAMS = re.compile(r"^(\d+(?:\.\d+)?)([MB])$")
_HOST = {"gdrive": "Google Drive", "dropbox": "Dropbox", "zenodo": "Zenodo", "box": "Box"}


class BuildError(RuntimeError):
    pass


def parse_params(text: str) -> int:
    m = _PARAMS.match(text.strip())
    if not m:
        raise BuildError(f"cannot parse params {text!r}; use a number followed by M or B, e.g. 355M or 1.5B")
    return round(float(m.group(1)) * (1e6 if m.group(2) == "M" else 1e9))


def check_folds(name: str, folds: list[float], mean: float, std: float) -> None:
    """Fail unless there are at least two folds and `mean`/`std` (ddof=1) match them."""
    if len(folds) < 2:
        raise BuildError(f"{name}: needs at least two folds")
    actual_mean, actual_std = statistics.mean(folds), statistics.stdev(folds)
    if abs(actual_mean - mean) > TOLERANCE or abs(actual_std - std) > TOLERANCE:
        raise BuildError(f"{name}: mean/std ({mean}, {std}) do not match its folds ({actual_mean:.4f}, {actual_std:.4f})")


def paper_link(paper: Paper | None) -> str | None:
    if not paper:
        return None
    return f"https://arxiv.org/abs/{paper.arxiv}" if paper.arxiv else paper.url


def checkpoint_link(c: Checkpoint) -> dict:
    """Label and URL of the evaluated checkpoint, shown in the page's Checkpoints section."""
    if c.kind == "hf":
        url = f"https://huggingface.co/{c.ref}" + (f"/blob/main/{c.path}" if c.path else "")
        return {"label": c.ref + (f"/{c.path}" if c.path else ""), "url": url}
    if c.kind == "github-release":
        return {"label": f"GitHub release {c.ref.rstrip('/').rsplit('/', 1)[-1]}", "url": c.ref}
    host = _HOST.get(c.kind, c.kind)
    on_readme = "github.com" in c.ref
    label = f"{c.path} ({host}{', linked from the README' if on_readme else ''})" if c.path else (
        f"{host}, linked from the README" if on_readme else host)
    return {"label": label, "url": c.ref}
