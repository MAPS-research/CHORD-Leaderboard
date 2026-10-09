"""CLI: python -m discovery {run,apply-decision} ..."""

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from discovery.config import load_config
from discovery.http import github_headers, make_client
from discovery.registry import REPO_ROOT


def cmd_run(args) -> int:
    import shutil

    from huggingface_hub import HfApi

    from discovery.github import GitHubClient
    from discovery.judge import run_claude
    from discovery.pipeline import Deps, RunSummary, notify_repeated, render_summary, repeated_failures, run

    cfg = load_config(args.config)
    if args.max_judge_calls is not None:
        cfg = cfg.model_copy(update={"max_judge_calls": args.max_judge_calls})
    since = args.since or date.today() - timedelta(days=cfg.lookback_days)
    if shutil.which("claude") is None:
        print("the claude CLI is not installed (npm install -g @anthropic-ai/claude-code)", file=sys.stderr)
        return 2
    github = GitHubClient(make_client(github_headers()), cfg.repo) if os.environ.get("GITHUB_TOKEN") else None
    if github is None:
        print("warning: GITHUB_TOKEN is not set; GitHub allows 60 requests/hour, so README and repo checks will hit rate limits",
              file=sys.stderr)
    if github is None and not args.dry_run:
        print("GITHUB_TOKEN is required unless --dry-run is given", file=sys.stderr)
        return 2
    deps = Deps(http=make_client(), hf=HfApi(), llm=run_claude, github=github)
    prev = None
    if args.prev_summary and Path(args.prev_summary).exists():
        prev = RunSummary.model_validate_json(Path(args.prev_summary).read_text(encoding="utf-8"))
    seen = prev.judged_on if prev and not args.no_dedupe else None
    summary, judged = run(cfg, deps, root=args.root, since=since, dry_run=args.dry_run, no_dedupe=args.no_dedupe, seen=seen)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for c in judged:
                row = {"reported": c.key in summary.reported, "digest": c.key in summary.digested,
                       "candidate": c.model_dump(mode="json", exclude={"readme"})}
                fh.write(json.dumps(row) + "\n")
    repeated = repeated_failures(prev, summary)
    if args.summary_out:
        Path(args.summary_out).write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    md = render_summary(summary, repeated)
    if repeated and not args.dry_run and (failure := notify_repeated(github, md)):
        md += f"\n\nCould not update the pipeline-failure issue: {failure}"
    print(md)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    return 1 if summary.all_sources_failed else 0


def cmd_apply_decision(args) -> int:
    from discovery.apply_decision import apply

    event = json.loads(args.event.read_text(encoding="utf-8"))
    try:
        body = apply(args.action, event, args.root, args.today or date.today())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    args.summary_out.write_text(body, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m discovery")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run one discovery pass")
    r.add_argument("--config", type=Path, default=None)
    r.add_argument("--root", type=Path, default=REPO_ROOT)
    r.add_argument("--since", type=date.fromisoformat, default=None)
    r.add_argument("--dry-run", action="store_true", help="open no issues")
    r.add_argument("--no-dedupe", action="store_true", help="keep candidates already in the registry (backtests)")
    r.add_argument("--out", type=Path, default=None, help="write judged candidates as JSON lines")
    r.add_argument("--summary-out", type=Path, default=None)
    r.add_argument("--prev-summary", type=Path, default=None)
    r.add_argument("--max-judge-calls", type=int, default=None, help="override config (0 = harvest and check only)")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("apply-decision", help="apply an accept/reject label to the registry")
    a.add_argument("--action", choices=["accept", "reject"], required=True)
    a.add_argument("--event", type=Path, required=True, help="GitHub event payload ($GITHUB_EVENT_PATH)")
    a.add_argument("--root", type=Path, default=REPO_ROOT)
    a.add_argument("--today", type=date.fromisoformat, default=None)
    a.add_argument("--summary-out", type=Path, required=True, help="where to write the comment posted on the issue")
    a.set_defaults(func=cmd_apply_decision)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
