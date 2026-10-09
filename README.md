# CHORD Leaderboard

Scores open-ended text generators with [CHORD](https://arxiv.org/abs/2609.34240) under the paper's Table 2 protocol.

This repository currently holds the **discovery pipeline**: a weekly job that finds new OpenWebText-trained
models with public weights and an official repository, and opens one review issue per paper.

## How it works

1. `.github/workflows/discover.yml` runs every Monday. It harvests arXiv keyword hits (cs.CL/cs.LG,
   see `discovery/config.yaml`) and new OWT models on the Hugging Face Hub.
2. Each candidate gets its repository, README and weight links, deterministic checks (public HF weights,
   live GitHub repo) and a Claude verdict (via the Claude Code CLI) backed by verbatim quotes.
3. New candidates become issues labelled `candidate` (plus `needs-manual-check` when something could not be verified).
4. Label an issue `accept`, or comment `reason: <why>` and label it `reject`. `apply-decision.yml` opens a PR
   that updates `registry/models.yaml` or `registry/rejected.yaml`; merging it closes the issue.

## Registry

- `registry/models.yaml`: one entry per checkpoint; `status: scored | queued | excluded`.
- `registry/rejected.yaml`: rejected papers/models and why.

## Running locally

```bash
conda create -y -n chord-leaderboard python=3.11 && conda activate chord-leaderboard
pip install -e ".[test]"
pytest                                   # offline tests
python -m discovery run --dry-run --out candidates.jsonl   # needs a logged-in `claude` CLI
```

## Setup (once)

- Actions secret: `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`.
- Settings → Actions → General: allow GitHub Actions to create pull requests.
