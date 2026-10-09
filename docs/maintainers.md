# Maintaining the CHORD Leaderboard

## How it works

1. **Discovery.** `.github/workflows/discover.yml` runs every Monday. It collects new arXiv papers matching
   the keywords in `discovery/config.yaml` (cs.CL/cs.LG) and new OpenWebText models on the Hugging Face Hub,
   then gathers each candidate's repository, README (or HF model card) and weight links, checks that the
   weights are public and the repository is live, and asks Claude (through the Claude Code CLI) whether the
   model was trained on OpenWebText, can generate unconditionally, and which of the three types it is. Every
   answer must quote the abstract or README word for word.
2. **Review.** A candidate confirmed as OpenWebText-trained, with an official GitHub repository, gets its own
   issue labelled `candidate`. Everything less certain goes into one `candidate-digest` issue per run.
   Candidates judged in an earlier run are not judged again.
3. **Decision.** Label a candidate issue `accept` or `reject`.
   `apply-decision.yml` writes the change to `registry/models.yaml` or `registry/rejected.yaml`, commits it
   to `main`, comments on the issue with what changed, and closes it. To fix a field afterwards, edit the
   registry directly. Duplicates and digests can simply be closed.
4. **Scoring.** Models are sampled and scored on a GPU cluster with jobs kept in a separate private repository:
   10 seeds × 500 samples, embedded with the 27B CHORD encoder and scored fold by fold against the same human
   reference as the paper. `scoring/export_results.py` turns the fold scores into `results/<id>.json`.
5. **Website.** When `registry/`, `results/` or `site/` change on `main`, `.github/workflows/pages.yml` runs
   `python -m leaderboard_site.build` to produce `site/data/leaderboard.json` and publishes `site/` to
   GitHub Pages.

## Requesting a model

Anyone can ask for a generator to be evaluated (or evaluated sooner) with the [model evaluation request](https://github.com/MAPS-research/CHORD-Leaderboard/issues/new?template=model-request.yml) form. Requests are labelled `model-request` and reviewed by hand; they do not trigger `apply-decision`.

## Registry

- `registry/models.yaml`: one entry per checkpoint, with `status: queued | scored | excluded`.
  An entry shown on the website (`scored`) also needs `group` (`ar`, `discrete` or `continuous`), `params`
  (for example `355M`), `paper.published` (first arXiv version) and either `paper.arxiv` or `paper.url`,
  plus a matching file in `results/`. The build fails if a scored entry has no results or a results file
  has no scored entry.
- `registry/rejected.yaml`: rejected papers and models.
- `results/<id>.json`: fold scores for a scored entry; `results/human-heldout.json` is the held-out human
  reference row. Mean and standard deviation are checked against the folds at build time.

To add a scored model: sample and score it with the cluster jobs, run `scoring/export_results.py`, copy
the new `results/*.json` here, set the entry's `status: scored` and site fields, and push.

## Running locally

```bash
conda create -y -n chord-leaderboard python=3.11 && conda activate chord-leaderboard
pip install -e ".[test]"
python -m pytest                                            # offline tests
python -m leaderboard_site.build                            # writes site/data/leaderboard.json
python -m http.server -d site 8000                          # then open http://localhost:8000
python -m discovery run --dry-run --out candidates.jsonl    # needs a logged-in `claude` CLI
```

## Setup (once)

- Actions secret `CLAUDE_CODE_OAUTH_TOKEN`, created with `claude setup-token` (uses a Claude subscription).
- Settings → Pages → Source: GitHub Actions.
