"""Convert fold scores (score_chord_folds.py CSV) into results/<registry id>.json for the website.

  python scoring/export_results.py --csv $LB_ROOT/scores/repeat_folds_biased.csv \
      --ids scoring/configs/model_ids.yaml --out results/ [--scored-on YYYY-MM-DD]

Only the 27B CHORD encoder is exported. Every model name in the CSV must be mapped in --ids.
"""

import argparse
import csv
import json
import statistics
from datetime import date
from pathlib import Path

import yaml

ENCODER = "qwen35-27b-prompteol-coherence-l62"
SETTING = "unconditional-owt"
METRIC = "rbf_mmd2_x100_biased"


def export(rows, ids: dict[str, str], scored_on: date) -> dict[str, dict]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        if row["encoder"] == ENCODER:
            by_model.setdefault(row["model"], []).append(row)
    unknown = sorted(set(by_model) - set(ids))
    if unknown:
        raise ValueError(f"no registry id for model(s): {', '.join(unknown)}; add them to the --ids file")
    out = {}
    for model, model_rows in by_model.items():
        model_rows.sort(key=lambda r: int(r["fold"]))
        folds = [float(r["mmd_x100"]) for r in model_rows]
        out[ids[model]] = {
            "id": ids[model], "setting": SETTING, "encoder": ENCODER, "metric": METRIC, "folds": folds,
            "samples_per_fold": int(model_rows[0]["n"]),
            "mean": round(statistics.mean(folds), 4), "std": round(statistics.stdev(folds), 4),
            "scored_on": scored_on.isoformat(),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scored-on", type=date.fromisoformat, default=None, help="default: the CSV's modification date")
    args = ap.parse_args(argv)
    scored_on = args.scored_on or date.fromtimestamp(args.csv.stat().st_mtime)
    with open(args.csv, newline="") as fh:
        results = export(csv.DictReader(fh), yaml.safe_load(args.ids.read_text()), scored_on)
    args.out.mkdir(parents=True, exist_ok=True)
    for id_, data in sorted(results.items()):
        (args.out / f"{id_}.json").write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {args.out / f'{id_}.json'}: {data['mean']:.2f} (±{data['std']:.2f})")


if __name__ == "__main__":
    main()
