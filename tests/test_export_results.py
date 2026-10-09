import csv
import json
from datetime import date
from pathlib import Path

import pytest

from scoring.export_results import ENCODER, export, main

IDS = {"GPT2-large-AR": "gpt2-large", "Human-packed": "human-heldout"}


def _rows(model, values, encoder=ENCODER):
    return [{"encoder": encoder, "model": model, "fold": str(k), "n": "500", "sigma": "1.0", "mmd_x100": str(v)} for k, v in enumerate(values)]


def test_export_keeps_27b_rows_orders_folds_and_computes_stats():
    rows = _rows("GPT2-large-AR", [3.0, 1.0, 2.0]) + _rows("GPT2-large-AR", [9.0, 9.0, 9.0], encoder="chord-qwen3.5-2b-student")
    rows = [rows[1], rows[0], rows[2]] + rows[3:]  # folds out of order in the CSV
    out = export(rows, IDS, scored_on=date(2026, 10, 9))
    r = out["gpt2-large"]
    assert r["folds"] == [3.0, 1.0, 2.0] and r["mean"] == 2.0 and r["std"] == 1.0
    assert r["samples_per_fold"] == 500 and r["scored_on"] == "2026-10-09" and r["encoder"] == ENCODER
    assert r["setting"] == "unconditional-owt" and r["metric"] == "rbf_mmd2_x100_biased"


def test_unknown_model_names_are_an_error():
    with pytest.raises(ValueError, match="SEDD-medium"):
        export(_rows("SEDD-medium", [1.0, 2.0]), IDS, scored_on=date(2026, 10, 9))


def test_main_writes_one_file_per_id(tmp_path: Path):
    src = tmp_path / "folds.csv"
    with open(src, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["encoder", "model", "fold", "n", "sigma", "mmd_x100"])
        w.writeheader()
        w.writerows(_rows("GPT2-large-AR", [1.0, 3.0]) + _rows("Human-packed", [0.1, 0.3]))
    ids = tmp_path / "ids.yaml"
    ids.write_text("GPT2-large-AR: gpt2-large\nHuman-packed: human-heldout\n")
    main(["--csv", str(src), "--ids", str(ids), "--out", str(tmp_path / "results"), "--scored-on", "2026-10-09"])
    assert sorted(p.name for p in (tmp_path / "results").iterdir()) == ["gpt2-large.json", "human-heldout.json"]
    assert json.loads((tmp_path / "results" / "human-heldout.json").read_text())["mean"] == 0.2
