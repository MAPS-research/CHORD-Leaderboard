#!/bin/bash
#SBATCH --job-name=lb-feat-27b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --requeue
# 27B CHORD features for every run in scoring/configs/runs.jsonl (cached runs are skipped). Needs >= 80 GB GPU memory.
#   sbatch -A <account> -p <partition> -o <logdir>/%x_%j.out scoring/jobs/featurize.slurm
set -uo pipefail
source scoring/jobs/common.sh
claim_or_exit
activate chord-gpu
cd "$CHORD_EXP" && export PYTHONPATH=$PWD
# refuse to embed partial runs: every sample file in the manifest must be complete
python - "$LB_REPO/scoring/configs/runs.jsonl" "$N_SAMPLES" <<'PY' || exit 1
import json, sys
from pathlib import Path
need, bad = int(sys.argv[2]), []
for line in open(sys.argv[1]):
    r = json.loads(line)
    p = Path(r["samples_path"])
    n = sum(1 for _ in open(p)) if p.exists() else 0
    if r["run_id"] != "r10-human" and n < need:
        bad.append(f"{r['run_id']}: {n} lines")
if bad:
    sys.exit("incomplete samples:\n" + "\n".join(bad))
print("all sample files complete")
PY
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m chord.featurize --config "$LB_REPO/scoring/configs/featurize.yaml" --only-encoder qwen35-27b-prompteol-coherence-l62
rc=$?; echo "FEAT_EXIT=$rc"; exit $rc
