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
activate chord-gpu
cd "$CHORD_EXP" && export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m chord.featurize --config "$LB_REPO/scoring/configs/featurize.yaml" --only-encoder qwen35-27b-prompteol-coherence-l62
rc=$?; echo "FEAT_EXIT=$rc"; exit $rc
