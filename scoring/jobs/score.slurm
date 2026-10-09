#!/bin/bash
#SBATCH --job-name=lb-score-27b
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# Fold scoring exactly as the paper's Table 2: biased RBF-MMD^2 x100, median bandwidth on 150 dev windows,
# mean +- std over 10 disjoint folds of 500. Writes $LB_ROOT/scores/repeat_folds_biased.csv and table2_27b.txt.
#   sbatch -A <account> -p cpu_short -o <logdir>/%x_%j.out scoring/jobs/score.slurm
set -uo pipefail
source scoring/jobs/common.sh
activate chord-gpu
cd "$CHORD_EXP" && export PYTHONPATH=$PWD
mkdir -p "$LB_ROOT/scores"
python -m experiments.generator_eval.score_chord_folds --config "$LB_REPO/scoring/configs/featurize.yaml" \
  --encoders qwen35-27b-prompteol-coherence-l62 | tee "$LB_ROOT/scores/table2_27b.txt"
rc=${PIPESTATUS[0]}; echo "SCORE_EXIT=$rc"; exit $rc
