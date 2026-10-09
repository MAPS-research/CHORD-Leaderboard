#!/bin/bash
#SBATCH --job-name=lb-gen-ar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
# Autoregressive sampling at the paper's GPT-2 operating point (nucleus, T=1.0, p=0.95, 480-512 new tokens).
# From the repository root, with account/partition on the command line:
#   sbatch -A <account> -p <partition> -o <logdir>/%x_%j.out --export=ALL,MODEL=gpt2-xl,TAG=gpt2xl scoring/jobs/gen_ar.slurm
set -uo pipefail
source scoring/jobs/common.sh
claim_or_exit
activate chord-gpu
cd "$CHORD_EXP" && export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total --format=csv
rc=0
for k in $SEEDS; do
  F=$SAMPLES/r10-$TAG-s$k.jsonl
  if have "$F"; then echo "skip $F"; continue; fi
  CFG=$SAMPLES/.gen_ar_${TAG}_$k.yaml
  printf 'model: %s\nn_samples: %s\nmax_new_tokens: 512\nmin_new_tokens: 480\ntemperature: 1.0\ntop_p: 0.95\nrepetition_penalty: 1.0\nbatch_size: %s\nseed: %s\noutput_path: %s\n' \
    "$MODEL" "$N_SAMPLES" "$BATCH" "$(seed_of $k)" "$F.tmp" > "$CFG"
  python -m chord.data.generate_ar --config "$CFG" && mv "$F.tmp" "$F" || rc=1
done
echo "GEN_EXIT=$rc"; exit $rc
