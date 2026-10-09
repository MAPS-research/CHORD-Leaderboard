#!/bin/bash
#SBATCH --job-name=lb-gen-elf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
# ELF sampling with the official sampler, then normalisation to the CHORD sample format.
# SAMPLING is the operating point from the ELF README (see scoring/configs/elf_*.yaml).
#   sbatch -A <account> -p <partition> -o <logdir>/%x_%j.out \
#     --export=ALL,SIZE=B,TAG=elfB,SAMPLING=scoring/configs/elf_b_sde32_sccfg3.yaml scoring/jobs/gen_elf.slurm
set -uo pipefail
source scoring/jobs/common.sh
SAMPLING_ABS=$LB_REPO/$SAMPLING
nvidia-smi --query-gpu=name,memory.total --format=csv
rc=0
for k in $SEEDS; do
  SEED=$(seed_of $k)
  F=$SAMPLES/r10-$TAG-s$k.jsonl
  if have "$F"; then echo "skip $F"; continue; fi
  activate elf
  cd "$ELF_ROOT"
  RAW=outputs/chord_leaderboard/$TAG/seed_$SEED; rm -rf "$RAW"
  NGPU=1 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-$SIZE.yml \
      --checkpoint_path embedded-language-flows/ELF-$SIZE-owt-torch --seed "$SEED" \
      --config_override use_bf16=true --config_override use_compile=true --config_override use_wandb=false \
      --config_override global_batch_size=none --config_override batch_size="$BATCH" \
      --config_override num_samples="$N_SAMPLES" --config_override output_dir="$RAW" \
      --config_override sampling_configs_path="$SAMPLING_ABS" || { rc=1; continue; }
  GEN=$(ls "$RAW"/*/all_generated_*.jsonl | head -1)
  activate chord-gpu
  cd "$CHORD_EXP" && export PYTHONPATH=$PWD
  python -m chord.data.generators.normalize_samples --format jsonl --input-key generated --model "ELF-$SIZE-OWT" \
      --run-id r10-$TAG-s$k --seed "$SEED" --input "$ELF_ROOT/$GEN" --output "$F.tmp" && mv "$F.tmp" "$F" || rc=1
done
echo "GEN_EXIT=$rc"; exit $rc
