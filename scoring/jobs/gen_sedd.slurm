#!/bin/bash
#SBATCH --job-name=lb-gen-sedd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
# SEDD sampling at the paper's operating point (256 steps, length 512), official repo under third_party/SEDD.
#   sbatch -A <account> -p <partition> -o <logdir>/%x_%j.out --export=ALL,HF_MODEL=louaaron/sedd-medium,TAG=seddmedium scoring/jobs/gen_sedd.slurm
set -uo pipefail
source scoring/jobs/common.sh
activate chord-generators
cd "$CHORD_EXP" && export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total --format=csv
MODEL_DIR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_MODEL'))") || exit 1
rc=0
for k in $SEEDS; do
  F=$SAMPLES/r10-$TAG-s$k.jsonl
  if have "$F"; then echo "skip $F"; continue; fi
  python -m chord.data.generators.sedd --repo third_party/SEDD --model "$MODEL_DIR" --run-id r10-$TAG-s$k --seed "$(seed_of $k)" \
    --num-samples "$N_SAMPLES" --batch-size "$BATCH" --steps 256 --length 512 --output "$F.tmp" && mv "$F.tmp" "$F" || rc=1
done
echo "GEN_EXIT=$rc"; exit $rc
