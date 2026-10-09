#!/bin/bash
# Copy the paper's 27B features (human reference, held-out human, six paper generators) into the
# leaderboard output tree so featurization only embeds new runs. Copies, never links: a cache miss
# would otherwise overwrite the paper's files through the link. Safe to rerun (cp -n).
set -euo pipefail
source scoring/jobs/common.sh
ENC=qwen35-27b-prompteol-coherence-l62
SRC=/scratch/jz5770/CHORD_open_source/outputs/casestudy/unconditional_generation/features/$ENC
DST=$LB_ROOT/features/$ENC
mkdir -p "$DST"
for f in reference r10-human $(cd "$SRC" && ls r10-*.npy | sed 's/\.npy$//' | grep -v '^r10-human$'); do
  cp -n "$SRC/$f.npy" "$SRC/$f.json" "$DST/"
done
ls "$DST" | wc -l
