# Shared settings for the Table 2 scoring jobs. Sourced from the repository root by every job.
CHORD_EXP=${CHORD_EXP:-/scratch/jz5770/CHORD_experiments}      # CHORD-Experiment checkout (data/outputs/third_party link to the library)
CHORD_LIB=${CHORD_LIB:-/scratch/jz5770/CHORD_open_source}      # CHORD library (not installed in every conda env)
LB_ROOT=${LB_ROOT:-/scratch/jz5770/chord_leaderboard/table2}    # leaderboard outputs: samples/, features/, scores/
ELF_ROOT=${ELF_ROOT:-/scratch/jz5770/ELF}                       # lillian039/ELF@pytorch_elf
CONDA_ACTIVATE=${CONDA_ACTIVATE:-/scratch/jz5770/miniconda/bin/activate}
SEEDS=${SEEDS:-"01 02 03 04 05 06 07 08 09 10"}                # fold k uses seed 20260800 + k, as in the paper
N_SAMPLES=${N_SAMPLES:-500}
BATCH=${BATCH:-50}
LB_REPO=$(pwd)
SAMPLES=$LB_ROOT/samples
export HF_HOME=${HF_HOME:-/scratch/jz5770/.huggingface} TOKENIZERS_PARALLELISM=false
mkdir -p "$SAMPLES"

activate() { source "$CONDA_ACTIVATE" "$1"; }
seed_of() { echo $((20260800 + 10#$1)); }
have() { [ -f "$1" ] && [ "$(wc -l < "$1")" -ge "$N_SAMPLES" ]; }

# The same job may be submitted to two partitions (e.g. b200 and a100) to start sooner.
# Only the lowest-numbered running copy does the work; a later copy exits without touching files.
claim_or_exit() {
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  local first
  first=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t RUNNING,COMPLETING -o %i | sort -n | head -1)
  if [ -n "$first" ] && [ "$first" != "$SLURM_JOB_ID" ]; then echo "copy $first of $SLURM_JOB_NAME is already running; exiting"; exit 0; fi
}
