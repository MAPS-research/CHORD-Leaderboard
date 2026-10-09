# Scoring (Table 2 protocol)

Jobs that sample registry models on the Torch cluster and score them exactly as in the CHORD paper's
Table 2: 10 seeds x 500 unconditional samples per model (seed 20260800 + k), the paper's human reference
(150 bandwidth windows + 10 x 500 OpenWebText windows), the 27B CHORD encoder, biased RBF-MMD^2 x100,
mean +- std over 10 disjoint folds. The paper's six generators are re-listed in `configs/runs.jsonl`, so
new models are ranked jointly with them.

The jobs call code in the CHORD-Experiment checkout on the cluster (`CHORD_EXP` in `jobs/common.sh`) and
write to `LB_ROOT` (`/scratch/jz5770/chord_leaderboard/table2`). Run them from the repository root on the
cluster; give account and partition on the command line.

| Step | Command |
|---|---|
| Copy the paper's 27B features (once) | `bash scoring/jobs/seed_features.sh` |
| Sample an AR model | `sbatch ... --export=ALL,MODEL=gpt2-xl,TAG=gpt2xl scoring/jobs/gen_ar.slurm` |
| Sample a SEDD model | `sbatch ... --export=ALL,HF_MODEL=louaaron/sedd-medium,TAG=seddmedium scoring/jobs/gen_sedd.slurm` |
| Sample an ELF model | `sbatch ... --export=ALL,SIZE=B,TAG=elfB,SAMPLING=scoring/configs/elf_b_sde32_sccfg3.yaml scoring/jobs/gen_elf.slurm` |
| 27B features for new runs | `sbatch ... scoring/jobs/featurize.slurm` |
| Fold scores | `sbatch ... -p cpu_short scoring/jobs/score.slurm` |

Operating points: GPT-2 nucleus T=1.0, p=0.95, 480-512 new tokens; SEDD 256 steps; ELF-B 32-step SDE
(gamma 1.5, SC-CFG 3) and ELF-M 64-step SDE (gamma 1.0, SC-CFG 3) as in the ELF README. The paper's ELF-L
row used 64 steps with SC-CFG 4.

Set `SEEDS=01 N_SAMPLES=8 BATCH=8 LB_ROOT=<scratch dir>` for a smoke run.
