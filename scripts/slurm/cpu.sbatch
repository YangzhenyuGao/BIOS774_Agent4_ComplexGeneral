#!/usr/bin/env bash
#SBATCH --account=rc_yunmli_pi
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/logs/slurm/%x-%j.log
# Generic CPU runner: sbatch [overrides] scripts/slurm/cpu.sbatch <python module or script> [args...]
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/scripts/slurm/common.sh
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-} cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is) cmd=$*"
SECONDS=0
python -B "$@"
echo "WALL_SECONDS=$SECONDS"
echo "end=$(date -Is)"
