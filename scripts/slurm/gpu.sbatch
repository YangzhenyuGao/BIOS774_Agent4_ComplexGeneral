#!/usr/bin/env bash
#SBATCH --account=h100_harlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/logs/slurm/%x-%j.log
# Generic GPU runner: sbatch --partition=<h100 partition> scripts/slurm/gpu.sbatch <python module or script> [args...]
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/scripts/slurm/common.sh
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-} partition=${SLURM_JOB_PARTITION} start=$(date -Is) cmd=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SECONDS=0
python -B "$@"
echo "WALL_SECONDS=$SECONDS"
echo "end=$(date -Is)"
