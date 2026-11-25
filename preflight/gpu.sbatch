#!/usr/bin/env bash
#SBATCH --job-name=a4-preflight-gpu
#SBATCH --account=h100_harlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/preflight/logs/gpu-%j.log
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/preflight/common.sh
# The same check is queued on two partitions; whichever starts first does the work, the other exits.
if ! mkdir "$AGENT4_ROOT/preflight/results/.gpu_lock" 2>/dev/null; then echo "sibling job already ran; exiting"; exit 0; fi
echo "host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
bash preflight/net_check.sh
python -B preflight/compute_check.py --gpu
echo "end=$(date -Is)"
