#!/usr/bin/env bash
#SBATCH --job-name=a4-preflight-cpu
#SBATCH --account=rc_yunmli_pi
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/preflight/logs/cpu-%j.log
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/preflight/common.sh
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
bash preflight/net_check.sh
python -B preflight/compute_check.py
echo "=== R ==="
set +u; source /etc/profile.d/lmod.sh 2>/dev/null || true; module load r/4.6.0; set -u
Rscript --vanilla preflight/r_check.R
echo "end=$(date -Is)"
