#!/usr/bin/env bash
# Shared environment for Agent4 preflight jobs. Mirrors Agent3's conventions.
set -euo pipefail
AGENT4_ROOT=/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral
AGENT_ENV=/vast/som/yunligrp/users/ygao/conda/envs/agent3
export AGENT4_ROOT AGENT_ENV
export PATH="$AGENT_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$AGENT4_ROOT/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export HF_HOME="$XDG_CACHE_HOME/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
# Node-local temp dir: long project paths exceed the Unix socket path limit (Agent3 lesson).
TMPDIR=$(mktemp -d "/tmp/agent4.${UID}.${SLURM_JOB_ID:-login}.XXXXXX"); export TMPDIR
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export LD_LIBRARY_PATH="$AGENT_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$AGENT_ENV/etc/fonts/fonts.conf" ]]; then
  export FONTCONFIG_FILE="$AGENT_ENV/etc/fonts/fonts.conf" FONTCONFIG_PATH="$AGENT_ENV/etc/fonts"
fi
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TORCH_HOME" "$HF_HOME"
cd "$AGENT4_ROOT"
python -c 'import sys; assert not any("/.local/" in p for p in sys.path), sys.path'
