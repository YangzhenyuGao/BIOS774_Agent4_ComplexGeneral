#!/usr/bin/env bash
#SBATCH --job-name=a4-preflight-ctrl
#SBATCH --account=rc_yunmli_pi
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/preflight/logs/ctrl-%j.log
# Can the whole controller live inside a CPU Slurm job instead of on the login node?
# It needs three things from a compute node: the Claude judge call, the Gemini/HF calls, and job submission.
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral/preflight/common.sh
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
echo "claude binary: $(bash preflight/claude_cli.sh --where)"
printf '%s' 'Reply with one JSON object and nothing else: {"ok": true, "word": "READY"}' > "$TMPDIR/sys.txt"
if echo "Confirm." | bash preflight/claude_cli.sh "$TMPDIR/sys.txt" claude-opus-5-5 low > "$TMPDIR/judge.json" 2> "$TMPDIR/judge.err"; then
  python - "$TMPDIR/judge.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("CLAUDE_FROM_COMPUTE_NODE", "PASS" if not d.get("is_error") else "FAIL", "| api_ms", d.get("duration_api_ms"), "| cost_usd", d.get("total_cost_usd"), "| result", (d.get("result") or "")[:80])
PY
else
  echo "CLAUDE_FROM_COMPUTE_NODE FAIL"; head -c 600 "$TMPDIR/judge.err"
fi
python -B - <<'PY'
import sys
sys.path.insert(0, "preflight")
import llm_check as L
for row in (L.gemini("gemini-4-neon"), L.hf_router("deepseek-ai/DeepSeek-V4-Pro")):
    print("LLM_FROM_COMPUTE_NODE", "PASS" if row.get("status") == 200 else "FAIL", row["model"], row.get("status"), row.get("seconds"), row.get("reply"))
PY
if out=$(sbatch --test-only preflight/cpu.sbatch 2>&1); then echo "SBATCH_FROM_COMPUTE_NODE PASS | $out"; else echo "SBATCH_FROM_COMPUTE_NODE FAIL | $out"; fi
echo "end=$(date -Is)"
