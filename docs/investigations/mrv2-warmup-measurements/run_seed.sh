#!/bin/bash
# Does a fixed Python hash seed make attention kernels hit the Spyre kernel cache across processes?
set -u
M=/home/senuser/.claude/jobs/709f9980/tmp/meas
cd /home/senuser/spyre-inference/.claude/worktrees/mrv2-investigation-906
mkdir -p "$M/c-S"
for name in S-cold S-warm; do
  echo "=== $(date +%T) start $name"
  PYTHONHASHSEED=0 MEAS_MODE=B MEAS_OUT="$M/res-$name.json" MEAS_MODEL=ibm-ai-platform/micro-g3.3-8b-instruct-1b \
  SPYRE_KERNEL_CACHE=1 TORCHINDUCTOR_CACHE_DIR="$M/c-S" \
    timeout 3600 uv run --no-sync python "$M/meas.py" > "$M/log-$name.log" 2>&1
  echo "=== $(date +%T) end $name exit=$?"
done
echo "=== ALL DONE"
