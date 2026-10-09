#!/bin/bash
# Sequential measurement matrix (one Spyre engine at a time).
# Usage: run_all.sh <model> <max_num_seqs>
set -u
M=/home/senuser/.claude/jobs/709f9980/tmp/meas
WT=/home/senuser/spyre-inference/.claude/worktrees/mrv2-investigation-906
MODEL=${1:-ibm-granite/granite-3.3-8b-instruct}
SEQS=${2:-32}
cd "$WT"

run() {  # name mode kernel_cache cache_dir
  local name=$1 mode=$2 kc=$3 dir=$4
  mkdir -p "$dir"
  echo "=== $(date +%T) start $name (mode=$mode kernel_cache=$kc dir=$(basename "$dir"))"
  MEAS_MODE=$mode MEAS_OUT="$M/res-$name.json" MEAS_MODEL="$MODEL" MEAS_MAX_SEQS="$SEQS" \
  SPYRE_KERNEL_CACHE=$kc TORCHINDUCTOR_CACHE_DIR="$dir" \
    timeout 7200 uv run --no-sync python "$M/meas.py" > "$M/log-$name.log" 2>&1
  echo "=== $(date +%T) end $name exit=$?"
}

# M1/M2: cold, kernel cache off (today's default), one fresh cache dir each.
run B-cold   B 0 "$M/c-B"
run D-cold   D 0 "$M/c-D"
run A-cold   A 0 "$M/c-A"
# M3: restart behaviour.
run B-restart-kc0 B 0 "$M/c-B"     # FX graph cache warm (from B-cold), kernel cache off
run B-cold-kc1    B 1 "$M/c-K"     # cold, kernel cache on (populates it)
run B-warm-kc1    B 1 "$M/c-K"     # warm restart, kernel cache on
echo "=== ALL DONE"
