#!/usr/bin/env bash
# shard.sh <label> <rpm-prefix> [seed-cache-dir] : smoke shard 2/11 in the worktree, own inductor cache dir.
set -uo pipefail
LABEL=$1; RPMS=$2; SEED=${3:-}
SO=${SO:-$CLAUDE_JOB_DIR/tmp/so}
cd ${WORKTREE:-$CLAUDE_JOB_DIR/tmp/sowt}
source $RPMS/env.sh >/dev/null 2>&1
export TORCHINDUCTOR_CACHE_DIR=$SO/cache-$LABEL SPYRE_KERNEL_CACHE=1 HF_HUB_OFFLINE=1
rm -rf $TORCHINDUCTOR_CACHE_DIR; mkdir -p $TORCHINDUCTOR_CACHE_DIR
[ -n "$SEED" ] && (cd $SEED && find . -type f \( -name '*.so' -o -name '*.o' \) -print0 | cpio -pdm0 --quiet $TORCHINDUCTOR_CACHE_DIR)
find $TORCHINDUCTOR_CACHE_DIR -type f \( -name '*.so' -o -name '*.o' \) -printf '%P\t%T@\n' | sort > $SO/$LABEL.pre.tsv
start=$(date +%s)
.venv/bin/python -m pytest -m 'not (distributed or distributed_tp4 or upstream or attention or probe or model_quality)' \
  --smoke-shards=11 --smoke-shard-id=2 -p no:cacheprovider -q --durations=0 \
  --junitxml=$SO/$LABEL.xml > $SO/$LABEL.log 2>&1
rc=$?
end=$(date +%s)
find $TORCHINDUCTOR_CACHE_DIR -type f \( -name '*.so' -o -name '*.o' \) -printf '%P\t%T@\n' | sort > $SO/$LABEL.post.tsv
echo "$LABEL rc=$rc wall=$((end-start))s ts=$(.venv/bin/python -c 'import torch, torch_spyre._version as v;print(v.__version__)' 2>/dev/null) so_o_pre=$(wc -l < $SO/$LABEL.pre.tsv) so_o_post=$(wc -l < $SO/$LABEL.post.tsv) $(tail -1 $SO/$LABEL.log)"
