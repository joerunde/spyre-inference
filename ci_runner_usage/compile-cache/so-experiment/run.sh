#!/usr/bin/env bash
# run.sh <label> <rpm-prefix> : one compiled micro-g3.3 test in the worktree, own inductor cache dir.
set -uo pipefail
LABEL=$1; RPMS=$2; TEST=${3:-'tests/e2e/test_compile.py::test_basic_llm_inference[micro-g3.3]'}
SO=${SO:-$CLAUDE_JOB_DIR/tmp/so}
cd ${WORKTREE:-$CLAUDE_JOB_DIR/tmp/sowt}
source $RPMS/env.sh >/dev/null 2>&1
export TORCHINDUCTOR_CACHE_DIR=$SO/cache-$LABEL
export SPYRE_KERNEL_CACHE=1
mkdir -p $TORCHINDUCTOR_CACHE_DIR
find $TORCHINDUCTOR_CACHE_DIR -type f \( -name '*.so' -o -name '*.o' \) -printf '%P\t%T@\n' | sort > $SO/$LABEL.pre.tsv
start=$(date +%s)
.venv/bin/python -m pytest "$TEST" -m 'not upstream' -p no:cacheprovider -q -s > $SO/$LABEL.log 2>&1
rc=$?
end=$(date +%s)
find $TORCHINDUCTOR_CACHE_DIR -type f \( -name '*.so' -o -name '*.o' \) -printf '%P\t%T@\n' | sort > $SO/$LABEL.post.tsv
echo "$LABEL rc=$rc wall=$((end-start))s torch_spyre=$(.venv/bin/python -c 'import torch_spyre._version as v;print(v.__version__)' 2>/dev/null) lib=$(md5sum < $LIB_VERSION_FILE | cut -c1-8)"
