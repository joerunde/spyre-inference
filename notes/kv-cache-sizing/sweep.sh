HERE=$(cd "$(dirname "$0")" && pwd)
cd "${REPO:?set REPO to a spyre-inference checkout with a working venv}"
export SPYRE_NUM_CPUS=16 VLLM_ALLOW_INSECURE_SERIALIZATION=1
R=${OUT:-$HERE/runs-new}; mkdir -p $R
run() { name=$1; shift; timeout 1700 .venv/bin/python "$HERE"/kv_probe.py $R/$name.json "$@" > $R/$name.log 2>&1; echo "$name rc=$?"; }
run kvbytes40g --kv-bytes $((40*1024**3))
run gmu05 --gmu 0.5
run blocks1025 --blocks 1025
run blocks4097 --blocks 4097
run blocks8000 --blocks 8000
run blocks8900 --blocks 8900
run blocks9200 --blocks 9200
echo SWEEP_DONE
