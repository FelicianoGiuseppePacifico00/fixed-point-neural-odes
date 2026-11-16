#!/bin/bash
# Runs experiments.exp1_dataset_size.train for every line "<n_data> <model_type> <param> <seed> [<min_lr> <max_steps>]"
# of a jobs file, <n_parallel> at a time, on the CPU (one thread per run, the default) or on the GPU (FP_DEVICE /
# CUDA_VISIBLE_DEVICES). Finished runs are skipped and interrupted ones resume from their last checkpoint, so
# the same jobs file can be launched again; a larger max_steps continues capped runs. Used by run_all.sh.
# The model is small: on a 32-core CPU, 28 parallel runs give about 4x the throughput of one GPU.
# Usage: tools/launch_jobs.sh <jobs_file> <n_parallel> [cpu|gpu]
#        (OUT=<results folder>, default results/ of the experiment; PYTHON=<interpreter>)
JOBS=$(realpath "$1")
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/../.." || exit 1   # repository root
if [ "${3:-cpu}" = cpu ]; then
  export CUDA_VISIBLE_DEVICES="" FP_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fi
export PYTHONUNBUFFERED=1 OUT="${OUT:-$HERE/results}"
PY="${PYTHON:-python}"
M=experiments.exp1_dataset_size.train
mkdir -p "$OUT/logs"
xargs -P "$2" -L 1 bash -c \
  '"$0" -m '"$M"' $1 $2 $3 $4 $5 $6 --out-dir "$OUT" >> "$OUT/logs/n$1_$2_p$3_seed$4.log" 2>&1' "$PY" < "$JOBS"
echo "$(basename "$JOBS") finished $(LC_ALL=C date)" >> "$OUT/logs/done.txt"
