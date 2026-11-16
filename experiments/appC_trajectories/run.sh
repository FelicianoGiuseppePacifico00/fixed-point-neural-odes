#!/bin/bash
# Launches (or resumes, from <OUT>/checkpoints/stack_last.pt) the stacked training on trajectories, detached,
# on a GPU: all the 60 runs share one process (about 35 h on one RTX A5500). Select the GPU with
# CUDA_VISIBLE_DEVICES. Then evaluate.py, check_solver.py and plot.py give Fig. 8 and Tables 2-3.
# Usage: experiments/appC_trajectories/run.sh [extra arguments of train.py]
#        (OUT=<results folder>, default results/ of this folder; PYTHON=<interpreter>)
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.." || exit 1   # repository root
OUT="${OUT:-$HERE/results}"
mkdir -p "$OUT/logs"
FP_DEVICE=${FP_DEVICE:-cuda} setsid nohup "${PYTHON:-python}" -u -m experiments.appC_trajectories.train --out-dir "$OUT" \
    "$@" >> "$OUT/logs/worker.log" 2>&1 < /dev/null &
echo "launched pid $!"
