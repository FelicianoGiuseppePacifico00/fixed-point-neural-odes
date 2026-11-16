#!/bin/bash
# All the fixed-epoch runs of Section 5.2 (N_train = 5e4, 400 epochs, seeds 0-4):
#   cosine schedule:   QR, free MLP, MLP + penalty with lambda in {1, 10, 50}   (Experiment 1, Figs. 2-4)
#   constant lr 1e-3:  the same models                                          (check with a constant learning rate)
# The QR runs with the cosine schedule run one at a time, so that their wall-clock time is meaningful; the
# other runs N_PARALLEL at a time. Device: FP_DEVICE / CUDA_VISIBLE_DEVICES (default: CUDA if available).
# Finished runs are skipped.
# Usage: experiments/exp1_fixed_epochs/run_all.sh
#        (OUT=<results folder>, default results/ of this folder; PYTHON=<interpreter>; N_PARALLEL, default 8)
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.." || exit 1   # repository root
export OUT="${OUT:-$HERE/results}"
PY="${PYTHON:-python}"
M=experiments.exp1_fixed_epochs.train
mkdir -p "$OUT/logs"
for SEED in 0 1 2 3 4; do
  "$PY" -m $M --model qr --seed $SEED --schedule cosine --out-dir "$OUT/cosine" >> "$OUT/logs/cosine_qr_lam0_seed$SEED.log" 2>&1
done
for SCHEDULE in cosine constant; do
  for SEED in 0 1 2 3 4; do
    [ $SCHEDULE = constant ] && echo "qr 0 $SEED $SCHEDULE"
    echo "free_mlp 0 $SEED $SCHEDULE"
    for LAM in 1 10 50; do echo "penalty_mlp $LAM $SEED $SCHEDULE"; done
  done
done | xargs -P "${N_PARALLEL:-8}" -L 1 bash -c \
  '"$0" -m '"$M"' --model $1 --lam $2 --seed $3 --schedule $4 --out-dir "$OUT/$4" >> "$OUT/logs/$4_$1_lam$2_seed$3.log" 2>&1' "$PY"
