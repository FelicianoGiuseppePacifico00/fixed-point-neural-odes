#!/bin/bash
# All the training of the dataset-size study, N_data in {5e4, 1e4, 1e3, 500, 200} (largest first), on the CPU:
#  stream A: tuning runs (penalty lambda in {0.01, 0.1, 1, 10}, augmentation rho in {0.1, 0.25, 0.5, 0.75},
#            seeds 0-2), then select_hyperparameters, then the chosen lambda / rho on seeds 3-9;
#  stream B, at the same time: QR and free MLP, seeds 0-9;
#  then the runs stopped by the 2e6-step cap continue to 6e6 steps, and the selection is repeated (the
#  choice does not change).
# Then evaluate.py and plot.py give the evaluation, Figs. 5-6 and Table 1.
# Usage: experiments/exp1_dataset_size/run_all.sh
#        (OUT=<results folder>, default results/ of this folder; PYTHON=<interpreter>;
#         P_A / P_B = parallel runs of the two streams, default 19 and 9)
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.." || exit 1   # repository root
export OUT="${OUT:-$HERE/results}"
PY="${PYTHON:-python}"
M=experiments.exp1_dataset_size
mkdir -p "$OUT/logs"
for N in 50000 10000 1000 500 200; do
  for SEED in 0 1 2; do
    for LAM in 0.01 0.1 1 10; do echo "$N penalty_mlp $LAM $SEED"; done
    for RHO in 0.1 0.25 0.5 0.75; do echo "$N aug_mlp $RHO $SEED"; done
  done
done > "$OUT/jobs_tuning.txt"
for N in 50000 10000 1000 500 200; do
  for SEED in 0 1 2 3 4 5 6 7 8 9; do echo "$N qr 0 $SEED"; echo "$N free_mlp 0 $SEED"; done
done > "$OUT/jobs_baselines.txt"
( "$HERE/tools/launch_jobs.sh" "$OUT/jobs_tuning.txt" "${P_A:-19}" cpu \
  && "$PY" -m $M.select_hyperparameters --results "$OUT" > "$OUT/logs/selection.txt" 2>&1 \
  && "$HERE/tools/launch_jobs.sh" "$OUT/jobs_selected.txt" "${P_A:-19}" cpu ) &
"$HERE/tools/launch_jobs.sh" "$OUT/jobs_baselines.txt" "${P_B:-9}" cpu &
wait
"$PY" -m $M.tools.list_capped_runs 6e6 --results "$OUT" > "$OUT/jobs_extend.txt"
"$HERE/tools/launch_jobs.sh" "$OUT/jobs_extend.txt" 25 cpu
"$PY" -m $M.select_hyperparameters --results "$OUT" > "$OUT/logs/selection_after_extension.txt" 2>&1
echo "all runs finished $(LC_ALL=C date)" >> "$OUT/logs/done.txt"
