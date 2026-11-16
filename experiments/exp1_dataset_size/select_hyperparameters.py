"""
Hyperparameter selection of the dataset-size study: for every N_data, the penalty weight lambda and the
augmentation fraction rho with the lowest validation MSE (geometric mean over the tuning seeds 0-2 of the
best validation MSE of each run). A choice at the upper end of a grid is flagged; the lower end is covered
by the free MLP (lambda = rho = 0).

Usage: python -m experiments.exp1_dataset_size.select_hyperparameters [--results DIR]
Outputs in results/ (or --results): selection.json and jobs_selected.txt (seeds 3-9 with the chosen values).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments import rel
from .train import RESULTS, run_name

N_VALUES = [50_000, 10_000, 1_000, 500, 200]
GRIDS = {"penalty_mlp": ("lam", [0.01, 0.1, 1, 10]), "aug_mlp": ("rho", [0.1, 0.25, 0.5, 0.75])}
TUNING_SEEDS = [0, 1, 2]
FINAL_SEEDS = range(3, 10)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    results = Path(ap.parse_args(argv).results)
    selection, jobs = {}, []
    for n in N_VALUES:
        for model, (prefix, grid) in GRIDS.items():
            score = {p: float(10 ** np.mean([np.log10(json.load(open(results / f"{run_name(model, n, p, s)}.json"))
                                                      ["config"]["best_val_mse"]) for s in TUNING_SEEDS]))
                     for p in grid}
            best = min(score, key=score.get)
            selection[f"{model}_n{n}"] = {"chosen": best, "val_mse": {f"{p:g}": v for p, v in score.items()},
                                          "upper_boundary": best == grid[-1]}
            print(f"N={n:>6} {model:12s} chosen {prefix}={best:g}"
                  + (" (UPPER END OF THE GRID)" if best == grid[-1] else "") + " | val MSE "
                  + ", ".join(f"{prefix}={p:g}: {v:.2e}" for p, v in score.items()))
            jobs += [f"{n} {model} {best:g} {s}" for s in FINAL_SEEDS]
    with open(results / "selection.json", "w") as fh:
        json.dump(selection, fh, indent=1)
    (results / "jobs_selected.txt").write_text("\n".join(jobs) + "\n")
    print(f"saved {rel(results / 'selection.json')} and {len(jobs)} jobs in {rel(results / 'jobs_selected.txt')}")


if __name__ == "__main__":
    main()
