"""
Quick check of the main results on a CPU (a few minutes): Experiment 1 of the paper (Section 5.2) retrained
with the protocol of the paper (N_train = 5e4, 400 epochs, cosine-annealed learning rate) and the training
script of the paper (exp1_fixed_epochs/train.py), for the QR-constrained model, the free MLP and the
MLP + penalty with lambda = 1, one seed by default. The learned fields are then compared with the target field
near the equilibria (local RMSE, Jacobian error) and along trajectories (error at t = 30 from the 500 initial
conditions of the dataset-size study); grid RMSE and fixed-point residual are those of the last epoch, as in the
paper.

Usage: python -m experiments.quick [--seeds 0 [1 ...]] [--out DIR] [--jobs N] [--device cpu]
Outputs in reproduced/quick/ (or --out): runs/ (JSON and checkpoint of every run), logs/, summary.txt,
grid_rmse.pdf (grid RMSE vs. epoch, as Fig. 3) and vector_field.pdf (QR model, as Fig. 2).
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from fpnode import data, metrics, plotting
from fpnode.integrators import rk4_trajectory
from experiments import REPO_ROOT, rel
from experiments.exp1_dataset_size.evaluate import DT, N_STEPS, SAVE_EVERY, initial_conditions
from experiments.exp1_fixed_epochs.train import RESULTS as PAPER_RESULTS, run_name
from experiments.systems import lotka_volterra as lv

OUT = REPO_ROOT / "reproduced" / "quick"
N_TRAIN = 50_000
MODELS = [("qr", 0.0, "QR-constrained", "tab:blue"), ("free_mlp", 0.0, "Free MLP", "tab:orange"),
          ("penalty_mlp", 1.0, "MLP + penalty (lambda=1)", "#1baf7a")]
COLUMNS = [("grid_rmse", "grid RMSE"), ("local_rmse", "local RMSE"), ("fixed_point_residual", "r_fp"),
           ("jac_err", "Jacobian err."), ("traj_end_error_gmean", "traj. err. t=30")]


def train_all(seeds, runs_dir, logs_dir, jobs, device):
    """The paper's training script, one process per run, `jobs` at a time."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    todo = [(kind, lam, seed) for seed in seeds for kind, lam, _, _ in MODELS]

    def run(job):
        kind, lam, seed = job
        name = run_name(kind, N_TRAIN, lam, seed)
        cmd = [sys.executable, "-m", "experiments.exp1_fixed_epochs.train", "--model", kind, "--lam", f"{lam:g}",
               "--seed", str(seed), "--schedule", "cosine", "--n-train", str(N_TRAIN), "--device", device,
               "--out-dir", str(runs_dir)]
        with open(logs_dir / f"{name}.log", "w") as log:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "MPLBACKEND": "Agg"})
        print(f"  finished {name}", flush=True)

    print(f"training {len(todo)} runs, {jobs} at a time (about 3 min per run on one CPU core; runs already in "
          f"{rel(runs_dir)} are reused)", flush=True)
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(run, todo))


def evaluate(kind, lam, seed, runs_dir, grid, local, x0, traj_true):
    name = run_name(kind, N_TRAIN, lam, seed)
    logged = json.load(open(runs_dir / f"{name}.json"))
    model = lv.build_model(kind, seed)
    model.load_state_dict(torch.load(runs_dir / f"{name}.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        F = model.field_fn()
        e = metrics.error_norms(F, lv.field, grid)
        traj = rk4_trajectory(F, x0, DT, N_STEPS, SAVE_EVERY)
    _, J = metrics.numpy_field_and_jacobian(model)
    end_err = (traj[-1] - traj_true[-1]).norm(dim=1).clamp_min(1e-300)
    return {"grid_rmse": logged["history"]["grid_rmse"][-1], "local_rmse": metrics.rms(e[local]),
            "fixed_point_residual": logged["history"]["fixed_point_residual"][-1],
            "jac_err": max(float(np.linalg.norm(J(p) - lv.jacobian(p), 2)) for p in lv.EQUILIBRIA.double().numpy()),
            "traj_end_error_gmean": float(10 ** end_err.log10().mean()),
            "history": logged["history"]["grid_rmse"]}


def paper_value(kind, lam, seed, key):
    """The same metric at the last epoch in the stored runs of the paper (GPU), if available."""
    path = PAPER_RESULTS / "cosine" / f"{run_name(kind, N_TRAIN, lam, seed)}.json"
    return json.load(open(path))["history"][key][-1] if path.exists() else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--jobs", type=int, default=None, help="parallel runs (default: all, up to the CPU count)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    runs_dir = out / "runs"
    n_runs = len(args.seeds) * len(MODELS)
    train_all(args.seeds, runs_dir, out / "logs", args.jobs or min(n_runs, os.cpu_count() or 1), args.device)

    grid = data.grid_points(lv.DOMAIN, lv.GRID_RES)
    local = metrics.near(grid, lv.EQUILIBRIA, lv.LOCAL_RADIUS)
    x0 = initial_conditions(torch.device("cpu"))
    traj_true = rk4_trajectory(lambda x: lv.field(x.double()), x0.double(), DT, N_STEPS, SAVE_EVERY)
    res = {kind: [evaluate(kind, lam, s, runs_dir, grid, local, x0, traj_true) for s in args.seeds]
           for kind, lam, _, _ in MODELS}
    gm = lambda kind, key: metrics.geometric_mean([r[key] for r in res[kind]])
    gm_paper = lambda kind, lam, key: metrics.geometric_mean([paper_value(kind, lam, s, key) for s in args.seeds])

    lines = [f"Experiment 1 (Section 5.2), N_train = 5e4, 400 epochs, cosine schedule, seeds {args.seeds}, "
             f"device {args.device}" + ("; geometric means over the seeds" if len(args.seeds) > 1 else ""),
             "", f"{'':26s}" + "".join(f"{label:>16s}" for _, label in COLUMNS)
             + f"{'paper: grid RMSE':>19s}{'paper: r_fp':>13s}"]
    for kind, lam, label, _ in MODELS:
        lines.append(f"{label:26s}" + "".join(f"{gm(kind, key):16.2e}" for key, _ in COLUMNS)
                     + f"{gm_paper(kind, lam, 'grid_rmse'):19.2e}{gm_paper(kind, lam, 'fixed_point_residual'):13.2e}")
    lines += ["", "Ratio to the QR-constrained model:"]
    ratio = lambda r: f"x{r:.1f}" if r < 100 else f"x{r:.0f}"
    for kind, _, label, _ in MODELS[1:]:
        lines.append(f"  {label:26s}" + "  ".join(f"{name} {ratio(gm(kind, key) / gm('qr', key))}"
                                                  for key, name in COLUMNS))
    lines += ["", "The paper runs were trained on a GPU, so the retrained values can differ slightly from them, most",
              "visibly the fixed-point residual of the QR-constrained model, which is at float32 round-off level."]
    summary = "\n".join(lines)
    print("\n" + summary)
    (out / "summary.txt").write_text(summary + "\n")

    with plotting.paper_style():
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        for kind, _, label, color in MODELS:
            log_hist = np.log10(np.array([r["history"] for r in res[kind]]))
            epochs = np.arange(1, log_hist.shape[1] + 1)
            ax.plot(epochs, 10 ** log_hist.mean(axis=0), color=color, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Grid RMSE")
        ax.legend(frameon=False)
        plotting.style_axes(ax, "major")
        fig.tight_layout()
        plotting.save(fig, out / "grid_rmse.pdf", bbox_inches="tight")
        plt.close(fig)
    model = lv.build_model("qr", args.seeds[0])
    model.load_state_dict(torch.load(runs_dir / f"{run_name('qr', N_TRAIN, 0, args.seeds[0])}.pt", map_location="cpu"))
    model.eval()
    plotting.plot_field_comparison(lv.field, model, lv.DOMAIN, lv.EQUILIBRIA.numpy(), out / "vector_field.pdf")
    for name in ("summary.txt", "grid_rmse.pdf", "vector_field.pdf"):
        print("saved", rel(out / name))


if __name__ == "__main__":
    main()
