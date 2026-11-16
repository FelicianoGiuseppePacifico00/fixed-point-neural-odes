"""
Figs. 5-6 and Table 1 of the paper (Section 5.2, dependence on the dataset size), from the evaluation files
eval_metrics.json, eval_trajectories.npz and jacobian_metrics.json written by evaluate.py (no training):
  - LV_dataset_size_grid_rmse.pdf: grid RMSE vs. N_data (geometric mean and +/- 1 log-std band over seeds);
  - LV_trajectory_error.pdf: trajectory error vs. time, one panel per N_data in {200, 1e3, 1e4, 5e4}
    (statistics over seeds and initial conditions);
  - table.tex: rows of Table 1, geometric mean x/ geometric standard deviation over the seeds of the grid RMSE,
    local RMSE, fixed-point residual, Jacobian error at the equilibria and trajectory error at t = 30; bold:
    lowest geometric mean of each column within an N_data block.

Usage: python -m experiments.exp1_dataset_size.plot [--results DIR] [--figures DIR]
(defaults: results/ and figures/ of this folder).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator

from fpnode import plotting
from fpnode.plotting import METHOD_STYLES
from experiments import rel
from .evaluate import METHODS, N_VALUES
from .train import RESULTS

FIGURES = Path(__file__).resolve().parent / "figures"
TABLE_N = [200, 1_000, 10_000, 50_000]
N_LABEL = {200: "200", 500: "500", 1_000: "1000", 10_000: r"$10^4$", 50_000: r"$5\times10^4$"}
TABLE_FIELDS = ("grid_rmse", "local_rmse", "fixed_point_residual", "jac_err", "traj_end_error_gmean")


def load(results_dir=RESULTS):
    """Per-run metrics (with the Jacobian error max_l jac_err added) and the trajectory statistics."""
    results_dir = Path(results_dir)
    with open(results_dir / "eval_metrics.json") as fh:
        results = json.load(fh)
    with open(results_dir / "jacobian_metrics.json") as fh:
        jac = json.load(fh)
    for key, rows in results.items():
        for row, jr in zip(rows, jac[key], strict=True):
            assert row["seed"] == jr["seed"] and row["tag"] == jr["tag"]
            row["jac_err"] = jr["max_jac_err"]
    return results, np.load(results_dir / "eval_trajectories.npz")


def values(results, method, n, field):
    return np.array([r[field] for r in results[f"{method}_n{n}"]])


def plot_grid_rmse(results, figures=FIGURES, close=True):
    with plotting.paper_style():
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        x = np.array(N_VALUES, dtype=float)
        for method in METHODS:
            label, color, ls = METHOD_STYLES[method]
            log = np.log10(np.array([values(results, method, n, "grid_rmse") for n in N_VALUES]))
            plotting.band(ax, x, log.mean(axis=1), log.std(axis=1, ddof=1), color, ls, label, marker="o", markersize=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(x, [N_LABEL[n] for n in N_VALUES])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"Dataset size $N_{\mathrm{data}}$")
        ax.set_ylabel("Grid RMSE")
        plotting.style_axes(ax, "major")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   columnspacing=1.6, handlelength=2.2)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        if figures is not None:
            plotting.save(fig, Path(figures) / "LV_dataset_size_grid_rmse.pdf", bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    return fig


def plot_trajectory_error(traj, figures=FIGURES, close=True):
    with plotting.paper_style():
        t = traj["t"][1:]  # at t = 0 the error is exactly zero
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex=True, sharey=True)
        for ax, n in zip(axes.flat, TABLE_N):
            for method in METHODS:
                label, color, ls = METHOD_STYLES[method]
                plotting.band(ax, t, traj[f"{method}_n{n}_log_mean"][1:], traj[f"{method}_n{n}_log_std"][1:],
                              color, ls, label, linewidth=1.3)
            ax.set_yscale("log")
            ax.set_xlim(0, t[-1])
            ax.set_title(rf"$N_{{\mathrm{{data}}}}$ = {N_LABEL[n]}", fontsize=10)
            plotting.style_axes(ax, "major")
        for ax in axes[1]:
            ax.set_xlabel("Time $t$")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\|x_\theta(t)-x(t)\|_2$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   columnspacing=1.4, handlelength=2.0)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        if figures is not None:
            plotting.save(fig, Path(figures) / "LV_trajectory_error.pdf", bbox_inches="tight")
    if close:
        plt.close(fig)
    return fig


def table(results):
    acc = [r["outcome_accuracy"] for rows in results.values() for r in rows]
    assert min(acc) == 1.0, "outcome accuracy below 100%: the table should report it"
    lines = []
    for n in TABLE_N:
        lines.append(rf"\multicolumn{{6}}{{l}}{{$N_{{\mathrm{{data}}}}={N_LABEL[n].strip('$')}$}}\\")
        vals = {m: {f: values(results, m, n, f) for f in TABLE_FIELDS} for m in METHODS}
        lines += plotting.table_rows(vals, METHODS, TABLE_FIELDS)
        if n != TABLE_N[-1]:
            lines.append(r"\midrule")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--figures", default=str(FIGURES))
    args = ap.parse_args(argv)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    results, traj = load(args.results)
    plot_grid_rmse(results, figures)
    plot_trajectory_error(traj, figures)
    tex = table(results)
    (figures / "table.tex").write_text(tex + "\n")
    print(tex)
    for name in ("LV_dataset_size_grid_rmse.pdf", "LV_trajectory_error.pdf", "table.tex"):
        print("saved", rel(figures / name))


if __name__ == "__main__":
    main()
