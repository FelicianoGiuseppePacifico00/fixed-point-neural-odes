"""
Fig. 8 and Tables 2-3 of the paper (Appendix C), from the evaluation files eval_metrics.json and
eval_trajectories.npz written by evaluate.py (no training):
  - LV_traj_training_error.pdf: trajectory error vs. time up to t = 30, the range beyond the data (t > T_OBS)
    shaded; geometric mean over seeds and initial conditions and +/- 1 log-std band;
  - table.tex (Table 2): geometric mean x/ geometric standard deviation over the seeds of the trajectory error
    within and beyond the time range of the data, the fixed-point residual and the Jacobian error (max over
    the equilibria);
  - table_equilibria.tex (Table 3): per equilibrium, share of training observations within 0.25, |F(x_l)|,
    Jacobian error (geometric means over the seeds) and runs with the correct stability type.

Usage: python -m experiments.appC_trajectories.plot [--results DIR] [--figures DIR]
(defaults: results/ and figures/ of this folder).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fpnode import plotting
from fpnode.plotting import METHOD_STYLES
from experiments import rel
from experiments.systems import lotka_volterra as lv
from . import common as td

FIGURES = Path(__file__).resolve().parent / "figures"
T_EVAL = 30.0
METHODS = ["qr", "free_mlp", "penalty_mlp"]
TABLE_FIELDS = ("traj_rms_horizon_gmean", "traj_rms_beyond_gmean", "fixed_point_residual", "jac_err_max")


def load(results_dir=td.RESULTS):
    results_dir = Path(results_dir)
    with open(results_dir / "eval_metrics.json") as fh:
        results = json.load(fh)
    return results, dict(np.load(results_dir / "eval_trajectories.npz"))


def plot_trajectory_error(traj, figures=FIGURES, close=True):
    with plotting.paper_style():
        t = traj["t"]
        keep = (t > 0) & (t <= T_EVAL + 1e-9)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.axvspan(td.T_OBS, T_EVAL, color="0.92", linewidth=0, zorder=0)
        for method in METHODS:
            label, color, ls = METHOD_STYLES[method]
            plotting.band(ax, t[keep], traj[f"{method}_log_mean"][keep], traj[f"{method}_log_std"][keep], color, ls,
                          label, linewidth=1.3)
        ax.set_yscale("log")
        ax.set_xlim(0, T_EVAL)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(r"$\|x_\theta(t)-x(t)\|_2$")
        ax.text(0.5 * (td.T_OBS + T_EVAL), 0.97, "beyond the time range of the data", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="0.35")
        plotting.style_axes(ax, "major")
        ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16), columnspacing=1.4, handlelength=2.0)
        if figures is not None:
            plotting.save(fig, Path(figures) / "LV_traj_training_error.pdf", bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    return fig


def table(results):
    vals = {m: {f: [r[f] for r in results[m]] for f in TABLE_FIELDS} for m in METHODS}
    return "\n".join(plotting.table_rows(vals, METHODS, TABLE_FIELDS))


def table_equilibria(results):
    lines = []
    for l, (name, kind) in enumerate(zip(lv.EQUILIBRIUM_NAMES, lv.EQUILIBRIUM_TYPES)):
        obs = 100 * np.mean([r["per_equilibrium"][l]["obs_fraction_near"] for r in results["qr"]])
        cells = [f"${name}$ & {kind} & ${obs:.2f}\\%$"]
        for f in ("residual_float64", "jac_err"):
            cells += [plotting.short_sci(plotting.geo([r["per_equilibrium"][l][f] for r in results[m]])[0]) for m in METHODS]
        cells += [f"{sum(r['per_equilibrium'][l]['eig_signs_ok'] for r in results[m])}/{len(results[m])}" for m in METHODS]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(td.RESULTS))
    ap.add_argument("--figures", default=str(FIGURES))
    args = ap.parse_args(argv)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    results, traj = load(args.results)
    plot_trajectory_error(traj, figures)
    tex, tex_eq = table(results), table_equilibria(results)
    (figures / "table.tex").write_text(tex + "\n")
    (figures / "table_equilibria.tex").write_text(tex_eq + "\n")
    print(tex + "\n\n" + tex_eq)
    for name in ("LV_traj_training_error.pdf", "table.tex", "table_equilibria.tex"):
        print("saved", rel(figures / name))


if __name__ == "__main__":
    main()
