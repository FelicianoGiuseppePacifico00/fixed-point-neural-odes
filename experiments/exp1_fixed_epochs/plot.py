"""
Figs. 2-4 of the paper (Section 5.2), from the stored runs (no training):
  - Vector_field_approximation_50k.pdf (Fig. 2): target Lotka-Volterra field, field learned by the
    QR-constrained model (N_train = 5e4, cosine schedule, seed 0) and log-scale error map;
  - LV_grid_rmse.pdf, LV_fp_residual.pdf (Figs. 3-4): grid RMSE and fixed-point residual vs. epoch for the
    QR-constrained model, the free MLP and the MLP + penalty (lambda = 1, 10, 50), N_train = 5e4, cosine
    schedule, 5 seeds (geometric mean and +/- 1 log-std band).

Usage: python -m experiments.exp1_fixed_epochs.plot [--results DIR] [--figures DIR] [--device cpu]
(defaults: results/ and figures/ of this folder).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from fpnode import get_device, plotting
from experiments import rel
from experiments.systems import lotka_volterra as lv
from .train import RESULTS, run_name

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
N_TRAIN = 50_000
EPOCHS = 400
SEEDS = [0, 1, 2, 3, 4]
FIG2_SEED = 0
# key: (model type, lambda, label, color)
SERIES = {
    "qr": ("qr", 0, "QR-constrained", "tab:blue"),
    "free_mlp": ("free_mlp", 0, "Free MLP", "tab:orange"),
    "pen_1": ("penalty_mlp", 1, r"MLP + penalty, $\lambda=1$", "#1baf7a"),
    "pen_10": ("penalty_mlp", 10, r"MLP + penalty, $\lambda=10$", "#4a3aa7"),
    "pen_50": ("penalty_mlp", 50, r"MLP + penalty, $\lambda=50$", "#e34948"),
}
CURVES = [("grid_rmse", "Grid RMSE", "LV_grid_rmse.pdf"),
          ("fixed_point_residual", r"Fixed-point residual $r_{\mathrm{fp}}$", "LV_fp_residual.pdf")]


def load_histories(schedule="cosine", n_train=N_TRAIN, results=RESULTS):
    """series key -> metric -> array [seeds, epochs]."""
    hist = {}
    for key, (mt, lam, _, _) in SERIES.items():
        runs = [json.load(open(Path(results) / schedule / f"{run_name(mt, n_train, lam, s)}.json"))["history"]
                for s in SEEDS]
        hist[key] = {m: np.array([r[m] for r in runs]) for m in ("fixed_point_residual", "grid_rmse")}
        assert hist[key]["grid_rmse"].shape == (len(SEEDS), EPOCHS)
    return hist


def plot_vector_field(results=RESULTS, figures=FIGURES, device=torch.device("cpu"), close=True):
    """Fig. 2 (needs the checkpoint results/cosine/qr_n50000_lam0_seed0.pt). Returns the figure."""
    model = lv.build_model("qr", FIG2_SEED, device)
    path = Path(results) / "cosine" / f"{run_name('qr', N_TRAIN, 0, FIG2_SEED)}.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    out = Path(figures) / "Vector_field_approximation_50k.pdf" if figures is not None else None
    return plotting.plot_field_comparison(lv.field, model, lv.DOMAIN, lv.EQUILIBRIA.numpy(), out, device=device,
                                          close=close)


def plot_training_curves(results=RESULTS, figures=FIGURES, close=True):
    """Figs. 3-4. Returns the two figures."""
    hist = load_histories(results=results)
    x = np.arange(1, EPOCHS + 1)
    figs = []
    with plotting.paper_style():
        for metric, ylabel, fname in CURVES:
            fig, ax = plt.subplots(figsize=(6.4, 3.9))
            for key, (_, _, label, color) in SERIES.items():
                st = plotting.log_stats(hist[key][metric])
                plotting.band(ax, x, st["log_mean"], st["log_std"], color, label=label)
            ax.set_yscale("log")
            ax.set_xlim(0, EPOCHS)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", frameon=False)
            plotting.style_axes(ax, "major")
            fig.tight_layout()
            if figures is not None:
                plotting.save(fig, Path(figures) / fname, bbox_inches="tight")
            if close:
                plt.close(fig)
            figs.append(fig)
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    Path(args.figures).mkdir(parents=True, exist_ok=True)
    plot_vector_field(args.results, args.figures, get_device(args.device))
    plot_training_curves(args.results, args.figures)
    for name in ["Vector_field_approximation_50k.pdf"] + [fname for _, _, fname in CURVES]:
        print("saved", rel(Path(args.figures) / name))


if __name__ == "__main__":
    main()
