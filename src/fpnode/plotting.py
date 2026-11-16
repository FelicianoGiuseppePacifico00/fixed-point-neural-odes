"""
Plotting and table helpers shared by the experiments.

Statistics over seeds are geometric: lines are the geometric mean, bands +/- 1 standard deviation of
log10 (i.e. the interval [GM / GSD, GM * GSD]), table cells "GM x/ GSD".
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# label, color, line style of each method
METHOD_STYLES = {
    "qr": ("QR-constrained", "tab:blue", "-"),
    "free_mlp": ("Free MLP", "tab:orange", "-"),
    "penalty_mlp": ("MLP + penalty", "#1baf7a", "-"),
    "aug_mlp": ("MLP + augmentation", "#4a3aa7", "--"),
}
PAPER_RC = {"font.size": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 9.5}
PDF_METADATA = {"CreationDate": None}  # reproducible PDF files


def paper_style():
    """Context manager with the font sizes of the paper figures (the global style is restored on exit)."""
    return plt.rc_context(PAPER_RC)


def save(fig, path, **kwargs):
    """savefig with a fixed PDF creation date, so that reruns give identical files."""
    meta = PDF_METADATA if str(path).endswith(".pdf") else None
    fig.savefig(path, metadata=meta, **kwargs)


def style_axes(ax, which="both"):
    ax.grid(True, alpha=0.25, which=which, linewidth=0.6)
    for spine in ax.spines.values():  # closed box around the axes
        spine.set_visible(True)
        spine.set_linewidth(0.9)


def log_stats(arr: np.ndarray, axis=0) -> dict:
    """Mean and std (ddof = 1) of the values and of their log10 along axis."""
    log = np.log10(np.clip(arr, 1e-300, None))
    return {"mean": arr.mean(axis=axis), "std": arr.std(axis=axis, ddof=1),
            "log_mean": log.mean(axis=axis), "log_std": log.std(axis=axis, ddof=1)}


def band(ax, x, log_mean, log_std, color, ls="-", label=None, linewidth=1.4, **line_kwargs):
    """Geometric-mean line and +/- 1 log-std band."""
    ax.plot(x, 10 ** log_mean, color=color, linestyle=ls, linewidth=linewidth, label=label, **line_kwargs)
    ax.fill_between(x, 10 ** (log_mean - log_std), 10 ** (log_mean + log_std), color=color, alpha=0.15, linewidth=0)


# ----------------------------------------------------------------------------- tables
def geo(a):
    """Geometric mean and geometric standard deviation (10^std of log10, ddof = 1)."""
    log = np.log10(np.asarray(a, dtype=float))
    return 10 ** log.mean(), 10 ** log.std(ddof=1)


def gsci(gmean, gsd):
    """LaTeX 'm x 10^e x/ s': mantissa to one decimal, geometric standard deviation to two."""
    e = int(np.floor(np.log10(gmean)))
    if round(gmean / 10 ** e, 1) >= 10:
        e += 1
    return rf"${gmean / 10 ** e:.1f}\times10^{{{e}}}\,{{\times}}/{{\div}}\,{gsd:.2f}$"


def short_sci(x):
    """LaTeX 'm x 10^e' with one decimal."""
    e = int(np.floor(np.log10(x)))
    if round(x / 10.0 ** e, 1) >= 10:
        e += 1
    return rf"${x / 10.0 ** e:.1f}\times10^{{{e}}}$"


def table_rows(values, methods, fields):
    """LaTeX rows 'label & GM x/ GSD & ...' for values[method][field] = per-seed values; in every column the
    lowest geometric mean is bold (ties at the printed precision included)."""
    means = {f: {m: geo(values[m][f])[0] for m in methods} for f in fields}
    printed = lambda x: gsci(x, 1.0).split(r"\,")[0]
    best = {f: {m for m in methods if printed(means[f][m]) == printed(min(means[f].values()))} for f in fields}
    lines = []
    for m in methods:
        cells = []
        for f in fields:
            cell = gsci(*geo(values[m][f]))
            cells.append(rf"{{\boldmath{cell}}}" if m in best[f] else cell)
        lines.append(f"{METHOD_STYLES[m][0]} & " + " & ".join(cells) + r" \\")
    return lines


# ----------------------------------------------------------------------------- vector fields
@torch.no_grad()
def _on_grid(F, box, res, device):
    (x_low, x_high), (y_low, y_high) = box
    xs = np.linspace(x_low, x_high, res)
    ys = np.linspace(y_low, y_high, res)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), device=device, dtype=torch.float32)
    v = F(pts).cpu().numpy()
    return Xg, Yg, v[:, 0].reshape(res, res), v[:, 1].reshape(res, res)


def plot_field_comparison(F_target, F_model, box, points, path=None, device=torch.device("cpu"),
                          stream_res=80, error_res=250, close=True):
    """Three panels on a 2-D box: streamlines of the target and of the learned field, and the log-scale
    error map ||F_model - F_target||; points (e.g. the prescribed equilibria, [C, 2]) are marked. Saved to
    path if given; with close=False the figure is left open (e.g. for a notebook). Returns the figure."""
    pts = np.asarray(points)
    Xg, Yg, Ut, Vt = _on_grid(F_target, box, stream_res, device)
    _, _, Um, Vm = _on_grid(F_model, box, stream_res, device)
    _, _, Ue_t, Ve_t = _on_grid(F_target, box, error_res, device)
    _, _, Ue_m, Ve_m = _on_grid(F_model, box, error_res, device)
    err = np.sqrt((Ue_m - Ue_t) ** 2 + (Ve_m - Ve_t) ** 2)
    (x_low, x_high), (y_low, y_high) = box

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for a, (U, V), title in ((ax[0], (Ut, Vt), "Target field $F$"), (ax[1], (Um, Vm), r"Learned field $F_\theta$")):
        a.streamplot(Xg, Yg, U, V, density=2.0, linewidth=1.0, arrowsize=1.1)
        a.scatter(pts[:, 0], pts[:, 1], s=80, marker="x", color="red")
        a.set_title(title)
        a.set_aspect("equal")
        a.set_xlim(x_low, x_high)
        a.set_ylim(y_low, y_high)
        a.grid(True, alpha=0.3)
    im = ax[2].imshow(err, origin="lower", extent=[x_low, x_high, y_low, y_high], aspect="equal",
                      norm=LogNorm(vmin=max(err.min(), 1e-6), vmax=err.max()))
    ax[2].scatter(pts[:, 0], pts[:, 1], s=80, marker="x", color="red")
    ax[2].set_title(r"Error map $\|F_\theta - F\|$ (log scale)")
    ax[2].set_xlim(x_low, x_high)
    ax[2].set_ylim(y_low, y_high)
    cbar = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.set_label(r"$\|F_\theta - F\|$")
    if path is not None:
        save(fig, path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return fig
