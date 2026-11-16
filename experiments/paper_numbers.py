"""
Every statistic quoted in the text for the Lotka-Volterra experiments, from the stored results (no training).
Over the seeds: geometric mean (GM) and geometric standard deviation (GSD = 10^std(log10 x), ddof = 1), i.e. the
center and the half-width of the bands of the figures; wall-clock times as arithmetic mean +/- std.
  Experiment 1 (QR, N = 5e4, cosine, 5 seeds)            exp1_fixed_epochs/results/cosine
  comparison during training (N = 5e4, cosine)           same
  check with a constant learning rate (lr 1e-3)          exp1_fixed_epochs/results/constant
  dependence on the dataset size                          exp1_dataset_size/results
  training on trajectories (Appendix C)                   appC_trajectories/results

Usage: python -m experiments.paper_numbers
"""
import json

import numpy as np

from fpnode.plotting import METHOD_STYLES
from .appC_trajectories import common as td
from .exp1_dataset_size import plot as DS
from .exp1_dataset_size.evaluate import METHODS as DS_METHODS, N_VALUES
from .exp1_fixed_epochs.plot import load_histories
from .exp1_fixed_epochs.train import RESULTS as FE_RESULTS, run_name as fe_run

log = lambda a: np.log10(np.asarray(a, dtype=float))
gm = lambda a: float(10 ** log(a).mean())
gsd = lambda a: float(10 ** log(a).std(ddof=1))
fmt = lambda a: f"{gm(a):.3e} x/ {gsd(a):.3f}"
KEYS = ["qr", "free_mlp", "pen_1", "pen_10", "pen_50"]


def exp1():
    runs = [json.load(open(FE_RESULTS / "cosine" / f"{fe_run('qr', 50_000, 0, s)}.json")) for s in range(5)]
    print("== Experiment 1 (QR, N=5e4, cosine, 5 seeds)")
    for k in ("grid_rmse", "max_error", "fixed_point_residual"):
        a = [r["final"][k] for r in runs]
        print(f"  {k:22s} GM {fmt(a)}   (arith {np.mean(a):.3e} +- {np.std(a, ddof=1):.2e})")
    t = [r["train_seconds"] / 60 for r in runs]
    print(f"  train minutes          arith {np.mean(t):.3f} +- {np.std(t, ddof=1):.3f}, GM {fmt(t)}")


def training_comparison():
    h = load_histories("cosine")
    print("== Comparison during training, N=5e4 cosine, last epoch (grid RMSE, r_fp)")
    for k in KEYS:
        g, r = h[k]["grid_rmse"][:, -1], h[k]["fixed_point_residual"][:, -1]
        print(f"  {k:9s} grid {fmt(g)}  r_fp GM {gm(r):.2e}")
    curve = {k: 10 ** log(h[k]["grid_rmse"]).mean(axis=0) for k in KEYS}
    for k in KEYS:
        below = np.nonzero(curve[k] <= 1e-1)[0]
        print(f"  {k:9s} first epoch with GM grid RMSE <= 1e-1: {below[0] + 1 if len(below) else 'never'}")
    lower = np.all([curve["qr"] < curve[k] for k in KEYS[1:]], axis=0)
    print(f"  QR GM below every baseline from epoch {np.argmax(np.cumprod(lower[::-1])[::-1]) + 1} on "
          f"(first epoch where it holds: {np.argmax(lower) + 1})")
    for k in ("pen_1", "pen_10", "pen_50"):
        print(f"  {k} / free grid ratio of GMs at last epoch: {curve[k][-1] / curve['free_mlp'][-1]:.2f}")
    for k in ("pen_1", "pen_10", "pen_50"):
        r = h[k]["fixed_point_residual"][:, -1]
        print(f"  {k} / QR residual ratio of GMs: {gm(r) / gm(h['qr']['fixed_point_residual'][:, -1]):.0f}")


def robustness():
    h = load_histories("constant")
    print("== Robustness N=5e4, constant lr, last epoch")
    for k in KEYS:
        g, r = h[k]["grid_rmse"][:, -1], h[k]["fixed_point_residual"][:, -1]
        late = h[k]["fixed_point_residual"][:, -100:]
        print(f"  {k:9s} grid {fmt(g)} (arith {g.mean():.2e} +- {g.std(ddof=1):.1e}) | r_fp last GM {gm(r):.1e}, "
              f"last 100 epochs min {late.min():.1e} max {late.max():.1e}")


def dataset_size():
    results, traj = DS.load()
    jac = json.load(open(DS.RESULTS / "jacobian_metrics.json"))
    v = lambda m, n, f: DS.values(results, m, n, f)
    print("== Dataset size: GM per method and N")
    for f in ("grid_rmse", "local_rmse", "fixed_point_residual", "jac_err", "traj_end_error_gmean"):
        print(f"  {f}")
        for n in N_VALUES:
            print(f"    N={n:>6} " + "  ".join(f"{m[:4]} {gm(v(m, n, f)):.2e}" for m in DS_METHODS))
    print("  grid RMSE ratio baseline / QR:")
    for n in N_VALUES:
        print(f"    N={n:>6} " + " ".join(f"{gm(v(m, n, 'grid_rmse')) / gm(v('qr', n, 'grid_rmse')):.2f}" for m in DS_METHODS[1:]))
    print("  penalty, aug vs free (ratio free / method, GM), grid / local / r_fp / traj:")
    for n in N_VALUES:
        print(f"    N={n:>6} " + " | ".join(
            f"{m[:4]} " + " ".join(f"{gm(v('free_mlp', n, f)) / gm(v(m, n, f)):.2f}"
                                   for f in ("grid_rmse", "local_rmse", "fixed_point_residual", "traj_end_error_gmean"))
            for m in ("penalty_mlp", "aug_mlp")))
    print("  local RMSE free / QR:", " ".join(f"{gm(v('free_mlp', n, 'local_rmse')) / gm(v('qr', n, 'local_rmse')):.1f}" for n in N_VALUES))
    print("  Jacobian baseline / QR:", " ".join(
        f"{min(gm(v(m, n, 'jac_err')) for m in DS_METHODS[1:]) / gm(v('qr', n, 'jac_err')):.2f}-"
        f"{max(gm(v(m, n, 'jac_err')) for m in DS_METHODS[1:]) / gm(v('qr', n, 'jac_err')):.2f}" for n in N_VALUES))
    print("  relative eigenvalue error GM: " + " | ".join(
        f"{m[:4]} " + " ".join(f"{gm([r['max_eig_rel'] for r in jac[f'{m}_n{n}']]):.1e}" for n in N_VALUES) for m in DS_METHODS))
    print("  QR at N=200 vs every baseline at N=5e4 (GM):")
    for f in ("local_rmse", "fixed_point_residual", "traj_end_error_gmean", "jac_err"):
        q = gm(v("qr", 200, f))
        best = min(gm(v(m, 50_000, f)) for m in DS_METHODS[1:])
        print(f"    {f:22s} QR@200 {q:.2e} vs best baseline@5e4 {best:.2e} -> {'lower' if q < best else 'NOT lower'}")


def trajectories():
    results = json.load(open(td.RESULTS / "eval_metrics.json"))
    methods = ["qr", "free_mlp", "penalty_mlp"]
    g = lambda m, f: gm([r[f] for r in results[m]])
    sel = json.load(open(td.RESULTS / "selection.json"))
    print(f"== Training on trajectories (Appendix C); penalty lambda chosen on validation: {results['selection']['lambda']:g} "
          f"({sel['gm_best_val_mse']})")
    for m in methods:
        rows = results[m]
        print(f"  {METHOD_STYLES[m][0]:16s} val {g(m, 'val_mse'):.2e} | traj (0,10] {g(m, 'traj_rms_horizon_gmean'):.2e} "
              f"(10,30] {g(m, 'traj_rms_beyond_gmean'):.2e} | t=30 {g(m, 'traj_end_error_gmean'):.2e} "
              f"t=100 {g(m, 'traj_long_error_gmean'):.2e} | outcome acc {100 * np.mean([r['outcome_accuracy'] for r in rows]):.2f}% "
              f"(min {100 * min(r['outcome_accuracy'] for r in rows):.1f}%) | r_fp {g(m, 'fixed_point_residual'):.1e} "
              f"| jac max {g(m, 'jac_err_max'):.2e} | local {g(m, 'local_rmse'):.2e} "
              f"| field: IC box {g(m, 'grid_rmse_ic_box'):.2e} whole box {g(m, 'grid_rmse_box'):.2e} near data {g(m, 'grid_rmse_near_data'):.2e} "
              f"| stop: {sorted(set(r['stop_reason'] for r in rows))}, median steps {int(np.median([r['stop_step'] for r in rows]))}")
    q = {m: g(m, "traj_rms_beyond_gmean") for m in methods}
    h = {m: g(m, "traj_rms_horizon_gmean") for m in methods}
    print("  ratios baseline/QR: within the data range " + ", ".join(f"{m} {h[m] / h['qr']:.0f}x" for m in methods[1:])
          + "; beyond it " + ", ".join(f"{m} {q[m] / q['qr']:.0f}x" for m in methods[1:]))
    print("  seeds where QR beats each baseline (beyond the data range): " + ", ".join(
        f"{m} {sum(a['traj_rms_beyond_gmean'] < b['traj_rms_beyond_gmean'] for a, b in zip(results['qr'], results[m]))}/10"
        for m in methods[1:]))


if __name__ == "__main__":
    exp1()
    training_comparison()
    robustness()
    dataset_size()
    trajectories()
