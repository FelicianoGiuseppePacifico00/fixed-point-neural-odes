"""
Evaluation of the trajectory-training runs (Appendix C) on the best-validation checkpoints. The penalty
weight is chosen as in the dataset-size study: lowest geometric mean over seeds 0-2 of the best validation MSE;
the 10 seeds of that lambda are reported. Per run:
  - trajectories from the 500 initial conditions of the dataset-size study (uniform in [0.2, 3.5]^2): learned
    model with its training integrator (RK4, DT_MODEL, float32), true field with the float64 reference, up to
    t = 100, compared every DT_OBS: per initial condition the RMS error over (0, T_OBS] (time range of the
    data) and over (T_OBS, 30] (beyond it), then the geometric mean over the initial conditions; error at
    t = 30 and t = 100; outcome accuracy (same nearest equilibrium at t = 30);
  - vector field on the 250 x 250 grid of [0, 4]^2: RMSE on the whole box, on the box of the initial
    conditions, near the training observations of the seed (0.1) and near the equilibria (0.25); residual;
  - for each equilibrium: Jacobian error ||DF_theta - DF||_2, agreement of the signs of the eigenvalues
    (stability type), distance of the nearest learned equilibrium (Newton from x_l), float64 residual.

Usage: python -m experiments.appC_trajectories.evaluate [--results DIR] [--out DIR] [--device cpu]
Outputs in results/ (or --out): selection.json, eval_metrics.json, eval_trajectories.npz, logs/eval.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fpnode import get_device, metrics
from fpnode.integrators import rk4_observe
from experiments.systems import lotka_volterra as lv
from experiments.exp1_dataset_size.evaluate import IC_BOX, initial_conditions
from . import common as td

SELECT_SEEDS = [0, 1, 2]
T_EVAL = 30.0
T_LONG = 100.0                     # check that nothing changes after T_EVAL
NEAR_DATA = 0.1
FP = lv.EQUILIBRIA.double().numpy()


def load(name, results_dir):
    return json.load(open(Path(results_dir) / f"{name}.json"))


def select_lambda(lines, results_dir):
    gm, dropped = {}, set()
    for lam in td.LAMBDAS:
        cfgs = [load(td.run_name("penalty_mlp", lam, s), results_dir)["config"] for s in SELECT_SEEDS]
        gm[lam] = float(10 ** np.mean(np.log10([c["best_val_mse"] for c in cfgs])))
        if any(c["stop_reason"] == "dropped" for c in cfgs):
            dropped.add(lam)
    chosen = min(gm, key=gm.get)
    lines.append("penalty selection (geometric mean of the best validation MSE over seeds "
                 f"{SELECT_SEEDS}): " + ", ".join(f"lambda={k:g}: {v:.3e}" + (" (dropped before convergence)" if k in dropped else "")
                                                   for k, v in gm.items()) + f" -> {chosen:g}")
    return chosen, gm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(td.RESULTS))
    ap.add_argument("--out", default=None, help="default: --results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    results_dir = Path(args.results)
    out = Path(args.out) if args.out else results_dir
    (out / "logs").mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    lines = []
    chosen, gm_sel = select_lambda(lines, results_dir)
    json.dump({"lambda": chosen, "gm_best_val_mse": {f"{k:g}": v for k, v in gm_sel.items()},
               "select_seeds": SELECT_SEEDS}, open(out / "selection.json", "w"), indent=1)
    methods = {"qr": 0.0, "free_mlp": 0.0, "penalty_mlp": chosen}

    eq = lv.EQUILIBRIA.to(device).float()
    pts = td.make_setup(device).grid
    trajectories = torch.load(results_dir / "data.pt")
    (lo, hi), _ = IC_BOX
    ic_box = ((pts >= lo) & (pts <= hi)).all(1)
    F_true = lv.field(pts)
    dist_eq = torch.cdist(pts, eq)
    local_masks = [dist_eq[:, l] <= lv.LOCAL_RADIUS for l in range(len(FP))]
    x0 = initial_conditions(torch.device("cpu"))
    n_obs_long = int(round(T_LONG / td.DT_OBS)) + 1
    i_eval = int(round(T_EVAL / td.DT_OBS))
    traj_true = td.reference(x0.double(), n_obs=n_obs_long, device=device)                 # [n_ic, n_obs, 2]
    end_true = metrics.nearest(traj_true[:, i_eval], lv.EQUILIBRIA)
    assert torch.equal(end_true, metrics.nearest(traj_true[:, -1], lv.EQUILIBRIA))
    t = td.DT_OBS * np.arange(n_obs_long)
    in_horizon = (t > 0) & (t <= td.T_OBS + 1e-9)
    beyond = (t > td.T_OBS + 1e-9) & (t <= T_EVAL + 1e-9)
    rms_gmean = lambda e, mask: float(10 ** e[:, mask].pow(2).mean(dim=1).sqrt().clamp_min(1e-300).log10().mean())
    lines.append(f"true flow: final counts {torch.bincount(end_true, minlength=4).tolist()}, "
                 f"max distance from the end equilibrium at t={T_EVAL:g}: "
                 f"{float((traj_true[:, i_eval] - lv.EQUILIBRIA.double()[end_true]).norm(dim=1).max()):.1e}")

    results, traj_stats = {"selection": {"lambda": chosen}}, {"t": t}
    for method, lam in methods.items():
        rows, errs = [], []
        for seed in td.SEEDS:
            name = td.run_name(method, lam, seed)
            obs = trajectories["train"][td.SEEDS.index(seed)].reshape(-1, 2).float().to(device)
            near_data = torch.cdist(pts, obs).min(dim=1).values <= NEAR_DATA
            obs_near_eq = (torch.cdist(obs, eq) <= lv.LOCAL_RADIUS).float().mean(dim=0).tolist()
            logged = load(name, results_dir)
            model = lv.build_model(method, seed, device)
            model.load_state_dict(torch.load(results_dir / "checkpoints" / f"{name}_best.pt", map_location=device))
            model.eval()
            with torch.no_grad():
                Fm = model.field_fn()
                e = (Fm(pts) - F_true).norm(dim=1)
                traj = rk4_observe(Fm, x0.to(device).float(), td.DT_MODEL, n_obs_long, td.DT_OBS).double().cpu()
            err_t = (traj - traj_true).norm(dim=2)                                     # [n_ic, n_obs]
            errs.append(err_t.numpy())
            end_err = err_t[:, i_eval].clamp_min(1e-300)
            Fn, Jn = metrics.numpy_field_and_jacobian(model)
            per_eq = []
            for l, p in enumerate(FP):
                Jl, Jt = Jn(p), lv.jacobian(p)
                ll, lt = np.linalg.eigvals(Jl), np.linalg.eigvals(Jt)
                q = metrics.newton(Fn, Jn, p.copy(), iters=50, tol=1e-14)
                per_eq.append({"obs_fraction_near": obs_near_eq[l],
                               "local_rmse": metrics.rms(e[local_masks[l]]),
                               "jac_err": float(np.linalg.norm(Jl - Jt, 2)),
                               "eig_signs_ok": bool(np.array_equal(np.sort(np.sign(ll.real)), np.sort(np.sign(lt.real)))),
                               "eq_displacement": float(np.linalg.norm(q - p)),
                               "residual_float64": float(np.linalg.norm(Fn(p)))})
            rows.append({
                "seed": seed, "lambda": lam,
                "grid_rmse_box": metrics.rms(e),
                "grid_max_error_box": float(e.max()),
                "grid_rmse_ic_box": metrics.rms(e[ic_box]),
                "grid_rmse_near_data": metrics.rms(e[near_data]),
                "near_data_fraction": float(near_data.float().mean()),
                "local_rmse": metrics.rms(e[torch.stack(local_masks).any(0)]),
                "fixed_point_residual": logged["final"]["fixed_point_residual"],
                "jac_err_max": max(q["jac_err"] for q in per_eq),
                "eq_displacement_max": max(q["eq_displacement"] for q in per_eq),
                "per_equilibrium": per_eq,
                "traj_rms_horizon_gmean": rms_gmean(err_t, in_horizon),
                "traj_rms_beyond_gmean": rms_gmean(err_t, beyond),
                "traj_long_error_gmean": float(10 ** err_t[:, -1].clamp_min(1e-300).log10().mean()),
                "traj_end_error_gmean": float(10 ** end_err.log10().mean()),
                "traj_end_error_mean": float(end_err.mean()),
                "outcome_accuracy": float((metrics.nearest(traj[:, i_eval], lv.EQUILIBRIA) == end_true).float().mean()),
                "val_mse": logged["final"]["val_mse"], "train_mse": logged["final"]["train_mse"],
                "best_step": logged["config"]["best_step"], "stop_step": logged["config"]["stop_step"],
                "stop_reason": logged["config"]["stop_reason"],
            })
        results[method] = rows
        errs = np.stack(errs)                                                           # [seeds, n_ic, n_obs]
        log_e = np.log10(np.clip(errs, 1e-300, None))
        traj_stats[f"{method}_log_mean"] = log_e.mean(axis=(0, 1))
        traj_stats[f"{method}_log_std"] = log_e.std(axis=(0, 1), ddof=1)
        traj_stats[f"{method}_mean"] = errs.mean(axis=(0, 1))
        gmean = lambda f: 10 ** np.mean([np.log10(max(r[f], 1e-300)) for r in rows])
        gmean_eq = lambda l, f: 10 ** np.mean([np.log10(max(r["per_equilibrium"][l][f], 1e-300)) for r in rows])
        lines.append(f"{method:12s} lam={lam:g}: val {gmean('val_mse'):.2e} | traj t<={td.T_OBS:g} {gmean('traj_rms_horizon_gmean'):.2e} "
                     f"| traj {td.T_OBS:g}<t<={T_EVAL:g} {gmean('traj_rms_beyond_gmean'):.2e} | t={T_EVAL:g} {gmean('traj_end_error_gmean'):.2e} "
                     f"t={T_LONG:g} {gmean('traj_long_error_gmean'):.2e} | outcome acc {np.mean([r['outcome_accuracy'] for r in rows]):.4f} "
                     f"| grid box {gmean('grid_rmse_box'):.2e} IC box {gmean('grid_rmse_ic_box'):.2e} near data {gmean('grid_rmse_near_data'):.2e} | local {gmean('local_rmse'):.2e} | r_fp {gmean('fixed_point_residual'):.1e} "
                     f"| jac max {gmean('jac_err_max'):.2e} | eq displ max {gmean('eq_displacement_max'):.1e} "
                     f"| steps {int(np.median([r['stop_step'] for r in rows]))}, converged "
                     f"{sum(r['stop_reason'] == 'min_lr' for r in rows)}/{len(rows)}")
        for l, nm in enumerate(lv.EQUILIBRIUM_NAMES):
            lines.append(f"    {nm:6s} train obs within {lv.LOCAL_RADIUS:g}: {100 * np.mean([r['per_equilibrium'][l]['obs_fraction_near'] for r in rows]):5.2f}%  "
                         f"|F(x_bar)| {gmean_eq(l, 'residual_float64'):.1e}  local {gmean_eq(l, 'local_rmse'):.2e}  jac {gmean_eq(l, 'jac_err'):.2e}  "
                         f"displacement {gmean_eq(l, 'eq_displacement'):.1e}  "
                         f"eig signs ok {sum(r['per_equilibrium'][l]['eig_signs_ok'] for r in rows)}/{len(rows)}")
    with open(out / "eval_metrics.json", "w") as fh:
        json.dump(results, fh, indent=1)
    np.savez(out / "eval_trajectories.npz", **traj_stats)
    (out / "logs" / "eval.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
