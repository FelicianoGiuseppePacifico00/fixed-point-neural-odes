"""
Evaluation of the dataset-size study on the best-validation checkpoints (QR, free MLP, and the penalty /
augmentation with the lambda / rho chosen by select_hyperparameters), 10 seeds each.

Fields and trajectories (eval_metrics.json, eval_trajectories.npz):
  - grid RMSE and max error on the 250 x 250 grid of K, local RMSE within 0.25 of the equilibria;
  - trajectories of the learned field and of the true field from 500 initial conditions uniform in
    [0.2, 3.5]^2 (seed 123), RK4 with dt = 0.01 up to t = 30 (true field in float64), saved every 0.1:
    error at t = 30 (geometric and arithmetic mean over the initial conditions), outcome accuracy (same
    nearest equilibrium at t = 30), and the statistics over seeds and initial conditions of the error vs. time.
Jacobians at the four equilibria (jacobian_metrics.json): DF_theta(x) = -I + A diag(sigma'(Bx + b)) B in float64
from the trained parameters, against the Jacobian of the Lotka-Volterra field. Per run and equilibrium x_l:
  jac_err / jac_rel   ||DF_theta(x_l) - DF(x_l)||_2 (spectral norm), absolute and relative;
  eig_err / eig_rel   max |lambda_theta - lambda| over the eigenvalues (sorted by real part), abs. and rel.;
  stable_ok           same stability type (signs of the real parts);
  *_eq, shift         the same at the equilibrium of the learned field closest to x_l (Newton from x_l)
                      and its distance from x_l.
The column "Jacobian error" of Table 1 is max_l jac_err.

Usage: python -m experiments.exp1_dataset_size.evaluate [--results DIR] [--out DIR] [--device cuda]
(reads the runs in results/, writes to --out, default results/). The stored evaluation was computed on a GPU;
on a CPU the trajectory error of the QR model at t = 30, which is at float32 round-off level, changes slightly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fpnode import data, get_device, metrics
from fpnode.integrators import rk4_trajectory
from experiments import rel
from experiments.systems import lotka_volterra as lv
from .train import RESULTS, run_name

N_VALUES = [200, 500, 1_000, 10_000, 50_000]
SEEDS = list(range(10))
METHODS = ["qr", "free_mlp", "penalty_mlp", "aug_mlp"]
IC_BOX, IC_SEED, N_IC = ((0.2, 3.5), (0.2, 3.5)), 123, 500
DT, T_FINAL, SAVE_EVERY = 0.01, 30.0, 10
N_STEPS = int(round(T_FINAL / DT))
FP = lv.EQUILIBRIA.double().numpy()


def chosen_run(method, n, seed, selection):
    """Run name and tag of the reported run (the selected lambda / rho for the penalty / augmentation)."""
    param = selection[f"{method}_n{n}"]["chosen"] if method in ("penalty_mlp", "aug_mlp") else 0
    run = run_name(method, n, param, seed)
    return run, run.split("_")[-2]


def load_model(method, seed, path, device):
    model = lv.build_model(method, seed, device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def initial_conditions(device):
    return data.uniform_points(N_IC, IC_BOX, IC_SEED).to(device)


def evaluate_fields(results, out, device):
    selection = json.load(open(results / "selection.json"))
    eq = lv.EQUILIBRIA.to(device)
    pts = data.grid_points(lv.DOMAIN, lv.GRID_RES, device)
    local = metrics.near(pts, eq, lv.LOCAL_RADIUS)
    x0 = initial_conditions(device)
    traj_true = rk4_trajectory(lambda x: lv.field(x.double()), x0.double(), DT, N_STEPS, SAVE_EVERY)
    end_true = metrics.nearest(traj_true[-1], lv.EQUILIBRIA)

    out_metrics, traj_stats = {}, {"t": DT * SAVE_EVERY * np.arange(traj_true.shape[0])}
    for n in N_VALUES:
        for method in METHODS:
            rows, errs = [], []
            for seed in SEEDS:
                run, tag = chosen_run(method, n, seed, selection)
                logged = json.load(open(results / f"{run}.json"))
                model = load_model(method, seed, results / "checkpoints" / f"{run}_best.pt", device)
                with torch.no_grad():
                    F = model.field_fn()
                    e = metrics.error_norms(F, lv.field, pts)
                    grid_rmse = metrics.rms(e)
                    assert abs(grid_rmse - logged["final"]["grid_rmse"]) < 1e-2 * grid_rmse, run
                    traj = rk4_trajectory(F, x0, DT, N_STEPS, SAVE_EVERY)
                err_t = (traj - traj_true).norm(dim=2)
                errs.append(err_t.numpy())
                end_err = err_t[-1].double().clamp_min(1e-300)
                cfg = logged["config"]
                rows.append({
                    "seed": seed, "tag": tag,
                    "grid_rmse": grid_rmse,
                    "grid_max_error": float(e.max()),
                    "local_rmse": metrics.rms(e[local]),
                    "fixed_point_residual": logged["final"]["fixed_point_residual"],
                    "outcome_accuracy": float((metrics.nearest(traj[-1], lv.EQUILIBRIA) == end_true).float().mean()),
                    "traj_end_error_gmean": float(10 ** end_err.log10().mean()),
                    "traj_end_error_mean": float(end_err.mean()),
                    "train_mse": logged["final"]["train_mse"], "val_mse": logged["final"]["val_mse"],
                    "best_step": cfg["best_step"], "stop_step": cfg["stop_step"],
                    "stop_reason": cfg["stop_reason"], "last_grid_rmse": logged["final_last"]["grid_rmse"],
                })
            key = f"{method}_n{n}"
            out_metrics[key] = rows
            errs = np.stack(errs)
            log_e = np.log10(np.clip(errs, 1e-300, None))
            traj_stats[f"{key}_mean"] = errs.mean(axis=(0, 2))
            traj_stats[f"{key}_std"] = errs.std(axis=(0, 2), ddof=1)
            traj_stats[f"{key}_log_mean"] = log_e.mean(axis=(0, 2))
            traj_stats[f"{key}_log_std"] = log_e.std(axis=(0, 2), ddof=1)
            gm = lambda f: metrics.geometric_mean([r[f] for r in rows])
            print(f"N={n:>6} {method:12s} grid {gm('grid_rmse'):.2e} local {gm('local_rmse'):.2e} "
                  f"r_fp {gm('fixed_point_residual'):.1e} acc {np.mean([r['outcome_accuracy'] for r in rows]):.4f} "
                  f"traj_end {10 ** traj_stats[f'{key}_log_mean'][-1]:.1e} "
                  f"converged {sum(r['stop_reason'] == 'min_lr' for r in rows)}/10", flush=True)

    with open(out / "eval_metrics.json", "w") as fh:
        json.dump(out_metrics, fh, indent=1)
    np.savez(out / "eval_trajectories.npz", **traj_stats)
    print(f"saved {rel(out / 'eval_metrics.json')} and {rel(out / 'eval_trajectories.npz')}")


def check_autograd(model, F, J, device):
    """The float64 Jacobian and field against autograd on the float32 model, at the first equilibrium."""
    x = lv.EQUILIBRIA[0].to(device)
    Ja = torch.autograd.functional.jacobian(lambda z: model(z[None])[0], x)
    assert np.allclose(Ja.double().cpu().numpy(), J(FP[0]), atol=1e-4), (Ja, J(FP[0]))
    assert np.allclose(model(x[None])[0].detach().double().cpu().numpy(), F(FP[0]), atol=1e-5)


def evaluate_jacobians(results, out, device):
    selection = json.load(open(results / "selection.json"))
    out_jac, checked = {}, False
    for n in N_VALUES:
        for method in METHODS:
            rows = []
            for seed in SEEDS:
                run, tag = chosen_run(method, n, seed, selection)
                model = load_model(method, seed, results / "checkpoints" / f"{run}_best.pt", device)
                F, J = metrics.numpy_field_and_jacobian(model)
                if not checked:
                    check_autograd(model, F, J, device)
                    checked = True
                per_fp = []
                for p in FP:
                    Jt = lv.jacobian(p)
                    e = metrics.jacobian_comparison(J(p), Jt)
                    q = metrics.newton(F, J, p.copy(), iters=50)
                    e_eq = metrics.jacobian_comparison(J(q), Jt)
                    e.update({f"{k}_eq": v for k, v in e_eq.items() if k in ("jac_err", "eig_err", "eig_rel", "stable_ok", "eig_learned")})
                    e["shift"] = float(np.linalg.norm(q - p))
                    e["residual_at_p"] = float(np.linalg.norm(F(p)))
                    per_fp.append(e)
                rows.append({"seed": seed, "tag": tag, "per_fp": per_fp,
                             **{f"max_{k}": max(e[k] for e in per_fp)
                                for k in ("jac_err", "jac_rel", "eig_err", "eig_rel", "jac_err_eq", "eig_err_eq", "shift")},
                             "all_stable_ok": all(e["stable_ok"] for e in per_fp)})
            out_jac[f"{method}_n{n}"] = rows
            gm = lambda k: metrics.geometric_mean([r[k] for r in rows])
            print(f"N={n:>6} {method:12s} max ||dJ|| {gm('max_jac_err'):.2e} (rel {gm('max_jac_rel'):.1e}) "
                  f"max |dlambda| {gm('max_eig_err'):.2e} (rel {gm('max_eig_rel'):.1e}) | at learned eq: "
                  f"||dJ|| {gm('max_jac_err_eq'):.2e}, shift {gm('max_shift'):.1e} | "
                  f"stability ok {sum(r['all_stable_ok'] for r in rows)}/10", flush=True)
    with open(out / "jacobian_metrics.json", "w") as fh:
        json.dump(out_jac, fh, indent=1)
    print(f"saved {rel(out / 'jacobian_metrics.json')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=None, help="default: --results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    results = Path(args.results)
    out = Path(args.out) if args.out else results
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    evaluate_fields(results, out, device)
    evaluate_jacobians(results, out, device)


if __name__ == "__main__":
    main()
