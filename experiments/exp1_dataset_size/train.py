"""
One run of the dataset-size study of Section 5.2 (Figs. 5-6, Table 1), trained until the validation error
stops improving.

N_data points uniform in K (seeded), split 80/20 into training and validation; mini-batches of
min(500, N_train); Adam with lr 1e-3, halved when the validation MSE has not improved for 10 evaluations
(one every 1000 steps); training stops when lr < min_lr (default 1e-6) or at max_steps (default 2e6); the
state with the best validation MSE is kept (fpnode.training.train_until_converged).

Models: qr, free_mlp, penalty_mlp (param = lambda), aug_mlp (free MLP whose training set contains copies of
the four equilibria with target 0, a fraction param = rho of it).

Usage (from the repository root):
  python -m experiments.exp1_dataset_size.train <n_data> <model_type> <param> <seed> [<min_lr> <max_steps>]
      [--device cpu] [--out-dir DIR]
Outputs in results/ (or --out-dir): <run>.json, checkpoints/<run>_best.pt and _last.pt, snapshots/<run>_step<k>.pt.
All the runs of the paper: run_all.sh.
"""
import argparse
from pathlib import Path

import torch

from fpnode import data, get_device, metrics
from fpnode.training import log_spaced_steps, train_until_converged
from experiments.systems import lotka_volterra as lv

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
VAL_FRACTION = 0.2
BATCH_SIZE = 500
TEST_SIZE, TEST_SEED = 100_000, 10**6   # independent test set (test RMSE)
EVAL_EVERY = 1000


def run_name(model_type, n_data, param, seed):
    tag = {"penalty_mlp": f"lam{param:g}", "aug_mlp": f"rho{param:g}"}.get(model_type, "lam0")
    return f"{model_type}_n{n_data}_{tag}_seed{seed}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("n_data", type=lambda s: int(float(s)))
    ap.add_argument("model_type", choices=["qr", "free_mlp", "penalty_mlp", "aug_mlp"])
    ap.add_argument("param", type=float)
    ap.add_argument("seed", type=int)
    ap.add_argument("min_lr", type=float, nargs="?", default=1e-6)
    ap.add_argument("max_steps", type=lambda s: int(float(s)), nargs="?", default=2_000_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None, help="default: results/")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    device = get_device(args.device)
    run = run_name(args.model_type, args.n_data, args.param, args.seed)

    model = lv.build_model(args.model_type, args.seed, device)
    x_all = data.sample_uniform(args.n_data, lv.DOMAIN, args.seed).to(device)
    x_fit, x_val = data.train_val_split(x_all, VAL_FRACTION)
    n_fit, n_val, n_copies = len(x_fit), len(x_val), 0
    if args.model_type == "aug_mlp":
        x_fit, n_copies = data.augment_with_points(x_fit, lv.EQUILIBRIA, args.param)  # the field vanishes there
    with torch.no_grad():
        y_fit, y_val = lv.field(x_fit), lv.field(x_val)
    n_train = len(x_fit)

    grid = data.grid_points(lv.DOMAIN, lv.GRID_RES, device)
    x_test = data.uniform_points(TEST_SIZE, lv.DOMAIN, TEST_SEED).to(device)

    def monitor(m):
        return {"fixed_point_residual": m.fixed_point_residual(),
                "grid_rmse": metrics.rms(metrics.error_norms(m, lv.field, grid))}

    @torch.no_grad()
    def final_metrics(m):
        e = metrics.error_norms(m, lv.field, grid)
        return {"grid_rmse": metrics.rms(e), "grid_max_error": float(e.max()), "grid_mean_error": float(e.mean()),
                "fixed_point_residual": m.fixed_point_residual(),
                "train_mse": metrics.mean_squared_error(m, x_fit, y_fit),
                "val_mse": metrics.mean_squared_error(m, x_val, y_val),
                "test_rmse": metrics.rms(metrics.error_norms(m, lv.field, x_test))}

    config = {"n_data": args.n_data, "n_fit": n_fit, "n_val": n_val, "n_copies": n_copies, "n_train": n_train,
              "model_type": args.model_type, "param": args.param}
    train_until_converged(
        model, {"x_fit": x_fit, "y_fit": y_fit, "x_val": x_val, "y_val": y_val}, run=run, out_dir=Path(args.out_dir) if args.out_dir else RESULTS,
        seed=args.seed, penalty_lambda=args.param if args.model_type == "penalty_mlp" else 0.0,
        batch_size=min(BATCH_SIZE, n_train), lr0=1e-3, eval_every=EVAL_EVERY, patience=10, factor=0.5,
        min_lr=args.min_lr, max_steps=args.max_steps, checkpoint_every=10,
        snapshot_steps=log_spaced_steps(EVAL_EVERY, 2000, 30), monitor=monitor, final_metrics=final_metrics,
        config=config, device=device)


if __name__ == "__main__":
    main()
