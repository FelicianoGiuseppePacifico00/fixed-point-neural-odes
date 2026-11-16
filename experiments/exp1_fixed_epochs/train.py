"""
One fixed-epoch run of the Lotka-Volterra vector-field regression (Section 5.2: Experiment 1, Figs. 2-4, and
the check with a constant learning rate).

N_train points uniform in K (seeded), targets F(x); Adam, mini-batches of 500, `epochs` epochs, lr 1e-3
constant or cosine-annealed to 0 (stepped per epoch). After every epoch: fixed-point residual and grid RMSE
on the 250 x 250 grid. At the end: grid MSE / RMSE, max and mean pointwise error, residual.

Usage (from the repository root):
  python -m experiments.exp1_fixed_epochs.train --model qr --seed 0 --schedule cosine
      [--lam 0] [--n-train 50000] [--epochs 400] [--device cuda] [--out-dir DIR]
Outputs in results/<schedule>/ (or --out-dir): <run>.json (config, history, final metrics, timings) and <run>.pt.
All the runs of the paper: run_all.sh.
"""
import argparse
import json
import platform
from pathlib import Path

import torch

from fpnode import data, get_device, metrics
from fpnode.training import train_fixed_epochs
from experiments.systems import lotka_volterra as lv

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BATCH_SIZE = 500
LR = 1e-3


def run_name(model_type, n_train, lam, seed):
    tag = f"lam{lam:g}" if model_type == "penalty_mlp" else "lam0"
    return f"{model_type}_n{n_train}_{tag}_seed{seed}"


@torch.no_grad()
def final_metrics(model, grid):
    err = metrics.error_norms(model, lv.field, grid)
    mse = float(err.pow(2).mean())
    return {"grid_mse": mse, "grid_rmse": mse ** 0.5, "max_error": float(err.max()), "mean_error": float(err.mean()),
            "fixed_point_residual": model.fixed_point_residual()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qr", "free_mlp", "penalty_mlp"])
    ap.add_argument("--lam", type=float, default=0.0, help="penalty weight (penalty_mlp only)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--schedule", required=True, choices=["cosine", "constant"])
    ap.add_argument("--n-train", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None, help="default: results/<schedule>/")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    device = get_device(args.device)
    lam = args.lam if args.model == "penalty_mlp" else 0.0

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / args.schedule
    out_dir.mkdir(parents=True, exist_ok=True)
    run = run_name(args.model, args.n_train, lam, args.seed)
    if (out_dir / f"{run}.json").exists():
        print(f"{run}.json already exists, skipping")
        return

    model = lv.build_model(args.model, args.seed, device)
    x_train = data.sample_uniform(args.n_train, lv.DOMAIN, args.seed)
    with torch.no_grad():
        y_train = lv.field(x_train.to(device)).cpu()
    grid = data.grid_points(lv.DOMAIN, lv.GRID_RES, device)
    monitor = lambda m: {"fixed_point_residual": m.fixed_point_residual(),
                         "grid_rmse": metrics.grid_rmse(m, lv.field, grid)}
    print(f"{run}: schedule={args.schedule} epochs={args.epochs} device={device}", flush=True)
    history, train_seconds, total_seconds = train_fixed_epochs(
        model, x_train, y_train, epochs=args.epochs, lr=LR, batch_size=BATCH_SIZE, schedule=args.schedule,
        penalty_lambda=lam, monitor=monitor, device=device, log_every=50, log_prefix=f"[{run}] ")

    config = {"model_type": args.model, "lambda": lam, "seed": args.seed, "n_train": args.n_train,
              "epochs": args.epochs, "batch_size": BATCH_SIZE, "lr": LR, "schedule": args.schedule,
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch": torch.__version__, "python": platform.python_version()}
    result = {"config": config, "history": history, "final": final_metrics(model, grid),
              "train_seconds": train_seconds, "total_seconds": total_seconds}
    with open(out_dir / f"{run}.json", "w") as fh:
        json.dump(result, fh)
    torch.save(model.state_dict(), out_dir / f"{run}.pt")
    print(f"[{run}] final {result['final']} train {train_seconds / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
