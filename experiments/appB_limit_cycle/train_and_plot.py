"""
Appendix B (Fig. 7): QR-constrained model with the equilibrium of the Brusselator-like field prescribed,
trained by regression on N_train = 1e6 points uniform in K (Adam, lr 1e-3, mini-batches of 500, 200 epochs),
and the figure with the target field, the learned field and the log-scale error map.

Note: the figure of the paper (figures/vectorfield_target_learned_error.pdf) was produced by an earlier notebook
version of this script, without a fixed seed, so a rerun gives an equivalent but not identical figure. The
rerun is written to results/ (or --out-dir) and does not overwrite it.

Usage (from the repository root):
  python -m experiments.appB_limit_cycle.train_and_plot [--seed 0] [--device cuda] [--out-dir DIR]
      [--n-train 1000000] [--epochs 200]
Outputs in results/ (or --out-dir): qr_seed<seed>.json and .pt, vectorfield_target_learned_error.pdf
"""
import argparse
import json
from pathlib import Path

import torch

from fpnode import QRConstrainedMLP, data, get_device, plotting, set_seed
from fpnode.training import train_fixed_epochs
from experiments import rel
from experiments.systems import limit_cycle as lc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
N_TRAIN, BATCH_SIZE, EPOCHS, LR = 1_000_000, 500, 200, 1e-3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args(argv)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model = QRConstrainedMLP(lc.N_VISIBLE, lc.M_HIDDEN, lc.EQUILIBRIA).to(device)
    x_train = data.sample_uniform(args.n_train, lc.DOMAIN, args.seed)
    with torch.no_grad():
        y_train = lc.field(x_train.to(device)).cpu()
    history, train_seconds, _ = train_fixed_epochs(
        model, x_train, y_train, epochs=args.epochs, lr=LR, batch_size=BATCH_SIZE, schedule="constant",
        monitor=lambda m: {"fixed_point_residual": m.fixed_point_residual()}, device=device, log_every=10)
    stem = out_dir / f"qr_seed{args.seed}"
    with open(stem.with_suffix(".json"), "w") as fh:
        json.dump({"config": {"seed": args.seed, "n_train": args.n_train, "epochs": args.epochs,
                              "batch_size": BATCH_SIZE, "lr": LR}, "history": history,
                   "train_seconds": train_seconds}, fh)
    torch.save(model.state_dict(), stem.with_suffix(".pt"))

    path = out_dir / "vectorfield_target_learned_error.pdf"
    plotting.plot_field_comparison(lc.field, model.field_fn(), lc.DOMAIN, lc.EQUILIBRIA.numpy(), path, device=device)
    print("saved", rel(path))


if __name__ == "__main__":
    main()
