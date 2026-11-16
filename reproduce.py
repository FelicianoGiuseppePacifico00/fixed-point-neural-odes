"""
Reproduce the results of the paper. Run from the repository root:

  python reproduce.py figures            figures, tables and numbers of the paper from the stored results
                                         (about 1 min on a CPU)
  python reproduce.py evaluate           recompute the evaluations from the stored checkpoints, then the
                                         figures and tables that depend on them (15-30 min on a CPU)
  python reproduce.py quick              retrain Experiment 1 for one seed with the protocol of the paper and
                                         compare the models (a few minutes on a CPU)
  python reproduce.py train EXPERIMENT   retrain one experiment from scratch, then evaluate and plot it
                                         (hours, see README.md)

Everything is written to reproduced/; the stored results in experiments/ are never modified.
Options: --device cpu|cuda (default: cpu, except for `train`, where the default is CUDA if available).
"""
import argparse
import filecmp
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reproduced"
EXPERIMENTS = ROOT / "experiments"
TRAIN = {
    "exp1_fixed_epochs": "Section 5.2, Experiment 1 and Figs. 2-4: 50 runs, about 2 min each on a GPU",
    "exp1_dataset_size": "Section 5.2, Figs. 5-6 and Table 1: 290 runs, about 5 h on 28 CPU cores",
    "appB_limit_cycle": "Appendix B, Fig. 7: one run, about 20 min on a GPU",
    "appC_trajectories": "Appendix C, Fig. 8 and Tables 2-3: 60 runs in one process, about 35 h on a GPU",
}


def rel(path):
    return os.path.relpath(path, ROOT)


def run(*args, device=None, env=None, stdout=None):
    """python -m <args> (or a shell script) from the repository root."""
    cmd = ["bash", *args] if str(args[0]).endswith(".sh") else [sys.executable, "-m", *args]
    shown = " ".join(rel(a) if isinstance(a, Path) else str(a) for a in cmd[1:] if a != "-m")
    print(f"\n$ {shown}", flush=True)
    full_env = {**os.environ, "MPLBACKEND": "Agg", **(env or {})}
    if device is not None:
        full_env["FP_DEVICE"] = device
    subprocess.run([str(c) for c in cmd], cwd=ROOT, env=full_env, check=True, stdout=stdout)


TABLE_COLUMNS = {
    "exp1_dataset_size/table.tex": ["grid RMSE", "local RMSE", "r_fp", "Jacobian error", "traj. error t=30"],
    "appC_trajectories/table.tex": ["traj. error t<=10", "traj. error 10<t<=30", "r_fp", "Jacobian error"],
    "appC_trajectories/table_equilibria.tex": ["type", "obs.", "residual QR", "residual Free", "residual Pen.",
                                               "Jacobian QR", "Jacobian Free", "Jacobian Pen.", "type QR",
                                               "type Free", "type Pen."],
}


def plain(cell):
    """A LaTeX table cell as plain text: 9.8e-3 x/ 2.25."""
    for a, b in ((r"\boldmath", ""), (r"\times10^", "e"), (r"\,{\times}/{\div}\,", " x/ "), ("$", ""),
                 ("{", ""), ("}", ""), (r"\\", ""), (r"\%", "%"), (r"\mathrm", "")):
        cell = cell.replace(a, b)
    return cell.strip()


def compare_tables(new_dir, exp):
    """Compares the regenerated LaTeX tables with the stored ones (those of the paper), cell by cell."""
    for new in sorted(Path(new_dir).glob("*.tex")):
        stored = EXPERIMENTS / exp / "figures" / new.name
        old_rows, new_rows = stored.read_text().splitlines(), new.read_text().splitlines()
        if old_rows == new_rows:
            print(f"  {rel(new)}: identical to {rel(stored)}")
            continue
        columns = TABLE_COLUMNS.get(f"{exp}/{new.name}", [])
        block, diffs = "", []
        for a, b in zip(old_rows, new_rows):
            if a.startswith(r"\multicolumn"):
                block = plain(a.split("{l}")[-1]) + ", "
            for i, (x, y) in enumerate(zip(a.split(" & ")[1:], b.split(" & ")[1:])):
                if x != y:
                    col = columns[i] if i < len(columns) else f"column {i + 2}"
                    diffs.append(f"    {block}{plain(a.split(' & ')[0])}, {col}: paper {plain(x)}, reproduced {plain(y)}")
        print(f"  {rel(new)}: {len(diffs)} cells differ from {rel(stored)}")
        print("\n".join(diffs))


def figures(device):
    out = OUT / "figures"
    run("experiments.exp1_fixed_epochs.plot", "--figures", out / "exp1_fixed_epochs", "--device", device)
    run("experiments.exp1_dataset_size.plot", "--figures", out / "exp1_dataset_size")
    run("experiments.appC_trajectories.plot", "--figures", out / "appC_trajectories")
    with open(out / "paper_numbers.txt", "w") as fh:
        run("experiments.paper_numbers", stdout=fh)
    print(f"saved {rel(out / 'paper_numbers.txt')} (every number quoted in the text)")
    print("\nTables:")
    compare_tables(out / "exp1_dataset_size", "exp1_dataset_size")
    compare_tables(out / "appC_trajectories", "appC_trajectories")
    print("Fig. 1 is a diagram; Fig. 7 (Appendix B) was produced by an earlier notebook without a fixed seed and is\n"
          "only stored (experiments/appB_limit_cycle/figures), see README.md.")


def evaluate(device):
    out = OUT / "evaluation"
    ds, tr = out / "exp1_dataset_size", out / "appC_trajectories"
    run("experiments.exp1_dataset_size.evaluate", "--out", ds, "--device", device)
    run("experiments.exp1_dataset_size.plot", "--results", ds, "--figures", ds / "figures")
    run("experiments.appC_trajectories.evaluate", "--out", tr, "--device", device)
    run("experiments.appC_trajectories.check_solver", "--out", tr, "--device", device)
    run("experiments.appC_trajectories.plot", "--results", tr, "--figures", tr / "figures")
    print("\nTables from the recomputed evaluations:")
    compare_tables(ds / "figures", "exp1_dataset_size")
    compare_tables(tr / "figures", "appC_trajectories")
    print("The stored dataset-size evaluation was computed on a GPU: on a CPU, the trajectory error of the QR\n"
          "model at t = 30 (last column of Table 1), which is at float32 round-off level, can change slightly.")


def train(exp, device):
    out = OUT / "train" / exp
    results, figs = out / "results", out / "figures"
    print(f"Retraining {exp} from scratch ({TRAIN[exp]}) into {rel(out)}.\n"
          "Interrupted runs resume from their checkpoints when the command is repeated.", flush=True)
    env = {"OUT": str(results), "PYTHON": sys.executable}
    if exp == "exp1_fixed_epochs":
        run(EXPERIMENTS / exp / "run_all.sh", device=device, env=env)
        run("experiments.exp1_fixed_epochs.plot", "--results", results, "--figures", figs, "--device", device or "cpu")
    elif exp == "exp1_dataset_size":
        run(EXPERIMENTS / exp / "run_all.sh", env=env)
        run("experiments.exp1_dataset_size.evaluate", "--results", results, "--device", device or "cpu")
        run("experiments.exp1_dataset_size.plot", "--results", results, "--figures", figs)
    elif exp == "appB_limit_cycle":
        run("experiments.appB_limit_cycle.train_and_plot", "--out-dir", results, device=device)
    elif exp == "appC_trajectories":
        run("experiments.appC_trajectories.train", "--out-dir", results, device=device)
        run("experiments.appC_trajectories.evaluate", "--results", results, "--device", device or "cpu")
        run("experiments.appC_trajectories.check_solver", "--out", results, "--device", device or "cpu")
        run("experiments.appC_trajectories.plot", "--results", results, "--figures", figs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["figures", "evaluate", "quick", "train"])
    ap.add_argument("experiment", nargs="?", choices=sorted(TRAIN), help="for train")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="for quick")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.command == "figures":
        figures(args.device or "cpu")
    elif args.command == "evaluate":
        evaluate(args.device or "cpu")
    elif args.command == "quick":
        run("experiments.quick", "--device", args.device or "cpu", "--seeds", *map(str, args.seeds))
    else:
        if args.experiment is None:
            ap.error("train needs an experiment: " + ", ".join(sorted(TRAIN)))
        train(args.experiment, args.device)


if __name__ == "__main__":
    main()
