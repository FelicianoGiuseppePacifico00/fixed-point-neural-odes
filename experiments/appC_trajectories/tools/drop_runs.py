"""
Removes runs from the stopped stacked training of train.py, so that it resumes without them. Each removed run
is finalized as at the end of training (JSON with history, metrics of the best and last parameters, config;
*_best.pt and *_last.pt), with stop_reason "dropped". The original stack checkpoint is kept as
checkpoints/stack_before_drop_step<step>.pt. Run it only while the training is NOT running.

In the paper, lambda = 10 was dropped at step 1.18e6 (validation MSE 20 times higher than lambda = 0.1):
  python -m experiments.appC_trajectories.tools.drop_runs penalty_mlp:10:all

Usage: python -m experiments.appC_trajectories.tools.drop_runs kind:lam:seed[,kind:lam:seed...] [--results DIR]
       (seed "all" = every seed of that kind and lambda still in the stack)
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

from fpnode import get_device
from .. import common as td
from ..train import MAX_STEPS, finalize_run, load_stack, save_stack


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", help="kind:lam:seed[,kind:lam:seed...]")
    ap.add_argument("--results", default=str(td.RESULTS))
    args = ap.parse_args(argv)
    device = get_device()
    out_dir = Path(args.results)
    stack_path = out_dir / "checkpoints" / "stack_last.pt"
    setup = td.make_setup(device)
    st, state, ck = load_stack(stack_path, setup)
    drop = set()
    for spec in args.runs.split(","):
        k, l, s = spec.split(":")
        drop |= {r for r in st.runs if r[0] == k and r[1] == float(l) and (s == "all" or r[2] == int(s))}
    if not drop:
        sys.exit("no run of the stack matches " + args.runs)

    trajectories, win = td.load_data(out_dir, device)
    backup = stack_path.parent / f"stack_before_drop_step{state.step}.pt"
    shutil.copy2(stack_path, backup)
    for i, run in enumerate(st.runs):
        if run in drop:
            finalize_run(st, i, "dropped", state, ck["train_seconds"], out_dir=out_dir, trajectories=trajectories,
                         win=win, max_steps=MAX_STEPS,
                         extra_config={"stop_note": "removed before convergence: lambda not competitive in validation",
                                       "lr_at_stop": float(st.lr[i])})
    st = st.subset([i for i, r in enumerate(st.runs) if r not in drop])
    gen = torch.Generator(device=device)
    gen.set_state(ck["gen_state"].cpu())
    save_stack(stack_path, st, state, gen, ck["train_seconds"])
    print(f"stack at step {state.step}: {len(drop)} runs dropped, {len(st)} left; backup {backup.name}", flush=True)


if __name__ == "__main__":
    main()
