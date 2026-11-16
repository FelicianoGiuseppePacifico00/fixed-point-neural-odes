"""
Training on trajectories through the ODE solver (Appendix C): QR-constrained model, free MLP and MLP + penalty
(lambda in {0.01, 0.1, 1, 10}), 10 seeds each, all trained together in one process (fpnode.stacked) until
the validation error stops improving (the protocol of the dataset-size study).

Every step each model draws 128 training windows of its seed uniformly with replacement and integrates them
with RK4 (common.DT_MODEL); loss = mean squared error over windows, observations and components
+ lambda * mean_l ||F(x_l)||^2. Every 1000 steps: validation MSE on all the validation windows ->
ReduceLROnPlateau (factor 0.5, patience 10 evaluations); the best state is kept; a model stops when its lr
falls below 1e-6, or at max_steps.

Usage (from the repository root; detached on a GPU: run.sh):
  python -m experiments.appC_trajectories.train [--out-dir DIR] [--max-steps N] [--no-graph]
      [--runs kind:lam:seed,...] [--device cuda]
Outputs in results/ (or --out-dir):
  data.pt                               trajectories of every seed (float64), train / validation split;
  <run>.json                            config, history at every evaluation, metrics of the best ("final")
                                        and of the last ("final_last") state;
  checkpoints/<run>_best.pt, _last.pt   state_dicts of the fpnode models;
  checkpoints/stack_last.pt             the whole stack, every 10 evaluations: an interrupted run resumes from
                                        here (renamed stack_final.pt at the end).
"""
import argparse
import json
from pathlib import Path

import torch

from fpnode import get_device
from fpnode import stacked as stk
from . import common as td

LR0, EVAL_EVERY, PATIENCE, FACTOR, THRESHOLD = 1e-3, 1000, 10, 0.5, 1e-4
MIN_LR, MAX_STEPS = 1e-6, 2_000_000
CHECKPOINT_EVERY = 10   # evaluations


def run_config(run, trajectories, max_steps):
    return {"model_type": run[0], "lambda": run[1], "seed": run[2], "n_traj": td.N_TRAJ,
            "n_train_traj": int(trajectories["train"].shape[1]), "n_val_traj": int(trajectories["val"].shape[1]),
            "dt_obs": td.DT_OBS, "t_obs": td.T_OBS, "window": td.H_WINDOW, "batch_windows": td.BATCH_WINDOWS,
            "dt_model": td.DT_MODEL, "integrator": "RK4, backprop through the solver", "lr0": LR0,
            "eval_every": EVAL_EVERY, "patience_evals": PATIENCE, "factor": FACTOR, "min_lr": MIN_LR,
            "max_steps": max_steps}


def finalize_run(st, i, reason, state, train_seconds, *, out_dir, trajectories, win, max_steps, best_params=None,
                 extra_config=None):
    """Writes <run>.json and the best / last checkpoints of model i of the stack."""
    run, setup = st.runs[i], st.setup
    name = td.run_name(*run)
    last = st.params_of(i)
    bp = best_params if best_params is not None else state.best[name]["params"]
    bp = bp if bp is not None else last
    ckpt_dir = Path(out_dir) / "checkpoints"
    td.save_model(run, bp, ckpt_dir / f"{name}_best.pt", setup.device)
    td.save_model(run, last, ckpt_dir / f"{name}_last.pt", setup.device)
    config = {**run_config(run, trajectories, max_steps), "stop_step": state.step, "stop_reason": reason,
              **(extra_config or {}), "best_step": state.best[name]["step"], "best_val_mse": state.best[name]["val"],
              "schedule": "Adam, ReduceLROnPlateau on the validation MSE, stop when lr < min_lr",
              "train_seconds_shared": train_seconds}
    with open(Path(out_dir) / f"{name}.json", "w") as fh:
        json.dump({**state.hist[name], "final": stk.single_metrics(run, bp, win, setup),
                   "final_last": stk.single_metrics(run, last, win, setup), "config": config}, fh)
    print(f"[{name}] stopped at step {state.step} ({reason}), best step {state.best[name]['step']}, "
          f"best val {state.best[name]['val']:.3e}", flush=True)


def save_stack(path, st, state, gen, train_seconds):
    torch.save({"step": state.step, "runs": st.runs, "P": {k: st.P[k].detach() for k in stk.PARAMS}, "m": st.m,
                "v": st.v, "lr": st.lr.tolist(), "t": float(st.t), "running": st.running, "hist": state.hist,
                "best": state.best, "sched": state.sched, "gen_state": gen.get_state(),
                "train_seconds": train_seconds}, path)


def load_stack(path, setup):
    ck = torch.load(path, map_location=setup.device)
    st = stk.Stack(ck["runs"], ck["P"], setup, ck["m"], ck["v"], lr=ck["lr"], t=ck["t"])
    st.running.copy_(ck["running"])
    state = stk.StackState(step=ck["step"], train_seconds=ck["train_seconds"], hist=ck["hist"], best=ck["best"],
                           sched=ck["sched"])
    return st, state, ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(td.RESULTS))
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--runs", default=None, help="comma-separated kind:lam:seed")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    for d in (out_dir, ckpt_dir, out_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    stack_path = ckpt_dir / "stack_last.pt"
    setup = td.make_setup(device)
    trajectories, win = td.load_data(out_dir, device)
    use_graph = not args.no_graph and device.type == "cuda"
    gen = torch.Generator(device=device)

    if stack_path.exists():
        st, state, ck = load_stack(stack_path, setup)
        gen.set_state(ck["gen_state"].cpu())
        print(f"resumed at step {state.step} with {len(st)} active models", flush=True)
    else:
        runs = td.all_runs() if args.runs is None else [
            (k, float(l), int(s)) for k, l, s in (r.split(":") for r in args.runs.split(","))]
        runs = stk.sort_runs([r for r in runs if not (out_dir / f"{td.run_name(*r)}.json").exists()])
        st = stk.Stack(runs, {k: torch.stack([td.initial_params(r[0], r[2])[k] for r in runs]) for k in stk.PARAMS}, setup)
        state = stk.StackState()
        state.add_runs([td.run_name(*r) for r in runs])
        gen.manual_seed(12345)
    if len(st) == 0:
        print("nothing to do")
        return
    print(f"{len(st)} models ({st.nq} QR), {td.BATCH_WINDOWS} windows of {td.H_WINDOW} observations, "
          f"RK4 dt={td.DT_MODEL:g} ({td.SUBSTEPS} steps/obs), graph={use_graph}, max_steps={args.max_steps}", flush=True)

    on_finish = lambda st_, i, reason, state_, secs: finalize_run(
        st_, i, reason, state_, secs, out_dir=out_dir, trajectories=trajectories, win=win, max_steps=args.max_steps)
    on_checkpoint = lambda st_, state_, secs: save_stack(stack_path, st_, state_, gen, secs)
    stk.train_stack(st, win, state, gen, run_name=td.run_name, on_finish=on_finish, on_checkpoint=on_checkpoint,
                    eval_every=EVAL_EVERY, patience=PATIENCE, factor=FACTOR, threshold=THRESHOLD, min_lr=MIN_LR,
                    max_steps=args.max_steps, checkpoint_every=CHECKPOINT_EVERY, use_graph=use_graph)
    if stack_path.exists():
        stack_path.rename(ckpt_dir / "stack_final.pt")
    print("all done", flush=True)


if __name__ == "__main__":
    main()
