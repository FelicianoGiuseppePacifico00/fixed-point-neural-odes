"""
Training loops for vector-field regression (one model per call).

* ``train_fixed_epochs``: Adam for a fixed number of epochs over a fixed training set, with a constant
  or cosine-annealed learning rate; the model is monitored after every epoch.
* ``train_until_converged``: Adam on mini-batches until the validation error stops improving: the
  learning rate is multiplied by ``factor`` when the validation MSE has not improved for ``patience``
  evaluations (ReduceLROnPlateau) and training stops when it falls below ``min_lr``; the state with the
  best validation MSE is kept. Interrupted runs resume from their last checkpoint.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import regression_loss
from .metrics import mean_squared_error
from .utils import sync


def train_fixed_epochs(model, x_train, y_train, *, epochs: int, lr: float, batch_size: int,
                       schedule: str = "constant", penalty_lambda: float = 0.0, monitor=None,
                       device=torch.device("cpu"), log_every: int = 0, log_prefix: str = ""):
    """Returns (history, train_seconds, total_seconds). history["train_mse"] is the mean training loss of
    every epoch (penalty included), history["lr"] the learning rate used in it, and every key returned by
    monitor(model) (evaluated after the epoch, in eval mode, without gradients) is added. train_seconds
    excludes the time spent in monitor."""
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    elif schedule == "constant":
        scheduler = None
    else:
        raise ValueError(f"unknown schedule {schedule!r}")

    history = {"train_mse": [], "lr": []}
    monitor_seconds = 0.0
    sync(device)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        lr_now = optimizer.param_groups[0]["lr"]
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = regression_loss(model, xb, yb, penalty_lambda)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu().item())
        if scheduler is not None:
            scheduler.step()
        sync(device)
        tm = time.perf_counter()
        model.eval()
        with torch.no_grad():
            metrics = monitor(model) if monitor is not None else {}
        history["train_mse"].append(running / max(len(loader), 1))
        history["lr"].append(lr_now)
        for key, value in metrics.items():
            history.setdefault(key, []).append(value)
        sync(device)
        monitor_seconds += time.perf_counter() - tm
        if log_every and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(f"{log_prefix}epoch {epoch}/{epochs} lr={lr_now:.2e} train={history['train_mse'][-1]:.3e} "
                  + " ".join(f"{k}={v[-1]:.3e}" for k, v in history.items() if k not in ("train_mse", "lr")), flush=True)
    sync(device)
    total_seconds = time.perf_counter() - t0
    model.eval()
    return history, total_seconds - monitor_seconds, total_seconds


def train_until_converged(model, data: dict, *, run: str, out_dir, seed: int, penalty_lambda: float = 0.0,
                          batch_size: int, lr0: float = 1e-3, eval_every: int = 1000, patience: int = 10,
                          factor: float = 0.5, min_lr: float = 1e-6, max_steps: int = 2_000_000,
                          checkpoint_every: int = 10, snapshot_steps=(), monitor=None, final_metrics=None,
                          config: dict | None = None, device=torch.device("cpu"), log_every: int = 50):
    """Trains model on data = {"x_fit", "y_fit", "x_val", "y_val"} (tensors on device).

    Every eval_every optimizer steps the history gets step, lr, the mean training loss since the last
    evaluation, the validation MSE (regression term only) and the entries of monitor(model); the validation
    MSE drives ReduceLROnPlateau. Outputs in out_dir:
      <run>.json                 history, final metrics of the best ("final") and of the last ("final_last")
                                 state (final_metrics(model)), config, training time;
      checkpoints/<run>_best.pt  state_dict with the best validation MSE;
      checkpoints/<run>_last.pt  full training state, saved every checkpoint_every evaluations: an
                                 interrupted run resumes from here, and a finished run continues from here
                                 if called again with a smaller min_lr or a larger max_steps (a resumed run
                                 starts a new pass over the training set, reseeded with seed + step);
      snapshots/<run>_step<k>.pt state_dict at the steps in snapshot_steps.
    Returns the path of the JSON file, or None if the run had already finished."""
    out_dir = Path(out_dir)
    ckpt_dir, snap_dir = out_dir / "checkpoints", out_dir / "snapshots"
    for d in (out_dir, ckpt_dir, snap_dir):
        d.mkdir(parents=True, exist_ok=True)
    out_path, last_path = out_dir / f"{run}.json", ckpt_dir / f"{run}_last.pt"
    x_fit, y_fit, x_val, y_val = data["x_fit"], data["y_fit"], data["x_val"], data["y_val"]
    n_train = x_fit.shape[0]
    snapshot_steps = set(snapshot_steps)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
    step, best_val, best_step, best_state, seconds_before = 0, float("inf"), 0, None, 0.0
    history = {"step": [], "lr": [], "train_loss": [], "val_mse": []}
    if last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        step, best_val, best_step, best_state = ck["step"], ck["best_val"], ck["best_step"], ck["best_state"]
        history, seconds_before = ck["history"], ck["train_seconds"]
        torch.manual_seed(seed + step)  # new pass over the training set
    current_lr = lambda: optimizer.param_groups[0]["lr"]
    finished = lambda: current_lr() < min_lr or step >= max_steps
    if out_path.exists() and finished():
        print(f"{out_path.name} already exists, skipping")
        return None

    t0 = time.perf_counter()

    def save_last():
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "best_val": best_val,
                    "best_step": best_step, "best_state": best_state, "history": history,
                    "train_seconds": seconds_before + time.perf_counter() - t0}, last_path)

    print(f"{run}: n_train={n_train} batch={batch_size} min_lr={min_lr:g} max_steps={max_steps} "
          f"resumed_at_step={step}", flush=True)
    F = lambda x: model(x)
    running, n_running = 0.0, 0
    while not finished():
        model.train()
        for idx in torch.randperm(n_train, device=device).split(batch_size):
            loss = regression_loss(model, x_fit[idx], y_fit[idx], penalty_lambda)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            running, n_running = running + float(loss.detach()), n_running + 1
            if step % eval_every:
                continue
            model.eval()
            val = mean_squared_error(F, x_val, y_val)
            entries = {"step": step, "lr": current_lr(), "train_loss": running / n_running, "val_mse": val}
            if monitor is not None:
                entries.update(monitor(model))
            for key, value in entries.items():
                history.setdefault(key, []).append(value)
            running, n_running = 0.0, 0
            if val < best_val:
                best_val, best_step = val, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            scheduler.step(val)
            if step in snapshot_steps:
                torch.save(model.state_dict(), snap_dir / f"{run}_step{step}.pt")
            if len(history["step"]) % checkpoint_every == 0:
                save_last()
            if log_every and len(history["step"]) % log_every == 0:
                print(f"[{run}] step {step} lr={history['lr'][-1]:.1e} val={val:.3e} best_step={best_step}", flush=True)
            model.train()
            if finished():
                break
    stop_reason = "min_lr" if current_lr() < min_lr else "max_steps"
    save_last()

    model.eval()
    final_last = final_metrics(model) if final_metrics is not None else {}
    model.load_state_dict(best_state)
    final = final_metrics(model) if final_metrics is not None else {}
    torch.save(best_state, ckpt_dir / f"{run}_best.pt")
    config = {**(config or {}), "seed": seed, "lambda": penalty_lambda, "batch_size": batch_size, "lr0": lr0,
              "eval_every": eval_every, "patience_evals": patience, "factor": factor, "min_lr": min_lr,
              "max_steps": max_steps, "stop_step": step, "stop_reason": stop_reason, "best_step": best_step,
              "best_val_mse": best_val,
              "schedule": "Adam, ReduceLROnPlateau on the validation MSE, stop when lr < min_lr"}
    train_seconds = seconds_before + time.perf_counter() - t0
    with open(out_path, "w") as fh:
        json.dump({**history, "final": final, "final_last": final_last, "config": config,
                   "train_seconds": train_seconds}, fh)
    print(f"[{run}] final {final} best_step={best_step} stop_step={step} ({stop_reason}) "
          f"{train_seconds / 60:.1f} min", flush=True)
    return out_path


def log_spaced_steps(eval_every: int, max_evals: int = 2000, n: int = 30):
    """About n evaluation steps, log-spaced between eval_every and max_evals * eval_every (snapshots)."""
    return set(int(k) * eval_every for k in np.unique(np.round(np.logspace(0, np.log10(max_evals), n))))
