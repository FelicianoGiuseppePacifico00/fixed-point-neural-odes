"""
Many independent models trained together through the ODE solver, in one process.

The parameters (W, d, B, b) of E models are stacked along a leading dimension, so that the matrix
products, the QR factorizations and the RK4 steps of all the models are batched. The models are
independent: the loss is the sum of the per-model losses, Adam is element-wise, and every model has its
own learning rate and its own ReduceLROnPlateau state, so each model follows the same algorithm as a
separate run (only the random mini-batches differ). A model that stops is removed from the stack. On a GPU
the training step is captured in a CUDA graph, recaptured whenever the stack changes.

A run is a tuple (kind, lambda, seed) with kind in {"qr", "free_mlp", "penalty_mlp"}; the QR models must
come first in the stack (``sort_runs``). The training windows of the seed (``seed_index``) are used.
The batched fields are those of fpnode.models (QRConstrainedMLP for "qr", FreeMLP otherwise), sigmoid
activation.
"""
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np
import torch

from .integrators import rk4_step

PARAMS = ("W", "d", "B", "b")


@dataclass
class StackSetup:
    equilibria: torch.Tensor          # [C, n] float32 on device: the prescribed points
    grid: torch.Tensor                # [G, n] evaluation grid on device
    target_field: Callable            # the true field (grid RMSE)
    dt: float                         # RK4 step of the models
    substeps: int                     # RK4 steps per observation interval
    horizon: int                      # observations predicted by a window
    batch_windows: int                # windows per model and step
    seeds: list                       # data blocks of the windows tensors, by seed
    device: torch.device
    lr0: float = 1e-3
    betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8

    def seed_index(self, seed):
        return self.seeds.index(seed)


def sort_runs(runs):
    """QR models first (output_matrix relies on it), otherwise the given order."""
    return [r for r in runs if r[0] == "qr"] + [r for r in runs if r[0] != "qr"]


# ----------------------------------------------------------------------------- batched field
def output_matrix(P, nq, xbar):
    """Output matrix A of every model: the QR parametrization of QRConstrainedMLP for the first nq models
    (A S = Y exactly), the free matrix W for the others."""
    W = P["W"]
    if nq == 0:
        return W
    Wq, dq, Bq, bq = W[:nq], P["d"][:nq], P["B"][:nq], P["b"][:nq]
    S = torch.sigmoid(xbar @ Bq.transpose(1, 2) + bq[:, None, :]).transpose(1, 2)   # [nq, m, C]
    Y = xbar.t()[None] - dq[:, :, None]                                             # [nq, n, C]
    Q, R = torch.linalg.qr(S, mode="reduced")
    Zt = torch.linalg.solve_triangular(R.transpose(1, 2), Y.transpose(1, 2), upper=False)  # [nq, C, n]
    Qt = Q.transpose(1, 2)
    Aq = Zt.transpose(1, 2) @ Qt + (Wq - (Wq @ Q) @ Qt)
    return Aq if nq == W.shape[0] else torch.cat([Aq, W[nq:]])


def field(P, A, x):
    """F(x) = -x + A sigmoid(B x + b) + d for every model; x: [E, N, n]."""
    return -x + torch.sigmoid(x @ P["B"].transpose(1, 2) + P["b"][:, None, :]) @ A.transpose(1, 2) + P["d"][:, None, :]


def rollout(P, A, x0, setup: StackSetup, h=None):
    """RK4 from x0 [E, N, n]; the states at the h next observations [E, N, h, n]."""
    F = lambda z: field(P, A, z)
    x, out = x0, []
    for _ in range(setup.horizon if h is None else h):
        for _ in range(setup.substeps):
            x = rk4_step(F, x, setup.dt)
        out.append(x)
    return torch.stack(out, dim=2)


# ----------------------------------------------------------------------------- stack
class Stack:
    """The active models: stacked parameters, Adam moments, learning rates, static buffers."""

    def __init__(self, runs, P, setup: StackSetup, m=None, v=None, lr=None, t=0.0):
        self.runs = list(runs)
        assert self.runs == sort_runs(self.runs)
        self.setup = setup
        dev = setup.device
        E = len(self.runs)
        self.nq = sum(r[0] == "qr" for r in self.runs)
        self.P = {k: P[k].to(dev).float().contiguous().requires_grad_() for k in PARAMS}
        self.m = {k: (m[k].to(dev) if m else torch.zeros_like(self.P[k])).detach().clone() for k in PARAMS}
        self.v = {k: (v[k].to(dev) if v else torch.zeros_like(self.P[k])).detach().clone() for k in PARAMS}
        self.lr = torch.tensor(lr if lr is not None else [setup.lr0] * E, device=dev, dtype=torch.float32)
        self.t = torch.tensor(float(t), device=dev)
        self.lam = torch.tensor([r[1] for r in self.runs], device=dev, dtype=torch.float32)
        self.sidx = torch.tensor([setup.seed_index(r[2]) for r in self.runs], device=dev)
        self.idx = torch.zeros(E, setup.batch_windows, dtype=torch.long, device=dev)
        self.running = torch.zeros(E, device=dev)

    def __len__(self):
        return len(self.runs)

    def mutable(self):
        """Tensors modified by train_step (saved / restored around the CUDA-graph warmup)."""
        return [*self.P.values(), *self.m.values(), *self.v.values(), self.t, self.running]

    def subset(self, keep):
        """New stack with the models of index list keep (state carried over)."""
        kk = torch.tensor(keep, device=self.setup.device, dtype=torch.long)
        sel = lambda d: {k: d[k].detach()[kk] for k in PARAMS}
        new = Stack([self.runs[i] for i in keep], sel(self.P), self.setup, sel(self.m), sel(self.v),
                    lr=self.lr[kk].tolist(), t=float(self.t))
        new.running.copy_(self.running[kk])
        return new

    def params_of(self, i):
        return {k: self.P[k].detach()[i].cpu().clone() for k in PARAMS}


def per_model_loss(st: Stack, x0, tgt):
    """Loss of every model on its windows x0 [E, N, n], tgt [E, N, h, n]: [E]. Mean squared error over
    windows, observations and components, plus lambda * mean_l ||F(x_l)||^2."""
    xbar = st.setup.equilibria
    A = output_matrix(st.P, st.nq, xbar)
    mse = (rollout(st.P, A, x0, st.setup) - tgt).pow(2).mean(dim=(1, 2, 3))
    pen = field(st.P, A, xbar[None].expand(len(st), -1, -1)).pow(2).sum(-1).mean(-1)
    return mse + st.lam * pen


def train_step(st: Stack, win):
    """One Adam step of every model on the windows st.idx of its seed."""
    X0, TGT = win["train"]
    sidx = st.sidx[:, None]
    loss = per_model_loss(st, X0[sidx, st.idx], TGT[sidx, st.idx])
    grads = torch.autograd.grad(loss.sum(), [st.P[k] for k in PARAMS])
    beta1, beta2 = st.setup.betas
    with torch.no_grad():
        st.running.add_(loss)
        st.t.add_(1.0)
        bc1 = 1.0 - beta1 ** st.t
        bc2_sqrt = (1.0 - beta2 ** st.t).sqrt()
        for k, g in zip(PARAMS, grads):
            p, m, v = st.P[k], st.m[k], st.v[k]
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            lr = st.lr.view(-1, *([1] * (p.dim() - 1)))
            p.sub_((lr / bc1) * m / (v.sqrt() / bc2_sqrt + st.setup.adam_eps))


def capture(st: Stack, win):
    """CUDA graph of train_step; the warmup steps are undone."""
    saved = [x.detach().clone() for x in st.mutable()]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            train_step(st, win)
    torch.cuda.current_stream().wait_stream(s)
    with torch.no_grad():
        for x, y in zip(st.mutable(), saved):
            x.copy_(y)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        train_step(st, win)
    return g


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def traj_mse(P, A, sidx, split, win, setup: StackSetup, chunk=1024):
    """Mean squared error of every model on all the windows of the split of its seed."""
    X0, TGT = win[split]
    n = X0.shape[1]
    sq = torch.zeros(len(sidx), device=setup.device)
    for i in range(0, n, chunk):
        sq += (rollout(P, A, X0[sidx, i:i + chunk], setup) - TGT[sidx, i:i + chunk]).pow(2).sum(dim=(1, 2, 3))
    return sq / (n * setup.horizon * X0.shape[-1])


@torch.no_grad()
def grid_rmse(P, A, setup: StackSetup, chunk=8192):
    E = A.shape[0]
    sq = torch.zeros(E, device=setup.device)
    for c in setup.grid.split(chunk):
        sq += (field(P, A, c[None].expand(E, -1, -1)) - setup.target_field(c)[None]).pow(2).sum(dim=(1, 2))
    return (sq / len(setup.grid)).sqrt()


@torch.no_grad()
def fp_residual(P, A, setup: StackSetup):
    return field(P, A, setup.equilibria[None].expand(A.shape[0], -1, -1)).norm(dim=-1).max(dim=1).values


@torch.no_grad()
def evaluate(st: Stack, win, splits=("val",)):
    A = output_matrix(st.P, st.nq, st.setup.equilibria)
    out = {f"{s}_mse": traj_mse(st.P, A, st.sidx, s, win, st.setup).tolist() for s in splits}
    out["grid_rmse"] = grid_rmse(st.P, A, st.setup).tolist()
    out["fixed_point_residual"] = fp_residual(st.P, A, st.setup).tolist()
    return out


def single_metrics(run, params, win, setup: StackSetup):
    """Train / validation MSE, grid RMSE and residual of one model with the given parameters."""
    st = Stack([run], {k: params[k][None] for k in PARAMS}, setup)
    r = evaluate(st, win, splits=("train", "val"))
    return {k: v[0] for k, v in r.items()}


# ----------------------------------------------------------------------------- training loop
@dataclass
class StackState:
    """Everything the loop needs besides the stack itself (saved in the stack checkpoints)."""
    step: int = 0
    train_seconds: float = 0.0
    hist: dict = dc_field(default_factory=dict)    # run name -> history
    best: dict = dc_field(default_factory=dict)    # run name -> {"val", "step", "params"}
    sched: dict = dc_field(default_factory=dict)   # run name -> ReduceLROnPlateau state {"best", "bad"}

    def add_runs(self, names):
        for name in names:
            self.hist[name] = {"step": [], "lr": [], "train_loss": [], "val_mse": [],
                               "fixed_point_residual": [], "grid_rmse": []}
            self.best[name] = {"val": float("inf"), "step": 0, "params": None}
            self.sched[name] = {"best": float("inf"), "bad": 0}


def train_stack(st: Stack, win, state: StackState, gen: torch.Generator, *, run_name, on_finish, on_checkpoint,
                eval_every=1000, patience=10, factor=0.5, threshold=1e-4, min_lr=1e-6, max_steps=2_000_000,
                checkpoint_every=10, use_graph=True, log_every=10):
    """Trains until every model has stopped (lr < min_lr, or max_steps reached).

    Every step each model draws setup.batch_windows training windows of its seed uniformly with replacement
    (generator gen). Every eval_every steps: validation MSE on all the validation windows of its seed ->
    ReduceLROnPlateau replica (mode min, relative threshold, cooldown 0, as torch); the parameters with the
    best validation MSE are kept. on_finish(st, i, reason, state, elapsed) is called for every model that
    stops, before it leaves the stack; on_checkpoint(st, state, elapsed) every checkpoint_every evaluations.
    Returns the final state."""
    setup = st.setup
    graph = capture(st, win) if use_graph else None
    t0 = time.perf_counter()
    elapsed = lambda: state.train_seconds + time.perf_counter() - t0
    n_evals = state.step // eval_every
    while len(st) > 0:
        for _ in range(eval_every):
            st.idx.copy_(torch.randint(0, win["train"][0].shape[1], st.idx.shape, device=setup.device, generator=gen))
            if graph is not None:
                graph.replay()
            else:
                train_step(st, win)
        state.step += eval_every
        n_evals += 1
        ev = evaluate(st, win)
        running = (st.running / eval_every).tolist()
        st.running.zero_()
        lrs = st.lr.tolist()
        done = []
        for i, run in enumerate(st.runs):
            name, val = run_name(*run), ev["val_mse"][i]
            h = state.hist[name]
            for key, value in (("step", state.step), ("lr", lrs[i]), ("train_loss", running[i]), ("val_mse", val),
                               ("fixed_point_residual", ev["fixed_point_residual"][i]), ("grid_rmse", ev["grid_rmse"][i])):
                h[key].append(value)
            if val < state.best[name]["val"]:
                state.best[name].update(val=val, step=state.step, params=st.params_of(i))
            sc = state.sched[name]
            if val < sc["best"] * (1.0 - threshold):
                sc["best"], sc["bad"] = val, 0
            else:
                sc["bad"] += 1
            if sc["bad"] > patience:
                new = lrs[i] * factor
                if lrs[i] - new > 1e-8:
                    lrs[i] = new
                sc["bad"] = 0
            if lrs[i] < min_lr or state.step >= max_steps:
                done.append(i)
        st.lr.copy_(torch.tensor(lrs, device=setup.device))
        if done:
            for i in done:
                on_finish(st, i, "min_lr" if lrs[i] < min_lr else "max_steps", state, elapsed())
            keep = [i for i in range(len(st)) if i not in done]
            del graph
            graph = None
            if not keep:
                break
            st = st.subset(keep)
            if setup.device.type == "cuda":
                torch.cuda.empty_cache()
            if use_graph and len(st) > 0:
                graph = capture(st, win)
        if len(st) > 0 and n_evals % checkpoint_every == 0:
            on_checkpoint(st, state, elapsed())
        if log_every and (n_evals % log_every == 0 or done):
            kinds = sorted({r[0] for r in st.runs})
            gm = lambda kind, key: 10 ** np.mean([np.log10(state.hist[run_name(*r)][key][-1]) for r in st.runs if r[0] == kind])
            print(f"step {state.step} ({elapsed() / 3600:.2f} h, {1e3 * elapsed() / state.step:.1f} ms/step avg) "
                  f"active {len(st)} | " + " | ".join(
                      f"{k}: val {gm(k, 'val_mse'):.2e} grid {gm(k, 'grid_rmse'):.2e} r_fp {gm(k, 'fixed_point_residual'):.1e}"
                      for k in kinds), flush=True)
    state.train_seconds = elapsed()
    return state
