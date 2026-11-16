"""
Setup shared by the scripts of the trajectory-training experiment (Appendix C): data, integrator and models.

Data: for each seed, N_TRAJ trajectories of the Lotka-Volterra field from initial conditions uniform in
IC_BOX = [0, 4]^2 (forward-invariant, contains the four equilibria), observed every DT_OBS up to T_OBS,
split by trajectory 80/20 into training and validation. The reference solution is RK4 in float64 with step
DT_REF (checked against DT_REF / 2 in check_solver.py). The data depend only on the seed.

Training: windows of H_WINDOW + 1 consecutive observations; the model integrates from the first one with
fixed-step RK4 (step DT_MODEL, SUBSTEPS steps per observation interval) and the loss compares the H_WINDOW
predicted states with the observations; gradients by backpropagation through the solver.
"""
from pathlib import Path

import torch

from fpnode import data
from fpnode.stacked import PARAMS, StackSetup
from experiments.systems import lotka_volterra as lv

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

IC_BOX = ((0.0, 4.0), (0.0, 4.0))
N_TRAJ = 100
VAL_FRACTION = 0.2
DT_OBS = 0.1
T_OBS = 10.0
N_OBS = int(round(T_OBS / DT_OBS)) + 1   # observations per trajectory, t = 0 included
DT_REF = 1e-4
H_WINDOW = 10                            # observations predicted by each window (1 time unit)
DATA_SEED_OFFSET = 7_000                 # data generator seed = DATA_SEED_OFFSET + seed
DT_MODEL = 0.0125                        # check_solver.py: max error 2.1e-6 on the windows
SUBSTEPS = int(round(DT_OBS / DT_MODEL))
BATCH_WINDOWS = 128
GRID_RES = 250                           # field RMSE on a grid of IC_BOX
SEEDS = list(range(10))
LAMBDAS = [0.01, 0.1, 1.0, 10.0]


def initial_conditions(n, seed):
    return data.uniform_points(n, IC_BOX, DATA_SEED_OFFSET + seed, dtype=torch.float64)


def reference(x0, dt_ref=DT_REF, n_obs=N_OBS, device=None):
    """Reference trajectories of the Lotka-Volterra field in float64: [N, n_obs, 2] on the CPU."""
    return data.reference_trajectories(lv.field, x0, dt_ref, n_obs, DT_OBS, device)


def run_name(kind, lam, seed):
    tag = f"lam{lam:g}" if kind == "penalty_mlp" else "lam0"
    return f"{kind}_ntraj{N_TRAJ}_{tag}_seed{seed}"


def all_runs():
    return ([("qr", 0.0, s) for s in SEEDS] + [("free_mlp", 0.0, s) for s in SEEDS]
            + [("penalty_mlp", lam, s) for lam in LAMBDAS for s in SEEDS])


def make_setup(device) -> StackSetup:
    return StackSetup(equilibria=lv.EQUILIBRIA.to(device).float(), grid=data.grid_points(IC_BOX, GRID_RES, device),
                      target_field=lv.field, dt=DT_MODEL, substeps=SUBSTEPS, horizon=H_WINDOW,
                      batch_windows=BATCH_WINDOWS, seeds=SEEDS, device=device)


def load_data(out_dir, device):
    """The trajectories of every seed (generated and saved to out_dir/data.pt the first time) and the
    training / validation windows, float32 on device: {"train": (x0 [S, Nw, 2], targets [S, Nw, h, 2]), "val": ...}."""
    path = Path(out_dir) / "data.pt"
    if path.exists():
        trajectories = torch.load(path)
    else:
        x0 = torch.cat([initial_conditions(N_TRAJ, s) for s in SEEDS])
        ref = reference(x0, device=device).reshape(len(SEEDS), N_TRAJ, N_OBS, 2)
        n_val = int(round(N_TRAJ * VAL_FRACTION))
        trajectories = {"seeds": SEEDS, "x0": x0.reshape(len(SEEDS), N_TRAJ, 2),
                        "train": ref[:, : N_TRAJ - n_val], "val": ref[:, N_TRAJ - n_val:],
                        "config": {"ic_box": list(IC_BOX[0]), "n_traj": N_TRAJ, "dt_obs": DT_OBS, "t_obs": T_OBS,
                                   "dt_ref": DT_REF, "val_fraction": VAL_FRACTION,
                                   "data_seed_offset": DATA_SEED_OFFSET}}
        torch.save(trajectories, path)
    win = {}
    for split in ("train", "val"):
        pairs = [data.trajectory_windows(trajectories[split][i], H_WINDOW) for i in range(len(SEEDS))]
        win[split] = (torch.stack([p[0] for p in pairs]).float().to(device),
                      torch.stack([p[1] for p in pairs]).float().to(device))
    return trajectories, win


def initial_params(kind, seed):
    """Initial parameters of a run: those of the models of the paper (system.build_model)."""
    model = lv.build_model(kind, seed)
    return {k: getattr(model, k).detach().float().cpu().clone() for k in PARAMS}


def save_model(run, params, path, device):
    """Saves the parameters of a run as a state_dict of the corresponding fpnode model."""
    model = lv.build_model(run[0], run[2], device)
    with torch.no_grad():
        for k in PARAMS:
            getattr(model, k).copy_(params[k].to(device))
    torch.save(model.state_dict(), path)
