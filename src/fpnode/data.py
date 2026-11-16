"""
Training and evaluation data for a vector field on a box K = [l_1, h_1] x ... x [l_n, h_n]
(``box`` is a sequence of (low, high) pairs, one per coordinate).
"""
import torch

from .integrators import rk4_observe
from .utils import set_seed


def sample_uniform(n: int, box, seed: int) -> torch.Tensor:
    """n points uniform in the box, float32 [n, dim]. Seeds the global generators (set_seed), so the data
    depend only on the seed."""
    set_seed(seed)
    x = torch.empty(n, len(box), dtype=torch.float32)
    for i, (low, high) in enumerate(box):
        x[:, i].uniform_(low, high)
    return x


def uniform_points(n: int, box, seed: int, dtype=torch.float32) -> torch.Tensor:
    """n points uniform in the box from a private generator (test sets, initial conditions): the global
    random state is not touched."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, len(box), generator=g, dtype=dtype)
    for i, (low, high) in enumerate(box):
        x[:, i] = low + (high - low) * x[:, i]
    return x


def train_val_split(x: torch.Tensor, val_fraction: float):
    """First (1 - val_fraction) of the samples for training, the rest for validation."""
    n_val = int(round(len(x) * val_fraction))
    return x[: len(x) - n_val], x[len(x) - n_val:]


def augment_with_points(x_fit: torch.Tensor, points: torch.Tensor, rho: float):
    """Appends copies of the prescribed points so that they make up a fraction rho of the training set
    (at least one copy each). Returns the augmented inputs and the number of added samples."""
    C = points.shape[0]
    reps = max(1, round(rho / (1 - rho) * len(x_fit) / C))
    return torch.cat([x_fit, points.to(x_fit.device).repeat(reps, 1)]), reps * C


def grid_points(box, res: int, device=None) -> torch.Tensor:
    """Uniform res x res grid on a 2-D box, [res^2, 2]."""
    (x_low, x_high), (y_low, y_high) = box
    xs = torch.linspace(x_low, x_high, res, device=device)
    ys = torch.linspace(y_low, y_high, res, device=device)
    Yg, Xg = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((Xg.reshape(-1), Yg.reshape(-1)), dim=1)


@torch.no_grad()
def reference_trajectories(F, x0: torch.Tensor, dt: float, n_obs: int, dt_obs: float, device=None) -> torch.Tensor:
    """Trajectories of F from x0 [N, n] in float64 (RK4, step dt), observed every dt_obs: [N, n_obs, n] on the CPU."""
    return rk4_observe(F, x0.to(device, torch.float64), dt, n_obs, dt_obs).cpu()


def trajectory_windows(traj: torch.Tensor, h: int):
    """All the windows of h + 1 consecutive observations of trajectories [N, n_obs, n]: start states
    [N * w, n] and the h following observations [N * w, h, n], w = n_obs - h windows per trajectory."""
    n, n_obs, dim = traj.shape
    starts = torch.arange(n_obs - h)
    idx = starts[:, None] + torch.arange(1, h + 1)[None, :]
    return traj[:, starts].reshape(-1, dim), traj[:, idx].reshape(-1, h, dim)
