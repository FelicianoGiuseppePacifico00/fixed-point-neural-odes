"""
Lotka-Volterra competition model of Section 5 and Appendix C of the paper:

    dx/dt = x (3 - x) - 2 x y,      dy/dt = y (2 - y) - x y,

on K = [-1, 4]^2, with the four equilibria (0, 0) unstable node, (0, 2) and (3, 0) stable nodes,
(1, 1) saddle, which are the prescribed points of every model.
"""
import numpy as np
import torch

from fpnode import make_model, set_seed

DOMAIN = ((-1.0, 4.0), (-1.0, 4.0))
EQUILIBRIA = torch.tensor([[0.0, 0.0], [0.0, 2.0], [3.0, 0.0], [1.0, 1.0]])
EQUILIBRIUM_NAMES = ["(0,0)", "(0,2)", "(3,0)", "(1,1)"]
EQUILIBRIUM_TYPES = ["unstable node", "stable node", "stable node", "saddle"]
N_VISIBLE, M_HIDDEN = 2, 256
GRID_RES = 250        # evaluation grid on K (grid RMSE)
LOCAL_RADIUS = 0.25   # "near an equilibrium"


def field(x: torch.Tensor) -> torch.Tensor:
    X = x[..., 0]
    Y = x[..., 1]
    dX = X * (3 - X) - 2 * X * Y
    dY = Y * (2 - Y) - X * Y
    return torch.stack([dX, dY], dim=-1)


def jacobian(p) -> np.ndarray:
    x, y = p
    return np.array([[3 - 2 * x - 2 * y, -2 * x], [-y, 2 - 2 * y - x]])


# model types of the experiments -> model class (the penalty and the augmentation only change the training)
MODEL_KINDS = {"qr": "qr", "free_mlp": "free", "penalty_mlp": "free", "aug_mlp": "free"}


def build_model(model_type: str, seed: int, device=torch.device("cpu")):
    """The model of the experiments, initialized from the seed (created on the CPU, then moved)."""
    set_seed(seed)
    return make_model(MODEL_KINDS[model_type], N_VISIBLE, M_HIDDEN, EQUILIBRIA).to(device)
