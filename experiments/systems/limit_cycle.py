"""
Limit-cycle system of Appendix B (Brusselator-like model, a = 0.06, b = 0.6):

    dx/dt = -x + a y + x^2 y,      dy/dt = b - a y - x^2 y,

on K = [-1, 4]^2: one unstable equilibrium (b, b / (a + b^2)), surrounded by a stable limit cycle; the
equilibrium is the prescribed point of the model.
"""
import torch

A_LC, B_LC = 0.06, 0.6
DOMAIN = ((-1.0, 4.0), (-1.0, 4.0))
EQUILIBRIA = torch.tensor([[B_LC, B_LC / (A_LC + B_LC ** 2)]])
N_VISIBLE, M_HIDDEN = 2, 256


def field(x: torch.Tensor) -> torch.Tensor:
    X = x[..., 0]
    Y = x[..., 1]
    dX = -X + A_LC * Y + X * X * Y
    dY = B_LC - A_LC * Y - X * X * Y
    return torch.stack([dX, dY], dim=-1)
