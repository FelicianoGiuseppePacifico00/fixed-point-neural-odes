"""Training losses."""
import torch

from .integrators import rk4_step


def fixed_point_penalty(model) -> torch.Tensor:
    """mean_l ||F(x_l)||^2 over the prescribed points of the model."""
    f = model.field(model.x_attractors.t())
    return (f ** 2).sum(dim=1).mean()


def regression_loss(model, x: torch.Tensor, y: torch.Tensor, penalty_lambda: float = 0.0) -> torch.Tensor:
    """Vector-field regression: mean over samples and components of (F(x) - y)^2, plus
    penalty_lambda * fixed_point_penalty when penalty_lambda > 0."""
    loss = torch.mean((model(x) - y) ** 2)
    if penalty_lambda > 0:
        loss = loss + penalty_lambda * fixed_point_penalty(model)
    return loss


def trajectory_loss(model, x0: torch.Tensor, targets: torch.Tensor, dt: float, substeps: int,
                    penalty_lambda: float = 0.0) -> torch.Tensor:
    """Training through the solver on windows: from the start states x0 [N, n], RK4 with step dt
    (substeps steps per observation interval) predicts the h observations targets [N, h, n]; loss = mean over
    windows, observations and components of the squared error (+ the fixed-point penalty). Single-model
    version of fpnode.stacked.per_model_loss."""
    A = model.output_matrix()
    F = lambda z: model.field(z, A)
    x, out = x0, []
    for _ in range(targets.shape[1]):
        for _ in range(substeps):
            x = rk4_step(F, x, dt)
        out.append(x)
    loss = (torch.stack(out, dim=1) - targets).pow(2).mean()
    if penalty_lambda > 0:
        loss = loss + penalty_lambda * model.field(model.x_attractors.t(), A).pow(2).sum(-1).mean()
    return loss
