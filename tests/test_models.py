"""Quick checks of the library (CPU, a few seconds): python -m pytest tests"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpnode import FreeMLP, PseudoinverseConstrainedMLP, QRConstrainedMLP, set_seed  # noqa: E402
from fpnode import stacked  # noqa: E402
from fpnode.losses import trajectory_loss  # noqa: E402

EQ = torch.tensor([[0.0, 0.0], [0.0, 2.0], [3.0, 0.0], [1.0, 1.0]])


def perturbed(model, seed=1, scale=0.5):
    """The model with random parameters (the constraints must not depend on them)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


@pytest.mark.parametrize("cls", [QRConstrainedMLP, PseudoinverseConstrainedMLP])
@pytest.mark.parametrize("dim,C", [(2, 4), (5, 3), (10, 11)])
def test_equilibria_are_exact(cls, dim, C):
    """F(x_l) = 0 at the prescribed points for any value of the parameters (float64)."""
    set_seed(0)
    points = 2 * torch.rand(C, dim) - 1
    model = perturbed(cls(dim, 64, points).double())
    with torch.no_grad():
        assert model(model.equilibria).abs().max() < 1e-10
    assert model.fixed_point_residual() < 1e-10


def test_qr_equals_pseudoinverse():
    """For the same parameters the QR and the pseudoinverse parametrizations give the same field."""
    set_seed(0)
    qr = perturbed(QRConstrainedMLP(2, 64, EQ).double())
    pinv = PseudoinverseConstrainedMLP(2, 64, EQ).double()
    pinv.load_state_dict(qr.state_dict())
    x = 5 * torch.rand(1000, 2, dtype=torch.float64) - 1
    with torch.no_grad():
        assert torch.allclose(qr(x), pinv(x), atol=1e-9)


def test_free_mlp_is_unconstrained():
    set_seed(0)
    model = FreeMLP(2, 64, EQ)
    assert torch.equal(model.output_matrix(), model.W)
    assert model.fixed_point_residual() > 1e-3


@pytest.mark.parametrize("cls", [QRConstrainedMLP, FreeMLP])
def test_jacobian_matches_autograd(cls):
    set_seed(0)
    model = perturbed(cls(2, 32, EQ).double())
    x = torch.rand(5, 2, dtype=torch.float64)
    J_auto = torch.stack([torch.autograd.functional.jacobian(lambda z: model(z[None])[0], xi) for xi in x])
    assert torch.allclose(model.jacobian(x), J_auto, atol=1e-10)


def test_stacked_models_match_single_models():
    """The batched fields of fpnode.stacked (several models in one stack) equal the single models, and the
    stacked trajectory loss equals losses.trajectory_loss."""
    set_seed(0)
    models = [QRConstrainedMLP(2, 32, EQ), QRConstrainedMLP(2, 32, EQ), FreeMLP(2, 32, EQ)]
    models = [perturbed(m, seed=i).double() for i, m in enumerate(models)]
    P = {k: torch.stack([getattr(m, k).detach() for m in models]) for k in stacked.PARAMS}
    A = stacked.output_matrix(P, 2, EQ.double())
    x = 5 * torch.rand(3, 100, 2, dtype=torch.float64) - 1
    with torch.no_grad():
        F = stacked.field(P, A, x)
        for i, m in enumerate(models):
            assert torch.allclose(F[i], m(x[i]), atol=1e-10)

    setup = stacked.StackSetup(equilibria=EQ.double(), grid=x[0], target_field=lambda z: z, dt=0.05, substeps=2,
                               horizon=3, batch_windows=4, seeds=[0], device=torch.device("cpu"))
    x0 = torch.rand(3, 7, 2, dtype=torch.float64)
    tgt = torch.rand(3, 7, 3, 2, dtype=torch.float64)
    lam = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)
    loss = stacked.per_model_loss(_DoubleStack(P, 2, setup, lam), x0, tgt)
    for i, m in enumerate(models):
        ref = trajectory_loss(m, x0[i], tgt[i], dt=0.05, substeps=2, penalty_lambda=float(lam[i]))
        assert torch.allclose(loss[i], ref, atol=1e-12)


class _DoubleStack:
    """The attributes of a fpnode.stacked.Stack used by per_model_loss, in float64 (Stack itself is float32)."""

    def __init__(self, P, nq, setup, lam):
        self.P, self.nq, self.setup, self.lam = P, nq, setup, lam

    def __len__(self):
        return len(self.lam)
