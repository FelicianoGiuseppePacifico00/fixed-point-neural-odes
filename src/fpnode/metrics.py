"""
Error metrics for a learned field F against a target field (both: x [N, n] -> [N, n]).

The grid RMSE sqrt(mean_x ||F(x) - F_target(x)||^2) is computed in two equivalent ways, which differ only
by float32 round-off: ``grid_rmse`` sums the squared components (used for the per-epoch monitoring of the
fixed-epoch runs), ``error_norms`` + ``rms`` square the pointwise norms (used everywhere else). Both are
kept so that the stored results are reproduced bit for bit.
"""
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def grid_rmse(F, target, points: torch.Tensor, chunk_size: int = 65536) -> float:
    """sqrt(mean over the points of sum_i (F_i - target_i)^2)."""
    sq = torch.cat([(F(c) - target(c)).pow(2).sum(dim=1) for c in points.split(chunk_size)])
    return float(sq.mean().sqrt().cpu())


@torch.no_grad()
def error_norms(F, target, points: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
    """Pointwise errors ||F(x) - target(x)||_2, [N]."""
    return torch.cat([(F(c) - target(c)).norm(dim=1) for c in points.split(chunk_size)])


def rms(e: torch.Tensor) -> float:
    """Root mean square of pointwise errors."""
    return float(e.pow(2).mean().sqrt())


@torch.no_grad()
def mean_squared_error(F, x: torch.Tensor, y: torch.Tensor, chunk_size: int = 65536) -> float:
    """Mean over samples and components of (F(x) - y)^2 (the normalization of the regression loss)."""
    sq = sum(float((F(xc) - yc).pow(2).sum()) for xc, yc in zip(x.split(chunk_size), y.split(chunk_size)))
    return sq / y.numel()


def near(points: torch.Tensor, centers: torch.Tensor, radius: float) -> torch.Tensor:
    """Mask of the points within radius of at least one center."""
    return torch.cdist(points, centers).min(dim=1).values <= radius


def nearest(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Index of the nearest center of every point (x is cast to float32, centers must be float32 on x's device)."""
    return torch.cdist(x.float(), centers).argmin(dim=1)


def geometric_mean(a) -> float:
    return float(10 ** np.mean(np.log10(np.asarray(a, dtype=float))))


def geometric_stats(a):
    """Geometric mean and geometric standard deviation (10^std of log10, ddof = 1)."""
    log = np.log10(np.asarray(a, dtype=float))
    return 10 ** log.mean(), 10 ** log.std(ddof=1)


# ----------------------------------------------------------------------------- Jacobians (float64, NumPy)
@torch.no_grad()
def numpy_field_and_jacobian(model):
    """F(x) and DF(x) = -I + A diag(sigma'(B x + b)) B of a sigmoid model as float64 NumPy functions of x [n]."""
    if not isinstance(model.activation, nn.Sigmoid):
        raise NotImplementedError("numpy_field_and_jacobian supports the sigmoid only")
    A = model.output_matrix().double().cpu().numpy()
    B = model.B.double().cpu().numpy()
    b = model.b.double().cpu().numpy().reshape(-1)
    d = model.d.double().cpu().numpy().reshape(-1)
    eye = np.eye(len(d))
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    F = lambda x: -x + A @ sig(B @ x + b) + d

    def J(x):
        s = sig(B @ x + b)
        return -eye + (A * (s * (1 - s))) @ B
    return F, J


def newton(F, J, x, iters: int = 50, tol: float | None = None):
    """Newton iteration for F(x) = 0 from x; stops early when the step is below tol (if given)."""
    for _ in range(iters):
        step = np.linalg.solve(J(x), F(x))
        x = x - step
        if tol is not None and np.linalg.norm(step) < tol:
            break
    return x


def sorted_eigenvalues(M):
    lam = np.linalg.eigvals(M)
    return lam[np.argsort(lam.real)]


def jacobian_comparison(J_learned, J_true) -> dict:
    """Errors of a learned Jacobian: spectral-norm error (absolute and relative), eigenvalue errors (both
    spectra sorted by real part) and whether the stability type (signs of the real parts) is the same."""
    lt, ll = sorted_eigenvalues(J_true), sorted_eigenvalues(J_learned)
    return {"dJ": (J_learned - J_true).tolist(), "eig_true": [float(v) for v in lt.real],
            "jac_err": float(np.linalg.norm(J_learned - J_true, 2)),
            "jac_rel": float(np.linalg.norm(J_learned - J_true, 2) / np.linalg.norm(J_true, 2)),
            "eig_err": float(np.max(np.abs(ll - lt))),
            "eig_rel": float(np.max(np.abs(ll - lt) / np.abs(lt))),
            "eig_complex": bool(np.any(np.abs(ll.imag) > 0)),
            "stable_ok": bool(np.array_equal(np.sign(ll.real), np.sign(lt.real))),
            "eig_learned": [float(v) for v in ll.real]}
