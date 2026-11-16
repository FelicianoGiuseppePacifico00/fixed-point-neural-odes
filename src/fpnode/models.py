"""
Shallow residual vector fields with prescribed equilibria.

Every model is a vector field on R^n of the form

    F(x) = -x + A sigma(B x + b) + d,        A in R^{n x m}, d in R^n, B in R^{m x n}, b in R^m,

with C prescribed points x_1, ..., x_C (stored as the buffer ``x_attractors``, shape [n, C]).

* ``FreeMLP``: A = W, no constraint (the prescribed points are only stored, e.g. for a penalty).
* ``QRConstrainedMLP`` and ``PseudoinverseConstrainedMLP``: A is built from the free parameter W so that
  F(x_l) = 0 holds exactly for every value of the parameters. With the feature matrix
  S = [sigma(B x_1 + b), ..., sigma(B x_C + b)] in R^{m x C} and Y = [x_1 - d, ..., x_C - d], the
  constraints read A S = Y, and every solution is

      A = Y S^+ + W (I - S S^+).

  The QR model computes this with a thin QR factorization S = Q R (A = Z Q^T + W - (W Q) Q^T with
  Z R = Y); the pseudoinverse model forms S^+ explicitly. Both give the same field for the same
  parameters when S has full column rank.

The initializations are those used for the results of the paper:
  constrained models: W ~ Xavier-uniform (gain 0.3), d = 0, then B ~ N(0, 1), b ~ U(-3 s, 3 s) with
                      s = max(1, max |x_l|), redrawn until sigma_min(S) > tol;
  free MLP:           W ~ Xavier-uniform (gain 0.3), d = 0, B orthogonal (gain 1), b = 0.
"""
import torch
import torch.nn as nn
from torch.nn import init

# Parameter names of older checkpoints (W1 -> W, b1 -> d, W2 -> B, b2 -> b) and buffers that older
# versions of the constrained models stored but never used: renamed / dropped on load.
_LEGACY_NAMES = {"W1": "W", "b1": "d", "W2": "B", "b2": "b"}
_UNUSED_BUFFERS = ("identity_n_visible", "identity_m_hidden")


def _load_hook(state_dict, prefix, *args):
    for old, new in _LEGACY_NAMES.items():
        if prefix + old in state_dict and prefix + new not in state_dict:
            state_dict[prefix + new] = state_dict.pop(prefix + old)
    for name in _UNUSED_BUFFERS:
        state_dict.pop(prefix + name, None)


class ResidualMLPField(nn.Module):
    """F(x) = -x + A sigma(B x + b) + d; subclasses define the output matrix A."""

    def __init__(self, n_visible: int, m_hidden: int, equilibria: torch.Tensor, activation: nn.Module | None = None):
        super().__init__()
        self._register_load_state_dict_pre_hook(_load_hook)
        X = equilibria.detach()
        if X.dim() != 2 or X.shape[1] != n_visible:
            raise ValueError(f"equilibria must have shape [C, {n_visible}], got {tuple(X.shape)}")
        if X.shape[0] > 1:
            D = torch.cdist(X.float(), X.float())
            D.fill_diagonal_(float("inf"))
            if not float(D.min()) > 1e-6:
                raise ValueError("the prescribed points must be distinct")
        self.n_visible, self.m_hidden, self.num_equilibria = n_visible, m_hidden, X.shape[0]
        self.W = nn.Parameter(torch.empty(n_visible, m_hidden))
        self.d = nn.Parameter(torch.empty(n_visible))
        self.B = nn.Parameter(torch.empty(m_hidden, n_visible))
        self.b = nn.Parameter(torch.empty(m_hidden))
        self.activation = activation if activation is not None else nn.Sigmoid()
        self.register_buffer("x_attractors", X.t().contiguous().to(dtype=self.W.dtype, device=self.W.device))  # [n, C]

    @property
    def equilibria(self) -> torch.Tensor:
        """The prescribed points, [C, n]."""
        return self.x_attractors.t()

    def output_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def field(self, x: torch.Tensor, A: torch.Tensor | None = None) -> torch.Tensor:
        """F(x) for x of shape [N, n]; A can be passed to avoid rebuilding it."""
        if A is None:
            A = self.output_matrix()
        return -x + self.activation(x @ self.B.T + self.b) @ A.T + self.d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.field(x)

    def field_fn(self):
        """F as a function of x with the output matrix computed once (for evaluation and integration)."""
        A = self.output_matrix()
        return lambda x: self.field(x, A)

    @torch.no_grad()
    def fixed_point_residual(self) -> float:
        """max_l ||F(x_l)||_2 over the prescribed points."""
        A = self.output_matrix()
        x = self.x_attractors.t()
        F = -x + self.activation(x @ self.B.t() + self.b) @ A.t() + self.d
        return float(F.norm(dim=1).max().cpu().item())

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """DF(x) = -I + A diag(sigma'(B x + b)) B for x of shape [N, n] (elementwise activation): [N, n, n]."""
        A = self.output_matrix()
        z = (x @ self.B.T + self.b).detach().requires_grad_(True)
        with torch.enable_grad():
            dsig = torch.autograd.grad(self.activation(z).sum(), z)[0]  # [N, m]
        eye = torch.eye(self.n_visible, dtype=x.dtype, device=x.device)
        return -eye + (A[None] * dsig[:, None, :]) @ self.B


class FreeMLP(ResidualMLPField):
    """Unconstrained field: A = W."""

    def __init__(self, n_visible, m_hidden, equilibria, activation=None):
        super().__init__(n_visible, m_hidden, equilibria, activation)
        init.xavier_uniform_(self.W, gain=0.3)
        init.zeros_(self.d)
        init.orthogonal_(self.B, gain=1.0)
        init.zeros_(self.b)

    def output_matrix(self) -> torch.Tensor:
        return self.W


class _ConstrainedMLP(ResidualMLPField):
    """Shared part of the constrained models: feature matrix S and initialization."""

    def __init__(self, n_visible, m_hidden, equilibria, activation=None, init_tol: float = 1e-4,
                 init_max_tries: int = 50):
        super().__init__(n_visible, m_hidden, equilibria, activation)
        if m_hidden < self.num_equilibria:
            raise ValueError(f"need m_hidden >= number of prescribed points, got {m_hidden} < {self.num_equilibria}")
        init.xavier_uniform_(self.W, gain=0.3)
        init.zeros_(self.d)
        self._init_hidden_layer(init_max_tries, init_tol)

    def feature_matrix(self) -> torch.Tensor:
        """S = [sigma(B x_1 + b), ..., sigma(B x_C + b)], [m, C]."""
        Xc = self.x_attractors.t()
        s = self.activation(Xc @ self.B.t() + self.b)
        return s.t().contiguous()

    @torch.no_grad()
    def _init_hidden_layer(self, max_tries, tol):
        """Random-feature initialization of (B, b), redrawn until sigma_min(S) > tol (rank(S) = C)."""
        for _ in range(max_tries):
            self.B.normal_(mean=0.0, std=1.0)
            coord_scale = float(self.x_attractors.abs().max().clamp_min(1.0))
            self.b.uniform_(-3.0 * coord_scale, 3.0 * coord_scale)
            S = self.feature_matrix().double()
            if float(torch.linalg.eigvalsh(S.t() @ S).min().cpu().item()) > tol * tol:
                return
        raise RuntimeError(f"could not draw a hidden layer with rank(S) = C in {max_tries} tries")


class QRConstrainedMLP(_ConstrainedMLP):
    """A S = Y exactly, via the thin QR factorization of S (no explicit pseudoinverse)."""

    def __init__(self, n_visible, m_hidden, equilibria, activation=None, ridge: float = 0.0, **kwargs):
        super().__init__(n_visible, m_hidden, equilibria, activation, **kwargs)
        self.ridge = float(ridge)  # optional diagonal added to R (0 in all the experiments)

    def output_matrix(self) -> torch.Tensor:
        S = self.feature_matrix()                                    # [m, C]
        Y = self.x_attractors - self.d.unsqueeze(1)                  # [n, C]
        Q, R = torch.linalg.qr(S, mode="reduced")                    # Q [m, C], R [C, C]
        if self.ridge > 0.0:
            R = R + self.ridge * torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
        Z = torch.linalg.solve_triangular(R.T, Y.T, upper=False).T  # Z R = Y, [n, C]
        A_part = Z @ Q.T                                             # A_part S = Y
        W_perp = self.W - (self.W @ Q) @ Q.T                         # W (I - Q Q^T)
        return A_part + W_perp


class PseudoinverseConstrainedMLP(_ConstrainedMLP):
    """A = Y S^+ + W (I - S S^+), with S^+ the Moore-Penrose pseudoinverse."""

    def __init__(self, n_visible, m_hidden, equilibria, activation=None, rcond: float = 1e-6, **kwargs):
        super().__init__(n_visible, m_hidden, equilibria, activation, **kwargs)
        self.rcond = float(rcond)

    def output_matrix(self) -> torch.Tensor:
        S = self.feature_matrix()
        Y = self.x_attractors - self.d.unsqueeze(1)
        S_pinv = torch.linalg.pinv(S, rcond=self.rcond)              # [C, m]
        eye = torch.eye(self.m_hidden, dtype=S.dtype, device=S.device)
        return Y @ S_pinv + self.W @ (eye - S @ S_pinv)


MODELS = {"qr": QRConstrainedMLP, "pinv": PseudoinverseConstrainedMLP, "free": FreeMLP}


def make_model(kind: str, n_visible: int, m_hidden: int, equilibria: torch.Tensor, activation=None, **kwargs):
    """kind: 'qr', 'pinv' or 'free'. The model is created on the CPU (move it with .to(device))."""
    return MODELS[kind](n_visible, m_hidden, equilibria, activation, **kwargs)
