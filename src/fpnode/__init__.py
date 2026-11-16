"""fpnode: Neural-ODE vector fields with exactly prescribed equilibria.

Modules (general, for any vector field):
  models       shallow residual fields F(x) = -x + A sigma(B x + b) + d: QR- and pseudoinverse-constrained, free
  data         sampling on a box, train / validation split, augmentation, trajectory datasets
  integrators  fixed-step RK4
  losses       regression loss, fixed-point penalty, trajectory loss
  training     fixed-epoch and train-until-converged loops
  stacked      many models trained together through the ODE solver
  metrics      grid / local errors, Jacobians, geometric statistics
  plotting     figure style, geometric bands, LaTeX table cells, field comparison figure
"""
from .models import FreeMLP, PseudoinverseConstrainedMLP, QRConstrainedMLP, make_model  # noqa: F401
from .utils import get_device, set_seed  # noqa: F401

__version__ = "0.1.0"
