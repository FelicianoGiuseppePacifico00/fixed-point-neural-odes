# fixed-point-neural-odes

Code for the paper *Exact Fixed-Point Constraints in Neural-ODEs: Injecting Prior Knowledge into the
Network Parameters*.

The library `fpnode` implements Neural-ODE vector fields

    F(x) = -x + A sigma(B x + b) + d,      A = Y S^+ + W (I - S S^+),

where S = [sigma(B x_1 + b), ..., sigma(B x_C + b)] are the hidden features at C prescribed points,
Y = [x_1 - d, ..., x_C - d], and S^+ is applied through a thin QR factorization of S. The prescribed points are
equilibria, F(x_l) = 0, for every value of the trainable parameters W, B, b, d, so the constraints hold exactly
at every step of training; the model is trained like an ordinary network.

## Quick start

```bash
conda env create -f environment.yml    # Python 3.10, PyTorch 2.7.1, NumPy, Matplotlib, Jupyter; installs fpnode
conda activate fpnode
python -m pytest tests                 # a few seconds
python reproduce.py figures            # every figure, table and number of the paper, from the stored results (~10 s)
python reproduce.py quick              # retrains Experiment 1 for one seed on a CPU and compares the models (~3 min)
```

Without conda: `pip install -e ".[test,tutorials]"` (Python >= 3.10). Everything runs on a CPU; a GPU is only
useful to retrain the experiments of the paper.

## Tutorials

* [`tutorials/01_quickstart.ipynb`](tutorials/01_quickstart.ipynb): how to use the library. A constrained model
  and an unconstrained network of the same architecture are trained on the Lotka-Volterra field of the paper and
  compared: residual at the equilibria, error over the domain, long-time trajectories (about 4 min on a CPU).
* [`tutorials/02_paper_results.ipynb`](tutorials/02_paper_results.ipynb): every figure, table and number of the
  paper from the stored results, in the order of the paper (about 10 s).

The notebooks are stored with their outputs and can be read on GitHub without running them.

## Using the library

```python
import torch
from fpnode import QRConstrainedMLP
from fpnode.losses import regression_loss


def target(x):                                    # the Lotka-Volterra field of the paper
    X, Y = x[:, 0], x[:, 1]
    return torch.stack([X * (3 - X) - 2 * X * Y, Y * (2 - Y) - X * Y], dim=1)


equilibria = torch.tensor([[0.0, 0.0], [0.0, 2.0], [3.0, 0.0], [1.0, 1.0]])
model = QRConstrainedMLP(n_visible=2, m_hidden=256, equilibria=equilibria)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(5000):
    x = 5 * torch.rand(500, 2) - 1                # samples in K = [-1, 4]^2
    loss = regression_loss(model, x, target(x))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"loss {loss.item():.1e}, max_l |F(x_l)| = {model.fixed_point_residual():.1e}")   # float32 round-off
```

| Module (`src/fpnode/`) | Content |
|---|---|
| `models.py` | `QRConstrainedMLP`, `PseudoinverseConstrainedMLP` (same family, explicit pseudoinverse), `FreeMLP` (unconstrained baseline) |
| `losses.py` | regression loss, soft penalty on the equilibria, loss on trajectories (backpropagation through RK4) |
| `training.py` | the training loops of the paper: fixed number of epochs; until the validation error stops improving |
| `integrators.py`, `data.py` | fixed-step RK4; sampling, train / validation split, augmentation, trajectory windows |
| `metrics.py`, `plotting.py` | errors on a grid, Jacobians, geometric statistics; figure and table helpers |
| `stacked.py` | many models trained together through the ODE solver, in one process (Appendix C) |

## Reproducing the paper

`reproduce.py` gives three levels of checks. Everything is written to `reproduced/`; the stored results are never
modified.

| Command | What it does | Time |
|---|---|---|
| `python reproduce.py figures` | figures, tables and numbers of the paper from the stored runs (JSON / NPZ), and a check that the regenerated tables are identical to those of the paper | ~10 s |
| `python reproduce.py evaluate` | recomputes the evaluations from the stored checkpoints of the trained models, then the tables and figures that depend on them, and compares the tables with those of the paper cell by cell | 15-30 min on a CPU |
| `python reproduce.py quick` | retrains Experiment 1 (seed 0, protocol of the paper) for the QR-constrained model, the free MLP and the penalized MLP, and compares them | ~3 min on a CPU |
| `python reproduce.py train <experiment>` | retrains one experiment from scratch, then evaluates and plots it | see below |

| Paper | Folder (`experiments/`) | Retraining (`reproduce.py train ...`) |
|---|---|---|
| Section 5.2, Experiment 1, Figs. 2-4 | `exp1_fixed_epochs/` | `exp1_fixed_epochs`: 50 runs, ~2 min each on a GPU |
| Section 5.2, dataset size, Figs. 5-6, Table 1 | `exp1_dataset_size/` | `exp1_dataset_size`: 290 runs, ~5 h on 28 CPU cores |
| Appendix B, limit cycle, Fig. 7 | `appB_limit_cycle/` | `appB_limit_cycle`: 1 run, ~20 min on a GPU |
| Appendix C, training on trajectories, Fig. 8, Tables 2-3 | `appC_trajectories/` | `appC_trajectories`: 60 runs in one process, ~35 h on a GPU |
| Numbers quoted in the text | `paper_numbers.py` | (from the stored results) |

### Layout of `experiments/`

```
systems/              target fields: lotka_volterra.py, limit_cycle.py
exp1_fixed_epochs/    train.py (one run), run_all.sh (all the runs), plot.py
exp1_dataset_size/    train.py, run_all.sh, select_hyperparameters.py, evaluate.py, plot.py, tools/
appB_limit_cycle/     train_and_plot.py
appC_trajectories/    common.py, train.py, run.sh, evaluate.py, check_solver.py, plot.py, tools/
paper_numbers.py      every number quoted in the text
quick.py              the check of `reproduce.py quick`
```

Each experiment folder has a `results/` folder (one JSON per run with history, configuration and final metrics;
the checkpoints of the trained models; logs) and a `figures/` folder (the figures and LaTeX tables of the paper).
`tools/` contains helpers used by `run_all.sh` or, once, during the original runs. Every script is run as a module
from the repository root and documents its options in its docstring and with `--help`, e.g.
`python -m experiments.exp1_fixed_epochs.train --model qr --seed 0 --schedule cosine`.

### Notes

* **Devices.** `--device`, or the environment variable `FP_DEVICE`, otherwise CUDA if available.
  `reproduce.py figures`, `evaluate` and `quick` use the CPU unless `--device` is given.
* **CPU and GPU.** The stored dataset-size evaluation and Fig. 2 were computed on a GPU, the other evaluations on
  a CPU. On a different device, the quantities that are themselves at float32 round-off level change slightly:
  the trajectory error at t = 30 of the QR model (about 4e-6, last column of Table 1) can shift by up to about
  20%, and so can the fixed-point residual of a retrained QR model.
* **Initializations.** The constrained and the free models use different initializations of the hidden layer
  (`fpnode/models.py`): random features B ~ N(0, 1), b ~ U(-3s, 3s) for the constrained models, so that S is
  well conditioned; an orthogonal B and b = 0 for the free MLP. These are the settings of the results of the
  paper.
* **Grid RMSE.** `fpnode.metrics` computes it in two equivalent ways that differ only by float32 round-off. Both
  are kept so that the stored results are reproduced exactly.
* **Limit cycle.** Fig. 7 was produced by an earlier notebook version of `appB_limit_cycle/train_and_plot.py`
  without a fixed seed, so a rerun gives an equivalent but not identical figure.
* **Trajectory experiment.** In the original run, the penalty weight lambda = 10 was stopped before convergence at
  step 1.18e6, with `python -m experiments.appC_trajectories.tools.drop_runs penalty_mlp:10:all`, because its
  validation error was 20 times that of lambda = 0.1; a retraining from scratch runs it until its own stopping
  criterion.
* **Checkpoints.** Only the checkpoints needed by the figures and the evaluations are versioned (the
  best-validation models, the Experiment 1 models of Fig. 2 and the trajectory data, 22 MB in total);
  last-state checkpoints and training snapshots are not.
