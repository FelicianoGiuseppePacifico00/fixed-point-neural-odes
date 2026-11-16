"""
Checks of the integrators of the trajectory experiment (Appendix C), on the data of seeds 0-2:
  - the float64 reference (RK4, dt = 1e-4) against dt = 5e-5;
  - the error of fixed-step RK4 with the training steps dt on the training windows, in float64 and float32
    (rms and max over all windows, max near the equilibria): dt = 0.0125 is the largest step whose error
    stays below the fixed-point residual of the QR model;
  - RK4 started exactly at the equilibria does not drift (explicit Runge-Kutta schemes keep exact equilibria).

Usage: python -m experiments.appC_trajectories.check_solver [--out DIR] [--device cpu]
Output: logs/solver_check.txt in results/ (or --out).
"""
import argparse
from pathlib import Path

import torch

from fpnode import get_device
from fpnode.data import trajectory_windows
from fpnode.integrators import rk4_observe
from experiments.systems import lotka_volterra as lv
from . import common as td

DTS = [0.05, 0.025, 0.02, 0.0125, 0.01]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(td.RESULTS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = get_device(args.device)
    out_dir = Path(args.out) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    say = lambda s: (print(s, flush=True), lines.append(s))

    x0 = torch.cat([td.initial_conditions(td.N_TRAJ, s) for s in [0, 1, 2]])
    ref = td.reference(x0, device=device)
    ref_half = td.reference(x0, dt_ref=td.DT_REF / 2, device=device)
    say(f"reference RK4 float64 dt={td.DT_REF:g} vs dt={td.DT_REF / 2:g} ({len(x0)} trajectories, T={td.T_OBS:g}): "
        f"max {float((ref - ref_half).norm(dim=-1).max()):.1e}")
    say(f"states: min {float(ref.min()):.3f}, max {float(ref.max()):.3f} (box [{td.IC_BOX[0][0]:g}, {td.IC_BOX[0][1]:g}]^2)")

    w0, wt = trajectory_windows(ref, td.H_WINDOW)
    near = torch.cdist(w0, lv.EQUILIBRIA.double()).min(dim=1).values <= lv.LOCAL_RADIUS
    say(f"{len(w0)} windows of {td.H_WINDOW} observations ({td.H_WINDOW * td.DT_OBS:g} time units), "
        f"{int(near.sum())} start within {lv.LOCAL_RADIUS} of an equilibrium")
    say("per-observation error ||RK4_dt(window) - reference|| over all windows (rms, max) and near the equilibria (max):")
    for dt in DTS:
        res = {}
        for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
            pred = rk4_observe(lv.field, w0.to(device, dtype), dt, td.H_WINDOW + 1, td.DT_OBS)[:, 1:].double().cpu()
            e = (pred - wt).norm(dim=-1)
            res[name] = (float(e.pow(2).mean().sqrt()), float(e.max()), float(e[near].max()))
        say(f"  dt={dt:<7g} ({int(round(td.DT_OBS / dt)):2d} steps/obs)  float64 rms {res['float64'][0]:.1e} "
            f"max {res['float64'][1]:.1e} near-eq max {res['float64'][2]:.1e}   |   float32 rms {res['float32'][0]:.1e} "
            f"max {res['float32'][1]:.1e} near-eq max {res['float32'][2]:.1e}")

    xb = lv.EQUILIBRIA.to(device).float()
    for dt in DTS:
        drift = (rk4_observe(lv.field, xb, dt, td.H_WINDOW + 1, td.DT_OBS) - xb[:, None]).norm(dim=-1).max()
        say(f"  RK4 from the equilibria, float32, dt={dt:g}: max drift over a window {float(drift):.1e}")
    (out_dir / "solver_check.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
