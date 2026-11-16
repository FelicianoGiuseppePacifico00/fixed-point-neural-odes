"""Fixed-step explicit Runge-Kutta integration of a vector field F(x) (x: [..., n])."""
import torch


def rk4_step(F, x, dt):
    """One classical RK4 step."""
    k1 = F(x)
    k2 = F(x + 0.5 * dt * k1)
    k3 = F(x + 0.5 * dt * k2)
    k4 = F(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_observe(F, x0, dt, n_obs, dt_obs):
    """Integrates from x0 [..., n] with step dt and returns the states at t = 0, dt_obs, ..., (n_obs - 1) dt_obs,
    stacked along a new dimension -2: [..., n_obs, n]. dt_obs must be a multiple of dt."""
    sub = int(round(dt_obs / dt))
    assert abs(sub * dt - dt_obs) < 1e-12, (dt, dt_obs)
    x, out = x0, [x0]
    for _ in range(n_obs - 1):
        for _ in range(sub):
            x = rk4_step(F, x, dt)
        out.append(x)
    return torch.stack(out, dim=-2)


@torch.no_grad()
def rk4_trajectory(F, x0, dt, n_steps, save_every):
    """n_steps RK4 steps from x0 [N, n]; the states every save_every steps (t = 0 included) as a float64 CPU
    tensor [n_saved, N, n]."""
    x, out = x0, [x0.double().cpu()]
    for i in range(1, n_steps + 1):
        x = rk4_step(F, x, dt)
        if i % save_every == 0:
            out.append(x.double().cpu())
    return torch.stack(out)
