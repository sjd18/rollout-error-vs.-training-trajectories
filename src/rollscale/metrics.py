"""Rollout error: how far a learned-force simulation strays from the true one.

From each held-out start, the system is integrated twice with the same
velocity-Verlet code, once with the true forces and once with the model's
forces, and the RMS position difference is recorded over time.

Molecular dynamics is chaotic, so any force error eventually grows until the
two runs are unrelated and the difference saturates at the cluster's size
(about 0.15 to 0.2 here, reached by t ≈ 5). Before that, the error reflects
model quality. The default horizons (t = 0.5, 1, 2.5) sit in that regime.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from eqxmlp import LennardJones, PhaseSpace, forces, simulate

__all__ = ["DEFAULT_HORIZONS", "rollout_rms_curve", "rollout_error"]

DEFAULT_HORIZONS = (0.5, 1.0, 2.5)


@eqx.filter_jit
def rollout_rms_curve(
    model_force, true_force, starts: PhaseSpace, *, dt: float, n_steps: int, save_every: int
):
    """RMS position difference over time, for a batch of starts.

    Returns ``(t, err)`` with ``err`` of shape ``(n_starts, n_saved)``.
    """

    def one(s):
        a = simulate(true_force, s, dt=dt, n_steps=n_steps, save_every=save_every)
        b = simulate(model_force, s, dt=dt, n_steps=n_steps, save_every=save_every)
        return a.t, jnp.sqrt(jnp.mean((a.q - b.q) ** 2, axis=(-2, -1)))

    t, err = jax.vmap(one)(starts)
    return t[0], err


def rollout_error(
    model_force,
    starts: PhaseSpace,
    *,
    horizons=DEFAULT_HORIZONS,
    dt: float = 0.005,
    save_every: int = 20,
    potential=LennardJones(),
) -> dict[float, float]:
    """Mean (over starts) RMS position error at each horizon time.

    Non-finite errors (a model whose simulation blew up) are reported as
    ``inf`` rather than silently averaged.
    """
    step_time = dt * save_every
    idx = [round(h / step_time) for h in horizons]
    if any(abs(i * step_time - h) > 1e-9 for i, h in zip(idx, horizons, strict=True)):
        raise ValueError("each horizon must be a multiple of dt * save_every")
    true_force = lambda q: forces(potential, q)  # noqa: E731
    _, err = rollout_rms_curve(
        model_force, true_force, starts, dt=dt, n_steps=max(idx) * save_every, save_every=save_every
    )
    out = {}
    for h, i in zip(horizons, idx, strict=True):
        e = err[:, i]
        out[h] = float(jnp.mean(e)) if bool(jnp.all(jnp.isfinite(e))) else float("inf")
    return out
