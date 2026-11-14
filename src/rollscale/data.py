"""Nested training sets built from one pool of ground-truth trajectories.

All training sets are drawn from the same pool of ``n_pool`` Lennard-Jones
trajectories (temperatures spread evenly from 0.21 to 0.49, see
``eqxmlp.generate_dataset``). A set of ``n`` trajectories takes ``n`` of them
spaced evenly through the pool, so every set covers the temperature range as
evenly as its size allows, and ``n = 1`` uses the trajectory nearest the
middle temperature (0.35), which is also the temperature of the test starts.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from eqxmlp import Dataset, generate_dataset, initial_state

__all__ = ["TrajectoryPool", "make_pool", "subset_indices", "training_set", "test_starts"]


class TrajectoryPool(NamedTuple):
    """Frames grouped by trajectory: every array has leading axes ``(n_pool, frames)``."""

    q: jax.Array
    energy: jax.Array
    forces: jax.Array

    @property
    def n_traj(self) -> int:
        return self.q.shape[0]

    @property
    def frames_per_traj(self) -> int:
        return self.q.shape[1]


def make_pool(key, n_pool: int = 64, **dataset_kwargs) -> TrajectoryPool:
    """Generate ``n_pool`` labelled trajectories with ``eqxmlp.generate_dataset``."""
    d = generate_dataset(key, n_traj=n_pool, **dataset_kwargs)
    frames = len(d) // n_pool
    return TrajectoryPool(
        q=d.q.reshape(n_pool, frames, *d.q.shape[1:]),
        energy=d.energy.reshape(n_pool, frames),
        forces=d.forces.reshape(n_pool, frames, *d.forces.shape[1:]),
    )


def subset_indices(n: int, n_pool: int) -> np.ndarray:
    """``n`` trajectory indices spread evenly over the pool (centred bins)."""
    if not 1 <= n <= n_pool:
        raise ValueError(f"need 1 <= n <= {n_pool}, got {n}")
    return ((np.arange(n) + 0.5) * n_pool / n).astype(int)


def training_set(pool: TrajectoryPool, n: int) -> Dataset:
    """Flatten ``n`` evenly spaced trajectories from the pool into an ``eqxmlp.Dataset``."""
    idx = subset_indices(n, pool.n_traj)
    take = lambda x: x[idx].reshape(-1, *x.shape[2:])  # noqa: E731
    return Dataset(q=take(pool.q), energy=take(pool.energy), forces=take(pool.forces))


def test_starts(key, n: int = 16, temperature: float = 0.35):
    """``n`` held-out initial conditions, batched as one ``PhaseSpace`` of shape ``(n, N, dim)``."""
    states = [initial_state(k, temperature=temperature) for k in jax.random.split(key, n)]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)
