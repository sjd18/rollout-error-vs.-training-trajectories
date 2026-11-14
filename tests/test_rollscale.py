import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eqxmlp import LennardJones, PairPotentialMLP, forces

import rollscale as rs
from rollscale.plotting import aggregate, pick_families

LJ = LennardJones()


@pytest.fixture(scope="module")
def pool():
    return rs.make_pool(jax.random.PRNGKey(0), n_pool=8, n_steps=400, save_every=40, burn_in=1)


@pytest.fixture(scope="module")
def starts():
    return rs.test_starts(jax.random.PRNGKey(1), n=3)


def test_subset_indices_are_even_and_centred():
    np.testing.assert_array_equal(rs.subset_indices(1, 64), [32])
    np.testing.assert_array_equal(rs.subset_indices(4, 64), [8, 24, 40, 56])
    assert len(set(rs.subset_indices(64, 64))) == 64
    with pytest.raises(ValueError):
        rs.subset_indices(0, 64)


def test_training_set_size(pool):
    d = rs.training_set(pool, 2)
    assert len(d) == 2 * pool.frames_per_traj
    assert d.q.shape[1:] == (16, 2)


def test_rollout_error_is_zero_for_true_forces(starts):
    err = rs.rollout_error(lambda q: forces(LJ, q), starts)
    assert all(v < 1e-12 for v in err.values())


def test_rollout_error_scales_linearly_with_small_force_error(starts):
    """Before chaos saturates it, rollout error is proportional to the force error."""

    def scaled(eps):
        return lambda q: (1 + eps) * forces(LJ, q)

    e1 = rs.rollout_error(scaled(1e-4), starts, horizons=(0.5,))[0.5]
    e2 = rs.rollout_error(scaled(2e-4), starts, horizons=(0.5,))[0.5]
    assert e1 > 0
    np.testing.assert_allclose(e2 / e1, 2.0, rtol=0.05)


def test_rollout_error_grows_with_horizon(starts):
    err = rs.rollout_error(lambda q: 1.001 * forces(LJ, q), starts)
    h = sorted(err)
    assert err[h[0]] < err[h[1]] < err[h[2]]


def test_rollout_error_reports_blowup_as_inf(starts):
    err = rs.rollout_error(lambda q: forces(LJ, q) * jnp.nan, starts, horizons=(0.5,))
    assert math.isinf(err[0.5])


def test_horizon_must_match_sampling(starts):
    with pytest.raises(ValueError):
        rs.rollout_error(lambda q: forces(LJ, q), starts, horizons=(0.33,))


def test_sweep_writes_resumes_and_shards(tmp_path, monkeypatch):
    cfg = rs.SweepConfig(n_trajs=(1, 2), widths=(4,), seeds=(0,), steps=3, n_pool=4, n_test_starts=2)
    small_pool = rs.make_pool(jax.random.PRNGKey(0), n_pool=4, n_steps=400, save_every=40, burn_in=1)
    monkeypatch.setattr("rollscale.sweep.make_pool", lambda key, n_pool: small_pool)
    monkeypatch.setattr(
        "rollscale.sweep.generate_dataset",
        lambda key, n_traj: rs.training_set(small_pool, 1),
    )
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    rs.run_sweep(cfg, a, shard=(0, 2), verbose=False)
    rs.run_sweep(cfg, b, shard=(1, 2), done_from=[a], verbose=False)
    ra, rb = rs.read_results(a), rs.read_results(b)
    assert len(ra) == 1 and len(rb) == 1
    assert {ra[0]["n_traj"], rb[0]["n_traj"]} == {1, 2}
    assert set(ra[0]) == set(rs.FIELDS)
    assert isinstance(ra[0]["width"], int) and isinstance(ra[0]["rollout_t1"], float)
    # Shard 1 restarting after shard 0 finished still owns exactly its own run.
    assert len(rs.run_sweep(cfg, b, shard=(1, 2), done_from=[a], verbose=False)) == 1
    # Re-running finds nothing left to do.
    assert len(rs.run_sweep(cfg, a, shard=(0, 1), done_from=[b], verbose=False)) == 1


def test_power_law_slope_recovers_exponent():
    x = np.array([1, 2, 4, 8, 16.0])
    assert rs.power_law_slope(x, 3 * x**-0.75) == pytest.approx(-0.75)
    assert math.isnan(rs.power_law_slope(x[:1], x[:1]))


def test_pick_families_keeps_ends_and_caps_count():
    assert pick_families([4, 8, 16, 32, 64, 128]) == [4, 8, 16, 64, 128]
    assert pick_families([1, 2, 4, 8, 16, 32, 64]) == [1, 4, 8, 16, 64]
    assert pick_families([1, 2]) == [1, 2]


def fake_rows():
    rows = []
    for n in (1, 2, 4):
        for w in (4, 8):
            for s in (0, 1):
                e = 0.1 * n**-0.5 * w**-1.0 * (1 + 0.1 * s)
                rows.append({"n_traj": n, "width": w, "seed": s, "rollout_t1": e, "force_rmse": 10 * e})
    return rows


def test_aggregate_median_and_band():
    agg = aggregate(fake_rows(), "n_traj", "width", "rollout_t1")
    xs, med, lo, hi = agg[4]
    np.testing.assert_array_equal(xs, [1, 2, 4])
    assert np.all(lo <= med) and np.all(med <= hi)
    assert rs.power_law_slope(xs, med) == pytest.approx(-0.5)


def test_plots_are_written(tmp_path):
    rows = fake_rows()
    for fn in (rs.plot_scaling, rs.plot_heatmap, rs.plot_force_vs_rollout):
        out = fn(rows, tmp_path / f"{fn.__name__}.png")
        assert out.exists() and out.stat().st_size > 1000


def test_model_forces_plug_into_metric(starts):
    m = PairPotentialMLP(width=4, depth=1, key=jax.random.PRNGKey(0))
    err = rs.rollout_error(m.forces, starts, horizons=(0.5,))
    assert err[0.5] > 0
