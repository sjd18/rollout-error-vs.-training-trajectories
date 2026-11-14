"""The grid sweep: train one model per (trajectories, width, seed) and score it.

Results are appended to a CSV after every run, and runs already present in
the file are skipped, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
from eqxmlp import PairPotentialMLP, evaluate, generate_dataset, train

from .data import make_pool, test_starts, training_set
from .metrics import DEFAULT_HORIZONS, rollout_error

__all__ = ["SweepConfig", "run_sweep", "read_results", "FIELDS"]


@dataclass(frozen=True)
class SweepConfig:
    n_trajs: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    widths: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
    seeds: tuple[int, ...] = (0, 1, 2)
    depth: int = 2
    steps: int = 2000
    batch_size: int = 64
    lr: float = 3e-3
    n_pool: int = 64
    n_test_starts: int = 16
    horizons: tuple[float, ...] = DEFAULT_HORIZONS
    data_key: int = 0
    test_key: int = 1000
    force_test_key: int = 100


FIELDS = [
    "n_traj",
    "n_frames",
    "width",
    "depth",
    "seed",
    "n_params",
    "steps",
    "final_loss",
    "force_rmse",
    "force_r2",
    *[f"rollout_t{h:g}" for h in DEFAULT_HORIZONS],
    "train_seconds",
]


def _key(row) -> tuple[int, int, int]:
    return int(row["n_traj"]), int(row["width"]), int(row["seed"])


def read_results(path) -> list[dict]:
    """Rows of a results CSV, with numeric fields converted to numbers."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [{k: _number(v) for k, v in r.items()} for r in rows]


def _number(v: str):
    try:
        return int(v)
    except ValueError:
        return float(v)


def run_sweep(
    cfg: SweepConfig, out_csv, *, shard: tuple[int, int] = (0, 1), done_from=(), verbose: bool = True
) -> list[dict]:
    """Run every grid point not already finished. Returns the rows in ``out_csv``.

    Args:
        shard: ``(i, k)`` runs only every ``k``-th grid point starting at
            ``i``, so ``k`` processes can split the grid between them.
        done_from: extra CSV files whose runs count as finished (e.g. the
            other shards' outputs).
    """
    if tuple(cfg.horizons) != DEFAULT_HORIZONS:
        raise ValueError(f"the CSV schema records horizons {DEFAULT_HORIZONS}")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = {_key(r) for path in {out_csv, *map(Path, done_from)} for r in read_results(path)}
    grid = [(n, w, s) for n in cfg.n_trajs for w in cfg.widths for s in cfg.seeds]
    # Split the full grid (not just the remaining runs), so each run always
    # belongs to the same shard no matter which shards have already finished.
    i_shard, k_shard = shard
    todo = [g for g in grid[i_shard::k_shard] if g not in done]
    if verbose:
        print(f"{len(done)} runs already in {out_csv}; {len(todo)} to go")
    if not todo:
        return read_results(out_csv)

    pool = make_pool(jax.random.PRNGKey(cfg.data_key), n_pool=cfg.n_pool)
    force_test = generate_dataset(jax.random.PRNGKey(cfg.force_test_key), n_traj=8)
    starts = test_starts(jax.random.PRNGKey(cfg.test_key), cfg.n_test_starts)

    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, (n, w, s) in enumerate(todo, 1):
            data = training_set(pool, n)
            model = PairPotentialMLP(width=w, depth=cfg.depth, key=jax.random.PRNGKey(10_000 + s))
            t0 = time.perf_counter()
            model, losses = train(
                model,
                data,
                key=jax.random.PRNGKey(20_000 + s),
                steps=cfg.steps,
                batch_size=min(cfg.batch_size, len(data)),
                lr=cfg.lr,
                verbose=False,
            )
            seconds = time.perf_counter() - t0
            fm = evaluate(model, force_test)
            ro = rollout_error(model.forces, starts, horizons=cfg.horizons)
            row = {
                "n_traj": n,
                "n_frames": len(data),
                "width": w,
                "depth": cfg.depth,
                "seed": s,
                "n_params": sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))),
                "steps": cfg.steps,
                "final_loss": float(losses[-100:].mean()),
                "force_rmse": fm["force_rmse"],
                "force_r2": fm["force_r2"],
                **{f"rollout_t{h:g}": v for h, v in ro.items()},
                "train_seconds": round(seconds, 2),
            }
            writer.writerow(row)
            f.flush()
            if verbose:
                print(
                    f"[{i:3d}/{len(todo)}] n_traj={n:3d} width={w:4d} seed={s}  "
                    f"rollout(t=1)={row['rollout_t1']:.2e}  force RMSE={row['force_rmse']:.2e}  "
                    f"({seconds:.0f} s)"
                )
    return read_results(out_csv)


def config_dict(cfg: SweepConfig) -> dict:
    return asdict(cfg)
