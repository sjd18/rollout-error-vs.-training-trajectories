"""Run the (training trajectories × width × seed) sweep, resumably.

    python scripts/run_sweep.py                 # full grid -> results/sweep.csv (~1-2 h on CPU)
    python scripts/run_sweep.py --quick         # tiny grid -> results/quick.csv (~2 min)
    python scripts/run_sweep.py --widths 8 32 --n-trajs 1 4 16 --seeds 0

Split across processes with --shard I K (each writes results/sweep.shardI.csv), then
merge with `python scripts/merge_results.py`.

Interrupt at any time; re-running the same command skips finished runs.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from rollscale import SweepConfig, run_sweep  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trajs", type=int, nargs="+")
    ap.add_argument("--widths", type=int, nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--quick", action="store_true", help="small grid, 300 steps, for a smoke test")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shard", type=int, nargs=2, metavar=("I", "K"), default=(0, 1))
    a = ap.parse_args()

    cfg = SweepConfig()
    if a.quick:
        cfg = SweepConfig(n_trajs=(1, 4, 16), widths=(8, 32), seeds=(0,), steps=300, n_test_starts=4)
    overrides = {
        k: tuple(v) for k, v in [("n_trajs", a.n_trajs), ("widths", a.widths), ("seeds", a.seeds)] if v
    }
    if a.steps:
        overrides["steps"] = a.steps
    cfg = SweepConfig(**{**asdict(cfg), **overrides})
    out = a.out or ROOT / "results" / ("quick.csv" if a.quick else "sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    print("config:", asdict(cfg))
    i, k = a.shard
    target = out if k == 1 else out.with_name(f"{out.stem}.shard{i}.csv")
    others = sorted(out.parent.glob(f"{out.stem}*.csv"))
    run_sweep(cfg, target, shard=(i, k), done_from=others)


if __name__ == "__main__":
    main()
