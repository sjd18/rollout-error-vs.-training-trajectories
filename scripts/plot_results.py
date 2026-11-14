"""Make the figures and the slope table from a results CSV.

python scripts/plot_results.py                       # results/sweep.csv -> assets/
python scripts/plot_results.py --csv results/quick.csv --metric rollout_t2.5
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from rollscale import (  # noqa: E402
    plot_force_vs_rollout,
    plot_heatmap,
    plot_scaling,
    plot_slopes,
    read_results,
    scaling_slopes,
)

ROOT = Path(__file__).resolve().parent.parent


def slope_table(rows, metric):
    sl = scaling_slopes(rows, metric)
    lines = [f"Power-law slopes of median {metric} (log-log least squares)", ""]
    lines.append("vs. training trajectories, per width:")
    lines += [f"  width {w:>4g}: slope {v:+.2f}" for w, v in sl["vs_data"].items()]
    lines.append("vs. width, per number of trajectories:")
    lines += [f"  {n:>3g} traj: slope {v:+.2f}" for n, v in sl["vs_width"].items()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results" / "sweep.csv")
    ap.add_argument("--metric", default="rollout_t1")
    ap.add_argument("--out", type=Path, default=ROOT / "assets")
    a = ap.parse_args()
    rows = read_results(a.csv)
    if not rows:
        raise SystemExit(f"no results in {a.csv}; run scripts/run_sweep.py first")
    suffix = "" if a.metric == "rollout_t1" else f"_{a.metric}"
    for fn, name in [
        (plot_scaling, "rollout_scaling"),
        (plot_heatmap, "rollout_heatmap"),
        (plot_slopes, "rollout_slopes"),
    ]:
        print("saved", fn(rows, a.out / f"{name}{suffix}.png", metric=a.metric))
    print("saved", plot_force_vs_rollout(rows, a.out / f"force_vs_rollout{suffix}.png", metric=a.metric))
    table = slope_table(rows, a.metric)
    print("\n" + table)
    best = min(rows, key=lambda r: r[a.metric] if np.isfinite(r[a.metric]) else np.inf)
    where = f"n_traj={best['n_traj']} width={best['width']} seed={best['seed']}"
    print(f"\nbest run: {where} -> {best[a.metric]:.2e}")
    (a.out / f"slopes{suffix}.txt").write_text(table + "\n")


if __name__ == "__main__":
    main()
