"""Log-log scaling plots of rollout error from sweep results.

Line colours are ordinal one-hue ramps (blue for model width, orange for the
number of training trajectories), each validated for visible steps between
neighbouring lines. At most five shades pass that check, so each line panel
shows five evenly spaced families; every grid point still appears along the
x-axis, and :func:`plot_heatmap` shows the complete grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "aggregate",
    "power_law_slope",
    "pick_families",
    "plot_scaling",
    "plot_heatmap",
    "plot_force_vs_rollout",
    "METRIC_LABELS",
]

BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ORANGES = ["#fd9d53", "#f57622", "#de5005", "#ab3803", "#7f2704"]
_SURFACE, _TEXT, _TEXT_2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_RC = {
    "figure.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "axes.edgecolor": _GRID,
    "axes.labelcolor": _TEXT_2,
    "axes.titlecolor": _TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": _GRID,
    "grid.linewidth": 0.8,
    "xtick.color": _TEXT_2,
    "ytick.color": _TEXT_2,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.labelcolor": _TEXT,
    "font.size": 10,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
}

METRIC_LABELS = {
    "rollout_t0.5": "rollout RMS position error at t = 0.5",
    "rollout_t1": "rollout RMS position error at t = 1",
    "rollout_t2.5": "rollout RMS position error at t = 2.5",
    "force_rmse": "held-out force RMSE",
}


def aggregate(rows, x: str, group: str, metric: str) -> dict:
    """``{group_value: (xs, median, lo, hi)}`` over seeds, sorted by ``x``.

    Median and min/max are taken across seeds at each grid point.
    """
    out = {}
    for g in sorted({r[group] for r in rows}):
        xs = sorted({r[x] for r in rows if r[group] == g})
        vals = [np.array([r[metric] for r in rows if r[group] == g and r[x] == xv], dtype=float) for xv in xs]
        out[g] = (
            np.array(xs, dtype=float),
            np.array([np.median(v) for v in vals]),
            np.array([v.min() for v in vals]),
            np.array([v.max() for v in vals]),
        )
    return out


def power_law_slope(x, y) -> float:
    """Least-squares slope of ``log y`` against ``log x`` (finite points only)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def pick_families(values, k: int = 5) -> list:
    """Up to ``k`` values spread evenly (by index) over the sorted ``values``, ends included."""
    values = sorted(values)
    if len(values) <= k:
        return values
    idx = np.unique(np.round(np.linspace(0, len(values) - 1, k)).astype(int))
    return [values[i] for i in idx]


def _x_ticks_at(ax, values):
    """Label the log x-axis at exactly the grid values."""
    from matplotlib.ticker import NullLocator

    values = sorted(values)
    ax.set_xticks(values, [f"{v:g}" for v in values])
    ax.xaxis.set_minor_locator(NullLocator())


def _panel(ax, agg, families, colors, family_label, x_label, x_values):
    for g, c in zip(families, colors, strict=False):
        xs, med, lo, hi = agg[g]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(xs, med, "o-", color=c, ms=6, mec=_SURFACE, mew=1.5, label=f"{g:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _x_ticks_at(ax, x_values)
    ax.set_xlabel(x_label)


def plot_scaling(rows, path, metric: str = "rollout_t1", families_width=None, families_traj=None) -> Path:
    """Two log-log panels: error vs training trajectories (one line per width)
    and error vs width (one line per number of trajectories).

    Lines are medians over seeds; shaded bands span min to max across seeds.
    """
    import matplotlib.pyplot as plt

    widths = sorted({r["width"] for r in rows})
    trajs = sorted({r["n_traj"] for r in rows})
    fw = families_width or pick_families(widths)
    ft = families_traj or pick_families(trajs)

    with plt.rc_context(_RC):
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        _panel(
            a, aggregate(rows, "n_traj", "width", metric), fw, BLUES, "width", "training trajectories", trajs
        )
        a.set_ylabel(METRIC_LABELS.get(metric, metric))
        a.set_title("Rollout error vs. training data")
        a.legend(
            loc="center left", bbox_to_anchor=(1.0, 0.5), title="MLP width", fontsize=9, title_fontsize=9
        )

        _panel(
            b, aggregate(rows, "width", "n_traj", metric), ft, ORANGES, "trajectories", "MLP width", widths
        )
        b.set_title("Rollout error vs. model width")
        b.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            title="training\ntrajectories",
            fontsize=9,
            title_fontsize=9,
        )
        b.tick_params(labelleft=True)
        fig.text(
            0.01,
            -0.02,
            "Lines: median over seeds. Bands: min to max across seeds. Log-log axes. "
            f"Each panel shows {len(fw)} of {len(widths)} widths / {len(ft)} of {len(trajs)} data sizes "
            "as lines; see the heatmap for the full grid.",
            fontsize=8,
            color=_TEXT_2,
        )
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_heatmap(rows, path, metric: str = "rollout_t1") -> Path:
    """Median error over seeds on the full (trajectories × width) grid, log colour scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    widths = sorted({r["width"] for r in rows})
    trajs = sorted({r["n_traj"] for r in rows})
    grid = np.full((len(widths), len(trajs)), np.nan)
    for i, w in enumerate(widths):
        for j, n in enumerate(trajs):
            v = [r[metric] for r in rows if r["width"] == w and r["n_traj"] == n]
            if v:
                grid[i, j] = np.median(v)
    cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        im = ax.imshow(
            grid, origin="lower", cmap=cmap, norm=LogNorm(np.nanmin(grid), np.nanmax(grid)), aspect="auto"
        )
        ax.grid(False)
        ax.set_xticks(range(len(trajs)), [f"{n:g}" for n in trajs])
        ax.set_yticks(range(len(widths)), [f"{w:g}" for w in widths])
        ax.set_xlabel("training trajectories (log₂ spaced)")
        ax.set_ylabel("MLP width (log₂ spaced)")
        ax.set_title("Full grid: median rollout error")
        mid = np.sqrt(np.nanmin(grid) * np.nanmax(grid))
        for i in range(len(widths)):
            for j in range(len(trajs)):
                if np.isfinite(grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.1e}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if grid[i, j] > mid else _TEXT,
                    )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(METRIC_LABELS.get(metric, metric))
        cb.outline.set_visible(False)
        fig.tight_layout()
        path = Path(path)
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_force_vs_rollout(rows, path, metric: str = "rollout_t1") -> Path:
    """Every run: rollout error against held-out force error, log-log, with a fitted slope."""
    import matplotlib.pyplot as plt

    f = np.array([r["force_rmse"] for r in rows], float)
    e = np.array([r[metric] for r in rows], float)
    ok = np.isfinite(e) & np.isfinite(f) & (e > 0) & (f > 0)
    slope, icpt = np.polyfit(np.log(f[ok]), np.log(e[ok]), 1)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.scatter(f[ok], e[ok], s=22, color=BLUES[2], alpha=0.7, edgecolor=_SURFACE, lw=0.6)
        xs = np.geomspace(f[ok].min(), f[ok].max(), 50)
        ax.plot(
            xs,
            np.exp(icpt) * xs**slope,
            color=_TEXT_2,
            ls="--",
            lw=1.2,
            label=f"fit: error ∝ (force RMSE)^{slope:.2f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("held-out force RMSE")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title("Rollout error tracks force error")
        ax.legend(loc="upper left")
        fig.tight_layout()
        path = Path(path)
        fig.savefig(path)
        plt.close(fig)
    return path
