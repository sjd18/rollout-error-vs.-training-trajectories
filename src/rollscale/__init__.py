"""rollscale: how rollout error scales with training data and model width.

Built on ``eqxmlp`` (https://github.com/sjd18/Equinox-MLP): an Equinox MLP
pair potential whose autodiff forces drive velocity-Verlet MD.
"""

from .data import TrajectoryPool, make_pool, subset_indices, test_starts, training_set
from .metrics import DEFAULT_HORIZONS, rollout_error, rollout_rms_curve
from .plotting import plot_force_vs_rollout, plot_heatmap, plot_scaling, power_law_slope
from .sweep import FIELDS, SweepConfig, read_results, run_sweep

__version__ = "0.1.0"

__all__ = [
    "TrajectoryPool",
    "make_pool",
    "subset_indices",
    "training_set",
    "test_starts",
    "DEFAULT_HORIZONS",
    "rollout_error",
    "rollout_rms_curve",
    "SweepConfig",
    "run_sweep",
    "read_results",
    "FIELDS",
    "plot_scaling",
    "plot_heatmap",
    "plot_force_vs_rollout",
    "power_law_slope",
]
