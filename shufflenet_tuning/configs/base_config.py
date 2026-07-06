"""
configs/base_config.py
-----------------------
Defines the full hyperparameter search grid and generates all ExperimentConfig
instances from the Cartesian product of that grid.

NOTE: This grid is used for reference / legacy Phase 1 grid-search only.
      The Bayesian Optimization (BO) sampler in optuna_optimize.py and
      hypervolume_optimize.py samples this space intelligently, evaluating
      only 100-150 trials instead of all 34,560+ combinations.

Adjust the GRID dict to narrow or expand the search space.
"""

import itertools
from typing import Iterator

from configs.experiment_config import ExperimentConfig


# ── Hyperparameter Search Grid ────────────────────────────────────────────────
#
# Hyperparameter       | Why it's here                                  | Points
# ---------------------|------------------------------------------------|-------
# width_multiplier     | Primary capacity axis — drives accuracy/energy |   4
# intra_op_threads     | Pure energy/latency shifter, zero acc. impact  |   4
# batch_size           | Cache locality lever, memory wall behaviour    |   6
# input_size           | Quadratic FLOPs scaling, spatial resolution    |   4
# dropout              | Regularisation — affects generalisation gap    |   5
# optimizer_name       | Optimisation trajectory & convergence speed    |   3
# scheduler_name       | LR annealing strategy                          |   3
# label_smoothing      | Soft-target regularisation for classifier      |   4
#
# Continuous params (sampled by BO, not enumerated here):
#   learning_rate  — log-uniform [1e-5, 3e-1]
#   weight_decay   — log-uniform [1e-6, 1e-2]
#   momentum       — uniform [0.80, 0.99]  (SGD / RMSprop only)
#
GRID: dict[str, list] = {
    "width_multiplier": [0.5, 1.0, 1.5, 2.0],               # 4 structural tiers
    "intra_op_threads": [1, 2, 4, 8],                         # 4 thread configs
    "batch_size":       [4, 8, 16, 32, 64, 128],              # 6 batch configs
    "input_size":       [20, 24, 28, 32],                      # 4 resolution configs
    "dropout":          [0.0, 0.1, 0.2, 0.3, 0.5],           # 5 dropout levels
    "optimizer_name":   ["adam", "sgd", "rmsprop"],            # 3 optimizers
    "scheduler_name":   ["cosine", "step", "onecycle"],        # 3 schedulers
    "label_smoothing":  [0.0, 0.05, 0.1, 0.15],              # 4 smoothing levels
}
# Total discrete combos = 4 × 4 × 6 × 4 × 5 × 3 × 3 × 4 = 34,560
# (+ continuous lr/wd/momentum dimensions on top)
# BO explores this efficiently: 100-150 trials vs 34,560 exhaustive runs.

# ── Shared training / benchmark constants ─────────────────────────────────────
TRAINING_DEFAULTS: dict = {
    "num_classes":          9,
    "in_channels":          3,
    "num_epochs":           10,
    "learning_rate":        1e-3,
    "weight_decay":         1e-4,
    "momentum":             0.9,
    "num_benchmark_runs":   100,
    "warmup_runs":          10,
}


def generate_configs() -> list[ExperimentConfig]:
    """
    Returns a list of ExperimentConfig objects covering the full discrete grid.
    WARNING: This produces 34,560 configs — use only for reference / inspection.
    Use Bayesian Optimization (optuna_optimize.py) for actual search.

    Example:
        configs = generate_configs()
        print(f"Total runs: {len(configs)}")  # → 34,560
        for cfg in configs[:5]:
            print(cfg)
    """
    keys   = list(GRID.keys())
    values = list(GRID.values())

    configs = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        cfg = ExperimentConfig(**params, **TRAINING_DEFAULTS)
        configs.append(cfg)

    return configs


def generate_configs_for_width(width_multiplier: float) -> list[ExperimentConfig]:
    """Returns only configs for a specific width multiplier — useful for staged runs."""
    return [c for c in generate_configs() if c.width_multiplier == width_multiplier]


def generate_single_config(**overrides) -> ExperimentConfig:
    """
    Quickly build one config with custom values, falling back to grid defaults.

    Example:
        cfg = generate_single_config(width_multiplier=0.5, batch_size=64)
    """
    defaults = {
        "width_multiplier": GRID["width_multiplier"][1],    # 1.0
        "intra_op_threads": GRID["intra_op_threads"][1],    # 2
        "batch_size":       GRID["batch_size"][2],           # 16
        "input_size":       GRID["input_size"][2],           # 28
        "dropout":          GRID["dropout"][0],              # 0.0
        "optimizer_name":   GRID["optimizer_name"][0],       # "adam"
        "scheduler_name":   GRID["scheduler_name"][0],       # "cosine"
        "label_smoothing":  GRID["label_smoothing"][0],      # 0.0
        **TRAINING_DEFAULTS,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)
