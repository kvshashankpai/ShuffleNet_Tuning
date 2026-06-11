"""
configs/base_config.py
-----------------------
Defines the full hyperparameter search grid and generates all ExperimentConfig
instances from the Cartesian product of that grid.

Adjust the GRID dict to narrow or expand the search space.
"""

import itertools
from typing import Iterator

from configs.experiment_config import ExperimentConfig


# ── Hyperparameter Search Grid ────────────────────────────────────────────────
#
# Hyperparameter       | Why it's here
# ---------------------|----------------------------------------------------------
# width_multiplier     | Primary capacity axis — drives accuracy AND energy tiers
# intra_op_threads     | Pure energy/latency shifter — zero accuracy impact
# batch_size           | Cache locality lever — affects memory wall behaviour
# input_size           | Quadratic FLOPs scaling — emergency energy reduction knob
#
GRID: dict[str, list] = {
    "width_multiplier": [0.5, 1.0, 1.5, 2.0],   # 4 structural tiers
    "intra_op_threads": [1, 2, 4],                # 3 thread configs (tune to your core count)
    "batch_size":       [16, 32, 64],             # 3 batch configs
    "input_size":       [24, 28],                 # 2 resolution configs
}
# Total = 4 × 3 × 3 × 2 = 72 unique configurations

# ── Shared training / benchmark constants ─────────────────────────────────────
TRAINING_DEFAULTS: dict = {
    "num_classes":       9,
    "in_channels":       3,
    "num_epochs":        10,
    "learning_rate":     1e-3,
    "weight_decay":      1e-4,
    "num_benchmark_runs": 100,
    "warmup_runs":        10,
}


def generate_configs() -> list[ExperimentConfig]:
    """
    Returns a list of ExperimentConfig objects covering the full grid.

    Example:
        configs = generate_configs()
        print(f"Total runs: {len(configs)}")  # → 72
        for cfg in configs:
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
        "width_multiplier": GRID["width_multiplier"][1],   # 1.0
        "intra_op_threads": GRID["intra_op_threads"][1],   # 2
        "batch_size":       GRID["batch_size"][1],          # 32
        "input_size":       GRID["input_size"][1],          # 28
        **TRAINING_DEFAULTS,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)
