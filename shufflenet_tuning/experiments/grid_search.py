"""
experiments/grid_search.py
---------------------------
Outer experiment loop.

For each config in the search grid:
  1. Train the model
  2. Evaluate accuracy on the test split
  3. Profile energy consumption and latency in isolation
  4. Append a result row to results/experiment_log.csv

Designed to be resumable: already-completed config_ids are skipped
if the CSV already exists (safe to re-run after a crash).
"""

import csv
import dataclasses
import os
from pathlib import Path

from configs.base_config import generate_configs
from configs.experiment_config import ExperimentConfig
from engine.evaluator import evaluate
from engine.profiler import profile
from engine.trainer import train
from models.shufflenet import ShuffleNetV2


# ── Output path ───────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOG_FILE    = RESULTS_DIR / "experiment_log.csv"

# ── CSV column order (matches ExperimentConfig + result fields) ───────────────
CSV_FIELDNAMES = [
    "config_id",
    "width_multiplier",
    "intra_op_threads",
    "batch_size",
    "input_size",
    "accuracy",
    "energy_kwh",
    "latency_sec",
    "throughput",
    "params",
]


def _load_completed_ids() -> set[str]:
    """Returns config_ids already present in the CSV (for resume support)."""
    if not LOG_FILE.exists():
        return set()
    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f)
        return {row["config_id"] for row in reader}


def _append_result(row: dict) -> None:
    """Appends one result row to the CSV, writing the header on first call."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one(cfg: ExperimentConfig) -> dict:
    """
    Runs the full train → evaluate → profile pipeline for one config.

    Returns:
        A dict matching CSV_FIELDNAMES, ready to be logged.
    """
    # 1. Train
    model, _, test_loader = train(cfg)

    # 2. Evaluate accuracy
    accuracy = evaluate(model, test_loader)

    # 3. Profile energy + latency (isolated, no data loading)
    profile_result = profile(model, cfg)

    # 4. Assemble result row
    row = {
        "config_id":        cfg.config_id,
        "width_multiplier": cfg.width_multiplier,
        "intra_op_threads": cfg.intra_op_threads,
        "batch_size":       cfg.batch_size,
        "input_size":       cfg.input_size,
        "accuracy":         round(accuracy, 4),
        "energy_kwh":       round(profile_result.energy_kwh, 8),
        "latency_sec":      round(profile_result.latency_sec, 6),
        "throughput":       round(profile_result.throughput, 2),
        "params":           model.count_parameters(),
    }

    return row


def run_grid(
    dry_run: bool = False,
    resume:  bool = True,
) -> None:
    """
    Runs the full hyperparameter grid search.

    Args:
        dry_run: If True, prints all configs but doesn't train anything.
                 Useful to verify your grid before committing compute time.
        resume:  If True, skips configs already present in the CSV.
                 Allows safe restart after hardware interruptions.
    """
    configs = generate_configs()
    print(f"\n{'='*60}")
    print(f"  Grid search: {len(configs)} configurations")
    print(f"  Log file:    {LOG_FILE}")
    print(f"{'='*60}\n")

    if dry_run:
        for i, cfg in enumerate(configs, 1):
            print(f"  [{i:3d}/{len(configs)}] {cfg}")
        print("\n  Dry run complete — no training performed.")
        return

    completed = _load_completed_ids() if resume else set()
    if completed:
        print(f"  Resuming — skipping {len(completed)} already-completed configs.\n")

    for i, cfg in enumerate(configs, 1):
        if cfg.config_id in completed:
            print(f"  [{i:3d}/{len(configs)}] SKIP (already done): {cfg.config_id}")
            continue

        print(f"\n  [{i:3d}/{len(configs)}] Starting: {cfg}")

        try:
            row = run_one(cfg)
            _append_result(row)
            print(
                f"  ✓ Logged: acc={row['accuracy']:.2f}%  "
                f"energy={row['energy_kwh']:.6f} kWh"
            )
        except Exception as e:
            print(f"  ✗ Config {cfg.config_id} failed: {e}")
            # Continue to next config rather than halting the entire grid
            continue

    print(f"\n{'='*60}")
    print(f"  Grid search complete. Results saved to:")
    print(f"  {LOG_FILE}")
    print(f"{'='*60}\n")
