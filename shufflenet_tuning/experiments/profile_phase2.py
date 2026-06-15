"""
experiments/profile_phase2.py
-----------------------------
Phase 2 of the Two-Phase Strategy: Energy Profiling Run (Strictly CPU).

This script benchmarks the CPU execution speed (latency), throughput, and
energy consumption (kWh) of all 72 configurations. It loads the corresponding
pre-trained weights from Phase 1, runs inference on CPU, and logs the metrics
along with the Phase 1 test accuracy to the final experiment log CSV.

Resumable: will skip already completed configs in the results CSV.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

# Ensure the parent directory (shufflenet_tuning/) is in the path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.base_config import generate_configs
from engine.profiler import profile
from models.shufflenet import ShuffleNetV2

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
RESULTS_DIR     = Path(__file__).resolve().parent.parent / "results"
LOG_FILE        = RESULTS_DIR / "experiment_log.csv"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Profile CPU energy and latency for all configurations."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing entries in CSV and profile everything from scratch."
    )
    return parser.parse_args()


def _load_completed_ids() -> set[str]:
    """Returns config_ids already present in the CSV."""
    if not LOG_FILE.exists():
        return set()
    try:
        with open(LOG_FILE, newline="") as f:
            reader = csv.DictReader(f)
            return {row["config_id"] for row in reader if row.get("config_id")}
    except Exception as e:
        print(f"Warning: Error reading log file ({e}). Starting clean.")
        return set()


def _append_result(row: dict) -> None:
    """Appends one result row to the CSV, writing the header on first call."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_phase2(no_resume: bool = False) -> None:
    # CPU check
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  Phase 2: CPU Energy & Latency Profiling Run starting")
    print(f"  Target Device: {device}")
    print(f"  Log file:      {LOG_FILE}")
    print(f"{'='*60}\n")

    # 1. Load accuracy registry from Phase 1
    registry_path = CHECKPOINTS_DIR / "accuracy_registry.json"
    if not registry_path.exists():
        print(f"Error: Accuracy registry not found at {registry_path}.")
        print("You must run Phase 1 (Accuracy Run) first to train the models.")
        sys.exit(1)

    try:
        with open(registry_path, "r") as f:
            accuracy_registry = json.load(f)
    except Exception as e:
        print(f"Error reading accuracy registry: {e}")
        sys.exit(1)

    # Load completed runs
    completed = _load_completed_ids() if not no_resume else set()
    if completed:
        print(f"Loaded {len(completed)} completed runs from log. Resuming...")

    # Load all grid configs
    configs = generate_configs()
    print(f"Grid search space size: {len(configs)} configurations.\n")

    for i, cfg in enumerate(configs, 1):
        if cfg.config_id in completed:
            print(f"  [{i:3d}/{len(configs)}] SKIP (already done): {cfg.config_id}")
            continue

        registry_key = f"w{cfg.width_multiplier}_r{cfg.input_size}"
        accuracy = accuracy_registry.get(registry_key)
        if accuracy is None:
            print(f"  [{i:3d}/{len(configs)}] WARNING: Accuracy not found in registry for {registry_key}. Skipping config {cfg.config_id}.")
            continue

        weight_filename = f"shufflenet_w{cfg.width_multiplier}_r{cfg.input_size}.pth"
        weight_path = CHECKPOINTS_DIR / weight_filename
        if not weight_path.exists():
            print(f"  [{i:3d}/{len(configs)}] WARNING: Weight file {weight_path} does not exist. Skipping config {cfg.config_id}.")
            continue

        print(f"\n  [{i:3d}/{len(configs)}] Profiling: {cfg}")

        try:
            # 1. Instantiate the model
            model = ShuffleNetV2(
                width_multiplier=cfg.width_multiplier,
                num_classes=cfg.num_classes,
                in_channels=cfg.in_channels,
                intra_op_threads=cfg.intra_op_threads,
            ).to(device)

            # 2. Load the pre-trained weights
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            model.eval()

            # 3. Profile forward pass on CPU (uses engine.profiler.profile)
            profile_result = profile(model, cfg)

            # 4. Assemble and write result
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

            _append_result(row)
            print(
                f"  [OK] Logged: acc={row['accuracy']:.2f}% | "
                f"energy={row['energy_kwh']:.6f} kWh | "
                f"latency={row['latency_sec']*1000:.2f} ms"
            )

        except Exception as e:
            print(f"  [FAIL] Config {cfg.config_id} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"  Phase 2 CPU Profiling Complete!")
    print(f"  Results saved to: {LOG_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    run_phase2(no_resume=args.no_resume)
