"""
main.py
-------
Single entrypoint for ShuffleNetV2 tuning and optimization experiments.

All experiments run on CPU only.

Usage:
    # Multi-Objective Bayesian Optimization (MOTPE) — default
    python main.py

    # v2 Refined MOTPE (new loss functions, FC layer, depth tuning)
    python main.py --v2
    python main.py --v2 --trials 150 --optuna-epochs 2

    # Single-Objective Hypervolume Maximization BO
    python main.py --hypervolume

    # Custom trial/epoch count
    python main.py --trials 150 --optuna-epochs 3
    python main.py --hypervolume --trials 150 --optuna-epochs 3

    # Run Phase 1 training (legacy accuracy-critical grid search)
    python main.py --phase1 --epochs 10 --train-batch-size 64

    # Skip search, finalize from existing MOTPE study
    python main.py --final-only
"""

import argparse
import atexit
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Ensure the parent directory is in the path
sys.path.append(str(Path(__file__).resolve().parent))

from experiments.optuna_optimize import run_optimization


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ShuffleNetV2 Hyperparameter Tuning — MedMNIST CPU Study\n"
            "Supports Multi-Objective MOTPE and Hypervolume Maximization BO."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode flags ─────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--optuna",
        action="store_true",
        default=True,
        help="[DEFAULT] Run Multi-Objective MOTPE Bayesian Optimization (3 objectives: Accuracy, Latency, Energy).",
    )
    mode.add_argument(
        "--v2",
        action="store_true",
        help=(
            "Run v2 Refined MOTPE: new loss functions (CE/KLDiv/Focal), "
            "optional FC hidden layer, tunable stage depth, Adam/SGD only, narrowed LR."
        ),
    )
    mode.add_argument(
        "--hypervolume",
        action="store_true",
        help=(
            "Run Single-Objective Hypervolume Maximization BO. "
            "Collapses Accuracy/Latency/Energy into a single HV indicator and maximizes it."
        ),
    )
    mode.add_argument(
        "--phase1",
        action="store_true",
        help="Run Phase 1: Legacy grid-search training on CPU.",
    )
    mode.add_argument(
        "--phase2-legacy",
        action="store_true",
        help="Run legacy Phase 2: CPU energy & latency profiling for all grid configurations.",
    )
    mode.add_argument(
        "--final-only",
        action="store_true",
        help="Skip Optuna search — run final retraining + PTQ from the existing MOTPE study database.",
    )

    # ── BO / Search settings ───────────────────────────────────────────────────
    parser.add_argument(
        "--trials",
        type=int,
        default=150,
        help="Number of BO trials (default: 150). Recommended: 100–200 for a 34,560-combo search space.",
    )
    parser.add_argument(
        "--optuna-epochs",
        type=int,
        default=2,
        help="Training epochs per BO trial (default: 2). Shorter = faster exploration.",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=8,
        help="Epochs to retrain the best config after the search phase (default: 8).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel Optuna workers against the shared SQLite study (default: 1).",
    )

    # ── Phase 1 / Training settings ────────────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Phase 1: Training epochs per model (default: 10).",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="Phase 1: Training batch size (default: 64).",
    )

    # ── Misc ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh — ignore any previously completed configs/checkpoints.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="run.log",
        help="Write stdout/stderr to this file while still printing to the terminal (default: run.log).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", buffering=1) as log_file:
        tee_out = _Tee(sys.__stdout__, log_file)
        tee_err = _Tee(sys.__stderr__, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[log] writing to {log_path}")
            print(f"[device] CPU-only mode (all GPU paths disabled)")

            if args.v2:
                from experiments.optuna_optimize_v2 import run_optimization_v2
                run_optimization_v2(
                    n_trials=args.trials,
                    search_epochs=args.optuna_epochs,
                    final_epochs=args.final_epochs,
                    n_jobs=args.workers,
                )

            elif args.hypervolume:
                from experiments.hypervolume_optimize import run_hypervolume_optimization
                run_hypervolume_optimization(
                    n_trials=args.trials,
                    search_epochs=args.optuna_epochs,
                    final_epochs=args.final_epochs,
                    n_jobs=args.workers,
                )

            elif args.phase1:
                from experiments.train_phase1 import run_phase1
                run_phase1(
                    epochs=args.epochs,
                    batch_size=args.train_batch_size,
                    device_str="cpu",
                    no_resume=args.no_resume,
                )

            elif args.phase2_legacy:
                from experiments.profile_phase2 import run_phase2
                run_phase2(no_resume=args.no_resume)

            elif args.final_only:
                from experiments.optuna_optimize import finalize_from_study
                finalize_from_study(final_epochs=args.final_epochs)

            else:
                # Default: Multi-Objective Bayesian Optimization (MOTPE)
                run_optimization(
                    n_trials=args.trials,
                    search_epochs=args.optuna_epochs,
                    final_epochs=args.final_epochs,
                    n_jobs=args.workers,
                )


if __name__ == "__main__":
    main()
