"""
main.py
-------
Single entrypoint for ShuffleNetV2 tuning and optimization experiments.

Usage:
    # Run the Multi-Objective Bayesian Optimization & PTQ (default)
    python main.py

    # Run the optimization with custom trial and epoch count
    python main.py --trials 30 --optuna-epochs 2

    # Run Phase 1 training (accuracy-critical configurations)
    python main.py --phase1 --epochs 10 --train-batch-size 64
"""

import argparse
import sys
from pathlib import Path

# Ensure the parent directory is in the path
sys.path.append(str(Path(__file__).resolve().parent))

from experiments.optuna_optimize import run_optimization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ShuffleNetV2 Hyperparameter Tuning — MedMNIST CPU Study"
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--optuna",
        action="store_true",
        default=True,
        help="Run Phase 2 Multi-Objective Bayesian Optimization using Optuna & PTQ (default)."
    )
    mode.add_argument(
        "--phase1",
        action="store_true",
        help="Run Phase 1: Train accuracy-critical configurations on GPU/CPU."
    )
    mode.add_argument(
        "--phase2-legacy",
        action="store_true",
        help="Run legacy Phase 2: CPU energy & latency profiling for all grid configurations."
    )

    # Optuna / Phase 2 settings
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of trials for the Optuna study (default: 20)."
    )
    parser.add_argument(
        "--optuna-epochs",
        type=int,
        default=1,
        help="Number of epochs to train the model per Optuna trial (default: 1)."
    )

    # Phase 1 / Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Phase 1: Training epochs per model (default: 10)."
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="Phase 1: Training batch size (default: 64)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda/cpu, default: cuda if available)."
    )

    # Legacy settings
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh — ignore any previously completed configs/checkpoints."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.phase1:
        from experiments.train_phase1 import run_phase1
        run_phase1(
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            device_str=args.device,
            no_resume=args.no_resume
        )

    elif args.phase2_legacy:
        from experiments.profile_phase2 import run_phase2
        run_phase2(no_resume=args.no_resume)

    else:
        # Default option: Multi-Objective Bayesian Optimization & PTQ
        run_optimization(
            n_trials=args.trials,
            epochs_per_trial=args.optuna_epochs,
            device_str=args.device
        )


if __name__ == "__main__":
    main()
