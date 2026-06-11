"""
main.py
-------
Single entrypoint for all experiment modes.

Usage:
    # Run the full 72-config grid search
    python main.py

    # Preview all configs without training (dry run)
    python main.py --dry-run

    # Run a single custom config (useful for quick sanity checks)
    python main.py --single --width 0.5 --threads 4 --batch 32 --res 28

    # Run only configs for one width multiplier
    python main.py --width-only 1.0
"""

import argparse

from configs.base_config import generate_configs_for_width, generate_single_config
from experiments.grid_search import run_grid, run_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ShuffleNetV2 Hyperparameter Tuning — MedMNIST CPU Study"
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all configs without training."
    )
    mode.add_argument(
        "--single",
        action="store_true",
        help="Run a single config (use --width, --threads, --batch, --res to set it)."
    )
    mode.add_argument(
        "--width-only",
        type=float,
        metavar="W",
        help="Only run configs for one width multiplier (e.g. --width-only 0.5)."
    )

    # Single-run options
    parser.add_argument("--width",   type=float, default=1.0, help="Width multiplier")
    parser.add_argument("--threads", type=int,   default=4,   help="CPU thread count")
    parser.add_argument("--batch",   type=int,   default=32,  help="Batch size")
    parser.add_argument("--res",     type=int,   default=28,  help="Input resolution")

    # Grid search options
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh — ignore any previously completed configs in the CSV."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        run_grid(dry_run=True)

    elif args.single:
        cfg = generate_single_config(
            width_multiplier=args.width,
            intra_op_threads=args.threads,
            batch_size=args.batch,
            input_size=args.res,
        )
        print(f"\nRunning single config: {cfg}\n")
        run_one(cfg)

    elif args.width_only is not None:
        configs = generate_configs_for_width(args.width_only)
        print(f"\nRunning {len(configs)} configs for width={args.width_only}x\n")
        for cfg in configs:
            run_one(cfg)

    else:
        # Default: full grid search
        run_grid(resume=not args.no_resume)


if __name__ == "__main__":
    main()
