"""
experiments/train_phase1.py
---------------------------
Phase 1 of the Two-Phase Strategy: Accuracy Run (Heavy Compute).

This script identifies the unique combinations of architecture variables
(width_multiplier and input_size) that affect test accuracy. It trains
these unique configurations on a GPU (if available) and saves their
weights and test accuracies for Phase 2 CPU profiling.

Resumable: will skip already trained and registered configurations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Ensure the parent directory (shufflenet_tuning/) is in the path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.base_config import generate_configs
from configs.experiment_config import ExperimentConfig
from engine.evaluator import evaluate
from engine.trainer import train

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Train accuracy-affecting configurations on GPU/CPU."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train each configuration (default: 10)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size to use for training (default: 64, optimized for speed)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g. cuda, cpu). Defaults to cuda if available."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and retrain everything from scratch."
    )
    return parser.parse_args()


def run_phase1(epochs: int = 10, batch_size: int = 64, device_str: str = None, no_resume: bool = False) -> None:
    # Determine training device
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Phase 1: Accuracy Run starting")
    print(f"  Target Device: {device}")
    print(f"  Training Epochs: {epochs}")
    print(f"  Training Batch Size: {batch_size}")
    print(f"{'='*60}\n")

    # Load all grid configs and extract unique (width_multiplier, input_size) combinations
    all_configs = generate_configs()
    unique_pairs = {}
    
    for cfg in all_configs:
        pair = (cfg.width_multiplier, cfg.input_size)
        if pair not in unique_pairs:
            unique_pairs[pair] = cfg

    print(f"Found {len(all_configs)} total grid configurations.")
    print(f"Identified {len(unique_pairs)} unique (width, input_size) pairs affecting accuracy:\n")
    for i, (width, res) in enumerate(unique_pairs.keys(), 1):
        print(f"  {i}. Width multiplier: {width}x, Input size: {res}x{res}")
    print()

    # Load existing registry if resuming is enabled
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = CHECKPOINTS_DIR / "accuracy_registry.json"
    registry = {}
    if registry_path.exists() and not no_resume:
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
            print(f"Loaded existing accuracy registry with {len(registry)} entries.")
        except Exception as e:
            print(f"Warning: Failed to load existing registry ({e}). Starting fresh.")

    # Train each unique configuration
    for idx, ((width, res), base_cfg) in enumerate(unique_pairs.items(), 1):
        registry_key = f"w{width}_r{res}"
        
        if registry_key in registry:
            print(f"[{idx}/{len(unique_pairs)}] Skipping {registry_key} (already trained, acc={registry[registry_key]:.2f}%)")
            continue

        print(f"\n[{idx}/{len(unique_pairs)}] Training model for: width={width}x, resolution={res}x{res}")

        # Create training config overriding batch size and epochs for faster training
        train_cfg = ExperimentConfig(
            width_multiplier=width,
            input_size=res,
            batch_size=batch_size,
            num_epochs=epochs,
            intra_op_threads=0, # Let PyTorch choose training thread count dynamically
            num_classes=base_cfg.num_classes,
            in_channels=base_cfg.in_channels,
            learning_rate=base_cfg.learning_rate,
            weight_decay=base_cfg.weight_decay
        )

        try:
            # 1. Train the model
            model, _, test_loader = train(train_cfg, device=device)

            # 2. Evaluate accuracy
            accuracy = evaluate(model, test_loader)

            # 3. Save model weights
            weight_filename = f"shufflenet_w{width}_r{res}.pth"
            weight_path = CHECKPOINTS_DIR / weight_filename
            torch.save(model.state_dict(), weight_path)
            print(f"Saved weights to {weight_path}")

            # 4. Record to registry
            registry[registry_key] = round(accuracy, 4)
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=4)
            print(f"Registered accuracy: {accuracy:.2f}%")

        except Exception as e:
            print(f"Error training config {registry_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"  Phase 1 Complete!")
    print(f"  Registry file: {registry_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    run_phase1(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_str=args.device,
        no_resume=args.no_resume
    )
