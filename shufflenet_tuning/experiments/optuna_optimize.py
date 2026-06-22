"""
experiments/optuna_optimize.py
------------------------------
Multi-Objective Bayesian Optimization (Optuna MOTPE) and Post-Training
Quantization (PTQ) pipeline for ShuffleNet V2 on MedMNIST.
"""

import os
import sys
import time
import copy
import io
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna

# Ensure parent directory is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.experiment_config import ExperimentConfig
from engine.trainer import train, build_dataloaders
from engine.evaluator import evaluate
from models.shufflenet import QuantizableShuffleNetV2

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def run_ptq(model: nn.Module, test_loader: DataLoader) -> nn.Module:
    """
    Applies Static Post-Training Quantization (PTQ) to the model.

    Steps:
      1. Force model and data to CPU.
      2. Fuse modules (Conv + BN + ReLU).
      3. Set qconfig for fbgemm (x86 CPU inference).
      4. Prepare model for static quantization.
      5. Calibrate observers using 10 batches.
      6. Convert model to INT8.
      7. Print footprint reduction.
    """
    print(f"\n{'='*60}")
    print("  Task 2: INT8 Post-Training Quantization (PTQ)")
    print(f"{'='*60}")

    # Ensure CPU device
    model = model.cpu()
    model.eval()

    # 1. Print before size
    float_buffer = io.BytesIO()
    torch.save(model.state_dict(), float_buffer)
    float_size_mb = len(float_buffer.getvalue()) / (1024 * 1024)
    print(f"  [PTQ] Float model size: {float_size_mb:.3f} MB")

    quantizable_model = copy.deepcopy(model)

    # 2. Fuse modules
    if hasattr(quantizable_model, "fuse_model"):
        print("  [PTQ] Fusing Conv2d + BatchNorm2d + ReLU modules...")
        quantizable_model.fuse_model()

    # 3. Set qconfig
    quantizable_model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

    # 4. Prepare for static quantization
    print("  [PTQ] Preparing model (inserting observers)...")
    prepared_model = torch.ao.quantization.prepare(quantizable_model, inplace=False)

    # 5. Calibration loop (5-10 batches)
    print("  [PTQ] Calibrating scale/zero-point observers on 10 batches...")
    prepared_model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if idx >= 10:
                break
            _ = prepared_model(images.cpu())

    # 6. Convert model to INT8
    print("  [PTQ] Converting model to INT8 (convert)...")
    quantized_model = torch.ao.quantization.convert(prepared_model, inplace=False)

    # 7. Print after size
    quant_buffer = io.BytesIO()
    torch.save(quantized_model.state_dict(), quant_buffer)
    quant_size_mb = len(quant_buffer.getvalue()) / (1024 * 1024)
    print(f"  [PTQ] Quantized INT8 model size: {quant_size_mb:.3f} MB")

    reduction = (1.0 - (quant_size_mb / float_size_mb)) * 100
    print(f"  [PTQ] Footprint reduction: {reduction:.2f}%\n")

    return quantized_model


def run_optimization(n_trials: int = 20, epochs_per_trial: int = 1, device_str: str = None) -> None:
    """
    Runs Multi-Objective Bayesian Optimization using MOTPE.
    After finding the Pareto front, it selects the trial with the highest validation
    accuracy, saves its weights, and runs PTQ on it.
    """
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("  Task 1: Multi-Objective Bayesian Optimization (Optuna)")
    print(f"  Trials: {n_trials} | Epochs per trial: {epochs_per_trial}")
    print(f"  Training Device: {device}")
    print(f"{'='*60}\n")

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        # 1. Suggest parameters from the search space
        width_multiplier = trial.suggest_categorical("width_multiplier", [0.5, 1.0, 1.5, 2.0])
        input_resolution = trial.suggest_categorical("input_resolution", [24, 28])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        intra_op_threads = trial.suggest_categorical("intra_op_threads", [1, 2, 4])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        # 2. Dynamically set thread count
        torch.set_num_threads(intra_op_threads)

        # 3. Create config
        cfg = ExperimentConfig(
            width_multiplier=width_multiplier,
            input_size=input_resolution,
            batch_size=batch_size,
            intra_op_threads=intra_op_threads,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=epochs_per_trial,
        )

        # 4. Train the model (uses engine.trainer.train)
        model, train_loader, test_loader = train(cfg, device=device)

        # Benchmark CPU metrics: Force to CPU
        model = model.cpu()
        torch.set_num_threads(intra_op_threads)

        # Warm up runs
        dummy_input = torch.randn(1, cfg.in_channels, cfg.input_size, cfg.input_size)
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)

        # 5. Measure latency and energy inside evaluation loop under CodeCarbon
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()

        start_time = time.perf_counter()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.cpu()
                labels = labels.squeeze().long().cpu()
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += images.size(0)

        end_time = time.perf_counter()
        tracker.stop()

        val_accuracy = 100.0 * correct / total
        total_time = end_time - start_time
        # CPU Latency: average latency per batch
        latency_sec = total_time / len(test_loader)

        # CPU Energy: CPU + RAM energy consumed during forward pass
        emissions_data = tracker.final_emissions_data
        energy_kwh = (emissions_data.cpu_energy or 0.0) + (emissions_data.ram_energy or 0.0)

        # Save trial weights
        trial_weight_path = CHECKPOINTS_DIR / f"trial_{trial.number}.pth"
        torch.save(model.state_dict(), trial_weight_path)

        print(
            f"  [Trial {trial.number}] params: width={width_multiplier}, res={input_resolution}, batch={batch_size}, threads={intra_op_threads} | "
            f"acc={val_accuracy:.2f}% | "
            f"latency={latency_sec*1000:.2f} ms | "
            f"energy={energy_kwh:.8f} kWh"
        )

        return val_accuracy, latency_sec, energy_kwh

    # Use MOTPE (Multi-Objective TPE) sampler via TPESampler
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name="shufflenet_multi_objective",
        directions=["maximize", "minimize", "minimize"],
        sampler=sampler,
    )

    study.optimize(objective, n_trials=n_trials)

    print(f"\n{'='*60}")
    print("  Optuna Optimization Study Complete!")
    print(f"{'='*60}")

    best_trials = study.best_trials
    print(f"\nFound {len(best_trials)} Pareto-optimal configurations:")
    for t in best_trials:
        print(
            f"  Trial {t.number:2d}: "
            f"acc={t.values[0]:.2f}% | "
            f"latency={t.values[1]*1000:.2f} ms | "
            f"energy={t.values[2]:.8f} kWh | "
            f"params: width={t.params['width_multiplier']}, res={t.params['input_resolution']}"
        )

    # Heuristic: Select Pareto trial with the highest validation accuracy
    best_trial = max(best_trials, key=lambda t: t.values[0])
    print(f"\nSelected Best Trial (Highest Accuracy on Pareto Front):")
    print(f"  Trial:    {best_trial.number}")
    print(f"  Accuracy: {best_trial.values[0]:.2f}%")
    print(f"  Latency:  {best_trial.values[1]*1000:.2f} ms")
    print(f"  Energy:   {best_trial.values[2]:.8f} kWh")
    print(f"  Params:   {best_trial.params}")

    # Reconstruct the config for the selected best model
    best_cfg = ExperimentConfig(
        width_multiplier=best_trial.params["width_multiplier"],
        input_size=best_trial.params["input_resolution"],
        batch_size=best_trial.params["batch_size"],
        intra_op_threads=best_trial.params["intra_op_threads"],
        learning_rate=best_trial.params["learning_rate"],
        weight_decay=best_trial.params["weight_decay"],
        num_epochs=epochs_per_trial,
    )

    # Load float model
    best_model = QuantizableShuffleNetV2(
        width_multiplier=best_cfg.width_multiplier,
        num_classes=best_cfg.num_classes,
        in_channels=best_cfg.in_channels,
        intra_op_threads=best_cfg.intra_op_threads,
    )
    best_weight_path = CHECKPOINTS_DIR / f"trial_{best_trial.number}.pth"
    best_model.load_state_dict(torch.load(best_weight_path, map_location="cpu"))

    # Save float model to best_model_float.pth
    best_float_path = CHECKPOINTS_DIR / "best_model_float.pth"
    torch.save(best_model.state_dict(), best_float_path)
    print(f"\n  [Float] Saved float weights to {best_float_path}")

    # Build loader for calibration
    _, calibration_loader = build_dataloaders(best_cfg, device=torch.device("cpu"))

    # Run Task 2: INT8 Post-Training Quantization
    quantized_model = run_ptq(best_model, calibration_loader)

    # Save quantized model
    best_quant_path = CHECKPOINTS_DIR / "best_model_quantized.pth"
    torch.save(quantized_model.state_dict(), best_quant_path)
    print(f"  [Quant] Saved quantized weights to {best_quant_path}")

    # Clean up trial temporary checkpoint files
    for trial in study.trials:
        p = CHECKPOINTS_DIR / f"trial_{trial.number}.pth"
        if p.exists():
            p.unlink()
    print("\n  Cleaned up temporary trial checkpoints.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Multi-Objective Bayesian Optimization & Post-Training Quantization."
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of trials.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per trial.")
    parser.add_argument("--device", type=str, default=None, help="Device to train on.")
    args = parser.parse_args()

    run_optimization(n_trials=args.trials, epochs_per_trial=args.epochs, device_str=args.device)
