"""
experiments/optuna_optimize.py
------------------------------
Multi-Objective Bayesian Optimization (Optuna MOTPE) and Post-Training
Quantization (PTQ) pipeline for ShuffleNet V2 on MedMNIST.

Search Space (CPU-only, 10 hyperparameters):
  Discrete (categorical):
    - width_multiplier:  [0.5, 1.0, 1.5, 2.0] (look at the layer depth as well)
    - input_resolution:  [20, 24, 28, 32] (add 64)
    - batch_size:        [4, 8, 16, 32, 64, 128]
    - intra_op_threads:  [1, 2, 4, 8]
    - dropout:           [0.0, 0.1, 0.2, 0.3, 0.5]
    - optimizer_name:    ["adam", "sgd", "rmsprop"]
    - scheduler_name:    ["cosine", "step", "onecycle"]
  Continuous (BO-sampled): 
    - learning_rate:     log-uniform [1e-5, 3e-1]
    - weight_decay:      log-uniform [1e-6, 1e-2]
    - label_smoothing:   uniform [0.0, 0.2]
    - momentum:          uniform [0.80, 0.99]  (SGD/RMSprop only)

Objectives (multi-objective MOTPE):
  [0] Maximize accuracy (%)
  [1] Minimize latency (seconds per batch)
  [2] Minimize energy (kWh, CPU + RAM)

All training and profiling is CPU-only.
"""

# try completely connected layer at the end withouyt changing the intermediaries 
# try with adam and sgd only 

import os
import sys
import time
import copy
import io
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna

# Ensure parent directory is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.experiment_config import ExperimentConfig
from engine.trainer import train, build_dataloaders
from engine.profiler import profile
from engine.evaluator import evaluate
from models.shufflenet import QuantizableShuffleNetV2

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
PTQ_CALIBRATION_BATCHES = 2
STUDY_DB   = CHECKPOINTS_DIR.parent / "optuna_study_wide.db"
STUDY_NAME = "shufflenet_multi_objective_wide_v3"

# CPU-only device
CPU = torch.device("cpu")


def run_ptq(model: nn.Module, test_loader: DataLoader) -> nn.Module:
    """
    Applies Static Post-Training Quantization (PTQ) to the model.

    Steps:
      1. Force model and data to CPU.
      2. Fuse modules (Conv + BN + ReLU).
      3. Set qconfig for fbgemm (x86 CPU inference).
      4. Prepare model for static quantization.
      5. Calibrate observers using PTQ_CALIBRATION_BATCHES batches.
      6. Convert model to INT8.
      7. Print footprint reduction.
    """
    print(f"\n{'='*60}")
    print("  Task 2: INT8 Post-Training Quantization (PTQ)")
    print(f"{'='*60}")

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

    # 3. Set qconfig for x86 CPU
    quantizable_model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

    # 4. Prepare for static quantization
    print("  [PTQ] Preparing model (inserting observers)...")
    prepared_model = torch.ao.quantization.prepare(quantizable_model, inplace=False)

    # 5. Calibration loop
    print(f"  [PTQ] Calibrating observers on {PTQ_CALIBRATION_BATCHES} batches...")
    prepared_model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if idx >= PTQ_CALIBRATION_BATCHES:
                break
            _ = prepared_model(images.cpu())

    # 6. Convert to INT8
    print("  [PTQ] Converting model to INT8...")
    quantized_model = torch.ao.quantization.convert(prepared_model, inplace=False)

    # 7. Print after size
    quant_buffer = io.BytesIO()
    torch.save(quantized_model.state_dict(), quant_buffer)
    quant_size_mb = len(quant_buffer.getvalue()) / (1024 * 1024)
    print(f"  [PTQ] Quantized INT8 model size: {quant_size_mb:.3f} MB")

    reduction = (1.0 - (quant_size_mb / float_size_mb)) * 100
    print(f"  [PTQ] Footprint reduction: {reduction:.2f}%\n")

    return quantized_model


def _build_best_config(params: dict, final_epochs: int) -> ExperimentConfig:
    """Reconstruct ExperimentConfig from an Optuna trial's param dict."""
    return ExperimentConfig(
        width_multiplier  = params["width_multiplier"],
        input_size        = params["input_resolution"],
        batch_size        = params["batch_size"],
        intra_op_threads  = params["intra_op_threads"],
        dropout           = params["dropout"],
        optimizer_name    = params["optimizer_name"],
        scheduler_name    = params["scheduler_name"],
        label_smoothing   = params["label_smoothing"],
        momentum          = params.get("momentum", 0.9),
        learning_rate     = params["learning_rate"],
        weight_decay      = params["weight_decay"],
        num_epochs        = final_epochs,
    )


def finalize_from_study(
    final_epochs: int = 8,
    device_str: str = None,  # kept for CLI compatibility, ignored (CPU-only)
) -> None:
    """
    Loads the existing study from disk, selects the best completed trial,
    retrains that configuration on CPU, and runs PTQ.
    """
    print(f"\n{'='*60}")
    print("  Task 2: Final Training + PTQ from Existing Study")
    print(f"  Final epochs: {final_epochs}")
    print(f"  Device: CPU (forced)")
    print(f"  Study DB: {STUDY_DB}")
    print(f"{'='*60}\n")

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
    )

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("No completed trials found in the study database.")

    best_trial = max(completed_trials, key=lambda t: t.values[0])
    print("  Best completed trial from existing study:")
    print(f"    Trial:    {best_trial.number}")
    print(f"    Accuracy: {best_trial.values[0]:.2f}%")
    print(f"    Latency:  {best_trial.values[1]*1000:.2f} ms")
    print(f"    Energy:   {best_trial.values[2]:.8f} kWh")
    print(f"    Params:   {best_trial.params}")

    best_cfg = _build_best_config(best_trial.params, final_epochs)

    best_model = QuantizableShuffleNetV2(
        width_multiplier = best_cfg.width_multiplier,
        num_classes      = best_cfg.num_classes,
        in_channels      = best_cfg.in_channels,
        intra_op_threads = best_cfg.intra_op_threads,
    )
    best_model.dropout.p = best_cfg.dropout

    best_weight_path = CHECKPOINTS_DIR / f"trial_{best_trial.number}.pth"
    if best_weight_path.exists():
        best_model.load_state_dict(torch.load(best_weight_path, map_location="cpu"))
    else:
        print(f"  [Warn] Missing checkpoint {best_weight_path}; retraining from scratch.")

    best_float_path = CHECKPOINTS_DIR / "best_model_float.pth"
    torch.save(best_model.state_dict(), best_float_path)
    print(f"\n  [Float] Saved float weights to {best_float_path}")

    _, calibration_loader = build_dataloaders(best_cfg, device=CPU)
    quantized_model = run_ptq(best_model, calibration_loader)

    best_quant_path = CHECKPOINTS_DIR / "best_model_quantized.pth"
    torch.save(quantized_model.state_dict(), best_quant_path)
    print(f"  [Quant] Saved quantized weights to {best_quant_path}")

    cpu_profile = profile(best_model.cpu(), best_cfg)
    energy_uh = cpu_profile.energy_kwh * 1e9
    print(f"  [CPU Bench] Inference-only energy (100 runs): {energy_uh:.2f} uWh")


def run_optimization(
    n_trials: int = 50,
    search_epochs: int = 2,
    final_epochs: int = 8,
    device_str: str = None,  # kept for CLI compatibility, ignored (CPU-only)
    n_jobs: int = 1,
) -> None:
    """
    Runs Multi-Objective Bayesian Optimization (MOTPE) over a 10-parameter
    search space on CPU only.

    After finding the Pareto front, it selects the trial with the highest
    validation accuracy, saves its weights, and runs INT8 PTQ.

    Search space (34,560+ discrete combos + continuous dims):
      categorical: width_multiplier, input_resolution, batch_size,
                   intra_op_threads, dropout, optimizer_name, scheduler_name
      continuous:  learning_rate, weight_decay, label_smoothing, momentum
    """
    print(f"\n{'='*60}")
    print("  Task 1: Multi-Objective Bayesian Optimization (Optuna MOTPE)")
    print(f"  Trials: {n_trials} | Search epochs: {search_epochs} | Final epochs: {final_epochs}")
    print(f"  Parallel workers: {n_jobs}")
    print(f"  Device: CPU (forced)")
    print(f"  Search space: 10 hyperparameters (34,560+ discrete combos)")
    print(f"{'='*60}\n")

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        # ── 1. Sample discrete hyperparameters ──────────────────────────────────
        width_multiplier = trial.suggest_categorical(
            "width_multiplier", [0.5, 1.0, 1.5, 2.0]
        )
        input_resolution = trial.suggest_categorical(
            "input_resolution", [20, 24, 28, 32]
        )
        batch_size = trial.suggest_categorical(
            "batch_size", [4, 8, 16, 32, 64, 128]
        )
        intra_op_threads = trial.suggest_categorical(
            "intra_op_threads", [1, 2, 4, 8]
        )
        dropout = trial.suggest_categorical(
            "dropout", [0.0, 0.1, 0.2, 0.3, 0.5]
        )
        optimizer_name = trial.suggest_categorical(
            "optimizer_name", ["adam", "sgd", "rmsprop"]
        )
        scheduler_name = trial.suggest_categorical(
            "scheduler_name", ["cosine", "step", "onecycle"]
        )

        # ── 2. Sample continuous hyperparameters ─────────────────────────────────
        learning_rate   = trial.suggest_float("learning_rate",   1e-5, 3e-1, log=True)
        weight_decay    = trial.suggest_float("weight_decay",    1e-6, 1e-2, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0,  0.20)

        # Momentum is only meaningful for SGD / RMSprop
        if optimizer_name in ("sgd", "rmsprop"):
            momentum = trial.suggest_float("momentum", 0.80, 0.99)
        else:
            momentum = 0.9  # stored but unused by Adam

        # ── 3. Set CPU thread count for this trial ───────────────────────────────
        torch.set_num_threads(intra_op_threads)

        # ── 4. Build config ──────────────────────────────────────────────────────
        cfg = ExperimentConfig(
            width_multiplier = width_multiplier,
            input_size       = input_resolution,
            batch_size       = batch_size,
            intra_op_threads = intra_op_threads,
            dropout          = dropout,
            optimizer_name   = optimizer_name,
            scheduler_name   = scheduler_name,
            label_smoothing  = label_smoothing,
            momentum         = momentum,
            learning_rate    = learning_rate,
            weight_decay     = weight_decay,
            num_epochs       = search_epochs,
        )

        # ── 5. Train the model (CPU-only) ────────────────────────────────────────
        model, train_loader, test_loader = train(cfg, device=CPU)
        model = model.cpu()
        torch.set_num_threads(intra_op_threads)

        # ── 6. Warm-up forward passes ────────────────────────────────────────────
        dummy_input = torch.randn(1, cfg.in_channels, cfg.input_size, cfg.input_size)
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)

        # ── 7. Measure latency + energy (CodeCarbon) over the test set ───────────
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()

        start_time = time.perf_counter()
        correct = 0
        total   = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.cpu()
                labels = labels.squeeze().long().cpu()
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total   += images.size(0)

        end_time = time.perf_counter()
        tracker.stop()

        val_accuracy = 100.0 * correct / total
        total_time   = end_time - start_time
        # Average latency per batch
        latency_sec  = total_time / max(1, len(test_loader))

        emissions_data = tracker.final_emissions_data
        energy_kwh = (emissions_data.cpu_energy or 0.0) + (emissions_data.ram_energy or 0.0)

        # ── 8. Save trial checkpoint ─────────────────────────────────────────────
        trial_weight_path = CHECKPOINTS_DIR / f"trial_{trial.number}.pth"
        torch.save(model.state_dict(), trial_weight_path)

        print(
            f"  [Trial {trial.number:3d}] "
            f"w={width_multiplier} res={input_resolution} bs={batch_size} "
            f"thr={intra_op_threads} do={dropout} "
            f"opt={optimizer_name} sched={scheduler_name} ls={label_smoothing:.3f} | "
            f"acc={val_accuracy:.2f}% | "
            f"lat={latency_sec*1000:.2f}ms | "
            f"energy={energy_kwh:.8f}kWh"
        )

        return val_accuracy, latency_sec, energy_kwh

    # ── Create / resume MOTPE study ──────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name  = STUDY_NAME,
        directions  = ["maximize", "minimize", "minimize"],
        sampler     = sampler,
        storage     = f"sqlite:///{STUDY_DB}",
        load_if_exists = True,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

    print(f"\n{'='*60}")
    print("  Optuna MOTPE Study Complete!")
    print(f"{'='*60}")

    best_trials = study.best_trials
    print(f"\nFound {len(best_trials)} Pareto-optimal configurations:")
    for t in best_trials:
        print(
            f"  Trial {t.number:3d}: "
            f"acc={t.values[0]:.2f}% | "
            f"lat={t.values[1]*1000:.2f}ms | "
            f"energy={t.values[2]:.8f}kWh | "
            f"w={t.params['width_multiplier']} res={t.params['input_resolution']} "
            f"opt={t.params['optimizer_name']} sched={t.params['scheduler_name']}"
        )

    # Heuristic: Pareto trial with the highest accuracy
    best_trial = max(best_trials, key=lambda t: t.values[0])
    print(f"\nSelected Best Trial (Highest Accuracy on Pareto Front):")
    print(f"  Trial:    {best_trial.number}")
    print(f"  Accuracy: {best_trial.values[0]:.2f}%")
    print(f"  Latency:  {best_trial.values[1]*1000:.2f} ms")
    print(f"  Energy:   {best_trial.values[2]:.8f} kWh")
    print(f"  Params:   {best_trial.params}")

    # ── Reconstruct best config and model ────────────────────────────────────────
    best_cfg = _build_best_config(best_trial.params, final_epochs)

    best_model = QuantizableShuffleNetV2(
        width_multiplier = best_cfg.width_multiplier,
        num_classes      = best_cfg.num_classes,
        in_channels      = best_cfg.in_channels,
        intra_op_threads = best_cfg.intra_op_threads,
    )
    best_model.dropout.p = best_cfg.dropout

    best_weight_path = CHECKPOINTS_DIR / f"trial_{best_trial.number}.pth"
    if best_weight_path.exists():
        best_model.load_state_dict(torch.load(best_weight_path, map_location="cpu"))
    else:
        print(f"  [Warn] Checkpoint not found at {best_weight_path}. Weights are from last training epoch.")

    # Save float model
    best_float_path = CHECKPOINTS_DIR / "best_model_float.pth"
    torch.save(best_model.state_dict(), best_float_path)
    print(f"\n  [Float] Saved float weights to {best_float_path}")

    # Build calibration loader and run PTQ
    _, calibration_loader = build_dataloaders(best_cfg, device=CPU)
    quantized_model = run_ptq(best_model, calibration_loader)

    best_quant_path = CHECKPOINTS_DIR / "best_model_quantized.pth"
    torch.save(quantized_model.state_dict(), best_quant_path)
    print(f"  [Quant] Saved quantized weights to {best_quant_path}")

    # Clean up per-trial checkpoints
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
    parser.add_argument("--trials", type=int, default=50, help="Number of trials.")
    parser.add_argument("--epochs", type=int, default=2,   help="Training epochs per trial.")
    args = parser.parse_args()

    run_optimization(n_trials=args.trials, search_epochs=args.epochs)
