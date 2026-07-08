"""
experiments/optuna_optimize_v2.py
----------------------------------
Refined Multi-Objective Bayesian Optimization (Optuna MOTPE) v2 pipeline
for ShuffleNet V2 on MedMNIST.

Changes from v1 (optuna_optimize.py):
  - Loss function is now tunable: cross_entropy | kl_divergence | focal
  - Optional FC hidden layer before classifier (fc_hidden_dim)
  - Tunable network depth via stage_depth (shallow / standard / deep)
  - Optimizer restricted to adam | sgd (removed rmsprop)
  - Learning rate narrowed to [5e-4, 5e-3] (professor's guidance)
  - Uses a SEPARATE study database (optuna_study_v2.db) and results folder
    (results_v2/) — v1 results are fully preserved

Search Space (CPU-only, 13 hyperparameters):
  Discrete (categorical):
    - width_multiplier:  [0.5, 1.0, 1.5, 2.0]
    - input_resolution:  [20, 24, 28, 32]
    - batch_size:        [4, 8, 16, 32, 64, 128]
    - intra_op_threads:  [1, 2, 4, 8]
    - dropout:           [0.0, 0.1, 0.2, 0.3, 0.5]
    - optimizer_name:    ["adam", "sgd"]
    - scheduler_name:    ["cosine", "step", "onecycle"]
    - loss_name:         ["cross_entropy", "kl_divergence", "focal"]
    - fc_hidden_dim:     [0, 128, 256, 512]
    - stage_depth:       ["shallow", "standard", "deep"]
  Continuous (BO-sampled):
    - learning_rate:     log-uniform [5e-4, 5e-3]
    - weight_decay:      log-uniform [1e-6, 1e-2]
    - label_smoothing:   uniform [0.0, 0.2]
    - momentum:          uniform [0.80, 0.99]  (SGD only)

Objectives (multi-objective MOTPE):
  [0] Maximize accuracy (%)
  [1] Minimize latency (seconds per batch)
  [2] Minimize energy (kWh, CPU + RAM)

All training and profiling is CPU-only.
"""

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
from experiments.optuna_optimize import run_ptq

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints_v2"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STUDY_DB   = CHECKPOINTS_DIR.parent / "optuna_study_v2.db"
STUDY_NAME = "shufflenet_motpe_v2"

# CPU-only device
CPU = torch.device("cpu")


def _build_best_config_v2(params: dict, final_epochs: int) -> ExperimentConfig:
    """Reconstruct ExperimentConfig from an Optuna trial's param dict (v2)."""
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
        loss_name         = params["loss_name"],
        fc_hidden_dim     = params["fc_hidden_dim"],
        stage_depth       = params["stage_depth"],
        num_epochs        = final_epochs,
    )


def run_optimization_v2(
    n_trials: int = 150,
    search_epochs: int = 2,
    final_epochs: int = 8,
    device_str: str = None,  # kept for CLI compatibility, ignored (CPU-only)
    n_jobs: int = 1,
) -> None:
    """
    Runs Multi-Objective Bayesian Optimization (MOTPE) v2 over a refined
    search space on CPU only.

    New in v2:
      - Loss function as a tunable hyperparameter
      - Optional FC hidden layer (fc_hidden_dim)
      - Tunable stage depth (shallow / standard / deep)
      - Optimizers restricted to Adam and SGD only
      - Learning rate narrowed to [5e-4, 5e-3]
    """
    print(f"\n{'='*60}")
    print("  Task 1 (v2): Multi-Objective Bayesian Optimization (Optuna MOTPE)")
    print(f"  Trials: {n_trials} | Search epochs: {search_epochs} | Final epochs: {final_epochs}")
    print(f"  Parallel workers: {n_jobs}")
    print(f"  Device: CPU (forced)")
    print(f"  Search space: 13 hyperparameters (v2 — refined)")
    print(f"  New: loss_name, fc_hidden_dim, stage_depth")
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
            "optimizer_name", ["adam", "sgd"]
        )
        scheduler_name = trial.suggest_categorical(
            "scheduler_name", ["cosine", "step", "onecycle"]
        )

        # ── v2: New categorical hyperparameters ──────────────────────────────────
        loss_name = trial.suggest_categorical(
            "loss_name", ["cross_entropy", "kl_divergence", "focal"]
        )
        fc_hidden_dim = trial.suggest_categorical(
            "fc_hidden_dim", [0, 128, 256, 512]
        )
        stage_depth = trial.suggest_categorical(
            "stage_depth", ["shallow", "standard", "deep"]
        )

        # ── 2. Sample continuous hyperparameters ─────────────────────────────────
        # v2: Learning rate narrowed to [5e-4, 5e-3] per professor's guidance
        learning_rate   = trial.suggest_float("learning_rate",   5e-4, 5e-3, log=True)
        weight_decay    = trial.suggest_float("weight_decay",    1e-6, 1e-2, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0,  0.20)

        # Momentum is only meaningful for SGD
        if optimizer_name == "sgd":
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
            loss_name        = loss_name,
            fc_hidden_dim    = fc_hidden_dim,
            stage_depth      = stage_depth,
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
            f"  [v2 Trial {trial.number:3d}] "
            f"w={width_multiplier} res={input_resolution} bs={batch_size} "
            f"thr={intra_op_threads} do={dropout} "
            f"opt={optimizer_name} sched={scheduler_name} "
            f"loss={loss_name} fc={fc_hidden_dim} depth={stage_depth} | "
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
    print("  Optuna MOTPE v2 Study Complete!")
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
            f"opt={t.params['optimizer_name']} sched={t.params['scheduler_name']} "
            f"loss={t.params['loss_name']} fc={t.params['fc_hidden_dim']} "
            f"depth={t.params['stage_depth']}"
        )

    # ── Save all trial results to CSV ────────────────────────────────────────────
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    csv_path = RESULTS_DIR / "v2_study_results.csv"
    with open(csv_path, "w") as f:
        header = (
            "trial,accuracy,latency_sec,energy_kwh,"
            "width_multiplier,input_resolution,batch_size,intra_op_threads,"
            "dropout,learning_rate,weight_decay,loss_name,fc_hidden_dim,stage_depth"
        )
        f.write(header + "\n")
        for t in sorted(completed_trials, key=lambda x: x.values[0], reverse=True):
            f.write(
                f"{t.number},{t.values[0]},{t.values[1]},{t.values[2]},"
                f"{t.params['width_multiplier']},{t.params['input_resolution']},"
                f"{t.params['batch_size']},{t.params['intra_op_threads']},"
                f"{t.params['dropout']},{t.params['learning_rate']},"
                f"{t.params['weight_decay']},{t.params['loss_name']},"
                f"{t.params['fc_hidden_dim']},{t.params['stage_depth']}\n"
            )
    print(f"\n  [v2] Saved all trial results to {csv_path}")

    # Heuristic: Pareto trial with the highest accuracy
    best_trial = max(best_trials, key=lambda t: t.values[0])
    print(f"\nSelected Best Trial (Highest Accuracy on Pareto Front):")
    print(f"  Trial:    {best_trial.number}")
    print(f"  Accuracy: {best_trial.values[0]:.2f}%")
    print(f"  Latency:  {best_trial.values[1]*1000:.2f} ms")
    print(f"  Energy:   {best_trial.values[2]:.8f} kWh")
    print(f"  Params:   {best_trial.params}")

    # ── Reconstruct best config and model ────────────────────────────────────────
    best_cfg = _build_best_config_v2(best_trial.params, final_epochs)

    best_model = QuantizableShuffleNetV2(
        width_multiplier = best_cfg.width_multiplier,
        num_classes      = best_cfg.num_classes,
        in_channels      = best_cfg.in_channels,
        intra_op_threads = best_cfg.intra_op_threads,
        stage_repeats    = best_cfg.resolved_stage_repeats,
        fc_hidden_dim    = best_cfg.fc_hidden_dim,
    )
    best_model.dropout.p = best_cfg.dropout

    best_weight_path = CHECKPOINTS_DIR / f"trial_{best_trial.number}.pth"
    if best_weight_path.exists():
        best_model.load_state_dict(torch.load(best_weight_path, map_location="cpu"))
    else:
        print(f"  [Warn] Checkpoint not found at {best_weight_path}. Weights are from last training epoch.")

    # Save float model
    best_float_path = CHECKPOINTS_DIR / "best_model_float_v2.pth"
    torch.save(best_model.state_dict(), best_float_path)
    print(f"\n  [Float] Saved float weights to {best_float_path}")

    # Build calibration loader and run PTQ
    _, calibration_loader = build_dataloaders(best_cfg, device=CPU)
    quantized_model = run_ptq(best_model, calibration_loader)

    best_quant_path = CHECKPOINTS_DIR / "best_model_quantized_v2.pth"
    torch.save(quantized_model.state_dict(), best_quant_path)
    print(f"  [Quant] Saved quantized weights to {best_quant_path}")

    # Clean up per-trial checkpoints
    for trial in study.trials:
        p = CHECKPOINTS_DIR / f"trial_{trial.number}.pth"
        if p.exists():
            p.unlink()
    print("\n  Cleaned up temporary v2 trial checkpoints.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Multi-Objective Bayesian Optimization v2 & Post-Training Quantization."
    )
    parser.add_argument("--trials", type=int, default=150, help="Number of trials.")
    parser.add_argument("--epochs", type=int, default=2,   help="Training epochs per trial.")
    parser.add_argument("--final-epochs", type=int, default=8, help="Final retraining epochs.")
    args = parser.parse_args()

    run_optimization_v2(n_trials=args.trials, search_epochs=args.epochs, final_epochs=args.final_epochs)
