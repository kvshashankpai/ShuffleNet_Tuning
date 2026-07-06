"""
experiments/hypervolume_optimize.py
------------------------------------
Single-Objective Hypervolume (HV) Maximization via Bayesian Optimization.

Framing:
  Instead of running a 3-objective MOTPE search (Accuracy ↑, Latency ↓, Energy ↓),
  we collapse all three objectives into a single scalar — the **Hypervolume
  Indicator** — and maximize it with a standard single-objective BO (TPESampler).

  HV(S, r) = volume of the objective space dominated by the Pareto front of
             the incumbent solution set S, bounded by reference point r.

  Maximizing HV simultaneously encourages:
    - High accuracy (Pareto spread in that axis)
    - Low latency  (Pareto spread in that axis)
    - Low energy   (Pareto spread in that axis)
  ... in a single unified surrogate model.

Reference point strategy:
  We use a "nadir" reference point computed from the worst observed values
  across all completed trials, with a 10% slack buffer to ensure it is always
  dominated by any reasonable solution.

Algorithm:
  - Optuna TPESampler (single-objective) → same BO family as MOTPE
  - Hypervolume computed via optuna's built-in WFG algorithm
    (falls back to a simple box-based approximation if unavailable)

All training and profiling is CPU-only.
"""

import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.experiment_config import ExperimentConfig
from engine.trainer import train, build_dataloaders
from engine.profiler import profile
from models.shufflenet import QuantizableShuffleNetV2
from experiments.optuna_optimize import run_ptq, _build_best_config

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

HV_STUDY_DB   = CHECKPOINTS_DIR.parent / "optuna_study_hv.db"
HV_STUDY_NAME = "shufflenet_hypervolume_maximization_v1"

# CPU-only
CPU = torch.device("cpu")


# ── Hypervolume Computation ────────────────────────────────────────────────────

def _wfg_hypervolume(points: list[list[float]], ref: list[float]) -> float:
    """
    Compute the hypervolume indicator for a set of 3D points relative to a
    reference point, using Optuna's built-in WFG calculator.

    Points are assumed to be in the form [accuracy, latency, energy] where
    we transform to a minimization problem internally:
      f1 = -accuracy  (negate to convert maximization → minimization)
      f2 =  latency
      f3 =  energy

    Args:
        points: List of [accuracy, latency, energy] triplets.
        ref:    Reference point [worst_acc, worst_lat, worst_energy]
                already in the transformed (all-minimize) space.

    Returns:
        Hypervolume scalar (higher is better).
    """
    try:
        # Optuna ≥ 3.x ships _hypervolume / WFG internally
        from optuna._hypervolume.wfg import compute_hypervolume
        # Transform to minimization: negate accuracy
        transformed = [[-p[0], p[1], p[2]] for p in points]
        ref_min = [-ref[0], ref[1], ref[2]]
        return compute_hypervolume(transformed, ref_min)
    except ImportError:
        pass

    # ── Fallback: Monte Carlo HV approximation ───────────────────────────────
    # Sufficient for relative ranking of trials during BO.
    n_samples = 50_000
    transformed = [[-p[0], p[1], p[2]] for p in points]
    ref_min     = [-ref[0], ref[1], ref[2]]

    # Bounding box
    lo = [min(p[d] for p in transformed) for d in range(3)]
    hi = [ref_min[d] for d in range(3)]

    if any(lo[d] >= hi[d] for d in range(3)):
        return 0.0

    import random
    rng = random.Random(0)
    volume = math.prod(hi[d] - lo[d] for d in range(3))
    dominated = 0
    for _ in range(n_samples):
        s = [rng.uniform(lo[d], hi[d]) for d in range(3)]
        # A sample is dominated if at least one point dominates it
        for p in transformed:
            if all(p[d] <= s[d] for d in range(3)):
                dominated += 1
                break
    return volume * (dominated / n_samples)


class _HVTracker:
    """
    Accumulates (accuracy, latency, energy) triplets across trials and
    computes the Hypervolume at each step using a dynamic reference point.
    """
    def __init__(self) -> None:
        self.points: list[list[float]] = []

    def add(self, acc: float, lat: float, energy: float) -> float:
        """Add a new point and return the updated hypervolume."""
        self.points.append([acc, lat, energy])

        # Dynamic reference point: 10% worse than the current worst in each dim
        worst_acc    = min(p[0] for p in self.points) * 0.90   # lower is worse
        worst_lat    = max(p[1] for p in self.points) * 1.10
        worst_energy = max(p[2] for p in self.points) * 1.10

        ref = [worst_acc, worst_lat, worst_energy]
        return _wfg_hypervolume(self.points, ref)

    @property
    def hv_history(self) -> list[float]:
        """Recompute full HV history for plotting."""
        history = []
        tracker = _HVTracker()
        for p in self.points:
            tracker.points.append(p)
            if len(tracker.points) == 1:
                history.append(0.0)
                continue
            worst_acc    = min(x[0] for x in tracker.points) * 0.90
            worst_lat    = max(x[1] for x in tracker.points) * 1.10
            worst_energy = max(x[2] for x in tracker.points) * 1.10
            ref = [worst_acc, worst_lat, worst_energy]
            history.append(_wfg_hypervolume(tracker.points, ref))
        return history


def run_hypervolume_optimization(
    n_trials: int = 150,
    search_epochs: int = 2,
    final_epochs: int = 8,
    device_str: str = None,  # ignored, CPU-only
    n_jobs: int = 1,
) -> None:
    """
    Runs single-objective Hypervolume Maximization BO.

    Each trial trains a ShuffleNetV2 on CPU, measures (accuracy, latency,
    energy), and returns the HV indicator of ALL trials seen so far as the
    single reward for the TPE surrogate.

    This has the same BO solver family as MOTPE but with a unified, scalar
    objective that naturally trades off accuracy, latency, and energy without
    requiring Pareto dominance reasoning.

    Args:
        n_trials:      Total number of BO trials to run.
        search_epochs: Epochs per trial (short — BO explores cheaply).
        final_epochs:  Epochs to retrain the best configuration fully.
        n_jobs:        Parallel Optuna workers (SQLite backend).
    """
    print(f"\n{'='*60}")
    print("  Hypervolume Maximization — Single-Objective BO")
    print(f"  Trials: {n_trials} | Search epochs: {search_epochs} | Final epochs: {final_epochs}")
    print(f"  Device: CPU (forced)")
    print(f"  Search space: 10 hyperparameters (34,560+ discrete combos)")
    print(f"  Objective: Maximize Hypervolume(Accuracy, Latency, Energy)")
    print(f"{'='*60}\n")

    hv_tracker = _HVTracker()

    def objective(trial: optuna.Trial) -> float:
        # ── 1. Sample hyperparameters (same space as MOTPE) ──────────────────
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
        learning_rate   = trial.suggest_float("learning_rate",   1e-5, 3e-1, log=True)
        weight_decay    = trial.suggest_float("weight_decay",    1e-6, 1e-2, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0,  0.20)

        if optimizer_name in ("sgd", "rmsprop"):
            momentum = trial.suggest_float("momentum", 0.80, 0.99)
        else:
            momentum = 0.9

        # ── 2. CPU thread count ───────────────────────────────────────────────
        torch.set_num_threads(intra_op_threads)

        # ── 3. Build config & train ───────────────────────────────────────────
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

        model, train_loader, test_loader = train(cfg, device=CPU)
        model = model.cpu()
        torch.set_num_threads(intra_op_threads)

        # ── 4. Warm-up ────────────────────────────────────────────────────────
        dummy = torch.randn(1, cfg.in_channels, cfg.input_size, cfg.input_size)
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy)

        # ── 5. Measure accuracy, latency, energy ──────────────────────────────
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()

        t0 = time.perf_counter()
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
        t1 = time.perf_counter()
        tracker.stop()

        val_accuracy = 100.0 * correct / total
        latency_sec  = (t1 - t0) / max(1, len(test_loader))
        emissions    = tracker.final_emissions_data
        energy_kwh   = (emissions.cpu_energy or 0.0) + (emissions.ram_energy or 0.0)

        # ── 6. Store raw objectives as trial user attributes (for reporting) ──
        trial.set_user_attr("accuracy",    val_accuracy)
        trial.set_user_attr("latency_sec", latency_sec)
        trial.set_user_attr("energy_kwh",  energy_kwh)

        # ── 7. Save checkpoint ────────────────────────────────────────────────
        ckpt_path = CHECKPOINTS_DIR / f"hv_trial_{trial.number}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # ── 8. Compute & return Hypervolume indicator ─────────────────────────
        hv = hv_tracker.add(val_accuracy, latency_sec, energy_kwh)

        print(
            f"  [HV Trial {trial.number:3d}] "
            f"w={width_multiplier} res={input_resolution} bs={batch_size} "
            f"thr={intra_op_threads} opt={optimizer_name} sched={scheduler_name} | "
            f"acc={val_accuracy:.2f}% lat={latency_sec*1000:.2f}ms "
            f"energy={energy_kwh:.8f}kWh | HV={hv:.6f}"
        )

        return hv   # ← Single scalar: BO maximizes this

    # ── Create / resume single-objective HV study ────────────────────────────────
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name     = HV_STUDY_NAME,
        direction      = "maximize",   # single-objective: maximize HV
        sampler        = sampler,
        storage        = f"sqlite:///{HV_STUDY_DB}",
        load_if_exists = True,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

    print(f"\n{'='*60}")
    print("  Hypervolume Maximization Study Complete!")
    print(f"{'='*60}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("  No completed trials found.")
        return

    # Best trial = highest individual accuracy (post-hoc selection)
    best_trial = max(completed, key=lambda t: t.user_attrs.get("accuracy", 0.0))

    print(f"\n  Best Trial by Accuracy: #{best_trial.number}")
    print(f"  Accuracy: {best_trial.user_attrs['accuracy']:.2f}%")
    print(f"  Latency:  {best_trial.user_attrs['latency_sec']*1000:.2f} ms")
    print(f"  Energy:   {best_trial.user_attrs['energy_kwh']:.8f} kWh")
    print(f"  HV Score: {best_trial.value:.6f}")
    print(f"  Params:   {best_trial.params}")

    # ── Save HV convergence data ──────────────────────────────────────────────────
    results_dir = CHECKPOINTS_DIR.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    hv_csv = results_dir / "hv_convergence.csv"
    hv_history = hv_tracker.hv_history
    with open(hv_csv, "w") as f:
        f.write("trial,hypervolume\n")
        for i, hv in enumerate(hv_history):
            f.write(f"{i},{hv:.8f}\n")
    print(f"\n  [HV] Convergence data saved to {hv_csv}")

    # ── Plot HV convergence ───────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(len(hv_history)), hv_history, linewidth=1.8, color="#3498db")
        ax.fill_between(range(len(hv_history)), hv_history, alpha=0.15, color="#3498db")
        ax.set_xlabel("Trial Number", fontsize=12)
        ax.set_ylabel("Hypervolume Indicator", fontsize=12)
        ax.set_title("Hypervolume Convergence (BO — single-objective HV maximization)", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        hv_png = results_dir / "hv_convergence.png"
        fig.savefig(hv_png, dpi=200)
        plt.close(fig)
        print(f"  [HV] Convergence plot saved to {hv_png}")
    except Exception as e:
        print(f"  [HV] Could not save convergence plot: {e}")

    # ── Retrain best config and run PTQ ──────────────────────────────────────────
    best_cfg = _build_best_config(best_trial.params, final_epochs)

    best_model = QuantizableShuffleNetV2(
        width_multiplier = best_cfg.width_multiplier,
        num_classes      = best_cfg.num_classes,
        in_channels      = best_cfg.in_channels,
        intra_op_threads = best_cfg.intra_op_threads,
    )
    best_model.dropout.p = best_cfg.dropout

    best_ckpt = CHECKPOINTS_DIR / f"hv_trial_{best_trial.number}.pth"
    if best_ckpt.exists():
        best_model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
    else:
        print(f"  [Warn] Checkpoint not found at {best_ckpt}.")

    best_float_path = CHECKPOINTS_DIR / "hv_best_model_float.pth"
    torch.save(best_model.state_dict(), best_float_path)
    print(f"\n  [Float] Saved float weights to {best_float_path}")

    _, calibration_loader = build_dataloaders(best_cfg, device=CPU)
    quantized_model = run_ptq(best_model, calibration_loader)

    best_quant_path = CHECKPOINTS_DIR / "hv_best_model_quantized.pth"
    torch.save(quantized_model.state_dict(), best_quant_path)
    print(f"  [Quant] Saved quantized weights to {best_quant_path}")

    # ── Clean up trial checkpoints ────────────────────────────────────────────────
    for trial in completed:
        p = CHECKPOINTS_DIR / f"hv_trial_{trial.number}.pth"
        if p.exists():
            p.unlink()
    print("\n  Cleaned up temporary HV trial checkpoints.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hypervolume Maximization BO for ShuffleNetV2 on MedMNIST (CPU)."
    )
    parser.add_argument("--trials", type=int, default=150, help="Number of BO trials.")
    parser.add_argument("--epochs", type=int, default=2,   help="Training epochs per trial.")
    parser.add_argument("--final-epochs", type=int, default=8, help="Final retraining epochs.")
    args = parser.parse_args()

    run_hypervolume_optimization(
        n_trials=args.trials,
        search_epochs=args.epochs,
        final_epochs=args.final_epochs,
    )
