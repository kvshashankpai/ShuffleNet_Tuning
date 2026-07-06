"""Generate summary artifacts from an Optuna study.

Generates:
  - final_study_results.csv      (all completed trials with all 11 hyperparameters)
  - accuracy_vs_latency.png      (2D Pareto slice)
  - accuracy_vs_energy.png       (2D Pareto slice)
  - pareto_3d.png                (3D Pareto front scatter)
  - hv_convergence.png           (Hypervolume indicator vs. trial — if HV study exists)
  - param_importance.png         (Optuna hyperparameter importance chart)
  - ShuffleNetV2_BO_Report.md    (text summary)
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
RESULTS     = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
DB_PATH     = ROOT / "optuna_study_wide.db"
HV_DB_PATH  = ROOT / "optuna_study_hv.db"
STUDY_NAME    = "shufflenet_multi_objective_wide_v3"
HV_STUDY_NAME = "shufflenet_hypervolume_maximization_v1"

import sys
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import optuna


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── Load MOTPE study ──────────────────────────────────────────────────────
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{DB_PATH}",
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    completed.sort(key=lambda t: t.values[0], reverse=True)

    # ── Extended CSV export (all 11 hyperparameters) ──────────────────────────
    csv_path = RESULTS / "final_study_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "accuracy",
            "latency_sec",
            "energy_kwh",
            # Hyperparameters
            "width_multiplier",
            "input_resolution",
            "batch_size",
            "intra_op_threads",
            "dropout",
            "optimizer_name",
            "scheduler_name",
            "label_smoothing",
            "momentum",
            "learning_rate",
            "weight_decay",
        ])
        for t in completed:
            writer.writerow([
                t.number,
                t.values[0],
                t.values[1],
                t.values[2],
                t.params.get("width_multiplier", ""),
                t.params.get("input_resolution", ""),
                t.params.get("batch_size", ""),
                t.params.get("intra_op_threads", ""),
                t.params.get("dropout", ""),
                t.params.get("optimizer_name", ""),
                t.params.get("scheduler_name", ""),
                t.params.get("label_smoothing", ""),
                t.params.get("momentum", ""),
                t.params.get("learning_rate", ""),
                t.params.get("weight_decay", ""),
            ])
    print(f"[Report] CSV saved to {csv_path}")

    # ── Pareto 2D plots ───────────────────────────────────────────────────────
    best  = completed[0]
    pareto = study.best_trials
    pareto_set = {t.number for t in pareto}

    width_to_color = {0.5: "#8e44ad", 1.0: "#3498db", 1.5: "#2ecc71", 2.0: "#f1c40f"}

    fig = plt.figure(figsize=(8, 6))
    for w in sorted(width_to_color):
        xs = [t.values[1] * 1000 for t in completed if t.params.get("width_multiplier") == w]
        ys = [t.values[0]         for t in completed if t.params.get("width_multiplier") == w]
        plt.scatter(xs, ys, s=18, alpha=0.8, label=str(w), color=width_to_color[w])
    px = [t.values[1] * 1000 for t in pareto]
    py = [t.values[0]         for t in pareto]
    plt.plot(px, py, "r--", marker="x", markersize=6, linewidth=1.2, label="Pareto Frontier")
    plt.xscale("log")
    plt.xlabel("Latency (ms, log scale)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Latency — MOTPE Pareto Front")
    plt.legend(title="Width", loc="lower right")
    plt.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_latency.png", dpi=200)
    plt.close(fig)
    print(f"[Report] Saved accuracy_vs_latency.png")

    fig = plt.figure(figsize=(8, 6))
    for w in sorted(width_to_color):
        xs = [t.values[2] for t in completed if t.params.get("width_multiplier") == w]
        ys = [t.values[0] for t in completed if t.params.get("width_multiplier") == w]
        plt.scatter(xs, ys, s=18, alpha=0.8, label=str(w), color=width_to_color[w])
    px = [t.values[2] for t in pareto]
    py = [t.values[0] for t in pareto]
    plt.plot(px, py, "r--", marker="x", markersize=6, linewidth=1.2, label="Pareto Frontier")
    plt.xscale("log")
    plt.xlabel("Energy (kWh, log scale)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Energy — MOTPE Pareto Front")
    plt.legend(title="Width", loc="lower right")
    plt.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_energy.png", dpi=200)
    plt.close(fig)
    print(f"[Report] Saved accuracy_vs_energy.png")

    # ── True 3D Pareto view ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        [t.values[1] * 1000 for t in completed],
        [t.values[2]         for t in completed],
        [t.values[0]         for t in completed],
        c=[t.params.get("width_multiplier", 1.0) for t in completed],
        cmap="viridis",
        s=18,
        alpha=0.8,
    )
    ax.scatter(
        [t.values[1] * 1000 for t in pareto],
        [t.values[2]         for t in pareto],
        [t.values[0]         for t in pareto],
        c="red",
        marker="x",
        s=60,
        linewidths=1.5,
        label="3D Pareto Optimal",
    )
    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("Energy per test set (kWh)")
    ax.set_zlabel("Test Accuracy (%)")
    ax.set_title("3D Pareto Frontier: Accuracy × Latency × Energy")
    fig.colorbar(scatter, ax=ax, label="Width Multiplier")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(RESULTS / "pareto_3d.png", dpi=200)
    plt.close(fig)
    print(f"[Report] Saved pareto_3d.png")

    # ── Hyperparameter Importance Plot (MOTPE study) ──────────────────────────
    try:
        importances = optuna.importance.get_param_importances(
            study,
            target=lambda t: t.values[0],  # accuracy as target
            target_name="accuracy",
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        params  = list(importances.keys())
        values  = list(importances.values())
        colors  = ["#3498db" if v >= max(values) * 0.5 else "#95a5a6" for v in values]
        ax.barh(params[::-1], values[::-1], color=colors[::-1])
        ax.set_xlabel("Relative Importance (FAnova)", fontsize=11)
        ax.set_title("Hyperparameter Importance — Accuracy Objective", fontsize=12)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(RESULTS / "param_importance.png", dpi=200)
        plt.close(fig)
        print(f"[Report] Saved param_importance.png")
    except Exception as e:
        print(f"[Report] Could not compute param importance: {e}")

    # ── HV Convergence Plot (from HV study, if it exists) ────────────────────
    if HV_DB_PATH.exists():
        try:
            hv_study = optuna.load_study(
                study_name=HV_STUDY_NAME,
                storage=f"sqlite:///{HV_DB_PATH}",
            )
            hv_completed = [
                t for t in hv_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            hv_completed.sort(key=lambda t: t.number)
            hv_values = [t.value for t in hv_completed]
            running_max = []
            cur_max = float("-inf")
            for v in hv_values:
                cur_max = max(cur_max, v)
                running_max.append(cur_max)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(range(len(hv_values)), hv_values, alpha=0.4,
                    linewidth=1, color="#3498db", label="Trial HV")
            ax.plot(range(len(running_max)), running_max, linewidth=2,
                    color="#e74c3c", label="Running Best HV")
            ax.fill_between(range(len(running_max)), running_max,
                            alpha=0.1, color="#e74c3c")
            ax.set_xlabel("Trial Number", fontsize=12)
            ax.set_ylabel("Hypervolume Indicator", fontsize=12)
            ax.set_title("Hypervolume Maximization BO — Convergence", fontsize=13)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(RESULTS / "hv_convergence.png", dpi=200)
            plt.close(fig)
            print(f"[Report] Saved hv_convergence.png")
        except Exception as e:
            print(f"[Report] HV convergence plot skipped: {e}")

    # ── Markdown report ───────────────────────────────────────────────────────
    float_mb = CHECKPOINTS.joinpath("best_model_float.pth")
    quant_mb  = CHECKPOINTS.joinpath("best_model_quantized.pth")

    float_size = f"{float_mb.stat().st_size / (1024**2):.3f} MB" if float_mb.exists() else "N/A"
    quant_size = f"{quant_mb.stat().st_size / (1024**2):.3f} MB" if quant_mb.exists() else "N/A"

    report = RESULTS / "ShuffleNetV2_BO_Report.md"
    report.write_text(
        "\n".join([
            "# ShuffleNetV2 Bayesian Optimization Report",
            "",
            "## Search Space",
            "| Parameter | Type | Range |",
            "|---|---|---|",
            "| width_multiplier | Categorical | [0.5, 1.0, 1.5, 2.0] |",
            "| input_resolution | Categorical | [20, 24, 28, 32] |",
            "| batch_size | Categorical | [4, 8, 16, 32, 64, 128] |",
            "| intra_op_threads | Categorical | [1, 2, 4, 8] |",
            "| dropout | Categorical | [0.0, 0.1, 0.2, 0.3, 0.5] |",
            "| optimizer_name | Categorical | [adam, sgd, rmsprop] |",
            "| scheduler_name | Categorical | [cosine, step, onecycle] |",
            "| learning_rate | Log-Uniform Float | [1e-5, 3e-1] |",
            "| weight_decay | Log-Uniform Float | [1e-6, 1e-2] |",
            "| label_smoothing | Uniform Float | [0.0, 0.20] |",
            "| momentum | Uniform Float | [0.80, 0.99] (SGD/RMSprop) |",
            "",
            "**Total discrete combos (categorical dims only): 34,560+**",
            "BO evaluates ~100-150 trials via TPESampler (MOTPE / HV-max).",
            "",
            "## MOTPE Study Summary",
            f"- Completed trials: {len(completed)}",
            f"- Pareto-optimal trials: {len(pareto)}",
            f"- Best trial: #{best.number}",
            f"- Best accuracy: {best.values[0]:.2f}%",
            f"- Best latency: {best.values[1] * 1000:.2f} ms",
            f"- Best energy: {best.values[2]:.8f} kWh",
            f"- Best params: {best.params}",
            "",
            "## Final PTQ (MOTPE Best)",
            f"- Float model size: {float_size}",
            f"- Quantized INT8 model size: {quant_size}",
            "",
            "## Generated Files",
            "- `final_study_results.csv`      — all trials, all 11 hyperparameters",
            "- `accuracy_vs_latency.png`      — 2D Pareto slice (width coloured)",
            "- `accuracy_vs_energy.png`       — 2D Pareto slice (width coloured)",
            "- `pareto_3d.png`                — 3D Pareto front scatter",
            "- `param_importance.png`         — FAnova hyperparameter importance",
            "- `hv_convergence.png`           — HV convergence (if HV study ran)",
        ])
    )
    print(f"[Report] Markdown report saved to {report}")


if __name__ == "__main__":
    main()
