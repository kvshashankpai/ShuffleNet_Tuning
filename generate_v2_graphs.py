import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def main():
    csv_path = Path(__file__).resolve().parent / "shufflenet_tuning" / "results_v2" / "v2_study_results.csv"
    output_dir = Path(__file__).resolve().parent / "output_graphs_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Results CSV not found at {csv_path}. Have you run the v2 training yet?")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values(by="trial")
    print(f"Loaded {len(df)} trials from CSV.")

    # 1. Optimization History for Accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(df["trial"], df["accuracy"], color="blue", alpha=0.6, label="Trial Accuracy")
    # Running maximum
    running_max = np.maximum.accumulate(df["accuracy"])
    plt.plot(df["trial"], running_max, color="red", label="Best Accuracy")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy (%)")
    plt.title("Optimization History (Accuracy)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_history_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved optimization_history_accuracy.png")

    # 2. Pareto Fronts (Pairwise 2D projections)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Latency vs Accuracy
    axes[0].scatter(df["latency_sec"] * 1000, df["accuracy"], color="purple", alpha=0.7)
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Latency vs Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Energy vs Accuracy
    axes[1].scatter(df["energy_kwh"] * 1000, df["accuracy"], color="orange", alpha=0.7)
    axes[1].set_xlabel("Energy (Wh)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Energy vs Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Latency vs Energy
    axes[2].scatter(df["latency_sec"] * 1000, df["energy_kwh"] * 1000, color="green", alpha=0.7)
    axes[2].set_xlabel("Latency (ms)")
    axes[2].set_ylabel("Energy (Wh)")
    axes[2].set_title("Latency vs Energy")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "pareto_fronts_2d.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pareto_fronts_2d.png")

    # 3. Hyperparameter importances (proxy: correlation with accuracy)
    # We will just do a simple correlation bar chart for numeric parameters
    numeric_cols = ["width_multiplier", "input_resolution", "batch_size", "intra_op_threads", 
                    "dropout", "learning_rate", "weight_decay"]
    corrs = df[numeric_cols].corrwith(df["accuracy"]).fillna(0).sort_values()
    
    plt.figure(figsize=(10, 6))
    corrs.plot(kind="barh", color="teal")
    plt.xlabel("Pearson Correlation with Accuracy")
    plt.title("Parameter Correlation with Accuracy (Proxy for Importance)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "param_importances_proxy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved param_importances_proxy.png")

    print(f"All generated graphs are in {output_dir}")

if __name__ == "__main__":
    main()
