import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Create output directory
output_dir = "output_graphs"
os.makedirs(output_dir, exist_ok=True)

# Load data
csv_path = "results/final_study_results.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Set style
sns.set_theme(style="whitegrid")

def plot_pareto_front(df, x_col, y_col, minimize_x=True, maximize_y=True, filename="pareto.png", title="Pareto Front"):
    plt.figure(figsize=(10, 6))
    
    # Scatter all points
    sns.scatterplot(data=df, x=x_col, y=y_col, color="lightgray", label="All Trials", s=50, alpha=0.6)
    
    # Find Pareto optimal points
    # Sort by x first
    sorted_df = df.sort_values(by=x_col, ascending=minimize_x).reset_index(drop=True)
    
    pareto_pts_x = []
    pareto_pts_y = []
    
    current_best_y = -np.inf if maximize_y else np.inf
    
    for _, row in sorted_df.iterrows():
        y_val = row[y_col]
        
        # If we are maximizing Y, we want the current Y to be strictly greater than the best Y seen so far
        is_pareto = False
        if maximize_y and y_val > current_best_y:
            is_pareto = True
            current_best_y = y_val
        elif not maximize_y and y_val < current_best_y:
            is_pareto = True
            current_best_y = y_val
            
        if is_pareto:
            pareto_pts_x.append(row[x_col])
            pareto_pts_y.append(row[y_col])
            
    # Plot Pareto points and the "hyperbolic" trade-off curve
    plt.plot(pareto_pts_x, pareto_pts_y, color="red", marker="o", linestyle="-", linewidth=2, markersize=8, label="Pareto Front")
    
    plt.xscale('log')
    
    plt.title(title)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 1. Pareto Front: Accuracy vs Latency (Hyperbolic trade-off curve)
plot_pareto_front(df, x_col="latency_sec", y_col="accuracy", minimize_x=True, maximize_y=True, 
                  filename="hyperbolic_pareto_latency.png", title="Accuracy vs Latency Trade-off (Pareto Front)")

# 2. Pareto Front: Accuracy vs Energy (Hyperbolic trade-off curve)
plot_pareto_front(df, x_col="energy_kwh", y_col="accuracy", minimize_x=True, maximize_y=True, 
                  filename="hyperbolic_pareto_energy.png", title="Accuracy vs Energy Trade-off (Pareto Front)")

# 3. Standard scatter plot colored by width_multiplier
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="latency_sec", y="accuracy", hue="width_multiplier", palette="viridis", s=100, alpha=0.7)
plt.title("Accuracy vs Latency by Width Multiplier")
plt.xlabel("Latency (seconds)")
plt.ylabel("Accuracy (%)")
plt.savefig(os.path.join(output_dir, "accuracy_vs_latency_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully generated and saved plots to {os.path.abspath(output_dir)}")
