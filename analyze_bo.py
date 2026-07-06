import pandas as pd
import numpy as np

# Load the Bayesian Optimization results
csv_path = '/home/user2/NLP/BO_opti/shufflenet_tuning/results/final_study_results.csv'
df = pd.read_csv(csv_path)

# Summary statistics
print("=== BASIC STATISTICS ===")
print(df[['accuracy', 'latency_sec', 'energy_kwh']].describe())

# Find Pareto frontier function for 3 objectives (Max Accuracy, Min Latency, Min Energy)
def find_pareto_3d(df):
    pareto_indices = []
    for idx, row in df.iterrows():
        acc = row['accuracy']
        lat = row['latency_sec']
        eng = row['energy_kwh']
        
        dominated = False
        for idx2, row2 in df.iterrows():
            if idx == idx2:
                continue
            acc2 = row2['accuracy']
            lat2 = row2['latency_sec']
            eng2 = row2['energy_kwh']
            
            # Row2 dominates Row if:
            # - Row2 is at least as good in all objectives
            # - Row2 is strictly better in at least one objective
            cond_acc = acc2 >= acc
            cond_lat = lat2 <= lat
            cond_eng = eng2 <= eng
            
            strict_acc = acc2 > acc
            strict_lat = lat2 < lat
            strict_eng = eng2 < eng
            
            if (cond_acc and cond_lat and cond_eng) and (strict_acc or strict_lat or strict_eng):
                dominated = True
                break
        
        if not dominated:
            pareto_indices.append(idx)
            
    return df.loc[pareto_indices]

pareto_3d_df = find_pareto_3d(df)
print("\n=== PARETO OPTIMAL TRIALS (3D: Accuracy, Latency, Energy) ===")
print(f"Number of Pareto-optimal configurations: {len(pareto_3d_df)}")
print(pareto_3d_df.sort_values(by='accuracy', ascending=False)[
    ['trial', 'accuracy', 'latency_sec', 'energy_kwh', 'width_multiplier', 'input_resolution', 'batch_size', 'intra_op_threads', 'dropout']
].to_string(index=False))

# Calculate parameter correlations
print("\n=== CORRELATIONS WITH OBJECTIVES ===")
corr = df[['width_multiplier', 'input_resolution', 'batch_size', 'intra_op_threads', 'dropout', 'learning_rate', 'weight_decay', 'accuracy', 'latency_sec', 'energy_kwh']].corr()
print(corr[['accuracy', 'latency_sec', 'energy_kwh']].loc[
    ['width_multiplier', 'input_resolution', 'batch_size', 'intra_op_threads', 'dropout', 'learning_rate', 'weight_decay']
])

# Let's inspect the best configuration
print("\n=== BEST CONFIGURATION DETAILS (TRIAL 103) ===")
best_trial = df[df['trial'] == 103]
if not best_trial.empty:
    print(best_trial.to_string(index=False))
else:
    print("Trial 103 not found!")

# Average metrics across different width multipliers
print("\n=== AVERAGE METRICS BY WIDTH MULTIPLIER ===")
print(df.groupby('width_multiplier')[['accuracy', 'latency_sec', 'energy_kwh']].mean())

# Average metrics across thread counts
print("\n=== AVERAGE METRICS BY INTRA-OP THREADS ===")
print(df.groupby('intra_op_threads')[['accuracy', 'latency_sec', 'energy_kwh']].mean())

# Average metrics across resolution
print("\n=== AVERAGE METRICS BY INPUT RESOLUTION ===")
print(df.groupby('input_resolution')[['accuracy', 'latency_sec', 'energy_kwh']].mean())
