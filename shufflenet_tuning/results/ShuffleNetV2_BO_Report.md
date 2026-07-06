# ShuffleNetV2 Bayesian Optimization Report

## Summary
- Completed trials: 115
- Pareto-optimal trials: 20
- Best completed trial: 103
- Best accuracy: 83.94%
- Best latency: 21.10 ms
- Best energy: 0.00055718 kWh
- Best params: {'width_multiplier': 0.5, 'input_resolution': 28, 'batch_size': 8, 'intra_op_threads': 4, 'dropout': 0.5, 'learning_rate': 0.0043317535771657326, 'weight_decay': 1.228889068344754e-05}

## Final PTQ
- Float model size: 1.477 MB
- Quantized model size: 0.511 MB

## Files
- `final_study_results.csv`
- `accuracy_vs_latency.png`
- `accuracy_vs_energy.png`
- `pareto_3d.png`