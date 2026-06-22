# ShuffleNetV2 Bayesian Hyperparameter Optimization & INT8 Quantization

An advanced experiment pipeline mapping the **Accuracy vs. Latency vs. Energy Pareto front** of ShuffleNetV2 on CPU, using MedMNIST pathology classification, optimized via Bayesian Optimization, and compressed via static Post-Training Quantization (PTQ).

---

## Project Structure

```
shufflenet_tuning/
├── configs/
│   └── experiment_config.py    # Per-run config dataclass
│
├── models/
│   ├── blocks.py               # ShuffleV2Block (channel shuffle + branch logic)
│   └── shufflenet.py           # ShuffleNetV2 & QuantizableShuffleNetV2 full models
│
├── engine/
│   ├── trainer.py              # Training loop using QuantizableShuffleNetV2
│   ├── evaluator.py            # Accuracy evaluation on test split
│   └── profiler.py             # Isolated energy + latency benchmarking
│
├── experiments/
│   ├── train_phase1.py         # Legacy Phase 1 training script
│   ├── profile_phase2.py       # Legacy CPU energy profiling script
│   └── optuna_optimize.py      # Task 1 & 2: Optuna MOTPE study + static INT8 PTQ
│
└── main.py                     # Single entrypoint — runs Optuna optimization by default
```

---

## Hyperparameters & Search Space

We transition from a narrow brute-force grid search to an expanded continuous/discrete search space optimized dynamically using Bayesian Optimization:

| Parameter | Type / Range | Description |
|---|---|---|
| `width_multiplier` | Categorical `[0.5, 1.0, 1.5, 2.0]` | Primary model capacity axis |
| `intra_op_threads` | Categorical `[1, 2, 4]` | Thread pool count for forward pass benchmarks |
| `batch_size` | Categorical `[8, 16, 32, 64]` | Cache locality lever for memory usage |
| `input_size` (Resolution) | Categorical `[24, 28]` | Spatial dimension scaling for input images |
| `learning_rate` | Log-Uniform Float `[1e-4, 1e-1]` | Initial Adam optimizer learning rate |
| `weight_decay` | Log-Uniform Float `[1e-5, 1e-2]` | L2 regularization coefficient |

---

## Quick Start

### 1. Install Dependencies
Ensure you have all the required Python packages installed:
```bash
pip install torch torchvision medmnist codecarbon optuna
```

### 2. Run the Optuna Optimization and PTQ Pipeline
Run the multi-objective optimization study using Optuna with the MOTPE sampler. The study optimizes for three objectives: **[Maximize Accuracy, Minimize Latency, Minimize Energy]**.

```bash
# Run the pipeline with custom trials and training epochs per trial
python main.py --trials 150 --optuna-epochs 10 --device cuda
```

### 3. Execution Workflow
1. **Multi-Objective Bayesian Optimization**: The sampler intelligently queries the search space. In each trial, the model is trained with the suggested parameters, and its validation accuracy, latency (seconds), and energy consumption (kWh) are recorded.
2. **Pareto Frontier Estimation**: Optuna identifies the non-dominated Pareto front.
3. **Model Selection**: The pipeline automatically selects the best configuration from the Pareto front (prioritizing the highest validation accuracy).
4. **Post-Training Quantization (PTQ)**:
   - Module fusion fusions `Conv2d + BatchNorm2d + ReLU` layers to maintain accuracy.
   - Standard static x86 CPU quantization observers are calibrated using 10 batches.
   - The model is converted to INT8 format.
   - The before-and-after model size footprint in MB is printed.
   - The final weights are saved to `checkpoints/best_model_float.pth` and `checkpoints/best_model_quantized.pth`.
