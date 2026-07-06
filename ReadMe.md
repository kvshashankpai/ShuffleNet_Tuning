# ShuffleNetV2 Bayesian Hyperparameter Optimization & INT8 Quantization

An advanced experiment pipeline mapping the **Accuracy vs. Latency vs. Energy Pareto front** of ShuffleNetV2 on CPU, using MedMNIST pathology classification, optimized via two Bayesian Optimization modes:

1. **Multi-Objective MOTPE** — Optuna TPESampler with 3 objectives (Accuracy ↑, Latency ↓, Energy ↓)
2. **Hypervolume Maximization** — Single-objective BO maximizing the Hypervolume Indicator over all 3 objectives simultaneously

All training and profiling is **CPU-only**.

---

## Project Structure

```
shufflenet_tuning/
├── configs/
│   ├── experiment_config.py    # Per-run config dataclass (11 hyperparameters)
│   └── base_config.py          # Full 34,560+ combo search grid + helpers
│
├── models/
│   ├── blocks.py               # ShuffleV2Block (channel shuffle + branch logic)
│   └── shufflenet.py           # ShuffleNetV2 & QuantizableShuffleNetV2 full models
│
├── engine/
│   ├── trainer.py              # CPU-only training loop (Adam/SGD/RMSprop + 3 schedulers)
│   ├── evaluator.py            # Accuracy evaluation on test split
│   └── profiler.py             # Isolated energy + latency benchmarking (CPU)
│
├── experiments/
│   ├── optuna_optimize.py      # Multi-Objective MOTPE BO + INT8 PTQ
│   ├── hypervolume_optimize.py # Single-Objective Hypervolume Maximization BO + PTQ
│   ├── generate_report.py      # CSV/plot/Markdown report generator
│   ├── train_phase1.py         # Legacy Phase 1 training script
│   └── profile_phase2.py       # Legacy CPU energy profiling script
│
└── main.py                     # Single entrypoint — runs MOTPE optimization by default
```

---

## Hyperparameters & Search Space

| Parameter | Type | Range / Choices | Description |
|---|---|---|---|
| `width_multiplier` | Categorical | `[0.5, 1.0, 1.5, 2.0]` | ShuffleNetV2 channel scaling |
| `input_resolution` | Categorical | `[20, 24, 28, 32]` | Spatial dimension (FLOPs scale quadratically) |
| `batch_size` | Categorical | `[4, 8, 16, 32, 64, 128]` | Mini-batch size |
| `intra_op_threads` | Categorical | `[1, 2, 4, 8]` | CPU thread pool size |
| `dropout` | Categorical | `[0.0, 0.1, 0.2, 0.3, 0.5]` | Classifier head dropout |
| `optimizer_name` | Categorical | `[adam, sgd, rmsprop]` | Gradient optimizer |
| `scheduler_name` | Categorical | `[cosine, step, onecycle]` | LR annealing strategy |
| `learning_rate` | Log-Uniform Float | `[1e-5, 3e-1]` | Initial learning rate |
| `weight_decay` | Log-Uniform Float | `[1e-6, 1e-2]` | L2 regularisation |
| `label_smoothing` | Uniform Float | `[0.0, 0.20]` | Classifier label smoothing ε |
| `momentum` | Uniform Float | `[0.80, 0.99]` | Momentum (SGD / RMSprop only) |

**Total discrete combos (categorical dims): 4 × 4 × 6 × 4 × 5 × 3 × 3 = 34,560+**  
BO explores this intelligently with only **100–150 trials** using TPESampler.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision medmnist codecarbon optuna matplotlib
```

### 2a. Multi-Objective MOTPE (3 objectives, Pareto front)

Optimizes for 3 objectives simultaneously: **[Maximize Accuracy, Minimize Latency, Minimize Energy]**.
Returns a Pareto-optimal set of configurations.

```bash
# Default: 150 trials, 2 training epochs per trial
python main.py

# Custom
python main.py --trials 200 --optuna-epochs 3 --final-epochs 10
```

### 2b. Hypervolume Maximization (single-objective scalar BO)

Collapses all 3 objectives into a single **Hypervolume Indicator** and maximizes it.
Uses the same TPESampler (BO solver). Useful when you want a single best config
rather than a Pareto front.

```bash
python main.py --hypervolume --trials 150 --optuna-epochs 2
```

### 3. Generate Report & Plots

```bash
cd shufflenet_tuning
python experiments/generate_report.py
```

Outputs to `results/`:
- `final_study_results.csv` — all trials with all 11 hyperparameters
- `accuracy_vs_latency.png` — 2D Pareto slice
- `accuracy_vs_energy.png` — 2D Pareto slice
- `pareto_3d.png` — full 3D Pareto front
- `param_importance.png` — FAnova hyperparameter importance
- `hv_convergence.png` — HV convergence plot (if `--hypervolume` was run)
- `ShuffleNetV2_BO_Report.md` — text summary

---

## Execution Workflow

### MOTPE Mode (default)
1. **Bayesian Optimization** (MOTPE): TPESampler suggests configs from 34,560+ combo space
2. **Pareto Frontier Estimation**: Optuna identifies the non-dominated set
3. **Model Selection**: Best config by accuracy on the Pareto front
4. **INT8 PTQ**: Conv+BN+ReLU fusion → observer calibration → INT8 conversion
5. **Checkpoints**: `best_model_float.pth` and `best_model_quantized.pth`

### Hypervolume Maximization Mode
1. **Single-Objective BO**: TPESampler maximizes HV(accuracy, latency, energy) as a scalar
2. **HV Tracking**: Running hypervolume computed after each trial using WFG algorithm
3. **Convergence Plot**: HV vs. trial number saved to `results/hv_convergence.png`
4. **Model Selection**: Best config by individual accuracy
5. **INT8 PTQ**: Same PTQ pipeline as MOTPE mode

---

## Choosing Between Modes

| | MOTPE | Hypervolume Maximization |
|---|---|---|
| **Objectives** | 3 separate | 1 scalar (HV) |
| **Output** | Pareto front (many configs) | Single best config |
| **Interpretability** | Trade-off curves, Pareto plots | Convergence plot, single winner |
| **When to use** | Exploring the full trade-off space | Deploying one optimal model |
| **BO solver** | TPESampler (multi-obj mode) | TPESampler (single-obj mode) |

Both use the **same 10-parameter search space** and the **same TPESampler** BO algorithm.
