# ShuffleNetV2 Hyperparameter Tuning — MedMNIST CPU Efficiency Study

A clean, structured experiment pipeline for mapping the **Accuracy vs. Energy Pareto front** of ShuffleNetV2 on CPU, using MedMNIST pathology classification.

---

## Project Structure

```
shufflenet_tuning/
├── configs/
│   ├── base_config.py          # Central hyperparameter grid definition
│   └── experiment_config.py    # Per-run config dataclass
│
├── models/
│   ├── blocks.py               # ShuffleV2Block (channel shuffle + branch logic)
│   └── shufflenet.py           # ShuffleNetV2 full model definition
│
├── engine/
│   ├── trainer.py              # Training loop (loss, optimizer, scheduler)
│   ├── evaluator.py            # Accuracy evaluation on test split
│   └── profiler.py             # Isolated energy + latency benchmarking
│
├── experiments/
│   └── grid_search.py          # Outer loop: runs all (width × threads × batch) combos
│
├── results/
│   └── experiment_log.csv      # Auto-generated flat results log (gitignored raw data)
│
├── scripts/
│   └── check_hardware.sh       # Pre-flight CPU vendor + core count check
│
└── main.py                     # Single entrypoint — run everything from here
```

---

## Hyperparameters Under Study

| Parameter          | Values Tested      | Energy Impact | Accuracy Impact |
|--------------------|--------------------|---------------|-----------------|
| `width_multiplier` | 0.5, 1.0, 1.5, 2.0 | Extreme       | Extreme         |
| `intra_op_threads` | 1, 2, 4            | High          | Zero            |
| `batch_size`       | 16, 32, 64         | Moderate–High | Minimal         |
| `input_size`       | 24, 28             | High          | Moderate        |

Total grid: **72 unique configurations** (4 × 3 × 3 × 2)

---

## Quick Start

```bash
# 1. Check your hardware first
bash scripts/check_hardware.sh

# 2. Install dependencies
pip install torch torchvision medmnist codecarbon

# 3. Run the full grid search
python main.py

# 4. Results land in results/experiment_log.csv
```

---

## Output Format

Each run appends one row to `results/experiment_log.csv`:

| config_id       | width_multiplier | threads | batch_size | input_size | accuracy | energy_kwh | latency_sec | params |
|-----------------|-----------------|---------|------------|------------|----------|------------|-------------|--------|
| w0.5_t4_b32_r28 | 0.5             | 4       | 32         | 28         | 87.3     | 0.000021   | 0.0043      | 209K   |
