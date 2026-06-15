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
│   ├── trainer.py              # Training loop (loss, optimizer, scheduler) [MODIFIED]
│   ├── evaluator.py            # Accuracy evaluation on test split
│   └── profiler.py             # Isolated energy + latency benchmarking [MODIFIED]
│
├── experiments/
│   ├── train_phase1.py         # Phase 1: GPU-based accuracy training [NEW]
│   └── profile_phase2.py       # Phase 2: CPU-only energy profiling [NEW]
│
├── results/
│   └── experiment_log.csv      # Auto-generated flat results log (gitignored raw data)
│
├── scripts/
│   └── check_hardware.sh       # Pre-flight CPU vendor + core count check
│
└── main.py                     # Single entrypoint — run everything from here [MODIFIED]
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

## Quick Start (Two-Phase Strategy)

To bypass the CPU training bottleneck, we decouple accuracy from energy profiling:

### 1. Install Dependencies
```bash
pip install torch torchvision medmnist codecarbon
```

### 2. Phase 1: The Accuracy Run (Run on GPU)
Train only the **8 unique configurations** that affect accuracy on a GPU (e.g. Google Colab, Kaggle, or local GPU):
```bash
python main.py --phase1 --epochs 10 --train-batch-size 64
```
* **Output**: Trained weights saved to `checkpoints/` and accuracy registry created in `checkpoints/accuracy_registry.json`.

### 3. Phase 2: The Energy Profiling Run (Run on CPU)
Transfer the `checkpoints/` folder to your target local Intel CPU and run the offline energy profiling for all **72 configurations** in under 2 minutes:
```bash
python main.py --phase2
```
* **Output**: Combined accuracy and energy/latency metrics logged to `results/experiment_log.csv`.


---

## Output Format

Each run appends one row to `results/experiment_log.csv`:

| config_id       | width_multiplier | threads | batch_size | input_size | accuracy | energy_kwh | latency_sec | params |
|-----------------|-----------------|---------|------------|------------|----------|------------|-------------|--------|
| w0.5_t4_b32_r28 | 0.5             | 4       | 32         | 28         | 87.3     | 0.000021   | 0.0043      | 209K   |
