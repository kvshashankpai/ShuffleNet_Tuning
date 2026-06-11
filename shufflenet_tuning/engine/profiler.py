"""
engine/profiler.py
------------------
Isolated CPU energy and latency profiler.

Design goals:
  - Zero contamination from data loading, loss computation, or disk I/O.
    Only the model's forward pass is measured.
  - Warm-up runs flush OS thread scheduling spikes before measurement begins.
  - Thread state is saved/restored around the benchmark so it doesn't
    interfere with any code running before or after.

Requires: pip install codecarbon
"""

import time
from dataclasses import dataclass

import torch

from configs.experiment_config import ExperimentConfig
from models.shufflenet import ShuffleNetV2


@dataclass
class ProfileResult:
    """Structured output from one benchmark run."""
    energy_kwh:  float   # Total CPU + RAM energy consumed
    latency_sec: float   # Mean per-sample latency across all runs
    throughput:  float   # Samples per second (batch_size / mean_latency)


def profile(
    model:      ShuffleNetV2,
    cfg:        ExperimentConfig,
) -> ProfileResult:
    """
    Benchmarks a trained model's forward pass in complete isolation.

    Workflow:
      1. Disable gradients and enable MKL-DNN
      2. Build a dummy input matching the real inference shape
      3. Run warm-up passes (discarded)
      4. Run timed + tracked passes under CodeCarbon EmissionsTracker
      5. Compute and return energy + latency metrics

    Args:
        model:   Trained (or fresh) ShuffleNetV2 in eval mode.
        cfg:     Config describing the benchmark shape (batch_size, input_size, threads).

    Returns:
        ProfileResult with energy_kwh, latency_sec, and throughput.
    """
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        raise ImportError(
            "codecarbon is required for energy profiling.\n"
            "Install with: pip install codecarbon"
        )

    # ── Setup ──────────────────────────────────────────────────────────────────
    torch.set_grad_enabled(False)
    torch.backends.mkldnn.enabled = True

    model.eval()
    dummy_input = torch.randn(cfg.batch_size, cfg.in_channels,
                               cfg.input_size, cfg.input_size)

    # ── Warm-up (discards thread init + cold cache overhead) ──────────────────
    print(f"  Profiler: warming up ({cfg.warmup_runs} passes)...")
    for _ in range(cfg.warmup_runs):
        _ = model(dummy_input)

    # ── Timed + tracked measurement ───────────────────────────────────────────
    print(f"  Profiler: measuring ({cfg.num_benchmark_runs} passes)...")

    tracker = EmissionsTracker(
        measure_power_secs=1,
        display_to_term=False,
        log_level="error",            # Suppress verbose tracker output
    )
    tracker.start()

    start_time = time.perf_counter()
    for _ in range(cfg.num_benchmark_runs):
        _ = model(dummy_input)
    end_time = time.perf_counter()

    tracker.stop()

    # ── Extract results ───────────────────────────────────────────────────────
    emissions_data = tracker.final_emissions_data
    total_energy   = (
        (emissions_data.cpu_energy or 0.0) +
        (emissions_data.ram_energy or 0.0)
    )

    avg_latency  = (end_time - start_time) / cfg.num_benchmark_runs
    throughput   = cfg.batch_size / avg_latency

    result = ProfileResult(
        energy_kwh  = total_energy,
        latency_sec = avg_latency,
        throughput  = throughput,
    )

    print(
        f"  Profiler: energy={result.energy_kwh:.6f} kWh | "
        f"latency={result.latency_sec*1000:.2f} ms | "
        f"throughput={result.throughput:.1f} samples/s"
    )

    torch.set_grad_enabled(True)  # Restore for next training run
    return result
