"""
configs/experiment_config.py
-----------------------------
Typed configuration for a single experiment run.

Using a dataclass (rather than a plain dict) gives you:
  - IDE autocomplete and type checking
  - A clean __repr__ for logging
  - Easy serialisation to CSV via dataclasses.asdict()
"""

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """
    All hyperparameters that define one training + evaluation run.

    Attributes:
        width_multiplier:  ShuffleNetV2 channel scaling (0.5 / 1.0 / 1.5 / 2.0).
        intra_op_threads:  CPU thread count during model forward pass.
        batch_size:        Mini-batch size for training and evaluation.
        input_size:        Spatial dimension to resize MedMNIST images to.
        dropout:           Dropout probability before the classifier head.

        optimizer_name:    Optimizer choice: "adam" | "sgd".
        scheduler_name:    LR scheduler: "cosine" | "step" | "onecycle".
        label_smoothing:   Label smoothing epsilon for CrossEntropyLoss [0, 0.2].
        momentum:          Momentum for SGD (ignored for Adam).

        loss_name:         Loss function: "cross_entropy" | "kl_divergence" | "focal".
        fc_hidden_dim:     Hidden dim for optional FC layer before classifier (0 = disabled).
        stage_depth:       Network depth preset: "shallow" [2,4,2] | "standard" [4,8,4] | "deep" [6,12,6].

        num_classes:       Fixed to 9 for PathMNIST — change for other MedMNIST tasks.
        in_channels:       3 for RGB, 1 for greyscale.
        num_epochs:        Training epochs per config.
        learning_rate:     Initial LR for the chosen optimizer.
        weight_decay:      L2 regularisation strength.

        num_benchmark_runs:  Forward passes to average over during energy profiling.
        warmup_runs:         Discarded warm-up passes (flushes thread init spikes).
    """

    # ── Primary tuning knobs ────────────────────────────────────────────────────
    width_multiplier: float = 1.0
    intra_op_threads: int   = 4
    batch_size:       int   = 32
    input_size:       int   = 28
    dropout:        float   = 0.0

    # ── New tuning knobs (expanded search space) ────────────────────────────────
    optimizer_name:   str   = "adam"    # "adam" | "sgd"
    scheduler_name:   str   = "cosine"  # "cosine" | "step" | "onecycle"
    label_smoothing: float  = 0.0       # [0.0, 0.2] — classifier regularisation
    momentum:        float  = 0.9       # relevant for sgd

    # ── v2 tuning knobs ─────────────────────────────────────────────────────────
    loss_name:       str   = "cross_entropy"  # "cross_entropy" | "kl_divergence" | "focal"
    fc_hidden_dim:   int   = 0               # 0 = no extra FC layer; 128, 256, 512
    stage_depth:     str   = "standard"      # "shallow" [2,4,2] | "standard" [4,8,4] | "deep" [6,12,6]

    # ── Fixed model settings ────────────────────────────────────────────────────
    num_classes: int   = 9
    in_channels: int   = 3

    # ── Training settings ───────────────────────────────────────────────────────
    num_epochs:    int   = 10
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4

    # ── Benchmark settings ──────────────────────────────────────────────────────
    num_benchmark_runs: int = 100
    warmup_runs:        int = 10

    # ── Auto-generated fields (set post-init) ───────────────────────────────────
    config_id: str = field(init=False)

    # ── Stage depth presets ───────────────────────────────────────────────────
    STAGE_DEPTH_MAP: dict = field(default_factory=lambda: {
        "shallow":  [2, 4, 2],
        "standard": [4, 8, 4],
        "deep":     [6, 12, 6],
    }, repr=False)

    def __post_init__(self) -> None:
        self.config_id = (
            f"w{self.width_multiplier}_"
            f"t{self.intra_op_threads}_"
            f"b{self.batch_size}_"
            f"r{self.input_size}_"
            f"d{self.dropout}_"
            f"{self.optimizer_name}_"
            f"{self.scheduler_name}_"
            f"{self.loss_name}_"
            f"fc{self.fc_hidden_dim}_"
            f"{self.stage_depth}"
        )

    @property
    def resolved_stage_repeats(self) -> list[int]:
        """Returns the [repeats_s2, repeats_s3, repeats_s4] list for the chosen depth."""
        return self.STAGE_DEPTH_MAP[self.stage_depth]

    def __str__(self) -> str:
        return (
            f"[{self.config_id}] "
            f"width={self.width_multiplier}x | "
            f"threads={self.intra_op_threads} | "
            f"batch={self.batch_size} | "
            f"res={self.input_size}x{self.input_size} | "
            f"dropout={self.dropout:.2f} | "
            f"opt={self.optimizer_name} | "
            f"sched={self.scheduler_name} | "
            f"ls={self.label_smoothing:.3f} | "
            f"loss={self.loss_name} | "
            f"fc={self.fc_hidden_dim} | "
            f"depth={self.stage_depth}"
        )
