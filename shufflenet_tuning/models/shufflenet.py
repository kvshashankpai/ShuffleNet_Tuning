"""
models/shufflenet.py
--------------------
Full ShuffleNetV2 model for MedMNIST (28x28 greyscale / RGB input).

Key design decisions:
  - stage1 uses stride=1 (not 2) to preserve spatial detail on small 28x28 inputs
  - intra_op_threads controls PyTorch's CPU thread pool scoped to this model's
    forward pass only — previous thread count is always restored afterward
  - width_multiplier selects the channel width tier (0.5x through 2.0x)
"""

import torch
import torch.nn as nn

from models.blocks import ShuffleV2Block


class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2 adapted for MedMNIST CPU inference benchmarking.

    Args:
        width_multiplier:  Channel width scaling factor. One of {0.5, 1.0, 1.5, 2.0}.
        num_classes:       Number of output classes (9 for PathMNIST).
        in_channels:       Input image channels (3 for RGB MedMNIST).
        intra_op_threads:  CPU thread count for this model's forward pass.
                           0 = use PyTorch default (no override).
    """

    # Maps width multiplier → [stage1_ch, stage2_ch, stage3_ch, stage4_ch, conv5_ch]
    STAGE_CHANNELS: dict[float, list[int]] = {
        0.5: [24,  48,  96,  192, 1024],
        1.0: [24, 116, 232, 464, 1024],
        1.5: [24, 176, 352, 704, 1024],
        2.0: [24, 244, 488, 976, 2048],
    }
    STAGE_REPEATS: list[int] = [4, 8, 4]

    def __init__(
        self,
        width_multiplier: float = 1.0,
        num_classes: int = 9,
        in_channels: int = 3,
        intra_op_threads: int = 0,
    ):
        super().__init__()

        if width_multiplier not in self.STAGE_CHANNELS:
            raise ValueError(
                f"width_multiplier must be one of {list(self.STAGE_CHANNELS.keys())}, "
                f"got {width_multiplier}"
            )

        self.width_multiplier = width_multiplier
        self.intra_op_threads = intra_op_threads
        channels = self.STAGE_CHANNELS[width_multiplier]

        # ── Stage 1: initial stem conv (stride=1 to keep spatial info on 28x28) ──
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # ── Stages 2–4: stacked ShuffleV2 blocks ─────────────────────────────────
        self.stage2 = self._make_stage(channels[0], channels[1], self.STAGE_REPEATS[0])
        self.stage3 = self._make_stage(channels[1], channels[2], self.STAGE_REPEATS[1])
        self.stage4 = self._make_stage(channels[2], channels[3], self.STAGE_REPEATS[2])

        # ── Conv5: pointwise projection to final feature dim ──────────────────────
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
        )

        # ── Head ──────────────────────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[4], num_classes)

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────────

    def _make_stage(self, in_ch: int, out_ch: int, repeats: int) -> nn.Sequential:
        """First block downsamples (stride=2), remaining blocks keep size (stride=1)."""
        layers: list[nn.Module] = [ShuffleV2Block(in_ch, out_ch, stride=2)]
        for _ in range(repeats - 1):
            layers.append(ShuffleV2Block(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scoped thread override: set → run → restore
        # This ensures one model config doesn't pollute the thread state
        # for other models running in the same process.
        _prev_threads: int | None = None
        if self.intra_op_threads > 0:
            _prev_threads = torch.get_num_threads()
            torch.set_num_threads(self.intra_op_threads)

        try:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            x = self.global_pool(x).flatten(1)
            x = self.classifier(x)
        finally:
            # Always restore — even if an exception is raised mid-forward
            if _prev_threads is not None:
                torch.set_num_threads(_prev_threads)

        return x

    # ─────────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ShuffleNetV2("
            f"width={self.width_multiplier}x, "
            f"threads={self.intra_op_threads}, "
            f"params={self.count_parameters():,})"
        )
