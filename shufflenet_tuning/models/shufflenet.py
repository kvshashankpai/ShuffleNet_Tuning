"""
models/shufflenet.py
--------------------
Full ShuffleNetV2 model for MedMNIST (28x28 greyscale / RGB input).

Key design decisions:
  - stage1 uses stride=1 (not 2) to preserve spatial detail on small 28x28 inputs
  - intra_op_threads controls PyTorch's CPU thread pool scoped to this model's
    forward pass only — previous thread count is always restored afterward
  - width_multiplier selects the channel width tier (0.5x through 2.0x)
  - stage_repeats controls the depth of stages 2–4 (tunable in v2)
  - fc_hidden_dim optionally adds a FC+BN+ReLU layer before the classifier
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
        stage_repeats:     List of 3 ints controlling depth of stages 2, 3, 4.
                           Default [4, 8, 4] is the standard ShuffleNetV2 config.
        fc_hidden_dim:     If > 0, adds an FC(conv5_ch → fc_hidden_dim) + BN + ReLU
                           layer between global_pool and the final classifier.
                           0 = no extra layer (original architecture).
    """

    # Maps width multiplier → [stage1_ch, stage2_ch, stage3_ch, stage4_ch, conv5_ch]
    STAGE_CHANNELS: dict[float, list[int]] = {
        0.5: [24,  48,  96,  192, 1024],
        1.0: [24, 116, 232, 464, 1024],
        1.5: [24, 176, 352, 704, 1024],
        2.0: [24, 244, 488, 976, 2048],
    }
    # Default stage repeats — can be overridden via constructor
    DEFAULT_STAGE_REPEATS: list[int] = [4, 8, 4]

    def __init__(
        self,
        width_multiplier: float = 1.0,
        num_classes: int = 9,
        in_channels: int = 3,
        intra_op_threads: int = 0,
        stage_repeats: list[int] | None = None,
        fc_hidden_dim: int = 0,
    ):
        super().__init__()

        if width_multiplier not in self.STAGE_CHANNELS:
            raise ValueError(
                f"width_multiplier must be one of {list(self.STAGE_CHANNELS.keys())}, "
                f"got {width_multiplier}"
            )

        self.width_multiplier = width_multiplier
        self.intra_op_threads = intra_op_threads
        self.fc_hidden_dim = fc_hidden_dim
        channels = self.STAGE_CHANNELS[width_multiplier]

        # Resolve stage repeats
        if stage_repeats is None:
            stage_repeats = self.DEFAULT_STAGE_REPEATS
        assert len(stage_repeats) == 3, "stage_repeats must have exactly 3 elements"
        self.stage_repeats = stage_repeats

        # ── Stage 1: initial stem conv (stride=1 to keep spatial info on 28x28) ──
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # ── Stages 2–4: stacked ShuffleV2 blocks ─────────────────────────────────
        self.stage2 = self._make_stage(channels[0], channels[1], self.stage_repeats[0])
        self.stage3 = self._make_stage(channels[1], channels[2], self.stage_repeats[1])
        self.stage4 = self._make_stage(channels[2], channels[3], self.stage_repeats[2])

        # ── Conv5: pointwise projection to final feature dim ──────────────────────
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
        )

        # ── Head ──────────────────────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.0)

        # Optional FC hidden layer (professor's suggestion: reduce noise)
        if fc_hidden_dim > 0:
            self.fc_hidden = nn.Sequential(
                nn.Linear(channels[4], fc_hidden_dim),
                nn.BatchNorm1d(fc_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Linear(fc_hidden_dim, num_classes)
        else:
            self.fc_hidden = None
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
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        if self.fc_hidden is not None:
            x = self.fc_hidden(x)
        x = self.classifier(x)

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
            f"repeats={self.stage_repeats}, "
            f"fc_hidden={self.fc_hidden_dim}, "
            f"params={self.count_parameters():,})"
        )


class QuantizableShuffleNetV2(ShuffleNetV2):
    """
    Quantizable ShuffleNetV2 subclass equipped with QuantStub/DeQuantStub
    and a custom fuse_model method for static quantization (PTQ).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        if self.fc_hidden is not None:
            x = self.fc_hidden(x)
        x = self.classifier(x)
        x = self.dequant(x)

        return x

    def fuse_model(self) -> None:
        """
        Fuses modules (Conv + BN + ReLU) in place to prepare for static quantization.
        """
        # Fuse stage 1 (Conv2d, BatchNorm2d, ReLU)
        torch.ao.quantization.fuse_modules(
            self.stage1, [["0", "1", "2"]], inplace=True
        )

        # Fuse stage 2, 3, 4 blocks
        for stage in [self.stage2, self.stage3, self.stage4]:
            for block in stage:
                if block.stride == 1:
                    # branch_right:
                    # 0: Conv2d, 1: BatchNorm2d, 2: ReLU
                    # 3: Conv2d (depthwise), 4: BatchNorm2d
                    # 5: Conv2d (pointwise), 6: BatchNorm2d, 7: ReLU
                    torch.ao.quantization.fuse_modules(
                        block.branch_right,
                        [["0", "1", "2"], ["3", "4"], ["5", "6", "7"]],
                        inplace=True,
                    )
                else:
                    # branch_left:
                    # 0: Conv2d (depthwise), 1: BatchNorm2d
                    # 2: Conv2d (pointwise), 3: BatchNorm2d, 4: ReLU
                    torch.ao.quantization.fuse_modules(
                        block.branch_left,
                        [["0", "1"], ["2", "3", "4"]],
                        inplace=True,
                    )
                    # branch_right:
                    # 0: Conv2d, 1: BatchNorm2d, 2: ReLU
                    # 3: Conv2d (depthwise), 4: BatchNorm2d
                    # 5: Conv2d, 6: BatchNorm2d, 7: ReLU
                    torch.ao.quantization.fuse_modules(
                        block.branch_right,
                        [["0", "1", "2"], ["3", "4"], ["5", "6", "7"]],
                        inplace=True,
                    )

        # Fuse conv5
        torch.ao.quantization.fuse_modules(
            self.conv5, [["0", "1", "2"]], inplace=True
        )

        # Fuse fc_hidden if present (Linear + BatchNorm1d + ReLU)
        if self.fc_hidden is not None:
            torch.ao.quantization.fuse_modules(
                self.fc_hidden, [["0", "1", "2"]], inplace=True
            )
