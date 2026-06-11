"""
models/blocks.py
----------------
Core building block for ShuffleNetV2.

ShuffleV2Block implements the two-branch depthwise-separable design:
  - stride=1: channel split + right branch transforms + concatenate + shuffle
  - stride=2: both branches downsample + concatenate + shuffle

The channel_shuffle operation is kept as a standalone function so it can
be unit-tested independently of the block.
"""

import torch
import torch.nn as nn


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Reshuffles channels across groups to enable cross-group information flow.

    Args:
        x:      Input tensor of shape (B, C, H, W)
        groups: Number of groups to shuffle across

    Returns:
        Tensor of same shape with channels reordered
    """
    B, C, H, W = x.shape
    assert C % groups == 0, (
        f"Channel count ({C}) must be divisible by groups ({groups})"
    )
    channels_per_group = C // groups
    x = x.view(B, groups, channels_per_group, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, H, W)
    return x


class ShuffleV2Block(nn.Module):
    """
    Single ShuffleNetV2 block.

    stride=1: Channel split. Left half passes through unchanged.
              Right half goes through 1x1 -> DWConv3x3 -> 1x1.
    stride=2: No split. Both branches process the full input with stride=2
              to halve spatial dimensions while doubling channels.

    Args:
        in_channels:  Number of input channels
        out_channels: Number of output channels (must be even)
        stride:       Spatial stride — 1 (same size) or 2 (downsample)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        assert stride in (1, 2), "stride must be 1 or 2"
        self.stride = stride
        branch_channels = out_channels // 2

        if stride == 1:
            # Only the right branch transforms; left passes through unchanged
            self.branch_right = nn.Sequential(
                # Pointwise expand
                nn.Conv2d(branch_channels, branch_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                # Depthwise spatial conv
                nn.Conv2d(branch_channels, branch_channels, 3, 1, 1,
                          groups=branch_channels, bias=False),
                nn.BatchNorm2d(branch_channels),
                # Pointwise project
                nn.Conv2d(branch_channels, branch_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),   # Bug fix: activation was missing here
            )

        else:  # stride == 2
            # Left branch: DWConv stride=2 + pointwise
            self.branch_left = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            # Right branch: pointwise + DWConv stride=2 + pointwise
            self.branch_right = nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_channels, branch_channels, 3, 2, 1,
                          groups=branch_channels, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.Conv2d(branch_channels, branch_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),   # Bug fix: activation was missing here
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            c = x.shape[1] // 2
            x_left, x_right = x[:, :c], x[:, c:]
            out = torch.cat([x_left, self.branch_right(x_right)], dim=1)
        else:
            out = torch.cat([self.branch_left(x), self.branch_right(x)], dim=1)

        return channel_shuffle(out, groups=2)
