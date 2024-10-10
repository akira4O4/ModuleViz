import torch
import torch.nn as nn


def autopad(kernel_size, padding=None, dilation=1):
    """Pad to 'same' shape outputs."""
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [dilation * (x - 1) + 1 for
                                                                                             x in
                                                                                             kernel_size]  # actual kernel-size
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]  # auto-pad
    return padding


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        groups=1,
        dilation=1
    ):
        super().__init__()
        padding = autopad(kernel_size, padding, dilation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def __str__(self) -> str:
        return 'conv_bn_silu'
