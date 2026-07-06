import torch
import torch.nn as nn


# Swish Activation
class SwishActivate(nn.Module):
    def __init__(self):
        super(SwishActivate, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


# SqueezeExcite
class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(1)

        self.excite = nn.Sequential(
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            SwishActivate(),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        squeezed = self.adaptive_avg_pool2d(x)
        channel_excitation = self.excite(squeezed)
        return x * channel_excitation


# MBConvBlock
class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        se_ratio=0.25,
    ):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = (self.stride == 1) and (
            self.in_channels == self.out_channels
        )

        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
                SwishActivate(),
            )
        else:
            self.expand = nn.Identity()
        self.expanded_channels = expanded_channels

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            SwishActivate(),
        )

        reduced_dim = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcite(expanded_channels, reduced_dim)

        self.project = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + identity

        return out


# Stage Builder
def make_stage(
    in_channels,
    out_channels,
    num_blocks,
    kernel_size,
    stride,
    expand_ratio,
    se_ratio=0.25,
):
    layers = []
    layers.append(
        MBConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            expand_ratio=expand_ratio,
        )
    )
    for _ in range(1, num_blocks - 1):
        layers.append(
            MBConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                expand_ratio=expand_ratio,
            )
        )
    return nn.Sequential(*layers)


# EfficientNetB0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            SwishActivate(),
        )
        cfg = [
            (32, 16, 3, 1, 1, 1),
            (16, 24, 3, 2, 6, 2),
            (24, 40, 5, 2, 6, 2),
            (40, 80, 3, 2, 6, 3),
            (80, 112, 5, 1, 6, 3),
            (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1),
        ]
        self.stages = nn.ModuleList()
        for i, (in_c, out_c, k, s, e, r) in enumerate(cfg):
            self.stages.append(make_stage(in_c, out_c, k, s, e, r))

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            SwishActivate(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.head(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
