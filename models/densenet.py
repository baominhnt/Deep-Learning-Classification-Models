import torch
import torch.nn as nn


# DenseLayer
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, inter_channels, kernel_size=1, stride=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inter_channels, growth_rate, padding=1, kernel_size=3, stride=1, bias=False
        )

        self.drop_rate = drop_rate
        if drop_rate > 0.5:
            self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_rate > 0:
            out = self.dropout(out)
        return out


# DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            concatenated = torch.cat(features, dim=1)
            new_features = layer(concatenated)
            features.append(new_features)
        return torch.cat(features, dim=1)


# TransitionLayer
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        if not (0 < compression <= 1):
            raise ValueError("Compression value must be between 0 and 1")

        out_channels = max(1, round(in_channels * compression))

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


# DenseNet121
class DenseNet121(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        bn_size=4,
        compression=0.5,
        stem_channels=64,
        drop_rate=0.0,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_channels = stem_channels
        blocks = []
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            blocks.append(block)
            num_channels = num_channels + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionLayer(num_channels, compression=compression)
                blocks.append(trans)
                num_channels = trans.out_channels

        self.dense_blocks = nn.Sequential(*blocks)

        self.final_bn = nn.BatchNorm2d(num_channels)
        self.final_relu = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_channels, num_classes)

        self.num_features = num_channels

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense_blocks(x)
        x = self.final_relu(self.final_bn(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
