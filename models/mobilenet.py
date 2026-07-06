import torch
import torch.nn as nn

# Smallest building block
class InvertedResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride,expand_ratio):
    super().__init__()
    self.stride = stride
    self.use_residual = (self.stride ==1) and (in_channels == out_channels)
    hidden_dim = in_channels * expand_ratio
    self.expand = expand_ratio !=1

    layers = []
    # Expansion
    if self.expand:
      layers.append(nn.Conv2d(in_channels, hidden_dim,kernel_size=1, bias=False))
      layers.append(nn.BatchNorm2d(hidden_dim))
      layers.append(nn.ReLU6(inplace=True))

    # Depthwise
    layers.append(nn.Conv2d(hidden_dim, hidden_dim,kernel_size=3, stride=stride, padding=1, groups = hidden_dim, bias=False))
    layers.append(nn.BatchNorm2d(hidden_dim))
    layers.append(nn.ReLU6(inplace=True))

    # Projection
    layers.append(nn.Conv2d(hidden_dim,out_channels,kernel_size=1, bias=False))
    layers.append(nn.BatchNorm2d(out_channels))

    self.block = nn.Sequential(*layers)

  def forward(self,x):
    out = self.block(x)
    if self.use_residual:
      return x + out
    return out

# Stage Builder
def make_stage(in_channels, out_channels, repeat, stride, expand_ratio):
  layers = []
  for i in range(repeat):
    s = stride if i == 0 else 1
    in_c = in_channels if i == 0 else out_channels
    layers.append(InvertedResidualBlock(in_c,out_channels, s, expand_ratio))
  return nn.Sequential(*layers)

# MobileNetV2
class MobileNetV2(nn.Module):
  cfg = [
      (1,16,1,1),
      (6,24,2,2),
      (6,32,3,2),
      (6,64,4,2),
      (6,96,3,1),
      (6,160,3,2),
      (6,320,1,1)]
  def __init__(self, num_classes=16, width_mult=1):
    super().__init__()
    input_channels = int(32 * width_mult)

    # Stem
    self.stem = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=input_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(input_channels),
        nn.ReLU6(inplace=True))

    #Stages
    stages = []
    in_channels = input_channels
    for t,c,n,s in self.cfg:
      stages.append(make_stage(in_channels, c, n, s, t))
      in_channels = c
    self.stages = nn.Sequential(*stages)

    # Head
    last_channel = int(1280 * width_mult)
    self.head = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=last_channel, kernel_size=1, bias=False),
        nn.BatchNorm2d(last_channel),
        nn.ReLU6(inplace=True))

    self.pool = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(0.2)
    self.classifier = nn.Linear(last_channel, num_classes)

    self._initialize_weights()

  def forward(self,x):
    x = self.stem(x)
    x = self.stages(x)
    x = self.head(x)
    x = self.pool(x)
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight,0,0.01)
        nn.init.constant_(m.bias,0)
