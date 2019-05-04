import torch as t
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """
    Depthwise separable convolution
    From pytorch documentation:
    At groups=in_channels, each input channel is convolved 
    with its own set of filters, of size: math.floor(C_in/C_out)"""
    def __init__(self, **kwargs):
        super().__init__()
        # expansion 1x1 pointwise conv
        self.expansion = nn.Conv2d(kwargs['in_channels'], kwargs['expansion_rate'] * kwargs['in_channels'], 1)
        # produces t * in_channels separable filters with depth = 1 from input
        self.depthwise_conv = nn.Conv2d(kwargs['expansion_rate'] * kwargs['in_channels'], 
                                        kwargs['expansion_rate'] * kwargs['in_channels'], kwargs['kernel_size'], 
                                        kwargs['stride'], kwargs['padding'], 
                                        groups=kwargs['expansion_rate'] * kwargs['in_channels'])
        # makes linear combination of filters above
        self.pointwise_conv = nn.Conv2d(kwargs['expansion_rate'] * kwargs['in_channels'], kwargs['out_channels'], 1)
        self.bn2 = nn.BatchNorm2d(kwargs['out_channels'])
        self.residual = kwargs['residual']

    def forward(self, x):
        filters = self.expansion(x)
        filters = F.relu6(filters)
        filters = self.depthwise_conv(filters)
        filters = F.relu6(filters)
        # linear combination of produced 3x3 filters
        features = self.pointwise_conv(filters)
        if self.residual:
            return features + x
        else:
            return features


class MobileNetV2(nn.Module):
    """
    MobileNet implementation
    """
    def __init__(self, width_mult=1, res_mult=1, input_channels=3, n_classes=1000):
        """
        :param width_mult: number of parameter shrinking multiplier
        :param res_mult: image resolution multiplier
        """
        assert 0.0315 < width_mult <= 1, "Width multiplier must be in interval (0.0315, 1]"
        super().__init__()
        output_channels = int(width_mult * 32)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, 2)
        self.bn1 = nn.BatchNorm2d(output_channels)
        # expansion_rate, output_channels, number_layers, stride
        params = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1)
        ]
        # 1 conv + 17 sep-depth conv layers
        layers = []
        in_channels = output_channels
        for t, c, n, s in params:
            for i in range(n):
                layers.append(
                    InvertedResidual(in_channels=in_channels, out_channels=int(width_mult * c), 
                                     expansion_rate=t,
                                     kernel_size=3, 
                                     stride=(s if i == 0 else 1), 
                                     padding=1,
                                     residual=i != 0)
                )
                in_channels = int(width_mult * c)
        output_channels = int(width_mult * 1280)
        self.conv2 = nn.Conv2d(c, output_channels, 1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.features = nn.Sequential(self.conv1, self.bn1, nn.ReLU6(), *layers, self.conv2, self.bn2)
        self.last_conv = nn.Conv2d(output_channels, n_classes, 1)

    def forward(self, x):
        x = self.features(x)
        # spatial size produced from the features
        avg_pool_kernel_size = x.size()[-2:]
        x = F.avg_pool2d(x, avg_pool_kernel_size)
        x = F.relu6(x)
        x = self.last_conv(x)
        # flatten output
        x = x.view(x.size(0), -1)
        return x