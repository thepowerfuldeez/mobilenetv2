import torch as t
import torch.nn as nn
import torch.nn.functional as F

class DepthWiseSepConv(nn.Module):
    """
    Depthwise separable convolution
    From pytorch documentation:
    At groups=in_channels, each input channel is convolved 
    with its own set of filters, of size: math.floor(C_in/C_out)"""
    def __init__(self, **kwargs):
        super().__init__()
        # produces in_channels separable filters with depth = 1
        self.depthwise_conv = nn.Conv2d(kwargs['in_channels'], kwargs['in_channels'], kwargs['kernel_size'], 
                                        kwargs['stride'], kwargs['padding'], groups=kwargs['in_channels'])
        self.bn1 = nn.BatchNorm2d(kwargs['in_channels'])
        # makes linear combination of filters above
        self.pointwise_conv = nn.Conv2d(kwargs['in_channels'], kwargs['out_channels'], 1)
        self.bn2 = nn.BatchNorm2d(kwargs['out_channels'])

    def forward(self, x):
        filters = self.depthwise_conv(x)
        filters = self.bn1(filters)
        filters = F.selu(filters)
        # linear combination of produced 3x3 filters
        features = self.pointwise_conv(filters)
        features = self.bn2(features)
        features = F.selu(features)
        return features


class MobileNet(nn.Module):
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
        self.conv1 = nn.Conv2d(input_channels, int(width_mult * 32), 3, 2)
        self.bn1 = nn.BatchNorm2d(int(width_mult * 32))
        # number_layers, in_channels, out_channels, kernel_size, stride, padding
        params = [
            (1, 32, 64, 3, 1, 1),
            (1, 64, 128, 3, 2, 1),
            (1, 128, 128, 3, 1, 1),
            (1, 128, 256, 3, 2, 1),
            (1, 256, 256, 3, 1, 1),
            (1, 256, 512, 3, 2, 1),
            (5, 512, 512, 3, 1, 1),
            (1, 512, 1024, 3, 2, 1),
            (1, 1024, 1024, 3, 1, 1)
        ]
        # 1 conv + 13 sep-depth conv layers
        layers = []
        for n, in_c, out_c, k, s, p in params:
            for _ in range(n):
                layers.append(
                    DepthWiseSepConv(in_channels=int(width_mult * in_c), out_channels=int(width_mult * out_c), 
                                     kernel_size=k, stride=s, padding=p)
                )
        self.features = nn.Sequential(self.conv1, self.bn1, nn.SELU(), *layers)
        self.fc = nn.Linear(self.features[-1].pointwise_conv.out_channels, n_classes)

    def forward(self, x):
        x = self.features(x)
        # spatial size produced from the features
        avg_pool_kernel_size = x.size()[-2:]
        x = F.avg_pool2d(x, avg_pool_kernel_size)
        # flatten output
        x = x.view(x.size(0), -1)
        x = F.selu(x)
        x = self.fc(x)
        return x