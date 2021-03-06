# YOLOv5 common modules

import torch
import torch.nn as nn
import math

import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())


def autopad(kernel, padding=None):
    # Equal to 'padding="same"' in tensorflow
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x//2 for x in kernel]
    return padding


class ConvBnAct(nn.Module):
    # Standard convolution block followed by batch normalization and activation function
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, bn=True):
        # channel_in, channel_out, kernel_size, stride, padding, groups, activation
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False) # bias = False for BN
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = nn.Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def DWConv(c1, c2, k=3, s=1, p=None, act=True, bn=True):
    # Depth-wise ConvBnAct
    return ConvBnAct(c1, c2, k, s, p=p, g=math.gcd(c1, c2), act=act, bn=bn)


def PWConv(c1, c2, s=1, g=1, p=None, act=True, bn=True):
    # Point-wise ConvBnAct
    return ConvBnAct(c1, c2, 1, s, p=p, g=g, act=act, bn=bn)


class DWSConv(nn.Module):
    # Depth-wise separable ConvBnAct
    # Faster than Basic conv when channel is large and size is small
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dwconv = DWConv(c1, c2, k, s, act=act)
        self.pwconv = PWConv(c2, c2, 1, act=act)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x


class SEModule(nn.Module):
    # Squeeze and excitation module. arXiv:1709.01507
    def __init__(self, c_in, reduction=16):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c_in, c_in//reduction, bias=False),
            nn.Mish(),
            nn.Linear(c_in//reduction, c_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, c, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, c)
        y = self.excitation(y).view(batch_size, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    # Residual block. arXiv:1512.03385
    def __init__(self, c1, c2, k=3, s=1, act=True, use_se=True, se_r=16, shortcut=True):
        # in channel, out_channel, kernel size, stride, activation function, using SE module, reduction in SE
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnAct(c1, c2, k, s),
            ConvBnAct(c2, c2, k, 1, act=None),
            SEModule(c2, reduction=se_r) if use_se else nn.Identity()
        )
        self.act = nn.Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if shortcut:
            self.downsampling = PWConv(c1, c2, s) if c1 != c2 \
                else (nn.MaxPool2d(k, stride=s) if s != 1 else nn.Identity())
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv(x)
        if self.shortcut:
            x = self.downsampling(x)
            return self.act(x + y)
        else:
            return self.act(y)


class Bottleneck(nn.Module):
    # Bottleneck block. arXiv:1512.03385. = Inverted residual block(when b_e > 1) = MBConv block
    def __init__(self, c1, c2, k=3, s=1, act=True, use_se=True, se_r=16, b_e=4, shortcut=True):
        # in channel, out_channel, kernel size, stride, activation, using SE module, reduction in SE, expansion ratio
        super().__init__()
        c_mid = int(c1 * b_e)
        self.conv = nn.Sequential(
            PWConv(c1, c_mid, 1),
            DWConv(c_mid, c_mid, k, s),
            SEModule(c_mid, reduction=se_r) if use_se else nn.Identity(),
            PWConv(c_mid, c2, 1, act=None)
        )
        self.act = nn.Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if shortcut:
            self.downsampling = PWConv(c1, c2, s) if c1 != c2 \
                else (nn.MaxPool2d(k, stride=s) if s != 1 else nn.Identity())
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv(x)
        if self.shortcut:
            x = self.downsampling(x)
            return self.act(x + y)
        else:
            return self.act(y)


class FusedBottleneck(nn.Module):
    # Fused bottleneck block. arXiv:2104.00298. = Fused MBConv block
    # Faster than Bottleneck when feature map resolution is high and channel is low
    def __init__(self, c1, c2, k=3, s=1, act=True, use_se=True, se_r=16, b_e=4, shortcut=True):
        # in channel, out_channel, kernel size, stride, activation, using SE module, reduction in SE, expansion ratio
        super().__init__()
        c_mid = int(c1 * b_e)
        self.conv = nn.Sequential(
            ConvBnAct(c1, c_mid, k, s),
            SEModule(c_mid, reduction=se_r) if use_se else nn.Identity(),
            PWConv(c_mid, c2, 1, act=None)
        )
        self.act = nn.Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if shortcut:
            self.downsampling = PWConv(c1, c2, s) if c1 != c2 \
                else (nn.MaxPool2d(k, stride=s, padding=autopad(k, None)) if s != 1 else nn.Identity())
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv(x)
        if self.shortcut:
            x = self.downsampling(x)
            return self.act(x + y)
        else:
            return self.act(y)


class BottleneckCSP(nn.Module):
    # CSP bottleneck used in YOLOv5
    def __init__(self, c1, c2, n=1, b_e=0.5, shortcut=True, fused=True):
        super().__init__()
        c_mid = int(c2 * b_e)
        self.conv1 = ConvBnAct(c1, c_mid, 1, 1)
        self.conv2 = ConvBnAct(c1, c_mid, 1, 1)
        self.conv3 = ConvBnAct(2 * c_mid, c2, 1)
        if fused:
            self.m = nn.Sequential(
                *[FusedBottleneck(c_mid, c_mid, shortcut=shortcut, b_e=1, use_se=False) for _ in range(n)])
        else:
            self.m = nn.Sequential(
                *[Bottleneck(c_mid, c_mid, shortcut=shortcut, b_e=1, use_se=False) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class ConcatLayer(nn.Module):
    # Implementation of torch.cat() in layer
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, xs):
        # xs = iterable of torch.tensor: (tensor, tensor, ...) or [tensor, tensor, ...]
        return torch.cat(xs, self.d)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 3)):
        super().__init__()
        c_mid = c1 // 2
        self.conv1 = ConvBnAct(c1, c_mid, 1, 1)
        self.conv2 = ConvBnAct(c_mid * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus layer used in YOLOv5
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        # in channel, out channel, kernel size, stride, groups, activation
        super().__init__()
        self.conv = ConvBnAct(c1 * 4, c2, k, s, p, g, act)  # point-wise conv when k=1

    def forward(self, x):
        # x(b, c, w, h) -> y(b, 4c, w/2, h/2)
        return self.conv(torch.cat([x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]],
                                   1))


class ConvDownSampling(nn.Module):
    # Downsampling module concatenating max pooling and convolution with stride > 1
    def __init__(self, c1, c2, k=3, s=2, g=1, act=True):
        # in channel, out channel, kernel size, stride, groups, activation
        super().__init__()
        self.conv = ConvBnAct(c1, c2, k, s, g=g, act=act)
        self.maxpool = nn.MaxPool2d(k, stride=s, padding=autopad(k, None))

    def forward(self, x):
        y = self.conv(x)
        x = self.maxpool(x)
        return torch.cat((x, y), dim=1)


class DeconvBnAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, g=1, act=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(c1, c2, k, s, padding=autopad(k, None), output_padding=autopad(k, None), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class ConvUpSampling(nn.Module):
    # Upsampling module concatenating upsampling and deconvolution with stride > 1
    def __init__(self, c1, c2, k=3, s=2, g=1, act=True, use_convT=False):
        # in channel, out channel, kernel size, stride, gropus, activation
        super().__init__()
        self.use_convT = use_convT
        self.deconv = DeconvBnAct(c1, c2, k, s, g, act) if use_convT else nn.Identity()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=s)

    def forward(self, x):
        y = self.deconv(x)
        x = self.upsample(x)
        if self.use_convT:
            return torch.cat((x, y), dim=1)
        else:
            return x


import time

if __name__ == "__main__":
    c_in = 256
    c_out = 256
    batch_size = 32
    w, h = 32, 32
    sample = torch.randn(batch_size, c_in, w, h)


    t1 = time.time()
    print("\nBasicConv: ", ConvBnAct(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nDepth-wise Conv: ", DWConv(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nPoint-wise Conv: ", PWConv(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nDepth-wise separable Conv: ", DWSConv(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nResidual block: ", ResidualBlock(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nBottleneck(MBConv): ", Bottleneck(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nFused MBConv: ", FusedBottleneck(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nBottleneck CSP: ", BottleneckCSP(c_in, c_out, fused=False, n=4)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nSPP: ", SPP(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print("\nFocus: ", Focus(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print(ConvDownSampling(c_in, c_out)(sample).size(), time.time() - t1)

    t1 = time.time()
    print(ConvUpSampling(c_in, c_out)(sample).size(), time.time() - t1)