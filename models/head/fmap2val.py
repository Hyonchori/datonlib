# Head that outputs a single value from each feature map

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomHead(nn.Module):
    # Designed to output a keypoint
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        outputs = []
        for x in xs:
            bs, _, w, h = x.size()
            x = F.avg_pool2d(x, kernel_size=(w, h)).view(bs, -1)
            x = torch.sigmoid(x)
            outputs.append(x)
        return outputs


if __name__ == "__main__":
    samples = [
        torch.randn(32, 51, 32, 16),
        torch.randn(32, 51, 16, 8),
        torch.randn(32, 51, 8, 4)
    ]

    model = CustomHead()
    pred = model(samples)
    for p in pred:
        print(p.size())