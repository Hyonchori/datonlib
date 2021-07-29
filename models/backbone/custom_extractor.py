# Custom feature extractor from input image

import torch
import torch.nn as nn
import numpy as np

from datonlib.models.building_blocks import common


class CustomExtractor(nn.Module):
    # Custom feature extractor v1
    def __init__(self, use_agg=False, agg_interval=2, agg_num=3):
        super().__init__()
        self.extractor = nn.Sequential(
            common.FusedBottleneck(3, 16, use_se=False),
            common.FusedBottleneck(16, 32, s=2, use_se=False),
            common.Bottleneck(32, 48, use_se=False),
            common.Bottleneck(48, 64, s=2, use_se=False),
            common.Bottleneck(64, 80),
            common.Bottleneck(80, 96, s=2),
            common.Bottleneck(96, 112),
            common.Bottleneck(112, 144, s=2),
            common.Bottleneck(144, 160)
        )

        self.use_agg = use_agg
        self.agg_interval = [x for x in range(len(self.extractor)-1, 0, -agg_interval)][:agg_num]

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            if i in self.agg_interval:
                skips.append(x)
        return x if not self.use_agg else skips


if __name__ == "__main__":
    model = CustomExtractor(use_agg=True)
    sample = torch.randn(32, 3, 128, 64)

    pred = model(sample)
    if isinstance(pred, list):
        for p in pred:
            print(p.size())
    else:
        print(pred.size())
