
import torch
import torch.nn as nn

from datonlib.models.building_blocks import common


class CustomExtractor(nn.Module):
    def __init__(self, use_agg=False, agg_idx=[]):
        super().__init__()
        self.extractor = nn.Sequential(
            common.Focus(3, 48, 3),
            common.ConvBnAct(48, 96, 3, 2),
            common.BottleneckCSP(96, 96, 2),
            common.ConvBnAct(96, 192, 3, 2),
            common.BottleneckCSP(192, 192, 6),
            common.ConvBnAct(192, 384, 3, 2),
            common.BottleneckCSP(384, 384, 6),
            common.ConvBnAct(384, 768, 3, 2),
            common.SPP(768, 768, [5, 9, 13]),
            common.BottleneckCSP(768, 768, 2, shortcut=False, fused=False),
            common.PWConv(768, 384)
        )

        self.use_agg = use_agg
        self.agg_idx = agg_idx

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            print(i, x.size())
            if i in self.agg_idx:
                skips.append(x)
        return x if not self.use_agg else skips


if __name__ == "__main__":
    model = CustomExtractor(use_agg=True, agg_idx=[10, 5, 3]).cuda()
    sample = torch.randn(1, 3, 512, 512).cuda()

    pred = model(sample)
    print("")
    if isinstance(pred, list):
        for p in pred:
            print(p.size())
    else:
        print(pred.size())