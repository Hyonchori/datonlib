# Bottom-Up feature map aggregation

import torch
import torch.nn as nn

from datonlib.models.building_blocks import common


class BottomUpAgg(nn.Module):
    # size in xs: [[bs, c1, w1, h1], [bs, c2, w1*2, h1*2], [bs, c3, w1*4, h1*4] ...]
    # use_concat = False : use residual => more faster
    def __init__(self,
                 c_ins: list,  # [int, int, int...]
                 c_outs: list,  # [int, int, int...]
                 use_concat: bool = False):
        super().__init__()
        bottomup = []
        num_in = len(c_ins)
        if use_concat:
            for i in range(num_in):
                if i == num_in - 1:
                    bottomup.append(common.Bottleneck(c_ins[i], c_outs[i]))
                else:
                    bottomup.append(common.Bottleneck(c_ins[i] + c_outs[i + 1], c_outs[i]))
        else:
            add_convs = []
            for i in range(num_in):
                if i == 0:
                    bottomup.append(nn.Identity())
                    add_convs.append(common.Bottleneck(c_ins[i], c_outs[0]))
                else:
                    bottomup.append(common.PWConv(c_ins[i], c_ins[i - 1]))
                    add_convs.append(common.Bottleneck(c_ins[i - 1], c_outs[0]))
            self.add_convs = nn.Sequential(*add_convs)
        self.bottomup = nn.Sequential(*bottomup)
        self.use_concat = use_concat
        self.downsampling = nn.MaxPool2d(3, stride=2, padding=common.autopad(3))

    def forward(self, xs):
        # size in xs: [[bs, c1, w1, h1], [bs, c2, w1/2, h1/2], [bs, c3, w1/4, h1/4] ...]
        outputs = []
        x = None
        for i, conv in reversed(list(enumerate(self.bottomup))):
            if x is None:
                x = xs.pop()
            else:
                x = self.downsampling(x)
                #print("\t{}, {}".format(x.size(), xs[-1].size()))
                if self.use_concat:
                    x = torch.cat((x, xs.pop()), dim=1)
                else:
                    x = x + xs.pop()
            x = conv(x)
            if self.use_concat:
                outputs.append(x)
            else:
                output = self.add_convs[i](x)
                outputs.append(output)
        return outputs


if __name__ == "__main__":
    import time

    samples = [
        torch.randn(32, 80, 8, 4),
        torch.randn(32, 112, 16, 8),
        torch.randn(32, 160, 32, 16)
    ]

    c_ins = [samples[i].size()[1] for i in range(3)]
    c_outs = [51, 51, 51]
    model = BottomUpAgg(c_ins, c_outs, use_concat=False)

    t1 = time.time()
    pred = model(samples)
    for p in pred:
        print(p.size())
    print(time.time() - t1)