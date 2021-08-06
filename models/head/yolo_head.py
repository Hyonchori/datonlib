
import torch
import torch.nn as nn


class YOLOHead(nn.Module):
    def __init__(self, nc=2, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


if __name__ == "__main__":
    x = [
        torch.randn(1, 192, 64, 64),
        torch.randn(1, 384, 32, 32),
        torch.randn(1, 384, 16, 16)
    ]

    h = YOLOHead(anchors=[[10, 13, 16, 30, 33, 23],
                          [30, 61, 62, 45, 59, 119],
                          [116, 90, 156, 198, 373, 326]],
                 ch=[192, 384, 384])
    h.eval()

    pred = h(x)[0]
    print(pred.size())
    from datonlib.utils.nms import non_maximum_suppression
    pred = non_maximum_suppression(pred)

