import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from datonlib.data.coco_dataset import get_COCODataloader
from datonlib.models.backbone.custom_extractor import CustomExtractor
from datonlib.models.neck.top_down import TopDownAgg
from datonlib.models.neck.bottom_up import BottomUpAgg
from datonlib.models.head.fmap2val import CustomHead
from datonlib.models.regularizations.dropblock import DBLinearSheduler
from datonlib.train.loss.focal_loss import FocalLossV2
from datonlib.train.keypoint_train import train_model


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = CustomExtractor(use_agg=True, agg_interval=2, agg_num=3)
        self.panet = nn.Sequential(
            TopDownAgg(
                c_ins=[80, 112, 160],
                c_outs=[51, 51, 51],
                use_concat=False
            ),
            BottomUpAgg(
                c_ins=[51, 51, 51],
                c_outs=[51, 51, 51],
                use_concat=False
            )
        )
        self.head = CustomHead()
        self.dropblock = DBLinearSheduler(5, 50)

    def forward(self, x, epoch=None):
        xs = self.extractor(x)
        extractor_output = []
        for x in xs:
            extractor_output.append(self.dropblock(x, epoch))

        xs = self.panet(extractor_output)
        panet_ouput = []
        for x in xs:
            panet_ouput.append(self.dropblock(x, epoch))

        xs = self.head(panet_ouput)
        return xs

if __name__ == "__main__":
    train_dataloader, valid_dataloader = get_COCODataloader(mode="keypoint",
                                                            train_batch=48)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    loss_fn = FocalLossV2()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)
    # exp_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=10,
                                                                         T_mult=1,
                                                                         eta_min=0.0002,
                                                                         last_epoch=-1)


    save_dir = "keypoint_det.pth"
    start_epoch, end_epoch = 0, 50
    max_batch_size = 100
    train_losses, valid_losses = train_model(model, optimizer, cos_scheduler, loss_fn, device, save_dir,
                                             start_epoch, end_epoch, train_dataloader, valid_dataloader,
                                             max_batch_size)

    plt.plot(train_losses["mse"][5: ])
    plt.plot(valid_losses["mse"][5: ])
    plt.show()