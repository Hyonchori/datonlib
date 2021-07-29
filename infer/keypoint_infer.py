import torch
import torch.nn as nn
import cv2
import numpy as np

from datonlib.data.coco_dataset import get_COCODataloader
from datonlib.models.backbone.custom_extractor import CustomExtractor
from datonlib.models.neck.top_down import TopDownAgg
from datonlib.models.neck.bottom_up import BottomUpAgg
from datonlib.models.regularizations.dropblock import DBLinearSheduler
from datonlib.models.head.fmap2val import CustomHead
from datonlib.utils.general import unnormalize_keypoint
from datonlib.utils.plot import draw_one_keypoint


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
    train_dataloader, valid_dataloader = get_COCODataloader(mode="keypoint")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model()

    save_dir = "keypoint_det.pth"
    model.load_state_dict(torch.load(save_dir))
    model = model.to(device)

    for imgs, keypoints in train_dataloader:
        print("\n######")
        imgs = imgs.float().to(device)
        pred = model(imgs)
        for img, keypoint, p1, p2, p3 in zip(imgs, keypoints, pred[0], pred[1], pred[2]):
            img_np = img.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            unorm_keypoint = unnormalize_keypoint(keypoint, (128, 128))
            p1 = p1.cpu().detach().numpy()
            p2 = p2.cpu().detach().numpy()
            p3 = p3.cpu().detach().numpy()

            unorm_p1 = unnormalize_keypoint(p1, (128, 128))
            unorm_p2 = unnormalize_keypoint(p2, (128, 128))
            unorm_p3 = unnormalize_keypoint(p3, (128, 128))

            gt_img = draw_one_keypoint(img_np, unorm_keypoint, get_return=True)
            pred_img1 = draw_one_keypoint(img_np, unorm_p1, get_return=True)
            pred_img2 = draw_one_keypoint(img_np, unorm_p2, get_return=True)
            pred_img3 = draw_one_keypoint(img_np, unorm_p3, get_return=True)
            result_img = np.hstack((gt_img, pred_img1, pred_img2, pred_img3))

            cv2.imshow("result_img", result_img)
            cv2.waitKey(0)