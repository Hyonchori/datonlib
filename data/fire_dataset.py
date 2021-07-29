# Dataloader from '화재 발생 예측 데이터' directory

import torch
from torchvision.transforms import transforms
import os
import numpy as np
import albumentations as A
import json
from PIL import Image

from datonlib.utils import general, plot


DEFAULT_ROOT = "../../open_dataset/화재 발생 예측 데이터/Training/화재씬_1"
DEFAULT_ANNOT_ROOT = "../../open_dataset/화재 발생 예측 데이터/Training/[라벨]화재씬"
DEFAULT_SIZE = (512, 512)
DEFAULT_TRANSFORM = A.Compose([
    A.ColorJitter(),
    A.HueSaturationValue(),
    A.RandomBrightnessContrast(),
])
DEFAULT_TARGET_TRANSFORM = A.Compose([
])
TOTENSOR = transforms.ToTensor()

class FireDatasetFromOneDir(torch.utils.data.Dataset):
    def __init__(self, root, annot_root):
        self.root = root
        self.annot_root = annot_root
        self.imgs = os.listdir(root)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        annot_path = os.path.join(self.annot_root, self.imgs[idx].replace("jpg", "json"))

        img = np.array(Image.open(img_path))
        bboxes = []
        with open(annot_path, "rt", encoding="UTF8") as f:
            data = json.load(f)
            annots = data["annotations"]
            for annot in annots:
                middle_cls = int(annot["middle classification"])
                cls = int(annot["class"])
                if "box" in annot:
                    bbox = annot["box"]
                    flag = annot["flags"]
                    bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], cls])
        return img, bboxes


def detection_collate_fn(data):
    img_batch, bbox_batch = [], []
    imgs, bboxes = zip(*data)
    for img, bbox in zip(imgs, bboxes):
        img, ratio, (dw, dh) = plot.letterbox(img, new_shape=DEFAULT_SIZE, auto=False)
        bbox = general.normalize_bboxes(bbox, ratio, dw, dh)
        if len(bbox) == 0:
            continue
        img_batch.append(TOTENSOR(img).unsqueeze(dim=0))
        bbox_batch.append(bbox)
    img_batch = torch.cat(img_batch)
    return img_batch, bbox_batch


def get_FireDataloader(root: str=DEFAULT_ROOT,
                       annot_root: str=DEFAULT_ANNOT_ROOT,
                       batch_size: int=32):
    dataset = FireDatasetFromOneDir(root, annot_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn
    )
    return dataloader


if __name__ == "__main__":
    import cv2

    dataloader = get_FireDataloader()
    for imgs, bboxes in dataloader:
        print(imgs.size())
        for img, bbox in zip(imgs, bboxes):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = plot.draw_bboxes(img_np, bbox, color_norm=True)
            cv2.imshow("img", img_np)
            cv2.waitKey(1)