# Dataloader from PASCAL VOC dataset directory

import torch
from torchvision.transforms import transforms
import os
import numpy as np
import albumentations as A
import xml.etree.ElementTree as Et
from PIL import Image

from datonlib.utils import general, plot


DEFAULT_ROOT = "../../open_dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
DEFAULT_TARGET_CLS = "PERSON"
DEFAULT_MODE = "detection"
DEFAULT_SIZE = (224, 224)
DEFAULT_TRANSFORM = A.Compose([
    A.ColorJitter(),
    A.HueSaturationValue(),
    A.RandomBrightnessContrast(),
])
DEFAULT_TARGET_TRANSFORM = A.Compose([
])
TOTENSOR = transforms.ToTensor()

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 target_cls: str=None,
                 trainval: str="train",
                 mode: str="detection",
                 transform: A.core.composition.Compose=None,
                 ):
        if trainval not in ["train", "val", "trainval", "test"]:
            raise Exception(f"'trainval' should be selected along ['train', 'val', 'trainval', 'test']")
        if "test" in root:
            trainval = "test"

        cls_dir = os.path.join(root, "ImageSets", "main")
        self.cls_list = sorted(list(set([x.split("_")[0] for x in os.listdir(cls_dir)
                            if (not x.split("_")[0].endswith("txt"))])))

        if target_cls is None or trainval == "test":
            target_file = "{}.txt".format(trainval)
            print(target_file)
            target_path = os.path.join(cls_dir, target_file)
            if os.path.isfile(target_path):
                with open(target_path, "r") as f:
                    self.target_imgs = [x.strip().split()[0] for x in f.readlines()]
            else:
                raise Exception(f"'{target_file}' is not found in VOC '{trainval}' dataset")
        else:
            target_file = "{}_{}.txt".format(target_cls, trainval)
            target_path = os.path.join(cls_dir, target_file)
            if os.path.isfile(target_path):
                with open(target_path, "r") as f:
                    self.target_imgs = [x.strip().split()[0] for x in f.readlines()
                                   if x.strip().split()[1] == "1"]
            else:
                raise Exception(f"class '{target_cls}' is not found in VOC '{trainval}' dataset")

        if mode not in ["detection", "semantic_seg", "instance_seg"]:
            raise Exception(f"'mode' should be selected along ['detection', 'semantic_seg', 'instance_seg']")
        self.mode = mode
        self.transform = transform

        self.img_dir = os.path.join(root, "JPEGImages")
        self.det_dir = os.path.join(root, "Annotations")
        self.semantic_seg_dir = os.path.join(root, "SegmentationClass")
        self.instance_seg_dir = os.path.join(root, "SegmentationObject")

    def __len__(self):
        return len(self.target_imgs)

    def __getitem__(self, idx):
        if self.mode == "detection":
            return self.get_det_item(idx)
        elif self.mode == "semantic_seg":
            return self.get_semantic_seg_item(idx)
        elif self.mode == "instance_seg":
            return self.get_instance_seg_item(idx)
        else:
            raise Exception(f"'{self.mode}' is invalid mode!")

    def get_det_item(self, idx):
        # Get image and bounding box(x_min, y_min, x_max, y_max, str: name, int: cls)
        img_path = os.path.join(self.img_dir, self.target_imgs[idx]+".jpg")
        img = np.array(Image.open(img_path))
        annot_path = os.path.join(self.det_dir, self.target_imgs[idx]+".xml")
        bboxes = []
        if os.path.isfile(annot_path):
            with open(annot_path, "r") as f:
                tree = Et.parse(f)
                root = tree.getroot()
                objects = root.findall("object")
                for obj in objects:
                    name = obj.find("name").text
                    cls = self.cls_list.index(name)
                    bbox = obj.find("bndbox")
                    x_min = int(bbox.find("xmin").text)
                    y_min = int(bbox.find("ymin").text)
                    x_max = int(bbox.find("xmax").text)
                    y_max = int(bbox.find("ymax").text)
                    bboxes.append([x_min, y_min, x_max, y_max, name, cls])

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
        return img, bboxes

    def get_semantic_seg_item(self, idx):
        raise Exception("Not implemented yet")

    def get_instance_seg_item(self, idx):
        raise Exception("Not implemented yet")


def detection_collate_fn(data):
    img_batch, bbox_batch = [], []
    imgs, bboxes = zip(*data)
    for img, bbox in zip(imgs, bboxes):
        img, ratio, (dw, dh) = plot.letterbox(img, new_shape=DEFAULT_SIZE, auto=False)
        bbox = general.normalize_bboxes(bbox, ratio, dw, dh)
        img_batch.append(TOTENSOR(img).unsqueeze(dim=0))
        bbox_batch.append(bbox)
    img_batch = torch.cat(img_batch)
    return img_batch, bbox_batch


collate_fn_dict = {
    "detection": detection_collate_fn,
}

def get_VOCDataloader(root: str=DEFAULT_ROOT,
                      target_cls: str=DEFAULT_TARGET_CLS,
                      mode: str=DEFAULT_MODE,
                      transform: A.core.composition.Compose=DEFAULT_TRANSFORM,
                      target_transform : A.core.composition.Compose=DEFAULT_TARGET_TRANSFORM,
                      train_batch: int=32,
                      valid_batch: int=8):
    train_dataset = VOCDataset(root, target_cls, "train", mode, transform)
    valid_dataset = VOCDataset(root, target_cls, "val", mode, target_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        collate_fn=collate_fn_dict[mode]
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch,
        shuffle=False,
        collate_fn=collate_fn_dict[mode]
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    import cv2

    train_dataloader, valid_dataloder = get_VOCDataloader()
    for imgs, bboxes in train_dataloader:
        print(imgs.shape)
        for img, bbox in zip(imgs, bboxes):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = plot.draw_bboxes(img_np, bbox, color_norm=True)
            cv2.imshow("img", img_np)
            cv2.waitKey(0)
        break

    for img, bboxes in valid_dataloder:
        print(img.shape)
        break