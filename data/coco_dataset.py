# Dataloader from PASCAL VOC dataset directory

import torch
from torchvision.transforms import transforms
import os
import numpy as np
from PIL import Image
import albumentations as A

from datonlib.utils import general, plot


DEFAULT_ROOT = "../../open_dataset/COCO2014"
DEFAULT_ANNOT_ROOT = "../../open_dataset/COCO2014annotations/"
DEFAULT_TARGET_CLS = "person"
DEFAULT_MODE = "detection"
DEFAULT_SIZE = (224, 224)
DEFAULT_CROP_SIZE = (128, 128)
DEFAULT_TRANSFORM = A.Compose([
    A.ColorJitter(),
    A.HueSaturationValue(),
    A.RandomBrightnessContrast(),
])
DEFAULT_TARGET_TRANSFORM = A.Compose([
])
TOTENSOR = transforms.ToTensor()

class COCODataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_root: str,
                 annot_root: str=None,
                 target_cls: str=None,
                 trainval: str="train",
                 mode: str="detection",
                 transform: A.core.composition.Compose=None,
                 ):
        from pycocotools.coco import COCO

        if trainval not in ["train", "val"]:
            raise Exception(f"'trainval' should be selected along ['train', 'val']")
        self.img_dir = img_root + trainval
        annot_root = annot_root if annot_root is not None else img_root + "annotations"

        if mode not in ["detection", "semantic_seg", "instance_seg", "keypoint"]:
            raise Exception(f"'mode' should be selected along ['detection', 'semantic_seg', 'instance_seg', 'keypoint']")
        self.mode = mode
        self.transform = transform

        year = img_root[-4:]
        if mode == "detection" or "seg" in mode:
            target_annot = os.path.join(annot_root, "instances_{}{}.json".format(trainval, year))
        elif mode == "keypoint":
            target_annot = os.path.join(annot_root, "person_keypoints_{}{}.json".format(trainval, year))

        if os.path.isfile(target_annot):
            self.coco = COCO(target_annot)
        else:
            raise Exception(f"'{target_annot}' is not found in COCO '{trainval}' dataset")

        self.img_ids = self.coco.getImgIds() if target_cls is None \
            else self.coco.getImgIds(catIds=self.coco.getCatIds(catNms=target_cls))


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.mode == "detection":
            return self.get_det_item(idx)
        elif self.mode == "semantic_seg":
            return self.get_semantic_seg_item(idx)
        elif self.mode == "instance_seg":
            return self.get_instance_seg_item(idx)
        elif self.mode == "keypoint":
            return self.get_keypoint_item(idx)
        else:
            raise Exception(f"'{self.mode}' is invalid mode!")

    def get_det_item(self, idx):
        # Get image and bounding box(x_min, y_min, x_max, y_max, int: cls)
        idx = self.img_ids[idx]
        img_name = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.img_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))

        ann_ids = self.coco.getAnnIds(imgIds=idx)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        for ann in anns:
            cls = ann["category_id"]
            x_min, y_min, width, height = ann["bbox"]
            x_min, y_min, x_max, y_max = general.xywh2xyxy(x_min, y_min, width, height)
            bboxes.append([x_min, y_min, x_max, y_max, cls])

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
        return img, bboxes

    def get_keypoint_item(self, idx):
        # Get image and keypoint([x, y, v], [x, y, v] ...)
        # v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible
        idx = self.img_ids[idx]
        img_name = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.img_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))

        ann_ids = self.coco.getAnnIds(imgIds=idx)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        keypoints = []
        for ann in anns:
            cls = ann["category_id"]
            x_min, y_min, width, height = ann["bbox"]
            x_min, y_min, x_max, y_max = general.xywh2xyxy(x_min, y_min, width, height)
            bboxes.append((x_min, y_min, x_max, y_max, cls))
            keypoint = ann["keypoints"]
            keypoints.append(keypoint)

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, keypoints=keypoints)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            keypoints = transformed["keypoints"]

        return img, bboxes, keypoints


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


def keypoint_collate_fn(data):
    img_batch, keypoint_batch = [], []
    imgs, bboxes, keypoints = zip(*data)
    for img, bbox, keypoint in zip(imgs, bboxes, keypoints):
        cropped_imgs, adj_keypoints = general.crop_bbox_with_keypoint(img, bbox, keypoint, letterbox_size=DEFAULT_CROP_SIZE)
        if len(adj_keypoints) > 0:
            img_batch.append(torch.from_numpy(np.transpose(cropped_imgs, (0, 3, 1, 2))))
            keypoint_batch.append(adj_keypoints)
    img_batch = torch.cat(img_batch) if len(img_batch) > 0 else img_batch
    keypoint_batch = np.concatenate(keypoint_batch) if len(keypoint_batch) > 0 else keypoint_batch
    return img_batch, keypoint_batch


collate_fn_dict = {
    "detection": detection_collate_fn,
    "keypoint": keypoint_collate_fn
}

def get_COCODataloader(root: str=DEFAULT_ROOT,
                       annot_root: str=DEFAULT_ANNOT_ROOT,
                       target_cls: str=DEFAULT_TARGET_CLS,
                       mode: str=DEFAULT_MODE,
                       transform: A.core.composition.Compose=DEFAULT_TRANSFORM,
                       target_transform : A.core.composition.Compose=DEFAULT_TARGET_TRANSFORM,
                       train_batch: int=32,
                       valid_batch: int=32):
    train_dataset = COCODataset(root, annot_root, target_cls, "train", mode, transform)
    valid_dataset = COCODataset(root, annot_root, target_cls, "val", mode, target_transform)
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

    # Detection dataset
    '''train_dataloader, valid_dataloader = get_COCODataloader()
    for imgs, bboxes in train_dataloader:
        print(imgs.shape)
        for img, bbox in zip(imgs, bboxes):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = plot.draw_bboxes(img_np, bbox)
            cv2.imshow("img", img_np)
            cv2.waitKey(0)
        break'''


    # Keypoint dataset
    train_dataloader, valid_dataloader = get_COCODataloader(mode="keypoint")
    for imgs, keypoints in train_dataloader:
        print(imgs.shape)
        print(keypoints.shape)
        for img, keypoint in zip(imgs, keypoints):
            img_np = img.permute(1, 2, 0).numpy()
            unorm_keypoint = general.unnormalize_keypoint(keypoint, DEFAULT_CROP_SIZE)
            img_np = plot.draw_one_keypoint(img_np, unorm_keypoint, get_return=True)
            cv2.imshow("img", img_np)
            cv2.waitKey(0)
        break
