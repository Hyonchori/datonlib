# Dataloader from image dataset
import cv2
import torch
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as Et
import albumentations as A

from pprint import pprint

import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

from utils import general
from utils import plot


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 target_cls: str=None,
                 trainval: str="train",
                 mode: str="det",
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

        if mode not in ["det", "semantic_seg", "instance_seg"]:
            raise Exception(f"'mode' should be selected along ['det', 'semantic_seg', 'instance_seg']")
        self.mode = mode
        self.transform = transform

        self.img_dir = os.path.join(root, "JPEGImages")
        self.det_dir = os.path.join(root, "Annotations")
        self.semantic_seg_dir = os.path.join(root, "SegmentationClass")
        self.instance_seg_dir = os.path.join(root, "SegmentationObject")

    def __len__(self):
        return len(self.target_imgs)

    def __getitem__(self, idx):
        if self.mode == "det":
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
                    bboxes.append((x_min, y_min, x_max, y_max, name, cls))

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
        return img, bboxes

    def get_semantic_seg_item(self, idx):
        raise Exception("Not implemented yet")

    def get_instance_seg_item(self, idx):
        raise Exception("Not implemented yet")


class COCODataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_root: str,
                 annot_root: str=None,
                 target_cls: str=None,
                 trainval: str="train",
                 mode: str="det",
                 transform: A.core.composition.Compose=None,
                 ):
        from pycocotools.coco import COCO

        if trainval not in ["train", "val"]:
            raise Exception(f"'trainval' should be selected along ['train', 'val']")
        self.img_dir = img_root + trainval
        annot_root = annot_root if annot_root is not None else img_root + "annotations"

        if mode not in ["det", "semantic_seg", "instance_seg", "keypoint"]:
            raise Exception(f"'mode' should be selected along ['det', 'semantic_seg', 'instance_seg', 'keypoint']")
        self.mode = mode
        self.transform = transform

        year = img_root[-4:]
        if mode == "det" or "seg" in mode:
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
        if self.mode == "det":
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
        img = np.array(Image.open(img_path))

        ann_ids = self.coco.getAnnIds(imgIds=idx)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        for ann in anns:
            cls = ann["category_id"]
            x_min, y_min, width, height = ann["bbox"]
            x_min, y_min, x_max, y_max = general.xywh2xyxy(x_min, y_min, width, height)
            bboxes.append((x_min, y_min, x_max, y_max, cls))

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
        img = np.array(Image.open(img_path))

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


class MOTDatasetTotal(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 trainval: str = "train",
                 mode: str = "det",
                 use_det : bool = False,
                 transform: A.core.composition.Compose = None,
                 ):
        self.use_dr = use_det
        self.transform = transform

        if trainval not in ["train", "test"]:
            raise Exception(f"'trainval' should be selected along ['train', 'test']")

        if mode not in ["det", "track"]:
            raise Exception(f"'mode' should be selected along ['det', 'track']")
        self.mode = mode

        self.data_dir = os.path.join(root, trainval)
        default_detector = "DPM" if "MOT17" in root else ""
        self.video_dirs = [os.path.join(self.data_dir, x) for x in os.listdir(self.data_dir)
                           if default_detector in x]
        self.video_datasets = [MOTDatasetVideo(video_dir, mode=self.mode, transform=self.transform)
                               for video_dir in self.video_dirs]

        self.img_dirs = [os.path.join(x, "img1") for x in self.video_dirs]
        self.det_dirs = [os.path.join(x, "det") for x in self.video_dirs]
        self.gt_dirs = [os.path.join(x, "gt") for x in self.video_dirs]

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        return self.video_datasets[idx]


class MOTDatasetVideo(torch.utils.data.Dataset):
    def __init__(self,
                 video_dir: str,
                 target_id: int=None,
                 target_cls: int=None,
                 mode: str="det",
                 only_confident: bool=True,
                 visible_thr: float=0.1,
                 transform: A.core.composition.Compose = None,
                 ):
        self.target_id = target_id
        # 1: pedestrian, 2: person on vehicle, 3:car, 4: bicycle, 5: motorbike, 6: non motorized vehicle
        # 7: static person, 8: distractor, 9: occluder, 10: occluder on the ground, 11: occluder full, 12: reflection
        self.target_cls = target_cls
        self.mode = mode
        self.only_confident = only_confident
        self.visible_thr = visible_thr
        self.transform = transform

        self.img_dir = os.path.join(video_dir, "img1")
        self.det_dir = os.path.join(video_dir, "det")
        self.gt_dir = os.path.join(video_dir, "gt")

        self.img_path = [os.path.join(self.img_dir, x) for x in os.listdir(self.img_dir)]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_path[idx]))
        bboxes = []

        gt_file = os.path.join(self.gt_dir, "gt.txt")
        # frame number, id, x_min, y_min, width, height, confidence, class, visibility
        if os.path.isfile(gt_file):
            with open(gt_file, "r") as f:
                if self.only_confident:
                    data = [x.strip().split(",") for x in f.readlines()
                            if (int(x.split(",")[0]) == idx+1) and (float(x.split(",")[-1]) >= self.visible_thr)
                            and float(x.split(",")[-3]) == self.only_confident]
                else:
                    data = [x.strip().split(",") for x in f.readlines()
                            if (int(x.split(",")[0]) == idx+1) and (float(x.split(",")[-1]) >= self.visible_thr)]
                for bbox in data:
                    cls = int(bbox[-2])
                    if self.target_cls is not None:
                        if cls != self.target_cls:
                            continue

                    id = int(bbox[1])
                    if self.target_id is not None:
                        if id != self.target_id:
                            continue
                    x_min, y_min, width, height = int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5])
                    x_min, y_min, x_max, y_max = general.xywh2xyxy(x_min, y_min, width, height)

                    if self.mode == "det":
                        bboxes.append((x_min, y_min, x_max, y_max, cls))
                    elif self.mode == "track":
                        bboxes.append((x_min, y_min, x_max, y_max, cls, id))

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
        return img, bboxes




transforms = A.Compose([
    #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
    A.ColorJitter(),
    A.HueSaturationValue(),
    #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=1.)
],
bbox_params=A.BboxParams(format="pascal_voc", min_area=300, min_visibility=0.15),
#keypoint_params=A.KeypointParams(format="xy")
)


if __name__ == "__main__":
    '''voc_root = "../../open_dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
    voc_dataset = VOCDataset(voc_root,
                             target_cls="person",
                             trainval="trainval",
                             mode="det",
                             transform=transforms
                             )
    for img, bboxes in voc_dataset:
        print(img.shape)
        print(len(bboxes))
        drawed_img = plot.draw_bboxes(img, bboxes)
        cv2.imshow("img", drawed_img)
        cv2.waitKey(0)'''


    '''coco_img_root = "../../open_dataset/COCO2014"
    coco_annot_root = "../../open_dataset/COCO2014annotations/"
    coco_dataset = COCODataset(coco_img_root,
                               #coco_annot_root,
                               trainval="val",
                               target_cls="person",
                               mode="det",
                               transform=transforms
                               )

    for img, bboxes in coco_dataset:
        print(img.shape)
        drawed_img = plot.draw_bboxes(img, bboxes)
        cv2.imshow("img", drawed_img)
        cv2.waitKey(0)'''

    '''coco_img_root = "../../open_dataset/COCO2014"
    coco_annot_root = "../../open_dataset/COCO2014annotations/"
    coco_dataset = COCODataset(coco_img_root,
                               # coco_annot_root,
                               trainval="val",
                               target_cls="person",
                               mode="keypoint",
                               transform=transforms
                               )
    
    for img, bboxes, keypoints in coco_dataset:
        print("\n---")
        print(img.shape)
        drawed_img = plot.draw_bboxes(img, bboxes)
        drawed_img = plot.draw_keypoints(drawed_img, keypoints)
        cv2.imshow("img", drawed_img)

        #cv2.imshow("img", img)
        cv2.waitKey(0)'''



    '''mot_root = "../../open_dataset/MOT15"
    mot_dataset = MOTDatasetTotal(mot_root,
                                  trainval="train",
                                  )
    for video_dataset in mot_dataset:
        for img, bboxes in video_dataset:
            print(img.shape)
            drawed_img = plot.draw_bboxes(img, bboxes)
            cv2.imshow("img", drawed_img)
            cv2.waitKey(1)'''

