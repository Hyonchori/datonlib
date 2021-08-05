# DataLoader from MOT dataset
import cv2
import torch
import os
import numpy as np
import albumentations as A

from PIL import Image

from datonlib.utils import general, plot

class MOTDatasetTotal(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 trainval: str="train",
                 mode: str="det",
                 use_det: bool=False,
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

        self.img_path = sorted([os.path.join(self.img_dir, x) for x in os.listdir(self.img_dir)])

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


if __name__ == "__main__":
    root = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17"
    mot_dataset = MOTDatasetTotal(root=root,
                                  trainval="train")
    for video_dataset in mot_dataset:
        for img, bboxes in video_dataset:
            print(img.shape)
            drawed_img = plot.draw_bboxes(img, bboxes)
            cv2.imshow("img", drawed_img)
            cv2.waitKey(1)
            