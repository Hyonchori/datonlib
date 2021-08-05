# Dataloader from 'Human behaviour video' directory
import cv2
import torch
import os
import numpy as np
import albumentations as A
import json
from torchvision.transforms import transforms
from PIL import Image

from datonlib.utils import general, plot


video_num, video_idx = 45, 1  # video_num = 45: smoking video
root = "/media/daton/D6A88B27A88B0569/dataset/사람동작 영상/"
video_root = os.path.join(root, "이미지", f"image_action_{video_num}", f"image_{video_num}-{video_idx}",
                          f"{video_num}-{video_idx}")
annot_root = os.path.join(root, "annotation", "Annotation_2D_tar", "2D", f"{video_num}-{video_idx}")
img_size = (512, 512)
img_transform = A.Compose([
    A.ColorJitter(),
    A.HueSaturationValue(),
    A.RandomBrightnessContrast(),
])
target_transform = A.Compose([
])
toTensor = transforms.ToTensor()

class SmokingDatasetFromOneDir(torch.utils.data.Dataset):
    def __init__(self,
                 video_root: str,
                 annot_root: str,
                 video_num: int=45,  # video_num = 45: smoking
                 video_idx: int=1,
                 _2d_or_3d: str="2d"):

        video_dir = os.path.join(video_root, f"image_action_{video_num}",
                                 f"image_{video_num}-{video_idx}", f"{video_num}-{video_idx}")
        if _2d_or_3d not in ["2d", "3d"]:
            raise Exception(f"'{_2d_or_3d}' should be one of ['2d', '3d']")
        if _2d_or_3d == "2d":
            annot_dir = os.path.join(annot_root, "Annotation_2D_tar", "2D", f"{video_num}-{video_idx}")
        elif _2d_or_3d == "3d":
            annot_dir = os.path.join(annot_root, "Annotation_3D_tar", "3D", f"{video_num}-{video_idx}")

        if not os.path.isdir(video_dir):
            raise Exception(f"'{video_dir}' should be directory!")
        videos = sorted(os.listdir(video_dir))
        self.videos_path = [os.path.join(video_dir, video) for video in videos]
        if not os.path.isdir(annot_dir):
            raise Exception(f"'{annot_dir}' should be directory!")
        annots = sorted(os.listdir(annot_dir))
        self.annots_path = [os.path.join(annot_dir, annot) for annot in annots
                            if not annot.startswith(".")]
        self.datasets = [SmokingDatasetFromOneVideo(video, annot) for video, annot
                         in zip(self.videos_path, self.annots_path)]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.datasets[idx]


class SmokingDatasetFromOneVideo(torch.utils.data.Dataset):
    def __init__(self, img_dir, annot_path):
        img_names = sorted(os.listdir(img_dir))
        self.imgs = [os.path.join(img_dir, img_name) for img_name in img_names]
        self.annot_path = annot_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # keypoint: ["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle",
        # "pelvis", "throax", "neck", "head_top",
        # "right_wrist", "right_elbow", "right_shoulder",
        # "left_shoulder", "left_elbow", "left_wrist"]

        img = np.array(Image.open(self.imgs[idx]).convert("RGB"))
        bbox = []
        keypoint = []
        if os.path.isfile(self.annot_path):
            with open(self.annot_path, "r") as f:
                data = json.load(f)
                images = data["images"]
                categories = data["categories"]
                annotations = data["annotations"]
                annot = annotations[idx]
                bbox = annot["bbox"] + ["smoking", 45]
                keypoint = annot["keypoints"]
        return img, bbox, keypoint


if __name__ == "__main__":
    root = "/media/daton/D6A88B27A88B0569/dataset/사람동작 영상/"
    video_root = os.path.join(root, "이미지")
    annot_root = os.path.join(root, "annotation")
    datasets = SmokingDatasetFromOneDir(video_root=video_root,
                                        annot_root=annot_root,
                                        video_num=45,
                                        video_idx=2)

    for dataset in datasets:
        for img, bbox, keypoint in dataset:
            drawed_img = plot.draw_one_bbox(img, bbox, get_return=True)
            drawed_img = plot.draw_one_keypoint(drawed_img, keypoint,
                                                connection_type="aihub",
                                                get_return=True)
            cv2.imshow("img", drawed_img)
            cv2.waitKey(1)