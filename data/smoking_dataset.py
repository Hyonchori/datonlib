# Dataloader from 'Human behaviour video' directory

import torch
import os
import numpy as np
import albumentations as A
from torchvision.transforms import transforms
from PIL import Image

from datonlib.utils import general, plot


video_num, video_idx = 45, 1
root = "/media/daton/D6A88B27A88B0569/dataset/사람동작 영상/비디오/"
video_root = os.path.join(root, f"video_action_{video_num}", f"video_{video_num}-{video_idx}",
                          f"{video_num}-{video_idx}")
annot_root = os.path.join(root, "Annotation_2D_tar", "2D", f"{video_num}-{video_idx}")
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
    def __init__(self, root, annot_root):
        self.root = root
        self.annot_root = annot_root
        self.videos = os.listdir(root)

    def __len__(self):
        return len(self.videos)


if __name__ == "__main__":
    dataset = SmokingDatasetFromOneDir(root=video_root,
                                       annot_root=annot_root)
    print(len(dataset))