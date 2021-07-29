# Dataloader from image stream

import os
import cv2
import time
import numpy as np
from threading import Thread


import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

from utils import general
from utils import plot

class LoadStreams:
    def __init__(self,
                 sources: str ="streams.txt",
                 img_size: int = 640,
                 stride: int = 32):
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        print("\n-----------------")
        print(n)
        self.imgs, self.fps, self.frames, self.threads = [None]*n, [0]*n, [0]*n, [None]*n
        self.sources = [general.clean_str(x) for x in sources]
        for i, s in enumerate(sources):
            print(f'{i+1}/{n}: {s}...', end='')
            s = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"Failed to open {s}"

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")

            _, self.imgs[i] = cap.read()
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print("")

        s = np.stack([plot.letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap):
        n, f, read = 0, self.frames[i], 1  # current_frame_number, left_frame_number, inference_every_"read"_frame
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            #time.sleep(1 / self.fps[i])

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration

        img0 = self.imgs.copy()

        # letterboxing img for preventing ratio during downsampling stage in model (stride)
        img = [plot.letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)


if __name__ == "__main__":
    cctv = "rtsp://datonai:datonai@172.30.1.49:554/stream2"
    webcam = "0"
    dataset = LoadStreams(cctv, img_size=640)
    for path, img, img0, vid_cap in dataset:
        print('\n-----')
        cv2.imshow("img0", img0[0])
        cv2.imshow("img", np.asarray(img[0].transpose(1, 2, 0)))
        print(img0[0].shape, img[0].transpose(1, 2, 0).shape)
        cv2.waitKey(1)
