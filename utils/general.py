# General utils

import re
import numpy as np
import torch

from datonlib.utils.plot import letterbox


def clean_str(s: str):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def xywh2xyxy(x: int,
              y: int,
              w: int,
              h: int):
    x_max = int(x + w)
    y_max = int(y + h)
    return x, y, x_max, y_max


def normalize_bboxes(bboxes,
                     ratio,
                     dw, dh):
    if bboxes:
        if type(bboxes[0]) == int:
            bboxes = [bboxes]
    result = []
    for bbox in bboxes:
        bbox[0] = int(bbox[0] * ratio[0] + dw)
        bbox[1] = int(bbox[1] * ratio[1] + dh)
        bbox[2] = int(bbox[2] * ratio[0] + dw)
        bbox[3] = int(bbox[3] * ratio[1] + dh)
        result.append(bbox)
    return result


def crop_bbox_with_keypoint(img: np.ndarray,
                            bboxes: list,
                            keypoints: list,
                            letterbox_size: (int, int) = None,
                            normalize: bool=True):
    # letterbox_size: desired size of letterboxed image (width, height)
    imgs = []
    adj_keypoints = []
    for bbox, keypoint in zip(bboxes, keypoints):
        if sum(keypoint) == 0:
            continue

        xs = keypoint[0::3]
        ys = keypoint[1::3]
        vs = keypoint[2::3]
        adj_keypoint = []

        bbox = list(map(int, bbox))
        cropped_img = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        if letterbox_size is not None:
            cropped_img, ratio, (dw, dh) = letterbox(cropped_img, new_shape=letterbox_size, auto=False)
            norm = (1, 1) if not normalize else letterbox_size
            for x, y, v in zip(xs, ys, vs):
                adj_keypoint += [((x - bbox[0]) * ratio[0] + dw) / norm[0]] \
                    if x != 0 else [0]
                adj_keypoint += [((y - bbox[1]) * ratio[1] + dh) / norm[1]] \
                    if y != 0 else [0]
                adj_keypoint += [1 if v > 0 else 0]
        else:
            norm = (1, 1) if not normalize else cropped_img.shape[:2]
            for x, y, v in zip(xs, ys, vs):
                adj_keypoint += [(x - bbox[0]) / norm[0]] if x != 0 else [0]
                adj_keypoint += [(y - bbox[1]) / norm[1]] if y != 0 else [0]
                adj_keypoint += [1 if v > 0 else 0]
        imgs.append(cropped_img)
        adj_keypoints.append(adj_keypoint)

    return np.array(imgs), np.array(adj_keypoints)


def normalize_keypoint(keypoint: list,
                       img_size: (int, int)):
    keypoint = np.array(keypoint, dtype=np.float)
    keypoint[0::3] /= img_size[0]
    keypoint[1::3] /= img_size[1]
    keypoint[2::3] = list(map(lambda x: 1 if x > 0 else 0, keypoint[2::3]))
    return keypoint


def unnormalize_keypoint(keypoint: np.ndarray,
                         img_size: (int, int)):
    keypoint = np.array(keypoint)
    keypoint[0::3] = list(map(lambda x: int(x * img_size[0]), keypoint[0::3]))
    keypoint[1::3] = list(map(lambda x: int(x * img_size[1]), keypoint[1::3]))
    keypoint[2::3] = list(map(lambda x: 1 if x > 0.5 else 0, keypoint[2::3]))
    return np.array(keypoint, dtype=np.int)