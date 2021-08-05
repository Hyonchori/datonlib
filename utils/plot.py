# Visualization utils

import cv2
import numpy as np


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False, norm=False):
        c = self.palette[int(i) % self.n] if not norm else [x/255. for x in self.palette[int(i) % self.n]]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()


def letterbox(
        img: np.ndarray,
        new_shape: tuple = (640, 640),
        color: tuple = (114, 114, 114),
        scaleup: bool = True,
        auto: bool = True,
        stride: int = 32):
    # letterboxing img for preventing ratio during downsampling stage in model (stride)
    img_size = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / img_size[0], new_shape[1] / img_size[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    #The modified size of the original image when scaled and stretched to fit the new size.
    new_unpad = int(round(img_size[1] * r)), int(round(img_size[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if img_size[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def draw_one_bbox(
        img: np.ndarray,
        bbox: list,
        color: tuple=(128, 128, 128),
        thick: int=3,
        get_return: bool=False):
    if get_return:
        img = img.copy()
    line_thickness = thick or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, p1, p2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    label = bbox[-2] if type(bbox[-2]) == str else str(bbox[-1])
    tf = max(line_thickness - 1, 1)
    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness/3, thickness=tf)[0]
    p2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
    cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (p1[0], p1[1]-2), 0, line_thickness/3, [225, 255, 255], thickness=thick, lineType=cv2.LINE_AA)
    if get_return:
        return img


def draw_bboxes(
        img: np.ndarray,
        bboxes: list,
        color_norm: bool=False):
    img = img.copy()
    for bbox in bboxes:
        cls = int(bbox[-1])
        color = colors(cls, True, color_norm)
        draw_one_bbox(img, bbox, color, thick=1)
    return img


coco_keypoint_label = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                       "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_hand", "right_hand",
                       "left_waist", "right_waist", "left_knee", "right_knee", "left_foot", "right_foot"]
aihub_keypoint_label = ["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle",
                        "pelvis", "throax", "neck", "head_top",
                        "right_wrist", "right_elbow", "right_shoulder",
                        "left_shoulder", "left_elbow", "left_wrist"]
keypoint_label_dict = {"coco": coco_keypoint_label,
                       "aihub": aihub_keypoint_label}
def draw_one_keypoint(
        img: np.ndarray,
        keypoint: list,
        color: tuple = (255, 255, 255),
        radius: int = 2,
        get_return: bool = False,
        connection_type: str = "coco",
        letterbox_offset: list = None):
    # Visualize COCO format keypoints
    # letterbox_offset: [ratio, dw, dh]

    if get_return:
        img = img.copy()

    xs = keypoint[0::3]
    ys = keypoint[1::3]
    vs = keypoint[2::3]

    keypoint_label = keypoint_label_dict[connection_type]
    kp = {key: [int(x), int(y), v] for (key, x, y, v) in zip(keypoint_label, xs, ys, vs)}

    if connection_type == "coco":
        connect_keypoint_coco(img, kp)
    elif connection_type == "aihub":
        connect_keypoint_aihub(img, kp)

    for key, value in kp.items():
        x = value[0]
        y = value[1]
        v = value[2]
        if v > 1:
            cv2.circle(img, (x, y), radius, color, -1)
    if get_return:
        return img


def connect_keypoint_coco(img, kp):
    connect_keypoint(img, kp, ["right_ear", "right_eye"])
    connect_keypoint(img, kp, ["right_eye", "nose"])
    connect_keypoint(img, kp, ["left_ear", "left_eye"])
    connect_keypoint(img, kp, ["left_eye", "nose"])
    connect_keypoint(img, kp, ["right_ear", "right_shoulder"])
    connect_keypoint(img, kp, ["left_ear", "left_shoulder"])
    connect_keypoint(img, kp, ["right_shoulder", "right_elbow"])
    connect_keypoint(img, kp, ["right_elbow", "right_hand"])
    connect_keypoint(img, kp, ["left_shoulder", "left_elbow"])
    connect_keypoint(img, kp, ["left_elbow", "left_hand"])
    connect_keypoint(img, kp, ["right_shoulder", "right_waist"])
    connect_keypoint(img, kp, ["right_waist", "right_knee"])
    connect_keypoint(img, kp, ["right_knee", "right_foot"])
    connect_keypoint(img, kp, ["left_shoulder", "left_waist"])
    connect_keypoint(img, kp, ["left_waist", "left_knee"])
    connect_keypoint(img, kp, ["left_knee", "left_foot"])


def connect_keypoint_aihub(img, kp):
    connect_keypoint(img, kp, ["right_ankle", "right_knee"])
    connect_keypoint(img, kp, ["right_hip", "right_knee"])
    connect_keypoint(img, kp, ["right_hip", "pelvis"])
    connect_keypoint(img, kp, ["left_ankle", "left_knee"])
    connect_keypoint(img, kp, ["left_hip", "left_knee"])
    connect_keypoint(img, kp, ["left_hip", "pelvis"])
    connect_keypoint(img, kp, ["right_wrist", "right_elbow"])
    connect_keypoint(img, kp, ["right_shoulder", "right_elbow"])
    connect_keypoint(img, kp, ["neck", "right_shoulder"])
    connect_keypoint(img, kp, ["left_wrist", "left_elbow"])
    connect_keypoint(img, kp, ["left_shoulder", "left_elbow"])
    connect_keypoint(img, kp, ["neck", "left_shoulder"])
    connect_keypoint(img, kp, ["neck", "head_top"])
    connect_keypoint(img, kp, ["neck", "throax"])
    connect_keypoint(img, kp, ["pelvis", "throax"])



def connect_keypoint(
        img: np.ndarray,
        keypoint: dict,
        key_pair: [str, str],
        color: tuple = (100, 100, 250),
        thick: int = 1):
    kp1 = keypoint[key_pair[0]]
    kp2 = keypoint[key_pair[1]]
    if kp1[2] == 0 or kp2[2] == 0:
        pass
    else:
        if kp1[2] == 2 or kp2[2] == 2:
            cv2.line(img, kp1[:2], kp2[:2], color, thickness=thick, lineType=cv2.LINE_AA)


def draw_keypoints(
        img: np.ndarray,
        keypoints: list):
    img = img.copy()
    for keypoint in keypoints:
        draw_one_keypoint(img, keypoint)
    return img
