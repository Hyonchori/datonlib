
import torch
import time


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_maximum_suppression(prediction, conf_thr=0.5, iou_thr=0.45,
                            classes=None, agnostic=False, multi_label=False, labels=()):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thr

    assert 0 <= conf_thr <= 1, f'Invalid Confidence threshold {conf_thr}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thr <= 1, f'Invalid IoU {iou_thr}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 4096
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for i, x in enumerate(prediction):
        x = x[xc[i]]
        if labels and len(labels[i]):
            l = labels[i]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])