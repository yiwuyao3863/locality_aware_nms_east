# Author: Yiwu Yao
# Date: 2019-05-29
# Description: GPU polygon nms

from rotate_polygon_nms import rotate_gpu_nms
import numpy as np

def rotate_nms(dets, thresh, device_id=0):
    """GPU NMS implementations."""
    if dets.shape[0] == 0:
        return np.array([])
    return rotate_gpu_nms(dets, thresh, device_id=device_id)
