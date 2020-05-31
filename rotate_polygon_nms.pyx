import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "rotate_gpu_nms.hpp":
    void _rotate_nms(np.float32_t*, int*, np.float32_t*, int, int, float, int)

def rotate_gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float_t thresh, np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets = dets[:, :]
    cdef np.ndarray[np.float32_t, ndim=2] \
        nms_out = np.zeros((boxes_num, boxes_dim), dtype=np.float32)
    cdef int num_out
    _rotate_nms(&nms_out[0, 0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)
    nms_out = nms_out[:num_out, :]
    return nms_out
