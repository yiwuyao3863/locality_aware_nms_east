// Author: Yiwu Yao
// Date: 2019-06-05
// Description: modified for polygon-nms: the shape of input array is N*9,
// containing coordinates of the 4 vertices and score. The NMS from MXnet
// MultiBoxDetection is used.

#include "rotate_gpu_nms.hpp"
#include <vector>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

int const threadsPerBlock = 1024;

__device__ inline float sqr_d(float x) { return x * x; }

__device__ inline float trangle_area(float * a, float * b, float * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}

__device__ inline float area(float * int_pts, int num_of_inter) {

  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}

__device__ inline float trangle_area_rect(const float * a, const float * b, const float * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}

__device__ inline float area_rect(const float * int_pts, int num_of_inter) {

  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area_rect(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}

__device__ inline void reorder_pts(float * int_pts, int num_of_inter) {

  if(num_of_inter > 0) {

    float center[2];
    center[0] = 0.0;
    center[1] = 0.0;

    for(int i = 0;i < num_of_inter;i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    float vs[16];
    float v[2];
    float d;
    for(int i = 0;i < num_of_inter;i++) {
      v[0] = int_pts[2 * i]-center[0];
      v[1] = int_pts[2 * i + 1]-center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if(v[1] < 0) {
        v[0]= - 2 - v[0];
      }
      vs[i] = v[0];
    }

    float temp,tx,ty;
    int j;
    for(int i=1;i<num_of_inter;++i){
      if(vs[i-1]>vs[i]){
        temp = vs[i];
        tx = int_pts[2*i];
        ty = int_pts[2*i+1];
        j=i;
        while(j>0&&vs[j-1]>temp){
          vs[j] = vs[j-1];
          int_pts[j*2] = int_pts[j*2-2];
          int_pts[j*2+1] = int_pts[j*2-1];
          j--;
        }
        vs[j] = temp;
        int_pts[j*2] = tx;
        int_pts[j*2+1] = ty;
      }
    }
  }

}
__device__ inline bool inter2line(float * pts1, float *pts2, int i, int j, float * temp_pts) {

  float a[2];
  float b[2];
  float c[2];
  float d[2];

  float area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);

  if(area_abc * area_abd >= 0) {
    return false;
  }

  area_cda = trangle_area(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= 0) {
    return false;
  }
  float t = area_cda / (area_abd - area_abc);

  float dx = t * (b[0] - a[0]);
  float dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}

__device__ inline bool in_rect(float pt_x, float pt_y, float * pts) {

  float ab[2];
  float ad[2];
  float ap[2];

  float abab;
  float abap;
  float adad;
  float adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];

  return abab >= abap and abap >= 0 and adad >= adap and adap >= 0;
}

__device__ inline int inter_pts(float * pts1, float * pts2, float * int_pts) {

  int num_of_inter = 0;

  for(int i = 0;i < 4;i++) {
    if(in_rect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
     if(in_rect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  float temp_pts[2];

  for(int i = 0;i < 4;i++) {
    for(int j = 0;j < 4;j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if(has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }


  return num_of_inter;
}

__device__ inline void convert_region(float * pts , float const * const region) {

  for(int i = 0;i < 4;i++) {
    pts[7 - 2 * i - 1] = region[2*i];
    pts[7 - 2 * i] = region[2*i+1];
  }
}


__device__ inline float inter(float const * const region1, float const * const region2) {

  float pts1[8], pts2[8];
  float int_pts[16];
  int num_of_inter;

  convert_region(pts1, region1);
  convert_region(pts2, region2);

  num_of_inter = inter_pts(pts1, pts2, int_pts);
  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);

}

__device__ inline float devRotateIoU(float const * const region1, float const * const region2) {

  float area1 = area_rect(region1, 4);
  float area2 = area_rect(region2, 4);
  float area_inter = inter(region1, region2);

  return area_inter / (area1 + area2 - area_inter);

}

__global__ void rotate_nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, float *out) {
  int index = threadIdx.x;

  // define dynamic shared memory to cache all boxes
  /*
  extern __shared__ float block_boxes[]; // 4 vertices and s

  for (int i = index; i < n_boxes; i += blockDim.x) {
    for (int j = 0; j < 9; j++) {
      block_boxes[i * 9 + j] = dev_boxes[i * 9 + j];
    }
  }
  __syncthreads();
  */
  for (int i = index; i < n_boxes; i += blockDim.x) {
    // store the first bbox
    out[i * 11] = 1; // 1: valid, -1: invalid
    out[i * 11 + 10] = -1;
    for (int j = 0; j < 9; j++) {
      out[i * 11 + j + 1] = dev_boxes[i * 9 + j];
    }
  }
  __syncthreads();

  // apply NMS, from MXNet MultiBoxDetection
  for (int compare_pos = 0; compare_pos < n_boxes; ++compare_pos) {
    float compare_id = out[compare_pos * 11];
    if (compare_id < 0) continue;  // not a valid positive detection, skip
    float *compare_loc_ptr = out + compare_pos * 11 + 1;
    for (int i = compare_pos + index + 1; i < n_boxes; i += blockDim.x) {
      float class_id = out[i * 11];
      if (class_id < 0) continue;
      if (devRotateIoU(compare_loc_ptr, out + i * 11 + 1) > nms_overlap_thresh) {
        out[i * 11] = -1;
        out[i * 11 + 10] = static_cast<float>(compare_pos);
      }
    }
    __syncthreads();
  }

  // post merge
  for (int i = index; i < n_boxes; i += blockDim.x) {
    int ref = i * 11; // the reference and update one
    if (out[ref] > 0) { // if valid
      //int count = 0;
      float score_t = out[ref + 9];
      for (int k=1; k < (n_boxes-i); k+=1) {
        int p_given = (i+k)*11; // the followed one
        int invalid_i = static_cast<int>(out[p_given+10]); // if invalid, corresponding to i
        if ((out[p_given] < 0) && (invalid_i == i)){
          //if (devRotateIoU(out + ref + 1, out + p_given + 1) > nms_overlap_thresh) {
            // merge
            /*
            if ((out[p_given+7] < out[ref+7]) ) {
                out[ref+7] = out[p_given+7];
                out[ref+8] = out[p_given+8];
            }
            if ((out[p_given+1] < out[ref+1]) ) {
                out[ref+1] = out[p_given+1];
                out[ref+2] = out[p_given+2];
            }

            if ((out[p_given+5] >= out[ref+5]) ) {
                out[ref+5] = out[p_given+5];
                out[ref+6] = out[p_given+6];
            }
            if ((out[p_given+3] >= out[ref+3]) ) {
                out[ref+3] = out[p_given+3];
                out[ref+4] = out[p_given+4];
            }
            */
            float p_s = out[p_given + 9];
    				for (int v = 0; v < 4; v++) {
              out[ref+v*2+1] = (out[ref+v*2+1]*score_t + out[p_given+v*2+1]*p_s) / (score_t + p_s);
              out[ref+v*2+2] = (out[ref+v*2+2]*score_t + out[p_given+v*2+2]*p_s) / (score_t + p_s);
            }
            score_t += p_s; // update the total score
            //if (count>1024) break;
            //count++;
          //}
        }
      }
      out[ref + 9] = score_t;
    }
  }
  __syncthreads();
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

// Host routine
void _rotate_nms(float *nms_out_host, int *num_out, const float* boxes_host, const int boxes_num, const int boxes_dim,
                 float nms_overlap_thresh, int device_id) {
  _set_device(device_id);

  float* boxes_dev = NULL;
  float* out_dev = NULL;

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&out_dev,
                        boxes_num * (boxes_dim+2) * sizeof(float)));

  dim3 threads(threadsPerBlock);
  rotate_nms_kernel<<<1, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  out_dev);

  // dynamic array allocation
  float (* out_host)[11] = new float[boxes_num][11]; // 11 = boxes_dim+2
  CUDA_CHECK(cudaMemcpy(&out_host[0][0],
                        out_dev,
                        sizeof(float) * boxes_num * (boxes_dim+2),
                        cudaMemcpyDeviceToHost));

  // delete the invalid bbox
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int id = int(out_host[i][0]);
    if (id > 0) {
      for (int j = 0; j < boxes_dim; j++) {
        nms_out_host[num_to_keep * boxes_dim + j] = out_host[i][j+1];
      }
      num_to_keep++;
    }
  }
  *num_out = num_to_keep;

  // clean up
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(out_dev));
  delete[] out_host;
}
