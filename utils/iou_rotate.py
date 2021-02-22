import jittor as jt 

CUDA_HEADER=r'''
#undef out
#include <executor.h>
#include <vector>
#include <iostream>
#include <cmath>
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
using namespace std;

int const threadsPerBlock = 1024;

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

__device__ inline bool equal(float const * const box1, float const * const box2){
  for(int i=0;i<5;i++){
    if (fabs(box1[i]-box2[i])>1e-5)
        return false;
  }
  return true;
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

  float angle = region[4];
  float a_cos = cos(angle/180.0*3.1415926535);
  float a_sin = sin(angle/180.0*3.1415926535);

  float ctr_x = region[0];
  float ctr_y = region[1];

  float w = region[2];
  float h = region[3];

  float pts_x[4];
  float pts_y[4];

  pts_x[0] = - w / 2;
  pts_x[1] = w / 2;
  pts_x[2] = w / 2;
  pts_x[3] = - w / 2;

  pts_y[0] = - h / 2;
  pts_y[1] = - h / 2;
  pts_y[2] = h / 2;
  pts_y[3] = h / 2;

  for(int i = 0;i < 4;i++) {
    pts[7 - 2 * i - 1] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
    pts[7 - 2 * i] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;
   
  }

}


__device__ inline float inter(float const * const region1, float const * const region2) {

  float pts1[8];
  float pts2[8];
  float int_pts[16];
  int num_of_inter;

  convert_region(pts1, region1);
  convert_region(pts2, region2);

  num_of_inter = inter_pts(pts1, pts2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);
  
  
}

__device__ inline float devRotateIoU(float const * const region1, float const * const region2) {
  if(equal(region1,region2))
      return 1.0;
  float area1 = region1[2] * region1[3];
  float area2 = region2[2] * region2[3];
  float area_inter = inter(region1, region2);
  return area_inter / (area1 + area2 - area_inter);
  
}
__global__ void rotate_iou_kernel(const int n_boxes,const float *dev_boxes1, const float *dev_boxes2,float *dev_iou) {
  const int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i < n_boxes) {
    const float *cur_box1 = dev_boxes1 + i * 5;
    const float *cur_box2 = dev_boxes2 + i * 5;
    dev_iou[i] = devRotateIoU(cur_box1, cur_box2);
  }
}
'''

CUDA_SRC=r'''
@alias(boxes1,in0)
@alias(boxes2,in1)
@alias(ious,out)
const int n_boxes = boxes1_shape0;
const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
rotate_iou_kernel<<<col_blocks,threadsPerBlock>>>(n_boxes,boxes1_p,boxes2_p,ious_p);
'''

def iou_rotate(boxes1,boxes2):
    assert boxes1.shape == boxes2.shape and boxes1.ndim==2
    if boxes1.shape[0]==0:
        return jt.zeros((0,),dtype="float32")
    return jt.code((boxes1.shape[0],),"float32",[boxes1,boxes2],cuda_header=CUDA_HEADER,cuda_src=CUDA_SRC)


def test():
    jt.flags.use_cuda=1
    boxes1 = jt.array([[1,1,2,2,0.1],[1,1,2,2,0.2],[1,1,2,2,0.2]]).float32()
    boxes2 = jt.array([[1,1,2,3,1],[1,1,2,2,0.3],[1,1,2,2,0.2]]).float32()
    ious = iou_rotate(boxes1=boxes1,boxes2=boxes2)
    print(ious)

if __name__ == "__main__":
    test()