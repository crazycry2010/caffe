#include "caffe/util/local_conv.hpp"

#include <iostream>
#include<iomanip>
using namespace std;

namespace caffe {
__device__ void
b_set_base(const int* b_dim, int* b_base, int n, int c, int h, int w, int g) {
  b_base[0] = n;
  b_base[1] = c;
  b_base[2] = h;
  b_base[3] = w;
//  if (b_base[3] >= b_dim[3]) {
//    b_base[2] += b_base[3] / b_dim[3];
//    b_base[3] = b_base[3] % b_dim[3];
//  }
  b_base[4] = g;
}
__device__ void
b_add_base(const int* b_dim, int* b_base, int n, int c, int h, int w) {
  b_base[0] += n;
  b_base[1] += c;
  b_base[2] += h;
  b_base[3] += w;
//  if (b_base[3] >= b_dim[3]) {
//    b_base[2] += b_base[3] / b_dim[3];
//    b_base[3] = b_base[3] % b_dim[3];
//  }
}
__device__ bool
b_check(const int* b_dim, int* b_base, int n, int c, int h, int w) {
  bool ret = (b_base[0] + n < b_dim[0]) && (b_base[0] + n >= 0);
  ret = ret && (b_base[1] + c < b_dim[1]) && (b_base[1] + c >= 0);
  ret = ret && (b_base[2] + h < b_dim[2]) && (b_base[2] + h >= 0);
  ret = ret && (b_base[3] + w < b_dim[3]) && (b_base[3] + w >= 0);
  ret = ret && (b_base[1] + c >= b_base[4] * (b_dim[1] / b_dim[4])) &&
      (b_base[1] + c < (b_base[4] + 1) * (b_dim[1] / b_dim[4]));
  return ret;
}
__device__ int
b_idx(const int* b_dim, int* b_base, int n, int c, int h, int w) {
  int ret = (b_base[0] + n) * b_dim[1] * b_dim[2] * b_dim[3];
  ret += (b_base[1] + c) * b_dim[2] * b_dim[3];
  ret += (b_base[2] + h) * b_dim[3];
  ret += (b_base[3] + w);
  return ret;
}

__device__ void
t_set_base(const int* t_dim, int* t_base, int n, int no, int ho, int wo, int g) {
  t_base[0] = n;
  t_base[1] = no;
  t_base[2] = ho;
  t_base[3] = wo;
//  if (t_base[3] >= t_dim[3]) {
//    t_base[2] += t_base[3] / t_dim[3];
//    t_base[3] = t_base[3] % t_dim[3];
//  }
  t_base[4] = g;
}
__device__ void
t_add_base(const int* t_dim, int* t_base, int n, int no, int ho, int wo) {
  t_base[0] += n;
  t_base[1] += no;
  t_base[2] += ho;
  t_base[3] += wo;
//  if (t_base[3] >= t_dim[3]) {
//    t_base[2] += t_base[3] / t_dim[3];
//    t_base[3] = t_base[3] % t_dim[3];
//  }
}
__device__ bool
t_check(const int* t_dim, int* t_base, int n, int no, int ho, int wo) {
  bool ret = (t_base[0] + n < t_dim[0]) && (t_base[0] + n >= 0);
  ret = ret && (t_base[1] + no < t_dim[1]) && (t_base[1] + no >= 0);
  ret = ret && (t_base[2] + ho < t_dim[2]) && (t_base[2] + ho >= 0);
  ret = ret && (t_base[3] + wo < t_dim[3]) && (t_base[3] + wo >= 0);
  ret = ret && (t_base[1] + no >= t_base[4] * (t_dim[1] / t_dim[4])) &&
      (t_base[1] + no < (t_base[4] + 1) * (t_dim[1] / t_dim[4]));
  return ret;
}
__device__ int
t_idx(const int* t_dim, int* t_base, int n, int no, int ho, int wo) {
  int ret = (t_base[0] + n) * t_dim[1] * t_dim[2] * t_dim[3];
  ret += (t_base[1] + no) * t_dim[2] * t_dim[3];
  ret += (t_base[2] + ho) * t_dim[3];
  ret += (t_base[3] + wo);
  return ret;
}

__device__ void
w_set_base(const int* w_dim, int* w_base, int no, int c, int kh, int kw, int ho, int wo, int g) {
  w_base[0] = no;
  w_base[1] = c;
  w_base[2] = kh;
  w_base[3] = kw;
//  if (w_base[3] >= w_dim[3]) {
//    w_base[2] += w_base[3] / w_dim[3];
//    w_base[3] = w_base[3] % w_dim[3];
//  }
  w_base[4] = ho;
  w_base[5] = wo;
//  if (w_base[5] >= w_dim[5]) {
//    w_base[4] += w_base[5] / w_dim[5];
//    w_base[5] = w_base[5] % w_dim[5];
//  }
  w_base[6] = g;
}
__device__ void
w_add_base(const int* w_dim, int* w_base, int no, int c, int kh, int kw, int ho, int wo) {
  w_base[0] += no;
  w_base[1] += c;
  w_base[2] += kh;
  w_base[3] += kw;
//  if (w_base[3] >= w_dim[3]) {
//    w_base[2] += w_base[3] / w_dim[3];
//    w_base[3] = w_base[3] % w_dim[3];
//  }
  w_base[4] += ho;
  w_base[5] += wo;
//  if (w_base[5] >= w_dim[5]) {
//    w_base[4] += w_base[5] / w_dim[5];
//    w_base[5] = w_base[5] % w_dim[5];
//  }
}
__device__ bool
w_check(const int* w_dim, int* w_base, int no, int c, int kh, int kw, int ho, int wo) {
  bool ret = (w_base[0] + no < w_dim[0]) && (w_base[0] + no >= 0);
  ret = ret && (w_base[1] + c < w_dim[1]) && (w_base[1] + c >= 0);
  ret = ret && (w_base[2] + kh < w_dim[2]) && (w_base[2] + kh >= 0);
  ret = ret && (w_base[3] + kw < w_dim[3]) && (w_base[3] + kw >= 0);
  ret = ret && (w_base[4] + ho < w_dim[4]) && (w_base[4] + ho >= 0);
  ret = ret && (w_base[5] + wo < w_dim[5]) && (w_base[5] + wo >= 0);
  ret = ret && (w_base[0] + no >= w_base[6] * (w_dim[0] / w_dim[6])) &&
      (w_base[0] + no < (w_base[6] + 1) * (w_dim[0] / w_dim[6]));
  return ret;
}
__device__ int
w_idx(const int* w_dim, int* w_base, int no, int c, int kh, int kw, int ho, int wo) {
  int ret = (w_base[0] + no) * w_dim[1] * w_dim[2] * w_dim[3] * w_dim[4] * w_dim[5];
  ret += (w_base[1] + c) * w_dim[2] * w_dim[3] * w_dim[4] * w_dim[5];
  ret += (w_base[2] + kh) * w_dim[3] * w_dim[4] * w_dim[5];
  ret += (w_base[3] + kw) * w_dim[4] * w_dim[5];
  ret += (w_base[4] + ho) * w_dim[5];
  ret += (w_base[5] + wo);
  return ret;
}
/*
 * top = bottom * weights
 * bottom:  num, group*(channels/group), height, width
 * weight:  group*(num_output/group), channels/group, kernel_h*kernel_w, height_out*width_out
 * top:     num, group*(num_output/group), height_out, width_out
 *
 *
 * grid:    DIVUP(num, BX*num_per_thread),
 *          group*DIVUP(num_output/group, BY*num_output_per_thread)
 *          height_out*width_out
 * block:   BX, BY, BZ
 * thread:  num_per_thread, num_output_per_thread, 1
 *          channels_per_thread(0 ~ channels/group)
 *          pix_per_thread(0 ~ kernel_h*kernel_w)
 * ==> top    (num_idx,num_output_idx(group_idx),pix_out_idx)
 * num_idx:           blockIdx.x*BX*num_per_thread
 * blocks_per_group:  DIVUP(num_output/group, BY*num_output_per_thread)
 * group_idx:         blockIdx.y/blocks_per_group
 * num_output_idx:    group_idx*(num_output/group) + (blockIdx.y%blocks_per_group)*BY*num_output_per_thread
 * channels_idx:      group_idx*(channels/group)
 * pix_out_idx:       blockIdx.z
 * ==> bottom (num_idx, channels_idx(group_idx), pix_idx)
 * num_idx:           num_idx
 * channels_idx:      group_idx*(channels/group) + (0~channels/group)
 * pix_idx:           stride_h-pad_h
 * hi = ho*stride_h-pad_h ~ ho*stride_h-pad_h + kernel_h
 *              wi = wo*stride_w-pad_w ~ wo*stride_w-pad_w + kernel_w
 * ==> weight
 * c_out_idx:   &c_out_idx
 * c_in:        &c_in
 * pix_w_idx:   kh = 0 ~ kernel_h
 *              kw = 0 ~ kernel_w
 */
template<typename Dtype, int BX, int BY,
    int num_per_thread, int num_output_per_thread,
    int channels_per_thread, int pix_per_thread>
__global__ void BxW2T_imp(const Dtype* bottom, const Dtype* weights, Dtype* top,
                          const int num, const int channels, const int height, const int width,
                          /*const int num, */const int num_output, const int height_out, const int width_out,
                          /*const int num_output, const int channels/group,*/
                          const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                          const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                          const int group,
                          const float scale_targets, const float scale_outputs,
                          const bool conv) {
  __shared__ Dtype sh_weights[channels_per_thread * pix_per_thread][BY * num_output_per_thread];
  __shared__ Dtype sh_bottom[channels_per_thread * pix_per_thread][BX * num_per_thread];

  const int num_idx = blockIdx.x * (BX * num_per_thread);
  const int blocks_per_group = DIVUP(num_output/group, BY*num_output_per_thread);
  const int group_idx = blockIdx.y / blocks_per_group;
  const int num_output_idx = group_idx * (num_output / group) + (blockIdx.y % blocks_per_group) * BY * num_output_per_thread;
  //const int pix_out_idx = blockIdx.z;
  const int pix_t_h = blockIdx.z / width_out;
  const int pix_t_w = blockIdx.z % width_out;
  const int t_dim[5] = { num, num_output, height_out, width_out, group };
  int t_base[5];
  t_set_base(t_dim, t_base, num_idx, num_output_idx, pix_t_h, pix_t_w, group_idx);

  const int channels_idx = group_idx * (channels / group);
  const int pix_b_h = -pad_h + pix_t_h * stride_h;
  const int pix_b_w = -pad_w + pix_t_w * stride_w;
  const int b_dim[5] = { num, channels, height, width, group };
  int b_base[5];
  b_set_base(b_dim, b_base, num_idx, channels_idx, pix_b_h, pix_b_w, group_idx);

  const int w_dim[7] = { num_output, channels / group, kernel_h, kernel_w, height_out, width_out, group };
  int w_base[7];
  w_set_base(w_dim, w_base, num_output_idx, 0, 0, 0, pix_t_h, pix_t_w, group_idx);

  const int tidx = threadIdx.y * BX + threadIdx.x;

  float prod[num_per_thread][num_output_per_thread];
  for (int f = 0; f < num_per_thread; f++) {
    for (int g = 0; g < num_output_per_thread; g++) {
      prod[f][g] = 0;
    }
  }

  for (int lc = 0; lc < channels / group; lc += channels_per_thread) {
    for (int lp = 0; lp < kernel_h * kernel_w; lp += pix_per_thread) {

      // load sh_bottom
      for (int i = 0;
          i < channels_per_thread * pix_per_thread * BX * num_per_thread;
          i += BX * BY) {
        //current:i+tidx
        const int c = (i + tidx) / (pix_per_thread * BX * num_per_thread);
        const int p = ((i + tidx) % (pix_per_thread * BX * num_per_thread)) / (BX * num_per_thread);
        const int n = (i + tidx) % (BX * num_per_thread);
        // lc+c,lp+p,n
        const int pix_k_h = (lp + p) / kernel_w;
        const int pix_k_w = (lp + p) % kernel_w;
        if (i + tidx < channels_per_thread * pix_per_thread * BX * num_per_thread) {
          if ((lp + p < kernel_h * kernel_w) && (lc + c < channels / group) &&
              b_check(b_dim, b_base, n, lc + c, pix_k_h, pix_k_w)) {
            sh_bottom[c * pix_per_thread + p][n] = bottom[b_idx(b_dim, b_base, n, lc + c, pix_k_h, pix_k_w)];
          } else {
            sh_bottom[c * pix_per_thread + p][n] = 0;
          }
        }
      }

      // load sh_weight
      for (int i = 0;
          i < channels_per_thread * pix_per_thread * BY * num_output_per_thread;
          i += BX * BY) {
        //current:i+tidx
        const int c = (i + tidx) / (pix_per_thread * BY * num_output_per_thread);
        const int p = ((i + tidx) % (pix_per_thread * BY * num_output_per_thread)) / (BY * num_output_per_thread);
        const int no = (i + tidx) % (BY * num_output_per_thread);

        const int pix_k_h = (lp + p) / kernel_w;
        const int pix_k_w = (lp + p) % kernel_w;

        if (i + tidx < channels_per_thread * pix_per_thread * BY * num_output_per_thread) {
          if ((lp + p < kernel_h * kernel_w) && (lc + c < channels / group) &&
              w_check(w_dim, w_base, no, lc + c, pix_k_h, pix_k_w, 0, 0)) {
            sh_weights[c * pix_per_thread + p][no] = weights[w_idx(w_dim, w_base, no, lc + c, pix_k_h, pix_k_w, 0, 0)];
          } else {
            sh_weights[c * pix_per_thread + p][no] = 0;
          }
        }
      }
      __syncthreads();

      for (int f = 0; f < num_per_thread; f++) {
        for (int g = 0; g < num_output_per_thread; g++) {
          for (int i = 0; i < channels_per_thread * pix_per_thread; i++) {
            prod[f][g] += sh_bottom[i][f * BX + threadIdx.x] * sh_weights[i][g * BY + threadIdx.y];
          }
        }
      }
      __syncthreads();
    }
  }

  for (int f = 0; f < num_per_thread; f++) {
    for (int g = 0; g < num_output_per_thread; g++) {
      if (t_check(t_dim, t_base, f * BX + threadIdx.x, g * BY + threadIdx.y, 0, 0)) {
        top[t_idx(t_dim, t_base, f * BX + threadIdx.x, g * BY + threadIdx.y, 0, 0)] =
            scale_targets * top[t_idx(t_dim, t_base, f * BX + threadIdx.x, g * BY + threadIdx.y, 0, 0)]
                + scale_outputs * prod[f][g];
      }
    }
  }
}

template<typename Dtype>
void BxW2T(const Dtype* bottom, const Dtype* weights, Dtype* top,
           const int num, const int channels, const int height, const int width,
           /*const int num, */const int num_output, const int height_out, const int width_out,
           /*const int num_output, const int channels/group,*/
           const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
           const int pad_h, const int pad_w, const int stride_h, const int stride_w,
           const int group,
           const float scale_targets, const float scale_outputs,
           const bool conv) {

  const int BX = 32;
  const int BY = 4;
  const int BZ = 1;
  const int num_per_thread = 4;
  const int num_output_per_thread = 4;

  //const int channels_per_thread = 4;
  //const int pix_per_thread = 4;
  dim3 threads(BX, BY, BZ);
  dim3 blocks = dim3(DIVUP(num, BX * num_per_thread),
                     group * DIVUP(num_output / group, BY * num_output_per_thread),
                     height_out * width_out);

  cudaFuncSetCacheConfig(BxW2T_imp<Dtype, 32, 4, 4, 4, 4, 4>, cudaFuncCachePreferShared);
  BxW2T_imp<Dtype, 32, 4, 4, 4, 4, 4> <<<blocks, threads>>>(bottom, weights, top,
                                                           num, channels, height, width,
                                                           num_output, height_out, width_out,
                                                           kernel_h, kernel_w,
                                                           pad_h, pad_w, stride_h, stride_w,
                                                           group,
                                                           scale_targets, scale_outputs,
                                                           conv);
}

template
void BxW2T<float>(const float* bottom, const float* weights, float* top,
                  const int num, const int channels, const int height, const int width,
                  /*const int num, */const int num_output, const int height_out, const int width_out,
                  /*const int num_output, const int channels/group,*/
                  const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                  const int group,
                  const float scale_targets, const float scale_outputs,
                  const bool conv);
template
void BxW2T<double>(const double* bottom, const double* weights, double* top,
                   const int num, const int channels, const int height, const int width,
                   /*const int num, */const int num_output, const int height_out, const int width_out,
                   /*const int num_output, const int channels/group,*/
                   const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                   const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                   const int group,
                   const float scale_targets, const float scale_outputs,
                   const bool conv);

/*
 * weight_diff = bottom_data * top_diff
 * bottom:  num, group*(channels/group), height, width
 * top:     num, group*(num_output/group), height_out, width_out
 * weight:  group*(num_output/group), channels/group, kernel_h*kernel_w, height_out*width_out
 *
 *
 * grid:    group*DIVUP(num_output/group, BX*num_output_per_thread),
 *          DIVUP(channels/group, BY*channels_per_thread),
 *          (kernel_h*kernel_w)*(height_out*width_out)
 * block:   BX, BY, BZ
 * thread:  num_output_per_thread, channels_per_thread, 1
 *          num_per_block(0 ~ num)
 *
 */
template<typename Dtype, int BX, int BY,
    int num_output_per_thread, int channels_per_thread,
    int num_per_block>
__global__ void BxT2W_imp(const Dtype* bottom_data, const Dtype* top_diff, Dtype* weight_diff,
                          const int num, const int channels, const int height, const int width,
                          /*const int num, */const int num_output, const int height_out, const int width_out,
                          /*const int num_output, const int channels/group,*/
                          const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                          const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                          const int group,
                          const float scale_targets, const float scale_outputs,
                          const bool conv) {
  __shared__ Dtype sh_bottom[channels_per_thread * BY][num_per_block];
  __shared__ Dtype sh_top_diff[num_output_per_thread * BX][num_per_block];

  const int num_output_per_group = DIVUP(num_output/group, BX*num_output_per_thread);
  const int group_idx = blockIdx.x / num_output_per_group;
  const int num_output_idx = group_idx * (num_output / group) + (blockIdx.x % num_output_per_group) * BX * num_output_per_thread;

  const int channels_ingroup_idx = blockIdx.y * (BY * channels_per_thread);

  const int pix_t_idx = blockIdx.z % (height_out * width_out);
  const int pix_k_idx = blockIdx.z / (height_out * width_out);
  const int pix_t_h = pix_t_idx / width_out;
  const int pix_t_w = pix_t_idx % width_out;
  const int pix_k_h = pix_k_idx / kernel_w;
  const int pix_k_w = pix_k_idx % kernel_w;

  const int w_dim[7] = { num_output, channels / group, kernel_h, kernel_w, height_out, width_out, group };
  int w_base[7];
  w_set_base(w_dim, w_base, num_output_idx, channels_ingroup_idx, pix_k_h, pix_k_w, pix_t_h, pix_t_w, group_idx);

  const int channels_idx = group_idx * (num_output / group) + channels_ingroup_idx;
  const int b_dim[5] = { num, channels, height, width, group };
  const int pix_b_h = -pad_h + pix_t_h * stride_h + pix_k_h;
  const int pix_b_w = -pad_w + pix_t_w * stride_w + pix_k_w;
  int b_base[5];
  b_set_base(b_dim, b_base, 0, channels_idx, pix_b_h, pix_b_w, group_idx);

  const int t_dim[5] = { num, num_output, height_out, width_out, group };
  int t_base[5];
  t_set_base(t_dim, t_base, 0, num_output_idx, pix_t_h, pix_t_w, group_idx);

  const int tidx = BX * threadIdx.y + threadIdx.x;

  Dtype prod[channels_per_thread][num_output_per_thread];
  for (int c = 0; c < channels_per_thread; c++) {
    for (int n = 0; n < num_output_per_thread; n++) {
      prod[c][n] = 0;
    }
  }
  for (int ln = 0; ln < num; ln += num_per_block) {

    // load sh_bottom
    for (int i = 0;
        i < channels_per_thread * BY * num_per_block;
        i += BX * BY) {
      //current:i+tidx
      const int c = (i + tidx) / num_per_block;
      const int n = (i + tidx) % num_per_block;
      if (i + tidx < channels_per_thread * BY * num_per_block) {
        if ((ln + n < num) &&
            b_check(b_dim, b_base, ln + n, c, 0, 0)) {
          sh_bottom[c][n] = bottom_data[b_idx(b_dim, b_base, ln + n, c, 0, 0)];
        } else {
          sh_bottom[c][n] = 0;
        }
      }
    }

    // load sh_top_diff
    for (int i = 0;
        i < num_output_per_thread * BX * num_per_block;
        i += BX * BY) {
      //current:i+tidx
      const int no = (i + tidx) / num_per_block;
      const int n = (i + tidx) % num_per_block;
      if (i + tidx < num_output_per_thread * BX * num_per_block) {
        if ((ln + n < num) &&
            t_check(t_dim, t_base, ln + n, no, 0, 0)) {
          sh_top_diff[no][n] = top_diff[t_idx(t_dim, t_base, ln + n, no, 0, 0)];
        } else {
          sh_top_diff[no][n] = 0;
        }
      }
    }

    __syncthreads();
    // compute
    for (int c = 0; c < channels_per_thread; c++) {
      for (int n = 0; n < num_output_per_thread; n++) {
        for (int i = 0; i < num_per_block; i++) {
          prod[c][n] += sh_bottom[threadIdx.y + c * BY][i] * sh_top_diff[threadIdx.x + n * BX][i];
        }
      }
    }
    __syncthreads();
  }

  for (int c = 0; c < channels_per_thread; c++) {
    for (int n = 0; n < num_output_per_thread; n++) {
      if (w_check(w_dim, w_base, threadIdx.x + n * BX, threadIdx.y + c * BY, 0, 0, 0, 0)) {
        weight_diff[w_idx(w_dim, w_base, threadIdx.x + n * BX, threadIdx.y + c * BY, 0, 0, 0, 0)] =
            scale_targets * weight_diff[w_idx(w_dim, w_base, threadIdx.x + n * BX, threadIdx.y + c * BY, 0, 0, 0, 0)]
                + scale_outputs * prod[c][n];
      }
    }
  }
}
template<typename Dtype>
void BxT2W(const Dtype* bottom_data, const Dtype* top_diff, Dtype* weight_diff,
           const int num, const int channels, const int height, const int width,
           /*const int num, */const int num_output, const int height_out, const int width_out,
           /*const int num_output, const int channels/group,*/
           const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
           const int pad_h, const int pad_w, const int stride_h, const int stride_w,
           const int group,
           const float scale_targets, const float scale_outputs,
           const bool conv) {
  int BX = 32;
  int BY = 4;
  int BZ = 1;
  int num_output_per_thread = 4;
  int channels_per_thread = 4;
  //int num_per_block = 4;

  dim3 threads(BX, BY, BZ);
  dim3 blocks = dim3(group * DIVUP(num_output/group, BX*num_output_per_thread),
                     DIVUP(channels/group, BY*channels_per_thread),
                     (kernel_h * kernel_w) * (height_out * width_out));
  cudaFuncSetCacheConfig(BxT2W_imp<Dtype, 32, 4, 4, 4, 4>, cudaFuncCachePreferShared);
  BxT2W_imp<Dtype, 32, 4, 4, 4, 4> <<<blocks, threads>>>(bottom_data, top_diff, weight_diff,
                                                        num, channels, height, width,
                                                        num_output, height_out, width_out,
                                                        kernel_h, kernel_w,
                                                        pad_h, pad_w, stride_h, stride_w,
                                                        group,
                                                        scale_targets, scale_outputs,
                                                        conv);
}
template
void BxT2W<float>(const float* bottom_data, const float* top_diff, float* weight_diff,
                  const int num, const int channels, const int height, const int width,
                  /*const int num, */const int num_output, const int height_out, const int width_out,
                  /*const int num_output, const int channels/group,*/
                  const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                  const int group,
                  const float scale_targets, const float scale_outputs,
                  const bool conv);
template
void BxT2W<double>(const double* bottom_data, const double* top_diff, double* weight_diff,
                   const int num, const int channels, const int height, const int width,
                   /*const int num, */const int num_output, const int height_out, const int width_out,
                   /*const int num_output, const int channels/group,*/
                   const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                   const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                   const int group,
                   const float scale_targets, const float scale_outputs,
                   const bool conv);

/*
 * bottom_diff = top_diff * weight
 * top:     num, group*(num_output/group), height_out, width_out
 * weight:  group*(num_output/group), channels/group, kernel_h*kernel_w, height_out*width_out
 * bottom:  num, group*(channels/group), height, width
 *
 *
 * grid:    DIVUP(num, BX*num_per_thread),
 *          group*DIVUP(channels/group, BY*channels_per_thread),
 *          height, width
 * block:   BX, BY, BZ
 * thread:  num_per_thread, channels_per_thread, 1
 *          num_output_per_block(0 ~ num_output/group)
 *
 */
template<typename Dtype, int BX, int BY,
    int num_per_thread, int channels_per_thread,
    int num_output_per_block>
__global__ void TxW2B_imp(const Dtype* top_diff, const Dtype* weights, Dtype* bottom_diff,
                          const int num, const int channels, const int height, const int width,
                          /*const int num, */const int num_output, const int height_out, const int width_out,
                          /*const int num_output, const int channels/group,*/
                          const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                          const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                          const int group,
                          const float scale_targets, const float scale_outputs,
                          const bool conv) {
  __shared__ Dtype sh_weights[channels_per_thread * BY][num_output_per_block];
  __shared__ Dtype sh_top_diff[num_output_per_block][BX * num_per_thread];

  const int num_idx = blockIdx.x * (BX * num_per_thread);

  const int channels_per_group = DIVUP(channels/group, BY*channels_per_thread);
  const int group_idx = blockIdx.y / channels_per_group;
  const int channels_ingroup_idx = (blockIdx.y % channels_per_group) * BY * channels_per_thread;
  const int channels_idx = group_idx * (channels / group) + channels_ingroup_idx;

  const int pix_b_idx = blockIdx.z;
  const int pix_b_h = pix_b_idx / width;
  const int pix_b_w = pix_b_idx % width;

  const int b_dim[5] = { num, channels, height, width, group };
  int b_base[5];
  b_set_base(b_dim, b_base, num_idx, channels_idx, pix_b_h, pix_b_w, group_idx);

  const int num_output_idx = group_idx * (num_output / group);
  const int t_dim[5] = { num, num_output, height_out, width_out, group };
  int t_base[5];
  t_set_base(t_dim, t_base, num_idx, num_output_idx, 0, 0, group_idx);

  const int w_dim[7] = { num_output, channels / group, kernel_h, kernel_w, height_out, width_out, group };
  int w_base[7];
  w_set_base(w_dim, w_base, num_output_idx, channels_ingroup_idx, 0, 0, 0, 0, group_idx);

  const int tidx = threadIdx.y * BX + threadIdx.x;

  const int pix_t_h_s =
      (pix_b_h + pad_h - kernel_h < 0) ? 0 : 1 + (pix_b_h + pad_h - kernel_h) / stride_h;
  const int pix_t_h_e = MIN(height_out, 1 + (pix_b_h + pad_h) / stride_h);
  const int pix_t_w_s =
      (pix_b_w + pad_w - kernel_w < 0) ? 0 : 1 + (pix_b_w + pad_w - kernel_w) / stride_w;
  const int pix_t_w_e = MIN(width_out, 1 + (pix_b_w + pad_w) / stride_w);

  float prod[num_per_thread][channels_per_thread];
  for (int n = 0; n < num_per_thread; n++) {
    for (int c = 0; c < channels_per_thread; c++) {
      prod[n][c] = 0;
    }
  }

  for (int pix_t_h = pix_t_h_s; pix_t_h < pix_t_h_e; pix_t_h++) {

    const int pix_k_h = pix_b_h - (-pad_h + pix_t_h * stride_h);
    for (int pix_t_w = pix_t_w_s; pix_t_w < pix_t_w_e; pix_t_w++) {
      const int pix_k_w = pix_b_w - (-pad_w + pix_t_w * stride_w);
      for (int lno = 0; lno < num_output / group;
          lno += num_output_per_block) {

        // load sh_top_diff
        for (int i = 0;
            i < num_output_per_block * BX * num_per_thread;
            i += BX * BY) {
          //current:i+tidx
          const int no = (i + tidx) / (BX * num_per_thread);
          const int n = (i + tidx) % (BX * num_per_thread);

          if (i + tidx < num_output_per_block * BX * num_per_thread) {
            if (t_check(t_dim, t_base, n, lno + no, pix_t_h, pix_t_w)) {
              sh_top_diff[no][n] = top_diff[t_idx(t_dim, t_base, n, lno + no, pix_t_h, pix_t_w)];
            } else {
              sh_top_diff[no][n] = 0;
            }
          }
        }

        // load sh_weight
        for (int i = 0;
            i < channels_per_thread * BY * num_output_per_block;
            i += BX * BY) {
          //current:i+tidx
          const int c = (i + tidx) / num_output_per_block;
          const int no = (i + tidx) % num_output_per_block;

          if (i + tidx < channels_per_thread * BY * num_output_per_block) {
            if (w_check(w_dim, w_base, lno + no, c, pix_k_h, pix_k_w, pix_t_h, pix_t_w)) {
              sh_weights[c][no] = weights[w_idx(w_dim, w_base, lno + no, c, pix_k_h, pix_k_w, pix_t_h, pix_t_w)];
            } else {
              sh_weights[c][no] = 0;
            }
          }
        }
        __syncthreads();

        // Do some actual computation
        for (int n = 0; n < num_per_thread; n++) {
          for (int c = 0; c < channels_per_thread; c++) {
            for (int i = 0; i < num_output_per_block; i++) {
              prod[n][c] += sh_weights[c * BY + threadIdx.y][i] * sh_top_diff[i][threadIdx.x + n * BX];
            }
          }
        }
        __syncthreads();
      }
    }
  }

  for (int n = 0; n < num_per_thread; n++) {
    for (int c = 0; c < channels_per_thread; c++) {
      if (b_check(b_dim, b_base, threadIdx.x + n * BX, c * BY + threadIdx.y, 0, 0)) {
        bottom_diff[b_idx(b_dim, b_base, threadIdx.x + n * BX, c * BY + threadIdx.y, 0, 0)] =
            scale_targets * bottom_diff[b_idx(b_dim, b_base, threadIdx.x + n * BX, c * BY + threadIdx.y, 0, 0)]
                + scale_outputs * prod[n][c];
      }
    }
  }
}
template<typename Dtype>
void TxW2B(const Dtype* top_diff, const Dtype* weight, Dtype* bottom_diff,
           const int num, const int channels, const int height, const int width,
           /*const int num, */const int num_output, const int height_out, const int width_out,
           /*const int num_output, const int channels/group,*/
           const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
           const int pad_h, const int pad_w, const int stride_h, const int stride_w,
           const int group,
           const float scale_targets, const float scale_outputs,
           const bool conv) {
  int BX = 32;
  int BY = 4;
  int BZ = 1;
  int num_per_thread = 4;
  int channels_per_thread = 4;
  //int num_output_per_block = 4;

  dim3 threads(BX, BY, BZ);
  dim3 blocks = dim3(DIVUP(num, BX*num_per_thread),
                     group * DIVUP(channels/group, BY*channels_per_thread),
                     height * width);
  cudaFuncSetCacheConfig(TxW2B_imp<Dtype, 32, 4, 4, 4, 4>, cudaFuncCachePreferShared);
  TxW2B_imp<Dtype, 32, 4, 4, 4, 4> <<<blocks, threads>>>(top_diff, weight, bottom_diff,
                                                        num, channels, height, width,
                                                        num_output, height_out, width_out,
                                                        kernel_h, kernel_w,
                                                        pad_h, pad_w, stride_h, stride_w,
                                                        group,
                                                        scale_targets, scale_outputs,
                                                        conv);
}
template
void TxW2B<float>(const float* top_diff, const float* weight, float* bottom_diff,
                  const int num, const int channels, const int height, const int width,
                  /*const int num, */const int num_output, const int height_out, const int width_out,
                  /*const int num_output, const int channels/group,*/
                  const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                  const int group,
                  const float scale_targets, const float scale_outputs,
                  const bool conv);
template
void TxW2B<double>(const double* top_diff, const double* weight, double* bottom_diff,
                   const int num, const int channels, const int height, const int width,
                   /*const int num, */const int num_output, const int height_out, const int width_out,
                   /*const int num_output, const int channels/group,*/
                   const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
                   const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                   const int group,
                   const float scale_targets, const float scale_outputs,
                   const bool conv);

}

