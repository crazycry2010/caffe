#ifndef CAFFE_LOCAL_CONV_H_
#define CAFFE_LOCAL_CONV_H_

#include "caffe/common.hpp"

#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#define MIN(a,b) (a) < (b) ? (a) : (b)

namespace caffe {

template<typename Dtype>
void BxW2T(const Dtype* bottom, const Dtype* weights, Dtype* top,
           const int num, const int channels, const int height, const int width,
           /*const int num, */const int num_output, const int height_out, const int width_out,
           /*const int num_output, const int channels/group,*/
           const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
           const int pad_h, const int pad_w, const int stride_h, const int stride_w,
           const int group,
           const float scale_targets, const float scale_outputs,
           const bool conv);
template<typename Dtype>
void BxT2W(const Dtype* bottom_data, const Dtype* top_diff, Dtype* weight_diff,
           const int num, const int channels, const int height, const int width,
           /*const int num, */const int num_output, const int height_out, const int width_out,
           /*const int num_output, const int channels/group,*/
           const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
           const int pad_h, const int pad_w, const int stride_h, const int stride_w,
           const int group,
           const float scale_targets, const float scale_outputs,
           const bool conv);
template<typename Dtype>
void TxW2B(const Dtype* top_diff, const Dtype* weight, Dtype* bottom_diff,
           const int num, const int channels, const int height, const int width,
           /*const int num, */const int num_output, const int height_out, const int width_out,
           /*const int num_output, const int channels/group,*/
           const int kernel_h, const int kernel_w, /*const int height_out, const int width_out,*/
           const int pad_h, const int pad_w, const int stride_h, const int stride_w,
           const int group,
           const float scale_targets, const float scale_outputs,
           const bool conv);
}  // namespace caffe

#endif   // CAFFE_UTIL_CUDACONV2_H_

