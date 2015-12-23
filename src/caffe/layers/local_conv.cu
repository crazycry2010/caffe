#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/local_conv.hpp"
#include "caffe/util/local_conv.hpp"

namespace caffe {

template<typename Dtype>
void
LocalConvLayer<Dtype>::Forward_gpu(
                                          const vector<Blob<Dtype>*>& bottom, const vector<
                                              Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    BxW2T<Dtype>(bottom_data, weight, top_data,
                 num_, channels_, height_, width_,
                 num_output_, height_out_, width_out_,
                 kernel_h_, kernel_w_,
                 pad_h_, pad_w_, stride_h_, stride_w_, group_,
                 0, 1, false);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_,
                            num_output_ * height_out_ * width_out_, 1, (Dtype) 1., bias_multiplier_.gpu_data(), bias,
                            (Dtype) 1., top_data);
    }
  }
}

template<typename Dtype>
void
LocalConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0),
                  this->blobs_[1]->mutable_gpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1,
                            num_output_ * height_out_ * width_out_, num_, (Dtype) 1., bias_multiplier_.gpu_data(), top_diff,
                            (Dtype) 1., bias_diff);
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        BxT2W<Dtype>(bottom_data, top_diff, weight_diff,
                     num_, channels_, height_, width_,
                     num_output_, height_out_, width_out_,
                     kernel_h_, kernel_w_,
                     pad_h_, pad_w_, stride_h_, stride_w_, group_,
                     1, 1, false);
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[i]) {
        TxW2B<Dtype>(top_diff, weight, bottom_diff,
                     num_, channels_, height_, width_,
                     num_output_, height_out_, width_out_,
                     kernel_h_, kernel_w_,
                     pad_h_, pad_w_, stride_h_, stride_w_, group_,
                     0, 1, false);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LocalConvLayer);

}  // namespace caffe

