#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/local_conv.hpp"

namespace caffe {

template<typename Dtype>
void LocalConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  LocalConvParameter local_conv_param = this->layer_param_.local_conv_param();
  CHECK(!local_conv_param.has_kernel_size() !=
      !(local_conv_param.has_kernel_h() && local_conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(local_conv_param.has_kernel_size() ||
      (local_conv_param.has_kernel_h() && local_conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!local_conv_param.has_pad() && local_conv_param.has_pad_h()
      && local_conv_param.has_pad_w())
      || (!local_conv_param.has_pad_h() && !local_conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!local_conv_param.has_stride() && local_conv_param.has_stride_h()
      && local_conv_param.has_stride_w())
      || (!local_conv_param.has_stride_h() && !local_conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (local_conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = local_conv_param.kernel_size();
  } else {
    kernel_h_ = local_conv_param.kernel_h();
    kernel_w_ = local_conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!local_conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = local_conv_param.pad();
  } else {
    pad_h_ = local_conv_param.pad_h();
    pad_w_ = local_conv_param.pad_w();
  }
  if (!local_conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = local_conv_param.stride();
  } else {
    stride_h_ = local_conv_param.stride_h();
    stride_w_ = local_conv_param.stride_w();
  }

  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.local_conv_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.local_conv_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";

  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  height_out_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  conv_ = this->layer_param_.local_conv_param().conv();

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.local_conv_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    if(conv_)
        this->blobs_[0].reset(new Blob<Dtype>(num_output_, channels_ / group_, kernel_h_ * kernel_w_, 1));
    else
        this->blobs_[0].reset(new Blob<Dtype>(num_output_, channels_ / group_, kernel_h_ * kernel_w_, height_out_ * width_out_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                                                              this->layer_param_.local_conv_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
        if(conv_)
            this->blobs_[1].reset(new Blob<Dtype>(1, num_output_, 1, 1));
        else
            this->blobs_[1].reset(new Blob<Dtype>(1, num_output_, height_out_, width_out_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                                                              this->layer_param_.local_conv_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void LocalConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
      " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }

  if (bias_term_) {
    bias_multiplier_.Reshape(num_, 1, 1, 1);
    caffe_set(bias_multiplier_.count(), Dtype(1),
              bias_multiplier_.mutable_cpu_data());
  }
}

template<typename Dtype>
void LocalConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
}

template<typename Dtype>
void LocalConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down, const vector<
                                                    Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(LocalConvLayer);
#endif

INSTANTIATE_CLASS(LocalConvLayer);
REGISTER_LAYER_CLASS(LocalConv);

}  // namespace caffe

