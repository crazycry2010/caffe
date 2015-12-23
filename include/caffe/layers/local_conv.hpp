#ifndef CAFFE_LOCALCONV_LAYER_HPP_
#define CAFFE_LOCALCONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class LocalConvLayer : public Layer<Dtype> {
 public:
  explicit LocalConvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "LocalConv";
  }
  virtual inline int MinBottomBlobs() const {
    return 1;
  }
  virtual inline int MinTopBlobs() const {
    return 1;
  }
  virtual inline bool EqualNumBottomTopBlobs() const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<
                                Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<
                                Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int num_;
  int channels_;
  int pad_h_, pad_w_;
  int height_, width_;
  int group_;
  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  bool conv_;

  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_LOCALCONV_LAYER_HPP_
