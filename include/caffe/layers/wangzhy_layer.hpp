#ifndef CAFFE_WANGZHY_LAYER_HPP_
#define CAFFE_WANGZHY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class WangzhyLayer : public Layer<Dtype> {
 public:
  explicit WangzhyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Crop"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  WangzhyParameter_Op op_;
  // crop
  int off_w;
  int off_h;
  // poly
  // mirror
  // affine
  Dtype angle;
  Dtype scale;
  int center_h;
  int center_w;
  Dtype m0, m1, m2, m3, m4, m5;
  Dtype im0, im1, im2, im3, im4, im5;
  int mirror;
  //int border;
  // embedaccuracy
  // Resize
  Dtype resize_scale;
};

}  // namespace caffe

#endif  // CAFFE_WANGZHY_LAYER_HPP_
