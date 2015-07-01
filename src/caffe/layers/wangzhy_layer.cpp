#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WangzhyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.wangzhy_param().operation();
    switch (op_) {
        case WangzhyParameter_Op_Crop:
            break;
        case WangzhyParameter_Op_Poly:
          if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
          } else {
            this->blobs_.resize(1);
            this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 6));
            Dtype* poly = this->blobs_[0]->mutable_cpu_data();
            poly[0] = -0.599572228154110;
            poly[1] = 1.256635022859541;
            poly[2] = -0.027359069456555;
            poly[3] = 0.000245929079637;
            poly[4] = -0.000000780820381;
            poly[5] = -0.000000000288062;
          }
          this->param_propagate_down_.resize(this->blobs_.size(), true);
          break;
    }

}

template <typename Dtype>
void WangzhyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    switch (op_) {
        case WangzhyParameter_Op_Crop:
            top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                    this->layer_param().wangzhy_param().crop_size(),
                    this->layer_param().wangzhy_param().crop_size());
            rand_h_vec_.Reshape(bottom[0]->num(),1,1,1);
            rand_w_vec_.Reshape(bottom[0]->num(),1,1,1);
            break;
        case WangzhyParameter_Op_Poly:
            top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                    bottom[0]->height(), bottom[0]->width());
            break;
    }
}

template <typename Dtype>
void WangzhyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void WangzhyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(WangzhyLayer);
#endif

INSTANTIATE_CLASS(WangzhyLayer);
REGISTER_LAYER_CLASS(Wangzhy);

}  // namespace caffe
