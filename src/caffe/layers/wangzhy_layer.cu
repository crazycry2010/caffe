#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CropRng(const int n, const int start, const int end, unsigned int* data) {
  CUDA_KERNEL_LOOP(index, n) {
    data[index] = start + data[index] % (end - start + 1);
  }
}

template <typename Dtype>
__global__ void CropForward(const int n, const int channels, const int height,
    const int width, const int height_out, const int width_out, 
    unsigned int* crop_at_h, unsigned int* crop_at_w,
    const Dtype* bottom_data, Dtype* top_data, const bool mirror) {
  CUDA_KERNEL_LOOP(index, n) {
    int nn = index / channels / height_out / width_out;
    int c = (index / height_out / width_out) % channels;
    int ho = (index / width_out) % height_out;
    int wo = index % width_out;
    int bottom_index = nn * channels * height * width + c * height * width
            + (crop_at_h[nn] + ho) * width + (crop_at_w[nn] + wo);
    int top_index = index;
    if(mirror) {
        top_index = nn * channels * height_out * width_out
            + c * height_out * width_out + ho * width_out
            + (width_out - 1 - wo);
    }
    top_data[top_index] = bottom_data[bottom_index];
  }
}

template <typename Dtype>
__global__ void CropBackward(const int n, const int channels, const int height,
    const int width, const int height_out, const int width_out, 
    unsigned int* crop_at_h, unsigned int* crop_at_w,
    const Dtype* top_diff, Dtype* bottom_diff, const bool mirror) {
  CUDA_KERNEL_LOOP(index, n) {
    int nn = index / channels / height_out / width_out;
    int c = (index / height_out / width_out) % channels;
    int ho = (index / width_out) % height_out;
    int wo = index % width_out;
    int bottom_index = nn * channels * height * width + c * height * width
            + (crop_at_h[nn] + ho) * width + (crop_at_w[nn] + wo);
    int top_index = index;
    if(mirror) {
        top_index = nn * channels * height_out * width_out
            + c * height_out * width_out + ho * width_out
            + (width_out - 1 - wo);
    }
    bottom_diff[bottom_index] = top_diff[top_index];
  }
}


template <typename Dtype>
__global__ void PolyForward(const int n, const Dtype* bottom, const Dtype* poly, Dtype* top) {
  CUDA_KERNEL_LOOP(index, n) {
      const Dtype x = bottom[index];
      if(x < 0)
          top[index] = 0;
      else {
          //top[index] = ((((poly[5]*x + poly[4])*x + poly[3])*x + poly[2])*x + poly[1])*x + poly[0];
          top[index] = ((poly[3]*x + poly[2])*x + poly[1])*x + poly[0];
      }

  }
}

template <typename Dtype>
__global__ void PolyBottomBackward(const int n, const Dtype* top_diff, const Dtype* bottom,
    const Dtype* poly, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
      const Dtype x = bottom[index];
      if(x < 0)
          bottom_diff[index] = 0;
      else {
          //bottom_diff[index] = (((5*poly[5]*x + 4*poly[4])*x + 3*poly[3])*x + 2*poly[2])*x + poly[1];
          bottom_diff[index] = (3*poly[3]*x + 2*poly[2])*x + poly[1];
          bottom_diff[index] = bottom_diff[index] * top_diff[index];
      }
  }
}
template <typename Dtype>
__global__ void PolyBlobBackward(const int n, const Dtype* top_diff, const Dtype* bottom,
        Dtype* poly_diff) {
  CUDA_KERNEL_LOOP(index, n) {
      const Dtype x = bottom[index];
      if(x < 0){
      } else {
          poly_diff[0] += top_diff[index];
          poly_diff[1] += top_diff[index] * pow(x,1);
          poly_diff[2] += top_diff[index] * pow(x,2);
          poly_diff[3] += top_diff[index] * pow(x,3);
          //poly_diff[4] += top_diff[index] * pow(x,4);
          //poly_diff[5] += top_diff[index] * pow(x,5);
      }
  }
}

template <typename Dtype>
__global__ void MirrorForward(const int n, const int channels, const int height,
    const int width, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
    int nn = index / channels / height / width;
    int c = (index / height / width) % channels;
    int h = (index / width) % height;
    int w = index % width;
    int bottom_index = index;
    int top_index = nn * channels * height * width
            + c * height * width + h * width
            + (width - 1 - w);
    top_data[top_index] = bottom_data[bottom_index];
  }
}

template <typename Dtype>
__global__ void MirrorBackward(const int n, const int channels, const int height,
    const int width, const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    int nn = index / channels / height / width;
    int c = (index / height / width) % channels;
    int h = (index / width) % height;
    int w = index % width;
    int bottom_index = index;
    int top_index = nn * channels * height * width
            + c * height * width + h * width
            + (width - 1 - w);
    bottom_diff[bottom_index] = top_diff[top_index];
  }
}

template <typename Dtype> void WangzhyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    switch (op_) {
    case WangzhyParameter_Op_Crop: 
    {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        unsigned int* crop_at_h = 
            static_cast<unsigned int*>(rand_h_vec_.mutable_gpu_data());
        unsigned int* crop_at_w = 
            static_cast<unsigned int*>(rand_w_vec_.mutable_gpu_data());

        if(this->layer_param().wangzhy_param().random()) {
            caffe_gpu_rng_uniform(bottom[0]->num(), crop_at_h);
            CropRng<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->num()), CAFFE_CUDA_NUM_THREADS>>>(
                    bottom[0]->num(), this->layer_param().wangzhy_param().start_h(),
                    this->layer_param().wangzhy_param().end_h(), crop_at_h);
            caffe_gpu_rng_uniform(bottom[0]->num(), crop_at_w);
            CropRng<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->num()), CAFFE_CUDA_NUM_THREADS>>>(
                    bottom[0]->num(), this->layer_param().wangzhy_param().start_w(),
                    this->layer_param().wangzhy_param().end_w(), crop_at_w);
        } else {
            caffe_gpu_set<int>(bottom[0]->num(), this->layer_param().wangzhy_param().crop_at_h(), 
                    (int*) crop_at_h);
            caffe_gpu_set<int>(bottom[0]->num(), this->layer_param().wangzhy_param().crop_at_w(), 
                    (int*) crop_at_w);
        }

        CropForward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                top[0]->count(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
                top[0]->height(), top[0]->width(), crop_at_h, crop_at_w, bottom_data, top_data,
                this->layer_param().wangzhy_param().mirror());
        CUDA_POST_KERNEL_CHECK;
    }
    break;
    case WangzhyParameter_Op_Poly:
    {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* poly = this->blobs_[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        PolyForward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                top[0]->count(), bottom_data, poly, top_data);
    }
    break;
    case WangzhyParameter_Op_Mirror: 
    {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        MirrorForward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                top[0]->count(), top[0]->channels(), top[0]->height(),
                top[0]->width(), bottom_data, top_data);
    }
    break;
    }
}

template <typename Dtype>
void WangzhyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    switch (op_) {
    case WangzhyParameter_Op_Crop:
    {
        if (propagate_down[0]) {
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            unsigned int* crop_at_h = 
                static_cast<unsigned int*>(rand_h_vec_.mutable_gpu_data());
            unsigned int* crop_at_w = 
                static_cast<unsigned int*>(rand_w_vec_.mutable_gpu_data());
            caffe_gpu_set<Dtype>(bottom[0]->count(), Dtype(0), bottom_diff);
            CropBackward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                    top[0]->count(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
                    top[0]->height(), top[0]->width(), crop_at_h, crop_at_w, top_diff, bottom_diff,
                    this->layer_param().wangzhy_param().mirror());
            CUDA_POST_KERNEL_CHECK;
        }
    }
    break;
    case WangzhyParameter_Op_Poly:
    {
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* poly = this->blobs_[0]->gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        Dtype* poly_diff = this->blobs_[0]->mutable_gpu_diff();
        if (this->param_propagate_down_[0]) {
            PolyBottomBackward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                    top[0]->count(), top_diff, bottom_data, poly, bottom_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[0]) {
            caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), poly_diff);
            PolyBlobBackward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                    top[0]->count(), top_diff, bottom_data, bottom_diff);
        }
    }
    break;
    case WangzhyParameter_Op_Mirror: 
    {
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[0]) {
            MirrorBackward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                    top[0]->count(), top[0]->channels(), top[0]->height(),
                    top[0]->width(), top_diff, bottom_diff);
        }
    }
    break;
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(WangzhyLayer);


}  // namespace caffe
