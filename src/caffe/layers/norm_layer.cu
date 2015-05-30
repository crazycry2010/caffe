#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_dim_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    Dtype maxval = -FLT_MAX;
    for (int d = 0; d < spatial_dim; ++d) {
      maxval = max(data[(n * channels + c) * spatial_dim + d], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_dim_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* scale_data, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int c = (index / spatial_dim) % channels;
    data[index] /= scale_data[n * channels + c];
  }
}

template <typename Dtype>
__global__ void kernel_dim_mul(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* scale_data, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int c = (index / spatial_dim) % channels;
    data[index] *= scale_data[n * channels + c];
  }
}

template <typename Dtype>
void NormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->count(2);
  int count = bottom[0]->count();
  
  caffe_copy(count, bottom_data, top_data);

  kernel_dim_max<Dtype><<<CAFFE_GET_BLOCKS(num * channels),
      CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
      scale_data);
  kernel_dim_div<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, num, channels, spatial_dim,
      scale_data, top_data);
}

template <typename Dtype>
void NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int count = top[0]->count();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->count(2);
  int count = bottom[0]->count();
  
  caffe_copy(count, top_diff, bottom_diff);

  /*const Dtype* bottom_data = bottom[0]->gpu_data();*/
  /*kernel_dim_max<Dtype><<<CAFFE_GET_BLOCKS(num * channels),*/
      /*CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_data,*/
      /*scale_data);*/
  kernel_dim_mul<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, num, channels, spatial_dim,
      scale_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(NormLayer);

}  // namespace caffe
