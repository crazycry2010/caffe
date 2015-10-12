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

template <typename Dtype>
__global__ void AffineForward(const int num, const int channels, const int top_height,
    const int top_width, const int bottom_height, const int bottom_width,
    const Dtype im0, const Dtype im1, const Dtype im2, const Dtype im3, const Dtype im4, const Dtype im5,
    const Dtype* bottom_data, Dtype* top_data, int mirror) {
  CUDA_KERNEL_LOOP(index, num) {
    int n = index / channels / top_height / top_width;
    int c = (index / top_height / top_width) % channels;
    int h = (index / top_width) % top_height;
    int w = index % top_width;
    // top(n,c,h,w) => bottom(n,c,hh,ww)
    Dtype ww = im0 * w + im1 * h + im2;
    Dtype hh = im3 * w + im4 * h + im5;
    int w1 = ww;
    int h1 = hh;
    int w2 = w1+1;
    int h2 = h1+1;

    Dtype out = 0;
    int src_index = 0;
    Dtype src_reg = 0;
    if(h1>=0 && h1<bottom_height && w1>=0 && w1<bottom_width ){
      src_index = n * channels * bottom_height * bottom_width
          + c * bottom_height * bottom_width + h1 * bottom_width
          + w1;
      src_reg = bottom_data[src_index];
      out = out + src_reg*((w2-ww)*(h2-hh));
    }
    if(h1>=0 && h1<bottom_height && w2>=0 && w2<bottom_width ){
      src_index = n * channels * bottom_height * bottom_width
          + c * bottom_height * bottom_width + h1 * bottom_width
          + w2;
      src_reg = bottom_data[src_index];
      out = out + src_reg*((ww-w1)*(h2-hh));
    }
    if(h2>=0 && h2<bottom_height && w1>=0 && w1<bottom_width ){
      src_index = n * channels * bottom_height * bottom_width
          + c * bottom_height * bottom_width + h2 * bottom_width
          + w1;
      src_reg = bottom_data[src_index];
      out = out + src_reg*((w2-ww)*(hh-h1));
    }
    if(h2>=0 && h2<bottom_height && w2>=0 && w2<bottom_width ){
      src_index = n * channels * bottom_height * bottom_width
          + c * bottom_height * bottom_width + h2 * bottom_width
          + w2;
      src_reg = bottom_data[src_index];
      out = out + src_reg*((ww - w1)*(hh-h1));
    }
    int top_index = index;
    if(mirror){
        top_index = n * channels * top_height * top_width
            + c * top_height * top_width + h * top_width
            + (top_width - 1 - w);
    }
    top_data[top_index] = out;
  }
}

template <typename Dtype>
__global__ void AffineBackward(const int num, const int channels, const int bottom_height,
    const int bottom_width, const int top_height, const int top_width,
    const Dtype m0, const Dtype m1, const Dtype m2, const Dtype m3, const Dtype m4, const Dtype m5,
    const Dtype* top_diff, Dtype* bottom_diff, int mirror) {
  CUDA_KERNEL_LOOP(index, num) {
    int n = index / channels / bottom_height / bottom_width;
    int c = (index / bottom_height / bottom_width) % channels;
    int h = (index / bottom_width) % bottom_height;
    int w = index % bottom_width;
    // bottom(n,c,h,w) => top(n,c,hh,ww)
    Dtype ww = m0 * w + m1 * h + m2;
    Dtype hh = m3 * w + m4 * h + m5;
    int w1 = ww;
    int h1 = hh;
    int w2 = w1+1;
    int h2 = h1+1;

    Dtype out = 0;
    int src_index = 0;
    Dtype src_reg = 0;
    if(h1>=0 && h1<top_height && w1>=0 && w1<top_width ){
      src_index = n * channels * top_height * top_width
          + c * top_height * top_width + h1 * top_width
          + w1;
      if(mirror){
          src_index = n * channels * top_height * top_width
              + c * top_height * top_width + h1 * top_width
              + (top_width-1-w1);
      }
      src_reg = top_diff[src_index];
      out = out + src_reg*((w2-ww)*(h2-hh));
    }
    if(h1>=0 && h1<top_height && w2>=0 && w2<top_width ){
      src_index = n * channels * top_height * top_width
          + c * top_height * top_width + h1 * top_width
          + w2;
      if(mirror){
          src_index = n * channels * top_height * top_width
              + c * top_height * top_width + h1 * top_width
              + (top_width-1-w2);
      }
      src_reg = top_diff[src_index];
      out = out + src_reg*((ww-w1)*(h2-hh));
    }
    if(h2>=0 && h2<top_height && w1>=0 && w1<top_width ){
      src_index = n * channels * top_height * top_width
          + c * top_height * top_width + h2 * top_width
          + w1;
      if(mirror){
          src_index = n * channels * top_height * top_width
              + c * top_height * top_width + h2 * top_width
              + (top_width-1-w1);
      }
      src_reg = top_diff[src_index];
      out = out + src_reg*((w2-ww)*(hh-h1));
    }
    if(h2>=0 && h2<top_height && w2>=0 && w2<top_width ){
      src_index = n * channels * top_height * top_width
          + c * top_height * top_width + h2 * top_width
          + w2;
      if(mirror){
          src_index = n * channels * top_height * top_width
              + c * top_height * top_width + h2 * top_width
              + (top_width-1-w2);
      }
      src_reg = top_diff[src_index];
      out = out + src_reg*((ww - w1)*(hh-h1));
    }
    bottom_diff[index] = out;
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
        CUDA_POST_KERNEL_CHECK;
    }
    break;
    case WangzhyParameter_Op_Mirror: 
    {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        MirrorForward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                top[0]->count(), top[0]->channels(), top[0]->height(),
                top[0]->width(), bottom_data, top_data);
        CUDA_POST_KERNEL_CHECK;
    }
    break;
    case WangzhyParameter_Op_Affine:
    {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        if (this->phase_ == TRAIN) {
          if(this->layer_param().wangzhy_param().d_angle() == 0){
              angle = 0;
          } else {
              caffe_rng_uniform<Dtype>(1, -this->layer_param().wangzhy_param().d_angle(), this->layer_param().wangzhy_param().d_angle(), &angle);
              angle = 0 + angle * 3.141592653 / 180;
          }
          if(this->layer_param().wangzhy_param().d_scale() == 0) {
              scale = 1;
          } else {
              caffe_rng_uniform<Dtype>(1, -this->layer_param().wangzhy_param().d_scale(), this->layer_param().wangzhy_param().d_scale(), &scale);
              scale = 1 + scale;
          }
          if(this->layer_param().wangzhy_param().d_center() == 0){
              center_w = bottom[0]->height() / 2;
              center_h = bottom[0]->width() / 2;
          }
          else {
              Dtype center = 0;
              caffe_rng_uniform<Dtype>(1, -this->layer_param().wangzhy_param().d_center(), this->layer_param().wangzhy_param().d_center(), &center);
              center_w = bottom[0]->height() / 2 + (int)center;
              center_h = bottom[0]->width() / 2 + (int)center;
          }
          caffe_rng_bernoulli(1, 0.5, &mirror);
          mirror = 0;

          Dtype alpha = cos(angle)*scale;
          Dtype beta = sin(angle)*scale;
          m0 = alpha;
          m1 = beta;
          /*m2 = (1-alpha)*center_w - beta*center_h + border;*/
          m2 = (1-alpha)*center_w - beta*center_h;
          m3 = -beta;
          m4 = alpha;
          /*m5 = beta*center_w + (1-alpha)*center_h + border;*/
          m5 = beta*center_w + (1-alpha)*center_h;
          Dtype D = m0*m4-m1*m3;
          D = (D != 0) ? 1./D : 0;
          im0 = m4*D;
          im1 = -m1*D;
          im3= -m3*D;
          im4 = m0*D;
          im2 = -im0*m2-im1*m5;
          im5 = -im3*m2-im4*m5;

          AffineForward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
              top[0]->count(), top[0]->channels(), top[0]->height(),
              top[0]->width(), bottom[0]->height(), bottom[0]->width(),
              im0, im1, im2,im3, im4, im5, bottom_data, top_data, mirror);
          CUDA_POST_KERNEL_CHECK;
        } else {
          /*for(int n = 0;n < bottom[0]->num();n++){*/
            /*for(int c = 0;c < bottom[0]->channels();c++){*/
                /*for(int h = 0;h < bottom[0]->height();h++){*/
                    /*int bottom_offset = n * bottom[0]->channels() * bottom[0]->height() * bottom[0]->width()*/
                        /*+ c * bottom[0]->height() * bottom[0]->width() + h * bottom[0]->width();*/
                    /*int top_offset = n * top[0]->channels() * top[0]->height() * top[0]->width()*/
                        /*+ c * top[0]->height() * top[0]->width() + h * top[0]->width() + border;*/
                    /*caffe_copy(bottom[0]->width(), bottom_data+bottom_offset, top_data+top_offset);*/
                /*}*/
            /*}*/
          /*}*/
            caffe_copy(bottom[0]->count(), bottom_data, top_data);
        }
    }
    break;
    case WangzhyParameter_Op_EmbedAccuracy:
    {
        const Dtype* bottom_data = bottom[0]->cpu_data(); // N * N_
        const Dtype* weight = bottom[1]->cpu_data();// K_ * N_
        Dtype* top_data = top[0]->mutable_cpu_data(); // N * K_
        const int N = bottom[0]->num();
        const int N_ = bottom[0]->count() / N;
        const int K_ = bottom[1]->num();
        Dtype loss = 0;
        for(int i = 0;i < N;i++){
            for(int j = 0;j < K_;j++){
                loss = 0;
                for(int k = 0;k < N_;k++){
                    loss += bottom_data[i*N_+k] * (weight[j*N_+k] - (bottom_data[i*N_+k] >= 0)) -
                        log(1 + exp(bottom_data[i*N_+k] - 2 * bottom_data[i*N_+k] * (bottom_data[i*N_+k] >= 0)));
                }
                top_data[i*K_+j] = loss;
            }
        }
    }
    break;
    case WangzhyParameter_Op_OneHot:
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int num = top[0]->num();
        const int dim = top[0]->count() / num;
        caffe_set(top[0]->count(), Dtype(0), top_data);
        for(int i = 0;i < num;i++){
            int value = bottom_data[i];
            top_data[i*dim+value] = 1;
        }
    }
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
            CUDA_POST_KERNEL_CHECK;
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[0]) {
            caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), poly_diff);
            PolyBlobBackward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                    top[0]->count(), top_diff, bottom_data, bottom_diff);
            CUDA_POST_KERNEL_CHECK;
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
            CUDA_POST_KERNEL_CHECK;
        }
    }
    break;
    case WangzhyParameter_Op_Affine:
        {
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[0]) {
                AffineBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                        bottom[0]->count(), bottom[0]->channels(), bottom[0]->height(),
                        bottom[0]->width(), top[0]->height(), top[0]->width(),
                        m0, m1, m2, m3, m4, m5, top_diff, bottom_diff, mirror);
                CUDA_POST_KERNEL_CHECK;
            }
        }
        break;
    case WangzhyParameter_Op_EmbedAccuracy:
        {
        }
        break;
    case WangzhyParameter_Op_OneHot:
        {
        }
        break;
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(WangzhyLayer);


}  // namespace caffe
