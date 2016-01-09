#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/wangzhy_layer.hpp"

namespace caffe {

  template <typename Dtype>
    __global__ void CropForward(const int count, const int channels, const int height, const int width, 
        const int off_h, const int off_w,
        const Dtype* bottom_data, Dtype* top_data) {
      CUDA_KERNEL_LOOP(index, count) {
        int n = index / channels / height/ width;
        int c = (index / height/ width) % channels;
        int h = (index / width) % height;
        int w = index % width;
        if(off_h + h >=0 && off_h + h < height && off_w + w >= 0 && off_w + w < width){
          int bottom_index = n * channels * height * width + c * height * width
            + (off_h + h) * width + (off_w + w);
          int top_index = index;
          top_data[top_index] = bottom_data[bottom_index];
        }
        else {
          int top_index = index;
          top_data[top_index] = 0;
        }
      }
    }

  template <typename Dtype>
    __global__ void CropBackward(const int count, const int channels, const int height, const int width, 
        const int off_h, const int off_w, 
        const Dtype* top_diff, Dtype* bottom_diff) {
      CUDA_KERNEL_LOOP(index, count) {
        int n = index / channels / height/ width;
        int c = (index / height/ width) % channels;
        int h = (index / width) % height;
        int w = index % width;
        if(h - off_h >=0 && h - off_h < height && w - off_w >= 0 && w - off_w < width){
          int top_index = n * channels * height * width + c * height * width
            + (h - off_h) * width + (w - off_w);
          int bottom_index = index;
          bottom_diff[bottom_index] = top_diff[top_index];
        }
        else {
          int bottom_index = index;
          bottom_diff[bottom_index] = 0;
        }
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
          if(this->layer_param().wangzhy_param().random()){
            Dtype rand= 0;
            caffe_rng_uniform<Dtype>(1, 0, 3, &rand);
            off_h = (int)rand - 1;
            caffe_rng_uniform<Dtype>(1, 0, 3, &rand);
            off_w = (int)rand - 1;
            if (this->phase_ == TEST) {
              off_h = 0;
              off_w = 0;
            }
          } else {
            off_h = this->layer_param().wangzhy_param().start_h();
            off_w = this->layer_param().wangzhy_param().start_w();
          }

          CropForward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
              top[0]->count(), top[0]->channels(), top[0]->height(), top[0]->width(),
              off_h, off_w, bottom_data, top_data);
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
          Dtype rand;
          if(this->layer_param().wangzhy_param().d_angle() == 0){
            angle = 0;
          } else {
            /*caffe_rng_uniform<Dtype>(1, 0, 2*this->layer_param().wangzhy_param().d_angle(), &rand);*/
            caffe_rng_gaussian<Dtype>(1, 0, this->layer_param().wangzhy_param().d_angle(), &rand);
            angle = 0 + (rand) * 3.1415926 / 180;
          }
          if(this->layer_param().wangzhy_param().d_scale() == 0) {
            scale = 1;
          } else {
            /*caffe_rng_uniform<Dtype>(1, 0, 2 * this->layer_param().wangzhy_param().d_scale(), &rand);*/
            caffe_rng_gaussian<Dtype>(1, 0, this->layer_param().wangzhy_param().d_scale(), &rand);
            scale = 1 + (rand);
          }
          if(this->layer_param().wangzhy_param().d_center() == 0){
            center_h = bottom[0]->height() / 2;
            center_w = bottom[0]->width() / 2;
          }
          else {
            /*caffe_rng_uniform<Dtype>(1, 0, 2*this->layer_param().wangzhy_param().d_center(), &rand);*/
            caffe_rng_gaussian<Dtype>(1, 0, this->layer_param().wangzhy_param().d_center(), &rand);
            center_h = bottom[0]->height() / 2 + (rand);
            caffe_rng_gaussian<Dtype>(1, 0, this->layer_param().wangzhy_param().d_center(), &rand);
            center_w = bottom[0]->width() / 2 + (rand);
            if(center_h < 0)
              center_h = 0;
            else if(center_h > bottom[0]->height())
              center_h = bottom[0]->height();
            if(center_w < 0)
              center_w = 0;
            else if(center_w > bottom[0]->width())
              center_w = bottom[0]->width();
          }

          if(this->layer_param().wangzhy_param().random()) {
            caffe_rng_uniform<Dtype>(1, 0, 3, &rand);
            off_h = (int)rand - 1;
            caffe_rng_uniform<Dtype>(1, 0, 3, &rand);
            off_w = (int)rand - 1;
          } else {
            off_h = this->layer_param().wangzhy_param().start_h();
            off_w = this->layer_param().wangzhy_param().start_w();
          }

          //LOG(INFO) << "angle: " << angle/3.1415926*180
            //<< ", scale:" << scale
            //<< ", center: (" << center_h << ", " << center_w << ")"
            //<< ", off: (" << off_h << ", " << off_w << ")";

          caffe_rng_bernoulli(1, 0.5, &mirror);
          mirror = 0;
          if (this->phase_ == TEST) {
            angle = 0;
            scale = 1;
            center_w = bottom[0]->height() / 2;
            center_h = bottom[0]->width() / 2;
            off_h = 0;
            off_w = 0;
          }

          Dtype alpha = cos(angle)*scale;
          Dtype beta = sin(angle)*scale;
          m0 = alpha;
          m1 = beta;
          m2 = (1-alpha)*center_w - beta*center_h + off_h;
          m3 = -beta;
          m4 = alpha;
          m5 = beta*center_w + (1-alpha)*center_h + off_w;
          Dtype D = m0*m4-m1*m3;
          D = (D != 0) ? 1./D : 0;
          im0 = m4*D;
          im1 = -m1*D;
          im3= -m3*D;
          im4 = m0*D;
          im2 = -im0*m2-im1*m5;
          im5 = -im3*m2-im4*m5;
          for(int i = 0;i < bottom.size(); i++){
            const Dtype* bottom_data = bottom[i]->gpu_data();
            Dtype* top_data = top[i]->mutable_gpu_data();
            AffineForward<Dtype><<<CAFFE_GET_BLOCKS(top[i]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                top[i]->count(), top[i]->channels(), top[i]->height(),
                top[i]->width(), bottom[i]->height(), bottom[i]->width(),
                im0, im1, im2,im3, im4, im5, bottom_data, top_data, mirror);
            CUDA_POST_KERNEL_CHECK;
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
        break;
      case WangzhyParameter_Op_Resize:
        {
          const Dtype* bottom_data = bottom[0]->gpu_data();
          Dtype* top_data = top[0]->mutable_gpu_data();
          angle = 0;
          scale = resize_scale;
          center_w = 0;
          center_h = 0;
          mirror = 0;

          Dtype alpha = cos(angle)*scale;
          Dtype beta = sin(angle)*scale;
          m0 = alpha;
          m1 = beta;
          m2 = (1-alpha)*center_w - beta*center_h;
          m3 = -beta;
          m4 = alpha;
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
        }
        break;
      case WangzhyParameter_Op_EuclideanAccuracy:
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
                loss -= pow(bottom_data[i*N_+k] - weight[j*N_+k],2);
              }
              top_data[i*K_+j] = loss;
            }
          }
        }
        break;
      default:
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
              caffe_gpu_set<Dtype>(bottom[0]->count(), Dtype(0), bottom_diff);
              CropBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                  bottom[0]->count(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
                  off_h, off_w, top_diff, bottom_diff);
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
            for(int i = 0;i < top.size();i++){
              const Dtype* top_diff = top[i]->gpu_diff();
              Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
              // gradient w.r.t. bottom data, if necessary.
              if (propagate_down[i]) {
                AffineBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[i]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                    bottom[i]->count(), bottom[i]->channels(), bottom[i]->height(),
                    bottom[i]->width(), top[i]->height(), top[i]->width(),
                    m0, m1, m2, m3, m4, m5, top_diff, bottom_diff, mirror);
                CUDA_POST_KERNEL_CHECK;
              }
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
        case WangzhyParameter_Op_Resize:
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
        case WangzhyParameter_Op_EuclideanAccuracy:
          {
          }
          break;
        default:
          break;
      }
    }


  INSTANTIATE_LAYER_GPU_FUNCS(WangzhyLayer);


}  // namespace caffe
