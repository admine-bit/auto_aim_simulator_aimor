#ifndef AUTO_AIM__TRT_ENGINE_HPP
#define AUTO_AIM__TRT_ENGINE_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <string>
#include <vector>

#include "infer_engine.hpp"

namespace auto_aim
{
// TensorRT 后端：加载离线生成的 .engine，使用 GPU 完成预处理和推理。
// engine 必须与目标 Jetson 的 GPU 架构和 TensorRT 版本匹配。
class TRTEngine : public InferEngine
{
public:
  struct Profile
  {
    float upload_ms{0.0f};
    float letterbox_ms{0.0f};
    float preprocess_ms{0.0f};
    float engine_ms{0.0f};
    float download_ms{0.0f};
    float total_ms{0.0f};
  };

  TRTEngine(const std::string & engine_path, int input_h, int input_w, int channels = 3);
  ~TRTEngine() override;

  cv::Mat infer(const cv::Mat & input_bgr_u8) override;
  cv::Mat infer(const cv::cuda::GpuMat & input_bgr_u8) override;
  cv::Mat infer_letterbox(
    const cv::Mat & input_bgr_u8, double & scale, int pad_value = 0) override;
  cv::Mat infer_letterbox_profiled(
    const cv::Mat & input_bgr_u8, double & scale, Profile * profile, int pad_value = 0);

  int input_h() const override { return input_h_; }
  int input_w() const override { return input_w_; }

private:
  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override;
  } logger_;

  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * ctx_{nullptr};

  cv::cuda::Stream cvstream_;
  // OpenCV CUDA 与 TensorRT 共用同一条 CUDA stream，保证执行顺序。
  cudaStream_t stream_{nullptr};

  struct OutputTensor
  {
    std::string name;
    nvinfer1::Dims dims{};
    size_t elements{0};
    size_t bytes{0};
    void * device{nullptr};
  };

  std::string in_name_;
  std::vector<OutputTensor> outputs_;
  int primary_output_{-1};
  int input_h_, input_w_, channels_;
  int out_dim1_{0}, out_dim2_{0};
  size_t out_bytes_{0};

  // 连续的 NCHW 输入缓冲区，planes_ 是其各通道的 GpuMat 视图。
  void * d_in_{nullptr};
  cv::cuda::GpuMat gpu_src_, gpu_in_, gpu_rgb_, gpu_f32_;
  std::vector<cv::cuda::GpuMat> planes_;

  void infer_gpu_internal();
  void preprocess_gpu_internal();
  void enqueue_internal();
  cv::Mat download_output();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRT_ENGINE_HPP
