#include "trt_engine.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <stdexcept>
#include <vector>

namespace auto_aim
{
void TRTEngine::Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
}

TRTEngine::TRTEngine(const std::string & engine_path, int input_h, int input_w, int channels)
: input_h_(input_h), input_w_(input_w), channels_(channels)
{
  // engine 必须由目标 Jetson 的 TensorRT 环境生成。
  std::ifstream file(engine_path, std::ios::binary);
  if (!file.good()) throw std::runtime_error("TRTEngine: cannot open engine file: " + engine_path);
  std::vector<char> data(
    (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  runtime_ = nvinfer1::createInferRuntime(logger_);
  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
  if (!engine_) throw std::runtime_error("TRTEngine: deserializeCudaEngine failed: " + engine_path);
  ctx_ = engine_->createExecutionContext();

  // TRT10 按名字识别输入/输出张量。部分 ONNX 会保留多个输出，这里绑定所有输出，
  // 业务默认返回元素数最大的那个检测/分类输出。
  // TensorRT 10 按张量名称绑定输入输出，主输出选择元素数最多的一项。
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      in_name_ = name;
    } else {
      OutputTensor output;
      output.name = name;
      outputs_.push_back(std::move(output));
    }
  }
  if (in_name_.empty() || outputs_.empty()) {
    throw std::runtime_error("TRTEngine: engine must have one input and at least one output: " + engine_path);
  }

  // 固定输入 batch=1（兼容静态/动态引擎），之后才能取到确定的输出形状。
  ctx_->setInputShape(
    in_name_.c_str(), nvinfer1::Dims4{1, channels_, input_h_, input_w_});

  // 为每个输出分配显存，兼容模型包含多个输出的情况。
  size_t max_elements = 0;
  for (int i = 0; i < static_cast<int>(outputs_.size()); ++i) {
    auto & output = outputs_[i];
    output.dims = ctx_->getTensorShape(output.name.c_str());
    output.elements = 1;
    for (int d = 0; d < output.dims.nbDims; ++d) {
      if (output.dims.d[d] < 0) {
        throw std::runtime_error("TRTEngine: dynamic output shape is not resolved: " + output.name);
      }
      output.elements *= static_cast<size_t>(output.dims.d[d]);
    }
    output.bytes = output.elements * sizeof(float);
    cudaMalloc(&output.device, output.bytes);
    ctx_->setTensorAddress(output.name.c_str(), output.device);

    if (output.elements > max_elements) {
      max_elements = output.elements;
      primary_output_ = i;
    }
  }
  if (primary_output_ < 0) {
    throw std::runtime_error("TRTEngine: failed to choose primary output: " + engine_path);
  }

  const auto & primary = outputs_[primary_output_];
  // 兼容 2D 输出 [batch, features]（如分类器 [1,9]）和 3D 输出 [batch, d1, d2]（如 YOLO）。
  out_dim1_ = primary.dims.nbDims >= 2 ? primary.dims.d[1] : primary.dims.d[0];
  out_dim2_ = primary.dims.nbDims >= 3 ? primary.dims.d[2] : 1;
  out_bytes_ = primary.bytes;

  stream_ = cv::cuda::StreamAccessor::getStream(cvstream_);

  // 连续 NCHW 输入缓冲；用 channels_ 个 step=w*sizeof(float) 的 GpuMat 头包裹其通道，
  // 使 split 的输出在 d_in_ 中紧密排列成 NCHW（GpuMat 自身分配会按行对齐补 pad，故不能直接切片）。
  cudaMalloc(
    &d_in_, static_cast<size_t>(channels_) * input_h_ * input_w_ * sizeof(float));
  auto * base = static_cast<float *>(d_in_);
  size_t plane = static_cast<size_t>(input_h_) * input_w_;
  size_t step = static_cast<size_t>(input_w_) * sizeof(float);
  planes_.reserve(channels_);
  for (int c = 0; c < channels_; ++c) {
    planes_.emplace_back(input_h_, input_w_, CV_32F, base + c * plane, step);
  }

  ctx_->setTensorAddress(in_name_.c_str(), d_in_);
}

TRTEngine::~TRTEngine()
{
  if (d_in_) cudaFree(d_in_);
  for (auto & output : outputs_) {
    if (output.device) cudaFree(output.device);
  }
  delete ctx_;
  delete engine_;
  delete runtime_;
}

cv::Mat TRTEngine::infer(const cv::Mat & input_bgr_u8)
{
  // GPU 预处理 + 推理（内部 upload）
  gpu_in_.upload(input_bgr_u8, cvstream_);
  infer_gpu_internal();
  return download_output();
}

cv::Mat TRTEngine::infer(const cv::cuda::GpuMat & input_bgr_u8)
{
  // GPU 预处理 + 推理（跳过 upload，输入已在 GPU 上，零拷贝）
  // 注意：input_bgr_u8 必须已对齐到当前 stream
  gpu_in_ = input_bgr_u8;
  infer_gpu_internal();
  return download_output();
}

cv::Mat TRTEngine::infer_letterbox(
  const cv::Mat & input_bgr_u8, double & scale, int pad_value)
{
  return infer_letterbox_profiled(input_bgr_u8, scale, nullptr, pad_value);
}

cv::Mat TRTEngine::infer_letterbox_profiled(
  const cv::Mat & input_bgr_u8, double & scale, Profile * profile, int pad_value)
{
  // 保持宽高比缩放，右侧和底部使用调用方指定的 letterbox 填充值。
  const auto x_scale = static_cast<double>(input_h_) / input_bgr_u8.rows;
  const auto y_scale = static_cast<double>(input_w_) / input_bgr_u8.cols;
  scale = std::min(x_scale, y_scale);
  const auto h = static_cast<int>(input_bgr_u8.rows * scale);
  const auto w = static_cast<int>(input_bgr_u8.cols * scale);

  if (!profile) {
    gpu_src_.upload(input_bgr_u8, cvstream_);
    gpu_in_.create(input_h_, input_w_, input_bgr_u8.type());
    if (h < input_h_) {
      gpu_in_(cv::Rect(0, h, input_w_, input_h_ - h))
        .setTo(cv::Scalar::all(pad_value), cvstream_);
    }
    if (w < input_w_) {
      gpu_in_(cv::Rect(w, 0, input_w_ - w, h))
        .setTo(cv::Scalar::all(pad_value), cvstream_);
    }
    cv::cuda::GpuMat roi = gpu_in_(cv::Rect(0, 0, w, h));
    cv::cuda::resize(gpu_src_, roi, {w, h}, 0.0, 0.0, cv::INTER_LINEAR, cvstream_);

    infer_gpu_internal();
    return download_output();
  }

  cudaEvent_t t0, t_upload, t_letterbox, t_preprocess, t_engine, t_download;
  cudaEventCreate(&t0);
  cudaEventCreate(&t_upload);
  cudaEventCreate(&t_letterbox);
  cudaEventCreate(&t_preprocess);
  cudaEventCreate(&t_engine);
  cudaEventCreate(&t_download);

  cudaEventRecord(t0, stream_);
  gpu_src_.upload(input_bgr_u8, cvstream_);
  cudaEventRecord(t_upload, stream_);

  gpu_in_.create(input_h_, input_w_, input_bgr_u8.type());
  if (h < input_h_) {
    gpu_in_(cv::Rect(0, h, input_w_, input_h_ - h))
      .setTo(cv::Scalar::all(pad_value), cvstream_);
  }
  if (w < input_w_) {
    gpu_in_(cv::Rect(w, 0, input_w_ - w, h))
      .setTo(cv::Scalar::all(pad_value), cvstream_);
  }
  cv::cuda::GpuMat roi = gpu_in_(cv::Rect(0, 0, w, h));
  cv::cuda::resize(gpu_src_, roi, {w, h}, 0.0, 0.0, cv::INTER_LINEAR, cvstream_);
  cudaEventRecord(t_letterbox, stream_);

  preprocess_gpu_internal();
  cudaEventRecord(t_preprocess, stream_);

  enqueue_internal();
  cudaEventRecord(t_engine, stream_);

  cv::Mat output(out_dim1_, out_dim2_, CV_32F);
  cudaMemcpyAsync(
    output.data, outputs_[primary_output_].device, out_bytes_, cudaMemcpyDeviceToHost, stream_);
  cudaEventRecord(t_download, stream_);
  cvstream_.waitForCompletion();

  cudaEventElapsedTime(&profile->upload_ms, t0, t_upload);
  cudaEventElapsedTime(&profile->letterbox_ms, t_upload, t_letterbox);
  cudaEventElapsedTime(&profile->preprocess_ms, t_letterbox, t_preprocess);
  cudaEventElapsedTime(&profile->engine_ms, t_preprocess, t_engine);
  cudaEventElapsedTime(&profile->download_ms, t_engine, t_download);
  cudaEventElapsedTime(&profile->total_ms, t0, t_download);

  cudaEventDestroy(t0);
  cudaEventDestroy(t_upload);
  cudaEventDestroy(t_letterbox);
  cudaEventDestroy(t_preprocess);
  cudaEventDestroy(t_engine);
  cudaEventDestroy(t_download);

  return output;
}

void TRTEngine::infer_gpu_internal()
{
  preprocess_gpu_internal();
  enqueue_internal();
}

void TRTEngine::preprocess_gpu_internal()
{
  if (channels_ == 1) {
    // u8 灰度 HWC -> f32 NCHW /255
    cv::cuda::GpuMat d_in_wrap(input_h_, input_w_, CV_32F, d_in_, input_w_ * sizeof(float));
    gpu_in_.convertTo(d_in_wrap, CV_32F, 1.0 / 255.0, 0.0, cvstream_);
  } else {
    // u8 BGR HWC -> f32 NCHW RGB /255
    cv::cuda::cvtColor(gpu_in_, gpu_rgb_, cv::COLOR_BGR2RGB, 0, cvstream_);
    gpu_rgb_.convertTo(gpu_f32_, CV_32F, 1.0 / 255.0, 0.0, cvstream_);
    cv::cuda::split(gpu_f32_, planes_.data(), cvstream_);  // 写入 d_in_（连续 NCHW，零拷贝输入）
  }
}

void TRTEngine::enqueue_internal()
{
  // 预处理和 TensorRT 推理使用同一条 stream。
  ctx_->enqueueV3(stream_);
}

cv::Mat TRTEngine::download_output()
{
  cv::Mat output(out_dim1_, out_dim2_, CV_32F);
  cudaMemcpyAsync(
    output.data, outputs_[primary_output_].device, out_bytes_, cudaMemcpyDeviceToHost, stream_);
  cvstream_.waitForCompletion();
  return output;
}

}  // namespace auto_aim
