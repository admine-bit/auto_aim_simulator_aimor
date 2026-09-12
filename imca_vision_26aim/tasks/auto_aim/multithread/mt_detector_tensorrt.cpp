#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <stdexcept>

namespace auto_aim
{
namespace multithread
{

void MultiThreadDetector::TRTLogger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
}

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();

  // 多槽位复用同一个 engine，减少反序列化和显存占用。
  // ---------- TensorRT 异步路径：共享 engine + NUM_SLOTS 个 context 槽位 ----------
  auto engine_path = yaml[yolo_name + "_engine_path"].as<std::string>();
  std::ifstream file(engine_path, std::ios::binary);
  if (!file.good())
    throw std::runtime_error("[MultiThreadDetector] cannot open engine: " + engine_path);
  std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  runtime_ = nvinfer1::createInferRuntime(trt_logger_);
  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
  if (!engine_)
    throw std::runtime_error("[MultiThreadDetector] deserializeCudaEngine failed: " + engine_path);

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      in_name_ = name;
    else
      out_name_ = name;
  }

  slots_.resize(NUM_SLOTS);
  for (int i = 0; i < NUM_SLOTS; ++i) {
    auto & s = slots_[i];
    s.ctx = engine_->createExecutionContext();
    s.ctx->setInputShape(in_name_.c_str(), nvinfer1::Dims4{1, 3, INPUT_HW, INPUT_HW});

    if (i == 0) {
      auto od = s.ctx->getTensorShape(out_name_.c_str());
      out_dim1_ = od.d[1];
      out_dim2_ = od.d[2];
      out_bytes_ = static_cast<size_t>(out_dim1_) * out_dim2_ * sizeof(float);
    }

    s.stream = cv::cuda::StreamAccessor::getStream(s.cvstream);
    cudaEventCreate(&s.done);
    cudaMalloc(&s.d_in, static_cast<size_t>(3) * INPUT_HW * INPUT_HW * sizeof(float));
    cudaMalloc(&s.d_out, out_bytes_);

    auto * base = static_cast<float *>(s.d_in);
    size_t plane = static_cast<size_t>(INPUT_HW) * INPUT_HW;
    size_t step = static_cast<size_t>(INPUT_HW) * sizeof(float);
    s.planes = {
      cv::cuda::GpuMat(INPUT_HW, INPUT_HW, CV_32F, base + 0 * plane, step),
      cv::cuda::GpuMat(INPUT_HW, INPUT_HW, CV_32F, base + 1 * plane, step),
      cv::cuda::GpuMat(INPUT_HW, INPUT_HW, CV_32F, base + 2 * plane, step)};

    s.ctx->setTensorAddress(in_name_.c_str(), s.d_in);
    s.ctx->setTensorAddress(out_name_.c_str(), s.d_out);
    s.host_out.create(out_dim1_, out_dim2_, CV_32F);

    free_slots_.push(i);
  }

  tools::logger()->info("[MultiThreadDetector] initialized (TensorRT, {} slots)!", NUM_SLOTS);
}

MultiThreadDetector::~MultiThreadDetector()
{
  for (auto & s : slots_) {
    if (s.done) cudaEventDestroy(s.done);
    if (s.d_in) cudaFree(s.d_in);
    if (s.d_out) cudaFree(s.d_out);
    delete s.ctx;
  }
  delete engine_;
  delete runtime_;
}

void MultiThreadDetector::trt_launch_(int slot, const cv::Mat & img)
{
  auto & s = slots_[slot];

  // 预处理、推理和回传全部进入该槽位独立的 CUDA stream。
  auto x_scale = static_cast<double>(INPUT_HW) / img.rows;
  auto y_scale = static_cast<double>(INPUT_HW) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // GPU letterbox + 预处理：u8 BGR -> f32 NCHW RGB /255，全程 s.stream
  s.host_src = img;
  s.gpu_src.upload(s.host_src, s.cvstream);
  s.gpu_in.create(INPUT_HW, INPUT_HW, CV_8UC3);
  s.gpu_in.setTo(cv::Scalar(0, 0, 0), s.cvstream);
  cv::cuda::GpuMat roi = s.gpu_in(cv::Rect(0, 0, w, h));
  cv::cuda::resize(s.gpu_src, roi, {w, h}, 0.0, 0.0, cv::INTER_LINEAR, s.cvstream);
  cv::cuda::cvtColor(s.gpu_in, s.gpu_rgb, cv::COLOR_BGR2RGB, 0, s.cvstream);
  s.gpu_rgb.convertTo(s.gpu_f32, CV_32F, 1.0 / 255.0, 0.0, s.cvstream);
  cv::cuda::split(s.gpu_f32, s.planes.data(), s.cvstream);

  s.ctx->enqueueV3(s.stream);
  cudaMemcpyAsync(s.host_out.data, s.d_out, out_bytes_, cudaMemcpyDeviceToHost, s.stream);
  cudaEventRecord(s.done, s.stream);
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  // TensorRT：取空闲槽位（无空闲则丢帧；假定单生产者线程）
  // 没有空闲槽位时丢帧，避免阻塞相机线程。
  if (free_slots_.empty()) {
    tools::logger()->debug("[MultiThreadDetector] no free slot, drop frame!");
    return;
  }
  int slot = free_slots_.pop();
  trt_launch_(slot, img);
  trt_queue_.push({slot, img.clone(), t});
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::pop_impl_()
{
  auto [slot, img, t] = trt_queue_.pop();
  // 等待当前槽位的 D2H 回传完成，再交给主分支后处理。
  cudaEventSynchronize(slots_[slot].done);  // 等待该槽位 D2H 完成
  cv::Mat output = slots_[slot].host_out;   // 槽位仍归本帧所有，数据稳定
  auto scale =
    std::min(static_cast<double>(INPUT_HW) / img.rows, static_cast<double>(INPUT_HW) / img.cols);
  auto armors = yolo_.postprocess(scale, output, img, 0);
  free_slots_.push(slot);  // 释放槽位（postprocess 完成后）
  return {img, std::move(armors), t};
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, armors, t] = pop_impl_();
  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  return pop_impl_();
}

}  // namespace multithread

}  // namespace auto_aim
