#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

// 多线程检测器按编译后端选择各自的异步实现。
#if defined(IMCA_BACKEND_TENSORRT)
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#else
#include <openvino/openvino.hpp>
#endif

#include <chrono>
#include <list>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{
class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);
#if defined(IMCA_BACKEND_TENSORRT)
  ~MultiThreadDetector();
#endif

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  YOLO yolo_;

#if defined(IMCA_BACKEND_TENSORRT)
  // TensorRT 版本共享 engine，每个槽位持有独立 context、stream 和缓冲区。
  struct TRTLogger : nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override;
  } trt_logger_;

  nvinfer1::IRuntime * runtime_ = nullptr;
  nvinfer1::ICudaEngine * engine_ = nullptr;
  std::string in_name_, out_name_;
  int out_dim1_ = 0, out_dim2_ = 0;
  size_t out_bytes_ = 0;

  struct Slot
  {
    nvinfer1::IExecutionContext * ctx = nullptr;
    cv::cuda::Stream cvstream;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    void * d_in = nullptr;
    void * d_out = nullptr;
    cv::cuda::GpuMat gpu_src, gpu_in, gpu_rgb, gpu_f32;
    std::vector<cv::cuda::GpuMat> planes;
    cv::Mat host_src;
    cv::Mat host_out;
  };

  static constexpr int INPUT_HW = 640;
  static constexpr int NUM_SLOTS = 4;
  std::vector<Slot> slots_;
  tools::ThreadSafeQueue<int> free_slots_{NUM_SLOTS};
  tools::ThreadSafeQueue<std::tuple<int, cv::Mat, std::chrono::steady_clock::time_point>>
    trt_queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] trt queue is full!"); }};

  void trt_launch_(int slot, const cv::Mat & img);
  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> pop_impl_();
#else
  // OpenVINO 版本保留主分支的异步 InferRequest 队列。
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::string device_;
  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, ov::InferRequest>>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};
#endif
};
}  // namespace multithread
}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP
