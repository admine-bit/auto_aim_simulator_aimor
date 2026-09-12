#ifndef AUTO_AIM__INFER_ENGINE_HPP
#define AUTO_AIM__INFER_ENGINE_HPP

#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <string>

namespace auto_aim
{
// OpenVINO 与 TensorRT 共用的推理接口，业务层不直接依赖后端 API。
class InferEngine
{
public:
  virtual ~InferEngine() = default;

  virtual cv::Mat infer(const cv::Mat & input_bgr_u8) = 0;
  virtual cv::Mat infer(const cv::cuda::GpuMat & input_bgr_u8) = 0;
  virtual cv::Mat infer_letterbox(
    const cv::Mat & input_bgr_u8, double & scale, int pad_value = 0) = 0;

  virtual int input_h() const = 0;
  virtual int input_w() const = 0;
};

// 后端由 cmake/inference_backend.cmake 在编译前确定。
std::unique_ptr<InferEngine> make_infer_engine(
  const std::string & config_path, const std::string & model_key, const std::string & engine_key,
  int input_h, int input_w, int channels = 3);

}  // namespace auto_aim

#endif  // AUTO_AIM__INFER_ENGINE_HPP
