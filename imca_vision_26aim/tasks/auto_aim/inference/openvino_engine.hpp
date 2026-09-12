#ifndef AUTO_AIM__OPENVINO_ENGINE_HPP
#define AUTO_AIM__OPENVINO_ENGINE_HPP

#include <openvino/openvino.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "infer_engine.hpp"

namespace auto_aim
{
class OpenVINOEngine : public InferEngine
{
public:
  OpenVINOEngine(
    const std::string & model_path, const std::string & device, int input_h, int input_w,
    int channels = 3);

  cv::Mat infer(const cv::Mat & input) override;
  cv::Mat infer(const cv::cuda::GpuMat & input) override;
  cv::Mat infer_letterbox(
    const cv::Mat & input, double & scale, int pad_value = 0) override;

  int input_h() const override { return input_h_; }
  int input_w() const override { return input_w_; }

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  int input_h_, input_w_, channels_;

  cv::Mat get_output(const ov::Tensor & tensor) const;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__OPENVINO_ENGINE_HPP
