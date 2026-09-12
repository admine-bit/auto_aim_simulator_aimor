#include "openvino_engine.hpp"

#include <algorithm>
#include <stdexcept>

namespace auto_aim
{
OpenVINOEngine::OpenVINOEngine(
  const std::string & model_path, const std::string & device, int input_h, int input_w,
  int channels)
: input_h_(input_h), input_w_(input_w), channels_(channels)
{
  auto model = core_.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  // 与主分支保持相同的输入布局和归一化方式。
  if (channels_ == 1) {
    input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, 1, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_)})
      .set_layout("NCHW");
    input.model().set_layout("NCHW");
  } else {
    input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_), 3})
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);
    input.model().set_layout("NCHW");
    input.preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
  }

  input.preprocess().convert_element_type(ov::element::f32).scale(255.0);
  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

cv::Mat OpenVINOEngine::infer(const cv::Mat & input)
{
  if (input.empty()) throw std::runtime_error("OpenVINOEngine: empty input");
  if (input.channels() != channels_)
    throw std::runtime_error("OpenVINOEngine: unexpected input channels");

  // OpenVINO Tensor 直接引用 Mat 数据，输入必须连续存储。
  cv::Mat contiguous = input.isContinuous() ? input : input.clone();
  ov::Shape shape;
  if (channels_ == 1) {
    shape = {1, 1, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_)};
  } else {
    shape = {1, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_), 3};
  }

  // 每次推理独立创建请求，避免调用间共享输出缓冲区。
  auto request = compiled_model_.create_infer_request();
  request.set_input_tensor(ov::Tensor(ov::element::u8, shape, contiguous.data));
  request.infer();
  return get_output(request.get_output_tensor());
}

cv::Mat OpenVINOEngine::infer(const cv::cuda::GpuMat & input)
{
  cv::Mat host_input;
  input.download(host_input);
  return infer(host_input);
}

cv::Mat OpenVINOEngine::infer_letterbox(const cv::Mat & input, double & scale, int pad_value)
{
  if (input.empty()) throw std::runtime_error("OpenVINOEngine: empty input");

  const auto x_scale = static_cast<double>(input_h_) / input.rows;
  const auto y_scale = static_cast<double>(input_w_) / input.cols;
  scale = std::min(x_scale, y_scale);
  const auto h = static_cast<int>(input.rows * scale);
  const auto w = static_cast<int>(input.cols * scale);

  // Letterbox 左上对齐，与原有 YOLO 推理逻辑一致。
  const int type = channels_ == 1 ? CV_8UC1 : CV_8UC3;
  cv::Mat letterboxed(input_h_, input_w_, type, cv::Scalar::all(pad_value));
  cv::resize(input, letterboxed(cv::Rect(0, 0, w, h)), {w, h});
  return infer(letterboxed);
}

cv::Mat OpenVINOEngine::get_output(const ov::Tensor & tensor) const
{
  const auto shape = tensor.get_shape();
  if (shape.size() < 2) throw std::runtime_error("OpenVINOEngine: unsupported output shape");

  const int rows = shape.size() == 2 ? static_cast<int>(shape[0]) : static_cast<int>(shape[1]);
  const int cols = shape.size() == 2 ? static_cast<int>(shape[1]) : static_cast<int>(shape[2]);
  // InferRequest 销毁后输出可能失效，因此返回独立 Mat。
  // cv::Mat 的外部数据构造函数需要 void *，clone 后不再依赖 OpenVINO 输出内存。
  auto data = const_cast<float *>(tensor.data<const float>());
  return cv::Mat(rows, cols, CV_32F, data).clone();
}
}  // namespace auto_aim
