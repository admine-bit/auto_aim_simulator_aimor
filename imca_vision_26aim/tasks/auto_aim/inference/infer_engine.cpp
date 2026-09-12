#include "infer_engine.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

  // 编译时只会保留一个后端，避免未使用的依赖参与链接。
#if defined(IMCA_BACKEND_OPENVINO)
#include "openvino_engine.hpp"
#elif defined(IMCA_BACKEND_TENSORRT)
#include "trt_engine.hpp"
#endif

namespace auto_aim
{
std::unique_ptr<InferEngine> make_infer_engine(
  const std::string & config_path, const std::string & model_key, const std::string & engine_key,
  int input_h, int input_w, int channels)
{
  auto yaml = YAML::LoadFile(config_path);

  // 根据CMake配置选择推理后端
#if defined(IMCA_BACKEND_OPENVINO)
  auto model_path = yaml[model_key].as<std::string>();
  auto device = yaml["device"] ? yaml["device"].as<std::string>() : std::string("CPU");
  return std::make_unique<OpenVINOEngine>(model_path, device, input_h, input_w, channels);
#elif defined(IMCA_BACKEND_TENSORRT)
  auto engine_path = yaml[engine_key].as<std::string>();
  return std::make_unique<TRTEngine>(engine_path, input_h, input_w, channels);
#else
  throw std::runtime_error("No inference backend has been selected");
#endif
}
}  // namespace auto_aim
