#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/inference/infer_engine.hpp"
#include "tools/logger.hpp"

namespace auto_buff
{
const std::vector<std::string> class_names = {"buff", "buf1"};

class YOLO11_BUFF
{
public:
  struct Object
  {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<cv::Point2f> kpt;
    std::vector<float> kpt_prob;

    bool is_active() const { return label == 1; }
  };

  explicit YOLO11_BUFF(const std::string & config);

  // 使用NMS，用来获取多个框
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);

  // 寻找置信度最高的框
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

  // 分阶段计数用于判断第二目标究竟在置信度门控、NMS 还是后续关联阶段消失。
  std::size_t last_score_count() const { return last_score_count_; }
  std::size_t last_candidate_count() const { return last_candidate_count_; }
  std::size_t last_flow_valid_count() const { return last_flow_valid_count_; }
  std::size_t last_active_count() const { return last_active_count_; }
  std::size_t last_nms_count() const { return last_nms_count_; }

private:
  std::unique_ptr<auto_aim::InferEngine> engine_;
  const int NUM_POINTS = 6;
  // 这些阈值由对应 YAML 加载，避免运行时使用隐藏的编译期常量。
  float confidence_threshold_{0.5F};
  float keypoint_threshold_{0.5F};
  float iou_threshold_{0.4F};
  std::size_t last_score_count_{0};
  std::size_t last_candidate_count_{0};
  std::size_t last_flow_valid_count_{0};
  std::size_t last_active_count_{0};
  std::size_t last_nms_count_{0};

  // 将image保存为"../result/$${programName}.jpg"
  void save(const std::string & programName, const cv::Mat & image);
};
}  // namespace auto_buff
#endif
