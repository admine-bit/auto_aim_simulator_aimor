#include "yolo11_buff.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <yaml-cpp/yaml.h>

namespace
{
// Ultralytics 导出的模型使用 114 做 letterbox 填充，坐标随后按 scale 还原到原图。
constexpr int LetterboxPadValue = 114;
constexpr int RCenterIndex = 4;
constexpr int BoxValueCount = 4;
constexpr int ClassCount = 2;
}  // namespace

namespace auto_buff
{
YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
: engine_(auto_aim::make_infer_engine(config, "model", "yolo11_buff_engine_path", 640, 640))
{
  const auto yaml = YAML::LoadFile(config);
  confidence_threshold_ = yaml["buff_yolo_confidence_threshold"].as<float>();
  keypoint_threshold_ = yaml["buff_yolo_keypoint_threshold"].as<float>();
  iou_threshold_ = yaml["buff_yolo_nms_iou_threshold"].as<float>();
  if (
    !std::isfinite(confidence_threshold_) || confidence_threshold_ < 0.0F ||
    confidence_threshold_ > 1.0F || !std::isfinite(keypoint_threshold_) ||
    keypoint_threshold_ < 0.0F || keypoint_threshold_ > 1.0F ||
    !std::isfinite(iou_threshold_) || iou_threshold_ < 0.0F || iou_threshold_ > 1.0F)
    throw std::runtime_error(
      "Buff YOLO thresholds must be finite values in the range [0, 1]");
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  last_score_count_ = 0;
  last_candidate_count_ = 0;
  last_flow_valid_count_ = 0;
  last_active_count_ = 0;
  last_nms_count_ = 0;
  const int64 start = cv::getTickCount();

  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  double scale = 1.0;
  cv::Mat det_output = engine_->infer_letterbox(image, scale, LetterboxPadValue);
  const float factor = static_cast<float>(1.0 / scale);
  // YOLO11 Pose 输出为 4 框参数 + 类别分数 + 6*(x,y,conf)。
  // 旧模型是 [1,23,8400]；加入 buf1 后为 [1,24,8400]。
  const int legacy_rows = BoxValueCount + 1 + NUM_POINTS * 3;
  const int multi_class_rows = BoxValueCount + ClassCount + NUM_POINTS * 3;
  int class_count = 0;
  int keypoint_offset = 0;
  if (det_output.rows == legacy_rows) {
    class_count = 1;
    keypoint_offset = BoxValueCount + class_count;
  } else if (det_output.rows == multi_class_rows) {
    class_count = ClassCount;
    keypoint_offset = BoxValueCount + class_count;
  }
  if (class_count == 0 || det_output.cols != 8400) {
    tools::logger()->error("Unexpected buff output shape: {}x{}", det_output.rows, det_output.cols);
    return {};
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> labels;
  std::vector<std::vector<float>> objects_keypoints;

  // 类别顺序固定：0=buff（带六点），1=buf1（激活状态标记，无关键点语义）。
  // 对 buff，关键点顺序固定为 kpt0~3 target 四角、kpt4 R 中心、kpt5 流水灯中心。
  for (int i = 0; i < det_output.cols; ++i) {
    int label = 0;
    float score = -std::numeric_limits<float>::infinity();
    for (int cls = 0; cls < class_count; ++cls) {
      const float class_score = det_output.at<float>(BoxValueCount + cls, i);
      if (std::isfinite(class_score) && class_score > score) {
        score = class_score;
        label = cls;
      }
    }
    if (!std::isfinite(score) || score <= confidence_threshold_) continue;
    ++last_score_count_;

    std::vector<float> keypoints;
    keypoints.reserve(NUM_POINTS * 3);
    bool valid = true;
    if (label == 0) {
      for (int j = 0; j < NUM_POINTS; ++j) {
        const float x = det_output.at<float>(keypoint_offset + j * 3, i);
        const float y = det_output.at<float>(keypoint_offset + j * 3 + 1, i);
        const float confidence = det_output.at<float>(keypoint_offset + j * 3 + 2, i);
        keypoints.push_back(x * factor);
        keypoints.push_back(y * factor);
        keypoints.push_back(confidence);

        // 五点 IPPE 使用 kpt0~3 + kpt5，因此这些点必须是本帧有效观测。
        // kpt4(R) 仍然只作 R 中心候选，置信度不足不会阻断 buff 几何检测。
        if ((j < 4 || j == 5) &&
            (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(confidence) ||
             confidence <= keypoint_threshold_))
          valid = false;
      }
    }
    if (!valid) continue;

    const float cx = det_output.at<float>(0, i) * factor;
    const float cy = det_output.at<float>(1, i) * factor;
    const float ow = det_output.at<float>(2, i) * factor;
    const float oh = det_output.at<float>(3, i) * factor;
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(ow) ||
        !std::isfinite(oh) || ow <= 0.0F || oh <= 0.0F)
      continue;

    boxes.emplace_back(
      static_cast<int>(cx - 0.5F * ow), static_cast<int>(cy - 0.5F * oh),
      static_cast<int>(ow), static_cast<int>(oh));
    confidences.push_back(score);
    labels.push_back(label);
    objects_keypoints.emplace_back(std::move(keypoints));
    if (label == 0) {
      const auto & saved_keypoints = objects_keypoints.back();
      if (
        std::isfinite(saved_keypoints[5 * 3]) && std::isfinite(saved_keypoints[5 * 3 + 1]) &&
        std::isfinite(saved_keypoints[5 * 3 + 2]) &&
        saved_keypoints[5 * 3 + 2] > keypoint_threshold_)
        ++last_flow_valid_count_;
    }
  }
  last_candidate_count_ = boxes.size();

  // 普通 AABB NMS 会把“共享同一个 R、但 target 不同”的两片真实扇叶互相压掉。
  // 这里仍按置信度做一次 NMS，但只有框重叠且四角中心也几乎相同时才认为是重复框。
  // 因此同一 target 的重复候选会被删除，而两个物理 target 可以同时进入 detect_dual()。
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return confidences[lhs] > confidences[rhs];
  });

  auto target_center_at = [&](int index) {
    if (labels[index] != 0 || objects_keypoints[index].size() < NUM_POINTS * 3)
      return cv::Point2f(
        boxes[index].x + boxes[index].width * 0.5F,
        boxes[index].y + boxes[index].height * 0.5F);

    cv::Point2f center{};
    for (int point = 0; point < 4; ++point) {
      center.x += objects_keypoints[index][point * 3];
      center.y += objects_keypoints[index][point * 3 + 1];
    }
    return center * 0.25F;
  };

  std::vector<int> indexes;
  indexes.reserve(order.size());
  for (const int candidate : order) {
    if (candidate < 0 || static_cast<std::size_t>(candidate) >= boxes.size()) continue;
    bool duplicate = false;
    const cv::Point2f candidate_center = target_center_at(candidate);
    for (const int kept : indexes) {
      if (kept < 0 || static_cast<std::size_t>(kept) >= boxes.size()) continue;
      // buf1 是与 buff 配对的激活状态标记，不能因为框重叠而被 buff NMS 删除。
      if (labels[candidate] != labels[kept]) continue;
      const cv::Rect intersection = boxes[candidate] & boxes[kept];
      const double intersection_area = static_cast<double>(intersection.area());
      const double union_area = static_cast<double>(boxes[candidate].area()) +
                                static_cast<double>(boxes[kept].area()) - intersection_area;
      const double iou = union_area > 0.0 ? intersection_area / union_area : 0.0;
      const double candidate_diagonal = std::hypot(boxes[candidate].width, boxes[candidate].height);
      const double kept_diagonal = std::hypot(boxes[kept].width, boxes[kept].height);
      const double same_target_gate =
        std::max(6.0, 0.20 * std::min(candidate_diagonal, kept_diagonal));
      const bool duplicate = labels[candidate] == 0
        ? iou > iou_threshold_ &&
          cv::norm(candidate_center - target_center_at(kept)) < same_target_gate
        : iou > iou_threshold_;
      if (duplicate) {
        break;
      }
    }
    if (!duplicate) indexes.push_back(candidate);
  }
  last_nms_count_ = indexes.size();

  std::vector<Object> object_result;
  object_result.reserve(indexes.size());
  for (const auto index : indexes) {
    Object obj;
    obj.rect = boxes[index];
    obj.label = labels[index];
    obj.prob = confidences[index];
    if (obj.label == 0) {
      // 按模型原始顺序保存六点坐标和各自置信度，供共享 R 与 PnP 分工使用。
      for (int i = 0; i < NUM_POINTS; ++i) {
        obj.kpt.emplace_back(
          objects_keypoints[index][i * 3], objects_keypoints[index][i * 3 + 1]);
        obj.kpt_prob.push_back(objects_keypoints[index][i * 3 + 2]);
      }
    } else {
      ++last_active_count_;
    }
    object_result.push_back(obj);

    const cv::Scalar box_color = obj.is_active() ? cv::Scalar(0, 0, 255)
                                                 : cv::Scalar(255, 255, 255);
    cv::rectangle(image, obj.rect, box_color, 1, 8);
    const std::string label = class_names.at(obj.label) + ":" +
                              std::to_string(obj.prob).substr(0, 4);
    const cv::Size text_size =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect text_box(
      obj.rect.tl().x, obj.rect.tl().y - 15, text_size.width, text_size.height + 5);
    cv::rectangle(image, text_box, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
      image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    for (int i = 0; i < static_cast<int>(obj.kpt.size()); ++i) {
      if (!std::isfinite(obj.kpt[i].x) || !std::isfinite(obj.kpt[i].y)) continue;
      // 原始 YOLO kpt4 使用空心橙色点显示，避免与后续计算出的唯一共享 R 混淆。
      const cv::Scalar color = i == RCenterIndex ? cv::Scalar(0, 165, 255)
                                                  : cv::Scalar(255, 255, 0);
      cv::circle(
        image, obj.kpt[i], i == RCenterIndex ? 3 : 2, color,
        i == RCenterIndex ? 1 : -1, cv::LINE_AA);
      cv::putText(
        image, std::to_string(i), obj.kpt[i] + cv::Point2f(5, -5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
  }

  const float t =
    (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40),
    cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  // 单目标接口复用多目标解析/NMS，保证六关键点步长和硬/弱门控只有一份实现。
  auto objects = get_multicandidateboxes(image);
  if (objects.empty()) return {};

  objects.erase(
    std::remove_if(objects.begin(), objects.end(), [](const Object & object) {
      return object.is_active() || object.kpt.size() != 6;
    }), objects.end());
  if (objects.empty()) return {};

  const auto best = std::max_element(
    objects.begin(), objects.end(),
    [](const Object & lhs, const Object & rhs) { return lhs.prob < rhs.prob; });
  if (best->prob < 0.7F) save(std::to_string(cv::getTickCount()), image);
  return {*best};
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}
}  // namespace auto_buff
