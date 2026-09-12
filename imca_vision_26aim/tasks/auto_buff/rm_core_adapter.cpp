#include "rm_core_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "vc/camera/camera_param.h"
#include "vc/feature/feature_node_child_feature_type.h"
#include "vc/feature/rune_center.h"
#include "vc/feature/rune_combo.h"
#include "vc/feature/rune_fan.h"
#include "vc/feature/rune_group.h"
#include "vc/feature/rune_target.h"
#include "vc/feature/rune_tracker.h"

namespace auto_buff
{
RmCoreAdapter::RmCoreAdapter(const std::string & config_path)
: detector_(RuneDetector::make_detector()),
  feature_nodes_(),
  last_powerrune_(std::nullopt),
  color_channel_(PixChannel::RED),
  red_color_threshold_(38),
  blue_color_threshold_(50),
  color_threshold_(38)
{
  loadConfig(config_path);
}

void RmCoreAdapter::setMyColor(uint8_t my_color)
{
  updateColorConfig(my_color);
}

void RmCoreAdapter::reset()
{
  feature_nodes_.clear();
  last_powerrune_.reset();
}

cv::Mat RmCoreAdapter::binary_debug_image() const
{
  return detector_->binaryImage().clone();
}

void RmCoreAdapter::set_color_threshold(int threshold)
{
  color_threshold_ = std::clamp(threshold, 0, 255);
  color_threshold_overridden_ = true;
}

cv::Mat RmCoreAdapter::binarize(const cv::Mat & img) const
{
  if (img.empty() || img.type() != CV_8UC3) return {};
  cv::Mat binary;
  // 复用 RuneDetector 自己的二值化，保证预览和检测里看到的是同一张图。
  RuneDetector::binary(img, binary, color_channel_, static_cast<uint8_t>(color_threshold_));
  return binary;
}

std::optional<cv::Point2f> RmCoreAdapter::detect_center(
  cv::Mat & img, const Eigen::Quaterniond & q_gimbal2world,
  std::chrono::steady_clock::time_point timestamp)
{
  DetectorInput input;
  DetectorOutput output;
  input.setImage(img);
  const auto tick =
    std::chrono::duration_cast<std::chrono::microseconds>(timestamp.time_since_epoch()).count();
  input.setTick(tick);
  input.setGyroData(quaternionToGyroData(q_gimbal2world));
  input.setColor(color_channel_);
  input.setColorThresh(color_threshold_);
  input.setFeatureNodes(feature_nodes_);

  detector_->detect(input, output);
  feature_nodes_ = output.getFeatureNodes();
  if (img.empty()) return std::nullopt;

  // 这里**故意不看 output.getValid()**。
  // getValid()==false 只说明 rm-core 没能完成整符 PnP / 五组角度构造 / 滤波更新，
  // 与"本帧是否找到了一个通过三层筛选的真 R 轮廓"是两件事。observedRuneCenter() 只在
  // findFeatures 成功的分支被写入（轮廓 + filterCenter 概率投票 + ≥6 点），因此它自身
  // 就是可信度凭据；用 getValid() 再卡一道，等于让姿态域的失败去否决轮廓域的成功，
  // 上层被迫退到 kpt5，实测把符中心拉偏。防误识别改由调用方的 kpt4 门控负责。
  const auto center = detector_->observedRuneCenter();
  if (!center.has_value() || !std::isfinite(center->x) || !std::isfinite(center->y))
    return std::nullopt;
  if (center->x < 0.0F || center->y < 0.0F || center->x >= img.cols || center->y >= img.rows)
    return std::nullopt;
  return center;
}

std::optional<PowerRune> RmCoreAdapter::detect(
  cv::Mat & img, const Eigen::Quaterniond & q_gimbal2world,
  std::chrono::steady_clock::time_point timestamp)
{
  DetectorInput input;
  DetectorOutput output;

  input.setImage(img);
  auto tick =
    std::chrono::duration_cast<std::chrono::microseconds>(timestamp.time_since_epoch()).count();
  input.setTick(tick);
  input.setGyroData(quaternionToGyroData(q_gimbal2world));
  input.setColor(color_channel_);
  input.setColorThresh(color_threshold_);
  input.setFeatureNodes(feature_nodes_);

  detector_->detect(input, output);
  feature_nodes_ = output.getFeatureNodes();
  if (!output.getValid() || feature_nodes_.empty()) {
    return std::nullopt;
  }

  auto rune_group = RuneGroup::cast(feature_nodes_.front());
  if (!rune_group) {
    return std::nullopt;
  }

  // 遍历所有 tracker，收集 PENDING_STRUCK combo 并记录其对应 tracker 索引
  struct TrackerInfo {
    FeatureNode_ptr combo;
    cv::Point2f target_center;
    cv::Point2f r_center;
  };
  std::vector<TrackerInfo> pending_infos;
  // 同时收集所有 tracker 的 fan 角点（用于角点回退）
  struct FanFallback {
    cv::Point2f target_center;  // 该 tracker 的靶心中心
    std::vector<cv::Point2f> corners;  // 该 tracker 的 fan 角点
  };
  std::vector<FanFallback> all_fan_fallbacks;

  for (const auto & tracker : rune_group->getTrackers()) {
    auto rune_tracker = RuneTracker::cast(tracker);
    if (!rune_tracker) continue;
    const auto & history = rune_tracker->getHistoryNodes();
    if (history.empty()) continue;
    auto combo = RuneCombo::cast(std::const_pointer_cast<FeatureNode>(history.front()));
    if (!combo) continue;

    const auto & cf = combo->childFeatures();
    auto fan_it = cf.find(FeatureNode::ChildFeatureType::RUNE_FAN);
    auto target_it = cf.find(FeatureNode::ChildFeatureType::RUNE_TARGET);
    auto center_it = cf.find(FeatureNode::ChildFeatureType::RUNE_CENTER);

    // 收集所有 tracker 的 fan 角点（包括 UNSTRUCK 的 PnP 重建角点）
    if (fan_it != cf.end() && target_it != cf.end()) {
      auto fan = RuneFan::cast(fan_it->second);
      auto t = RuneTarget::cast(target_it->second);
      if (fan && t && !fan->getActiveFlag()) {
        const auto & c = fan->imageCache().getCorners();
        if (c.size() >= 4) {
          all_fan_fallbacks.push_back({
            t->imageCache().getCenter(),
            {c[0], c[1], c[2], c[3]}
          });
        }
      }
    }

    // 收集 PENDING_STRUCK combo
    if (combo->getRuneType() == RuneType::PENDING_STRUCK) {
      cv::Point2f tc(0, 0), rc(0, 0);
      if (target_it != cf.end()) {
        auto t = RuneTarget::cast(target_it->second);
        if (t) tc = t->imageCache().getCenter();
      }
      if (center_it != cf.end()) {
        auto c = RuneCenter::cast(center_it->second);
        if (c) rc = c->imageCache().getCenter();
      }
      pending_infos.push_back({combo, tc, rc});
    }
  }

  if (pending_infos.empty()) {
    return std::nullopt;
  }

  // 选择策略：多个 PENDING_STRUCK 时用上一帧 target 角度做最近匹配
  size_t selected = 0;
  if (pending_infos.size() > 1 && last_powerrune_.has_value()) {
    cv::Point2f last_dir = last_powerrune_->target().center - last_powerrune_->r_center;
    double last_angle = std::atan2(last_dir.y, last_dir.x);
    double min_diff = 1e9;
    for (size_t i = 0; i < pending_infos.size(); i++) {
      cv::Point2f dir = pending_infos[i].target_center - pending_infos[i].r_center;
      double cur_angle = std::atan2(dir.y, dir.x);
      double diff = std::abs(cur_angle - last_angle);
      if (diff > CV_PI) diff = 2 * CV_PI - diff;
      if (diff < min_diff) {
        min_diff = diff;
        selected = i;
      }
    }
  }

  const auto & chosen = pending_infos[selected];
  const auto & child_features = chosen.combo->childFeatures();
  auto fan_it = child_features.find(FeatureNode::ChildFeatureType::RUNE_FAN);
  auto target_it = child_features.find(FeatureNode::ChildFeatureType::RUNE_TARGET);
  auto center_it = child_features.find(FeatureNode::ChildFeatureType::RUNE_CENTER);
  if (fan_it == child_features.end() || target_it == child_features.end() ||
      center_it == child_features.end()) {
    return std::nullopt;
  }

  auto fan = RuneFan::cast(fan_it->second);
  auto target = RuneTarget::cast(target_it->second);
  auto center = RuneCenter::cast(center_it->second);
  if (!fan || !target || !center) {
    return std::nullopt;
  }

  cv::Point2f target_center = target->imageCache().getCenter();
  cv::Point2f r_center = center->imageCache().getCenter();
  std::vector<cv::Point2f> fan_points;

  // 角点获取：优先用 PENDING_STRUCK 自身的 inactive fan 角点
  bool use_fallback = false;
  if (!fan->getActiveFlag()) {
    const auto & corners = fan->imageCache().getCorners();
    if (corners.size() >= 4) {
      fan_points = {corners[0], corners[1], corners[2], corners[3]};
    }
  }

  // 角点回退：当 target fan 不可用（active/角点不足）时，
  // 从同角度位置的其他 tracker 的 fan 角点替代
  if (fan_points.empty() && !all_fan_fallbacks.empty()) {
    double min_dist = 1e9;
    size_t best_idx = 0;
    for (size_t i = 0; i < all_fan_fallbacks.size(); i++) {
      double d = cv::norm(all_fan_fallbacks[i].target_center - target_center);
      if (d < min_dist) {
        min_dist = d;
        best_idx = i;
      }
    }
    fan_points = all_fan_fallbacks[best_idx].corners;
    use_fallback = true;
    tools::logger()->debug("[Adapter] fan fallback: dist={:.1f}px", min_dist);
  }

  if (fan_points.size() < 4) {
    return std::nullopt;
  }

  std::vector<FanBlade> fanblades;
  fanblades.emplace_back(fan_points, target_center, _target);

  PowerRune power_rune(fanblades, r_center, last_powerrune_);
  if (power_rune.is_unsolve()) {
    return std::nullopt;
  }

  last_powerrune_ = power_rune;
  return power_rune;
}

std::pair<std::optional<PowerRune>, std::optional<PowerRune>> RmCoreAdapter::detect_dual(
  cv::Mat & img, const Eigen::Quaterniond & q_gimbal2world,
  std::chrono::steady_clock::time_point timestamp)
{
  DetectorInput input;
  DetectorOutput output;

  input.setImage(img);
  auto tick =
    std::chrono::duration_cast<std::chrono::microseconds>(timestamp.time_since_epoch()).count();
  input.setTick(tick);
  input.setGyroData(quaternionToGyroData(q_gimbal2world));
  input.setColor(color_channel_);
  input.setColorThresh(color_threshold_);
  input.setFeatureNodes(feature_nodes_);

  detector_->detect(input, output);
  feature_nodes_ = output.getFeatureNodes();
  if (!output.getValid() || feature_nodes_.empty()) {
    return {std::nullopt, std::nullopt};
  }

  auto rune_group = RuneGroup::cast(feature_nodes_.front());
  if (!rune_group) {
    return {std::nullopt, std::nullopt};
  }

  // 遍历所有 tracker，收集 PENDING_STRUCK combo
  struct TrackerInfo {
    FeatureNode_ptr combo;
    cv::Point2f target_center;
    cv::Point2f r_center;
  };
  std::vector<TrackerInfo> pending_infos;
  struct FanFallback {
    cv::Point2f target_center;
    std::vector<cv::Point2f> corners;
  };
  std::vector<FanFallback> all_fan_fallbacks;

  for (const auto & tracker : rune_group->getTrackers()) {
    auto rune_tracker = RuneTracker::cast(tracker);
    if (!rune_tracker) continue;
    const auto & history = rune_tracker->getHistoryNodes();
    if (history.empty()) continue;
    auto combo = RuneCombo::cast(std::const_pointer_cast<FeatureNode>(history.front()));
    if (!combo) continue;

    const auto & cf = combo->childFeatures();
    auto fan_it = cf.find(FeatureNode::ChildFeatureType::RUNE_FAN);
    auto target_it = cf.find(FeatureNode::ChildFeatureType::RUNE_TARGET);
    auto center_it = cf.find(FeatureNode::ChildFeatureType::RUNE_CENTER);

    if (fan_it != cf.end() && target_it != cf.end()) {
      auto fan = RuneFan::cast(fan_it->second);
      auto t = RuneTarget::cast(target_it->second);
      if (fan && t && !fan->getActiveFlag()) {
        const auto & c = fan->imageCache().getCorners();
        if (c.size() >= 4) {
          all_fan_fallbacks.push_back({t->imageCache().getCenter(), {c[0], c[1], c[2], c[3]}});
        }
      }
    }

    if (combo->getRuneType() == RuneType::PENDING_STRUCK) {
      cv::Point2f tc(0, 0), rc(0, 0);
      if (target_it != cf.end()) {
        auto t = RuneTarget::cast(target_it->second);
        if (t) tc = t->imageCache().getCenter();
      }
      if (center_it != cf.end()) {
        auto c = RuneCenter::cast(center_it->second);
        if (c) rc = c->imageCache().getCenter();
      }
      pending_infos.push_back({combo, tc, rc});
    }
  }

  if (pending_infos.empty()) {
    return {std::nullopt, std::nullopt};
  }

  // 按上一帧角度排序，选出 primary（最近）和 secondary（次近）
  size_t primary_idx = 0;
  size_t secondary_idx = SIZE_MAX;  // 无效标记

  if (pending_infos.size() > 1 && last_powerrune_.has_value()) {
    cv::Point2f last_dir = last_powerrune_->target().center - last_powerrune_->r_center;
    double last_angle = std::atan2(last_dir.y, last_dir.x);

    // 计算所有角度差并排序
    std::vector<std::pair<double, size_t>> angle_diffs;
    for (size_t i = 0; i < pending_infos.size(); i++) {
      cv::Point2f dir = pending_infos[i].target_center - pending_infos[i].r_center;
      double cur_angle = std::atan2(dir.y, dir.x);
      double diff = std::abs(cur_angle - last_angle);
      if (diff > CV_PI) diff = 2 * CV_PI - diff;
      angle_diffs.push_back({diff, i});
    }
    std::sort(angle_diffs.begin(), angle_diffs.end());
    primary_idx = angle_diffs[0].second;
    secondary_idx = angle_diffs[1].second;
  } else if (pending_infos.size() > 1) {
    // 没有上一帧参考，primary=0, secondary=1
    primary_idx = 0;
    secondary_idx = 1;
  }

  // 构建 PowerRune 的 lambda
  auto build_powerrune = [&](size_t idx) -> std::optional<PowerRune> {
    const auto & info = pending_infos[idx];
    const auto & child_features = info.combo->childFeatures();
    auto fan_it = child_features.find(FeatureNode::ChildFeatureType::RUNE_FAN);
    auto target_it = child_features.find(FeatureNode::ChildFeatureType::RUNE_TARGET);
    auto center_it = child_features.find(FeatureNode::ChildFeatureType::RUNE_CENTER);
    if (fan_it == child_features.end() || target_it == child_features.end() ||
        center_it == child_features.end()) {
      return std::nullopt;
    }
    auto fan = RuneFan::cast(fan_it->second);
    auto target = RuneTarget::cast(target_it->second);
    auto center = RuneCenter::cast(center_it->second);
    if (!fan || !target || !center) return std::nullopt;

    cv::Point2f target_center = target->imageCache().getCenter();
    cv::Point2f r_center = center->imageCache().getCenter();
    std::vector<cv::Point2f> fan_points;

    if (!fan->getActiveFlag()) {
      const auto & corners = fan->imageCache().getCorners();
      if (corners.size() >= 4) {
        fan_points = {corners[0], corners[1], corners[2], corners[3]};
      }
    }
    if (fan_points.empty() && !all_fan_fallbacks.empty()) {
      double min_dist = 1e9;
      size_t best_idx = 0;
      for (size_t i = 0; i < all_fan_fallbacks.size(); i++) {
        double d = cv::norm(all_fan_fallbacks[i].target_center - target_center);
        if (d < min_dist) { min_dist = d; best_idx = i; }
      }
      fan_points = all_fan_fallbacks[best_idx].corners;
    }
    if (fan_points.size() < 4) return std::nullopt;

    std::vector<FanBlade> fanblades;
    fanblades.emplace_back(fan_points, target_center, _target);
    PowerRune pr(fanblades, r_center, last_powerrune_);
    if (pr.is_unsolve()) return std::nullopt;
    return pr;
  };

  // 构建 primary
  auto primary = build_powerrune(primary_idx);
  if (primary.has_value()) {
    last_powerrune_ = primary;
  }

  // 构建 secondary
  std::optional<PowerRune> secondary = std::nullopt;
  if (secondary_idx != SIZE_MAX) {
    secondary = build_powerrune(secondary_idx);
  }

  return {primary, secondary};
}

GyroData RmCoreAdapter::quaternionToGyroData(const Eigen::Quaterniond & q) const
{
  const double qw = q.w();
  const double qx = q.x();
  const double qy = q.y();
  const double qz = q.z();
  double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  double sinp = 2.0 * (qw * qy - qz * qx);
  sinp = std::clamp(sinp, -1.0, 1.0);
  double pitch = std::asin(sinp);

  GyroData gyro_data;
  gyro_data.rotation.yaw = yaw * 180.0 / CV_PI;
  gyro_data.rotation.pitch = pitch * 180.0 / CV_PI;
  return gyro_data;
}

void RmCoreAdapter::updateColorConfig(uint8_t my_color)
{
  // Buff 与自身同色，而自瞄识别敌方颜色：这里保持与自瞄相反。
  // gimbal my_color: 0 蓝，1 红。
  if (my_color == 0U) {
    color_channel_ = PixChannel::BLUE;
    if (!color_threshold_overridden_) color_threshold_ = blue_color_threshold_;
  } else {
    color_channel_ = PixChannel::RED;
    if (!color_threshold_overridden_) color_threshold_ = red_color_threshold_;
  }
}

void RmCoreAdapter::loadConfig(const std::string & config_path)
{
  try {
    YAML::Node config = YAML::LoadFile(config_path);

    if (config["camera_matrix"]) {
      const auto cm = config["camera_matrix"].as<std::vector<double>>();
      if (cm.size() == 9) {
        camera_param.cameraMatrix = cv::Matx33f(
          static_cast<float>(cm[0]), static_cast<float>(cm[1]), static_cast<float>(cm[2]),
          static_cast<float>(cm[3]), static_cast<float>(cm[4]), static_cast<float>(cm[5]),
          static_cast<float>(cm[6]), static_cast<float>(cm[7]), static_cast<float>(cm[8]));
      }
    }
    if (config["distort_coeffs"]) {
      const auto dc = config["distort_coeffs"].as<std::vector<double>>();
      if (dc.size() >= 5) {
        camera_param.distCoeff = cv::Matx<float, 5, 1>(
          static_cast<float>(dc[0]), static_cast<float>(dc[1]), static_cast<float>(dc[2]),
          static_cast<float>(dc[3]), static_cast<float>(dc[4]));
      }
    }
    if (config["R_camera2gimbal"]) {
      const auto r = config["R_camera2gimbal"].as<std::vector<double>>();
      if (r.size() == 9) {
        camera_param.cam2joint_rmat = cv::Matx33f(
          static_cast<float>(r[0]), static_cast<float>(r[1]), static_cast<float>(r[2]),
          static_cast<float>(r[3]), static_cast<float>(r[4]), static_cast<float>(r[5]),
          static_cast<float>(r[6]), static_cast<float>(r[7]), static_cast<float>(r[8]));
      }
    }
    if (config["t_camera2gimbal"]) {
      const auto t = config["t_camera2gimbal"].as<std::vector<double>>();
      if (t.size() == 3) {
        camera_param.cam2joint_tvec = cv::Matx<float, 3, 1>(
          static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[2]));
      }
    }
    if (config["color_threshold_red"]) {
      red_color_threshold_ = config["color_threshold_red"].as<int>();
    }
    if (config["color_threshold_blue"]) {
      blue_color_threshold_ = config["color_threshold_blue"].as<int>();
    }
    if (config["color_threshold"]) {
      const auto threshold = config["color_threshold"].as<int>();
      red_color_threshold_ = threshold;
      blue_color_threshold_ = threshold;
    }
    // 离线回放没有云台 my_color 输入；enemy_color 是自瞄识别色，Buff 取其反色。
    // 实时入口每帧仍由 setMyColor() 按 0→BLUE、1→RED 覆盖。
    if (config["enemy_color"]) {
      const auto enemy_color = config["enemy_color"].as<std::string>();
      color_channel_ = (enemy_color == "red") ? PixChannel::BLUE : PixChannel::RED;
    }
    color_threshold_ =
      (color_channel_ == PixChannel::RED) ? red_color_threshold_ : blue_color_threshold_;
    tools::logger()->info(
      "[RmCoreAdapter] color thresholds red={} blue={} current={}",
      red_color_threshold_, blue_color_threshold_, color_threshold_);

  } catch (const std::exception & e) {
    tools::logger()->warn("[RmCoreAdapter] Failed to load config: {}", e.what());
  }
}
}  // namespace auto_buff
