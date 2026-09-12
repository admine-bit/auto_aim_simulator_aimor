#ifndef AUTO_BUFF__RM_CORE_ADAPTER_HPP
#define AUTO_BUFF__RM_CORE_ADAPTER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "buff_type.hpp"
#include "vc/detector/rune_detector.h"

namespace auto_buff
{
class RmCoreAdapter
{
public:
  explicit RmCoreAdapter(const std::string & config_path);

  void setMyColor(uint8_t my_color);
  void reset();

  /// 运行一次完整 RuneDetector，但只返回本帧轮廓真实观测到的 R 标中心。
  /// 强制构造、历史重投影和越界中心均返回 nullopt。
  ///
  /// 与 rm-core 自身的 PnP/姿态阶段成败**解耦**：只要轮廓阶段（轮廓查找 +
  /// filterCenter 概率投票 + 轮廓 ≥6 点）成功就返回中心，即使 rm-core 随后
  /// 因 angle not finite、checkPoseDiff 创新门限、极端姿态等原因整帧判无效。
  /// 防误识别不在这一层做，由调用方用 YOLO kpt4 门控。
  std::optional<cv::Point2f> detect_center(
    cv::Mat & img, const Eigen::Quaterniond & q_gimbal2world,
    std::chrono::steady_clock::time_point timestamp);

  std::optional<PowerRune> detect(
    cv::Mat & img, const Eigen::Quaterniond & q_gimbal2world,
    std::chrono::steady_clock::time_point timestamp);

  /// 大符双目标检测：返回 {primary, secondary}，secondary 为第二个 PENDING_STRUCK 目标
  /// 小符不调用此接口，保持 detect() 不变
  std::pair<std::optional<PowerRune>, std::optional<PowerRune>> detect_dual(
    cv::Mat & img, const Eigen::Quaterniond & q_gimbal2world,
    std::chrono::steady_clock::time_point timestamp);

  /// 强制设置 last_powerrune_，用于大符双目标切换跟踪
  void set_last_powerrune(const PowerRune & pr) { last_powerrune_ = pr; }

  /// 返回本次 rm-core 检测真正使用的二值图快照，仅供调用方调试显示。
  cv::Mat binary_debug_image() const;

  /// 当前生效的颜色二值化阈值。setMyColor() 会按敌我颜色改写它。
  int color_threshold() const { return color_threshold_; }

  /// 手动覆盖颜色二值化阈值（调试拖动条用）。
  /// 覆盖后 setMyColor() 不再改写阈值，只切换通道，否则每帧都会被 yaml 值顶回去。
  void set_color_threshold(int threshold);

  /// 用当前通道和阈值单独跑一遍二值化，不触发检测。
  /// 调试界面暂停时拖动阈值也能立刻看到效果。
  cv::Mat binarize(const cv::Mat & img) const;

  // 获取 feature_nodes 用于调试绘制
  const std::vector<FeatureNode_ptr> & getFeatureNodes() const { return feature_nodes_; }

private:
  void loadConfig(const std::string & config_path);
  GyroData quaternionToGyroData(const Eigen::Quaterniond & q) const;
  void updateColorConfig(uint8_t my_color);

  std::shared_ptr<RuneDetector> detector_;
  std::vector<FeatureNode_ptr> feature_nodes_;
  std::optional<PowerRune> last_powerrune_;

  PixChannel color_channel_;
  int red_color_threshold_;
  int blue_color_threshold_;
  int color_threshold_;
  // 调试拖动条一旦写过阈值，就不再让 setMyColor() 用 yaml 值覆盖。
  bool color_threshold_overridden_{false};
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__RM_CORE_ADAPTER_HPP
