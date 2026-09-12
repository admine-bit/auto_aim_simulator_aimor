#ifndef AUTO_BUFF__DETECTOR_HPP
#define AUTO_BUFF__DETECTOR_HPP

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "buff_type.hpp"
#include "rm_core_adapter.hpp"
#include "yolo11_buff.hpp"

namespace auto_buff
{
/// R 中心的二维（像素域）融合滤波器：把两个互补的 R 源融成一个。
///
/// 为什么在像素域、PnP 之前滤：PnP 的 tvec 对 R 像素近乎 1:1 敏感，在这里降噪
/// 等价于直接降 tvec 的噪声。等到 PnP 之后（下游 RCenterFilter 的 0.30 m 创新门）
/// 再处理，坏 R 已经污染了整个位姿，只能整帧丢弃。
///
/// 两个源的分工来自它们正交的失效模式：
///   rm-core 轮廓 R —— 准（小 σ），但会丢识别、会误识别（单帧离群）
///   融合 kpt4      —— 一直在，但不准（大 σ，噪声主导，可按 样本散布/√n 在线估）
/// 因此这不是"二选一"，而是两个测量按各自协方差加权。关键收益：拒掉某一帧的
/// rm 测量不会饿死估计（kpt4 仍在更新），所以**不需要任何逃生机制**，也就没有
/// 定期放行一帧坏 R 的延迟代价。
///
/// 运动模型：恒速度。车不平移、R 是固定结构，图像里的 R 只随云台转动漂移，
/// 实测量级约 3 px/帧（0.7 m 半径 / 7 m 距离 / 符 1 rad/s / 60 fps），是一条匀速项，
/// 恒速度模型可以吃掉，只在云台加速段留残差。云台补偿预测（K·R·K⁻¹ 无穷远单应，
/// 纯旋转下不需要深度）是更彻底的做法，但要引入相机内参并依赖外参与时间同步的
/// 正确性——那两项目前还在待查清单上，所以先不引入，把残差留成诊断量。
class RPixelFusion
{
public:
  struct Measurement
  {
    cv::Point2f point;
    double sigma_px;  // 该测量的各向同性标准差
  };

  /// 用本帧可得的测量推进一步，返回融合后的 R 像素位置。
  /// 两个测量都可缺省；都缺省且滤波器未初始化时返回 nullopt。
  std::optional<cv::Point2f> update(
    const std::optional<Measurement> & rm, const std::optional<Measurement> & kpt4,
    double timestamp_s);

  void reset();

  bool initialized() const { return initialized_; }
  /// 融合后位置标准差（两轴较大者），PnP 权重由它决定。
  double position_sigma_px() const;
  /// 上一帧鲁棒更新对 rm 测量协方差的膨胀倍数；1 表示未膨胀。
  double last_rm_inflation() const { return last_rm_inflation_; }

private:
  void predict(double dt);
  /// 单个测量的标准卡尔曼校正。inflation 用于鲁棒（Huber）软降权。
  void correct(const cv::Point2f & z, double sigma_px, double * inflation_out);

  bool initialized_{false};
  double last_timestamp_s_{0.0};
  double last_rm_inflation_{1.0};
  // 状态 [x, y, vx, vy]，单位 px 与 px/s。
  cv::Matx41d x_{0.0, 0.0, 0.0, 0.0};
  cv::Matx44d P_{cv::Matx44d::eye()};
};

/// 将一帧内全部 YOLO Pose 候选合并为同一个五槽 PowerRune。
/// 主目标和副目标只是同一刚体的不同槽位视图，共享 R 中心和 frame_id。
class Buff_Detector
{
public:
  explicit Buff_Detector(const std::string & config);

  // YOLO 仍只有 buff 一个类别；颜色仅转发给 rm-core R 轮廓检测。
  void setMyColor(uint8_t my_color);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);

  std::optional<PowerRune> detect(
    cv::Mat & bgr_img, const Eigen::Quaterniond & q_gimbal2world,
    std::chrono::steady_clock::time_point timestamp);

  /// 主目标槽位不会因一次漏检被其他扇叶顶替；副槽短时漏检可由整符刚体关系补全。
  std::pair<std::optional<PowerRune>, std::optional<PowerRune>> detect_dual(
    cv::Mat & bgr_img, const Eigen::Quaterniond & q_gimbal2world,
    std::chrono::steady_clock::time_point timestamp);

  /// 大符开火切换后的唯一显式换槽入口。
  void set_last_powerrune(const PowerRune & pr);

  void reset();

  /// 注入"某槽此刻应当出现在图像的哪个圆周角"的预测器（由 Target 的相位模型提供）。
  /// 关联参考角优先用它：换组黑屏期间它仍在推进，不会像上一帧像素角那样陈旧
  /// （1~2 rad/s 下 200ms 就陈旧 11°~24°，吃掉 36° 角门限的大半预算）。
  /// 未注入或该槽未知时自动退回上一帧像素角。
  /// calibrator 是反向回灌：一次可信关联后把该槽的图像圆周角告诉 Target 完成标定。
  void set_slot_angle_predictor(
    std::function<std::optional<double>(std::size_t)> predictor,
    std::function<void(std::size_t, double)> calibrator = {})
  {
    predicted_slot_angle_ = std::move(predictor);
    calibrate_image_angle_ = std::move(calibrator);
  }

  std::size_t last_yolo_candidate_count() const { return model_.last_candidate_count(); }
  std::size_t last_yolo_score_count() const { return model_.last_score_count(); }
  std::size_t last_yolo_flow_valid_count() const { return model_.last_flow_valid_count(); }
  std::size_t last_yolo_active_count() const { return model_.last_active_count(); }
  std::size_t last_yolo_nms_count() const { return model_.last_nms_count(); }
  bool rm_center_enabled() const { return rm_center_enabled_; }
  /// kpt4 门控当前的连续拒绝帧数。长期贴着逃生阈值说明门限过紧。
  int r_gate_reject_streak() const { return r_gate_reject_streak_; }
  bool last_rm_center_valid() const { return last_rm_center_.has_value(); }
  bool last_rm_center_fallback() const
  {
    return rm_center_enabled_ && !last_rm_center_.has_value();
  }
  std::optional<cv::Point2f> last_rm_center() const { return last_rm_center_; }
  cv::Mat last_rm_binary_image() const { return rm_center_detector_.binary_debug_image(); }
  /// rm-core 颜色二值化阈值。调试界面用拖动条改它，正式入口不调用。
  int rm_color_threshold() const { return rm_center_detector_.color_threshold(); }
  void set_rm_color_threshold(int threshold)
  {
    rm_center_detector_.set_color_threshold(threshold);
  }
  /// 不跑检测，只按当前阈值单独二值化一张图，供暂停时预览阈值。
  cv::Mat binarize_with_rm_threshold(const cv::Mat & img) const
  {
    return rm_center_detector_.binarize(img);
  }
  double last_rm_center_ms() const { return last_rm_center_ms_; }
  double last_yolo_buff_ms() const { return last_yolo_buff_ms_; }
  /// 当前槽号纪元。每次 last_powerrune_ 被清空、五槽编号需要重新自举时 +1。
  std::uint64_t slot_epoch() const { return slot_epoch_; }

private:
  // 配置项：整帧没有有效候选时，历史目标保留的最大帧数。
  int loss_max_frames_{20};
  // 配置项：副目标漏检时，继续使用相邻槽推断结果的最大帧数。
  int secondary_hold_max_frames_{20};

  // kpt0~3 和 kpt5 是运行时硬观测；kpt4 只保留作模型质量诊断。
  std::optional<FanBlade> make_fanblade(const YOLO11_BUFF::Object & object) const;
  // 每帧仅执行一次 YOLO/NMS，再把全部候选组装为一个 PowerRune。
  std::optional<PowerRune> detect_group(
    cv::Mat & bgr_img, const Eigen::Quaterniond & q_gimbal2world,
    std::chrono::steady_clock::time_point timestamp);
  std::optional<PowerRune> make_power_rune(
    const std::vector<YOLO11_BUFF::Object> & objects, std::uint64_t frame_id,
    const std::optional<cv::Point2f> & rm_center);
  /// 决定本帧 PnP 用哪个 R：rm-core R 先过融合 kpt4 门控，超门限则退用融合 kpt4；
  /// 连续被拦 kRGateEscapeStreak 帧后放行一帧 rm R（防融合 kpt4 系统性跑偏时
  /// 把持续正确的 rm R 永久锁在门外）。**不再退 kpt5**。
  /// 二维融合滤波器（RPixelFusion）已实现但尚未接入此路径。
  void apply_r_center(
    PowerRune & power_rune, const std::optional<cv::Point2f> & rm_center);
  // 由 target/kpt5 构造关联或旧 PnP 回退所需的几何中心；不代表真实 R 观测。
  cv::Point2f shared_r_center(const std::vector<FanBlade> & fanblades);
  // 有历史时锁定 selected_slot；关联门控失败只报主目标丢失，绝不自动提升副目标。
  std::size_t assign_slots(
    std::vector<FanBlade> & fanblades, const cv::Point2f & r_center) const;
  void handle_loss();

  YOLO11_BUFF model_;
  RmCoreAdapter rm_center_detector_;
  bool rm_center_enabled_{false};
  std::optional<cv::Point2f> last_rm_center_;
  double last_rm_center_ms_{0.0};
  double last_yolo_buff_ms_{0.0};
  std::optional<PowerRune> last_powerrune_;
  std::optional<std::size_t> secondary_slot_;
  int lose_count_{0};
  int secondary_miss_count_{0};
  // 粗筛连续拒绝 rm R 的帧数。**只作诊断**，不再驱动任何逃生动作——
  // 滤波器下拒绝不会饿死估计，长期贴顶只说明粗筛定得太窄，该调宽而不是放行。
  int r_gate_reject_streak_{0};
  RPixelFusion r_fusion_;
  Track_status status_{LOSE};
  std::uint64_t next_frame_id_{1};
  // 槽号纪元：assign_slots 只有在 last_powerrune_ 存在时才能延续旧编号，
  // 一旦它被清空，下一帧的槽 0 会重新绑定到置信度最高的那片扇叶。
  std::uint64_t slot_epoch_{0};
  // 相位模型注入的槽角度预测器，用于关联参考角；未注入时退回上一帧像素角。
  std::function<std::optional<double>(std::size_t)> predicted_slot_angle_;
  // 反向回灌：一次可信关联后把图像圆周角告诉 Target，完成 roll↔图像角的标定。
  mutable std::function<void(std::size_t, double)> calibrate_image_angle_;
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__DETECTOR_HPP
