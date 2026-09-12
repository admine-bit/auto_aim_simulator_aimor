#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <cstdint>
#include <deque>
#include <eigen3/Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <limits>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"
namespace auto_buff
{
const int INF = 1000000;
// 尚未完成槽位关联的哨兵值。必须大于 PowerRune::SLOT_COUNT——后者被 assign_slots
// 复用为"该片撞槽且置信度更低，直接丢弃"的标记，两者含义不同不能混用。
constexpr std::size_t PowerRune_SLOT_UNASSIGNED = 99;
enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE };

// PnP 使用的 R 观测来自谁。诊断用，回放时判断门控是否按预期工作。
// 区分"rm 缺失"和"rm 被门控拦下"看 r_gate_distance 是否有限。
enum class RCenterSource {
  None = 0,           // 无任何可信 R，PnP 不加 R 点、走 kpt0~3+kpt5 旧路径
  RmCore = 1,         // rm-core R 通过门控被采纳（最佳）
  RmCoreUngated = 2,  // 有 rm-core R 但无可信融合 kpt4 参照，未做门控直接采纳
  RmCoreEscaped = 3,  // 连续超门限后逃生采纳的 rm-core R（防门控自身成为故障源）
  FusedKpt4 = 4       // rm R 缺失或被门控拦下，本帧退用融合 kpt4
};

class FanBlade
{
public:
  cv::Rect_<float> rect{};  // YOLO buff 框，供 buf1 激活标记与带关键点目标关联
  cv::Point2f center{};  // target 四角对角线交点，对应物理半径 0.700 m
  // YOLO 六关键点：target 四角、R 中心、流水灯中心，严格保留模型标注顺序。
  std::vector<cv::Point2f> points;
  std::vector<float> keypoint_confidences;  // 与 points 一一对应，便于诊断弱关键点 kpt4
  float object_confidence{0.0F};            // YOLO 目标框置信度，用于同槽候选去重
  bool active{false};  // buf1 关联到的已激活目标
  bool active_observed{false};  // 当前帧是否实际观测到 buf1
  float active_confidence{0.0F};
  // 模型直接给出的 kpt4。单片精度不足以直接进 PnP，但两个用途成立：
  //   1) 作 rm-core R 的防误识别门控参照——门控只需粗精度；
  //   2) 跨片置信度加权融合后作 R 的回退来源（噪声被平均，优于 kpt5 外推）。
  // 这撤销了本文件早期"kpt4 仅作可视化和质量诊断"的定位，是有意变更，非笔误。
  cv::Point2f raw_r_center{};
  float raw_r_confidence{0.0F};
  bool raw_r_valid{false};
  bool observed{false};       // true 表示该槽在当前帧确实被检测到，可以参与 PnP
  // 五边形物理槽号。SLOT_COUNT(=5) 是 assign_slots 的"丢弃"标记（与另一片撞到同一
  // 格位且置信度更低）；大于 SLOT_COUNT 表示尚未完成槽位关联，由 PowerRune 按空槽补位。
  std::size_t slot_index{PowerRune_SLOT_UNASSIGNED};

  // Solver 写入的当前帧原始 PnP 诊断结果，不包含 EKF 滤波或运动预测。
  std::vector<cv::Point2f> raw_reprojected_points;
  double raw_reprojection_rmse{std::numeric_limits<double>::infinity()};
  double raw_reprojection_max_error{std::numeric_limits<double>::infinity()};
  // 分项质量：kpt0~3 四角误差、kpt5 流水灯误差；kpt5 异常帧回退四角 PnP 时置位。
  double raw_corner_rmse{std::numeric_limits<double>::infinity()};
  double raw_flow_error{std::numeric_limits<double>::infinity()};
  // 与四角回退分开记录：是否实际使用 rm-core R，以及 rm-core 缺失后是否走旧 kpt5 路径。
  bool raw_used_rm_center{false};
  bool raw_rm_center_fallback{false};
  bool raw_pnp_fallback{false};
  double angle{0.0};
  FanBlade_type type{_unlight};  // 类型

  FanBlade() = default;

  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(FanBlade_type t);
};

class PowerRune
{
public:
  // Buff 是固定五边形整体。fanblades 始终保持五个物理槽，槽间相位固定为 72°。
  static constexpr std::size_t SLOT_COUNT = 5;

  cv::Point2f r_center;  // 整个 Buff 唯一的共享 R 中心，而非每个扇叶各自一个
  // 送进 PnP 的当帧 R 中心 = 二维融合滤波器的输出（rm-core R 与融合 kpt4 按各自
  // 协方差加权，跨帧递推）。无可靠来源时为 NaN。
  // 名字保留 _raw 是为了不动下游一堆引用；它已经不是"裸测量"，而是"未经 PnP 的 R"。
  cv::Point2f r_center_raw{
    std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};
  // true 表示本帧有可信 R 可用于 PnP。kpt5 外推不再冒充实测，也不再作为 R 缺失时的回退。
  bool r_center_observed{false};
  // 本帧参与融合的测量，仅诊断用，不参与任何解算。
  RCenterSource r_center_source{RCenterSource::None};
  // 粗筛的实测偏差与阈值（像素）。未做粗筛（无 kpt4 参照或无 rm R）时为 NaN。
  float r_gate_distance{std::numeric_limits<float>::quiet_NaN()};
  float r_gate_threshold{std::numeric_limits<float>::quiet_NaN()};
  // 滤波后 R 位置的标准差（像素，取两轴较大者）。PnP 权重由它决定。
  float r_filter_sigma_px{std::numeric_limits<float>::quiet_NaN()};
  // 在线估计的融合 kpt4 标准差（像素）。噪声主导下按 样本散布/√n 收缩。
  float r_kpt4_sigma_px{std::numeric_limits<float>::quiet_NaN()};
  // 鲁棒更新对 rm 测量协方差的膨胀倍数：1 = 未膨胀，>1 = 本帧 rm 创新偏大被软降权。
  float r_rm_inflation{1.0F};
  // 参与 kpt4 融合的扇叶数；0 表示无可信 kpt4 参照。
  int r_kpt4_fused_count{0};
  // 融合 kpt4 本身（门控圆心）。即使 rm R 被采纳也保留，否则可视化画不出门限圆。
  // 仅诊断与可视化用，不参与解算。
  cv::Point2f r_kpt4_fused_center{
    std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};
  // 配置是否要求使用 rm-core。关闭时保持纯 YOLO 旧路径，不能把每帧都标成降质回退。
  bool rm_center_enabled{false};
  // 槽号纪元。Detector 每次重新自举五槽编号（丢失超时后 last_powerrune_ 被清空）
  // 都会 +1：同一个槽号在纪元变化前后可能指向不同的物理扇叶。
  std::uint64_t slot_epoch{0};
  std::vector<FanBlade> fanblades;  // 固定五槽，槽间相位为 72°

  int light_num{0};

  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();  // 单位：m
  Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();  // 单位：rad
  Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();  // 球坐标系

  Eigen::Vector3d blade_xyz_in_world = Eigen::Vector3d::Zero();  // 单位：m
  Eigen::Vector3d blade_ypd_in_world = Eigen::Vector3d::Zero();  // 球坐标系, 单位: m

  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune, std::uint64_t frame_id = 0,
    std::size_t selected_slot = 0);
  PowerRune() : fanblades(SLOT_COUNT), unsolvable_(true) {}

  // PowerRune 本体保存完整五槽；target()/with_target() 只改变“当前查看哪个槽”。
  // 因此主、副目标可以共享同一组槽位、R 中心以及同帧 group/frame 标识。
  FanBlade & target() { return fanblades[selected_slot_]; }
  const FanBlade & target() const { return fanblades[selected_slot_]; }

  std::size_t selected_slot() const { return selected_slot_; }
  std::size_t target_slot() const { return selected_slot_; }
  // 同一帧生成的主、副视图具有相同 frame_id/group_id，可防止跨帧目标误配。
  std::uint64_t frame_id() const { return frame_id_; }
  std::uint64_t group_id() const { return frame_id_; }
  bool slot_observed(std::size_t slot) const
  {
    return slot < fanblades.size() && fanblades[slot].observed;
  }
  // 返回完整 PowerRune 的轻量视图副本，只把指定槽设为 target，不重新检测或解算。
  // 若该槽只是历史残留（observed=false），返回值会被标记为不可解。
  PowerRune with_target(std::size_t slot) const;

  // 副目标短时漏检时只补全槽位身份；Solver 会依据同一刚体的 72° 相位关系推导姿态，
  // 不会拿历史像素重新做 PnP。
  PowerRune with_inferred_target(std::size_t slot) const;

  bool is_unsolve() const { return unsolvable_; }

private:
  double target_angle_{0.0};
  bool unsolvable_ = false;
  std::size_t selected_slot_{0};
  std::uint64_t frame_id_{0};

  double atan_angle(cv::Point2f v) const;  // [0, 2CV_PI]
};
}  // namespace auto_buff
#endif  // BUFF_TYPE_HPP
