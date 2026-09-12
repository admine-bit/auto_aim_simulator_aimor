#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <optional>
#include <vector>

#include "../auto_aim/planner/planner.hpp"
#include "buff_target.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "buff_detector.hpp"

namespace auto_buff
{
class Aimer
{
public:
  explicit Aimer(const std::string & config_path);

  io::Command aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    bool to_now = true);

  auto_aim::Plan mpc_aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
    bool to_now = true);

  // 大符副目标预计算（二迭代弹道解算，不污染 mpc_aim 状态）
  bool compute_secondary_plan(
    BigTarget * bt, const std::optional<PowerRune> & secondary,
    const io::GimbalState & gs, std::chrono::steady_clock::time_point t);

  // 大符开火状态机：开火 → 等 barrel_delay → 切副目标
  void big_buff_send(
    auto_aim::Plan & plan, io::Gimbal & gimbal, Buff_Detector & detector);

  // 调试可视化只在完整弹道迭代成功后绘制预测框，不能显示半预测状态。
  bool prediction_valid() const { return last_prediction_valid_; }
  bool debug_aim_r_center() const { return debug_aim_r_center_; }

  double angle{0.0};  ///
  double t_gap = 0;  ///

private:
  double yaw_offset_;
  double pitch_offset_;

  double fire_gap_time_;
  double predict_time_;
  double barrel_delay_;
  // 第一发后是否自动切副目标（yaml: buff_auto_secondary_switch，缺省 true）。
  bool auto_secondary_switch_{true};
  // 调试模式：瞄准固定 R 标并关闭旋转预测/前馈/换槽状态机。
  bool debug_aim_r_center_{false};

  int mistake_count_ = 0;
  bool switch_fanblade_{true};

  double last_yaw_ = 0;
  double last_pitch_ = 0;
  bool has_last_angle_{false};
  std::chrono::steady_clock::time_point last_angle_t_{};
  std::optional<std::size_t> last_aim_target_slot_;
  // 槽位去抖：同一新槽连续 2 帧才视为显式换叶；单帧标签闪断走角度跳变门控。
  std::optional<std::size_t> pending_slot_;
  int pending_slot_count_{0};

  // for mpc
  bool first_in_aimer_ = true;
  bool last_prediction_valid_{false};

  // RMCS 风格的能量机关 idle/shoot 周期。计划通过后仍需落在 shoot 窗口才允许发送 fire。
  double rune_idle_duration_{0.4};
  double rune_shoot_duration_{0.2};
  bool use_rmcs_fire_cycle_{true};
  bool rune_cycle_started_{false};
  std::chrono::steady_clock::time_point rune_cycle_start_{};

  std::chrono::steady_clock::time_point last_fire_t_;

  bool get_send_angle(
    auto_buff::Target & target, const double predict_time, const double bullet_speed,
    const bool to_now, double & yaw, double & pitch, bool commit_prediction = true);

  bool rune_fire_window(bool ready, std::chrono::steady_clock::time_point now);

  // 槽位去抖判定：返回“已确认的显式换叶”，并维护 last_aim_target_slot_。
  bool confirm_slot_switch(const std::optional<std::size_t> & current_slot);

  // 丢失或模式重建后清除上一回合角度门控，保证新目标首帧即可建立 yaw 基准。
  void reset_tracking_gate();

  // 大符双目标开火切换状态机
  // TRACK_PRIMARY →(第一发+有副方案) WAIT_BARREL →(barrel_delay 到) TRACK_SECONDARY
  //   →(第二发) WAIT_NEW_PAIR →(短暂等待新组) TRACK_PRIMARY
  enum class BigSwitchState { TRACK_PRIMARY, WAIT_BARREL, TRACK_SECONDARY, WAIT_NEW_PAIR };
  struct BigBuffState {
    BigSwitchState state = BigSwitchState::TRACK_PRIMARY;
    std::chrono::steady_clock::time_point fire_time;
    std::chrono::steady_clock::time_point pair_time;
    auto_aim::Plan plan_current{};
    auto_aim::Plan plan_next{};
    bool has_plan_next = false;
    std::optional<PowerRune> secondary_pr;
    // 第一发后锁定的是副目标“槽位身份”，不是过期的 yaw/pitch。
    std::optional<std::size_t> locked_secondary_slot;
  } big_state_;
};

// 开火二次门控：误差阈值随距离缩放（大符/小符通用）
bool fire_gate_check(
  const auto_aim::Plan & plan, const Target & target, const io::GimbalState & gs);


}  // namespace auto_buff
#endif  // AUTO_AIM__AIMER_HPP
