#include "buff_aimer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
double buff_fire_dis_;
namespace auto_buff
{
namespace
{
constexpr double kSecondaryAngle = 2.0 * CV_PI / 5.0;
constexpr double kSecondaryAngleMaxError = CV_PI / 5.0;
constexpr int kBallisticIterations = 4;
constexpr double kBallisticConvergenceSeconds = 1e-3;
constexpr double kBaseAngleJump = 5.0 / 57.3;
constexpr double kMaxBuffAngularSpeed = 2.5;
constexpr double kMaxAngleGateDt = 0.2;
constexpr double kTargetCenterRadius = 0.700;
// 规则：大符命中第一块后 1s 内可打第二块。切换预算须给第二发飞行时间留余量。
constexpr double kSecondarySwitchBudgetS = 0.8;

template<typename TargetType>
bool solve_ballistic_prediction(
  const TargetType & observed_target, double system_delay, double bullet_speed,
  double yaw_offset, double pitch_offset, bool aim_r_center, TargetType & predicted_target,
  double & yaw, double & pitch, double & roll)
{
  if (
    !std::isfinite(system_delay) || !std::isfinite(bullet_speed) ||
    bullet_speed <= 0.0)
    return false;

  // 飞行时间与目标位置互相依赖。每轮都从同一个观测状态重新预测到
  // system_delay + fly_time，避免在上轮副本上重复累计飞行时间。
  double fly_time_estimate = 0.0;
  double final_pitch = 0.0;
  Eigen::Vector3d final_aim = Eigen::Vector3d::Zero();
  for (int iteration = 0; iteration < kBallisticIterations; ++iteration) {
    predicted_target = observed_target;
    if (!aim_r_center) predicted_target.predict(system_delay + fly_time_estimate);

    const Eigen::Vector3d point_in_buff = aim_r_center
      ? Eigen::Vector3d::Zero()
      : Eigen::Vector3d(0.0, 0.0, kTargetCenterRadius);
    const Eigen::Vector3d aim = predicted_target.point_buff2world(point_in_buff);
    if (!aim.allFinite()) return false;
    const double distance = aim.head<2>().norm();
    const double height = aim[2];
    if (!std::isfinite(distance) || !std::isfinite(height)) return false;

    tools::Trajectory trajectory(bullet_speed, distance, height);
    if (
      trajectory.unsolvable || !std::isfinite(trajectory.fly_time) ||
      !std::isfinite(trajectory.pitch))
      return false;

    const double residual = trajectory.fly_time - fly_time_estimate;
    fly_time_estimate = trajectory.fly_time;
    final_pitch = trajectory.pitch;
    final_aim = aim;
    if (
      iteration > 0 && std::abs(residual) <= kBallisticConvergenceSeconds)
      break;
  }

  yaw = std::atan2(final_aim[1], final_aim[0]) + yaw_offset;
  pitch = final_pitch + pitch_offset;
  const Eigen::VectorXd state = predicted_target.ekf_x();
  if (state.size() <= 5 || !state.allFinite()) return false;
  roll = state[5];
  return std::isfinite(yaw) && std::isfinite(pitch) && std::isfinite(roll);
}
}  // namespace

Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  const double aim_yaw_offset = yaml["yaw_offset"].as<double>();
  const double aim_pitch_offset = yaml["pitch_offset"].as<double>();
  const double buff_yaw_offset = yaml["buff_yaw_offset"].as<double>();
  const double buff_pitch_offset =yaml["buff_pitch_offset"].as<double>();

  yaw_offset_ = (aim_yaw_offset + buff_yaw_offset) / 57.3;      // degree to rad
  pitch_offset_ = (aim_pitch_offset + buff_pitch_offset) / 57.3;  // degree to rad

  buff_fire_dis_ = yaml["buff_fire_dis"].as<double>();
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  predict_time_ = yaml["predict_time"].as<double>();
  barrel_delay_ = yaml["barrel_delay"].as<double>();
  rune_idle_duration_ = yaml["rune_idle_duration"]
                          ? yaml["rune_idle_duration"].as<double>()
                          : 0.4;
  rune_shoot_duration_ = yaml["rune_shoot_duration"]
                           ? yaml["rune_shoot_duration"].as<double>()
                           : 0.2;
  use_rmcs_fire_cycle_ = yaml["buff_use_rmcs_fire_cycle"]
                           ? yaml["buff_use_rmcs_fire_cycle"].as<bool>()
                           : true;
  if (
    !std::isfinite(rune_idle_duration_) || !std::isfinite(rune_shoot_duration_) ||
    rune_idle_duration_ < 0.0 || rune_shoot_duration_ <= 0.0)
    throw std::runtime_error("Rune fire durations must be finite and positive");
  // 可选开关：旧 yaml 未写时保持“第一发后自动切副目标”的现行为。
  auto_secondary_switch_ = yaml["buff_auto_secondary_switch"]
                             ? yaml["buff_auto_secondary_switch"].as<bool>()
                             : true;
  debug_aim_r_center_ = yaml["buff_debug_aim_r_center"]
                          ? yaml["buff_debug_aim_r_center"].as<bool>()
                          : false;
  // R 与槽位无关，调偏置时不运行主副目标开火切换状态机。
  if (debug_aim_r_center_) auto_secondary_switch_ = false;

  last_fire_t_ = std::chrono::steady_clock::now();
}

bool Aimer::rune_fire_window(bool ready, std::chrono::steady_clock::time_point now)
{
  if (!use_rmcs_fire_cycle_) return ready;
  if (!ready) {
    rune_cycle_started_ = false;
    return false;
  }
  if (!rune_cycle_started_) {
    rune_cycle_started_ = true;
    rune_cycle_start_ = now;
    return false;
  }

  const double elapsed = std::max(0.0, tools::delta_time(now, rune_cycle_start_));
  const double period = rune_idle_duration_ + rune_shoot_duration_;
  const double phase = std::fmod(elapsed, period);
  return phase >= rune_idle_duration_;
}

bool Aimer::confirm_slot_switch(const std::optional<std::size_t> & current_slot)
{
  // 槽位去抖：同一新槽连续 2 帧才确认为显式换叶。单帧标签闪断不确认，
  // 其带来的 yaw 突变由角度跳变门控吸收（压一帧输出），避免乒乓压枪。
  if (!has_last_angle_ || !last_aim_target_slot_.has_value() || !current_slot.has_value()) {
    last_aim_target_slot_ = current_slot;
    pending_slot_.reset();
    pending_slot_count_ = 0;
    return false;
  }
  if (*last_aim_target_slot_ == *current_slot) {
    pending_slot_.reset();
    pending_slot_count_ = 0;
    return false;
  }
  if (pending_slot_.has_value() && *pending_slot_ == *current_slot) {
    ++pending_slot_count_;
  } else {
    pending_slot_ = current_slot;
    pending_slot_count_ = 1;
  }
  if (pending_slot_count_ >= 2) {
    last_aim_target_slot_ = current_slot;
    pending_slot_.reset();
    pending_slot_count_ = 0;
    return true;
  }
  return false;
}

void Aimer::reset_tracking_gate()
{
  has_last_angle_ = false;
  mistake_count_ = 0;
  switch_fanblade_ = true;
  first_in_aimer_ = true;
  last_prediction_valid_ = false;
  last_aim_target_slot_.reset();
  pending_slot_.reset();
  pending_slot_count_ = 0;
  rune_cycle_started_ = false;
  last_fire_t_ = std::chrono::steady_clock::now();
  // 丢失期间不能继续沿用已经缓存的主/副开火切换方案，否则 barrel_delay 到时
  // 可能向旧副目标发送一次过期控制。
  big_state_ = BigBuffState{};
}

io::Command Aimer::aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp,
  double bullet_speed, bool to_now)
{
  io::Command command = {false, false, 0, 0};
  last_prediction_valid_ = false;
  if (target.is_unsolve()) {
    // 目标重捕获后不能再和上一回合的 yaw 比较，否则会被 5° 门控额外压住数帧。
    reset_tracking_gate();
    return command;
  }

  // 如果子弹速度小于10，将其设为24
  if (bullet_speed < 10) bullet_speed = 24;

  auto now = std::chrono::steady_clock::now();

  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  double yaw, pitch;

  bool explicit_slot_switch = false;
  switch_fanblade_ = true;
  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch)) {
    last_prediction_valid_ = true;
    command.yaw = yaw;
    command.pitch = -pitch;  //世界坐标系下的pitch向上为负
    explicit_slot_switch =
      !debug_aim_r_center_ && confirm_slot_switch(target.target_slot());
    const double gate_dt = has_last_angle_
      ? std::clamp(tools::delta_time(now, last_angle_t_), 0.0, kMaxAngleGateDt)
      : 0.0;
    // 5° 只是静态噪声底线；低帧率下正常大符在帧间就可以转过更大角度。
    // 按真实时间差放宽门限，避免 Jetson 帧率下降时周期性压住 yaw。
    const double angle_jump_limit = kBaseAngleJump + kMaxBuffAngularSpeed * gate_dt;
    if (!has_last_angle_) {
      // 第一个有效解没有“上一帧”可比较，直接建立基准并输出 yaw；
      // 旧代码拿它和初始 0 比较，目标偏离中轴超过 5° 时会无条件等待四帧。
      has_last_angle_ = true;
      switch_fanblade_ = false;
      mistake_count_ = 0;
      command.control = true;
    } else if (explicit_slot_switch) {
      // Detector 已经明确给出换叶，新槽位 yaw 必须当帧发送；仅压住当帧开火。
      switch_fanblade_ = true;
      mistake_count_ = 0;
      command.control = true;
    } else if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      command.control = true;
    } else if (
      std::abs(tools::limit_rad(yaw - last_yaw_)) > angle_jump_limit ||
      std::abs(last_pitch_ - pitch) > angle_jump_limit) {
      switch_fanblade_ = true;
      mistake_count_++;
      command.control = false;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      command.control = true;
    }
    last_yaw_ = yaw;
    last_pitch_ = pitch;
    last_angle_t_ = now;
  }

  if (switch_fanblade_) {
    command.shoot = false;
    rune_fire_window(false, now);
    // 显式换槽是大符开火状态机安排的正常切换：只禁止当帧开火，
    // 不能把第一发的真实时间清掉，否则 barrel_delay 后还会重新等待完整 fire_gap_time。
    if (!explicit_slot_switch) last_fire_t_ = now;
  } else if (use_rmcs_fire_cycle_) {
    command.shoot = rune_fire_window(command.control, now);
  } else if (tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    command.shoot = true;
    last_fire_t_ = now;
  }

  return command;
}

auto_aim::Plan Aimer::mpc_aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
  bool to_now)
{
  auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  last_prediction_valid_ = false;
  if (target.is_unsolve()) {
    reset_tracking_gate();
    return plan;
  }

  double bullet_speed;
  // 如果子弹速度小于10，将其设为23
  if (gs.bullet_speed < 10)
    bullet_speed = 23;
  else
    bullet_speed = gs.bullet_speed;

  auto now = std::chrono::steady_clock::now();

  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  double yaw, pitch;

  // MPC 的前后样本必须从同一个“未预测观测状态”出发。负样本只用于前馈，
  // commit_prediction=false 保证它不会覆盖后面用于主 yaw/蓝框的正样本。
  const double derivative_dt = std::abs(predict_time_);
  double previous_yaw = 0.0;
  double previous_pitch = 0.0;
  const bool derivative_valid =
    !debug_aim_r_center_ && derivative_dt > 1e-6 &&
    get_send_angle(
      target, future - 2.0 * derivative_dt, bullet_speed, to_now,
      previous_yaw, previous_pitch, false);

  bool explicit_slot_switch = false;
  switch_fanblade_ = true;
  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch)) {
    last_prediction_valid_ = true;
    plan.yaw = yaw;
    plan.pitch = -pitch;  //世界坐标系下的pitch向上为负
    explicit_slot_switch =
      !debug_aim_r_center_ && confirm_slot_switch(target.target_slot());
    const double gate_dt = has_last_angle_
      ? std::clamp(tools::delta_time(now, last_angle_t_), 0.0, kMaxAngleGateDt)
      : 0.0;
    const double angle_jump_limit = kBaseAngleJump + kMaxBuffAngularSpeed * gate_dt;
    if (!has_last_angle_) {
      // 首个可靠 Buff 姿态立即输出；速度/加速度在下面按 first_in_aimer_ 清零。
      has_last_angle_ = true;
      switch_fanblade_ = false;
      mistake_count_ = 0;
      plan.control = true;
      first_in_aimer_ = true;
    } else if (explicit_slot_switch) {
      // 槽位切换是 Detector 的显式事件，不再当成 PnP 突变连续丢四帧。
      // 当帧仍输出新 yaw，但清零 MPC 导数并禁止立即开火。
      switch_fanblade_ = true;
      mistake_count_ = 0;
      plan.control = true;
      first_in_aimer_ = true;
    } else if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      plan.control = true;
      first_in_aimer_ = true;
    } else if (
      std::abs(tools::limit_rad(yaw - last_yaw_)) > angle_jump_limit ||
      std::abs(last_pitch_ - pitch) > angle_jump_limit) {
      switch_fanblade_ = true;
      mistake_count_++;
      plan.control = false;

      first_in_aimer_ = true;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      plan.control = true;
    }
    last_yaw_ = yaw;
    last_pitch_ = pitch;
    last_angle_t_ = now;

    if (plan.control) {
      if (first_in_aimer_) {
        plan.yaw_vel = 0;
        plan.yaw_acc = 0;
        plan.pitch_vel = 0;
        plan.pitch_acc = 0;
        first_in_aimer_ = false;
      } else {
        const double dt = derivative_dt;
        if (!derivative_valid) {
          // 速度/加速度只是可选前馈；负样本失败时保留已经有效的主 yaw，
          // 本帧把前馈安全降级为 0，下一帧再重新建立中心差分。
          plan.yaw_vel = 0.0;
          plan.yaw_acc = 0.0;
          plan.pitch_vel = 0.0;
          plan.pitch_acc = 0.0;
          first_in_aimer_ = true;
        } else {
          plan.yaw_vel = tools::limit_rad(yaw - previous_yaw) / (2 * dt);
          // plan.yaw_vel = tools::limit_min_max(plan.yaw_vel, -6.28, 6.28);
          plan.yaw_acc =
            (tools::limit_rad(yaw - gs.yaw) - tools::limit_rad(gs.yaw - previous_yaw)) /
            std::pow(dt, 2);
          // plan.yaw_acc = tools::limit_min_max(plan.yaw_acc, -50, 50);

          plan.pitch_vel = tools::limit_rad(-pitch + previous_pitch) / (2 * dt);
          // plan.pitch_vel = tools::limit_min_max(plan.pitch_vel, -6.28, 6.28);
          plan.pitch_acc =
            (-pitch - gs.pitch - (gs.pitch + previous_pitch)) / std::pow(dt, 2);
          // plan.pitch_acc = tools::limit_min_max(plan.pitch_acc, -100, 100);
        }
      }
    }
  }

  if (switch_fanblade_) {
    plan.fire = false;
    rune_fire_window(false, now);
    // 槽位切换是正常的离散目标事件，保持第一发的 fire_gap 计时。
    // 只有异常跳变或首次建立跟踪才重新等待，避免大符两发间隔被 barrel_delay + fire_gap_time 叠加。
    if (!explicit_slot_switch) last_fire_t_ = now;
  } else if (use_rmcs_fire_cycle_) {
    plan.fire = rune_fire_window(plan.control, now);
  } else if (tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    plan.fire = true;
    last_fire_t_ = now;
  }

  return plan;
}

bool Aimer::get_send_angle(
  auto_buff::Target & target, const double predict_time, const double bullet_speed,
  const bool to_now, double & yaw, double & pitch, bool commit_prediction)
{
  (void)to_now;
  if (auto * big_target = dynamic_cast<BigTarget *>(&target)) {
    BigTarget predicted = *big_target;
    if (!solve_ballistic_prediction(
          *big_target, predict_time, bullet_speed, yaw_offset_, pitch_offset_,
          debug_aim_r_center_, predicted, yaw, pitch, angle))
      return false;
    if (commit_prediction) *big_target = std::move(predicted);
    return true;
  }
  if (auto * small_target = dynamic_cast<SmallTarget *>(&target)) {
    SmallTarget predicted = *small_target;
    if (!solve_ballistic_prediction(
          *small_target, predict_time, bullet_speed, yaw_offset_, pitch_offset_,
          debug_aim_r_center_, predicted, yaw, pitch, angle))
      return false;
    if (commit_prediction) *small_target = std::move(predicted);
    return true;
  }
  tools::logger()->warn("[Aimer] Unknown buff Target subtype");
  return false;
}

// 为副目标做弹道预计算，结果存入 big_state_.plan_next（不污染 Aimer 主状态）。
// WAIT_BARREL 期间锁定的是槽位身份，方案本身每帧刷新，不冻结 yaw/pitch。
bool Aimer::compute_secondary_plan(
  BigTarget * bt, const std::optional<PowerRune> & secondary,
  const io::GimbalState & gs, std::chrono::steady_clock::time_point t)
{
  big_state_.has_plan_next = false;
  if (!auto_secondary_switch_) return false;
  if (bt == nullptr || bt->is_unsolve()) return false;

  const bool waiting = big_state_.state == BigSwitchState::WAIT_BARREL;

  std::optional<std::size_t> slot;
  std::optional<PowerRune> secondary_view = secondary;
  if (waiting && big_state_.locked_secondary_slot.has_value()) {
    // 第一发已出：只认锁定槽。副槽漏检时按固定槽位继续预测，不换目标。
    slot = big_state_.locked_secondary_slot;
    if (
      secondary_view.has_value() &&
      (secondary_view->is_unsolve() || secondary_view->target_slot() != *slot))
      secondary_view.reset();
  } else if (secondary.has_value() && !secondary->is_unsolve()) {
    // 合法性检查：副目标必须是同一五边形上相隔 1~2 槽的扇叶。
    const auto primary_state = bt->ekf_x();
    if (primary_state.size() <= 5 || !primary_state.allFinite() ||
        !secondary->ypr_in_world.allFinite())
      return false;
    const double roll_delta =
      std::abs(tools::limit_rad(secondary->ypr_in_world[2] - primary_state[5]));
    int slot_steps = static_cast<int>(std::lround(roll_delta / kSecondaryAngle));
    if (bt->target_slot().has_value()) {
      const std::size_t primary_slot = *bt->target_slot();
      const std::size_t secondary_slot = secondary->target_slot();
      const std::size_t forward_steps =
        (secondary_slot + PowerRune::SLOT_COUNT - primary_slot) % PowerRune::SLOT_COUNT;
      slot_steps = static_cast<int>(
        std::min(forward_steps, PowerRune::SLOT_COUNT - forward_steps));
    }
    if (
      slot_steps < 1 || slot_steps > 2 ||
      std::abs(roll_delta - slot_steps * kSecondaryAngle) > kSecondaryAngleMaxError)
      return false;
    slot = secondary->target_slot();
  }
  if (!slot.has_value()) {
    if (!waiting) big_state_.secondary_pr.reset();
    return false;
  }

  // 副目标优先按槽位从共享相位模型构造（clone_for_slot）；该槽相位尚未学习时
  // 退回用当前帧观测 roll 沿共享模型外推（clone_with_roll）。
  BigTarget bt_next = bt->clone_for_slot(*slot);
  if (bt_next.is_unsolve() && secondary_view.has_value() && !secondary_view->is_unsolve())
    bt_next = bt->clone_with_roll(secondary_view->ypr_in_world[2]);
  if (bt_next.is_unsolve()) return false;

  const double bullet_speed =
    std::isfinite(gs.bullet_speed) && gs.bullet_speed >= 10.0 ? gs.bullet_speed : 23.0;
  // 切换发生在 barrel_delay 之后，副目标预测时域必须包含这段等待。
  const double future =
    tools::delta_time(std::chrono::steady_clock::now(), t) + predict_time_ + barrel_delay_;
  if (!std::isfinite(future)) return false;

  BigTarget predicted_next = bt_next;
  double secondary_yaw = 0.0;
  double secondary_pitch = 0.0;
  double secondary_roll = 0.0;
  if (!solve_ballistic_prediction(
        bt_next, future, bullet_speed, yaw_offset_, pitch_offset_, debug_aim_r_center_,
        predicted_next, secondary_yaw, secondary_pitch, secondary_roll))
    return false;
  (void)secondary_roll;

  big_state_.plan_next = {};
  big_state_.plan_next.control = true;
  big_state_.plan_next.yaw = secondary_yaw;
  big_state_.plan_next.pitch = -secondary_pitch;
  big_state_.has_plan_next = true;
  if (secondary_view.has_value()) big_state_.secondary_pr = secondary_view;
  if (!waiting) big_state_.locked_secondary_slot = slot;
  return true;
}

// 大符双目标开火切换状态机：
// TRACK_PRIMARY →(第一发+有副方案) WAIT_BARREL →(barrel_delay) TRACK_SECONDARY
//   →(第二发) WAIT_NEW_PAIR →(等新一组点亮) TRACK_PRIMARY
void Aimer::big_buff_send(
  auto_aim::Plan & plan, io::Gimbal & gimbal, Buff_Detector & detector)
{
  auto now = std::chrono::steady_clock::now();
  auto send = [&](const auto_aim::Plan & p, bool fire) {
    gimbal.send(p.control, fire, p.yaw, p.yaw_vel, p.yaw_acc,
                p.pitch, p.pitch_vel, p.pitch_acc);
  };

  // 开关关闭：完全旁路自动切换（不锁槽、不发 plan_next、不改 Detector 主槽参考），
  // 大符按普通单目标流程开火；detect_dual 仍在上游为共享观测服务。
  if (!auto_secondary_switch_) {
    send(plan, plan.control && plan.fire);
    return;
  }

  switch (big_state_.state) {
    case BigSwitchState::TRACK_PRIMARY: {
      const bool fire = plan.control && plan.fire;
      if (fire && big_state_.has_plan_next && big_state_.locked_secondary_slot.has_value()) {
        // 第一发真正发出：锁定副槽身份进入等待；无合法副槽时正常打，不进状态机。
        big_state_.state = BigSwitchState::WAIT_BARREL;
        big_state_.fire_time = now;
        big_state_.plan_current = plan;
      }
      send(plan, fire);
      return;
    }
    case BigSwitchState::WAIT_BARREL: {
      const double waited = tools::delta_time(now, big_state_.fire_time);
      if (waited < barrel_delay_) {
        // 出膛窗口：保持主目标指向，禁止重复开火。
        send(big_state_.plan_current, false);
        return;
      }
      // 规则：命中第一块后 1s 内可打第二块。超预算仍无可靠方案则放弃自动切换。
      if (waited > kSecondarySwitchBudgetS) {
        tools::logger()->debug("[Aimer] 副目标方案超时，放弃自动切换");
        big_state_ = BigBuffState{};
        send(plan, false);
        return;
      }
      if (big_state_.has_plan_next) {
        // 用最新一帧刷新的副方案切换；只改 Detector 主槽参考，不重置任何共享滤波器。
        send(big_state_.plan_next, false);
        if (big_state_.secondary_pr.has_value())
          detector.set_last_powerrune(*big_state_.secondary_pr);
        big_state_.state = BigSwitchState::TRACK_SECONDARY;
        big_state_.pair_time = now;
      } else {
        send(big_state_.plan_current, false);
      }
      return;
    }
    case BigSwitchState::TRACK_SECONDARY: {
      const bool fire = plan.control && plan.fire;
      if (fire) {
        big_state_.state = BigSwitchState::WAIT_NEW_PAIR;
        big_state_.pair_time = now;
      }
      send(plan, fire);
      return;
    }
    case BigSwitchState::WAIT_NEW_PAIR: {
      // 第二发已出。规则允许换组前 ≤200ms 黑屏；短暂等待后回到主跟踪，
      // 本组主副身份作废（locked slot 清空），共享 R/相位模型不受影响。
      send(plan, plan.control && plan.fire);
      if (tools::delta_time(now, big_state_.pair_time) > 0.3)
        big_state_ = BigBuffState{};
      return;
    }
  }
}

// 开火二次门控：误差阈值随距离缩放（大符/小符通用）
bool fire_gate_check(
  const auto_aim::Plan & plan, const Target & target, const io::GimbalState & gs)
{
  if (!plan.control || !plan.fire || target.is_unsolve()) return false;

  const auto state = target.ekf_x();
  if (state.size() <= 3 || !state.allFinite() || !std::isfinite(plan.yaw) ||
      !std::isfinite(plan.pitch) || !std::isfinite(gs.yaw) || !std::isfinite(gs.pitch) ||
      state[3] <= 0.0)
    return false;

  const double min_dis = buff_fire_dis_ / state[3];  // 角度阈值随距离缩放
  if (!std::isfinite(min_dis) || min_dis <= 0.0) return false;
  return std::abs(tools::limit_rad(plan.yaw - gs.yaw)) < min_dis &&
         std::abs(plan.pitch - gs.pitch) < min_dis;
}

}  // namespace auto_buff
