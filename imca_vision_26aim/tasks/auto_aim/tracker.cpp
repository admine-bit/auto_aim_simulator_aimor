#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "io/gimbal/gimbal.hpp"

namespace auto_aim
{
namespace
{
constexpr int kExactMatchLostThreshold = 4;
// 绝对位置轻微抖动经过 /dt 会被放大；低于该速度视为停车并立即清零，不留下长尾。
constexpr double kSelfSpeedDeadband = 0.05;  // m/s
// 只平滑供未来外推使用的速度；当前帧的 delta_self_world 不经过该滤波。
constexpr double kSelfSpeedFilterAlpha = 0.6;
}

Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  exact_match_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  io::Gimbal gimbal(config_path);
  auto gim = gimbal.state();
  auto yaml = YAML::LoadFile(config_path);
  // enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  enemy_color_ = gim.my_color == 0 ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
}

std::string Tracker::state() const { return state_; }
std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color,
  Eigen::Vector2d self_world_xy)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 当前帧 EKF 使用绝对位置之差做精确平移补偿；速度只供 Planner 外推未来自身位移
  Eigen::Vector2d delta_self_world = Eigen::Vector2d::Zero();
  Eigen::Vector2d v_self_world = Eigen::Vector2d::Zero();
  if (!self_world_xy.allFinite()) {
    // 无效样本不能作为下一帧差分基准，避免恢复后产生一次错误的大位移
    has_last_self_world_xy_ = false;
    filtered_v_self_world_.setZero();
  } else if (has_last_self_world_xy_ && dt > 0 && dt <= 0.1) {
    // 只对连续图像帧做差；跨断帧的大位移不应一次性注入当前目标 EKF
    delta_self_world = self_world_xy - last_self_world_xy_;
    const Eigen::Vector2d raw_v_self_world = delta_self_world / dt;
    if (raw_v_self_world.norm() < kSelfSpeedDeadband) {
      // 停车时直接归零，避免低通滤波像旧 vx/vy 方案一样拖尾。
      filtered_v_self_world_.setZero();
    } else {
      filtered_v_self_world_ = kSelfSpeedFilterAlpha * raw_v_self_world +
                               (1 - kSelfSpeedFilterAlpha) * filtered_v_self_world_;
    }
    v_self_world = filtered_v_self_world_;
  } else {
    filtered_v_self_world_.setZero();
  }
  if (self_world_xy.allFinite()) {
    last_self_world_xy_ = self_world_xy;
    has_last_self_world_xy_ = true;
  }

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t, delta_self_world);
  }

  // 当前帧 predict 已经使用真实 delta；这里保存速度仅供返回的 Target 副本预测未来
  target_.set_v_self_world(v_self_world);

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) 
  {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 收敛效果检测：
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) 
  {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  return targets;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color, Eigen::Vector2d self_world_xy)
{
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 与普通 track() 保持相同语义：真实位移用于当前帧，差分速度用于未来预测
  Eigen::Vector2d delta_self_world = Eigen::Vector2d::Zero();
  Eigen::Vector2d v_self_world = Eigen::Vector2d::Zero();
  if (!self_world_xy.allFinite()) {
    // 无效样本不能作为下一帧差分基准
    has_last_self_world_xy_ = false;
    filtered_v_self_world_.setZero();
  } else if (has_last_self_world_xy_ && dt > 0 && dt <= 0.1) {
    // 与离线检测共用 0.1 s 边界，跨断帧时从当前位置重新建立差分基准
    delta_self_world = self_world_xy - last_self_world_xy_;
    const Eigen::Vector2d raw_v_self_world = delta_self_world / dt;
    if (raw_v_self_world.norm() < kSelfSpeedDeadband) {
      filtered_v_self_world_.setZero();
    } else {
      filtered_v_self_world_ = kSelfSpeedFilterAlpha * raw_v_self_world +
                               (1 - kSelfSpeedFilterAlpha) * filtered_v_self_world_;
    }
    v_self_world = filtered_v_self_world_;
  } else {
    filtered_v_self_world_.setZero();
  }
  if (self_world_xy.allFinite()) {
    last_self_world_xy_ = self_world_xy;
    has_last_self_world_xy_ = true;
  }

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }

  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }

  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t, delta_self_world);
  }

  // omniperception 返回的 Target 同样携带最新自身速度，供 Planner 对副本前向预测
  target_.set_v_self_world(v_self_world);

  pre_state_ = state_;
  // 更新状态机
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

  std::list<Target> targets = {target_};
  return {switch_target, targets};
}

void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }

  else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }

  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        //前哨站的temp_lost_count需要设置的大一些
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  exact_match_lost_count_ = 0;

  auto & armor = armors.front();
  solver_.solve(armor);

  // 根据兵种优化初始化参数
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }

  else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0.6}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  }

  else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }

  else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

bool Tracker::update_target(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
  const Eigen::Vector2d & delta_self_world)
{
  // 视觉 x/y 是“以当前云台为原点、轴方向与 W0 对齐”的相对坐标，
  // 因此预测到当前帧时必须先减去这一帧自身已经发生的 W0 位移。
  target_.predict(t, delta_self_world);

  int found_count = 0;
  double min_x = 1e10;  // 画面最左侧
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
    min_x = armor.center.x < min_x ? armor.center.x : min_x;
  }

  // if (found_count == 0) {
  //   if (state_ == "tracking") {
  //     exact_match_lost_count_++;
  //     if (exact_match_lost_count_ < kExactMatchLostThreshold) return true;
  //   }
  //   exact_match_lost_count_ = 0;
  //   return false;
  // }
  // exact_match_lost_count_ = 0;

  if (found_count == 0) return false;
  for (auto & armor : armors) {
    if (
      armor.name != target_.name || armor.type != target_.armor_type
      //  || armor.center.x != min_x
    )
      continue;

    solver_.solve(armor);

    target_.update(armor);
  }

  return true;
}

}  // namespace auto_aim
