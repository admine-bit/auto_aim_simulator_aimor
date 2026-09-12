#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  // self_world_xy 是图像时刻云台安装点在固定世界系 W0 下的绝对位置；
  // 旧调用若始终使用默认 (0,0)，等价于关闭自身平移补偿。
  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true, Eigen::Vector2d self_world_xy = Eigen::Vector2d::Zero());

  // omniperception 入口与普通入口使用相同的自身位置语义
  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true,
    Eigen::Vector2d self_world_xy = Eigen::Vector2d::Zero());

private:
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int exact_match_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  // 与 last_timestamp_ 对应的自身绝对位置，用来计算已经发生的逐帧真实位移
  Eigen::Vector2d last_self_world_xy_ = Eigen::Vector2d::Zero();
  // false 表示还没有有效历史位置；此时当前帧自身位移补偿保持为零
  bool has_last_self_world_xy_ = false;
  // 仅供尚未发生的未来预测使用；当前帧 EKF 仍使用未经滤波的真实位置差。
  Eigen::Vector2d filtered_v_self_world_ = Eigen::Vector2d::Zero();
  ArmorPriority omni_target_priority_;

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  // delta_self_world 是上一图像帧到 t 之间，云台安装点在 W0 下已发生的真实位移
  bool update_target(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    const Eigen::Vector2d & delta_self_world);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
