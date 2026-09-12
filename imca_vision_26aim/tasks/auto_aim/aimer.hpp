#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

// 包含标准库头文件
#include <Eigen/Dense>
#include <chrono>
#include <list>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "target.hpp"

namespace auto_aim
{

struct AimPoint
{
  bool valid;
  Eigen::Vector4d xyza;
};
// 自动瞄准器类：负责目标瞄准计算、弹道解算和射击决策
class Aimer
{
public:
  AimPoint debug_aim_point;
  explicit Aimer(const std::string & config_path);   // 参数：config_path - YAML配置文件的路径
  
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    io::ShootMode shoot_mode, bool to_now = true);

private:
  double yaw_offset_;
  std::optional<double> left_yaw_offset_, right_yaw_offset_;
  double pitch_offset_;    // pitch轴偏移量（弧度）
  double comming_angle_;   // 小陀螺模式的进入角度（弧度）
  double leaving_angle_;  // 小陀螺模式的离开角度（弧度）
  double lock_id_ = -1;   // 锁定的装甲板ID（防止频繁切换）
  double high_speed_delay_time_; // 高速目标的延迟时间（秒）
  double low_speed_delay_time_;// 低速目标的延迟时间（秒）
  double decision_speed_;  // 速度阈值（区分高低速目标）
 
  AimPoint choose_aim_point(const Target & target);// 调试用的瞄准点信息
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP