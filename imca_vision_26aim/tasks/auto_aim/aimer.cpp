#include "aimer.hpp" // 包含Aimer类的头文件

#include <yaml-cpp/yaml.h>// 包含YAML配置文件解析库


// 包含数学相关库
#include <cmath>
#include <vector>

// 包含工具类：日志器、数学工具、弹道解算
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

// 命名空间：自动瞄准模块
namespace auto_aim
{
// Aimer类构造函数，接收配置文件路径
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);   // 加载YAML配置文件
  yaw_offset_ = yaml["yaw_offset"].as<double>() *3.14159/180.0;        // 读取yaw偏移量（度转弧度）
  pitch_offset_ = yaml["pitch_offset"].as<double>() *3.14159/180.0;    // 读取pitch偏移量（度转弧度）
  comming_angle_ = yaml["comming_angle"].as<double>() *3.14159/180.0;  // 读取进入角度（度转弧度）
  leaving_angle_ = yaml["leaving_angle"].as<double>() *3.14159/180.0;  // 读取离开角度（度转弧度）
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();// 读取高速延迟时间
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();// 读取低速延迟时间
  decision_speed_ = yaml["decision_speed"].as<double>();           // 读取决策速度阈值
  // 如果配置文件中定义了左右yaw偏移，则加载（度转弧度）
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) 
  {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() *3.14159/180.0;    // degree to rad
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() *3.14159/180.0;  // degree to rad
    // 记录日志：成功加载射击模式
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

// 瞄准方法：计算瞄准命令
// 参数：目标列表、时间戳、子弹速度、是否使用当前时间
io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  // 如果目标列表为空，返回空命令（不开火，角度为0）
  if (targets.empty()) return {false, false, 0, 0};

  auto target = targets.front();// 获取第一个目标（默认优先目标）

  auto ekf = target.ekf();// 获取目标的EKF状态估计器

  // 根据目标角速度选择延迟时间（高于决策速度用高速延迟，否则用低速延迟）
  double delay_time = target.ekf_x()[7] > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
    
  // 如果子弹速度低于14，默认设为23（可能是最低速度限制）
  if (bullet_speed < 14) bullet_speed = 23;

  // 考虑detecor和tracker所消耗的时间，此外假设aimer的用时可忽略不计
  auto future = timestamp;
  if (to_now) 
  {
    // 计算从当前时间到时间戳的差值加上延迟时间，作为预测时间增量
    double dt;
    dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));// 计算未来时间点（时间戳 + dt）
    target.predict(future);// 预测目标在未来时间点的状态
  }

  else 
  {
    // 固定时间增量：检测器-瞄准器耗时0.005秒 + 发弹延迟（如0.1秒）
    auto dt = 0.005 + delay_time;  //detector-aimer耗时0.005+发弹延时0.1
    // tools::logger()->info("dt is {:.4f} second", dt);
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);// 预测目标在未来时间点的状态
  }

  auto aim_point0 = choose_aim_point(target);// 选择瞄准点
  debug_aim_point = aim_point0;// 保存调试用的瞄准点

  //如果瞄准点无效，返回空命令
  if (!aim_point0.valid) 
  {
    // tools::logger()->debug("Invalid aim_point0.");
    return {false, false, 0, 0};
  }


  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);// 提取瞄准点的xyz坐标（前3个元素）
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);// 计算水平距离（xy平面内的距离）
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);// 初始化弹道解算器（子弹速度、水平距离、z方向距离）
  // 如果弹道不可解，记录日志并返回空命令
  if (trajectory0.unsolvable) 
  {
    tools::logger()->debug("[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
      
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  // 迭代求解飞行时间 (最多10次，收敛条件：相邻两次fly_time差 <0.001)
  bool converged = false;
  double prev_fly_time = trajectory0.fly_time;// 初始飞行时间
  tools::Trajectory current_traj = trajectory0;// 当前弹道
  std::vector<Target> iteration_target(10, target);  // 创建10个目标副本用于迭代预测

  // 迭代计算
  for (int iter = 0; iter < 10; ++iter) 
  {
    // 预测目标在 future + prev_fly_time 时刻的位置（考虑子弹飞行时间后的目标位置）
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    // 计算瞄准点
    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;

    // 如果瞄准点无效，返回空命令
    if (!aim_point.valid) 
    {
      return {false, false, 0, 0};
    }

    // 计算新弹道
    Eigen::Vector3d xyz = aim_point.xyza.head(3);// 提取瞄准点的xyz坐标
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());// 计算水平距离
    current_traj = tools::Trajectory(bullet_speed, d, xyz.z()); // 重新计算弹道

    // 检查弹道是否可解
    if (current_traj.unsolvable) 
    {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    // 检查收敛条件
    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) 
    {
      converged = true;
      break;
    }

    prev_fly_time = current_traj.fly_time;// 更新上一次飞行时间
  }

  // 计算最终角度
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);  //世界坐标系下pitch向上为负
  return {true, false, yaw, pitch};
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  double yaw_offset;
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;
  }

  auto command = aim(targets, timestamp, bullet_speed, to_now);
  command.yaw = command.yaw - yaw_offset_ + yaw_offset;

  return command;
}

AimPoint Aimer::choose_aim_point(const Target & target)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();
  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  if (!target.jumped) return {true, armor_xyza_list[0]};

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 如果delta_angle为0，则该装甲板中心和整车中心的连线在世界坐标系的xy平面过原点
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // 不考虑小陀螺
  if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost) {
    // 选择在可射击范围内的装甲板
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }
    // 绝无可能
    if (id_list.empty()) {
      tools::logger()->warn("Empty id list!");
      return {false, armor_xyza_list[0]};
    }

    // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
    if (id_list.size() > 1) {
      int id0 = id_list[0], id1 = id_list[1];

      // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
      if (lock_id_ != id0 && lock_id_ != id1)
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

      return {true, armor_xyza_list[lock_id_]};
    }

    // 只有一个装甲板在可射击范围内时，退出锁定模式
    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  
  double coming_angle, leaving_angle;// 小陀螺或前哨站情况：定义进入和离开角度
  if (target.name == ArmorName::outpost) 
  {
    coming_angle = 70 *3.14159/180.0;// 前哨站进入角度70度（转弧度）
    leaving_angle = 25 *3.14159/180.0;// 前哨站离开角度25度（转弧度）
  } 
  else 
  {
    coming_angle = comming_angle_;// 小陀螺进入角度（配置文件读取）
    leaving_angle = leaving_angle_;// 小陀螺离开角度（配置文件读取）
  }

  // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
  for (int i = 0; i < armor_num; i++) 
  {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue; // 装甲板角度在进入角度范围内
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};// 目标顺时针旋转（角速度>0），选择角度差小于离开角度的装甲板（正在进入视野）
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]}; // 目标逆时针旋转（角速度<0），选择角度差大于-离开角度的装甲板（正在进入视野）
  }
  // 无符合条件的装甲板，返回无效
  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim