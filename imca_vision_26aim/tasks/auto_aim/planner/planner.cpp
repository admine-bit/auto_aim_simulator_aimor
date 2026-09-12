#include "planner.hpp"

#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

#include "io/gimbal/gimbal.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
// 构造函数：从配置文件初始化规划器参数，并设置MPC求解器
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);  // 加载YAML配置文件
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.325;  // 读取偏航角补偿值并转换为弧度
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.325;  // 读取俯仰角补偿值并转换为弧度
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");  // 读取开火判断的位置误差阈值
  fire_hold_time_ = tools::read<double>(yaml, "fire_hold_time");  // 读取开火短脉冲补齐时间
  decision_speed_ = tools::read<double>(yaml, "decision_speed");  // 读取速度决策阈值（区分高速/低速）
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");  // 高速时的预测延迟时间
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");  // 低速时的预测延迟时间

  // 初始化fire脉冲补齐状态
  fire_high_start_ = std::chrono::steady_clock::time_point::min();
  fire_hold_until_ = std::chrono::steady_clock::time_point::min();

  setup_yaw_solver(config_path);  // 初始化偏航轴MPC求解器
  setup_pitch_solver(config_path);  // 初始化俯仰轴MPC求解器
}

// 规划函数（输入确定的Target）：生成云台控制量与开火决策
Plan Planner::plan(Target target, double bullet_speed)
{
  // 0. 检查子弹速度有效性
  if (bullet_speed < 10 || bullet_speed > 25) {  // 若子弹速度超出合理范围（10~25 m/s）
    bullet_speed = 23;  // 设为默认值23 m/s
  }

  // 1. 预测子弹飞行时间，并据此预测目标未来位置
  Eigen::Vector3d xyz;  // 存储装甲板的三维坐标(x,y,z)
  auto min_dist = 1e10;  // 初始化最小装甲板距离为极大值
  for (auto & xyza : target.armor_xyza_list()) 
  {  // 遍历所有装甲板信息
    auto dist = xyza.head<2>().norm();  // 计算装甲板在xy平面的距离（sqrt(x²+y²)）
    if (dist < min_dist) {  // 找到距离最小的装甲板（优先瞄准最近装甲板）
      min_dist = dist;
      xyz = xyza.head<3>();  // 记录该装甲板的三维坐标
    }
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());  // 计算子弹轨迹（含飞行时间）
  target.predict(bullet_traj.fly_time);  // 根据子弹飞行时间预测目标未来位置

  // 2. 生成MPC所需的参考轨迹
  double yaw0;  // 存储初始偏航角（用于轨迹坐标系对齐）
  Trajectory traj;  // 存储MPC的参考轨迹（yaw, yaw_vel, pitch, pitch_vel）
  try {
    yaw0 = aim(target, bullet_speed)(0);  // 计算初始瞄准偏航角
    traj = get_trajectory(target, yaw0, bullet_speed);  // 生成未来HORIZON步的参考轨迹
  } catch (const std::exception & e) {  // 捕获轨迹生成/瞄准过程中的异常
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);  // 记录警告日志
    return {false};  // 返回无效规划（control=false）
  }

  // 3. 求解偏航轴MPC
  Eigen::VectorXd x0(2);  // 偏航轴初始状态向量（位置、速度）
  x0 << traj(0, 0), traj(1, 0);  // 设置偏航轴初始状态为轨迹第0步的状态
  tiny_set_x0(yaw_solver_, x0);  // 向偏航轴MPC求解器设置初始状态

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);  // 设置偏航轴参考轨迹（位置、速度）
  tiny_solve(yaw_solver_);  // 求解偏航轴MPC优化问题

  // 4. 求解俯仰轴MPC
  x0 << traj(2, 0), traj(3, 0);  // 设置俯仰轴初始状态为轨迹第0步的状态
  tiny_set_x0(pitch_solver_, x0);  // 向俯仰轴MPC求解器设置初始状态

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);  // 设置俯仰轴参考轨迹（位置、速度）
  tiny_solve(pitch_solver_);  // 求解俯仰轴MPC优化问题

  Plan plan;  // 初始化规划结果结构体
  plan.control = true;  // 标记为需要云台控制

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);  // 设置目标偏航角（参考轨迹中间步 + 初始角）
  plan.target_pitch = traj(2, HALF_HORIZON);  // 设置目标俯仰角（参考轨迹中间步）

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);  // 设置偏航角控制量（MPC求解结果 + 初始角）
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);  // 设置偏航角速度控制量（MPC求解结果）
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);  // 设置偏航角加速度控制量（MPC求解结果）

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);  // 设置俯仰角控制量（MPC求解结果）
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);  // 设置俯仰角速度控制量（MPC求解结果）
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);  // 设置俯仰角加速度控制量（MPC求解结果）

  auto shoot_offset_ = 2;  // 开火判断的轨迹步长偏移（提前判断未来几步的误差）
  plan.fire =
    std::hypot(  // 计算位置误差的欧氏距离
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;  // 若误差小于阈值则开火

  // 对短开火脉冲做补齐：短于 fire_hold_time_ 才补齐，长于则不补
  auto now = std::chrono::steady_clock::now();
  bool raw_fire = plan.fire;  // 原始开火判定（由MPC误差给出）
  // 上升沿：记录原始fire拉高时刻
  if (raw_fire && !last_raw_fire_) {
    fire_high_start_ = now;
  }
  // 下降沿：若本次高电平持续时间不足，则补齐到 fire_hold_time_
  if (!raw_fire && last_raw_fire_) {
    auto high_dt = tools::delta_time(now, fire_high_start_);
    if (high_dt < fire_hold_time_) {
      fire_hold_until_ =
        now + std::chrono::microseconds(static_cast<int>((fire_hold_time_ - high_dt) * 1e6));
    } else {
      fire_hold_until_ = std::chrono::steady_clock::time_point::min();
    }
  }
  // 最终开火：原始fire为真，或处于补齐保持窗口内
  plan.fire = raw_fire || (now < fire_hold_until_);
  last_raw_fire_ = raw_fire;

  
  return plan;  // 返回规划结果
}

// 规划函数（输入可选Target）：处理目标存在性，添加预测延迟后调用确定Target的plan函数
Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};  // 若目标不存在，返回无效规划

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;  // 根据目标速度选择预测延迟时间

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));  // 计算未来时间点（当前时间 + 延迟）

  target->predict(future);  // 预测目标到未来时间点的位置

  return plan(*target, bullet_speed);  // 调用带确定Target的plan函数
}

// 初始化偏航轴MPC求解器：从配置读取参数，设置状态转移、约束、权重等
void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);  // 加载YAML配置文件
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");  // 读取偏航轴最大允许角加速度
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");  // 读取偏航轴状态权重（位置、速度）
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");  // 读取偏航轴控制输入权重（加速度）

  Eigen::MatrixXd A{{1, DT}, {0, 1}};  // 偏航轴状态转移矩阵（匀加速运动模型：x_k+1 = x_k + v_k*DT；v_k+1 = v_k + a_k*DT）
  Eigen::MatrixXd B{{0}, {DT}};  // 偏航轴控制输入矩阵（加速度对速度的影响：v_k+1 = v_k + a_k*DT）
  Eigen::VectorXd f{{0, 0}};  // 偏航轴状态偏移向量（无偏移）
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());  // 构造状态权重对角矩阵
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());  // 构造控制输入权重对角矩阵
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);  // 初始化MPC求解器

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);  // 偏航轴状态（位置、速度）最小值约束
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);  // 偏航轴状态最大值约束
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);  // 偏航轴控制输入（加速度）最小值约束
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);  // 偏航轴控制输入最大值约束
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);  // 设置MPC求解的约束

  yaw_solver_->settings->max_iter = 10;  // 设置偏航轴MPC求解的最大迭代次数
}

// 初始化俯仰轴MPC求解器：逻辑与偏航轴一致，针对俯仰自由度
void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);  // 加载YAML配置文件
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");  // 读取俯仰轴最大允许角加速度
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");  // 读取俯仰轴状态权重（位置、速度）
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");  // 读取俯仰轴控制输入权重（加速度）

  Eigen::MatrixXd A{{1, DT}, {0, 1}};  // 俯仰轴状态转移矩阵（匀加速运动模型）
  Eigen::MatrixXd B{{0}, {DT}};  // 俯仰轴控制输入矩阵
  Eigen::VectorXd f{{0, 0}};  // 俯仰轴状态偏移向量
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());  // 构造状态权重对角矩阵
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());  // 构造控制输入权重对角矩阵
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);  // 初始化MPC求解器

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);  // 俯仰轴状态最小值约束
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);  // 俯仰轴状态最大值约束
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);  // 俯仰轴控制输入最小值约束
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);  // 俯仰轴控制输入最大值约束
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);  // 设置MPC求解的约束

  pitch_solver_->settings->max_iter = 10;  // 设置俯仰轴MPC求解的最大迭代次数
}

// 瞄准计算：根据装甲板位置计算所需偏航角和俯仰角
Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;  // 存储装甲板三维坐标(x,y,z)
  double yaw;  // 存储装甲板的偏航角
  auto min_dist = 1e10;  // 初始化最小装甲板距离为极大值

  for (auto & xyza : target.armor_xyza_list()) {  // 遍历所有装甲板信息
    auto dist = xyza.head<2>().norm();  // 计算装甲板在xy平面的距离
    if (dist < min_dist) {  // 找到距离最小的装甲板
      min_dist = dist;
      xyz = xyza.head<3>();  // 记录三维坐标
      yaw = xyza[3];  // 记录偏航角
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);  // 记录调试用的装甲板坐标与偏航角

  auto azim = std::atan2(xyz.y(), xyz.x());  // 计算装甲板的方位角（绕z轴的角度）
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());  // 计算子弹轨迹（含俯仰角）
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");  // 若轨迹不可解则抛异常

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};  // 返回瞄准所需的偏航角（方位角+补偿）和俯仰角（负轨迹俯仰角-补偿）
}

// 生成MPC参考轨迹：基于目标未来位置，生成HORIZON步的偏航/俯仰位置与速度
Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;  // 初始化轨迹矩阵（4行HORIZON列：yaw, yaw_vel, pitch, pitch_vel）

  target.predict(-DT * (HALF_HORIZON + 1));  // 将目标回退到过去的时间点（用于生成完整轨迹的起始点）
  auto yaw_pitch_last = aim(target, bullet_speed);  // 计算回退后时间点的瞄准角度

  target.predict(DT);  // 预测目标到下一个时间点
  auto yaw_pitch = aim(target, bullet_speed);  // 计算当前时间点的瞄准角度

  for (int i = 0; i < HORIZON; i++) {  // 生成HORIZON步的轨迹
    target.predict(DT);  // 预测目标到下一个时间点
    auto yaw_pitch_next = aim(target, bullet_speed);  // 计算下一时间点的瞄准角度

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);  // 中心差分计算偏航角速度
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);  // 中心差分计算俯仰角速度

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;  // 填充轨迹的当前列（位置相对初始角、速度）

    yaw_pitch_last = yaw_pitch;  // 更新上一时刻的瞄准角度
    yaw_pitch = yaw_pitch_next;  // 更新当前时刻的瞄准角度
  }

  return traj;  // 返回生成的参考轨迹
}

}  // namespace auto_aim