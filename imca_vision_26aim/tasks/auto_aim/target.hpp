// 条件编译宏，防止target.hpp头文件被重复包含
#ifndef AUTO_AIM__TARGET_HPP
// 定义宏AUTO_AIM__TARGET_HPP，标记头文件已被包含
#define AUTO_AIM__TARGET_HPP

// 包含Eigen库的稠密矩阵模块，用于矩阵和向量运算
#include <Eigen/Dense>
// 包含chrono库，用于处理时间戳相关操作
#include <chrono>
// 包含optional库，用于处理可选值（当前文件未直接使用，可能为后续扩展预留）
#include <optional>
// 包含queue库，用于队列数据结构（当前文件未直接使用，可能为后续扩展预留）
#include <queue>
// 包含string库，用于字符串操作
#include <string>
// 包含vector库，用于动态数组操作
#include <vector>

// 包含装甲板类的头文件，Target类依赖Armor类的定义
#include "armor.hpp"
// 包含扩展卡尔曼滤波器工具类的头文件，Target类使用EKF进行状态估计
#include "tools/extended_kalman_filter.hpp"

// 自动瞄准命名空间，封装相关类和变量，避免命名冲突
namespace auto_aim
{
// 声明外部全局变量：目标的偏航角（yaw），供外部模块访问
extern float target_yaw;
// 声明外部全局变量：目标的俯仰角（pitch），供外部模块访问
extern float target_pitch;

// 目标类，用于通过EKF估计目标（如机器人）的运动状态，管理装甲板匹配和状态更新
class Target
{
public:
  // 目标名称（如"outpost"前哨站，定义在ArmorName中）
  ArmorName name;
  // 装甲板类型（如小装甲板、大装甲板，定义在ArmorType中）
  ArmorType armor_type;
  // 目标优先级（用于多目标时的选择，定义在ArmorPriority中）
  ArmorPriority priority;
  // 标记是否发生装甲板跳变（匹配到非0ID的装甲板）
  bool jumped;
  // 上一次匹配的装甲板ID（仅用于调试）
  int last_id;  // debug only

  // 默认构造函数，创建一个空的Target对象
  Target() = default;
  // 构造函数：从检测到的装甲板初始化目标
  // 参数：armor-检测到的装甲板；t-当前时间戳；radius-目标旋转半径；armor_num-装甲板总数；P0_dig-EKF初始协方差对角线
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  // 构造函数：手动初始化目标（用于测试/仿真）
  // 参数：x-初始x位置；vyaw-初始偏航角速度；radius-旋转半径；h-高度差
  Target(double x, double vyaw, double radius, double h);

  void init(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);

  // 使用保存的自身速度预测到指定时间戳，供 Aimer/Planner 外推尚未发生的自身位移
  void predict(std::chrono::steady_clock::time_point t);
  // 使用已知真实位移预测到当前图像时间戳，供 Tracker 做逐帧 EKF 预测
  void predict(
    std::chrono::steady_clock::time_point t, const Eigen::Vector2d & delta_self_world);
  // 使用保存的自身速度预测 dt 秒，主要用于弹丸飞行时间和 MPC 轨迹预测
  void predict(double dt);
  // EKF 核心预测：delta_self_world 是该 dt 内云台安装点在 W0 下的位移
  void predict(double dt, const Eigen::Vector2d & delta_self_world);
  // 根据新检测到的装甲板更新目标状态（EKF的更新步骤）
  // 参数：armor-新检测到的装甲板
  void update(const Armor & armor);

  // 获取EKF当前的状态向量（11维，包含位置、速度、角度等）
  Eigen::VectorXd ekf_x() const;
  // 获取EKF对象的常量引用（用于访问EKF内部状态如协方差）
  const tools::ExtendedKalmanFilter & ekf() const;
  // 生成目标所有装甲板的xyza列表（x,y,z坐标和角度）
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // 判断EKF是否发散（基于旋转半径等参数是否在合理范围）
  bool diverged() const;

  // 判断EKF是否收敛（基于更新次数和是否发散）
  bool convergened();

  // 标记目标是否已初始化
  bool isinit = false;

  // 检查目标是否已初始化
  bool checkinit();

  // 保存短时匀速外推所需的自身 W0 系速度；不参与 Tracker 当前帧的真实位移补偿
  void set_v_self_world(const Eigen::Vector2d & v);

private:
  // 目标的装甲板总数（如4个）
  int armor_num_;
  // 装甲板切换次数（累计ID变化的次数）
  int switch_count_;
  // EKF状态更新的总次数
  int update_count_;


  // 标记当前是否发生装甲板切换（当前ID与上一次不同）
  bool is_switch_,
    // 标记EKF是否已收敛
    is_converged_;

  // 独立z参考值，用于ID验证时的几何估计（哨站专用，避免循环依赖EKF）
  double ref_z_ = 0;

  // 由相邻绝对位置差分得到，只在没有显式 delta 的未来 predict() 中使用
  Eigen::Vector2d v_self_world_ = Eigen::Vector2d::Zero();

  // 扩展卡尔曼滤波器实例，用于目标状态估计
  tools::ExtendedKalmanFilter ekf_;
  // 上一次更新/预测的时间戳
  std::chrono::steady_clock::time_point t_;

  // 基于ypda（yaw,pitch,distance,angle）观测值更新EKF
  // 参数：armor-新检测的装甲板；id-匹配到的装甲板ID
  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  // 计算指定ID装甲板的xyz坐标（基于目标状态x）
  // 参数：x-EKF状态向量；id-装甲板ID；返回值-装甲板的xyz坐标
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  // 计算观测模型h的雅可比矩阵（用于EKF更新时的线性化）
  // 参数：x-EKF状态向量；id-装甲板ID；返回值-4x11的雅可比矩阵
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

// 条件编译结束，对应开头的#ifndef
#endif  // AUTO_AIM__TARGET_HPP
