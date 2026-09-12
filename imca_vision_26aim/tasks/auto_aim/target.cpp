// 包含Target类的实现依赖头文件，定义目标状态估计的核心逻辑
#include "target.hpp"

// 包含数值计算工具头文件，用于累加、均值等操作
#include <iostream>
#include <numeric>

// 包含日志工具头文件，用于打印调试/信息日志，辅助问题定位
#include "tools/logger.hpp"
// 包含数学工具头文件，提供角度限制、坐标转换、时间差计算等基础函数
#include "tools/math_tools.hpp"

// 进入自动瞄准命名空间，避免与其他模块命名冲突
namespace auto_aim
{
namespace
{
// 自身位移和视觉世界坐标之间存在少量配准误差，不能把 Bu 当作完全无噪声的精确输入。
constexpr double kSelfDisplacementStdRatio = 0.0;
}

// 定义全局变量：目标当前的偏航角（yaw），外部模块（如Planner）可直接访问该角度用于控制
float target_yaw = 0;
// 定义全局变量：目标当前的俯仰角（pitch），外部模块可直接访问该角度用于控制
float target_pitch = 0;

// 【Target类构造函数1】从实际检测到的装甲板初始化目标，适用于真实场景
// 参数说明：
// - armor：检测到的装甲板对象（含世界坐标系下的xyz、ypr、ypd等状态）
// - t：当前时间戳（用于EKF时间同步，计算预测时间差）
// - radius：目标旋转中心到装甲板的距离（如机器人底盘半径，需根据目标尺寸手动设置）
// - armor_num：目标的装甲板总数（如标准机器人为4个，前哨站为3个）
// - P0_dig：EKF初始状态协方差矩阵的对角线元素（表示各状态的初始不确定性）
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),        // 初始化目标名称（与装甲板名称一致，如"outpost"前哨站）
  armor_type(armor.type),  // 初始化装甲板类型（如"small"小装甲板、"large"大装甲板）
  jumped(false),           // 初始化装甲板跳变标记（初始无跳变，跳变指匹配到非0ID装甲板）
  last_id(0),              // 初始化上一次匹配的装甲板ID（初始为0，用于判断切换）
  update_count_(0),        // 初始化EKF状态更新次数（用于判断EKF是否收敛）
  armor_num_(armor_num),   // 初始化目标的装甲板总数
  t_(t),                   // 初始化时间戳（记录当前初始化时间，用于后续预测的时间差计算）
  is_switch_(false),       // 初始化装甲板切换标记（初始无切换）
  is_converged_(false),    // 初始化EKF收敛标记（初始未收敛）
  switch_count_(0)         // 初始化装甲板切换次数（用于判断目标是否异常运动）
{
  auto r = radius;            // 简化变量名：目标旋转半径
  priority = armor.priority;  // 初始化目标优先级（与装甲板优先级一致，高优先级目标优先打击）
  const Eigen::VectorXd & xyz = armor.xyz_in_world;  // 获取装甲板在世界坐标系下的xyz坐标
  const Eigen::VectorXd & ypr = armor.ypr_in_world;  // 获取装甲板在世界坐标系下的yaw/pitch/roll角

  ref_z_ = xyz[2];  // 初始化独立z参考值

  // 计算目标旋转中心的世界坐标系坐标（装甲板位置 + 旋转半径方向的偏移）
  auto center_x =
    xyz[0] + r * std::cos(ypr[0]);  // 旋转中心x：装甲板x + r*cos(装甲板yaw)（沿yaw方向偏移）
  auto center_y = xyz[1] + r * std::sin(ypr[0]);  // 旋转中心y：装甲板y + r*sin(装甲板yaw)
  auto center_z = xyz[2];                         // 旋转中心z：与装甲板z一致（假设无竖直方向偏移）

  // 初始化EKF的状态向量x0（11维，顺序固定，对应目标核心运动状态）
  // 状态维度定义：[0:x位置, 1:x速度, 2:y位置, 3:y速度, 4:z位置, 5:z速度, 6:yaw角, 7:yaw角速度, 8:旋转半径r, 9:长度差l, 10:高度差h]
  double init_h = 0;
  if(name == ArmorName::outpost){
    init_h = -0.1;
  }
  Eigen::VectorXd x0{
    {center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, init_h}};  // 初始速度/长度差/高度差设为0

  Eigen::MatrixXd P0 =
    P0_dig.asDiagonal();  // 构建EKF初始协方差矩阵（对角矩阵，不确定性由P0_dig定义）

  // 定义EKF状态更新的"加法函数"：处理角度溢出，确保yaw角始终在[-π, π]范围内
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;      // 状态向量加法（预测/更新时的状态增量计算）
    c[6] = tools::limit_rad(c[6]);  // 第6维（yaw角）用limit_rad函数限制范围，避免角度累积溢出
    return c;
  };

  // 初始化扩展卡尔曼滤波器（EKF）：传入初始状态x0、初始协方差P0、角度处理加法函数x_add
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  // 完成EKF初始化，用于后续状态估计
}

// 【Target类构造函数2】手动初始化目标，适用于测试/仿真场景（无需实际检测装甲板）
// 参数说明：
// - x：目标初始x位置（世界坐标系）
// - vyaw：目标初始yaw角速度（rad/s，用于模拟旋转运动）
// - radius：目标旋转半径（m）
// - h：目标高度差（m，用于区分长短轴装甲板的z坐标）
Target::Target(double x, double vyaw, double radius, double h)
: armor_num_(4)  // 默认目标有4个装甲板
{
  // 手动初始化EKF状态向量x0（11维，仅x、vyaw、radius、h非0，其余状态设为0）
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  // 手动初始化EKF协方差矩阵对角线（全0，表示初始状态完全确定，仅用于测试）
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();  // 构建协方差矩阵（测试场景无不确定性）

  // 定义EKF状态加法函数：处理yaw角溢出（同构造函数1）
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  // 初始化EKF（测试场景专用，初始状态无不确定性）
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  // 完成EKF初始化
}

void Target::init(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
{
  name = armor.name;
  armor_type = armor.type;
  jumped = false;
  last_id = 0;
  update_count_ = 0;
  armor_num_ = armor_num;
  t_ = t;
  switch_count_ = 0;
  is_switch_ = false;
  is_converged_ = false;

  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  ref_z_ = xyz[2];  // 初始化独立z参考值

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, -0.1}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

// Aimer/Planner 的未来预测：未来自身位置未知，用当前自身速度近似未来位移
void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt, v_self_world_ * dt);
  t_ = t;
}

void Target::predict(
  std::chrono::steady_clock::time_point t, const Eigen::Vector2d & delta_self_world)
{
  // Tracker 已经从两个同步绝对位置算出了真实位移，不再经过速度积分
  auto dt = tools::delta_time(t, t_);
  predict(dt, delta_self_world);
  t_ = t;
}

// Aimer/Planner 按时间增量预测：短时间内假设自身保持最近一次差分速度
void Target::predict(double dt)
{
  predict(dt, v_self_world_ * dt);
}

void Target::predict(double dt, const Eigen::Vector2d & delta_self_world)
{
  // 从EKF当前状态中提取速度分量（用于动态调整过程噪声，默认注释，可按需启用）
  double vx = ekf_.x[1];  // 第1维：x方向速度（m/s）
  double vy = ekf_.x[3];  // 第3维：y方向速度（m/s）
  double vz = ekf_.x[5];  // 第5维：z方向速度（m/s）
  // 计算目标合速度（用于动态调整过程噪声，速度越大噪声越大，默认注释）
  double speed = std::sqrt(vx * vx + vy * vy + vz * vz);

  // 构建EKF的状态转移矩阵F（11x11，基于匀加速运动模型，假设加速度为过程噪声）
  // 矩阵含义：每行对应一个状态的转移关系，如x位置 = 上一x位置 + x速度*dt
  // clang-format off（禁用代码格式化，保持矩阵结构清晰）
  Eigen::MatrixXd F{
    {1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // 0:x位置 = x + vx*dt
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},   // 1:x速度 = vx（匀速假设，加速度由噪声体现）
    {0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0},  // 2:y位置 = y + vy*dt
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},   // 3:y速度 = vy
    {0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0},  // 4:z位置 = z + vz*dt
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},   // 5:z速度 = vz
    {0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0},  // 6:yaw角 = yaw + wyaw*dt
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},   // 7:yaw角速度 = wyaw
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},   // 8:旋转半径r = r（固定，无变化）
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},   // 9:长度差l = l（固定）
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}    //10:高度差h = h（固定）
  };
  // clang-format on（恢复代码格式化）

  // 【过程噪声矩阵Q的构建】基于Piecewise White Noise Model（分段白噪声模型）
  // 参考链接：https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1,
    v2;  // v1：线加速度方差（控制x/y/z方向的过程噪声）；v2：角加速度方差（控制yaw方向的过程噪声）
  if (name == ArmorName::outpost)  // 若目标是前哨站（运动规律固定，速度慢，噪声小）
  {
    v1 = 10;   // 前哨站线加速度方差（小值，避免过度敏感）
    v2 = 0.1;  // 前哨站角加速度方差（小值，前哨站旋转稳定）
  } else       // 若目标是普通机器人（运动灵活，速度快，噪声大）
  {
    v1 = 100;  // 普通目标线加速度方差（大值，适应快速移动）
    v2 = 350;  // 普通目标角加速度方差（大值，适应快速转向）
    // 动态调整噪声（默认注释，按需启用：低速时噪声小，高速时噪声大）
    // v1 = speed < 0.05 ? 0.1 : 100;
    // v2 = speed < 0.1 ? 0.1 : 400;
  }
  // 计算过程噪声矩阵的系数（基于时间差dt的幂次，符合白噪声模型的数学推导）
  auto a = dt * dt * dt * dt / 4;  // 位置噪声系数（dt^4/4，影响x/y/z/yaw位置的不确定性）
  auto b = dt * dt * dt / 2;       // 位置-速度交叉项系数（dt^3/2，关联位置与速度的噪声）
  auto c = dt * dt;                // 速度噪声系数（dt^2，影响x/y/z/yaw速度的不确定性）

  // 构建过程噪声矩阵Q（11x11，仅线运动和角运动有噪声，固定参数r/l/h无噪声）
  // clang-format off
    Eigen::MatrixXd Q{
      {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},  // x位置噪声
      {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},  // x速度噪声
      {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},  // y位置噪声
      {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},  // y速度噪声
      {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},  // z位置噪声
      {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},  // z速度噪声
      {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},  // yaw角噪声
      {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},  // yaw角速度噪声
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},  // r无噪声
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},  // l无噪声
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}   // h无噪声
    };
  // clang-format on

  // 将自身位移输入的不确定度只加到目标中心 x/y。
  // 这样平移时剩余的小残差会优先修正中心，而不是通过观测耦合持续推大 r。
  const double self_displacement_std =
    kSelfDisplacementStdRatio * delta_self_world.norm();
  const double self_displacement_variance = self_displacement_std * self_displacement_std;
  Q(0, 0) += self_displacement_variance;
  Q(2, 2) += self_displacement_variance;

  // 状态 x/y 是以当前云台为原点、轴方向与 W0 对齐的目标相对位置。
  // 自身沿 W0 正方向移动 delta 后，固定目标的相对坐标应反向减少同样的 delta。
  Eigen::VectorXd Bu = Eigen::VectorXd::Zero(11);
  Bu(0) = -delta_self_world.x();  // 中心相对 x 减去自身 W0-x 位移
  Bu(2) = -delta_self_world.y();  // 中心相对 y 减去自身 W0-y 位移

  // 定义EKF预测的"状态转移函数"f：输入当前状态x，输出预测后的先验状态x_prior
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x + Bu;       // 目标匀速状态转移 + 自身平移位移补偿
    x_prior[6] = tools::limit_rad(x_prior[6]);  // 限制yaw角在[-π, π]，避免预测过程中角度溢出
    return x_prior;
  };

  // 【前哨站特殊处理】若EKF已收敛且前哨站yaw角速度过大，强制限制角速度（避免预测发散）
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 1.3)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.513: -2.513;  //0.8pai 限制在±2.51rad/s（约144°/s，符合前哨站实际转速）

  if (this->name == ArmorName::outpost && std::abs(this->ekf_.x[10]) < 0.10) {
    this->ekf_.x[10] = -0.1;
  }

  // 调用EKF的预测方法：传入状态转移矩阵F、过程噪声Q、状态转移函数f，完成先验状态估计
  ekf_.predict(F, Q, f);
}

// 【更新函数】根据新检测到的装甲板更新EKF状态（EKF核心更新步骤，融合观测值）
// 参数：armor - 新检测到的装甲板对象（含当前帧的观测数据）
void Target::update(const Armor & armor)
{
  // 装甲板匹配：从目标的装甲板列表中，找到与当前检测装甲板最匹配的ID
  int id;                       // 存储匹配到的装甲板ID（0~armor_num_-1）
  auto min_angle_error = 1e10;  // 初始化最小角度误差为极大值（用于筛选最优匹配）
  const std::vector<Eigen::Vector4d> & xyza_list =
    armor_xyza_list();  // 获取目标当前预测的所有装甲板xyza（x,y,z,angle）

  // 构建"装甲板xyza + ID"的配对列表（方便后续按距离排序）
  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});  // 每个元素存储：{装甲板xyza, 装甲板ID}
  }

  // 对装甲板列表按"距离"升序排序（距离越小越靠前，减少远距离异常装甲板的干扰）
  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      // 将装甲板xyz坐标转换为ypd（yaw偏航角、pitch俯仰角、distance距离）
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];  // 按距离（ypd[2]）升序排序，近距离装甲板优先
    });

  // 从排序后的前3个近距离装甲板中，选择角度误差最小的作为最优匹配
  for (int i = 0; i < 3; i++) {
    const auto & xyza = xyza_i_list[i].first;            // 当前候选装甲板的xyza（x,y,z,angle）
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));  // 候选装甲板的ypd（yaw/pitch/distance）
    // 计算角度误差：装甲板yaw角误差 + ypd yaw角误差（均用limit_rad限制在[-π, π]，避免跨0点误差）
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    // 更新最小角度误差和对应的装甲板ID
    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;     // 记录最优匹配的装甲板ID
      min_angle_error = angle_error;  // 更新最小角度误差
    }
  }

  // 标记是否发生"装甲板跳变"：若匹配ID非0，说明目标快速旋转导致ID跳变（需关注是否异常）
  if (id != 0) jumped = true;

  bool is_init = false;
  // 标记是否发生"装甲板切换"：当前匹配ID与上一次ID不同，说明装甲板视角切换
  if (id != last_id)
  {
    is_switch_ = true;  // 发生切换
  }
  else
  {
    is_switch_ = false;  // 未切换
  }

  if (is_switch_)
  {
    switch_count_++;

    // 追踪哨站的最大观测z，作为独立的几何参考（不依赖EKF，避免循环依赖）
    if (name == ArmorName::outpost && armor.xyz_in_world[2] > ref_z_) {
      ref_z_ = armor.xyz_in_world[2];
    }

    // 混合z估计：EKF未收敛时用独立几何估计，收敛后用EKF预测
    double expected_z;
    if (name == ArmorName::outpost && !convergened()) {
      expected_z = ref_z_ - static_cast<double>(id) * 0.1;
    } else {
      expected_z = h_armor_xyz(ekf_.x, id)[2];
    }

    double z_deviation = armor.xyz_in_world[2] - expected_z;
    std::cout << " dis " << z_deviation << " armor " << armor.xyz_in_world[2]
              << " pred_z " << expected_z << " id " << id
              << " convergened:" << convergened() << std::endl;
    if (name == ArmorName::outpost) {
      // 动态计算Z轴偏差阈值（随距离增加而放宽）
      double z_threshold = 0.15;
      // if (armor.ypd_in_world[2] > 3.0) {
      //   z_threshold += 0.008 * (armor.ypd_in_world[2] - 3.0);
      // }

      // 分层EKF重置策略
      if (z_deviation > z_threshold) {
        is_init = true;  // 标记已处理，跳过后续的常规 update_ypda

        if (z_deviation < 0.25) {
          // 轻度偏差：仅修正Z轴漂移，保留yaw和角速度
          ekf_.x[4] = armor.xyz_in_world[2];
          ekf_.x[10] = -0.1;
          ekf_.P(4, 4) = 1.0;
          ekf_.P(10, 10) = 0.6;

          tools::logger()->warn("[Outpost] Z高度差: {:.3f}m, 软重置 Z", z_deviation);
        } else {
          // 重度偏差：认为装甲板ID匹配错误，按前哨站三块装甲板的120度间隔修正yaw
          ekf_.x[4] = armor.xyz_in_world[2];
          ekf_.x[10] = -0.1;

          double yaw_correction = (ekf_.x[7] > 0) ? -2.094 : 2.094;
          ekf_.x[6] = tools::limit_rad(ekf_.x[6] + yaw_correction);

          ekf_.P(4, 4) = 1.0;
          ekf_.P(6, 6) = 1.0;
          ekf_.P(7, 7) = 10.0;
          ekf_.P(10, 10) = 0.6;

          last_id = 0;
          switch_count_ = 0;

          tools::logger()->warn(
            "[Outpost] ID错误! Z高度差{:.3f}m, 角度旋转: {:.1f}°", z_deviation,
            yaw_correction * 57.3);
        }
      }
    }
  }

  if (!is_init) {
    // 更新上一次匹配的ID为当前ID（作为下一次更新的基准）
    last_id = id;
    // 累计EKF状态更新次数（用于判断EKF是否收敛）
    update_count_++;

    // 调用update_ypda函数，基于匹配的装甲板ID和观测值，完成EKF的状态更新
    update_ypda(armor, id);
  }
}

// 【EKF更新核心函数】基于ypda（yaw/pitch/distance/angle）观测值更新EKF状态
// 参数：armor - 新检测的装甲板；id - 匹配到的装甲板ID
void Target::update_ypda(const Armor & armor, int id)
{
  // 计算观测矩阵H（雅可比矩阵）：描述"目标状态x"与"观测值z"之间的线性化关系（通过h_jacobian函数）
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // 固定观测噪声（默认注释，按需启用，当前使用动态计算的R）
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};

  // 动态计算观测噪声：基于装甲板yaw角与旋转中心yaw角的偏差，偏差越大噪声越大（降低异常观测的权重）
  auto center_yaw =
    std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);  // 旋转中心的yaw角（atan2(y,x)）
  auto delta_angle =
    tools::limit_rad(armor.ypr_in_world[0] - center_yaw);  // 装甲板yaw与中心yaw的偏差（限制范围）
  // 构建观测噪声协方差矩阵的对角线R_dig（4维，对应观测值z的4个维度，动态调整噪声大小）xiugai
  Eigen::VectorXd R_dig{{
    4e-3,                                // 维度0：ypd.yaw的噪声（固定小值，yaw观测稳定）
    4e-3,                                // 维度1：ypd.pitch的噪声（固定小值，pitch观测稳定）
    log(std::abs(delta_angle) + 1) + 1,  // 维度2：ypd.distance的噪声（偏差越大，噪声越大）
    log(std::abs(armor.ypd_in_world[2]) + 1) / 200 +
      9e-2  // 维度3：armor.yaw的噪声（距离越远，噪声越大）
  }};

  // 构建观测噪声协方差矩阵R（对角矩阵，对角线元素为R_dig，描述观测值的不确定性）
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性观测函数h：输入目标状态x，输出预测的观测值z_hat（4维，与实际观测z维度对应）
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);   // 基于状态x和ID，计算装甲板的xyz坐标
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);  // 将xyz转换为ypd（yaw/pitch/distance）
    // 计算装甲板的yaw角：状态x的yaw角（x[6]） + ID对应的角度偏移（2π*ID/装甲板总数，均分圆周）
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};  // 返回预测观测值z_hat：[yaw, pitch, distance, angle]
  };

  // 定义观测值减法函数：处理角度溢出，确保观测值与预测值的残差在[-π, π]范围内
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;      // 计算残差：实际观测z - 预测观测z_hat
    c[0] = tools::limit_rad(c[0]);  // 维度0：ypd.yaw残差限制
    c[1] = tools::limit_rad(c[1]);  // 维度1：ypd.pitch残差限制
    c[3] = tools::limit_rad(c[3]);  // 维度3：armor.yaw残差限制
    return c;
  };

  // 从装甲板对象中提取实际观测值z的原始数据
  const Eigen::VectorXd & ypd = armor.ypd_in_world;  // 装甲板的ypd观测（yaw/pitch/distance）
  const Eigen::VectorXd & ypr = armor.ypr_in_world;  // 装甲板的ypr观测（yaw/pitch/roll）
  // 构建实际观测值z（4维，与观测函数h的输出z_hat维度完全对应）
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  // 更新全局变量：将当前观测的yaw和pitch赋值给全局变量，供外部模块（如Planner）直接使用
  target_yaw = ypd[0];
  target_pitch = ypd[1];
  // 调试日志：打印当前观测的yaw和pitch（默认注释，需调试时打开，验证观测值是否合理）
  // tools::logger()->info("[TARGET]yaw: {} pitch: {}",ypd[0] ,ypd[1]);

  // 调用EKF的更新方法：传入实际观测z、观测矩阵H、观测噪声R、观测函数h、残差处理函数z_subtract
  // 完成EKF后验状态估计（融合观测值，修正先验状态）
  ekf_.update(z, H, R, h, z_subtract);
}

// 【获取EKF状态】返回当前EKF的状态向量x（11维），外部模块（如Planner）通过该函数获取目标运动状态
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

// 【获取EKF对象】返回EKF的常量引用，外部模块可通过该引用访问EKF内部状态（如协方差矩阵）
const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

// 【生成装甲板列表】根据当前EKF状态，生成目标所有装甲板的xyza（x,y,z,angle）列表
// 返回值：std::vector<Eigen::Vector4d> - 每个元素对应一个装甲板的xyza信息
std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;  // 存储生成的装甲板列表

  // 遍历目标的所有装甲板ID（0~armor_num_-1）
  for (int i = 0; i < armor_num_; i++) {
    // 计算当前ID装甲板的yaw角：EKF状态yaw（x[6]） + ID对应的角度偏移（2π*ID/装甲板总数）
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    // 计算当前ID装甲板的xyz坐标（调用h_armor_xyz函数，考虑长短轴差异）
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    // 将xyz和angle组合为4维向量（xyza），添加到列表中
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;  // 返回所有装甲板的xyza列表（用于可视化或二次检测）
}

// 【判断EKF是否发散】基于EKF状态中的旋转半径r和长度差l的合理性判断
// 返回值：true-发散（状态异常）；false-未发散（状态正常）
bool Target::diverged() const
{
  // 判断旋转半径r是否在合理范围（0.05~0.5m，需根据目标实际尺寸调整，如机器人底盘半径约0.2m）
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  // 判断总半径（r+l）是否在合理范围（0.05~0.5m，l是长短轴的长度差，通常较小）
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  // 若r和l均在合理范围，说明EKF状态正常，返回未发散
  if (r_ok && l_ok) return false;

  // 调试日志：打印发散时的r和l值（帮助定位发散原因，如检测错误导致r异常）
  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;  // r或l异常，返回发散
}

// 【判断EKF是否收敛】基于更新次数和是否发散判断，收敛后状态才能用于控制
// 返回值：true-收敛；false-未收敛
bool Target::convergened()
{
  // 普通目标收敛条件：EKF更新次数>3次（获取足够观测数据）且未发散
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;  // 标记为已收敛
  }

  // 前哨站收敛条件：更新次数>10次（前哨站运动慢，需更多数据稳定）且未发散
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;  // 标记为已收敛
  }

  return is_converged_;  // 返回收敛状态
}

// 【计算装甲板xyz】基于目标状态x和装甲板ID，计算该装甲板的世界坐标系xyz坐标（考虑长短轴差异）
// 参数：x - EKF状态向量；id - 装甲板ID
// 返回值：Eigen::Vector3d - 装甲板的xyz坐标
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  // 计算当前装甲板的yaw角：状态x的yaw（x[6]） + ID对应的角度偏移（2π*ID/装甲板总数）
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  // 判断是否使用长轴参数：仅当装甲板总数为4且ID为1/3时（长轴装甲板），使用r+l作为半径
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  auto use_outpost = (name == ArmorName::outpost) && (armor_num_ == 3);

  // 计算装甲板的旋转半径：长轴用r+l（x[8]+x[9]），短轴用r（x[8]）
  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  // 计算装甲板x坐标：旋转中心x（x[0]） - r*cos(angle)（与旋转方向相反，确保位置正确）
  auto armor_x = x[0] - r * std::cos(angle);
  // 计算装甲板y坐标：旋转中心y（x[2]） - r*sin(angle)
  auto armor_y = x[2] - r * std::sin(angle);
  // 计算装甲板z坐标：长轴用z+h（x[4]+x[10]），短轴用z（x[4]）
  auto armor_z = x[4];
  if (use_l_h) {
    armor_z += x[10];
  } else if (use_outpost) 
  {
    armor_z += id * x[10];
  }

  return {armor_x, armor_y, armor_z};  // 返回装甲板的xyz坐标
}

// 【计算观测雅可比矩阵】基于目标状态x和装甲板ID，计算观测函数h的雅可比矩阵H（用于EKF线性化）
// 参数：x - EKF状态向量；id - 装甲板ID
// 返回值：Eigen::MatrixXd - 4x11的雅可比矩阵（观测值z对状态x的偏导数）
Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  // 计算当前装甲板的yaw角（同h_armor_xyz函数）
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  // 判断是否使用长轴参数（同h_armor_xyz函数）
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  auto use_outpost = (armor_num_ == 3) && (name == ArmorName::outpost) && (id == 1 || id == 2);

  // 计算装甲板的旋转半径（同h_armor_xyz函数）
  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  // 雅可比元素：装甲板x坐标对yaw角（x[6]）的偏导数 = r*sin(angle)
  auto dx_da = r * std::sin(angle);
  // 雅可比元素：装甲板y坐标对yaw角（x[6]）的偏导数 = -r*cos(angle)
  auto dy_da = -r * std::cos(angle);

  // 雅可比元素：装甲板x坐标对r（x[8]）的偏导数 = -cos(angle)
  auto dx_dr = -std::cos(angle);
  // 雅可比元素：装甲板y坐标对r（x[8]）的偏导数 = -sin(angle)
  auto dy_dr = -std::sin(angle);
  // 雅可比元素：装甲板x坐标对l（x[9]）的偏导数（长轴用dx_dr，短轴用0）
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  // 雅可比元素：装甲板y坐标对l（x[9]）的偏导数（长轴用dy_dr，短轴用0）
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  // 雅可比元素：装甲板z坐标对h（x[10]）的偏导数（长轴用1，短轴用0）
  auto dz_dh = (use_l_h) ? 1.0 : ((use_outpost) ? (id == 1 ? 1.0 : 2.0) : 0.0);

  // 【第一步：构建装甲板xyza对状态x的雅可比矩阵H_armor_xyza（4x11）】
  // 行含义：0-x坐标，1-y坐标，2-z坐标，3-angle；列含义：EKF的11个状态维度
  // clang-format off
    Eigen::MatrixXd H_armor_xyza{
      {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},  // x对各状态的偏导
      {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},  // y对各状态的偏导
      {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},  // z对各状态的偏导
      {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}  // angle对各状态的偏导（仅对x[6]偏导为1）
    };
  // clang-format on

  // 【第二步：构建ypd对装甲板xyz的雅可比矩阵H_armor_ypd（3x3）】
  // 调用工具函数xyz2ypd_jacobian，获取xyz坐标到ypd的线性化关系（偏导数矩阵）
  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);

  // 【第三步：构建ypda对xyza的雅可比矩阵H_armor_ypda（4x4）】
  // 行含义：0-ypd.yaw，1-ypd.pitch，2-ypd.distance，3-angle；列含义：0-x，1-y，2-z，3-angle
  // clang-format off
    Eigen::MatrixXd H_armor_ypda{
      {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},  // ypd.yaw对xyz的偏导
      {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},  // ypd.pitch对xyz的偏导
      {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},  // ypd.distance对xyz的偏导
      {                0,                 0,                 0, 1}   // angle对自身的偏导（为1）
    };
  // clang-format on

  // 【最终雅可比矩阵H】通过链式求导得到：H = H_armor_ypda * H_armor_xyza（4x11）
  return H_armor_ypda * H_armor_xyza;
}

// 【判断目标是否初始化】返回isinit标记（需确保isinit在类中正确赋值，当前默认初始化为false）
bool Target::checkinit() { return isinit; }

// 保存当前自身 W0 系速度，供 Planner/Aimer 估计尚未发生的未来自身位移
void Target::set_v_self_world(const Eigen::Vector2d & v)
{
  v_self_world_ = v;
}

}  // namespace auto_aim
