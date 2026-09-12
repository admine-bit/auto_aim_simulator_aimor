// 包含Solver类的头文件
#include "solver.hpp"

// 包含YAML配置文件解析库
#include <yaml-cpp/yaml.h>

// 包含标准容器vector头文件
#include <cstdlib>
#include <cmath>
#include <vector>

// 包含工具类：日志器、数学工具
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

// 命名空间：自动瞄准模块
namespace auto_aim
{
// 常量定义：灯条长度（单位：米）
constexpr double LIGHTBAR_LENGTH = 56e-3;     // m
// 常量定义：大装甲板宽度（单位：米）
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // m
// 常量定义：小装甲板宽度（单位：米）
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // m

// 大装甲板的3D特征点（装甲板坐标系）
// 坐标定义：x=0（装甲板平面法向），y±半宽，z±灯条半长
const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};
// 小装甲板的3D特征点（装甲板坐标系）
// 坐标定义：x=0（装甲板平面法向），y±半宽，z±灯条半长
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

bool talos_simulator_enabled()
{
  return std::getenv("IMCA_TALOS_SIMULATOR") == std::string("1");
}

// Solver类构造函数：通过配置文件路径初始化参数
// 初始化云台到世界的旋转矩阵为单位矩阵
Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity()), t_gimbal2world_(Eigen::Vector3d::Zero())
{
  // 加载YAML配置文件
  auto yaml = YAML::LoadFile(config_path);

  // 从配置文件读取旋转矩阵数据（云台到IMU本体）
  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  // 从配置文件读取旋转矩阵数据（相机到云台）
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  // 从配置文件读取平移向量数据（相机到云台）
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  // 将向量数据转换为Eigen旋转矩阵（行优先）
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  // 将向量数据转换为Eigen平移向量
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  // 从配置文件读取相机内参矩阵数据
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  // 从配置文件读取畸变系数数据
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  // 将向量数据转换为Eigen相机内参矩阵（行优先）
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  // 将向量数据转换为Eigen畸变系数
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  // 将Eigen矩阵转换为OpenCV矩阵（用于后续PnP计算）
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  if (talos_simulator_enabled()) {
    // Daedalus 默认相机为 1440x1080、垂直视场角 45°，Talos 发布的图像
    // 使用 OpenCV 相机坐标。仿真时不应复用实车标定参数。
    constexpr double width = 1440.0;
    constexpr double height = 1080.0;
    constexpr double fov_y = 45.0 * M_PI / 180.0;
    const double fy = height / (2.0 * std::tan(fov_y / 2.0));
    const double fov_x = 2.0 * std::atan(std::tan(fov_y / 2.0) * width / height);
    const double fx = width / (2.0 * std::tan(fov_x / 2.0));
    camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0.0, width / 2.0,
      0.0, fy, height / 2.0,
      0.0, 0.0, 1.0);
    distort_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
    R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
    R_camera2gimbal_ << 0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0,
      0.0, -1.0, 0.0;
    t_camera2gimbal_ = Eigen::Vector3d::Zero();
  }
}

// UAV远距离相机通过后缀读取独立的相机内参和外参
Solver::Solver(const std::string & config_path, const std::string & key_suffix) : Solver(config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto key = [&key_suffix](const std::string & name) { return name + key_suffix; };

  // 云台到IMU的参数共用，只覆盖远距离相机参数
  auto R_camera2gimbal_data = yaml[key("R_camera2gimbal")].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml[key("t_camera2gimbal")].as<std::vector<double>>();
  R_camera2gimbal_ =
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml[key("camera_matrix")].as<std::vector<double>>();
  auto distort_coeffs_data = yaml[key("distort_coeffs")].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

// 获取云台到世界坐标系的旋转矩阵
Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

// 设置云台到世界坐标系的旋转矩阵
// 参数：q - 四元数（IMU本体到IMU绝对坐标系的旋转）
void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  // 将四元数转换为旋转矩阵（IMU本体到IMU绝对坐标系）
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  // 计算云台到世界坐标系的旋转矩阵（坐标变换链：云台→IMU本体→IMU绝对坐标系→世界）
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

// 设置云台到世界坐标系的平移向量（用于补偿车体高度）
// 参数：z_chassis - 底盘离地高度 (m)
void Solver::set_t_gimbal2world(double z_chassis)
{
  // 云台到世界的 z 偏移 = 底盘离地高度
  // x, y 没有绝对位置信息（用户只给速度），保持 0
  t_gimbal2world_ = Eigen::Vector3d(0, 0, z_chassis);
}

void Solver::solve(Armor & armor) const
{
  // 根据装甲板类型选择对应的3D特征点（大/小装甲板）
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  // 旋转向量（rvec）和平移向量（tvec）（相机坐标系下）
  cv::Vec3d rvec, tvec;
  // 调用OpenCV的solvePnP求解位姿（IPPE算法，适合平面特征）
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  // 将平移向量从OpenCV格式转换为Eigen格式（相机坐标系下装甲板中心坐标）
  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  // 转换坐标到云台坐标系：云台坐标 = R(相机→云台)×相机坐标 + t(相机→云台)
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  // 转换坐标到世界坐标系：世界坐标 = R(云台→世界)×云台坐标 + t(云台→世界)
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal + t_gimbal2world_;

  // 将旋转向量转换为旋转矩阵（相机坐标系下装甲板姿态）
  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  // 转换姿态到云台坐标系：R(装甲板→云台) = R(相机→云台)×R(装甲板→相机)
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  // 转换姿态到世界坐标系：R(装甲板→世界) = R(云台→世界)×R(装甲板→云台)
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  // 计算云台坐标系下的欧拉角（yaw, pitch, roll，顺序2,1,0）
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  // 计算世界坐标系下的欧拉角（yaw, pitch, roll，顺序2,1,0）
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  // 计算世界坐标系下的yaw（方位角）、pitch（俯仰角）、distance（距离）
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  // 平衡机器人不做yaw优化（因pitch假设不成立）
  // 判断是否为平衡机器人（大装甲板且编号为3/4/5）
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);
  if (is_balance) return;

  // 对非平衡机器人优化yaw角
  optimize_yaw(armor);
}

// 重投影装甲板：将世界坐标系下的装甲板投影到图像平面
// 参数：
//   xyz_in_world - 世界坐标系下装甲板中心坐标
//   yaw - 装甲板yaw角
//   type - 装甲板类型（大/小）
//   name - 装甲板编号（影响pitch角）
// 返回：图像平面上的重投影点
std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  // 计算yaw角的正弦和余弦值
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  // 根据装甲板编号设置pitch角（前哨站为-15度，其他为15度，转换为弧度）
  auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  // 计算pitch角的正弦和余弦值
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // 构造装甲板到世界坐标系的旋转矩阵（yaw-pitch-roll，roll=0）
  // clang-format off
  const Eigen::Matrix3d R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // 获取装甲板到世界的平移向量（即装甲板中心世界坐标）
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  // 计算装甲板到相机的旋转矩阵：R = R(相机→云台)转置 × R(云台→世界)转置 × R(装甲板→世界)
  Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  // 计算装甲板到相机的平移向量：t = R(相机→云台)转置 × (R(云台→世界)转置 × (t_armor2world - t_gimbal2world) - t(相机→云台))
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * (t_armor2world - t_gimbal2world_) - t_camera2gimbal_);

  // 将旋转矩阵转换为旋转向量（OpenCV格式）
  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  // 平移向量转换为OpenCV格式
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // 重投影：将3D特征点投影到图像平面
  std::vector<cv::Point2f> image_points;
  // 根据装甲板类型选择3D特征点
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

// 计算前哨站装甲板的重投影误差（用于优化）
// 参数：armor - 装甲板对象，pitch - 待优化的pitch角
// 返回：重投影误差
double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  // 根据装甲板类型选择对应的3D特征点
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  // 求解PnP获取旋转和平移向量（相机坐标系下）
  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  // 转换平移向量到Eigen格式（相机坐标系下）
  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  // 转换到云台坐标系
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  // 转换到世界坐标系
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal + t_gimbal2world_;

  // 旋转向量转旋转矩阵（相机坐标系下装甲板姿态）
  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  // 转换到云台坐标系姿态
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  // 转换到世界坐标系姿态
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  // 计算欧拉角
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  // 计算yaw、pitch、距离
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  // 获取当前yaw角和世界坐标
  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  // 计算yaw角的正余弦
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  // 计算输入pitch角的正余弦
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // 构造装甲板到世界的旋转矩阵（基于当前yaw和输入pitch）
  // clang-format off
  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // 获取装甲板到世界的平移向量
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  // 计算装甲板到相机的旋转矩阵
  Eigen::Matrix3d _R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * _R_armor2world;
  // 计算装甲板到相机的平移向量
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * (t_armor2world - t_gimbal2world_) - t_camera2gimbal_);

  // 旋转矩阵转旋转向量（OpenCV格式）
  cv::Vec3d _rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(_R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, _rvec);
  // 平移向量转换为OpenCV格式
  cv::Vec3d _tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // 重投影3D点到图像平面
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, _rvec, _tvec, camera_matrix_, distort_coeffs_, image_points);

  // 计算重投影误差：所有点的像素距离之和
  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

// 优化装甲板的yaw角（通过搜索最小重投影误差）
// 参数：armor - 装甲板对象（输出优化后的yaw角）
void Solver::optimize_yaw(Armor & armor) const
{
  // 计算云台坐标系下的欧拉角（yaw, pitch, roll）
  Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);

  // 搜索范围：140度（转换为弧度）
  constexpr double SEARCH_RANGE = 120;  // degree
  // 计算初始yaw角（云台yaw减去搜索范围的一半，归一化到[-π, π]）
  auto yaw0 = tools::limit_rad(gimbal_ypr[0] - SEARCH_RANGE / 2 * CV_PI / 180.0);

  // 初始化最小误差和最佳yaw角
  auto min_error = 1e10;
  auto best_yaw = armor.ypr_in_world[0];

  // 遍历搜索范围内的所有yaw角（步长1度）
  for (int i = 0; i < SEARCH_RANGE; i++) 
  {
    // 计算当前yaw角（初始yaw加上i度，归一化）
    double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    // 计算当前yaw角对应的重投影误差（inclined为角度偏移）
    auto error = armor_reprojection_error(armor, yaw, (i - SEARCH_RANGE / 2) * CV_PI / 180.0);

    // 更新最小误差和最佳yaw角
    if (error < min_error) 
    {
      min_error = error;
      best_yaw = yaw;
    }
  }

  // 保存原始yaw角，更新优化后的yaw角
  armor.yaw_raw = armor.ypr_in_world[0];
  armor.ypr_in_world[0] = best_yaw;
}

// 自定义代价函数（SJTU代价）：综合像素距离和角度距离
// 参数：
//   cv_refs - 参考点（重投影点）
//   cv_pts - 实际点（图像检测点）
//   inclined - 倾斜系数（影响像素/角度代价权重）
// 返回：计算得到的代价
double Solver::SJTU_cost(
  const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
  const double & inclined) const
{
  // 点的数量
  std::size_t size = cv_refs.size();
  // 转换为Eigen向量（便于计算）
  std::vector<Eigen::Vector2d> refs;
  std::vector<Eigen::Vector2d> pts;
  for (std::size_t i = 0u; i < size; ++i) {
    refs.emplace_back(cv_refs[i].x, cv_refs[i].y);
    pts.emplace_back(cv_pts[i].x, cv_pts[i].y);
  }
  // 总代价
  double cost = 0.;
  for (std::size_t i = 0u; i < size; ++i) {
    // 下一个点的索引（形成闭合多边形）
    std::size_t p = (i + 1u) % size;
    // 计算线段向量（参考与实际）
    Eigen::Vector2d ref_d = refs[p] - refs[i];  // 标准线段向量
    Eigen::Vector2d pt_d = pts[p] - pts[i];     // 实际线段向量
    // 像素距离代价：(起点误差+终点误差)/2 + 长度差，归一化到线段长度
    double pixel_dis =  // dis 是指方差平面内到原点的距离
      (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm()) +
       std::fabs(ref_d.norm() - pt_d.norm())) /
      ref_d.norm();
    // 角度距离代价：线段夹角（弧度），归一化到线段长度
    double angular_dis = ref_d.norm() * tools::get_abs_angle(ref_d, pt_d) / ref_d.norm();
    // 综合代价：像素代价×sin(inclined)² + 角度代价×cos(inclined)² × 2
    // （重投影像素误差越大，角度代价权重越高）
    double cost_i =
      tools::square(pixel_dis * std::sin(inclined)) +
      tools::square(angular_dis * std::cos(inclined)) * 2.0;  // DETECTOR_ERROR_PIXEL_BY_SLOPE
    // 累加平方根代价（降低大误差的影响）
    cost += std::sqrt(cost_i);
  }
  return cost;
}

// 计算装甲板的重投影误差
// 参数：
//   armor - 装甲板对象（含图像检测点）
//   yaw - 待评估的yaw角
//   inclined - 倾斜系数（用于SJTU代价）
// 返回：重投影误差（像素距离之和）
double Solver::armor_reprojection_error(
  const Armor & armor, double yaw, const double & inclined) const
{
  // 重投影装甲板到图像平面
  auto image_points = reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  // 计算所有点的像素距离之和
  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  // 可选：使用SJTU代价函数计算误差
  // auto error = SJTU_cost(image_points, armor.points, inclined);

  return error;
}

// 世界坐标到像素坐标的转换
// 参数：worldPoints - 世界坐标系下的3D点集
// 返回：图像平面上的像素点集（仅保留z>0的有效点）
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  // 计算世界到相机的旋转矩阵：R = R(相机→云台)转置 × R(云台→世界)转置
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  // 计算世界到相机的平移向量：t = R(相机→云台)转置 × (-R(云台→世界)转置 × t_gimbal2world - t(相机→云台))
  Eigen::Vector3d t_world2camera = R_camera2gimbal_.transpose() * (-R_gimbal2world_.transpose() * t_gimbal2world_ - t_camera2gimbal_);

  // 转换旋转和平移矩阵为OpenCV格式
  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  // 筛选有效点（相机坐标系下z>0，即点在相机前方）
  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    // 转换世界点到Eigen格式
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    // 转换到相机坐标系
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    // 仅保留z>0的点
    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }
  // 无有效点时返回空
  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  // 重投影有效3D点到图像平面
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}
}  // namespace
