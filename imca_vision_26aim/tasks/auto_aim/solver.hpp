#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

// 包含Eigen库头文件（矩阵、向量运算），需在opencv2/core/eigen.hpp前包含
#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>  // Eigen几何模块（四元数、旋转矩阵等）
// 包含OpenCV与Eigen矩阵转换工具
#include <opencv2/core/eigen.hpp>

// 包含装甲板相关定义（Armor类、ArmorType、ArmorName等）
#include "armor.hpp"

// 自动瞄准模块命名空间
namespace auto_aim
{
// 位姿解算器类：负责通过PnP算法求解装甲板位姿、坐标转换及重投影优化
class Solver
{
public:
  // 构造函数：通过配置文件路径初始化相机参数和坐标转换矩阵
  // 参数：config_path - YAML配置文件路径（包含相机内参、外参等）
  explicit Solver(const std::string & config_path);

  // UAV远距离相机使用带后缀的相机标定参数
  Solver(const std::string & config_path, const std::string & key_suffix);

  // 获取云台坐标系到世界坐标系的旋转矩阵
  // 返回：Eigen::Matrix3d - 3x3旋转矩阵
  Eigen::Matrix3d R_gimbal2world() const;

  // 设置云台坐标系到世界坐标系的旋转矩阵
  // 参数：q - 四元数（用于计算IMU本体到IMU绝对坐标系的旋转）
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  // 设置云台坐标系到世界坐标系的平移向量（用于补偿车体高度）
  // 参数：z_chassis - 底盘离地高度 (m)
  void set_t_gimbal2world(double z_chassis);

  // 求解装甲板位姿：通过PnP算法计算装甲板在各坐标系下的坐标和姿态
  // 参数：armor - 装甲板对象（输入图像检测点，输出位姿信息）
  void solve(Armor & armor) const;

  // 重投影装甲板：将世界坐标系下的装甲板投影到图像平面
  // 参数：
  //   xyz_in_world - 装甲板在世界坐标系下的中心坐标
  //   yaw - 装甲板的yaw角（方位角）
  //   type - 装甲板类型（大/小）
  //   name - 装甲板编号（影响pitch角设置）
  // 返回：std::vector<cv::Point2f> - 图像平面上的重投影点集合
  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  // 计算前哨站装甲板的重投影误差（用于pitch角优化）
  // 参数：
  //   armor - 装甲板对象（含图像检测点）
  //   picth - 待评估的pitch角（俯仰角）
  // 返回：double - 重投影误差（像素距离之和）
  double oupost_reprojection_error(Armor armor, const double & picth);

  // 世界坐标系到像素坐标系的转换：将3D世界点投影到图像平面
  // 参数：worldPoints - 世界坐标系下的3D点集合
  // 返回：std::vector<cv::Point2f> - 对应图像平面上的像素点集合（仅保留相机前方的有效点）
  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & worldPoints);

private:
  // 相机内参矩阵（3x3）：包含焦距、主点坐标等相机固有参数
  cv::Mat camera_matrix_;
  // 相机畸变系数（1x5）：用于校正镜头畸变
  cv::Mat distort_coeffs_;
  // 云台坐标系到IMU本体坐标系的旋转矩阵（3x3）：坐标转换用
  Eigen::Matrix3d R_gimbal2imubody_;
  // 相机坐标系到云台坐标系的旋转矩阵（3x3）：坐标转换用
  Eigen::Matrix3d R_camera2gimbal_;
  // 相机坐标系到云台坐标系的平移向量（3x1）：坐标转换用
  Eigen::Vector3d t_camera2gimbal_;
  // 云台坐标系到世界坐标系的旋转矩阵（3x3）：实时更新，用于姿态转换
  Eigen::Matrix3d R_gimbal2world_;
  // 云台坐标系到世界坐标系的平移向量（3x1）：用于补偿车体高度
  Eigen::Vector3d t_gimbal2world_;

  // 优化装甲板yaw角：通过搜索最小重投影误差优化yaw角
  // 参数：armor - 装甲板对象（输出优化后的yaw角）
  void optimize_yaw(Armor & armor) const;

  // 计算装甲板重投影误差：评估重投影点与实际检测点的偏差
  // 参数：
  //   armor - 装甲板对象（含实际检测点）
  //   yaw - 待评估的yaw角
  //   inclined - 倾斜系数（用于SJTU代价函数）
  // 返回：double - 重投影误差（像素距离之和或SJTU代价）
  double armor_reprojection_error(const Armor & armor, double yaw, const double & inclined) const;

  // SJTU代价函数：综合像素距离和角度距离的自定义代价计算
  // 参数：
  //   cv_refs - 参考点（重投影点）
  //   cv_pts - 实际点（图像检测点）
  //   inclined - 倾斜系数（调整像素/角度代价权重）
  // 返回：double - 计算得到的综合代价
  double SJTU_cost(
    const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
    const double & inclined) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP  // 头文件保护宏结束
