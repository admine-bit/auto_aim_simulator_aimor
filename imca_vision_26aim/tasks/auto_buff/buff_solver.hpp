#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在 opencv2/core/eigen.hpp 前包含
#include <opencv2/core/eigen.hpp>

#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{
const double THETA = 2.0 * CV_PI / 5.0;

class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;
  void set_R_gimbal2world(const Eigen::Quaterniond & q);
  void reset();

  // 使用 kpt0~3 四角和 kpt5 流水灯中心做五点共面 PnP，kpt4 不进入位姿链路。
  void solve(std::optional<PowerRune> & power_rune) const;

  // 主副目标属于同一刚体：有当前观测时分别做五点 PnP，副槽短时漏检时按固定相位补全。
  void solve_dual(
    std::optional<PowerRune> & primary, std::optional<PowerRune> & secondary) const;

  // 六点仅用于可视化：target 四角、R 中心、流水灯中心。
  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, double yaw, double row) const;

  // —— 只读诊断量：不参与任何判决，仅供回放归因 ——
  /// 本帧被采纳的符面法向相对上一帧的夹角（rad）。IPPE 在两支镜像分支间切换时，
  /// 这个量会突然跳到大角度，而重投影 RMSE 几乎看不出差别。
  double last_normal_step_rad() const { return last_normal_step_rad_; }
  /// 本帧两支候选各自的世界系符面 pitch（度）。符竖直安装 ⇒ 真解 ≈0、
  /// 镜像解 ≈±2×视线仰角。两者是否分离，决定 pitch 判据能否成立。
  double last_pose_pitch_a() const { return last_pose_pitch_a_; }
  double last_pose_pitch_b() const { return last_pose_pitch_b_; }
  /// 两支候选的重投影 RMSE 差（px）。实测远小于 kPoseTiePixels ⇒ RMSE 选边等于抛硬币。
  /// 整符点集张开后这个值应当明显变大。
  double last_rmse_gap() const { return last_rmse_gap_; }
  /// 本帧参与整符联合解算的扇叶片数（1=小符或大符只看到一片，2=大符正常）。
  int last_blade_count() const { return last_blade_count_; }
  /// 整符解算自标定出的槽差符号（+1/-1，0=未标定）。
  int last_slot_sign() const { return last_slot_sign_; }

private:
  // 镜像消歧锚点的最长丢失寿命（帧）。60fps 下约 5 秒，远超规则的 200ms 换组黑屏；
  // 只在真正长时间失去目标后才清空，正常打符期间锚点始终有效。
  static constexpr int kAnchorMaxMisses = 300;

  void note_miss() const;

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  // 五点总重投影 RMSE 超过该值且 kpt5 残差显著异常时，当前帧回退四角 PnP。
  double pnp_max_rmse_px_{4.0};
  // 镜像消歧：符竖直安装 ⇒ 真解符面 pitch ≈ 0。候选的 |pitch| 必须小于该上限，
  // 且与另一支拉开 pitch_separation_rad_ 以上，pitch 先验才被采信。
  double max_plane_pitch_rad_{20.0 / 57.3};
  double pitch_separation_rad_{8.0 / 57.3};

  // IPPE 对共面点存在双解；法向保存在世界系，云台转动后仍可跨帧比较。
  // 符螺栓固定在场地、打符的车不平移 ⇒ 该法向在一局内是常数，
  // 因此只在 reset()（模式切换，可能换成对方符）时清空，不因短时丢失而清空。
  mutable std::optional<cv::Vec3d> last_primary_normal_world_;
  mutable int consecutive_misses_{0};

  // 只读诊断量，reset() 时一并清零。
  mutable double last_normal_step_rad_{0.0};
  mutable double last_pose_pitch_a_{0.0};
  mutable double last_pose_pitch_b_{0.0};
  mutable double last_rmse_gap_{0.0};
  mutable int last_blade_count_{0};
  mutable int last_slot_sign_{0};
  // 最近一次真正标定成功（两片同时可见）的槽差符号。只有一片可见时沿用它，
  // 而不是每帧重新猜——猜错会让副视图的 roll 差反向 144°。
  mutable int calibrated_slot_sign_{0};

  // 顺序严格对应模型标注。PnP 显式选 0、1、2、3、5，避免把不稳定 kpt4 混入。
  const std::vector<cv::Point3f> OBJECT_POINTS = {
    cv::Point3f(0, 0, 0.827),       // kpt0: 离 R 最远的 target 角点
    cv::Point3f(0, 0.127, 0.700),   // kpt1: 从 kpt0 起逆时针下一角点
    cv::Point3f(0, 0, 0.573),       // kpt2: target 内侧角点
    cv::Point3f(0, -0.127, 0.700),  // kpt3: target 角点
    cv::Point3f(0, 0, 0),           // kpt4: R 中心，仅可视化
    cv::Point3f(0, 0, 0.344)};      // kpt5: 流水灯中心，参与 PnP
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__SOLVER_HPP
