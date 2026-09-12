#include "buff_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>

#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{
constexpr std::array<std::size_t, 5> kPnpIndices{0, 1, 2, 3, 5};
constexpr std::array<std::size_t, 4> kCornerIndices{0, 1, 2, 3};
constexpr double kTargetCenterRadius = 0.700;
// IPPE 镜像双解的等价窗口。五点又细又长（径向 0.83m、横向 ±0.127m），真实关键点
// 噪声下两支解的 RMSE 差常在亚像素量级（实测 pnp_rmse 0.375），这个窗口取多大都
// 无法靠 RMSE 选边——真正的判据是下面的 plane pitch 先验和时序法向。
constexpr double kPoseTiePixels = 0.5;
// kpt5 残差绝对下限与相对四角误差的倍数门限：两者同时超限才认定流水灯飞点。
constexpr double kFlowErrorMinPx = 5.0;
constexpr double kFlowCornerRatio = 2.5;
// 相邻扇叶的固定相位间隔。
constexpr double kSlotAngle = 2.0 * CV_PI / 5.0;
// 点加权用重复计入实现。rm-core R 是独立轮廓观测，但不应比任一角点更强。
constexpr int kCornerWeight = 2;
constexpr int kFlowWeight = 1;
constexpr int kRCenterWeight = kCornerWeight;

// from_slot 到 to_slot 的槽位差，折到 [-2, +2]；乘以 kSlotAngle 即两片的 roll 差。
// 正五边形只有五个槽，顺时针走 3 步等于逆时针走 2 步，不折叠就会把副片摆到 144° 之外。
// 整符点集拼装（solve_group_pose）和副视图 roll 推导（solve_dual）必须用同一套折叠，
// 否则两处对"同一对槽"给出的角差会差 360°，故只留这一处实现。
int signed_slot_steps(std::size_t from_slot, std::size_t to_slot)
{
  const int forward =
    static_cast<int>((to_slot + PowerRune::SLOT_COUNT - from_slot) % PowerRune::SLOT_COUNT);
  return forward <= static_cast<int>(PowerRune::SLOT_COUNT / 2)
           ? forward
           : forward - static_cast<int>(PowerRune::SLOT_COUNT);
}

struct PoseSolution
{
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  cv::Matx33d rotation = cv::Matx33d::eye();
  std::vector<cv::Point2f> reprojected_points;
  double reprojection_rmse{std::numeric_limits<double>::infinity()};
  double reprojection_max_error{std::numeric_limits<double>::infinity()};
  // kpt0~3 的 RMSE 与 kpt5 的单点误差；四角回退解的 flow_error 为无穷。
  double corner_rmse{std::numeric_limits<double>::infinity()};
  double flow_error{std::numeric_limits<double>::infinity()};
  bool used_fallback{false};
  bool used_rm_center{false};
  bool rm_center_fallback{false};
  // 世界系符面 pitch（rad）。符竖直安装 ⇒ 真解 ≈0，镜像解 ≈±2×视线仰角。
  double plane_pitch_world{std::numeric_limits<double>::quiet_NaN()};
};

bool finite_point(const cv::Point2f & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool finite_vec(const cv::Vec3d & value)
{
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

cv::Vec3d mat_to_vec3d(const cv::Mat & value)
{
  cv::Mat flat = value.reshape(1, 1);
  cv::Mat converted;
  flat.convertTo(converted, CV_64F);
  const auto * data = converted.ptr<double>();
  return {data[0], data[1], data[2]};
}

cv::Matx33d rvec_to_rotation(const cv::Vec3d & rvec)
{
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  cv::Mat converted;
  rotation.convertTo(converted, CV_64F);
  return cv::Matx33d(
    converted.at<double>(0, 0), converted.at<double>(0, 1), converted.at<double>(0, 2),
    converted.at<double>(1, 0), converted.at<double>(1, 1), converted.at<double>(1, 2),
    converted.at<double>(2, 0), converted.at<double>(2, 1), converted.at<double>(2, 2));
}

cv::Vec3d plane_normal(const cv::Matx33d & rotation)
{
  // Buff 的物理点都位于 X=0 平面，因此旋转矩阵第一列就是符面的相机系法向。
  return {rotation(0, 0), rotation(1, 0), rotation(2, 0)};
}

cv::Vec3d plane_normal_in_world(
  const cv::Matx33d & rotation, const Eigen::Matrix3d & R_camera2world)
{
  const cv::Vec3d normal_camera = plane_normal(rotation);
  const Eigen::Vector3d normal_world =
    R_camera2world * Eigen::Vector3d(normal_camera[0], normal_camera[1], normal_camera[2]);
  return {normal_world[0], normal_world[1], normal_world[2]};
}

double normal_angle(const cv::Vec3d & lhs, const cv::Vec3d & rhs)
{
  const double lhs_norm = cv::norm(lhs);
  const double rhs_norm = cv::norm(rhs);
  if (lhs_norm < 1e-9 || rhs_norm < 1e-9)
    return std::numeric_limits<double>::infinity();
  const double cosine = std::clamp(lhs.dot(rhs) / (lhs_norm * rhs_norm), -1.0, 1.0);
  return std::acos(cosine);
}

bool points_in_front(
  const std::vector<cv::Point3f> & object_points, const cv::Matx33d & rotation,
  const cv::Vec3d & translation)
{
  for (const auto & point : object_points) {
    const double z = rotation(2, 0) * point.x + rotation(2, 1) * point.y +
                     rotation(2, 2) * point.z + translation[2];
    if (!std::isfinite(z) || z <= 0.0) return false;
  }
  return true;
}

std::pair<double, double> reprojection_error(
  const std::vector<cv::Point2f> & observed, const std::vector<cv::Point2f> & projected)
{
  if (observed.size() != projected.size() || observed.empty()) {
    return {std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
  }

  double squared_sum = 0.0;
  double max_error = 0.0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    const double error = cv::norm(observed[i] - projected[i]);
    squared_sum += error * error;
    max_error = std::max(max_error, error);
  }
  return {std::sqrt(squared_sum / static_cast<double>(observed.size())), max_error};
}

// 用给定关键点下标集合执行一次 IPPE，返回全部通过正深度/有限性检查的候选解。
// 五点与四角回退共用此路径，保证质量指标和双解处理逻辑一致。
template <std::size_t N>
std::vector<PoseSolution> collect_pose_candidates(
  const FanBlade & blade, const std::array<std::size_t, N> & indices,
  const std::vector<cv::Point3f> & all_object_points, const cv::Mat & camera_matrix,
  const cv::Mat & distort_coeffs)
{
  std::vector<PoseSolution> candidates;
  if (blade.points.size() <= indices.back() || all_object_points.size() <= indices.back())
    return candidates;

  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> object_points;
  image_points.reserve(indices.size());
  object_points.reserve(indices.size());
  for (const std::size_t index : indices) {
    image_points.push_back(blade.points[index]);
    object_points.push_back(all_object_points[index]);
  }
  if (!std::all_of(image_points.begin(), image_points.end(), finite_point)) return candidates;

  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  try {
    const int solution_count = cv::solvePnPGeneric(
      object_points, image_points, camera_matrix, distort_coeffs, rvecs, tvecs, false,
      cv::SOLVEPNP_IPPE);
    if (solution_count <= 0) return candidates;
  } catch (const cv::Exception & exception) {
    tools::logger()->warn("Buff IPPE exception ({} pts): {}", indices.size(), exception.what());
    return candidates;
  }

  for (std::size_t i = 0; i < rvecs.size() && i < tvecs.size(); ++i) {
    PoseSolution candidate;
    candidate.rvec = mat_to_vec3d(rvecs[i]);
    candidate.tvec = mat_to_vec3d(tvecs[i]);
    if (!finite_vec(candidate.rvec) || !finite_vec(candidate.tvec) || candidate.tvec[2] <= 0.0)
      continue;

    candidate.rotation = rvec_to_rotation(candidate.rvec);
    if (!points_in_front(object_points, candidate.rotation, candidate.tvec)) continue;

    cv::projectPoints(
      object_points, candidate.rvec, candidate.tvec, camera_matrix, distort_coeffs,
      candidate.reprojected_points);
    const auto [rmse, max_error] =
      reprojection_error(image_points, candidate.reprojected_points);
    candidate.reprojection_rmse = rmse;
    candidate.reprojection_max_error = max_error;
    if (!std::isfinite(candidate.reprojection_rmse)) continue;

    // 分项质量：前四个下标固定是 kpt0~3；第五个（若有）是 kpt5。
    double corner_squared_sum = 0.0;
    for (std::size_t j = 0; j < 4 && j < image_points.size(); ++j) {
      const double error = cv::norm(image_points[j] - candidate.reprojected_points[j]);
      corner_squared_sum += error * error;
    }
    candidate.corner_rmse = std::sqrt(corner_squared_sum / 4.0);
    candidate.flow_error =
      image_points.size() > 4
        ? cv::norm(image_points[4] - candidate.reprojected_points[4])
        : std::numeric_limits<double>::infinity();
    candidate.used_fallback = indices.size() == 4;
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

// ============ 整符点集 ============
// 五片扇叶不是五个独立目标，而是同一刚体的五个视图。把当前帧所有可见片的点按各自
// 槽差 k×72° 旋转到同一个符坐标系，与共享 R 一起拼成一个点集只解一次 PnP。
//
// 为什么这样能治镜像：单片五点全挤在一条 0.83m 的细线上（横向仅 ±0.127m），
// IPPE 两支镜像解的重投影几乎完全相同（实测 RMSE 0.375px），RMSE 分不开；
// 两片拼起来后点集横向张开到 0.66m 以上，两支解的重投影差异变得显著。
struct PointSet
{
  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> object_points;
  // 与上面一一对应：该点属于哪一片（0=主片），以及是否是共享 R。
  std::vector<int> owner_blade;
  int blade_count{0};
  bool has_r_center{false};
};

// 物体点绕符坐标系 x 轴（= 符面法向，物体点全部位于 x=0 平面内）旋转 angle。
cv::Point3f rotate_about_normal(const cv::Point3f & point, double angle)
{
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {
    point.x, static_cast<float>(c * point.y - s * point.z),
    static_cast<float>(s * point.y + c * point.z)};
}

// 追加一片扇叶的点（按 slot_shift×72° 旋转）。weight 用重复计入实现。
void append_blade_points(
  PointSet & set, const FanBlade & blade, const std::vector<cv::Point3f> & all_object_points,
  double slot_shift_angle, int blade_id, bool include_flow)
{
  for (const std::size_t index : kPnpIndices) {
    if (index == 5 && !include_flow) continue;
    if (blade.points.size() <= index || all_object_points.size() <= index) continue;
    const cv::Point2f & image_point = blade.points[index];
    if (!finite_point(image_point)) continue;
    const cv::Point3f object_point =
      rotate_about_normal(all_object_points[index], slot_shift_angle);
    const int weight = (index == 5) ? kFlowWeight : kCornerWeight;
    for (int w = 0; w < weight; ++w) {
      set.image_points.push_back(image_point);
      set.object_points.push_back(object_point);
      set.owner_blade.push_back(blade_id);
    }
  }
}

// 从整符点集解 IPPE。返回全部有效候选（共面点集仍有二重解，交给调用方选边）。
std::vector<PoseSolution> solve_point_set(
  const PointSet & set, const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs)
{
  std::vector<PoseSolution> candidates;
  if (set.image_points.size() < 4 || set.image_points.size() != set.object_points.size())
    return candidates;

  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  try {
    const int solution_count = cv::solvePnPGeneric(
      set.object_points, set.image_points, camera_matrix, distort_coeffs, rvecs, tvecs, false,
      cv::SOLVEPNP_IPPE);
    if (solution_count <= 0) return candidates;
  } catch (const cv::Exception & exception) {
    tools::logger()->warn(
      "Buff group IPPE exception ({} pts): {}", set.image_points.size(), exception.what());
    return candidates;
  }

  for (std::size_t i = 0; i < rvecs.size() && i < tvecs.size(); ++i) {
    PoseSolution candidate;
    candidate.rvec = mat_to_vec3d(rvecs[i]);
    candidate.tvec = mat_to_vec3d(tvecs[i]);
    if (!finite_vec(candidate.rvec) || !finite_vec(candidate.tvec) || candidate.tvec[2] <= 0.0)
      continue;
    candidate.rotation = rvec_to_rotation(candidate.rvec);
    if (!points_in_front(set.object_points, candidate.rotation, candidate.tvec)) continue;

    cv::projectPoints(
      set.object_points, candidate.rvec, candidate.tvec, camera_matrix, distort_coeffs,
      candidate.reprojected_points);
    const auto [rmse, max_error] =
      reprojection_error(set.image_points, candidate.reprojected_points);
    candidate.reprojection_rmse = rmse;
    candidate.reprojection_max_error = max_error;
    if (!std::isfinite(candidate.reprojection_rmse)) continue;
    candidate.corner_rmse = rmse;
    candidate.flow_error = std::numeric_limits<double>::infinity();
    candidate.used_fallback = false;
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

// 计算候选解在世界系的符面 pitch。
// R_buff2world 的第一列就是符面法向（物体点全部位于 x=0 平面），
// 而 rotation_matrix(yaw,pitch,roll) 的第一列 = (cy·cp, sy·cp, −sp)
// ⇒ 法向的 z 分量 = −sin(pitch) ⇒ pitch = −asin(n_z)。
double plane_pitch_in_world(
  const cv::Matx33d & rotation, const Eigen::Matrix3d & R_camera2world)
{
  const cv::Vec3d normal = plane_normal_in_world(rotation, R_camera2world);
  const double norm = cv::norm(normal);
  if (norm < 1e-9) return std::numeric_limits<double>::quiet_NaN();
  return -std::asin(std::clamp(normal[2] / norm, -1.0, 1.0));
}

// IPPE 双解选择。
//
// 判据优先级：
//   1) plane pitch 先验——符是竖直安装的，真解符面 pitch ≈ 0；镜像解是把法向绕视线
//      对称翻折得到的，其法向仰角 ≈ −2ε（ε 为视线仰角），对应 pitch ≈ +2ε。
//      我们从地面打高处的符，ε 恒 >0（实测约 11°），所以两支解的 |pitch| 分离约 22°。
//      这个判据是单帧的、不依赖任何历史，因此没有"一次坏初始化被永久锁死"的风险。
//   2) 两支 pitch 接近（分离度不足，例如与符等高的退化工况）时，才用时序法向连续性。
//   3) 都没有时退回最小 RMSE——但要清楚 RMSE 对镜像几乎没有区分度
//      （实测两支差远小于 kPoseTiePixels），这只是最后的兜底。
// 本帧双解的诊断快照。镜像判据能否成立，全靠回放里看这几个数。
struct BranchDiagnostics
{
  double pitch_a_deg{0.0};
  double pitch_b_deg{0.0};
  double rmse_gap{0.0};
  int candidate_count{0};
};

std::optional<PoseSolution> select_pose_candidate(
  std::vector<PoseSolution> & candidates,
  const std::optional<cv::Vec3d> & previous_normal_world,
  const Eigen::Matrix3d & R_camera2world, double max_plane_pitch_rad,
  double pitch_separation_rad, BranchDiagnostics * diagnostics = nullptr)
{
  if (candidates.empty()) return std::nullopt;

  for (auto & candidate : candidates)
    candidate.plane_pitch_world = plane_pitch_in_world(candidate.rotation, R_camera2world);

  if (diagnostics != nullptr) {
    diagnostics->candidate_count = static_cast<int>(candidates.size());
    diagnostics->pitch_a_deg = candidates[0].plane_pitch_world * 57.3;
    if (candidates.size() > 1) {
      diagnostics->pitch_b_deg = candidates[1].plane_pitch_world * 57.3;
      diagnostics->rmse_gap =
        std::abs(candidates[0].reprojection_rmse - candidates[1].reprojection_rmse);
    } else {
      diagnostics->pitch_b_deg = diagnostics->pitch_a_deg;
      diagnostics->rmse_gap = 0.0;
    }
  }

  const auto min_rmse_it = std::min_element(
    candidates.begin(), candidates.end(), [](const PoseSolution & lhs, const PoseSolution & rhs) {
      return lhs.reprojection_rmse < rhs.reprojection_rmse;
    });
  if (candidates.size() == 1) return *min_rmse_it;

  const double min_rmse = min_rmse_it->reprojection_rmse;

  // 只在重投影近似等价的候选里选边；明显更差的解直接排除。
  std::vector<const PoseSolution *> tied;
  for (const auto & candidate : candidates)
    if (candidate.reprojection_rmse <= min_rmse + kPoseTiePixels) tied.push_back(&candidate);
  if (tied.empty()) return *min_rmse_it;
  if (tied.size() == 1) return *tied.front();

  // —— 判据 1：plane pitch 先验 ——
  const PoseSolution * flattest = nullptr;
  double flattest_pitch = std::numeric_limits<double>::infinity();
  double second_pitch = std::numeric_limits<double>::infinity();
  for (const auto * candidate : tied) {
    if (!std::isfinite(candidate->plane_pitch_world)) continue;
    const double magnitude = std::abs(candidate->plane_pitch_world);
    if (magnitude < flattest_pitch) {
      second_pitch = flattest_pitch;
      flattest_pitch = magnitude;
      flattest = candidate;
    } else if (magnitude < second_pitch) {
      second_pitch = magnitude;
    }
  }
  // 要求最平的那支确实"够平"，且与次平的那支拉开足够距离，先验才算有效。
  if (
    flattest != nullptr && flattest_pitch < max_plane_pitch_rad &&
    std::isfinite(second_pitch) && second_pitch - flattest_pitch > pitch_separation_rad)
    return *flattest;

  // —— 判据 2：时序法向连续性 ——
  if (previous_normal_world.has_value()) {
    const PoseSolution * best = nullptr;
    double best_normal_angle = std::numeric_limits<double>::infinity();
    for (const auto * candidate : tied) {
      const double angle = normal_angle(
        plane_normal_in_world(candidate->rotation, R_camera2world), *previous_normal_world);
      if (best == nullptr || angle < best_normal_angle) {
        best = candidate;
        best_normal_angle = angle;
      }
    }
    if (best != nullptr) return *best;
  }

  // —— 判据 3：兜底 ——
  return *min_rmse_it;
}

// 小符新路径：只取当前 target 的 YOLO kpt0~3，加一次共享 R（按角点同权重复）。
// 不读取 kpt5，也不把同帧其他扇叶的角点混入小符 PnP。
// 这里的 R 取自 r_center_raw，来源可能是 rm-core 轮廓或融合 kpt4（见 apply_r_center）。
std::optional<PoseSolution> solve_pose_with_rm_center(
  const PowerRune & power_rune, const std::vector<cv::Point3f> & all_object_points,
  const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs,
  const std::optional<cv::Vec3d> & previous_normal_world,
  const Eigen::Matrix3d & R_camera2world, double max_plane_pitch_rad,
  double pitch_separation_rad, BranchDiagnostics * diagnostics = nullptr)
{
  const auto & blade = power_rune.target();
  if (
    blade.points.size() < 4 || all_object_points.size() <= 4 ||
    !power_rune.r_center_observed || !finite_point(power_rune.r_center_raw))
    return std::nullopt;

  PointSet set;
  append_blade_points(set, blade, all_object_points, 0.0, 0, false);
  for (int w = 0; w < kRCenterWeight; ++w) {
    set.image_points.push_back(power_rune.r_center_raw);
    set.object_points.push_back(all_object_points[4]);
    set.owner_blade.push_back(-1);
  }
  set.blade_count = 1;
  set.has_r_center = true;

  auto candidates = solve_point_set(set, camera_matrix, distort_coeffs);
  auto solution = select_pose_candidate(
    candidates, previous_normal_world, R_camera2world, max_plane_pitch_rad,
    pitch_separation_rad, diagnostics);
  if (solution.has_value()) solution->used_rm_center = true;
  return solution;
}

std::optional<PoseSolution> solve_pose_points(
  const PowerRune & power_rune, const std::vector<cv::Point3f> & all_object_points,
  const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs,
  const std::optional<cv::Vec3d> & previous_normal_world,
  const Eigen::Matrix3d & R_camera2world, const double max_rmse_px,
  const double max_plane_pitch_rad, const double pitch_separation_rad,
  BranchDiagnostics * diagnostics = nullptr)
{
  const auto & blade = power_rune.target();

  // 正常路径：五点 IPPE（kpt0~3 + kpt5）。这条单片路径仍不使用 kpt4——
  // kpt4 的新用途（门控参照 + 融合回退）只在 Detector 侧决定 r_center_raw 时生效，
  // 不改变本函数的点集含义。
  auto five_point_candidates = collect_pose_candidates(
    blade, kPnpIndices, all_object_points, camera_matrix, distort_coeffs);
  auto best = select_pose_candidate(
    five_point_candidates, previous_normal_world, R_camera2world, max_plane_pitch_rad,
    pitch_separation_rad, diagnostics);

  // 异常路径：kpt0~3 稳定但 kpt5 残差显著异常（飞点）时，仅当前帧回退四角 IPPE。
  // 回退不改变模型含义，也绝不引入 kpt4。
  const bool flow_outlier =
    best.has_value() && best->reprojection_rmse > max_rmse_px &&
    best->flow_error > std::max(kFlowErrorMinPx, kFlowCornerRatio * best->corner_rmse);
  if (!best.has_value() || flow_outlier) {
    auto corner_candidates = collect_pose_candidates(
      blade, kCornerIndices, all_object_points, camera_matrix, distort_coeffs);
    auto fallback = select_pose_candidate(
      corner_candidates, previous_normal_world, R_camera2world, max_plane_pitch_rad,
      pitch_separation_rad, diagnostics);
    if (fallback.has_value()) {
      if (flow_outlier)
        tools::logger()->debug(
          "Buff kpt5 outlier (flow_error {:.1f}px, corner {:.1f}px), four-corner fallback",
          best->flow_error, best->corner_rmse);
      best = std::move(fallback);
    }
  }
  return best;
}

// 整符联合解算：把所有可见片的点旋转到同一符坐标系，与共享 R 拼成一个点集解一次。
// 返回的位姿是"符坐标系（原点在 R，roll 以 primary 片为零点）到相机"。
//
// slot_sign：图像圆周角递增方向对应符坐标系 roll 的正负。物体点绕法向旋转 +72° 时
// 投影到图像上是顺时针还是逆时针，取决于关键点标注手性与相机朝向，读码断言不了。
// 这里两个符号都试、取 RMSE 明显更低的一个——符号错会把副片点放到 144° 之外，
// RMSE 差距是数量级的，不存在歧义。
//
std::optional<PoseSolution> solve_group_pose(
  const PowerRune & power_rune, const std::vector<cv::Point3f> & all_object_points,
  const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs,
  const std::optional<cv::Vec3d> & previous_normal_world,
  const Eigen::Matrix3d & R_camera2world, double max_plane_pitch_rad,
  double pitch_separation_rad, int & used_blade_count, int & used_slot_sign,
  BranchDiagnostics * diagnostics = nullptr)
{
  used_blade_count = 0;
  used_slot_sign = 0;

  const std::size_t primary_slot = power_rune.target_slot();
  if (primary_slot >= PowerRune::SLOT_COUNT) return std::nullopt;

  // 收集本帧真正被观测到的片（推断视图没有当帧二维点，绝不参与）。
  std::vector<std::size_t> visible;
  for (std::size_t slot = 0; slot < PowerRune::SLOT_COUNT; ++slot) {
    if (!power_rune.slot_observed(slot)) continue;
    if (power_rune.fanblades[slot].points.size() <= 5) continue;
    visible.push_back(slot);
  }
  if (visible.empty()) return std::nullopt;

  const bool use_r_center =
    power_rune.r_center_observed && finite_point(power_rune.r_center_raw);

  auto build = [&](int sign) {
    PointSet set;
    for (const std::size_t slot : visible) {
      const int steps = signed_slot_steps(primary_slot, slot);
      // 有可信 R 时彻底排除存在逐扇叶偏差的 kpt5；无 R 时才走旧联合 PnP 回退。
      // 两片时两条路径均为 18 点：16 个角点重复点 + 2 个 R 或 kpt5 点。
      //
      // 注意：Detector 侧的 apply_r_center 现在会用融合 kpt4 兜住 rm R 缺失的帧，
      // 所以 use_r_center 长期恒为 true，下面这个 include_flow 分支基本不再进入。
      // 保留它只是为了 rm_center_enabled=false（纯 YOLO 旧路径）以及两个来源都失效的
      // 极端帧。这正是要的效果：逐帧在"含 R"和"含 kpt5"两种点集间切换等于逐帧换标定
      // 模型，tvec 会跟着阶跃，是 pitch/yaw 震荡的直接来源。
      append_blade_points(
        set, power_rune.fanblades[slot], all_object_points, sign * steps * kSlotAngle,
        static_cast<int>(slot), !use_r_center);
    }
    if (use_r_center) {
      // rm-core 实测 R 对应符坐标系原点，与 roll 无关，整组点集中只加入一次。
      for (int w = 0; w < kRCenterWeight; ++w) {
        set.image_points.push_back(power_rune.r_center_raw);
        set.object_points.push_back(all_object_points[4]);
        set.owner_blade.push_back(-1);
      }
      set.has_r_center = true;
    }
    set.blade_count = static_cast<int>(visible.size());
    return set;
  };

  std::optional<PoseSolution> best;
  int best_sign = 0;
  for (const int sign : {+1, -1}) {
    PointSet set = build(sign);
    auto candidates = solve_point_set(set, camera_matrix, distort_coeffs);
    BranchDiagnostics local{};
    auto picked = select_pose_candidate(
      candidates, previous_normal_world, R_camera2world, max_plane_pitch_rad,
      pitch_separation_rad, &local);
    if (!picked.has_value()) continue;
    if (!best.has_value() || picked->reprojection_rmse < best->reprojection_rmse) {
      best = std::move(picked);
      best_sign = sign;
      if (diagnostics != nullptr) *diagnostics = local;
    }
    // 只有一片可见时两个符号完全等价，不必试第二个。
    if (visible.size() == 1) break;
  }

  if (!best.has_value()) return std::nullopt;
  best->used_rm_center = use_r_center;
  // 降级 = 没有 R（走 kpt5）**或** R 来自融合 kpt4。后者精度低于 rm-core 轮廓 R，
  // 若不算降级，Target 的 good_quality 门会被本次改动顺带放松（隐藏改动）。
  best->rm_center_fallback =
    power_rune.rm_center_enabled &&
    (!use_r_center || power_rune.r_center_source == RCenterSource::FusedKpt4);
  used_blade_count = static_cast<int>(visible.size());
  // 只有一片可见时两个符号解出的点集完全相同（steps 恒为 0），此时符号并没有被标定，
  // 报 0 让调用方知道"这一帧学不到符号"，而不是把默认值当成标定结果喂给相位模型。
  used_slot_sign = visible.size() >= 2 ? best_sign : 0;
  return best;
}

void apply_pose(
  PowerRune & power_rune, const PoseSolution & solution,
  const Eigen::Matrix3d & R_camera2gimbal, const Eigen::Vector3d & t_camera2gimbal,
  const Eigen::Matrix3d & R_gimbal2world)
{
  Eigen::Matrix3d R_buff2camera;
  cv::Mat rotation_cv(solution.rotation);
  cv::cv2eigen(rotation_cv, R_buff2camera);
  const Eigen::Vector3d t_buff2camera{
    solution.tvec[0], solution.tvec[1], solution.tvec[2]};

  // OBJECT_POINTS 全部以 R 为坐标原点，因此五点 PnP 的 tvec 本身就是 R 在相机系的位置。
  // 这里没有使用 Detector 的二维几何 R（shared_r_center，那个只喂槽位关联）。
  // r_center_raw 可能来自 rm-core 轮廓或融合 kpt4，选择在 Detector::apply_r_center 完成。
  const Eigen::Matrix3d R_buff2gimbal = R_camera2gimbal * R_buff2camera;
  const Eigen::Vector3d r_center_in_gimbal =
    R_camera2gimbal * t_buff2camera + t_camera2gimbal;
  const Eigen::Matrix3d R_buff2world = R_gimbal2world * R_buff2gimbal;

  power_rune.xyz_in_world = R_gimbal2world * r_center_in_gimbal;
  power_rune.ypd_in_world = tools::xyz2ypd(power_rune.xyz_in_world);
  power_rune.ypr_in_world = tools::eulers(R_buff2world, 2, 1, 0);

  // hero 的 Target/EKF 状态明确假设能量机关平面在世界系 pitch=0。
  // 因此 z2 的 target 中心也必须由同一模型生成，不能把完整 PnP pitch 下的中心
  // 与 EKF 的 pitch=0 预测混在一起，否则两组测量会持续互相拉扯。
  const Eigen::Matrix3d R_buff_model = tools::rotation_matrix(Eigen::Vector3d(
    power_rune.ypr_in_world[0], 0.0, power_rune.ypr_in_world[2]));
  power_rune.blade_xyz_in_world =
    power_rune.xyz_in_world +
    R_buff_model * Eigen::Vector3d(0.0, 0.0, kTargetCenterRadius);
  power_rune.blade_ypd_in_world = tools::xyz2ypd(power_rune.blade_xyz_in_world);

  auto & blade = power_rune.target();
  blade.raw_reprojected_points = solution.reprojected_points;
  blade.raw_reprojection_rmse = solution.reprojection_rmse;
  blade.raw_reprojection_max_error = solution.reprojection_max_error;
  blade.raw_corner_rmse = solution.corner_rmse;
  blade.raw_flow_error = solution.flow_error;
  blade.raw_used_rm_center = solution.used_rm_center;
  blade.raw_rm_center_fallback = solution.rm_center_fallback;
  blade.raw_pnp_fallback = solution.used_fallback;
}
}  // namespace

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ =
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ =
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  // 可选配置：五点 PnP 质量门限，旧 yaml 未写时使用默认值。
  if (yaml["buff_pnp_max_rmse_px"]) {
    pnp_max_rmse_px_ = yaml["buff_pnp_max_rmse_px"].as<double>();
    if (!std::isfinite(pnp_max_rmse_px_) || pnp_max_rmse_px_ <= 0.0)
      pnp_max_rmse_px_ = 4.0;
  }
  // 镜像消歧的 pitch 先验门限。默认按"符竖直、视线仰角约 10°"给：
  // 真解 |pitch| 应远小于 20°，镜像解约 2×仰角 ≈ 22°，分离阈值 8° 留足噪声余量。
  if (yaml["buff_max_plane_pitch_deg"]) {
    const double value = yaml["buff_max_plane_pitch_deg"].as<double>();
    if (std::isfinite(value) && value > 0.0) max_plane_pitch_rad_ = value / 57.3;
  }
  if (yaml["buff_pitch_separation_deg"]) {
    const double value = yaml["buff_pitch_separation_deg"].as<double>();
    if (std::isfinite(value) && value > 0.0) pitch_separation_rad_ = value / 57.3;
  }
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

// 丢失时的锚点寿命管理。
//
// 旧实现是连续 7 帧（60fps 下才 117ms）就清空 last_primary_normal_world_，
// 而清空后 select_pose_candidate 只能退回按 RMSE 选边——实测两支解的 RMSE 差
// 远小于容差窗，等于抛硬币。符螺栓固定在场地、打符的车不平移，所以世界系法向
// 在一局内是常数，117ms 的丢失完全没有理由清它。
//
// 但也不能永不清空：模式切换后可能看到对方的符（规则 2681，两侧背对背、法向相反），
// 那时旧锚点是有害的。reset() 覆盖了模式切换；这里再留一道很长的兜底，
// 长到远超规则允许的 200ms 换组黑屏，只在真正长时间失去目标后才生效。
void Solver::note_miss() const
{
  if (++consecutive_misses_ > kAnchorMaxMisses) last_primary_normal_world_.reset();
}

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  const Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::reset()
{
  last_primary_normal_world_.reset();
  consecutive_misses_ = 0;
  last_normal_step_rad_ = 0.0;
  last_pose_pitch_a_ = 0.0;
  last_pose_pitch_b_ = 0.0;
  last_rmse_gap_ = 0.0;
  last_blade_count_ = 0;
  last_slot_sign_ = 0;
  calibrated_slot_sign_ = 0;
}

void Solver::solve(std::optional<PowerRune> & power_rune) const
{
  if (!power_rune.has_value()) {
    note_miss();
    return;
  }

  const Eigen::Matrix3d R_camera2world = R_gimbal2world_ * R_camera2gimbal_;
  // 诊断需要“上一帧”的法向，先于本帧覆盖前留快照。
  const std::optional<cv::Vec3d> previous_normal_world = last_primary_normal_world_;

  // 小符只使用当前 target：有真实 rm-core R 时为 kpt0~3+R；无 R 时完整复用
  // kpt0~3+kpt5 及其飞点后的四角回退。即使 YOLO 同帧看到多片也不做联合解算。
  int blade_count = 0;
  BranchDiagnostics diagnostics{};
  std::optional<PoseSolution> solution;
  const bool use_rm_center =
    power_rune->r_center_observed && finite_point(power_rune->r_center_raw);
  if (use_rm_center) {
    solution = solve_pose_with_rm_center(
      power_rune.value(), OBJECT_POINTS, camera_matrix_, distort_coeffs_,
      previous_normal_world, R_camera2world, max_plane_pitch_rad_, pitch_separation_rad_,
      &diagnostics);
  } else {
    solution = solve_pose_points(
      power_rune.value(), OBJECT_POINTS, camera_matrix_, distort_coeffs_,
      previous_normal_world, R_camera2world, pnp_max_rmse_px_, max_plane_pitch_rad_,
      pitch_separation_rad_, &diagnostics);
    if (solution.has_value())
      solution->rm_center_fallback = power_rune->rm_center_enabled;
  }
  // rm_center_fallback 的语义是"本帧 R 质量降级"，Target 用它卡 good_quality。
  // 融合 kpt4 虽然填上了 R 槽位（因此 use_rm_center 为真、不走 kpt5），精度仍低于
  // rm-core 轮廓 R，必须照样算降级——否则本次改动会顺带放松 EKF 的质量门，
  // 变成一个没人要求的隐藏改动。RmCoreEscaped 用的仍是 rm-core R，不算降级。
  if (solution.has_value() && power_rune->r_center_source == RCenterSource::FusedKpt4)
    solution->rm_center_fallback = power_rune->rm_center_enabled;
  blade_count = solution.has_value() ? 1 : 0;
  if (!solution.has_value()) {
    tools::logger()->warn("Buff IPPE failed or returned an invalid pose");
    power_rune.reset();
    note_miss();
    return;
  }

  apply_pose(
    power_rune.value(), solution.value(), R_camera2gimbal_, t_camera2gimbal_,
    R_gimbal2world_);
  last_primary_normal_world_ =
    plane_normal_in_world(solution->rotation, R_camera2world);
  last_normal_step_rad_ =
    previous_normal_world.has_value()
      ? normal_angle(*last_primary_normal_world_, *previous_normal_world)
      : 0.0;
  last_blade_count_ = blade_count;
  last_slot_sign_ = calibrated_slot_sign_;
  last_pose_pitch_a_ = diagnostics.pitch_a_deg;
  last_pose_pitch_b_ = diagnostics.pitch_b_deg;
  last_rmse_gap_ = diagnostics.rmse_gap;
  consecutive_misses_ = 0;
}

void Solver::solve_dual(
  std::optional<PowerRune> & primary, std::optional<PowerRune> & secondary) const
{
  if (!primary.has_value()) {
    secondary.reset();
    note_miss();
    return;
  }

  const Eigen::Matrix3d R_camera2world = R_gimbal2world_ * R_camera2gimbal_;
  // 选边需要“上一帧”的符面法向，先于本帧覆盖前留快照。
  const std::optional<cv::Vec3d> previous_normal_world = last_primary_normal_world_;

  // ============ 整符一次解算 ============
  // 主副不是两个独立目标，而是同一刚体的两个视图。旧实现给两者各解一次 PnP 再在
  // 位姿层加权融合——两支各自都可能是镜像解，融合两个可能翻的解没有意义，
  // 还要靠 kSharedPoseMaxRDistance 事后仲裁。现在把两片的点旋转到同一符坐标系拼成
  // 一个点集只解一次：同帧只有一个位姿，没有第二个可以矛盾，仲裁分支随之作废。
  int blade_count = 0;
  int slot_sign = 0;
  BranchDiagnostics diagnostics{};
  auto solution = solve_group_pose(
    primary.value(), OBJECT_POINTS, camera_matrix_, distort_coeffs_, previous_normal_world,
    R_camera2world, max_plane_pitch_rad_, pitch_separation_rad_, blade_count, slot_sign,
    &diagnostics);
  if (!solution.has_value()) {
    solution = solve_pose_points(
      primary.value(), OBJECT_POINTS, camera_matrix_, distort_coeffs_, previous_normal_world,
      R_camera2world, pnp_max_rmse_px_, max_plane_pitch_rad_, pitch_separation_rad_,
      &diagnostics);
    if (solution.has_value()) {
      blade_count = 1;
      slot_sign = 0;
    }
  }
  // 同上：没用 R，或用的是融合 kpt4，都算 R 质量降级。
  if (
    solution.has_value() && primary->rm_center_enabled &&
    (!solution->used_rm_center || primary->r_center_source == RCenterSource::FusedKpt4))
    solution->rm_center_fallback = true;
  if (!solution.has_value()) {
    tools::logger()->warn("Buff group IPPE failed or returned an invalid pose");
    primary.reset();
    secondary.reset();
    note_miss();
    return;
  }

  apply_pose(
    primary.value(), solution.value(), R_camera2gimbal_, t_camera2gimbal_, R_gimbal2world_);
  last_primary_normal_world_ = plane_normal_in_world(solution->rotation, R_camera2world);
  last_normal_step_rad_ =
    previous_normal_world.has_value()
      ? normal_angle(*last_primary_normal_world_, *previous_normal_world)
      : 0.0;
  last_blade_count_ = blade_count;
  if (slot_sign != 0) calibrated_slot_sign_ = slot_sign;
  last_slot_sign_ = calibrated_slot_sign_;
  last_pose_pitch_a_ = diagnostics.pitch_a_deg;
  last_pose_pitch_b_ = diagnostics.pitch_b_deg;
  last_rmse_gap_ = diagnostics.rmse_gap;
  consecutive_misses_ = 0;

  if (!secondary.has_value()) return;

  // ============ 副视图由同一刚体推导 ============
  // 位置、符面姿态与主视图完全相同（本就是同一个位姿）；roll 差严格锁到 72° 整数倍。
  // 符号由整符解算自标定得到（slot_sign），不再依赖 phase_roll_offset_——
  // 那个量是从可能翻转的独立 PnP 学来的，本身就被镜像污染过。
  const int signed_steps = signed_slot_steps(primary->target_slot(), secondary->target_slot());
  const int sign = calibrated_slot_sign_ != 0 ? calibrated_slot_sign_ : 1;

  secondary->xyz_in_world = primary->xyz_in_world;
  secondary->ypd_in_world = primary->ypd_in_world;
  secondary->ypr_in_world = primary->ypr_in_world;
  secondary->ypr_in_world[2] =
    tools::limit_rad(primary->ypr_in_world[2] + sign * signed_steps * THETA);

  for (PowerRune * view : {&primary.value(), &secondary.value()}) {
    // 瞄准点仍按 Target/EKF 的 pitch=0 模型生成，与 point_buff2world 保持全链路一致。
    const Eigen::Matrix3d view_rotation = tools::rotation_matrix(
      Eigen::Vector3d(view->ypr_in_world[0], 0.0, view->ypr_in_world[2]));
    view->blade_xyz_in_world =
      view->xyz_in_world + view_rotation * Eigen::Vector3d(0.0, 0.0, kTargetCenterRadius);
    view->blade_ypd_in_world = tools::xyz2ypd(view->blade_xyz_in_world);
  }
}

std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double row) const
{
  // Target/EKF 仍沿用 hero 的“符面 pitch=0”状态模型；这里与其保持完全一致。
  const Eigen::Matrix3d R_buff2world =
    tools::rotation_matrix(Eigen::Vector3d(yaw, 0.0, row));
  const Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  const Eigen::Vector3d t_buff2camera = R_camera2gimbal_.transpose() *
                                       (R_gimbal2world_.transpose() * xyz_in_world -
                                        t_camera2gimbal_);

  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Vec3d rvec;
  cv::Rodrigues(R_buff2camera_cv, rvec);
  const cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(
    OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}
}  // namespace auto_buff
