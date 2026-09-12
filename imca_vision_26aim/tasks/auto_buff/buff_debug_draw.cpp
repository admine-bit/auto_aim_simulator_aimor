#include "buff_debug_draw.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{
void draw_whole_buff(
  cv::Mat & img, const Solver & solver, const Target & target, const cv::Scalar & arm_color,
  const cv::Scalar & active_color, int thickness, const std::string & label)
{
  const Eigen::Vector3d r_world = target.point_buff2world(Eigen::Vector3d::Zero());
  if (!r_world.allFinite()) return;
  const Eigen::VectorXd state = target.ekf_x();
  if (state.size() <= 5 || !state.allFinite()) return;

  const double plane_yaw = state[4];
  // state[5] 即 compose_common_state() 里的 current_roll()（含 roll 覆写）。
  // 拿它当骨架基准，骨架第 0 条臂必然与当前 target 臂重合，两层不会整体错开。
  const double base_roll = state[5];
  const auto active_slot = target.target_slot();

  std::optional<cv::Point2f> r_pixel;
  auto draw_arm = [&](double roll, const cv::Scalar & color, int width, const std::string & tag,
                      const std::string & sub_tag) {
    const auto pts = solver.reproject_buff(r_world, plane_yaw, roll);
    if (pts.size() < 6) return;
    // OBJECT_POINTS 顺序：kpt0~3 是 target 四角（0→1→2→3 成菱形），kpt4 是 R，kpt5 是流水灯。
    tools::draw_points(img, std::vector<cv::Point2f>(pts.begin(), pts.begin() + 4), color, width);
    // 臂脊 R→流水灯→target 中心，让五边形骨架连起来。
    const cv::Point2f target_center = (pts[0] + pts[1] + pts[2] + pts[3]) * 0.25F;
    cv::line(img, pts[4], pts[5], color, width, cv::LINE_AA);
    cv::line(img, pts[5], target_center, color, width, cv::LINE_AA);
    if (!tag.empty())
      cv::putText(
        img, tag, target_center + cv::Point2f(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        cv::LINE_AA);
    if (!sub_tag.empty())
      cv::putText(
        img, sub_tag, target_center + cv::Point2f(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        cv::LINE_AA);
    // R 位于符坐标系原点，与 roll 无关：任取一条臂的 kpt4 即可。
    if (!r_pixel.has_value()) r_pixel = pts[4];
  };

  // slot_offset 是按 72° 整数倍学的，已学臂必然落在某条骨架臂上。
  // 被认领的骨架臂不再单独画，避免 ? 和槽号叠字。
  // 容差只兜浮点与 clone_with_roll 的 roll 覆写，不是"差不多就算"。
  constexpr double kOnGridTolRad = 5.0 / 57.3;
  std::array<bool, PowerRune::SLOT_COUNT> claimed{};
  for (std::size_t slot = 0; slot < PowerRune::SLOT_COUNT; ++slot) {
    const auto roll = target.roll_of_slot(slot);
    if (!roll.has_value()) continue;
    for (std::size_t k = 0; k < PowerRune::SLOT_COUNT; ++k) {
      const double off = tools::limit_rad(*roll - base_roll - k * THETA);
      if (std::abs(off) < kOnGridTolRad) claimed[k] = true;
    }
  }

  // 图层 1｜骨架：暗、细、标 ?，含义是"这条臂一定在，但模型还不知道它是几号槽"。
  const cv::Scalar skeleton_color(arm_color[0] * 0.5, arm_color[1] * 0.5, arm_color[2] * 0.5);
  for (std::size_t k = 0; k < PowerRune::SLOT_COUNT; ++k) {
    if (claimed[k]) continue;
    draw_arm(tools::limit_rad(base_roll + k * THETA), skeleton_color, 1, "?", "");
  }

  // 图层 2｜已学槽：带槽号，当前 target 槽最粗最亮。
  for (std::size_t slot = 0; slot < PowerRune::SLOT_COUNT; ++slot) {
    const auto roll = target.roll_of_slot(slot);
    if (!roll.has_value()) continue;
    const bool active = active_slot.has_value() && *active_slot == slot;
    draw_arm(
      *roll, active ? active_color : arm_color, active ? thickness : std::max(1, thickness / 3),
      std::to_string(slot), active ? label : "");
  }

  if (r_pixel.has_value()) tools::draw_point(img, *r_pixel, active_color, 6);
}

void draw_r_gate(cv::Mat & img, const PowerRune & power_rune)
{
  if (img.empty()) return;

  const char * source_name = "None";
  cv::Scalar source_color(128, 128, 128);
  switch (power_rune.r_center_source) {
    case RCenterSource::RmCore:
      source_name = "RmCore";
      source_color = cv::Scalar(0, 255, 0);
      break;
    case RCenterSource::RmCoreUngated:
      source_name = "RmCoreUngated";
      source_color = cv::Scalar(0, 255, 255);
      break;
    case RCenterSource::RmCoreEscaped:
      source_name = "RmCoreEscaped";
      source_color = cv::Scalar(0, 128, 255);
      break;
    case RCenterSource::FusedKpt4:
      source_name = "FusedKpt4";
      source_color = cv::Scalar(255, 0, 255);
      break;
    case RCenterSource::None:
      break;
  }

  // 门限圆：只有真正做过门控的帧才有有限的 threshold。圆心就是融合 kpt4，
  // 因为被拦时 PnP 用的正是圆心那个点。
  const bool gated = std::isfinite(power_rune.r_gate_threshold) &&
                     std::isfinite(power_rune.r_gate_distance) &&
                     std::isfinite(power_rune.r_kpt4_fused_center.x) &&
                     std::isfinite(power_rune.r_kpt4_fused_center.y);
  if (gated) {
    // 圆心恒为融合 kpt4，与本帧采纳了谁无关：这样"R 在圆内/圆外"始终可读。
    cv::circle(
      img, power_rune.r_kpt4_fused_center, static_cast<int>(power_rune.r_gate_threshold),
      cv::Scalar(255, 0, 255), 1);
    tools::draw_point(img, power_rune.r_kpt4_fused_center, cv::Scalar(255, 0, 255), 3);
  }

  if (std::isfinite(power_rune.r_center_raw.x) && std::isfinite(power_rune.r_center_raw.y)) {
    tools::draw_point(img, power_rune.r_center_raw, source_color, 5);
  }

  std::string label = std::string("R:") + source_name;
  if (gated) {
    label += cv::format(
      " d=%.0f/%.0f", static_cast<double>(power_rune.r_gate_distance),
      static_cast<double>(power_rune.r_gate_threshold));
  }
  label += cv::format(" k4n=%d", power_rune.r_kpt4_fused_count);
  cv::putText(
    img, label, cv::Point(10, img.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, source_color, 1);
}

}  // namespace auto_buff
