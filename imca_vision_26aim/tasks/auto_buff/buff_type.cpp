#include "buff_type.hpp"

#include <cmath>
#include <limits>

namespace auto_buff
{
FanBlade::FanBlade(
  const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t)
: center(keypoints_center), type(t), observed(t != _unlight)
{
  points.insert(points.end(), kpt.begin(), kpt.end());
}

FanBlade::FanBlade(FanBlade_type t) : type(t), observed(t != _unlight)
{
  if (t != _unlight) exit(-1);
}

PowerRune::PowerRune(
  std::vector<FanBlade> & ts, const cv::Point2f center, std::optional<PowerRune> last_powerrune,
  std::uint64_t frame_id, std::size_t selected_slot)
: r_center(center), frame_id_(frame_id)
{
  fanblades.resize(SLOT_COUNT);
  selected_slot_ = selected_slot < SLOT_COUNT ? selected_slot : 0;

  // 继承上一帧五槽几何只用于跨帧关联。所有继承槽都先标记 observed=false，
  // 这样没有被本帧 YOLO 重新观测到的旧角点绝不会进入当前帧 PnP。
  if (last_powerrune.has_value() && last_powerrune->fanblades.size() == SLOT_COUNT) {
    fanblades = last_powerrune->fanblades;
    for (std::size_t slot = 0; slot < SLOT_COUNT; ++slot) {
      fanblades[slot].observed = false;
      fanblades[slot].type = _unlight;
      fanblades[slot].slot_index = slot;
      fanblades[slot].active = false;
      fanblades[slot].active_observed = false;
      fanblades[slot].active_confidence = 0.0F;
      // 历史槽只保留关联几何，不能把上一帧网络 R 伪装成本帧弱观测。
      fanblades[slot].raw_r_valid = false;
      fanblades[slot].raw_r_confidence = 0.0F;
      fanblades[slot].raw_reprojected_points.clear();
      fanblades[slot].raw_reprojection_rmse = std::numeric_limits<double>::infinity();
      fanblades[slot].raw_reprojection_max_error = std::numeric_limits<double>::infinity();
      fanblades[slot].raw_corner_rmse = std::numeric_limits<double>::infinity();
      fanblades[slot].raw_flow_error = std::numeric_limits<double>::infinity();
      fanblades[slot].raw_pnp_fallback = false;
    }
  }

  std::size_t next_free_slot = 0;
  for (auto blade : ts) {
    std::size_t slot = blade.slot_index;
    // assign_slots 用 slot_index == SLOT_COUNT 标记"该片与另一片撞到同一格位且置信度更低"，
    // 这种片必须直接丢弃：塞进任意空槽会把一个错误的物理扇叶伪装成那个槽的观测。
    if (slot == SLOT_COUNT) continue;
    // 兼容未预先分配槽号的调用：按空槽依次放入，但仍保持固定五槽结构。
    if (slot > SLOT_COUNT) {
      while (next_free_slot < SLOT_COUNT && fanblades[next_free_slot].observed)
        ++next_free_slot;
      if (next_free_slot >= SLOT_COUNT) break;
      slot = next_free_slot++;
    }

    // 同一帧多个候选落入同一物理槽时，只保留目标框置信度更高的观测。
    if (fanblades[slot].observed &&
        fanblades[slot].object_confidence >= blade.object_confidence)
      continue;

    blade.slot_index = slot;
    blade.observed = true;
    fanblades[slot] = std::move(blade);
  }

  light_num = 0;
  for (std::size_t slot = 0; slot < SLOT_COUNT; ++slot) {
    auto & blade = fanblades[slot];
    blade.slot_index = slot;
    if (blade.observed) {
      ++light_num;
      blade.angle = atan_angle(blade.center);
      blade.type = slot == selected_slot_ ? _target : _light;
    } else {
      blade.type = _unlight;
    }
  }

  unsolvable_ = light_num == 0 || !std::isfinite(r_center.x) || !std::isfinite(r_center.y);
}

PowerRune PowerRune::with_target(std::size_t slot) const
{
  // 主目标与副目标不是两个独立 PowerRune，而是同一个五槽整体的两个槽位视图。
  // 复制后只修改 selected_slot_ 和类型标签，公共 R、frame_id 及其他槽数据均保持一致。
  PowerRune view = *this;
  if (slot >= SLOT_COUNT) {
    view.unsolvable_ = true;
    return view;
  }

  view.selected_slot_ = slot;
  for (std::size_t i = 0; i < view.fanblades.size(); ++i) {
    view.fanblades[i].type = view.fanblades[i].observed
                                 ? (i == slot ? _target : _light)
                                 : _unlight;
  }
  view.unsolvable_ = !view.fanblades[slot].observed;
  return view;
}

PowerRune PowerRune::with_inferred_target(std::size_t slot) const
{
  PowerRune view = with_target(slot);
  if (slot >= SLOT_COUNT) return view;

  // observed=false 明确告诉 Solver：这个槽位没有当前帧二维点，只能从主目标刚体姿态推导。
  // 至少曾经观测过六点才允许建立推断视图，避免把默认空槽伪装成真实副目标。
  const auto & blade = view.fanblades[slot];
  view.unsolvable_ = !(blade.points.size() >= 6 && std::isfinite(blade.center.x) &&
                       std::isfinite(blade.center.y));
  return view;
}

double PowerRune::atan_angle(cv::Point2f point) const
{
  auto v = point - r_center;
  auto angle = std::atan2(v.y, v.x);
  return angle >= 0 ? angle : angle + CV_2PI;
}
}  // namespace auto_buff
