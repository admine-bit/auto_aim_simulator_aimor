#ifndef AUTO_BUFF__DEBUG_DRAW_HPP
#define AUTO_BUFF__DEBUG_DRAW_HPP

#include <opencv2/opencv.hpp>
#include <string>

#include "buff_solver.hpp"
#include "buff_target.hpp"

namespace auto_buff
{
/// 整符建模可视化：把模型认为的五条臂全部画出来。
///
/// 分两层画，两层回答的问题不同：
///   骨架层：符是刚体正五边形，任意一条臂定了，另外四条就是 ±72° 的整数倍，
///           与"那片扇叶亮没亮过"无关。解算成功的第一帧就能把整符补全，
///           刚看到符时也能立刻判断建模位置对不对。
///   已学层：只有 PhaseModel 真学到 72° 偏移的槽才带槽号。骨架等距外推、
///           永远不会自己重叠，所以"某槽偏移学错"只能靠这层暴露：两个槽号
///           叠在同一条臂上、另一条臂只剩带 ? 的骨架，就是学错了。
///
/// 三个调试程序（auto_buff_debug_mpc / auto_buff_debug_yolo / auto_buff_test）画的是
/// 同一套模型，只有配色和线宽不同，因此实现放在这里一份，避免改判读逻辑时漏改某一处。
/// 只读 Target/Solver，不改变任何状态。
void draw_whole_buff(
  cv::Mat & img, const Solver & solver, const Target & target, const cv::Scalar & arm_color,
  const cv::Scalar & active_color, int thickness, const std::string & label);

/// kpt4 门控可视化：画出融合 kpt4（门控圆心）、门限圆（半径 = r_gate_threshold）、
/// 以及本帧真正送进 PnP 的 R，并标注来源。
///
/// 判读方式：rm-core R 落在圆内 ⇒ 被采纳（r_src=RmCore）；落在圆外 ⇒ 被拦，
/// PnP 用的是圆心那个融合 kpt4（r_src=FusedKpt4）。圆一直很小而 R 反复在圆外，
/// 说明门限过紧；圆大到套住半个符，说明门限过松、拦不住误识别。
/// 只读 PowerRune，不改变任何状态。
void draw_r_gate(cv::Mat & img, const PowerRune & power_rune);

}  // namespace auto_buff

#endif  // AUTO_BUFF__DEBUG_DRAW_HPP
