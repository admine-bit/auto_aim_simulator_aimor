#include <fmt/format.h>

#include <cmath>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_debug_draw.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "vc/core/debug_tools.h"

// 定义命令行参数
const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   |configs/sentry.yaml | yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // 关键点尺度诊断要把像素跨度翻成米，需要 fx。Solver 没有暴露内参，直接读同一份 yaml，
  // 不为一个诊断给生产类加接口。读不到就把 d_span_* 置 -1，其余诊断照常输出。
  double fx = 0.0;
  try {
    const auto camera_matrix =
      YAML::LoadFile(config_path)["camera_matrix"].as<std::vector<double>>();
    if (!camera_matrix.empty()) fx = camera_matrix[0];
  } catch (const std::exception & e) {
    tools::logger()->warn("读取 camera_matrix 失败，隐含距离诊断关闭：{}", e.what());
  }

  // 初始化绘图器、录制器、退出器
  tools::Plotter plotter;
  tools::Recorder recorder;
  tools::Exiter exiter;

  // 初始化云台、相机
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 初始化识别器、解算器、瞄准器
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto_buff::Aimer aimer(config_path);

  // 参考 standard_mpc：常驻大小符目标对象，按模式切换使用
  auto_buff::SmallTarget buff_small_target(config_path);
  auto_buff::BigTarget buff_big_target(config_path);
  auto_buff::Target * target = nullptr;
  auto last_buff_mode = io::GimbalMode::IDLE;

  // 关联参考角改用相位模型外推：换组黑屏期间它仍在推进，不会像上一帧像素角那样陈旧。
  auto * active_buff_target = static_cast<auto_buff::Target *>(&buff_small_target);
  detector.set_slot_angle_predictor(
    [&](std::size_t slot) { return active_buff_target->image_angle_of_slot(slot); },
    [&](std::size_t slot, double angle) {
      active_buff_target->calibrate_image_angle(slot, angle);
    });

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  while (!exiter.exit()) {
    camera.read(img, t);
    q = gimbal.q(t);
    auto gs = gimbal.state();
    auto mode = gimbal.mode();
    recorder.record(img, q, t);

    if (last_buff_mode != mode) {
      if (mode == io::GimbalMode::SMALL_BUFF || mode == io::GimbalMode::BIG_BUFF) {
        detector.reset();
        // 新一回合不能沿用上一回合的 IPPE 平面分支。
        solver.reset();
        buff_small_target = auto_buff::SmallTarget(config_path);
        buff_big_target = auto_buff::BigTarget(config_path);
        aimer = auto_buff::Aimer(config_path);
      }
      last_buff_mode = mode;
    }

 
    // -------------- 打符核心逻辑 --------------

    solver.set_R_gimbal2world(q);
    detector.setMyColor(gs.my_color);

    // rm-core 的绘制接口依赖本帧原图缓存；必须在 YOLO 往 img 上画框之前设置。
    // RuneDetector 会在同一次检测中直接绘制到该缓存，不会为可视化重复跑检测。
    if (detector.rm_center_enabled()) DebugTools::get()->setImage(img);

    std::optional<auto_buff::PowerRune> power_runes;
    std::optional<auto_buff::PowerRune> power_runes_next;
    auto_aim::Plan plan{};
    std::unique_ptr<auto_buff::Target> target_clone;

    if (mode == io::GimbalMode::BIG_BUFF) {
      target = &buff_big_target;
      active_buff_target = target;
      // ====== 大符双目标路径 ======
      // detect_dual(): 同时返回主目标(power_runes)和副目标(power_runes_next)
      auto [first, next] = detector.detect_dual(img, q, t);
      power_runes = first;
      power_runes_next = next;

      // 两片的点旋转到同一符坐标系拼成一个点集，只解一次 PnP（整符一个刚体一个位姿）。
      solver.solve_dual(power_runes, power_runes_next);

      // 整符解算自标定出的槽差符号回灌相位模型（槽偏移刚体推导的唯一未知量）。
      if (solver.last_slot_sign() != 0) target->set_slot_sign(solver.last_slot_sign());

      target->get_target(power_runes, t);
      auto * big_target = static_cast<auto_buff::BigTarget *>(target);
      // 副片 roll 也喂相位模型（观测率翻倍 + 72° 自洽性检查）。
      big_target->observe_secondary(power_runes_next);

      // 在副本上做 mpc 预测，避免污染主跟踪状态
      target_clone = std::make_unique<auto_buff::BigTarget>(*big_target);
      plan = aimer.mpc_aim(*target_clone, t, gs, true);

      // 预计算副目标方案
      if (!target->is_unsolve())
        aimer.compute_secondary_plan(big_target, power_runes_next, gs, t);

        
      // 二次门控
      plan.fire = auto_buff::fire_gate_check(plan, *target, gs);
      // 开火状态机发送
      aimer.big_buff_send(plan, gimbal, detector);

    } else {
      target = &buff_small_target;
      active_buff_target = target;
      // ====== 小符 / 其他模式：原逻辑完全不变 ======
      power_runes = detector.detect(img, q, t);

      solver.solve(power_runes);

      if (solver.last_slot_sign() != 0) target->set_slot_sign(solver.last_slot_sign());

      target->get_target(power_runes, t);
      auto * small_target = static_cast<auto_buff::SmallTarget *>(target);

      // 在副本上做 mpc 预测，避免污染主跟踪状态
      target_clone = std::make_unique<auto_buff::SmallTarget>(*small_target);

      plan = aimer.mpc_aim(*target_clone, t, gs, true);

      // plan.pitch = - plan.pitch;
      // plan.pitch_vel = - plan.pitch_vel;
      // plan.pitch_acc = - plan.pitch_acc;
      // 二次门控
      plan.fire = auto_buff::fire_gate_check(plan, *target, gs);


      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);
    }
    // -------------- 调试输出 --------------

    nlohmann::json data;

    // buff原始观测数据
    if (power_runes.has_value()) {
      const auto & p = power_runes.value();
      data["buff_R_yaw"] = p.ypd_in_world[0];
      data["buff_R_pitch"] = p.ypd_in_world[1];
      data["buff_R_dis"] = p.ypd_in_world[2];
      data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
      data["buff_pitch"] = p.ypr_in_world[1] * 57.3;
      data["buff_roll"] = p.ypr_in_world[2] * 57.3;
    }

    // -------------- 归因键位（固定键数，缺失时给 0/-1，PlotJuggler 不断线） --------------
    // 判读表：
    //   pose_pitch_a/b 分离约 2×视线仰角     -> 镜像判据成立（真解 pitch≈0）
    //   pnp_rmse_gap 接近 0                  -> RMSE 对镜像无区分度（改造前的常态）
    //   pnp_blade_count = 2                  -> 大符整符联合解算生效
    //   换组瞬间 slot 没变但 phase_innov 跳 72° -> 关联把相邻扇叶认成了同一片
    //   fold_steps 非零                       -> 该帧槽号记账错位了这么多格，已折叠进
    //                                           基准槽（θ 与拟合历史保持连续）
    //   fold_count 持续增长（不只在换目标时） -> assign_slots 的槽号关联在抖，根因在那
    //   换组瞬间 slot_epoch 变了             -> 槽号重新自举（纪元感知应已重设基准槽）
    //   dual_roll_delta 偏离整数             -> 槽号体系或位姿有问题
    //   theta_unwrap_step 出现 72° 或任意大跳 -> 槽偏移错位 / 解卷绕混叠
    //   obs_dt 超过 unwrap_max_dt_s          -> 该帧应当触发 history_reset
    //   r_innov_radial 成串同号 + normal_step 大跳 -> IPPE 镜像分支
    //   r_gate_streak 持续增长               -> 门控长期拒绝，R 停在旧值（不会瞬移，
    //                                           但瞄偏）；成因多半是观测坏而非模型漂
    //   buff_pitch 非零且随 buff_roll 周期变化 -> 固定 pitch=0 的系统性偏置
    //   d_span_* 稳定但 buff_R_dis 按 slot 分组 -> 关键点尺度没问题，锅在 PnP 侧
    //   d_span_* 本身按 slot 分成两组           -> YOLO 关键点尺度问题，改 C++ 无用
    //   kp_r_gap 随扇叶变化                    -> 流水灯/二维外推 R 的几何问题
    //   残差随 angle 而非 slot 变化             -> pitch=0 模型误差，独立议题
    data["slot"] = power_runes.has_value() ? static_cast<int>(power_runes->target_slot()) : -1;
    data["secondary_slot"] =
      power_runes_next.has_value() ? static_cast<int>(power_runes_next->target_slot()) : -1;
    data["slot_epoch"] = static_cast<int>(detector.slot_epoch());
    data["light_num"] = power_runes.has_value() ? power_runes->light_num : 0;
    data["rm_center_ms"] = detector.last_rm_center_ms();
    data["yolo_buff_ms"] = detector.last_yolo_buff_ms();
    data["rm_center_valid"] = detector.last_rm_center_valid() ? 1 : 0;
    data["rm_center_fallback"] = detector.last_rm_center_fallback() ? 1 : 0;
    const auto rm_center = detector.last_rm_center();
    data["rm_r_x"] = rm_center.has_value() ? rm_center->x : -1.0;
    data["rm_r_y"] = rm_center.has_value() ? rm_center->y : -1.0;

    // 主副同帧 roll 差必须是 72° 的整数倍；偏离整数即槽偏移错位。
    const bool dual_valid = power_runes.has_value() && power_runes_next.has_value() &&
                            !power_runes_next->is_unsolve();
    data["dual_valid"] = dual_valid ? 1 : 0;
    data["dual_roll_delta"] =
      dual_valid ? tools::limit_rad(
                     power_runes_next->ypr_in_world[2] - power_runes->ypr_in_world[2]) /
                     auto_buff::THETA
                 : 0.0;

    if (power_runes.has_value()) {
      const auto & blade = power_runes->target();
      // 四角回退帧的 flow_error 是 infinity，直接塞进 json 会序列化成 null 断掉曲线。
      auto finite_or = [](double v, double fallback) {
        return std::isfinite(v) ? v : fallback;
      };
      data["pnp_rmse"] = finite_or(blade.raw_reprojection_rmse, -1.0);
      data["pnp_corner_rmse"] = finite_or(blade.raw_corner_rmse, -1.0);
      data["pnp_flow_err"] = finite_or(blade.raw_flow_error, -1.0);
      data["pnp_fallback"] = blade.raw_pnp_fallback ? 1 : 0;
      data["pnp_used_rm_r"] = blade.raw_used_rm_center ? 1 : 0;
      data["pnp_rm_r_fallback"] = blade.raw_rm_center_fallback ? 1 : 0;
      data["r_center_observed"] = power_runes->r_center_observed ? 1 : 0;
    } else {
      data["pnp_used_rm_r"] = 0;
      data["pnp_rm_r_fallback"] = 0;
    }

    // -------------- R 来源与 kpt4 门控（本次改动的验收键位） --------------
    // r_src: 0=None 1=RmCore 2=RmCoreUngated 3=RmCoreEscaped 4=FusedKpt4
    // 判读：
    //   原先姿态检查失败的帧 r_src 由 0 变 1，且 pnp_used_rm_r 由 0 变 1 ⇒ 解耦生效；
    //   这些帧 xyz_in_world 不再与切换同步阶跃                          ⇒ 结构切换已消；
    //   真误识别帧出现 r_src=4                                          ⇒ 门控确实在拦；
    //   r_gate_streak 长期贴着逃生阈值（8）                             ⇒ 门限定得过紧。
    if (power_runes.has_value()) {
      // 未做门控的帧（无 kpt4 参照或无 rm R）这两个量是 NaN，塞进 json 会变 null，
      // 与其他键位一致地降为 -1.0。
      data["r_src"] = static_cast<int>(power_runes->r_center_source);
      data["r_gate_dist"] = std::isfinite(power_runes->r_gate_distance)
                              ? static_cast<double>(power_runes->r_gate_distance)
                              : -1.0;
      data["r_gate_thresh"] = std::isfinite(power_runes->r_gate_threshold)
                                ? static_cast<double>(power_runes->r_gate_threshold)
                                : -1.0;
      data["r_kpt4_fused_n"] = power_runes->r_kpt4_fused_count;
    } else {
      data["r_src"] = 0;
      data["r_kpt4_fused_n"] = 0;
    }
    data["r_gate_streak"] = detector.r_gate_reject_streak();

    // -------------- 关键点尺度诊断（只在本调试程序，不进 standard_mpc 生产日志） --------------
    // 这几个量**完全不经过 PnP**：直接由像素跨度和实物尺寸反推距离，是把像素读数翻译成
    // “米”、与尺量距离对齐的唯一途径。它们把两条互斥的根因一刀切开：
    //   d_span_* 稳定在实测值而 buff_R_dis 按扇叶分组 ⇒ 问题在 PnP 侧（点集或 IPPE 选边）；
    //   d_span_* 本身就按扇叶分成两组、跨度差 5%~6%   ⇒ 问题在 YOLO 关键点尺度，
    //   C++ 侧怎么改都没用（9.2/8.7 = 1.0575，正好是 5.7%），归到标注/推理去修，
    //   **不在这里加逐扇叶距离补偿**。
    // 固定键数，缺失时给 -1，PlotJuggler 不断线。
    {
      // OBJECT_POINTS：kpt0(0,0,0.827) kpt1(0,0.127,0.700) kpt2(0,0,0.573)
      //                kpt3(0,-0.127,0.700) kpt5(0,0,0.344)，target 中心在 0.700。
      constexpr double kSpan02M = 0.827 - 0.573;  // 0.254 m，径向对角线
      constexpr double kSpan13M = 0.127 * 2.0;    // 0.254 m，切向对角线
      constexpr double kSpanC5M = 0.700 - 0.344;  // 0.356 m，正是外推 R 要除的那一段

      double span02 = -1.0, span13 = -1.0, spanc5 = -1.0;
      double r_gap = -1.0, kp4_conf = 0.0, kp_conf_min = -1.0;

      if (power_runes.has_value()) {
        const auto & blade = power_runes->target();
        if (blade.observed && blade.points.size() > 5) {
          span02 = cv::norm(blade.points[0] - blade.points[2]);
          span13 = cv::norm(blade.points[1] - blade.points[3]);
          spanc5 = cv::norm(blade.center - blade.points[5]);
          // 进 PnP 的五个点里最弱的一个。跨度异常时先看它，排除“某片某个角本身没识别准”。
          for (const std::size_t i : {0U, 1U, 2U, 3U, 5U}) {
            if (i >= blade.keypoint_confidences.size()) continue;
            const double c = blade.keypoint_confidences[i];
            if (kp_conf_min < 0.0 || c < kp_conf_min) kp_conf_min = c;
          }
        }
        // 二维外推 R 与 YOLO kpt4 的像素距离：直接回答“两个 R 谁在骗人”。
        // 连同 kp4_conf 一起看，也是“kpt4 够不够格进 PnP”的准入依据。
        kp4_conf = blade.raw_r_confidence;
        if (
          blade.raw_r_valid && std::isfinite(power_runes->r_center_raw.x) &&
          std::isfinite(power_runes->r_center_raw.y))
          r_gap = cv::norm(power_runes->r_center_raw - blade.raw_r_center);
      }

      auto implied = [&](double span_px, double span_m) {
        return (fx > 0.0 && span_px > 1e-3) ? fx * span_m / span_px : -1.0;
      };

      data["kp_span_02"] = span02;
      data["kp_span_13"] = span13;
      data["kp_span_c5"] = spanc5;
      data["d_span_02"] = implied(span02, kSpan02M);
      data["d_span_13"] = implied(span13, kSpan13M);
      data["d_span_c5"] = implied(spanc5, kSpanC5M);
      // 两条对角线物理等长，比值随扇叶角变化只反映符面倾斜；某一片单独偏了才是标注问题。
      data["kp_span_ratio"] = (span02 > 0.0 && span13 > 1e-3) ? span02 / span13 : -1.0;
      data["kp_r_gap"] = r_gap;
      // rm-core 本帧 R 与槽位关联使用的融合/平滑中心之间的像素差。
      data["rm_r_assoc_gap"] =
        (power_runes.has_value() && power_runes->r_center_observed &&
         std::isfinite(power_runes->r_center_raw.x) && std::isfinite(power_runes->r_center.x))
          ? cv::norm(power_runes->r_center_raw - power_runes->r_center)
          : -1.0;
      data["kp4_conf"] = kp4_conf;
      data["kp_conf_min"] = kp_conf_min;
    }

    data["normal_step"] = solver.last_normal_step_rad() * 57.3;
    // 镜像判据的核心三个键：两支候选各自的符面 pitch（度）与它们的 RMSE 差。
    // 符竖直 ⇒ 真解 pitch≈0、镜像解≈±2×视线仰角（实测仰角约 11° ⇒ 分离约 22°）。
    // rmse_gap 实测远小于 0.5px ⇒ RMSE 选边等于抛硬币；整符点集张开后它应明显变大。
    data["pose_pitch_a"] = solver.last_pose_pitch_a();
    data["pose_pitch_b"] = solver.last_pose_pitch_b();
    data["pnp_rmse_gap"] = solver.last_rmse_gap();
    data["pnp_blade_count"] = solver.last_blade_count();
    data["slot_sign"] = solver.last_slot_sign();

    {
      const auto & phase = target->phase();
      const auto & rf = target->r_filter();
      data["good_quality"] = target->last_good_quality() ? 1 : 0;
      data["target_epoch"] = static_cast<int>(target->last_slot_epoch());
      data["theta_obs"] = phase.last_theta_obs() * 57.3;
      data["theta_pred"] = phase.last_theta_pred() * 57.3;
      data["phase_innov"] = phase.last_innovation() * 57.3;
      data["slot_offset"] = phase.last_slot_offset() * 57.3;
      data["phase_gated"] = phase.last_gated() ? 1 : 0;
      data["phase_reanchor"] = phase.last_reanchored() ? 1 : 0;
      data["slot_learned"] = phase.last_slot_learned() ? 1 : 0;
      data["gate_streak"] = phase.gate_streak();
      data["dir_votes"] = phase.dir_votes();
      data["theta_unwrapped"] = phase.theta_unwrapped() * 57.3;
      data["fit_rms"] = phase.last_fit_rms() * 57.3;
      data["fit_samples"] = static_cast<int>(phase.last_fit_samples());
      data["fit_span"] = phase.fit_span();
      data["r_innov"] = rf.last_innovation_m();
      data["r_innov_radial"] = rf.last_innovation_radial_m();
      data["r_gated"] = rf.last_gated() ? 1 : 0;
      data["r_relocate"] = rf.last_relocated() ? 1 : 0;
      data["r_gate_streak"] = rf.gate_streak();
      data["r_updates"] = rf.update_count();
      // 解卷绕守卫：obs_dt 是相位观测间隔；unwrap_step 应约等于单帧转角，
      // 72° = 槽偏移错位、任意大跳 = 混叠；history_reset 说明本帧丢弃了拟合历史。
      data["obs_dt"] = phase.last_obs_dt();
      data["theta_unwrap_step"] = phase.last_unwrap_step() * 57.3;
      data["unwrap_residual"] = phase.last_unwrap_residual() * 57.3;
      data["history_reset"] = phase.last_history_reset() ? 1 : 0;
      // 整槽折叠：本帧把多少格 72° 错位吸收进了基准槽。偶发几次 = 换目标瞬间的正常
      // 抖动；fold_count 持续增长 = detector 的槽号关联本身在抖，根因在 assign_slots。
      data["fold_steps"] = phase.last_fold_steps();
      data["fold_count"] = phase.fold_count();
      data["dual_roll_mismatch"] = target->last_dual_roll_mismatch() * 57.3;
      data["dual_roll_ok"] = target->last_dual_roll_consistent() ? 1 : 0;
      data["unsolvable"] = target->is_unsolve() ? 1 : 0;
    }

    if (!target->is_unsolve() && power_runes.has_value()) {
      auto & p = power_runes.value();

      // 红：本帧 YOLO 的原始观测（当前 target 四角 + 中心 + 关联用共享 R）
      for (int i = 0; i < 4; i++) tools::draw_point(img, p.target().points[i]);
      tools::draw_point(img, p.target().center, {0, 0, 255}, 5);
      tools::draw_point(img, p.r_center, {0, 0, 255}, 5);

      // 绿：当前帧模型状态的整符，亮绿是当前 target 槽。
      auto_buff::draw_whole_buff(img, solver, *target, {0, 140, 0}, {0, 255, 0}, 6, "NOW");

      // 品红：kpt4 门控圆（圆心 = 融合 kpt4，半径 = 门限），加本帧 R 的来源标注。
      auto_buff::draw_r_gate(img, p);

      // 蓝：预测时刻的整符，整轮转过多少角度直接可见，比只看一片更容易判断预测好坏。
      // 只在完整弹道迭代成功后才画，避免把“没有预测”误诊成预测位置跳变。
      if (aimer.prediction_valid() && target_clone && !target_clone->is_unsolve())
        auto_buff::draw_whole_buff(
          img, solver, *target_clone, {140, 60, 0}, {255, 100, 0}, 6, "PREDICT");

      // 观测器内部数据
      Eigen::VectorXd x = target->ekf_x();
      data["R_yaw"] = x[0];
      data["R_V_yaw"] = x[1];
      data["R_pitch"] = x[2];
      data["R_dis"] = x[3];
      data["yaw"] = x[4] * 57.3;

      data["angle"] = x[5] * 57.3;
      data["spd"] = x[6] * 57.3;
      if (x.size() >= 10) {
        data["spd"] = x[6];
        data["a"] = x[7];
        data["w"] = x[8];
        data["fi"] = x[9];
        data["spd0"] = target->spd;
      }
    }

    // 云台响应情况
    data["gimbal_yaw"] = gs.yaw * 57.3;
    data["gimbal_pitch"] = gs.pitch * 57.3;
    data["gimbal_yaw_vel"] = gs.yaw_vel * 57.3;
    data["gimbal_pitch_vel"] = gs.pitch_vel * 57.3;

    
    data["plan_yaw"] = plan.yaw * 57.3;
    data["plan_pitch"] = plan.pitch * 57.3;
    data["plan_yaw_vel"] = plan.yaw_vel * 57.3;
    data["plan_pitch_vel"] = plan.pitch_vel * 57.3;
    data["plan_yaw_acc"] = plan.yaw_acc * 57.3;
    data["plan_pitch_acc"] = plan.pitch_acc * 57.3;
    data["shoot"] = plan.fire ? 1 : 0;
    

    // PlotJuggler 保留最重要的 plan ↔ gimbal 响应对照，以及关联/PnP 根因指标；
    // 只删重复的派生误差和临时诊断，不能删掉控制主波形。
    nlohmann::json compact_data;
    compact_data["gimbal_yaw"] = data["gimbal_yaw"];
    compact_data["plan_yaw"] = data["plan_yaw"];
    compact_data["gimbal_pitch"] = data["gimbal_pitch"];
    compact_data["plan_pitch"] = data["plan_pitch"];
    compact_data["gimbal_yaw_vel"] = data["gimbal_yaw_vel"];
    compact_data["plan_yaw_vel"] = data["plan_yaw_vel"];
    compact_data["gimbal_pitch_vel"] = data["gimbal_pitch_vel"];
    compact_data["plan_pitch_vel"] = data["plan_pitch_vel"];
    compact_data["plan_yaw_acc"] = data["plan_yaw_acc"];
    compact_data["plan_pitch_acc"] = data["plan_pitch_acc"];
    compact_data["my_color"] = static_cast<int>(gs.my_color);
    compact_data["rm_center_valid"] = data.value("rm_center_valid", 0);
    compact_data["rm_center_fallback"] = data.value("rm_center_fallback", 0);
    compact_data["slot"] = data.value("slot", -1);
    compact_data["fold_steps"] = data.value("fold_steps", 0);
    compact_data["theta_obs"] = data.value("theta_obs", 0.0);
    compact_data["pnp_used_rm_r"] = data.value("pnp_used_rm_r", 0);
    compact_data["pnp_rmse"] = data.value("pnp_rmse", -1.0);
    // R 来源与门控进精简曲线：本次改动的主验收量，和 pnp_used_rm_r 对照看。
    compact_data["r_src"] = data.value("r_src", 0);
    compact_data["r_gate_dist"] = data.value("r_gate_dist", -1.0);
    compact_data["r_gate_thresh"] = data.value("r_gate_thresh", -1.0);
    compact_data["r_gate_streak"] = data.value("r_gate_streak", 0);
    compact_data["buff_R_dis"] = data.value("buff_R_dis", -1.0);
    compact_data["rm_r_assoc_gap"] = data.value("rm_r_assoc_gap", -1.0);
    compact_data["shoot"] = plan.fire ? 1 : 0;
    plotter.plot(compact_data);

    // 与 result 共用下面的 waitKey：同时显示 rm-core 原生特征图和本次检测实际使用的二值图。
    if (detector.rm_center_enabled()) {
      DebugTools::get()->show("rm-core");
      const cv::Mat binary = detector.last_rm_binary_image();
      if (!binary.empty()) {
        cv::Mat binary_show;
        cv::resize(binary, binary_show, {}, 0.5, 0.5, cv::INTER_NEAREST);
        cv::imshow("rm-core binary", binary_show);
      }
    }

    cv::resize(img, img, {}, 0.5, 0.5);
    // 画面中心准星
    auto ctr = cv::Point(img.cols / 2, img.rows / 2);
    cv::line(img, {ctr.x - 12, ctr.y}, {ctr.x + 12, ctr.y}, cv::Scalar(0, 255, 255), 2);
    cv::line(img, {ctr.x, ctr.y - 12}, {ctr.x, ctr.y + 12}, cv::Scalar(0, 255, 255), 2);
    cv::imshow("result", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
