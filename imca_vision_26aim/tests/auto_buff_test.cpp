#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

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
#include "vc/core/debug_tools.h"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{config-path c  | configs/sentry.yaml    | yaml配置文件的路径}"
  "{buff-mode m    | big                    | 测试模式：small 或 big}"
  "{start-index s  | 0                      | 视频起始帧下标    }"
  "{end-index e    | 0                      | 视频结束帧下标    }"
  "{@input-path    | assets/demo/1        | avi和txt文件的路径}";

namespace
{
constexpr const char * kResultWindow = "result";
constexpr const char * kFrameBar = "frame";
constexpr const char * kThreshBar = "rm thresh";
}  // namespace

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto buff_mode = cli.get<std::string>("buff-mode");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");
  if (buff_mode != "small" && buff_mode != "big") {
    tools::logger()->error("buff-mode 只能是 small 或 big，当前值：{}", buff_mode);
    return 1;
  }
  const bool test_big_buff = buff_mode == "big";

  // 关键点尺度诊断要把像素跨度翻成米，需要 fx。Solver 没有暴露内参，直接读同一份 yaml，
  // 不为一个诊断给生产类加接口。读不到就把 d_span_* 置 -1，其余诊断照常输出。
  // my_color 只用于日志：离线没有云台回传，rm-core 的通道由 yaml enemy_color 反色决定。
  double fx = 0.0;
  uint8_t my_color = 1U;
  try {
    const auto yaml = YAML::LoadFile(config_path);
    const auto camera_matrix = yaml["camera_matrix"].as<std::vector<double>>();
    if (!camera_matrix.empty()) fx = camera_matrix[0];
    if (yaml["enemy_color"])
      my_color = yaml["enemy_color"].as<std::string>() == "red" ? 0U : 1U;
  } catch (const std::exception & e) {
    tools::logger()->warn("读取 camera_matrix 失败，隐含距离诊断关闭：{}", e.what());
  }

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);

  if (!video.isOpened() || !text.is_open()) {
    tools::logger()->error("无法打开视频或姿态文件：{}", input_path);
    return 1;
  }

  double t, w, x, y, z;
  int pose_count = 0;
  while (text >> t >> w >> x >> y >> z) pose_count++;

  int last_frame = pose_count - 1;
  if (end_index > 0 && end_index < last_frame) last_frame = end_index;
  if (start_index < 0 || start_index > last_frame) {
    tools::logger()->error("start-index {} 超出范围，姿态行数 {}", start_index, pose_count);
    return 1;
  }

  auto seek_video = [&](int frame) {
    video.release();
    text.clear();
    text.seekg(0);
    if (!video.open(video_path)) return false;
    for (int i = 0; i < frame; i++) {
      if (!video.grab() || !(text >> t >> w >> x >> y >> z)) return false;
    }
    return true;
  };

  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);

  // 两种派生 Target 各自常驻，避免通过注释切换时把 SmallTarget 和 BigTarget 互相赋值。
  auto_buff::SmallTarget small_target(config_path);
  auto_buff::BigTarget big_target(config_path);
  auto_buff::Target * target = test_big_buff
                                 ? static_cast<auto_buff::Target *>(&big_target)
                                 : static_cast<auto_buff::Target *>(&small_target);

  auto_buff::Aimer aimer(config_path);

  // 关联参考角改用相位模型外推：换组黑屏期间它仍在推进，不会像上一帧像素角那样陈旧。
  detector.set_slot_angle_predictor(
    [&](std::size_t slot) { return target->image_angle_of_slot(slot); },
    [&](std::size_t slot, double angle) { target->calibrate_image_angle(slot, angle); });

  cv::Mat img;
  // YOLO 会往 img 上画框，rm-core 的颜色二值化必须看未污染的原图。
  // 暂停时拖阈值预览也用这一份。
  cv::Mat raw_img;
  auto t0 = std::chrono::steady_clock::now();

  if (!seek_video(start_index)) {
    tools::logger()->error("视频无法跳转到第 {} 帧", start_index);
    return 1;
  }

  cv::namedWindow(kResultWindow, cv::WINDOW_AUTOSIZE);
  cv::createTrackbar(kFrameBar, kResultWindow, nullptr, last_frame);
  cv::setTrackbarPos(kFrameBar, kResultWindow, start_index);

  // 二值化阈值拖动条：初值取 yaml 当前生效值，用户没拖过就不覆盖按颜色选阈值的逻辑。
  int last_threshold = detector.rm_color_threshold();
  if (detector.rm_center_enabled()) {
    cv::createTrackbar(kThreshBar, kResultWindow, nullptr, 255);
    cv::setTrackbarPos(kThreshBar, kResultWindow, last_threshold);
  }

  // 拖动条变化时才写回 Detector，返回值表示本次是否需要刷新二值化预览。
  auto sync_rm_threshold = [&]() {
    if (!detector.rm_center_enabled()) return false;
    const int pos = cv::getTrackbarPos(kThreshBar, kResultWindow);
    if (pos == last_threshold) return false;
    last_threshold = pos;
    detector.set_rm_color_threshold(pos);
    return true;
  };

  auto show_binary = [](const cv::Mat & binary) {
    if (binary.empty()) return;
    cv::Mat binary_show;
    cv::resize(binary, binary_show, {}, 0.5, 0.5, cv::INTER_NEAREST);
    cv::imshow("rm-core binary", binary_show);
  };

  bool paused = false;

  for (int frame_count = start_index; !exiter.exit();) {
    if (paused) {
      int key = cv::waitKey(30);
      // 暂停时改阈值不重跑检测，只用干净原图单独二值化一次，避免污染跟踪状态。
      if (sync_rm_threshold() && !raw_img.empty())
        show_binary(detector.binarize_with_rm_threshold(raw_img));
      if (key == 'q' || key == 27) break;
      if (key != ' ') continue;

      int selected = cv::getTrackbarPos(kFrameBar, kResultWindow);
      if (selected != frame_count) {
        frame_count = selected;
        if (!seek_video(frame_count)) {
          tools::logger()->error("视频无法跳转到第 {} 帧", frame_count);
          break;
        }
        detector.reset();
        // 录像跳转会破坏帧间连续性，Detector/Target/Solver 必须同时重置。
        solver.reset();
        small_target = auto_buff::SmallTarget(config_path);
        big_target = auto_buff::BigTarget(config_path);
        aimer = auto_buff::Aimer(config_path);
      }
      paused = false;
    }

    video.read(img);
    if (img.empty()) break;

    if (!(text >> t >> w >> x >> y >> z)) break;
    auto timestamp =
      t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             std::chrono::duration<double>(t));
    Eigen::Quaterniond q(w, x, y, z);

    // 运行中拖阈值：下一帧的 rm-core 检测就用新值，无需暂停。
    sync_rm_threshold();

    // 离线没有云台回传。yaw/pitch 用录制四元数反解（与 gimbal_test 同一套
    // eulers(2,1,0) 约定），这样 fire_gate_check 比较的是录像自身的姿态；
    // 速度项无法从单帧姿态得到，留 0，因此 shoot 键位离线只作参考。
    io::GimbalState gs{};
    const Eigen::Vector3d gimbal_ypr = tools::eulers(q, 2, 1, 0);
    gs.yaw = static_cast<float>(gimbal_ypr[0]);
    gs.pitch = static_cast<float>(gimbal_ypr[1]);
    gs.yaw_vel = 0.0F;
    gs.pitch_vel = 0.0F;
    gs.bullet_speed = 23.0F;
    gs.my_color = my_color;

    /// 打符核心逻辑

    solver.set_R_gimbal2world(q);
    // 与正式入口同一条颜色链路。阈值一旦被拖动条改过，这里只切通道不改阈值。
    detector.setMyColor(gs.my_color);

    // rm-core 的绘制接口依赖本帧原图缓存；必须在 YOLO 往 img 上画框之前设置。
    // RuneDetector 会在同一次检测中直接绘制到该缓存，不会为可视化重复跑检测。
    img.copyTo(raw_img);
    if (detector.rm_center_enabled()) DebugTools::get()->setImage(img);

    std::optional<auto_buff::PowerRune> power_runes;
    std::optional<auto_buff::PowerRune> power_runes_next;
    auto_aim::Plan plan{};
    std::unique_ptr<auto_buff::Target> target_clone;
    bool secondary_plan_valid = false;

    if (test_big_buff) {
      // ====== 大符双目标路径 ======
      // detect_dual(): 同时返回主目标(power_runes)和副目标(power_runes_next)
      std::tie(power_runes, power_runes_next) = detector.detect_dual(img, q, timestamp);
    } else {
      // 小符只跟踪当前 target，不运行副目标预计算和大符切换状态机。
      power_runes = detector.detect(img, q, timestamp);
    }

    // solve_dual() 会原地清除 PnP 失败的 optional。先保存 Detector 快照，才能区分
    // “YOLO/关联没有返回目标”和“目标已检测但五点 PnP 失败”这两个完全不同的阶段。
    const auto detected_primary = power_runes;
    const auto detected_secondary = power_runes_next;

    if (test_big_buff) {
      // 两片的点旋转到同一符坐标系拼成一个点集，只解一次 PnP（整符一个刚体一个位姿）。
      solver.solve_dual(power_runes, power_runes_next);

      // 整符解算自标定出的槽差符号回灌相位模型（槽偏移刚体推导的唯一未知量）。
      if (solver.last_slot_sign() != 0) target->set_slot_sign(solver.last_slot_sign());

      target->get_target(power_runes, timestamp);
      auto * bt = static_cast<auto_buff::BigTarget *>(target);
      // 副片 roll 也喂相位模型（观测率翻倍 + 72° 自洽性检查）。
      bt->observe_secondary(power_runes_next);

      // 在副本上做 mpc 预测，避免污染主跟踪状态。
      // to_now=false：离线 timestamp 走录像时钟，回放比实时快，
      // 用 now-timestamp 当系统延迟会得到一个越来越大的假值。
      target_clone = std::make_unique<auto_buff::BigTarget>(*bt);
      plan = aimer.mpc_aim(*target_clone, timestamp, gs, false);

      // 预计算副目标方案：区分“检测/PnP 有 S”与“Aimer 最终接受了 S”。
      // compute_secondary_plan 只用这个 t 算 now-t 的系统延迟（不参与目标模型时基），
      // 离线传当前时刻即把延迟置 0，与上面 to_now=false 的口径一致。
      if (!target->is_unsolve())
        secondary_plan_valid = aimer.compute_secondary_plan(
          bt, power_runes_next, gs, std::chrono::steady_clock::now());

      // 二次门控
      plan.fire = auto_buff::fire_gate_check(plan, *target, gs);
      // 离线没有 io::Gimbal，跳过 big_buff_send() 的开火/切槽状态机；
      // 主副目标本身仍逐帧解算，切换时序需要在实车上验证。
    } else {
      solver.solve(power_runes);

      if (solver.last_slot_sign() != 0) target->set_slot_sign(solver.last_slot_sign());

      target->get_target(power_runes, timestamp);
      auto * st = static_cast<auto_buff::SmallTarget *>(target);

      // 在副本上做 mpc 预测，避免污染主跟踪状态
      // to_now=false 的理由同大符分支：离线时钟不能当作系统延迟。
      target_clone = std::make_unique<auto_buff::SmallTarget>(*st);
      plan = aimer.mpc_aim(*target_clone, timestamp, gs, false);

      // 二次门控
      plan.fire = auto_buff::fire_gate_check(plan, *target, gs);
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

    // 阶段区分：检测有没有给目标 / PnP 有没有成功 / 副目标方案有没有被接受。
    data["detector_primary"] = detected_primary.has_value() ? 1 : 0;
    data["detector_secondary"] = detected_secondary.has_value() ? 1 : 0;
    data["solver_primary"] = power_runes.has_value() ? 1 : 0;
    data["solver_secondary"] = power_runes_next.has_value() ? 1 : 0;
    data["secondary_plan_valid"] = secondary_plan_valid ? 1 : 0;

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
    data["rm_color_thresh"] = detector.rm_color_threshold();
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
      // R 点落在圆外 = 被门控拦下、PnP 用的是圆心那个融合 kpt4。
      auto_buff::draw_r_gate(img, p);

      // 蓝：预测时刻的整符，整轮转过多少角度直接可见，比只看一片更容易判断预测好坏。
      // 只在完整弹道迭代成功后才画，避免把“没有预测”误诊成预测位置跳变。
      if (aimer.prediction_valid() && target_clone && !target_clone->is_unsolve())
        auto_buff::draw_whole_buff(
          img, solver, *target_clone, {140, 60, 0}, {255, 100, 0}, 6, "PREDICT");

      // 观测器内部数据
      Eigen::VectorXd x_state = target->ekf_x();
      data["R_yaw"] = x_state[0];
      data["R_V_yaw"] = x_state[1];
      data["R_pitch"] = x_state[2];
      data["R_dis"] = x_state[3];
      data["yaw"] = x_state[4] * 57.3;

      data["angle"] = x_state[5] * 57.3;
      data["spd"] = x_state[6] * 57.3;
      if (x_state.size() >= 10) {
        data["spd"] = x_state[6];
        data["a"] = x_state[7];
        data["w"] = x_state[8];
        data["fi"] = x_state[9];
        data["spd0"] = target->spd;
      }
    }

    // 云台响应情况：离线来自录制四元数，速度项恒 0。
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

    data["my_color"] = static_cast<int>(gs.my_color);
    // buff_layout.xml 用的是旧 cmd_* 键名；离线全量发送，顺带保留这两个别名，
    // 免得回放时还要改布局文件。debug_mpc 那边为了 UDP 带宽只发 compact 子集，
    // 离线没有这个约束，全量键更利于归因。
    data["cmd_yaw"] = data["plan_yaw"];
    data["cmd_pitch"] = data["plan_pitch"];

    plotter.plot(data);

    // 逐帧回放的最小 HUD：定位到某一帧后不用去看曲线也能知道卡在哪个阶段。
    cv::putText(
      img,
      fmt::format(
        "frame:{} DET P:{} S:{} | PNP P:{} S:{} | PLAN S:{} | thresh:{}", frame_count,
        detected_primary.has_value() ? 1 : 0, detected_secondary.has_value() ? 1 : 0,
        power_runes.has_value() ? 1 : 0, power_runes_next.has_value() ? 1 : 0,
        secondary_plan_valid ? 1 : 0, detector.rm_color_threshold()),
      {20, 75}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 255}, 2, cv::LINE_AA);
    cv::putText(
      img,
      fmt::format(
        "YOLO score:{} corner:{} flow:{} NMS:{}", detector.last_yolo_score_count(),
        detector.last_yolo_candidate_count(), detector.last_yolo_flow_valid_count(),
        detector.last_yolo_nms_count()),
      {20, 100}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 255, 255}, 2, cv::LINE_AA);

    // 与 result 共用下面的 waitKey：同时显示 rm-core 原生特征图和本次检测实际使用的二值图。
    if (detector.rm_center_enabled()) {
      DebugTools::get()->show("rm-core");
      show_binary(detector.last_rm_binary_image());
    }

    cv::resize(img, img, {}, 0.75, 0.75);
    // 画面中心准星
    const cv::Point center(img.cols / 2, img.rows / 2);
    const int cross_len = 12;
    const int cross_thickness = 2;
    const cv::Scalar cross_color(0, 255, 255);
    cv::line(img, {center.x - cross_len, center.y}, {center.x + cross_len, center.y}, cross_color, cross_thickness);
    cv::line(img, {center.x, center.y - cross_len}, {center.x, center.y + cross_len}, cross_color, cross_thickness);
    cv::imshow(kResultWindow, img);

    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) break;

    if (cv::getTrackbarPos(kFrameBar, kResultWindow) != frame_count) {
      paused = true;
      continue;
    }
    frame_count++;
    if (frame_count > last_frame) break;
    cv::setTrackbarPos(kFrameBar, kResultWindow, frame_count);
    if (key == ' ') paused = true;
  }
  cv::destroyAllWindows();
  text.close();  // 关闭文件

  return 0;
}
