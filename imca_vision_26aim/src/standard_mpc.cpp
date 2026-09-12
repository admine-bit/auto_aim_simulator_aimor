#include <chrono>
#include <cstdlib>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

#include "tasks/auto_aim/target.hpp"

const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   |configs/sentry.yaml | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char *argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path"))
  {
    cli.printMessage();
    return 0;
  }

  auto yaml = tools::load(config_path);                   // 加载YAML配置文件
  auto fire_dis_ = tools::read<double>(yaml, "fire_dis"); // 读取开火判断的位置误差阈值
  tools::logger()->info("[auto_aim_debug_mpc] fire_dis: {}", fire_dis_);

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  const auto *debug_env = std::getenv("IMCA_VISION_DEBUG_WINDOWS");
  const bool vision_debug_windows = debug_env != nullptr && std::string(debug_env) == "1";
  auto_aim::YOLO yolo(config_path, vision_debug_windows);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target(config_path);
  auto_buff::BigTarget buff_big_target(config_path);
  auto_buff::Aimer buff_aimer(config_path);
  int buff_lost_count = 0;

  auto reset_buff = [&]()
  {
    buff_detector.reset();
    // Detector、Target 与 IPPE 双解的时序参考必须在同一次回合重置中一起清除。
    buff_solver.reset();
    buff_small_target = auto_buff::SmallTarget(config_path);
    buff_big_target = auto_buff::BigTarget(config_path);
    buff_aimer = auto_buff::Aimer(config_path);
    buff_lost_count = 0;
  };

  // 关联参考角改用相位模型外推：换组黑屏期间它仍在推进，不会像上一帧像素角那样陈旧。
  // 两个 Target 都要挂，按当前模式取用。
  auto *active_buff_target = static_cast<auto_buff::Target *>(&buff_small_target);
  buff_detector.set_slot_angle_predictor(
      [&](std::size_t slot)
      { return active_buff_target->image_angle_of_slot(slot); },
      [&](std::size_t slot, double angle)
      {
        active_buff_target->calibrate_image_angle(slot, angle);
      });

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  std::atomic<bool> quit = false;

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  auto plan_thread = std::thread([&]()
                                 {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      if (!target_queue.empty() && mode != io::GimbalMode::SMALL_BUFF && mode != io::GimbalMode::BIG_BUFF) //&& mode == io::GimbalMode::AUTO_AIM
      {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = planner.plan(target, gs.bullet_speed);// 发送规划结果到下位机（控制使能、开火指令、角度、速度、加速度等）
        
        plan.pitch = - plan.pitch;
        plan.pitch_vel = - plan.pitch_vel;
        plan.pitch_acc = - plan.pitch_acc;

        double min_dist = 1.0;  // 无目标时默认值，避免除零
        if (target.has_value()) 
        {
          min_dist = 1e10;
          for (const auto & xyza : target->armor_xyza_list()) 
          {
            const double dist =  xyza.head<2>().norm();  // sqrt(x^2 + y^2)
            if (dist < min_dist) min_dist = dist;
          }
          if (min_dist < 0.1) min_dist = 0.1;  // 保护
        }
        if(min_dist<1.1)
        {
          plan.fire = (plan.fire == 1 || ( (abs(gs.yaw -plan.yaw) <  fire_dis_/2.0/min_dist)  && (abs(gs.pitch -plan.pitch) < fire_dis_/2.0/min_dist) ));
        }
        else
        {
          plan.fire = (plan.fire == 1 && ( (abs(gs.yaw -plan.yaw) <  fire_dis_/min_dist)  && (abs(gs.pitch -plan.pitch) < fire_dis_/min_dist) ));
        }
        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        // tools::logger()->info("aim");
        std::this_thread::sleep_for(9ms);
      } else
        std::this_thread::sleep_for(100ms);
    } });

  while (!exiter.exit())
  {
    mode = gimbal.mode();

    auto new_mode = mode.load();
    if (last_mode != new_mode)
    {
      if (new_mode == io::GimbalMode::SMALL_BUFF || new_mode == io::GimbalMode::BIG_BUFF)
      {
        reset_buff();
      }
      tools::logger()->info("Switch to {}", gimbal.str(new_mode));
      last_mode = new_mode;
    }

    camera.read(img, t);
    auto q = gimbal.q(t - 1ms);
    // 自身位置与姿态使用同一个图像时刻，避免位移补偿和视觉观测错帧
    // auto self_world_xy = gimbal.xy(t-1ms);
    auto self_world_xy = Eigen::Vector2d::Zero();

    auto gs = gimbal.state();
    recorder.record(img, q, t); // 录像
    solver.set_R_gimbal2world(q);
    solver.set_t_gimbal2world(gs.z_chassis); // 设置解算器中云台到世界坐标系的平移（底盘高度补偿）

    /// 打符
    if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF)
    {

      auto q_buff = gimbal.q(t);

      buff_solver.set_R_gimbal2world(q_buff);
      buff_detector.setMyColor(gs.my_color);

      std::optional<auto_buff::PowerRune> power_runes;
      std::optional<auto_buff::PowerRune> power_runes_next;
      auto_aim::Plan buff_plan{};
      std::unique_ptr<auto_buff::Target> target_clone;
      auto_buff::Target *target = nullptr;

      if (mode.load() == io::GimbalMode::BIG_BUFF)
      { // mode.load() == io::GimbalMode::BIG_BUFF
        target = &buff_big_target;
        active_buff_target = target;
        // ====== 大符双目标路径 ======
        // detect_dual(): 同时返回主目标(power_runes)和副目标(power_runes_next)
        auto [first, next] = buff_detector.detect_dual(img, q_buff, t);
        power_runes = first;
        power_runes_next = next;

        // 主副目标各用自己的 kpt0~3+kpt5 五点立即 PnP；solve_dual 再统一共享 R/yaw/pitch，
        // 副目标异常不会阻塞主目标和当帧 yaw 输出。
        buff_solver.solve_dual(power_runes, power_runes_next);

        // 整符解算自标定出的槽差符号回灌相位模型（槽偏移刚体推导的唯一未知量）。
        if (buff_solver.last_slot_sign() != 0)
          target->set_slot_sign(buff_solver.last_slot_sign());

        target->get_target(power_runes, t);
        auto *big_target = static_cast<auto_buff::BigTarget *>(target);
        // 副片 roll 也喂相位模型（观测率翻倍 + 72° 自洽性检查）。
        big_target->observe_secondary(power_runes_next);

        // 在副本上做 mpc 预测，避免污染主跟踪状态
        target_clone = std::make_unique<auto_buff::BigTarget>(*big_target);
        buff_plan = buff_aimer.mpc_aim(*target_clone, t, gs, true);

        // 预计算副目标方案
        if (!target->is_unsolve())
          buff_aimer.compute_secondary_plan(big_target, power_runes_next, gs, t);

        // 二次门控
        buff_plan.fire = auto_buff::fire_gate_check(buff_plan, *target, gs);
        // 开火状态机发送
        buff_aimer.big_buff_send(buff_plan, gimbal, buff_detector);
      }
      else
      {
        target = &buff_small_target;
        active_buff_target = target;
        // ====== 小符路径 ======
        power_runes = buff_detector.detect(img, q_buff, t);

        buff_solver.solve(power_runes);

        if (buff_solver.last_slot_sign() != 0)
          target->set_slot_sign(buff_solver.last_slot_sign());

        target->get_target(power_runes, t);
        auto *small_target = static_cast<auto_buff::SmallTarget *>(target);

        // 在副本上做 mpc 预测，避免污染主跟踪状态
        target_clone = std::make_unique<auto_buff::SmallTarget>(*small_target);

        buff_plan = buff_aimer.mpc_aim(*target_clone, t, gs, true);

        // 二次门控
        buff_plan.fire = auto_buff::fire_gate_check(buff_plan, *target, gs);

        gimbal.send(
            buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel,
            buff_plan.yaw_acc, buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);
      }

      if (power_runes.has_value())
        buff_lost_count = 0;
      else if (++buff_lost_count > 50)
        reset_buff();
    }
    // 自瞄
    else
    {
      auto armors = yolo.detect(img);
      auto targets = tracker.track(armors, t, true, self_world_xy); // 传入与图像同时刻的世界系x/y位置
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);
    }
    // else
    //    gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  }

  quit = true;
  if (plan_thread.joinable())
    plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
