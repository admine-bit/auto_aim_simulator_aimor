#include <fmt/core.h>// 包含fmt库，用于格式化输出

// 包含原子操作、chrono时间库、JSON处理、OpenCV、线程库等标准库
#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <thread>

// 包含项目内部的IO模块（相机、云台）、自瞄任务模块（规划器、解算器、追踪器、YOLO识别）及工具类
#include "io/hikrobot/hikrobot.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"  
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"


#include "tasks/auto_aim/target.hpp"

using namespace std::chrono_literals;// 使用chrono_literals命名空间，方便使用ms等时间字面量

// 定义命令行参数键值，包含帮助信息和配置文件路径参数
const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/uav.yaml | 位置参数，yaml配置文件路径 }";

struct HikRobotConfig
{
  double exposure_ms;
  double gain;
  std::string vid_pid;
  std::string serial;
};

HikRobotConfig read_hikrobot_config(
  const YAML::Node & yaml, const std::string & suffix = "")
{
  auto key = [&suffix](const std::string & name) { return name + suffix; };
  auto name = tools::read<std::string>(yaml, key("camera_name"));
  auto serial = tools::read<std::string>(yaml, key("camera_serial"));
  if (name != "hikrobot") throw std::runtime_error("UAV debug requires HikRobot cameras");
  if (serial.empty() || serial.find("REPLACE_") == 0)
    throw std::runtime_error("UAV debug camera serial is not configured");
  return {
    tools::read<double>(yaml, key("exposure_ms")),
    tools::read<double>(yaml, key("gain")),
    tools::read<std::string>(yaml, key("vid_pid")), serial};
}

int main(int argc, char * argv[])
{
  //FPS
  int frame_count = 0;
  auto last_time = std::chrono::steady_clock::now();
  double fps = 0.0;
  float min_dist = 0;

  tools::Exiter exiter;// 初始化程序退出管理器（用于处理程序退出逻辑）
  tools::Plotter plotter;// 初始化绘图工具（用于绘制调试数据曲线）

  cv::CommandLineParser cli(argc, argv, keys);// 解析命令行参数
  auto config_path = cli.get<std::string>(0);// 获取配置文件路径（默认configs/demo.yaml）

  // 如果请求帮助或配置文件路径为空，输出帮助信息并退出
  if (cli.has("help") || config_path.empty()) 
  {
    cli.printMessage();
    return 0;
  }
  auto yaml = tools::load(config_path);  // 加载YAML配置文件
  auto fire_dis_ = tools::read<double>(yaml, "fire_dis");  // 读取开火判断的位置误差阈值

  // 远距镜头（LONG 模式）相对近距的弹道偏置增量（degree→rad）。
  // 不修改共用 Planner：仅本文件在发送前对 plan 叠加增量，其他兵种零影响。
  const double base_yaw_offset = tools::read<double>(yaml, "yaw_offset");
  const double base_pitch_offset = tools::read<double>(yaml, "pitch_offset");
  const double long_yaw_offset_delta =
    ((yaml["yaw_offset_long"] ? yaml["yaw_offset_long"].as<double>() : base_yaw_offset) -
     base_yaw_offset) /
    57.325;
  const double long_pitch_offset_delta =
    ((yaml["pitch_offset_long"] ? yaml["pitch_offset_long"].as<double>() : base_pitch_offset) -
     base_pitch_offset) /
    57.325;
  tools::logger()->info("[auto_aim_debug_mpc] fire_dis: {}", fire_dis_); 

  io::Gimbal gimbal(config_path);// 初始化串口通信（传入配置文件路径）

  // 与 uav 使用同一套双 HikRobot 相机，只有当前模式对应的相机保持采集。
  auto camera_config = read_hikrobot_config(yaml);
  auto camera_config_long = read_hikrobot_config(yaml, "_long");
  if (camera_config.serial == camera_config_long.serial)
    throw std::runtime_error("UAV debug camera serials must be different");
  io::HikRobot camera(
    camera_config.exposure_ms, camera_config.gain, camera_config.vid_pid,
    camera_config.serial, false);
  io::HikRobot camera_long(
    camera_config_long.exposure_ms, camera_config_long.gain, camera_config_long.vid_pid,
    camera_config_long.serial, false);
  camera.start();

  auto_aim::YOLO yolo(config_path, true);// 初始化YOLO识别器（传入配置文件路径，true表示使用某种模式）
  auto_aim::Solver solver(config_path);// 初始化解算器（处理坐标变换等）
  auto_aim::Solver solver_long(config_path, "_long");// 远距相机使用独立标定参数
  auto_aim::Tracker tracker(config_path, solver);// 初始化追踪器（追踪装甲板目标，关联解算器）
  auto_aim::Tracker tracker_long(config_path, solver_long);// 远距相机使用独立追踪器状态
  auto_aim::Planner planner(config_path);// 初始化规划器（计算瞄准位置和开火决策）

  // 创建线程安全队列，用于传递目标信息（容量为1，覆盖模式）
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);// 初始推入空目标

  std::atomic<bool> quit = false;// 原子变量，用于控制规划线程退出
  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode = io::GimbalMode::IDLE;

  // 启动规划线程
  auto plan_thread = std::thread([&]() 
  {
    auto t0 = std::chrono::steady_clock::now(); // 记录线程启动时间（用于时间戳计算）
    uint16_t last_bullet_count = 0;// 记录上一次子弹数量（用于检测是否开火）


    // 线程循环，直到收到退出信号
    while (!quit) 
    {
      auto target = target_queue.front();// 从队列获取目标信息
      auto gs = gimbal.state();// 获取当前云台状态（包含角度、速度、子弹数量等）
      auto plan = planner.plan(target, gs.bullet_speed); // 调用规划器计算规划结果（传入目标和子弹速度）
      if (mode.load() == io::GimbalMode::LONG) {
        // 远距镜头独立弹道偏置：在共用 Planner 输出上叠加增量（yaw_offset_long/pitch_offset_long）
        plan.yaw = tools::limit_rad(plan.yaw + long_yaw_offset_delta);
        plan.pitch -= long_pitch_offset_delta;
      }

      plan.yaw = - plan.yaw;
      plan.yaw_vel = - plan.yaw_vel;
      plan.yaw_acc = - plan.yaw_acc;
      
      if(min_dist<1.1)
      {
        plan.fire = (plan.fire == 1 || ( (abs(gs.yaw -plan.yaw) <  fire_dis_/2.0/min_dist)  && (abs(gs.pitch -plan.pitch) < fire_dis_/2.0/min_dist) ));
      }
      else
      {
        plan.fire = (plan.fire == 1 && ( (abs(gs.yaw -plan.yaw) <  fire_dis_/min_dist)  && (abs(gs.pitch -plan.pitch) < fire_dis_/min_dist) ));
      }

      gimbal.send(
        plan.control, plan.fire, plan.yaw , plan.yaw_vel, plan.yaw_acc, plan.pitch , plan.pitch_vel,
        plan.pitch_acc);

      // 判断是否有子弹射出（当前子弹数大于上一次记录）
      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count; // 判断是否有子弹射出（当前子弹数大于上一次记录）

      nlohmann::json data;// 构建JSON对象，存储调试数据
      data["1t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);// 记录当前时间（相对于线程启动时间）

      // 记录云台当前角度和速度
      data["2gimbal_yaw"] = gs.yaw*57.3;
      data["3gimbal_yaw_vel"] = gs.yaw_vel;
      data["4gimbal_pitch"] = gs.pitch *57.325;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

      // 记录目标的期望角度
      data["5real_yaw"] =auto_aim::target_yaw*57.3;//auto_aim::target_yaw *57.325
      data["6target_yaw"] =plan.target_yaw*57.3;//auto_aim::target_yaw *57.325
      data["7real_pitch"] = auto_aim::target_pitch*57.325;//plan.target_pitch
      data["8target_pitch"] = plan.target_pitch*57.3;//plan.target_pitch

      // 记录规划的角度、速度、加速度
      data["9plan_yaw"] = plan.yaw*57.325;
      data["10plan_yaw_vel"] = plan.yaw_vel *57.325;
      data["11plan_yaw_acc"] = plan.yaw_acc;

      data["12plan_pitch"] = plan.pitch *57.325;
      data["13plan_pitch_vel"] = plan.pitch_vel*57.325;
      data["14plan_pitch_acc"] = plan.pitch_acc;

      // 记录开火指令和实际开火状态（1表示真，0表示假）
      data["15fire"] = (plan.fire ? 1 : 0 );

      data["16dist"] = min_dist;
      data["17kaihuo"] = abs(gs.yaw -plan.yaw)*min_dist;
  

      // 如果有目标，记录目标的z坐标和z方向速度
      if (target.has_value()) 
      {
        data["18target_z"] = target->ekf_x()[4];   //z
        data["19target_vz"] = target->ekf_x()[5];  //vz

        data["20target_x"] = target->ekf_x()[0];   //z
        data["21target_vx"] = target->ekf_x()[1];  //vz
      }

      // 如果有目标，记录目标的旋转角速度；否则为0
      if (target.has_value()) 
      {
        data["22w"] = target->ekf_x()[7] * 57.325;
      } 
      else 
      {
        data["22w"] = 0.0;
      }

      plotter.plot(data);// 将数据发送到绘图工具

      std::this_thread::sleep_for(8ms);// 线程休眠10毫秒，控制循环频率
    }
  });
  // 声明图像矩阵和时间戳变量
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  // 主循环，直到收到退出信号（如Exiter检测到退出或按键q）
  while (!exiter.exit())
  {
    mode = gimbal.mode();
    const auto new_mode = mode.load();
    if (last_mode != new_mode) {
      if (new_mode == io::GimbalMode::LONG) {
        camera.stop();
        camera_long.start();
      } else if (last_mode == io::GimbalMode::LONG) {
        camera_long.stop();
        camera.start();
      }
      tools::logger()->info("UAV debug switch to {}", gimbal.str(new_mode));
      last_mode = new_mode;
    }

    const bool use_camera_long = mode.load() == io::GimbalMode::LONG;
    auto & camera_active = use_camera_long ? camera_long : camera;
    auto & solver_active = use_camera_long ? solver_long : solver;
    auto & tracker_active = use_camera_long ? tracker_long : tracker;

    camera_active.read(img, t);
    auto q = gimbal.q(t);// 获取该时间戳对应的云台姿态四元数- 1ms
    // auto self_world_xy = gimbal.xy(t);// 获取与姿态同一时刻的 W0 系绝对x/y位置
    auto self_world_xy = Eigen::Vector2d::Zero();
    auto gs = gimbal.state();// 获取当前云台状态（角度、子弹信息、底盘高度等）


    solver_active.set_R_gimbal2world(q);
    solver_active.set_t_gimbal2world(gs.z_chassis);
    auto armors = yolo.detect(img);// 使用YOLO识别器检测图像中的装甲板
    auto targets = tracker_active.track(armors, t, true, self_world_xy);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

        // 如果有目标，推入队列；否则推入空
    if (!targets.empty())
    {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      int i = 0;
      for (const Eigen::Vector4d & xyza : armor_xyza_list) 
      {
        auto image_points =
        solver_active.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});

        auto info = "id = " + std::to_string(i++);
        tools::draw_text(img,info,image_points[3]+cv::Point2f(0,20),{255,255,0});
      }
      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver_active.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
      min_dist = std::max( std::hypot( aim_xyza.head(3)[0], aim_xyza.head(3)[1] ) , 0.05 );  // 最小距离下限
  
    }
    
    // 计算并显示FPS
    frame_count++;
    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = current_time - last_time;
    if (elapsed.count() >= 1.0) {  // 每秒更新一次FPS
        fps = frame_count / elapsed.count();
        frame_count = 0;
        last_time = current_time;
    }
    // 在图像上显示FPS
    std::string fps_text = fmt::format("FPS: {:.2f}", fps);
    cv::putText(img, fps_text, cv::Point(10, 30), 
    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                

    
    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);// 等待1毫秒按键输入，若按下q则退出循环
    if (key == 'q') break;
  }

  quit = true;// 发送退出信号给规划线程
  if (plan_thread.joinable()) plan_thread.join();// 等待规划线程结束
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);// 发送停止指令给云台（关闭控制和开火）
  camera.stop();
  camera_long.stop();

  return 0;
}



