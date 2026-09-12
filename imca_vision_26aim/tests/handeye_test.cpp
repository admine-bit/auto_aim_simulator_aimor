
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/core/eigen.hpp>

#include "io/camera.hpp"

#include "io/gimbal/gimbal.hpp"

#include "io/cboard.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{d display      |                     | 显示视频流       }";

// 世界坐标到像素坐标的转换

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  auto config_path = cli.get<std::string>("config-path");
  auto display = cli.has("display");
  auto yaml = YAML::LoadFile(config_path);
  // auto height = yaml["height"].as<double>();
  // auto grid_num = yaml["grid_num"].as<int>();
  // auto grid_size = yaml["grid_size"].as<double>();
  // auto delay = yaml["delay"].as<int>();

  auto height = 0.2;
  auto grid_num = 10;
  auto grid_size = 5.0;
  auto delay = 0;

  // io::CBoard cboard(config_path);
  io::Gimbal gimbal(config_path);// 初始化串口通信（传入配置文件路径）
  io::Camera camera(config_path);
  auto_aim::Solver solver(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;
  std::vector<cv::Point3f> points;
  for (int x = 0; x < grid_num; x++) 
  {
    for (int y = 0; y < grid_num; y++) 
    {
      points.emplace_back(x * grid_size, y * grid_size - grid_num * grid_size / 2, -height);
      points.emplace_back(-x * grid_size, y * grid_size - grid_num * grid_size / 2, -height);
    }
  }


  while (!exiter.exit()) {
    camera.read(img, t);
    // q = cboard.imu_at(t - 1ms * delay);
    auto q = gimbal.q(t);// 获取该时间戳对应的云台姿态四元数- 1ms
    fmt::print("q = [{:.3f}, {:.3f}, {:.3f}, {:.3f}]\n", q.w(), q.x(), q.y(), q.z());
    solver.set_R_gimbal2world(q);
    cv::Mat result = img.clone();
    std::vector<cv::Point2f> projectedPoints = solver.world2pixel(points);
    for (const auto & point : projectedPoints) 
    {
      if (point.x >= 0 && point.x < result.cols && point.y >= 0 && point.y < result.rows) 
      {
        cv::circle(result, point, 3, cv::Scalar(255, 255, 255), -1);
      }
    }
    Eigen::Vector3d euler = solver.R_gimbal2world().eulerAngles(2, 1, 0) * 180.0 / M_PI;




    // // ========== 修正 yaw 到 -180°~180° ==========
    // if (euler[0] > 180.0) {
    //     euler[0] -= 360.0;
    // } else if (euler[0] < -180.0) {
    //     euler[0] += 360.0;
    // }
    // // ========== 修正 pitch 到 -90°~90°（解决半圈跳变） ==========
    // if (euler[1] > 90.0) {
    //     euler[1] = 180.0 - euler[1];
    //     euler[0] += 180.0;  // pitch 翻转时，yaw 需同步调整180°
    //     euler[0] = fmod(euler[0] + 180.0, 360.0) - 180.0;  // 保持 yaw 在 -180°~180°
    // } else if (euler[1] < -90.0) {
    //     euler[1] = -180.0 - euler[1];
    //     euler[0] += 180.0;
    //     euler[0] = fmod(euler[0] + 180.0, 360.0) - 180.0;
    // }


    tools::draw_text(result, fmt::format("yaw   {:.2f}", euler[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(result, fmt::format("pitch {:.2f}", euler[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(result, fmt::format("roll  {:.2f}", euler[2]), {40, 120}, {0, 0, 255});
    if (!display) continue;
    cv::imshow("result", result);
    if (cv::waitKey(1) == 'q') break;
  }
}