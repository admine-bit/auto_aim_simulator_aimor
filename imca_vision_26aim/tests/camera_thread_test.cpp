#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>  // 用于时间戳处理

#include "io/camera.hpp"       // 摄像头接口
#include "io/gimbal/gimbal.hpp"// 云台接口
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/exiter.hpp"    // 用于安全退出

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }";

// 生成标定板三维坐标（世界坐标系下，单位：mm）
std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;
  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});
  return centers_3d;
}

// 实时采集数据（摄像头图像+云台四元数）
void collect_data(
  const std::string & config_path,
  std::vector<double> & R_gimbal2imubody_data,
  std::vector<cv::Mat> & R_gimbal2world_list,
  std::vector<cv::Mat> & t_gimbal2world_list,
  std::vector<cv::Mat> & rvecs,
  std::vector<cv::Mat> & tvecs)
{
  // 读取配置参数
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  // 初始化参数
  cv::Size pattern_size(pattern_cols, pattern_rows);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());
  cv::Matx33d camera_matrix(camera_matrix_data.data());
  cv::Mat distort_coeffs(distort_coeffs_data);

  // 初始化设备
  io::Camera camera(config_path);  // 摄像头
  io::Gimbal gimbal(config_path);  // 云台
  tools::Exiter exiter;            // 安全退出工具

  cv::Mat img;
  std::chrono::steady_clock::time_point img_time;  // 图像时间戳
  int collected_count = 0;                         // 已采集数据数量

  fmt::print("开始采集数据...\n");
  fmt::print("提示：\n");
  fmt::print("  1. 转动云台到不同姿态，确保标定板在视野内\n");
  fmt::print("  2. 按 's' 键保存当前帧数据（需标定板识别成功）\n");
  fmt::print("  3. 按 'q' 键结束采集并开始标定\n");

  while (!exiter.exit()) {
    // 1. 读取摄像头图像和时间戳
  camera.read(img, img_time);  // 直接调用read（无返回值）
  if (img.empty()) {  // 通过图像是否为空判断读取失败
  fmt::print("摄像头读取失败，重试...\n");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  continue;
  } 

    // 2. 获取与图像同步的云台四元数（关键：时间戳对齐）
    Eigen::Quaterniond q = gimbal.q(img_time);  // 用图像时间戳获取对应云台姿态

    // 3. 计算云台到世界坐标系的旋转矩阵
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;  // 弧度转角度

    // 4. 识别标定板
    std::vector<cv::Point2f> centers_2d;
    bool success = cv::findCirclesGrid(img, pattern_size, centers_2d);  // 识别圆点标定板

    // 5. 显示实时信息（图像+欧拉角+采集状态）
    cv::Mat drawing = img.clone();
    // 显示云台欧拉角
    tools::draw_text(drawing, fmt::format("yaw: {:.2f}°", ypr[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("pitch: {:.2f}°", ypr[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("roll: {:.2f}°", ypr[2]), {40, 120}, {0, 0, 255});
    // 显示采集状态
    tools::draw_text(drawing, fmt::format("已采集: {} 组", collected_count), {40, 160}, {0, 255, 0});
    // 绘制标定板识别结果
    cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
    // 显示提示
    tools::draw_text(drawing, "按 's' 保存，'q' 退出", {40, 200}, {255, 0, 0});

    cv::imshow("手眼标定采集", drawing);
    int key = cv::waitKey(1);

    // 6. 处理键盘输入
    if (key == 'q') {  // 结束采集
      if (collected_count < 5) {  // 至少需要5组数据（越多越准）
        fmt::print("数据不足！至少需要5组，当前仅{}组\n", collected_count);
        continue;
      }
      fmt::print("采集结束，共{}组数据，开始标定...\n", collected_count);
      break;
    } else if (key == 's') {  // 保存当前帧数据
      if (!success) {
        fmt::print("标定板识别失败，无法保存！\n");
        continue;
      }
      // 计算标定板在相机坐标系下的外参（rvec, tvec）
      cv::Mat rvec, tvec;
      auto centers_3d_ = centers_3d(pattern_size, center_distance_mm);
      cv::solvePnP(
        centers_3d_, centers_2d, camera_matrix, distort_coeffs,
        rvec, tvec, false, cv::SOLVEPNP_IPPE);

      // 保存数据到列表
      cv::Mat R_gimbal2world_cv;
      cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
      R_gimbal2world_list.push_back(R_gimbal2world_cv);
      t_gimbal2world_list.push_back((cv::Mat_<double>(3,1) << 0,0,0));  // 云台平移暂设为0（假设世界系原点在云台）
      rvecs.push_back(rvec);
      tvecs.push_back(tvec);

      collected_count++;
      fmt::print("已保存第{}组数据\n", collected_count);
    }
  }
  cv::destroyAllWindows();
}

// 输出标定结果为YAML
void print_yaml(
  const std::vector<double> & R_gimbal2imubody_data,
  const cv::Mat & R_camera2gimbal,
  const cv::Mat & t_camera2gimbal,
  const Eigen::Vector3d & ypr)
{
  YAML::Emitter result;
  std::vector<double> R_camera2gimbal_data(
    R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
  std::vector<double> t_camera2gimbal_data(
    t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

  result << YAML::BeginMap;
  result << YAML::Key << "R_gimbal2imubody";
  result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
  result << YAML::Newline;
  result << YAML::Comment(fmt::format(
    "相机同理想安装的偏角: yaw{:.2f}° pitch{:.2f}° roll{:.2f}°", ypr[0], ypr[1], ypr[2]));
  result << YAML::Key << "R_camera2gimbal";
  result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
  result << YAML::Key << "t_camera2gimbal";
  result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
  result << YAML::EndMap;

  fmt::print("\n标定结果:\n{}\n", result.c_str());
}

int main(int argc, char * argv[])
{
  // 解析命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>("config-path");

  // 实时采集标定数据
  std::vector<double> R_gimbal2imubody_data;
  std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
  std::vector<cv::Mat> rvecs, tvecs;
  collect_data(
    config_path, R_gimbal2imubody_data,
    R_gimbal2world_list, t_gimbal2world_list,
    rvecs, tvecs);

  // 执行手眼标定
  cv::Mat R_camera2gimbal, t_camera2gimbal;
  cv::calibrateHandEye(
    R_gimbal2world_list, t_gimbal2world_list,
    rvecs, tvecs,
    R_camera2gimbal, t_camera2gimbal);
  t_camera2gimbal /= 1e3;  // 毫米转米

  // 计算相机与理想安装的偏差（辅助调试）
  Eigen::Matrix3d R_camera2gimbal_eigen;
  cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
  Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};  // 理想安装姿态（根据实际定义调整）
  Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
  Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // 弧度转角度

  // 输出结果
  print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);
  return 0;
}