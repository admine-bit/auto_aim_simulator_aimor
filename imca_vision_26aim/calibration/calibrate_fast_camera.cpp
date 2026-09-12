#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{threads t      | 16                         | 标定线程数 }"
  "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }";

namespace
{
struct ReprojectionCost
{
  ReprojectionCost(const cv::Point3f & point_3d, const cv::Point2f & point_2d)
  : x_(point_3d.x), y_(point_3d.y), z_(point_3d.z), u_(point_2d.x), v_(point_2d.y)
  {
  }

  template<typename T>
  bool operator()(
    const T * const intrinsics, const T * const distortion, const T * const pose,
    T * residuals) const
  {
    const T & fx = intrinsics[0];
    const T & fy = intrinsics[1];
    const T & cx = intrinsics[2];
    const T & cy = intrinsics[3];
    const T & k1 = distortion[0];
    const T & k2 = distortion[1];
    const T & p1 = distortion[2];
    const T & p2 = distortion[3];

    T p[3] = {T(x_), T(y_), T(z_)};
    T pc[3];
    ceres::AngleAxisRotatePoint(pose, p, pc);
    pc[0] += pose[3];
    pc[1] += pose[4];
    pc[2] += pose[5];

    const T xn = pc[0] / pc[2];
    const T yn = pc[1] / pc[2];
    const T r2 = xn * xn + yn * yn;
    const T r4 = r2 * r2;
    const T radial = T(1.0) + k1 * r2 + k2 * r4;
    const T x_tan = T(2.0) * p1 * xn * yn + p2 * (r2 + T(2.0) * xn * xn);
    const T y_tan = p1 * (r2 + T(2.0) * yn * yn) + T(2.0) * p2 * xn * yn;
    const T x_dist = xn * radial + x_tan;
    const T y_dist = yn * radial + y_tan;
    const T u_proj = fx * x_dist + cx;
    const T v_proj = fy * y_dist + cy;

    residuals[0] = u_proj - T(u_);
    residuals[1] = v_proj - T(v_);
    return true;
  }

  double x_, y_, z_;
  double u_, v_;
};

void calibrate_camera_ceres(
  const std::vector<std::vector<cv::Point3f>> & obj_points,
  const std::vector<std::vector<cv::Point2f>> & img_points, const cv::Size & img_size, int threads,
  cv::Mat & camera_matrix, cv::Mat & distort_coeffs, std::vector<cv::Mat> & rvecs,
  std::vector<cv::Mat> & tvecs)
{
  camera_matrix = cv::initCameraMatrix2D(obj_points, img_points, img_size);
  cv::Mat init_dist = cv::Mat::zeros(1, 5, CV_64F);

  double intrinsics[4] = {
    camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1), camera_matrix.at<double>(0, 2),
    camera_matrix.at<double>(1, 2)};
  double distortion[4] = {0.0, 0.0, 0.0, 0.0};

  std::vector<std::array<double, 6>> poses(obj_points.size(), {0, 0, 0, 0, 0, 0});
  for (size_t i = 0; i < obj_points.size(); i++) {
    cv::Mat rvec, tvec;
    bool ok = cv::solvePnP(
      obj_points[i], img_points[i], camera_matrix, init_dist, rvec, tvec, false,
      cv::SOLVEPNP_ITERATIVE);
    if (!ok) continue;
    poses[i][0] = rvec.at<double>(0);
    poses[i][1] = rvec.at<double>(1);
    poses[i][2] = rvec.at<double>(2);
    poses[i][3] = tvec.at<double>(0);
    poses[i][4] = tvec.at<double>(1);
    poses[i][5] = tvec.at<double>(2);
  }

  ceres::Problem problem;
  for (size_t i = 0; i < obj_points.size(); i++) {
    for (size_t j = 0; j < obj_points[i].size(); j++) {
      auto * cost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 4, 4, 6>(
        new ReprojectionCost(obj_points[i][j], img_points[i][j]));
      problem.AddResidualBlock(cost, nullptr, intrinsics, distortion, poses[i].data());
    }
  }

  problem.SetParameterLowerBound(intrinsics, 0, 1e-6);
  problem.SetParameterLowerBound(intrinsics, 1, 1e-6);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.max_num_iterations = 1;
  options.num_threads = std::max(1, threads);
  options.minimizer_progress_to_stdout = false;

  std::array<double, 4> best_intrinsics{intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]};
  std::array<double, 4> best_distortion{distortion[0], distortion[1], distortion[2], distortion[3]};
  std::vector<std::array<double, 6>> best_poses = poses;
  double best_cost = std::numeric_limits<double>::infinity();
  int best_iteration = -1;
  double initial_cost = 0.0;
  double total_time = 0.0;
  int total_inner_iterations = 0;

  fmt::print(
    "Ceres start: images={}, residuals={}, threads={}, max_iterations={}\n", obj_points.size(),
    problem.NumResiduals(), options.num_threads, 100);
  for (int outer_iter = 0; outer_iter < 100; outer_iter++) {
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (outer_iter == 0) initial_cost = summary.initial_cost;
    total_time += summary.total_time_in_seconds;
    total_inner_iterations += static_cast<int>(summary.iterations.size());

    double current_cost = summary.final_cost;
    if (current_cost < best_cost) {
      best_cost = current_cost;
      best_iteration = outer_iter;
      std::copy_n(intrinsics, 4, best_intrinsics.begin());
      std::copy_n(distortion, 4, best_distortion.begin());
      best_poses = poses;
    }

    fmt::print(
      "outer_iter={:3d}/100  inner_iters={}  cost={:.12f}  best_cost={:.12f}  termination={}\n",
      outer_iter + 1, summary.iterations.size(), current_cost, best_cost,
      summary.termination_type == ceres::CONVERGENCE ? "CONVERGENCE" :
      summary.termination_type == ceres::NO_CONVERGENCE ? "NO_CONVERGENCE" :
      summary.termination_type == ceres::FAILURE ? "FAILURE" :
      summary.termination_type == ceres::USER_SUCCESS ? "USER_SUCCESS" :
      summary.termination_type == ceres::USER_FAILURE ? "USER_FAILURE" : "UNKNOWN");
  }

  std::copy(best_intrinsics.begin(), best_intrinsics.end(), intrinsics);
  std::copy(best_distortion.begin(), best_distortion.end(), distortion);
  poses = best_poses;
  fmt::print(
    "Ceres done: residuals={}, outer_iterations=100, inner_iterations={}, time={:.3f}s, initial_cost={:.6f}, final_cost={:.6f}, best_iteration={}, best_cost={:.6f}\n",
    problem.NumResiduals(), total_inner_iterations, total_time, initial_cost, best_cost,
    best_iteration + 1, best_cost);

  camera_matrix = (cv::Mat_<double>(3, 3) << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1],
                   intrinsics[3], 0, 0, 1);
  distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  distort_coeffs.at<double>(0, 0) = distortion[0];
  distort_coeffs.at<double>(0, 1) = distortion[1];
  distort_coeffs.at<double>(0, 2) = distortion[2];
  distort_coeffs.at<double>(0, 3) = distortion[3];
  distort_coeffs.at<double>(0, 4) = 0.0;  // keep k3 fixed

  rvecs.clear();
  tvecs.clear();
  rvecs.reserve(poses.size());
  tvecs.reserve(poses.size());
  for (const auto & pose : poses) {
    rvecs.emplace_back((cv::Mat_<double>(3, 1) << pose[0], pose[1], pose[2]));
    tvecs.emplace_back((cv::Mat_<double>(3, 1) << pose[3], pose[4], pose[5]));
  }
}
}  // namespace

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;

  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});

  return centers_3d;
}

void load(
  const std::string & input_folder, const std::string & config_path, cv::Size & img_size,
  std::vector<std::vector<cv::Point3f>> & obj_points,
  std::vector<std::vector<cv::Point2f>> & img_points)
{
  // 读取yaml参数
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  cv::Size pattern_size(pattern_cols, pattern_rows);

  for (int i = 1; true; i++) {
    // 读取图片
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto img = cv::imread(img_path);
    if (img.empty()) break;

    // 设置图片尺寸
    img_size = img.size();

    // 识别标定板
    std::vector<cv::Point2f> centers_2d;
    auto success = cv::findCirclesGrid(img, pattern_size, centers_2d, cv::CALIB_CB_SYMMETRIC_GRID);

    // 显示识别结果
    auto drawing = img.clone();
    cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
    cv::resize(drawing, drawing, {}, 0.5, 0.5);  // 缩小图片尺寸便于显示完全
    cv::imshow("Press any to continue", drawing);
    cv::waitKey(0);

    // 输出识别结果
    fmt::print("[{}] {}\n", success ? "success" : "failure", img_path);
    if (!success) continue;

    // 记录所需的数据
    img_points.emplace_back(centers_2d);
    obj_points.emplace_back(centers_3d(pattern_size, center_distance_mm));
  }
}

void print_yaml(const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs, double error)
{
  YAML::Emitter result;
  std::vector<double> camera_matrix_data(
    camera_matrix.begin<double>(), camera_matrix.end<double>());
  std::vector<double> distort_coeffs_data(
    distort_coeffs.begin<double>(), distort_coeffs.end<double>());

  result << YAML::BeginMap;
  result << YAML::Comment(fmt::format("重投影误差: {:.4f}px", error));
  result << YAML::Key << "camera_matrix";
  result << YAML::Value << YAML::Flow << camera_matrix_data;
  result << YAML::Key << "distort_coeffs";
  result << YAML::Value << YAML::Flow << distort_coeffs_data;
  result << YAML::Newline;
  result << YAML::EndMap;

  fmt::print("\n{}\n", result.c_str());
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_folder = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto worker_count = cli.get<int>("threads");

  // 从输入文件夹中加载标定所需的数据
  cv::Size img_size;
  std::vector<std::vector<cv::Point3f>> obj_points;
  std::vector<std::vector<cv::Point2f>> img_points;
  load(input_folder, config_path, img_size, obj_points, img_points);
  if (obj_points.empty()) {
    fmt::print("No valid calibration samples detected.\n");
    return 1;
  }

  // 相机标定
  cv::Mat camera_matrix, distort_coeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  calibrate_camera_ceres(
    obj_points, img_points, img_size, worker_count, camera_matrix, distort_coeffs, rvecs, tvecs);

  // 重投影误差
  double error_sum = 0;
  size_t total_points = 0;
  for (size_t i = 0; i < obj_points.size(); i++) {
    std::vector<cv::Point2f> reprojected_points;
    cv::projectPoints(
      obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected_points);

    total_points += reprojected_points.size();
    for (size_t j = 0; j < reprojected_points.size(); j++)
      error_sum += cv::norm(img_points[i][j] - reprojected_points[j]);
  }
  auto error = error_sum / total_points;

  // 输出yaml
  print_yaml(camera_matrix, distort_coeffs, error);
}
