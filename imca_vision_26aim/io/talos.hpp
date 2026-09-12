#ifndef IO__TALOS_HPP
#define IO__TALOS_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"

namespace io
{

struct TalosState
{
  double yaw = 0.0;
  double yaw_vel = 0.0;
  double pitch = 0.0;
  double pitch_vel = 0.0;
  uint8_t my_color = 1;  // 仿真己方为红色，敌方为蓝色
  double bullet_speed = 25.0;
  uint16_t bullet_count = 0;
  double x = 0.0;
  double y = 0.0;
  double z_chassis = 0.0;
};

// Talos IPC 的单进程客户端。Camera 和 Gimbal 共享同一个实例，保证每个
// triple-buffer 只有一个消费者，符合 simulator 的 IPC 协议约束。
class TalosTransport
{
public:
  static std::shared_ptr<TalosTransport> shared();

  TalosTransport();
  ~TalosTransport();

  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
  Eigen::Quaterniond gimbal_q() const;
  TalosState state() const;

  void send(bool control, bool fire, double yaw, double yaw_vel, double yaw_acc, double pitch,
    double pitch_vel, double pitch_acc);

private:
  struct Mapping;
  std::unique_ptr<Mapping> meta_;
  std::unique_ptr<Mapping> image_pool_;

  mutable std::mutex mutex_;
  Eigen::Quaterniond gimbal_to_world_;
  TalosState state_;
  uint64_t frame_seq_ = 0;
};

class TalosCamera final : public CameraBase
{
public:
  explicit TalosCamera(std::shared_ptr<TalosTransport> transport);
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;

private:
  std::shared_ptr<TalosTransport> transport_;
};

}  // namespace io

#endif  // IO__TALOS_HPP
