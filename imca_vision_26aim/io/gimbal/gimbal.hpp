#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{
class TalosTransport;

struct __attribute__((packed)) GimbalToVision
{
  uint8_t head = {0x5a};
  uint8_t mode;  // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符, 4: 远距离自瞄, 5: 前哨站
  float  q[4];
  float yaw;
  float yaw_vel;
  float pitch;
  // 云台安装点在上电时建立的固定世界系 W0 下的位置；不会随当前车头方向旋转
  float x;        // W0 的 x 位置 (m)，正方向为上电时正前方
  float y;        // W0 的 y 位置 (m)，正方向为上电时左方
  float z;        // 底盘离地高度 (m)
  uint8_t my_color; //1 red 0 blue
  uint8_t tail = {0xa5};

};

static_assert(sizeof(GimbalToVision) <= 64);

struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head = {0x5a};
  uint8_t mode;  // 0: 不控制, 1: 控制云台但不开火，2: 控制云台且开火
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  uint8_t tail = {0xa5};
};

static_assert(sizeof(VisionToGimbal) <= 64);

enum class GimbalMode
{
  IDLE = 0,                 // 空闲
  AUTO_AIM = 1,             // 自瞄
  SMALL_BUFF = 2,           // 小符
  BIG_BUFF = 3,             // 大符
  LONG = 4,                 // 远距离自瞄
  OUTPOST = 5               // 前哨站
};

struct GimbalState
{

  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  uint8_t my_color;
  float bullet_speed = 23.7;
  uint16_t bullet_count;
  float x = 0;           // 云台安装点在固定世界系 W0 下的 x 位置 (m)
  float y = 0;           // 云台安装点在固定世界系 W0 下的 y 位置 (m)
  float z_chassis = 0;    // 底盘离地高度 (m)
};

class Gimbal
{
public:
  
  Gimbal(const std::string & config_path);
  ~Gimbal();



  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);
  // 获取图像时间戳 t 对应的 W0 系绝对 x/y；与 q(t) 使用相同的串口接收时间戳插值
  Eigen::Vector2d xy(std::chrono::steady_clock::time_point t);

  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  void send(io::VisionToGimbal VisionToGimbal);

private:
  bool talos_backend_ = false;
  std::shared_ptr<TalosTransport> talos_;
  serial::Serial serial_;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  GimbalToVision rx_data_;
  VisionToGimbal tx_data_;

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};
  // 独立消费位置数据，避免改变既有四元数队列及 q(t) 的插值行为
  tools::ThreadSafeQueue<std::tuple<Eigen::Vector2d, std::chrono::steady_clock::time_point>>
    xy_queue_{1000};

  bool read(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();

  int i;
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
