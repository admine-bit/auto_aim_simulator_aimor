#include "talos.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "tools/logger.hpp"

namespace io
{
namespace
{
constexpr uint32_t SHM_MAGIC = 0x54414C05;
constexpr uint32_t SHM_VERSION = 2;
constexpr uint8_t FLAG_NEW = 0x80;
constexpr uint8_t INDEX_MASK = 0x03;
constexpr uint32_t IMAGE_WIDTH = 1440;
constexpr uint32_t IMAGE_HEIGHT = 1080;
constexpr uint32_t IMAGE_CHANNELS = 3;
constexpr std::size_t IMAGE_SIZE =
  static_cast<std::size_t>(IMAGE_WIDTH) * IMAGE_HEIGHT * IMAGE_CHANNELS;
constexpr std::size_t IMAGE_POOL_SIZE = IMAGE_SIZE * 3;
constexpr std::size_t META_SIZE = 3712;

struct alignas(32) ImageMeta
{
  uint64_t seq;
  uint64_t timestamp_ns;
  uint32_t width;
  uint32_t height;
  uint8_t buffer_id;
  uint8_t format;
  uint8_t pad[6];
};

struct alignas(64) PoseMeta
{
  uint64_t frame_seq;
  float position[3];
  float quaternion[4];
  uint64_t timestamp_ns;
  uint8_t pad[16];
};

struct alignas(32) GimbalCmd
{
  uint64_t timestamp_ns;
  float yaw_deg;
  float pitch_deg;
  float distance_m;
  uint8_t fire_advice;
  uint8_t pad[11];
};

struct alignas(64) CameraInfo
{
  uint64_t timestamp_ns;
  double fx;
  double fy;
  double cx;
  double cy;
  double distortion[5];
  uint32_t width;
  uint32_t height;
  uint8_t pad[24];
};

struct alignas(64) ChassisObservation
{
  uint64_t frame_seq;
  uint64_t timestamp_ns;
  float dt_s;
  float v_body[2];
  float wz_radps;
  float wheel_linear_mps[4];
  float wheel_angular_radps[4];
  float a_body[2];
  float alpha_z_radps2;
  float rpy_rad[3];
  float gyro_xyz_radps[3];
  float accel_xyz_mps2[3];
  uint8_t pad[16];
};

struct alignas(64) RuntimeState
{
  uint64_t timestamp_ns;
  uint8_t following;
  uint8_t pad[55];
};

template <typename Meta, std::size_t Count>
struct alignas(64) TripleBuffer
{
  uint8_t state;
  uint8_t write_idx;
  uint8_t read_idx;
  uint8_t pad[61];
  Meta slots[Count];
};

struct alignas(64) ShmHeader
{
  uint32_t magic;
  uint32_t version;
  uint64_t created_ns;
  uint64_t heartbeat_ns;
  uint32_t image_width;
  uint32_t image_height;
  uint8_t pad[32];
};

struct alignas(64) ShmMetaRegion
{
  ShmHeader header;
  TripleBuffer<ImageMeta, 3> image;
  TripleBuffer<PoseMeta, 3> poses[5];
  TripleBuffer<GimbalCmd, 3> gimbal_cmd;
  CameraInfo camera_info;
  ChassisObservation chassis_observation;
  uint8_t ground_truth[1664];
  RuntimeState runtime_state;
};

static_assert(sizeof(ImageMeta) == 32);
static_assert(sizeof(PoseMeta) == 64);
static_assert(sizeof(GimbalCmd) == 32);
static_assert(sizeof(CameraInfo) == 128);
static_assert(sizeof(ChassisObservation) == 128);
static_assert(sizeof(TripleBuffer<ImageMeta, 3>) == 192);
static_assert(sizeof(TripleBuffer<PoseMeta, 3>) == 256);
static_assert(sizeof(TripleBuffer<GimbalCmd, 3>) == 192);
static_assert(offsetof(ShmMetaRegion, camera_info) == 1728);
static_assert(offsetof(ShmMetaRegion, chassis_observation) == 1856);
static_assert(offsetof(ShmMetaRegion, runtime_state) == 3648);
static_assert(sizeof(ShmMetaRegion) == META_SIZE);

struct PoseIndex
{
  static constexpr std::size_t gimbal = 0;
  static constexpr std::size_t odom = 1;
  static constexpr std::size_t muzzle = 2;
  static constexpr std::size_t camera = 3;
};

uint64_t now_ns()
{
  using namespace std::chrono;
  return static_cast<uint64_t>(
    duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count());
}

template <typename Meta, std::size_t Count>
std::optional<Meta> consume(TripleBuffer<Meta, Count> & buffer)
{
  std::atomic_ref<uint8_t> state(buffer.state);
  for (int attempt = 0; attempt < 2; ++attempt) {
    uint8_t expected = state.load(std::memory_order_acquire);
    if ((expected & FLAG_NEW) == 0) return std::nullopt;

    const uint8_t ready_idx = expected & INDEX_MASK;
    const uint8_t desired = buffer.read_idx;
    if (state.compare_exchange_weak(
        expected, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      buffer.read_idx = ready_idx;
      return buffer.slots[ready_idx];
    }
  }
  return std::nullopt;
}

}  // namespace

struct TalosTransport::Mapping
{
  int fd = -1;
  void * data = MAP_FAILED;
  std::size_t size = 0;

  Mapping(const char * path, std::size_t required_size)
  {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
      fd = ::open(path, O_RDWR | O_CLOEXEC);
      if (fd >= 0) {
        struct stat info{};
        if (::fstat(fd, &info) == 0 && static_cast<std::size_t>(info.st_size) >= required_size) {
          data = ::mmap(nullptr, required_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
          if (data != MAP_FAILED) {
            size = required_size;
            return;
          }
        }
        ::close(fd);
        fd = -1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    throw std::runtime_error(
      std::string("Timed out waiting for Talos shared memory: ") + path);
  }

  ~Mapping()
  {
    if (data != MAP_FAILED) ::munmap(data, size);
    if (fd >= 0) ::close(fd);
  }
};

std::shared_ptr<TalosTransport> TalosTransport::shared()
{
  static std::mutex mutex;
  static std::weak_ptr<TalosTransport> cached;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto transport = cached.lock()) return transport;
  auto transport = std::make_shared<TalosTransport>();
  cached = transport;
  return transport;
}

TalosTransport::TalosTransport()
: meta_(std::make_unique<Mapping>("/tmp/talos_ipc_meta", META_SIZE)),
  image_pool_(std::make_unique<Mapping>("/tmp/talos_ipc_image_pool", IMAGE_POOL_SIZE)),
  gimbal_to_world_(Eigen::Quaterniond::Identity())
{
  const auto * meta = static_cast<const ShmMetaRegion *>(meta_->data);
  if (meta->header.magic != SHM_MAGIC || meta->header.version != SHM_VERSION ||
      meta->header.image_width != IMAGE_WIDTH || meta->header.image_height != IMAGE_HEIGHT) {
    throw std::runtime_error("Invalid Talos shared memory header");
  }
  state_.my_color = 1;
  state_.bullet_speed = 25.0;
  tools::logger()->info("[Talos] Connected to simulator shared memory");
}

TalosTransport::~TalosTransport() = default;

void TalosTransport::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  while (true) {
    auto * meta = static_cast<ShmMetaRegion *>(meta_->data);
    auto frame = consume(meta->image);
    if (!frame.has_value()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    if (frame->buffer_id >= 3 || frame->width != IMAGE_WIDTH || frame->height != IMAGE_HEIGHT) {
      throw std::runtime_error("Invalid Talos image metadata");
    }

    if (auto pose = consume(meta->poses[PoseIndex::gimbal]); pose.has_value()) {
      std::lock_guard<std::mutex> lock(mutex_);
      gimbal_to_world_ = Eigen::Quaterniond(
        pose->quaternion[0], pose->quaternion[1], pose->quaternion[2], pose->quaternion[3])
        .normalized();
    }
    if (auto pose = consume(meta->poses[PoseIndex::odom]); pose.has_value()) {
      std::lock_guard<std::mutex> lock(mutex_);
      state_.x = pose->position[0];
      state_.y = pose->position[1];
      state_.z_chassis = pose->position[2];
    }
    // Talos 的同步发布器会等待 image、gimbal、odom、muzzle、camera 五个
    // triple-buffer 都被消费后才发布下一帧。视觉侧暂时不使用 muzzle/camera
    // 位姿，但必须消费它们，否则读图会在第一帧后永久等待。
    (void)consume(meta->poses[PoseIndex::muzzle]);
    (void)consume(meta->poses[PoseIndex::camera]);

    const auto * rgb = static_cast<const uint8_t *>(image_pool_->data) +
      static_cast<std::size_t>(frame->buffer_id) * IMAGE_SIZE;
    cv::Mat rgb_view(static_cast<int>(IMAGE_HEIGHT), static_cast<int>(IMAGE_WIDTH), CV_8UC3,
      const_cast<uint8_t *>(rgb));
    cv::cvtColor(rgb_view, img, cv::COLOR_RGB2BGR);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      frame_seq_ = frame->seq;
      const auto ypr = gimbal_to_world_.toRotationMatrix().eulerAngles(2, 1, 0);
      state_.yaw = ypr[0];
      state_.pitch = ypr[1];
    }
    timestamp = std::chrono::steady_clock::now();
    return;
  }
}

Eigen::Quaterniond TalosTransport::gimbal_q() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return gimbal_to_world_;
}

TalosState TalosTransport::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

void TalosTransport::send(
  bool control, bool fire, double yaw, double yaw_vel, double yaw_acc, double pitch,
  double pitch_vel, double pitch_acc)
{
  (void)yaw_vel;
  (void)yaw_acc;
  (void)pitch_vel;
  (void)pitch_acc;

  std::lock_guard<std::mutex> lock(mutex_);
  auto * meta = static_cast<ShmMetaRegion *>(meta_->data);
  auto & buffer = meta->gimbal_cmd;
  const auto write_idx = buffer.write_idx;
  auto & command = buffer.slots[write_idx];
  command.timestamp_ns = now_ns();
  command.yaw_deg = static_cast<float>(yaw * 180.0 / M_PI);
  // Daedalus interprets Talos pitch as an angle from the vertical and negates it
  // before converting to the local gimbal pitch. Source pitch is horizontal=0,
  // upward<0, hence -(90 + pitch).
  command.pitch_deg = static_cast<float>(-(90.0 + pitch * 180.0 / M_PI));
  command.distance_m = control ? 1.0F : -1.0F;
  command.fire_advice = static_cast<uint8_t>(fire ? 1 : 0);

  std::atomic_ref<uint8_t> state(buffer.state);
  const auto old = state.exchange(static_cast<uint8_t>(write_idx | FLAG_NEW),
    std::memory_order_acq_rel);
  buffer.write_idx = old & INDEX_MASK;
}

TalosCamera::TalosCamera(std::shared_ptr<TalosTransport> transport)
: transport_(std::move(transport))
{
}

void TalosCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  transport_->read(img, timestamp);
}

}  // namespace io
