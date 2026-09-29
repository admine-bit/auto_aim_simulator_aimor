#include "MvCameraControl.h"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

// MVS API surface used by the unmodified imca_vision_26 camera class.
// The Talos frame is RGB8. Expose it as BayerRG8 because that class always
// demosaics a raw MVS frame before passing it to the detector.
namespace {
constexpr size_t kMetaSize = 3712;
constexpr size_t kWidth = 1440;
constexpr size_t kHeight = 1080;
constexpr size_t kPixels = kWidth * kHeight;
constexpr size_t kPoolSize = kPixels * 3 * 3;
constexpr uint8_t kNew = 0x80;
constexpr uint8_t kIndex = 0x03;

struct Camera {
  int meta_fd = -1;
  int pool_fd = -1;
  uint8_t *meta = nullptr;
  const uint8_t *pool = nullptr;
  bool grabbing = false;
  uint64_t last_seq = 0;
  std::vector<uint8_t> bayer = std::vector<uint8_t>(kPixels);
};

MV_CC_DEVICE_INFO device{};
std::once_flag device_once;

bool connect(Camera &camera) {
  camera.meta_fd = open("/tmp/talos_ipc_meta", O_RDWR | O_CLOEXEC);
  camera.pool_fd = open("/tmp/talos_ipc_image_pool", O_RDONLY | O_CLOEXEC);
  if (camera.meta_fd < 0 || camera.pool_fd < 0) return false;
  struct stat meta_stat{}, pool_stat{};
  if (fstat(camera.meta_fd, &meta_stat) || fstat(camera.pool_fd, &pool_stat) ||
      meta_stat.st_size < static_cast<off_t>(kMetaSize) ||
      pool_stat.st_size < static_cast<off_t>(kPoolSize)) return false;
  camera.meta = static_cast<uint8_t *>(mmap(nullptr, kMetaSize, PROT_READ | PROT_WRITE,
                                              MAP_SHARED, camera.meta_fd, 0));
  camera.pool = static_cast<const uint8_t *>(mmap(nullptr, kPoolSize, PROT_READ,
                                                    MAP_SHARED, camera.pool_fd, 0));
  if (camera.meta == MAP_FAILED || camera.pool == MAP_FAILED) {
    camera.meta = nullptr;
    camera.pool = nullptr;
    return false;
  }
  uint32_t magic{}, version{};
  memcpy(&magic, camera.meta, sizeof(magic));
  memcpy(&version, camera.meta + 4, sizeof(version));
  return magic == 0x54414c05 && version == 2;
}

void disconnect(Camera &camera) {
  if (camera.meta) munmap(camera.meta, kMetaSize);
  if (camera.pool) munmap(const_cast<uint8_t *>(camera.pool), kPoolSize);
  if (camera.meta_fd >= 0) close(camera.meta_fd);
  if (camera.pool_fd >= 0) close(camera.pool_fd);
  camera.meta = nullptr;
  camera.pool = nullptr;
  camera.meta_fd = -1;
  camera.pool_fd = -1;
}

// Same one-consumer triple-buffer exchange as talos-ipc::TripleBufferConsumer.
int consume(uint8_t *buffer) {
  auto *state = buffer;
  for (int attempt = 0; attempt < 2; ++attempt) {
    uint8_t expected = __atomic_load_n(state, __ATOMIC_ACQUIRE);
    if (!(expected & kNew)) return -1;
    const uint8_t slot = expected & kIndex;
    const uint8_t spare = buffer[2];
    if (__atomic_compare_exchange_n(state, &expected, spare, true,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      buffer[2] = slot;
      return slot;
    }
  }
  return -1;
}
} // namespace

extern "C" {
int MV_CC_EnumDevices(unsigned int, MV_CC_DEVICE_INFO_LIST *list) {
  if (!list) return -1;
  std::call_once(device_once, [] {
    device.nTLayerType = MV_USB_DEVICE;
    strncpy(reinterpret_cast<char *>(device.SpecialInfo.stUsb3VInfo.chSerialNumber),
            "DAEDALUS_SIM", sizeof(device.SpecialInfo.stUsb3VInfo.chSerialNumber) - 1);
    strncpy(reinterpret_cast<char *>(device.SpecialInfo.stUsb3VInfo.chModelName),
            "Talos Camera", sizeof(device.SpecialInfo.stUsb3VInfo.chModelName) - 1);
  });
  memset(list, 0, sizeof(*list));
  list->nDeviceNum = 1;
  list->pDeviceInfo[0] = &device;
  return MV_OK;
}

int MV_CC_CreateHandle(void **handle, const MV_CC_DEVICE_INFO *) {
  if (!handle) return -1;
  auto camera = std::make_unique<Camera>();
  if (!connect(*camera)) {
    disconnect(*camera);
    return -1;
  }
  *handle = camera.release();
  return MV_OK;
}

int MV_CC_DestroyHandle(void *handle) {
  if (!handle) return -1;
  auto camera = std::unique_ptr<Camera>(static_cast<Camera *>(handle));
  disconnect(*camera);
  return MV_OK;
}

int MV_CC_OpenDevice(void *handle, unsigned int, unsigned short) {
  return handle ? MV_OK : -1;
}
int MV_CC_CloseDevice(void *handle) {
  return handle ? MV_OK : -1;
}
int MV_CC_SetEnumValue(void *handle, const char *, unsigned int) {
  return handle ? MV_OK : -1;
}
int MV_CC_SetFloatValue(void *handle, const char *, float) {
  return handle ? MV_OK : -1;
}
int MV_CC_SetFrameRate(void *handle, float) {
  return handle ? MV_OK : -1;
}
int MV_CC_StartGrabbing(void *handle) {
  if (!handle) return -1;
  static_cast<Camera *>(handle)->grabbing = true;
  return MV_OK;
}
int MV_CC_StopGrabbing(void *handle) {
  if (!handle) return -1;
  static_cast<Camera *>(handle)->grabbing = false;
  return MV_OK;
}
int MV_CC_GetImageBuffer(void *handle, MV_FRAME_OUT *frame, unsigned int timeout_ms) {
  if (!handle || !frame) return -1;
  auto &camera = *static_cast<Camera *>(handle);
  if (!camera.grabbing) return -1;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  do {
    int slot = consume(camera.meta + 64);
    if (slot >= 0) {
      const uint8_t *entry = camera.meta + 128 + slot * 32;
      uint64_t seq{};
      uint32_t width{}, height{};
      memcpy(&seq, entry, sizeof(seq));
      memcpy(&width, entry + 16, sizeof(width));
      memcpy(&height, entry + 20, sizeof(height));
      const uint8_t pool_id = entry[24];
      if (seq && width == kWidth && height == kHeight && pool_id < 3) {
        const uint8_t *rgb = camera.pool + pool_id * kPixels * 3;
        for (size_t y = 0; y < kHeight; ++y) {
          for (size_t x = 0; x < kWidth; ++x) {
            const size_t p = y * kWidth + x;
            const size_t channel = (y % 2) ? ((x % 2) ? 2 : 1)
                                                   : ((x % 2) ? 1 : 0);
            camera.bayer[p] = rgb[p * 3 + channel];
          }
        }
      }
      // The publisher will not overwrite a frame until all four poses are consumed.
      for (int i = 0; i < 4; ++i) consume(camera.meta + 256 + i * 256);
      if (seq && seq != camera.last_seq && width == kWidth && height == kHeight && pool_id < 3) {
        camera.last_seq = seq;
        memset(frame, 0, sizeof(*frame));
        frame->pBufAddr = camera.bayer.data();
        frame->stFrameInfo.nWidth = kWidth;
        frame->stFrameInfo.nHeight = kHeight;
        frame->stFrameInfo.nFrameLen = kPixels;
        frame->stFrameInfo.nFrameNum = seq;
        frame->stFrameInfo.enPixelType = PixelType_Gvsp_BayerRG8;
        return MV_OK;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  return 0x80000007;
}
int MV_CC_FreeImageBuffer(void *handle, MV_FRAME_OUT *frame) {
  return handle && frame ? MV_OK : -1;
}
}
