#include "hikrobot.hpp"

#include <libusb-1.0/libusb.h>

#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace io
{
HikRobot::HikRobot(
  double exposure_ms, double gain, const std::string & vid_pid,
  const std::string & camera_serial, bool auto_start)
: exposure_us_(exposure_ms * 1e3),
  gain_(gain),
  camera_serial_(camera_serial),
  daemon_quit_(false),
  desired_running_(false),
  handle_(nullptr),
  capturing_(false),
  capture_quit_(true),
  opened_(false),
  grabbing_(false),
  queue_(1),
  vid_(-1),
  pid_(-1)
{
  set_vid_pid(vid_pid);

  static bool libusb_initialized = false;
  if (!libusb_initialized) {
    if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");
    libusb_initialized = true;
  }

  try {
    open_device();
  } catch (const std::exception & e) {
    tools::logger()->warn("HikRobot initial open failed: {}", e.what());
  }

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started. (SN=\"{}\")", camera_serial_);

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (!desired_running_ || capturing_) continue;

      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      if (!desired_running_ || capturing_) continue;

      capture_stop();
      close_device();
      // 两台同 VID/PID 相机并存时不能按 VID/PID 复位，否则可能复位另一台相机。
      if (camera_serial_.empty()) reset_usb();

      try {
        open_device();
        capture_start();
      } catch (const std::exception & e) {
        tools::logger()->warn("HikRobot recovery failed: {}", e.what());
      }
    }

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};

  // 普通程序保持原来的自动采集行为，UAV 双相机由外部手动启动。
  if (auto_start) start();
}

HikRobot::~HikRobot()
{
  desired_running_ = false;
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();

  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  capture_stop();
  close_device();
  tools::logger()->info("HikRobot destructed. (SN=\"{}\")", camera_serial_);
}

void HikRobot::start()
{
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  desired_running_ = true;
  try {
    if (!opened_) open_device();
    capture_start();
  } catch (const std::exception & e) {
    tools::logger()->warn("HikRobot start failed: {}", e.what());
  }
}

void HikRobot::stop()
{
  desired_running_ = false;
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  capture_stop();
}

bool HikRobot::is_running() const { return capturing_; }

void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  if (!desired_running_) {
    tools::logger()->error("HikRobot read while stopped. (SN=\"{}\")", camera_serial_);
    img.release();
    timestamp = std::chrono::steady_clock::now();
    return;
  }

  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

void HikRobot::open_device()
{
  if (opened_) return;

  unsigned int ret;

  MV_CC_DEVICE_INFO_LIST device_list;
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    throw std::runtime_error("MV_CC_EnumDevices failed");
  }

  if (device_list.nDeviceNum == 0) {
    throw std::runtime_error("Not found camera");
  }

  // 打印所有在线相机序列号
  for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
    auto * info = device_list.pDeviceInfo[i];
    if (info->nTLayerType == MV_USB_DEVICE) {
      std::string sn(reinterpret_cast<char *>(info->SpecialInfo.stUsb3VInfo.chSerialNumber));
      std::string model(reinterpret_cast<char *>(info->SpecialInfo.stUsb3VInfo.chModelName));
      tools::logger()->info("Camera [{}]: SN=\"{}\"  Model=\"{}\"", i, sn, model);
    }
  }

  // 按序列号匹配相机，空串则取第一个
  int device_index = -1;
  if (camera_serial_.empty()) {
    device_index = 0;
  } else {
    for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
      auto * info = device_list.pDeviceInfo[i];
      if (info->nTLayerType == MV_USB_DEVICE) {
        std::string sn(reinterpret_cast<char *>(info->SpecialInfo.stUsb3VInfo.chSerialNumber));
        if (sn == camera_serial_) {
          device_index = static_cast<int>(i);
          break;
        }
      }
    }
  }

  if (device_index < 0) {
    throw std::runtime_error("Camera with SN=\"" + camera_serial_ + "\" not found");
  }

  tools::logger()->info("Using camera [{}] (target SN=\"{}\")", device_index, camera_serial_);

  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[device_index]);
  if (ret != MV_OK) {
    handle_ = nullptr;
    throw std::runtime_error("MV_CC_CreateHandle failed");
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    throw std::runtime_error("MV_CC_OpenDevice failed");
  }

  // 强制设置为连续采集，避免相机残留在触发模式导致取流超时(0x80000007)
  // set_enum_value("TriggerMode", MV_TRIGGER_MODE_OFF);
  // set_enum_value("AcquisitionMode", MV_ACQ_MODE_CONTINUOUS);

  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);
  MV_CC_SetFrameRate(handle_, 150);

  opened_ = true;
  tools::logger()->info("HikRobot opened. (SN=\"{}\")", camera_serial_);
}

void HikRobot::close_device()
{
  if (!opened_) return;

  auto ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK) tools::logger()->warn("MV_CC_CloseDevice failed: {:#x}", ret);

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK) tools::logger()->warn("MV_CC_DestroyHandle failed: {:#x}", ret);

  handle_ = nullptr;
  opened_ = false;
}

void HikRobot::capture_start()
{
  if (!opened_ || grabbing_) return;

  capture_quit_ = false;

  auto ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StartGrabbing failed: {:#x}", ret);
    return;
  }

  grabbing_ = true;
  capturing_ = true;

  capture_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's capture thread started. (SN=\"{}\")", camera_serial_);

    MV_FRAME_OUT raw;
    MV_CC_PIXEL_CONVERT_PARAM cvt_param;

    while (!capture_quit_) {
      std::this_thread::sleep_for(1ms);

      unsigned int ret;
      unsigned int nMsec = 100;

      ret = MV_CC_GetImageBuffer(handle_, &raw, nMsec);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_GetImageBuffer failed: {:#x}", ret);
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

      cvt_param.nWidth = raw.stFrameInfo.nWidth;
      cvt_param.nHeight = raw.stFrameInfo.nHeight;

      cvt_param.pSrcData = raw.pBufAddr;
      cvt_param.nSrcDataLen = raw.stFrameInfo.nFrameLen;
      cvt_param.enSrcPixelType = raw.stFrameInfo.enPixelType;

      cvt_param.pDstBuffer = img.data;
      cvt_param.nDstBufferSize = img.total() * img.elemSize();
      cvt_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

      // ret = MV_CC_ConvertPixelType(handle_, &cvt_param);
      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
      cv::cvtColor(img, dst_image, type_map.at(pixel_type));
      img = dst_image;

      queue_.push({img, timestamp});

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_FreeImageBuffer failed: {:#x}", ret);
        break;
      }
    }

    capturing_ = false;
    tools::logger()->info("HikRobot's capture thread stopped. (SN=\"{}\")", camera_serial_);
  }};
}

void HikRobot::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();
  capturing_ = false;

  if (grabbing_) {
    auto ret = MV_CC_StopGrabbing(handle_);
    if (ret != MV_OK) tools::logger()->warn("MV_CC_StopGrabbing failed: {:#x}", ret);
    grabbing_ = false;
  }

  queue_.clear();
}

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    tools::logger()->warn("Unable to open usb!");
    return;
  }

  if (libusb_reset_device(handle))
    tools::logger()->warn("Unable to reset usb!");
  else
    tools::logger()->info("Reset usb successfully :)");

  libusb_close(handle);
}

}  // namespace io
