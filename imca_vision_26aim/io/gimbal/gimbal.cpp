#include "gimbal.hpp"
#include "io/talos.hpp"

#include <cstdlib>
// 包含云台类的头文件，声明了Gimbal类的成员函数和变量
#include "tools/crc.hpp" // 包含日志工具头文件，CRC

#include "tools/logger.hpp"// 包含数学工具头文件，提供时间计算、插值等功能
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io // 定义io命名空间，封装输入输出相关类
{
uint8_t reconnect_flag = 0;//0正常 1重联中 
// 云台类构造函数，接收配置文件路径参数
Gimbal::Gimbal(const std::string & config_path)
{
  i=0;
  auto yaml = tools::load(config_path);  // 加载YAML配置文件

  const bool talos_enabled =
    (yaml["gimbal_backend"] && yaml["gimbal_backend"].as<std::string>() == "talos") ||
    std::getenv("IMCA_TALOS_SIMULATOR") == std::string("1");
  if (talos_enabled) {
    talos_backend_ = true;
    talos_ = TalosTransport::shared();
    tools::logger()->info("[Gimbal] Using Talos simulator backend");
    return;
  }

  auto com_port = tools::read<std::string>(yaml, "com_port");  // 从配置文件中读取串口端口号（如"COM3"或"/dev/ttyUSB0"）

  try {
    serial_.setPort(com_port);// 设置串口端口
    serial_.open(); // 打开串口
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what()); // 串口打开失败时输出错误日志并退出程序
    exit(1);
  }

  // 创建并启动读取线程，用于持续接收云台数据
  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop(); // 等待并获取第一个四元数数据（阻塞操作）
  // q(t) 和 xy(t) 都需要前后两帧做插值，因此两条队列同步丢弃同一个首帧
  xy_queue_.pop();
  tools::logger()->info("[Gimbal] First q received.");  // 输出日志表示成功接收到第一个数据
}

// 云台类析构函数
Gimbal::~Gimbal()
{
  if (talos_backend_) return;

  // 设置退出标志，通知读取线程终止
  quit_ = true;
  if (thread_.joinable()) thread_.join();// 等待读取线程结束（如果线程可连接）
  serial_.close();// 关闭串口
}

// 获取当前云台工作模式
GimbalMode Gimbal::mode() const
{
  if (talos_backend_) return GimbalMode::AUTO_AIM;

  std::lock_guard<std::mutex> lock(mutex_); // 使用互斥锁保护共享变量，确保线程安全
  return mode_;// 返回当前模式
}

// 获取当前云台状态
GimbalState Gimbal::state() const
{
  if (talos_backend_) {
    const auto state = talos_->state();
    return {
      static_cast<float>(state.yaw), static_cast<float>(state.yaw_vel),
      static_cast<float>(state.pitch), static_cast<float>(state.pitch_vel), state.my_color,
      static_cast<float>(state.bullet_speed), state.bullet_count,
      static_cast<float>(state.x), static_cast<float>(state.y), static_cast<float>(state.z_chassis)};
  }

  // 使用互斥锁保护共享变量，确保线程安全
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;// 返回当前状态
}

// 将云台模式枚举转换为字符串，便于日志输出
std::string Gimbal::str(GimbalMode mode) const
{
  // 根据枚举值返回对应的字符串
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";    // 空闲模式
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";  // 自动瞄准模式
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";// 小能量机关模式
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";// 大能量机关模式
    case GimbalMode::LONG:
      return "LONG";
    case GimbalMode::OUTPOST:
      return "OUTPOST";
    default:
      return "INVALID"; // 无效模式
  }
}
// 根据指定时间点获取插值后的四元数
Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  if (talos_backend_) {
    (void)t;
    return talos_->gimbal_q();
  }

   // 循环直到找到合适的插值数据
  while (true) {
    auto [q_a, t_a] = queue_.pop(); // 从队列中取出前一个四元数及其时间戳
    auto [q_b, t_b] = queue_.front(); // 获取队列中下一个四元数及其时间戳（不弹出）
    auto t_ab = tools::delta_time(t_a, t_b);// 计算两个时间点之间的时间差
    auto t_ac = tools::delta_time(t_a, t);// 计算目标时间与前一时间点的时间差
    auto k = t_ac / t_ab;  // 计算插值比例
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();// 使用球面线性插值(SLERP)计算目标时间的四元数并归一化
    if (t < t_a) return q_c;  // 如果目标时间早于第一个时间点，直接返回插值结果
    if (!(t_a < t && t <= t_b)) continue;// 如果目标时间不在两个时间点之间，继续循环查找

    return q_c;// 返回插值结果
  }
}

// 根据图像时间戳线性插值云台安装点在固定世界系 W0 下的绝对 x/y
Eigen::Vector2d Gimbal::xy(std::chrono::steady_clock::time_point t)
{
  if (talos_backend_) {
    (void)t;
    const auto state = talos_->state();
    return {state.x, state.y};
  }

  while (true) {
    auto [xy_a, t_a] = xy_queue_.pop();
    auto [xy_b, t_b] = xy_queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    // delta_time(a, b) 返回 a-b，因此分子分母同号，k 仍为 [t_a,t_b] 内的插值比例
    auto k = t_ac / t_ab;
    Eigen::Vector2d xy_c = xy_a + k * (xy_b - xy_a);
    if (t < t_a) return xy_c;
    if (!(t_a < t && t <= t_b)) continue;

    return xy_c;
  }
}

// 发送视觉到云台的数据
void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  if (talos_backend_) {
    talos_->send(
      VisionToGimbal.mode != 0, VisionToGimbal.mode == 2, VisionToGimbal.yaw,
      VisionToGimbal.yaw_vel, VisionToGimbal.yaw_acc, VisionToGimbal.pitch,
      VisionToGimbal.pitch_vel, VisionToGimbal.pitch_acc);
    return;
  }

  // 将输入数据复制到发送缓冲区
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;

// 将发送缓冲区的数据通过串口发送
  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());// 发送失败时输出警告日志
  }
}

// 发送控制数据
void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  if (talos_backend_) {
    talos_->send(control, fire, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc);
    return;
  }

  
  if(reconnect_flag == 0)
  {
    // 0: 不控制, 1: 控制但不发射, 2: 控制且发射
    tx_data_.mode = control ? (fire ? 2 : 1) : 0;
    // 填充姿态控制参数
    tx_data_.yaw = yaw;// + rx_data_.yaw
    tx_data_.yaw_vel = yaw_vel;
    tx_data_.yaw_acc = yaw_acc;
    tx_data_.pitch = pitch;
    tx_data_.pitch_vel = pitch_vel;
    tx_data_.pitch_acc = pitch_acc;
      reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;
    //zhongduan
      float  tx_yaw =rx_data_.yaw;
      float  send_yaw = tx_data_.yaw;
      float  send_pitch = tx_data_.pitch;
      float  send_control = tx_data_.mode;
    // tools::logger()->info("yaw:{} pitch:{} mode:{} ", send_yaw ,send_pitch,send_control );
    // 将发送缓冲区的数据通过串口发送
    try {
      serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
    }
  }
  else
  {
    tools::logger()->warn("[Gimbal] reconnect_flag: {}", reconnect_flag);
  }
}

// 从串口读取指定大小的数据到缓冲区
bool Gimbal::read(uint8_t * buffer, size_t size)
{
  // 尝试读取数据，返回是否读取到指定大小的数据
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;// 读取异常时返回失败
  }
}


// 读取云台数据的线程函数
void Gimbal::read_thread()
{
  // 输出线程启动日志
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;// 错误计数器，用于检测连续读取错误

// 循环读取数据，直到收到退出信号
  while (!quit_) 
  {
    // 当连续错误超过5000次时，尝试重新连接串口
    if (error_count > 50000) 
    {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    
    // 读取帧头（1字节）
    if (!read(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) 
    {
      // tools::logger()->warn("[Gimbal] head {}",rx_data_.head);
      error_count++;
      continue;
    }

     // 验证帧头是否正确（必须为0x5a）
    if (rx_data_.head != 0x5a  )
    {
      error_count++;
      continue;
    }

    // 读取帧头之后的剩余数据
    if (!read(reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
    sizeof(rx_data_) - sizeof(rx_data_.head))) 
    {
      error_count++;
      continue;
    }

    auto t = std::chrono::steady_clock::now() ; // 获取当前时间戳（用于时间同步）+  std::chrono::milliseconds(10)
    
     // 验证帧end是否正确（必须为0x5a）
    if (rx_data_.tail != 0xa5  )
    {
      // tools::logger()->warn("[Gimbal] tail {}",rx_data_.tail);
      error_count++;
      continue;
    } 
    reconnect_flag = 0;
    //zhongduan
    float  Rx_yaw = rx_data_.yaw*57.3;
    float  Rx_pitch = rx_data_.pitch*57.3;
    uint8_t  Rx_mode = rx_data_.mode;
    // tools::logger()->info("[Gimbalrx] yaw:{} pitcch:{} mode:{} ", Rx_yaw ,Rx_pitch ,Rx_mode);


    // tools::logger()->info("[Gimbal]i={}",i++);
    error_count = 0;// 所有验证通过，重置错误计数器
    Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);// 从接收数据中构造四元数（w,x,y,z）
    queue_.push({q, t});   // 将四元数和时间戳存入队列         
    // q 与 x/y 使用同一个串口帧接收时间戳，供图像时刻分别插值
    xy_queue_.push({Eigen::Vector2d(rx_data_.x, rx_data_.y), t});

    std::lock_guard<std::mutex> lock(mutex_); // 使用互斥锁保护共享变量，确保线程安全

    state_.yaw = rx_data_.yaw;
    state_.my_color = rx_data_.my_color;
    state_.yaw_vel = rx_data_.yaw_vel;
    state_.pitch = rx_data_.pitch;
    state_.x = rx_data_.x;
    state_.y = rx_data_.y;
    state_.z_chassis = rx_data_.z;
    // state_.pitch_vel = rx_data_.pitch_vel;
    // state_.bullet_speed = rx_data_.bullet_speed;
    // state_.bullet_count = rx_data_.bullet_count;

  // 根据接收的模式值更新云台模式
    switch (rx_data_.mode) {
      tools::logger()->info("[Gimbal]  mode: {}", rx_data_.mode);
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      case 4:
        mode_ = GimbalMode::LONG;
        break;
      case 5:
        mode_ = GimbalMode::OUTPOST;
        break;
      default:
        mode_ = GimbalMode::IDLE;// 未知模式时默认设为空闲模式并输出警告
        tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
        break;
    }
  }
  // 线程退出时输出日志
  tools::logger()->info("[Gimbal] read_thread stopped.");
}

// 重新连接串口的函数
void Gimbal::reconnect()
{
  // 最大重试次数
  int max_retry_count = 10;
  // 循环重试连接
  for (int i = 0; i < max_retry_count && !quit_; ++i) 
  {
    reconnect_flag = 1;
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    // 关闭当前串口
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();  // 尝试重新打开
      queue_.clear();
      // 丢弃队列中断线前的位置样本，避免正常调用跨重连间隔插值
      xy_queue_.clear();
      // 输出重连成功日志
      reconnect_flag = 0;
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  reconnect_flag = 0;
}

}  // namespace io
