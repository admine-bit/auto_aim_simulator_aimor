## 本项目亮点
- 无ROS依赖*，新同学无需学习ROS相关知识就能上手自瞄。
- 完整工作流，包含开发、编译、调试、部署全流程。
- 模块化架构，包含各模块的功能说明、代码实现、以及独立的测试程序。
- 我们提出了“轨迹视角下的自瞄理论”，并据此设计了“自瞄轨迹规划算法”。相比传统自瞄决策算法，该算法实现更简洁、调试更方便、效果更好，击杀时间仅需10s（300HP，14rad/s，2m）。该理论也为后续自瞄技术的改进提供了方向。

*为实现哨兵“追击”功能，本项目预留了与导航通信的ROS接口。


## 1 功能介绍
自瞄的定义和意义。我们对自瞄的定义为“针对移动装甲板目标的自动瞄准和自动火控软件”。当操作手切换成自瞄模式后，自瞄会接管云台的控制权，通过对敌方运动轨迹的预测和弹道解算，控制云台进行追踪；同时，自瞄还会接管发射机构的控制权，判断开火时机。自瞄的意义在于提高我方作战能力，实现短击杀时间、高命中率的作战效果。 

自瞄的模块组成。经过多年的发展，大部份参赛队伍都具备了自瞄的能力。通过和一些队伍的交流，目前各队的自瞄类似于如图1.1所示的模块划分。装甲板识别器和目标状态估计器作为“感知模块”，提供敌方运动信息。决策器根据上述信息，进行瞄准位置的计算和开火判断，并最终交由控制器进行执行。
![自瞄模块划分](https://github.com/user-attachments/assets/0e3e960e-0bf0-430a-95f5-d0803d1a6ade)
图1.1 自瞄模块划分 

自瞄是分布式软件。识别器、估计器、决策器由于算法较为复杂，一般运行在算力较高的小电脑上；出于对实时性和稳定性的考量，控制器一般运行在嵌入式单片机上。模块划分解耦了视觉组和电控组的职责，明确了分工，提高了开发效率。

本项目包含了本队自瞄相关的视觉组部分的全部代码，包括但不限于：多线程相机驱动程序、电控通信协议和具体实现、图像-四元数对齐算法、相机内参标定和手眼标定程序、装甲板识别算法（含传统、神经网络多种实现）、装甲板位姿解算、坐标变换和yaw优化算法、基于拓展卡尔曼的整车估计器、瞄准位置和开火决策逻辑、以及各模块独立测试程序等。

本项目建立在往年[imca_vision_23](https://github.com/TongjiSuperPower/sp_vision_23)、[imca_vision_24](https://github.com/TongjiSuperPower/sp_vision_24)项目的基础之上。其中，imca_vision_23完全基于Python开发，跑通了自瞄从识别到射击的完整流程，因其运行效率不佳（在导航运行时，帧率仅60FPS），imca_vision_24采用C++重写（帧率约100FPS），并且具备参数文件加载、日志记录、离线检测与重启、类似rosbag（时间戳+视频+四元数）录制等辅助功能，以及各个模块的独立测试程序，实现“赛前问题排查、赛后问题重现”的能力，保证了自瞄的可靠性和稳定性。

本项目在上述基础上改进了识别器和决策器。其中，识别器由传统的图像处理方案更换为其它战队开源的基于神经网络的四点检测模型[1,2]，以期提高识别器的召回能力。而决策器则由基于经验主义的分段决策逻辑改为基于轨迹优化的规划器，简化代码的同时，实现了更优的云台控制效果和开火决策逻辑，其思路会在后文详细介绍。


## 2 效果展示
轨迹跟随效果。本项目创新性采用“自瞄轨迹规划算法”，降低控制难度的同时，提升轨迹跟随效果。如图2.1所示，跟随单块装甲板时，稳态误差小于0.01rad，装甲板切换时，具备提前减速的能力。
![轨迹跟随效果](https://github.com/user-attachments/assets/ac65ab13-bc2c-45c0-9c06-613442fc104d)
图2.1 轨迹跟随效果（5rad/s，3m）

调试场景演示。本项目容易上手，对新人友好。如视频2.1所示，我们通过无线局域网或插网线方式，使用NoMachine远程桌面软件连接机器人搭载的小电脑，配合PlotJuggler曲线图绘制软件，显示调试时常用的数据信息，如云台yaw关节执行情况、开火决策、击打目标的角速度等。

https://github.com/user-attachments/assets/606c2907-2f11-4392-b3fe-7ee4b7b6fd29


## 3 详细信息
### 3.1 项目环境
操作系统：Ubuntu 22.04\
运算平台：NUC12WSKI7（i7-1260P，16GB）\
相机型号：海康MV-CS016-10UC\
镜头型号：海康官方6mm镜头\
下位机型号：RoboMaster开发板C型（STM32F407）\
IMU型号：使用C板内置BMI088作为IMU\
通信方式：USB2CAN（旧）、MicroUSB虚拟串口（新）\
辅助工具：NoMachine（远程桌面）、PlotJuggler（绘制曲线图）

### 3.2 编译方式
1. 安装依赖项：
   - [MindVision SDK](https://mindvision.com.cn/category/software/sdk-installation-package/)或[HikRobot SDK](https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_GML_V2.1.2_231116.zip)
   - [OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)
   - [Ceres](http://ceres-solver.org/installation.html)
   - 其余：
    ```bash
    sudo apt install -y \
        git \
        build-essential \
        cmake \
        pkg-config \
        can-utils \
        libopencv-dev \
        libfmt-dev \
        libeigen3-dev \
        libspdlog-dev \
        libyaml-cpp-dev \
        libusb-1.0-0-dev \
        libceres-dev \
        nlohmann-json3-dev \
        openssh-server \
        screen
    ```

2. 编译：
    ```bash
    cmake -B build
    make -C build -j12
    ```
    默认构建使用系统 OpenVINO，并自动忽略当前 shell 中激活的 Conda 包路径。若旧的
    `build/` 曾缓存 Conda 依赖，先执行一次 `cmake --fresh -B build`，之后即可继续使用
    上述普通命令。

    只有明确需要整套 Conda 依赖时，才同时指定
    `-DIMCA_ALLOW_CONDA_PACKAGES=ON` 和
    `-DOpenVINO_DIR=<OpenVINOConfig.cmake所在目录>`；不要用该模式替代缺失的系统开发包。

3. 运行demo:
    ```bash
    ./build/auto_aim_test
    ```

### 3.2.1 与 Daedalus 仿真平台联调

项目目录中的 `bevy_robomaster_simulator` 通过 Talos 共享内存提供仿真图像、云台位姿和云台指令接口。
源码侧的 `io` 层在 `IMCA_TALOS_SIMULATOR=1` 时自动切换到该接口，实车的工业相机和串口路径不受影响。

先启动仿真器（仿真器目录需要作为工作目录，以便加载 `assets/` 和 `config.toml`）：

```bash
cd <repo-root>/bevy_robomaster_simulator
cargo run --release
```

再启动视觉程序（源码目录需要作为工作目录，以便加载 `assets/` 和配置文件）：

```bash
cd <repo-root>/imca_vision_26aim
IMCA_TALOS_SIMULATOR=1 ./build/standard_mpc configs/sentry.yaml
```

也可以直接从仿真器目录使用一键脚本启动仿真器和本程序：

```bash
cd <repo-root>/bevy_robomaster_simulator
./start_simulation.sh
```

脚本支持通过 `--program`、`--workdir` 和 `--` 指定项目外的其他视觉程序，详见
仿真器目录中的 `start_simulation.sh`；自瞄与手动云台切换规则见仿真器目录中的
《仿真自瞄与云台控制使用手册》。

仿真器默认要求按 `F5` 开启外部自瞄；也可以在启动时设置 `DAEDALUS_AUTO_AIM=1` 自动开启：

```bash
cd <repo-root>/bevy_robomaster_simulator
DAEDALUS_AUTO_AIM=1 cargo run --release
```

两端通过 `/tmp/talos_ipc_meta` 和 `/tmp/talos_ipc_image_pool` 通信，应先启动仿真器再启动视觉程序。
一键脚本默认启动 `auto_aim_debug_mpc`；也可以通过 `--program` 指定 `standard_mpc` 或项目外的其他程序。脚本默认关闭源码 OpenCV 调试窗口，避免其抢走仿真器键盘焦点；需要调试画面时设置 `IMCA_VISION_DEBUG_WINDOWS=1`。打符和实车 CAN/串口设备不在此仿真适配范围内。

4. 注册自启：
    1. 确保已安装`screen`:
        ```
        sudo apt install screen
        ```
    2. 创建`.desktop`文件:
        ```
        mkdir ~/.config/autostart/
        touch ~/.config/autostart/imca_vision.desktop
        ```
    3. 在该文件中写入:
        ```
        [Desktop Entry]
        Type=Application
        Exec=/home/imca03/imca_vision/autostart.sh
        Name=imca_vision
        ```
        注: [Exec](https://specifications.freedesktop.org/desktop-entry-spec/desktop-entry-spec-latest.html)必须为绝对路径.
    4. 确保`autostart.sh`有可执行权限:
        ```
        chmod +x autostart.sh
        ```

5. USB2CAN设置（可选）
    1. 创建`.rules`文件:
        ```
        sudo touch /etc/udev/rules.d/99-can-up.rules
        ```
    2. 在该文件中写入:
        ```
        ACTION=="add", KERNEL=="can0", RUN+="/sbin/ip link set can0 up type can bitrate 1000000"
        ACTION=="add", KERNEL=="can1", RUN+="/sbin/ip link set can1 up type can bitrate 1000000"

6. 使用GPU推理（可选）
    ```
    mkdir neo  
    cd neo  

    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13463.18/intel-igc-core_1.0.13463.18_amd64.deb  
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13463.18/intel-igc-opencl_1.0.13463.18_amd64.deb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-level-zero-gpu-dbgsym_1.3.25812.14_amd64.ddeb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-level-zero-gpu_1.3.25812.14_amd64.deb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-opencl-icd-dbgsym_23.09.25812.14_amd64.ddeb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-opencl-icd_23.09.25812.14_amd64.deb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/libigdgmm12_22.3.0_amd64.deb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/ww09.sum  

    sha256sum -c ww09.sum  
    sudo dpkg -i *.deb  
    ```
    注：如果使用 GPU 异步推理（async-infer），最高显示分辨率限制为 1920×1080 (24Hz)

7. 串口设置
    1. 授予权限
        ```
        sudo usermod -a -G dialout $USER
        ```
    2. 获取端口 ID（serial, idVendor, idProduct）
        ```
        udevadm info -a -n /dev/ttyACM0 | grep -E '({serial}|{idVendor}|{idProduct})'
        ```
        将 /dev/ttyACM0 替换为实际设备名。
    3. 创建 udev 规则文件
        ```
        sudo touch /etc/udev/rules.d/99-usb-serial.rules
        ```sudo nano /etc/udev/rules.d/99-usb-serial.rules
        然后在文件中写入如下内容（用真实 ID 替换示例，SYMLINK 是规则应用后固定的串口名）：
        ```
        SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", ATTRS{serial}=="205F389A4232", SYMLINK+="gimbal"
        ```
    4. 重新加载 udev 规则
        ```
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        ```
    5. 检查结果
        ```
        ls -l /dev/gimbal
        # Expected output (example):
        # lrwxrwxrwx 1 root root 7 Jul 21 10:00 /dev/gimbal -> ttyACM0
        ```

### 3.3 数据流图
视觉相关模块如图3.1所示。其中，相机线程产生图像、时间戳，通过下位机线程获取对应的云台姿态四元数；图像经过识别器，获得装甲板的四个顶点像素坐标，以及其图案类别；估计器根据装甲板信息，获得目标单位的运动状态；决策器则根据当前的目标运动状态信息，预测目标的运动轨迹，从而判断最佳瞄准位置和最佳开火时机，形成指令发送给下位机；最后控制器和执行机构则根据该指令进行执行，从而完成一个完整的自瞄流程。
![数据流图](https://github.com/user-attachments/assets/b89ce42f-a769-49c5-b82a-d69aeac02925)
图3.1 数据流图

### 3.4 软件架构
各兵种需要实现的功能往往有多个，但是每个功能不能作为独立的程序。例如步兵需要自瞄和打符，显然，这两个功能都需要从相机获取图像，但是相机只能被一个程序打开，这会导致另一个程序无法正常工作（相机被占用）。

为了解决这个问题，我们提出了视觉框架（imca_vision）：自瞄、打符等功能，被拆解为该框架下的一个个功能组，每个功能组包含多个类或函数（如识别器、解算器、预测器等等）；而运行的程序（main函数）只有一个，它负责从相机获取图像，再根据电控发来的信号（自瞄档or打符档），选择对应的功能组执行，不同兵种的执行逻辑各不相同，即不同的main函数。此外，为了方便大家合作开发，视觉框架还提供了许多常用的工具函数，减少因重复造轮子所导致的出错概率和时间成本。本框架的组成如图3.2所示。
![软件架构](https://github.com/user-attachments/assets/2603a3b3-ae1d-4fa8-afbb-490efde30d77)
图3.2 软件架构

### 3.5 文件结构
```
imca_vision
├── assets         // 包含demo素材、网络权重等
│   └── ...
├── calibration    // 标定相关程序
│   ├── calibrate_camera.cpp             // 相机内参标定程序
│   ├── calibrate_handeye.cpp            // 手眼标定程序
│   ├── calibrate_robotworld_handeye.cpp // 手眼标定程序（同时计算标定板位置）
│   └── capture.cpp                      // 相机标定数据采集程序
├── CMakeLists.txt // CMake配置文件
├── configs        // 每台机器人的YAML配置文件
│   └── ...
├── io             // 硬件抽象层，见3.4软件架构
│   └── ...
├── src            // 应用层，见3.4软件架构
│   └── ...
├── tasks          // 功能层，见3.4软件架构
│   ├── auto_aim       // 自瞄相关算法实现
│   │   └── ...
│   ├── auto_buff      // 打符相关算法实现
│   │   └── ...
│   └── omniperception // 全向感知相关算法实现
│   │   └── ...
├── tests
│   ├── auto_aim_test.cpp         // 自瞄录制视频测试程序
│   ├── auto_buff_test.cpp        // 打符录制视频测试程序
│   ├── camera_detect_test.cpp    // 识别器测试程序（工业相机）
│   ├── camera_test.cpp           // 相机测试程序
│   ├── camera_thread_test.cpp    // 相机线程测试程序
│   ├── cboard_test.cpp           // C板测试程序
│   ├── detector_video_test.cpp   // 识别器测试程序（视频）
│   ├── dm_test.cpp               // 达妙IMU测试程序
│   ├── fire_test.cpp             // 开火测试程序
│   ├── gimbal_response_test.cpp  // 云台响应测试程序
│   ├── gimbal_test.cpp           // 云台通信测试程序
│   ├── handeye_test.cpp          // 手眼标定测试程序
│   ├── minimum_vision_system.cpp // 最小视觉系统测试程序
│   ├── multi_usbcamera_test.cpp  // 多USB摄像头测试程序
│   ├── planner_test_offline.cpp  // 规划器测试程序（离线）
│   ├── planner_test.cpp          // 规划器测试程序（实车）
│   ├── publish_test.cpp          // ROS发送测试程序
│   ├── subscribe_test.cpp        // ROS接收测试程序
│   ├── topic_loop_test.cpp       // ROS话题循环测试程序
│   ├── usbcamera_detect_test.cpp // 识别器测试程序（USB相机）
│   ├── usbcamera_test.cpp        // USB相机测试程序
│   └── ...
└── tools          // 工具层，见3.4软件架构
    ├── crc.hpp                    // CRC校验
    ├── exiter.hpp                 // 退出检测
    ├── extended_kalman_filter.hpp // 扩展卡尔曼滤波器
    ├── img_tools.hpp              // 图像处理工具
    ├── logger.hpp                 // 日志记录器
    ├── math_tools.hpp             // 数学工具
    ├── plotter.hpp                // 曲线图绘制工具
    ├── recorder.hpp               // 视频录制器
    ├── thread_safe_queue.hpp      // 线程安全队列
    ├── trajectory.hpp             // 弹道解算
    ├── yaml.hpp                   // YAML配置文件解析器
    └── ...
```    


## 4 轨迹视角下的自瞄理论
### 4.1 引言
在自瞄的研发和测试过程中，我们发现决策器和控制器是目前的短板。

决策器实现繁琐。不同兵种、不同敌方移动状态（平移、低速小陀螺、高速小陀螺）需要不同的“自瞄行为”，例如：英雄只瞄准旋转中心位置，根据时间误差判断是否开火；平移和低速时，步兵的瞄准位置则持续跟随敌方装甲板移动，同时依据位置误差判断是否开火，而高速时则退化为类似英雄的情况。不同的自瞄行为需要不同的代码实现，而且需要设计不同的判断条件，这一方面提高了代码维护的成本，另一方面也引入额外参数，增大了调试的负担。

控制调参困难。传统控制指标的输入一般为阶跃或斜坡函数，而自瞄控制器的输入更类似于三角波，且周期和幅值随敌方运动状态变化而变化，我们无法找到合适的理论将二者联系起来，因此调参过程中更依赖于经验猜测而不是理论指导。此外，我们无法判断云台的控制效果上限在哪里，甚至对于其影响因素都缺乏充分的认识。这导致电控调参十分“坐牢”，调试一台机器人需要消耗大量时间和资源，并且调试后的机器人表现效果参差不齐。

因此，我们希望能够找到一个“理论”，统一不同决策逻辑、精简代码，同时提高我们对于自瞄以及控制的认识，指导我们更加高效、可靠地调试自瞄，最终提高整体的自瞄效果。

### 4.2 前置概念
命中率。命中率越高，意味着所发射的子弹中命中装甲板的数量越多，造成的伤害越高，是用来衡量自瞄效果最常见的指标。然而，“不谈射频的命中率是没有意义的”[3]，因为如果自瞄只在最有把握的时候射击，只要时间足够长，早晚能打死对面，所以除了命中率还需要考虑击杀时间。

击杀时间。击杀时间衡量了打死敌方单位所需的时间，是衡量自瞄好坏的“金标准”：一方面，装甲板击打检测和发射机构射频均存在上限，杜绝了依靠超高射频刷击杀时间的可行性；另一方面，该指标便于测量，仅需要秒表和靶机即可，比赛时也可以通过回放视频来快速估算。在测量该指标时，需要控制敌方血量、距离和运动状态等变量，这样才能保证时间的可比较性。

目标运动状态。目标运动状态由估计器计算得出，用于决策器预测目标在一段时间后的位置。目前主流的估计算法为基于扩展卡尔曼滤波的整车估计器[3,4]，其定义的目标运动状态包括目标旋转中心位置、速度、yaw旋转角度、yaw旋转角速度、各装甲板半径和高度差，根据这些信息，决策器可以计算出每个装甲板的位置，并在其中选择击打目标。为了方便后续的论述，这里有两点假设：假设一，估计器是完美的，无需考虑估计导致的误差；假设二，目标处于匀速运动状态，从而无需考虑预测时因目标变速导致的误差。上述两点假设简化了自瞄使用场景的复杂性，帮助我们专注于决策器的改进。

目标轨迹和云台轨迹。已知目标运动状态，根据匀速运动公式，可以计算任意时刻t下目标各装甲板位置，选择距离云台最近的装甲板，计算其相对于云台的方位角yaw，得到函数yaw_target(t)，记为目标轨迹。由于pitch计算方式类似，这里的“轨迹”只讨论yaw。云台轨迹同理，代表对应时刻云台相对于初始位置所呈的夹角yaw，记为yaw_gimbal(t)。

射击轨迹。子弹出膛后需经过子弹飞行时间t_fly后到达装甲板，将目标轨迹提前t_fly可获得射击轨迹，即：
```math
yaw_{shoot}(t) = yaw_{target}(t + t_{fly})
```
如果t时刻云台角度yaw_gimbal和对应的射击角度yaw_shoot满足一定的误差范围（半个装甲板），则此时从枪管射出的子弹会命中装甲板。更进一步，如果云台轨迹和射击轨迹重合，则任意时刻从枪管射出的子弹均会命中装甲板，“子弹发射如水流一样”[3]。如图4.1所示，射击轨迹的瞄准位置是14ms后目标轨迹的位置。
![轨迹示意图](https://github.com/user-attachments/assets/80019e43-8cdc-47d1-98cd-dd8ae3d5e20b)
图4.1 轨迹示意图

### 4.3 什么是好自瞄
在轨迹视角下，“云台轨迹”和“射击轨迹”重合度越高，击杀时间越短。击杀时间反映了DPS（Damage Per Second），公式如下：
```math
击杀时间 = \frac{血量}{DPS}
```
```math
DPS = 单位时间射击窗口占比 \times 射频 \times 单发子弹伤害
```
不难看出，重合度越高，单位时间射击窗口占比越高，DPS越高，击杀时间越短，自瞄效果越好。

轨迹重合度和云台控制能力息息相关。云台最大加速度的高低决定了云台控制能力的好坏，云台能产生的瞬时加速度越大，跟随轨迹变化的能力越强，轨迹重合度越高。云台最大加速度计算公式如下：
```math
云台最大加速度 = \frac{云台电机最大扭矩}{云台惯量}
```
其中，云台电机最大扭矩一般由电机厂家给出，云台惯量通过系统辨识[5]得出。除了云台最大加速度，控制算法的设计也会影响最终的云台控制效果，为了尽可能降低控制难度，我们采取了多种措施：
1. 我们提出了“自瞄轨迹规划器”，根据云台最大加速度优化射击轨迹，使其更容易被云台跟随。
2. 我们计算了轨迹对应的速度和加速度，作为前馈量一并发送给电控，提高云台的响应。
3. 我们采用了计算力矩控制算法[6]，其结合PID控制和动力学模型，相比双环PID调参更加简便。

除了击杀时间，命中率也可以在轨迹视角下进行分析。我们可以将云台轨迹划分为“重合射击轨迹段”和“偏离射击轨迹段”，二者的比例决定了理论命中率上限：
```math
命中率 \le 重合度 = \frac{重合射击轨迹段}{(重合射击轨迹段 + 偏离射击轨迹段)}
```
云台在移动过程中，不可避免地会出现偏离射击轨迹的情况，当该情况发生时，应停止开火，减少弹丸的浪费，保证命中率。在自瞄轨迹规划器中，我们将上述情况作为开火决策的依据，在保证击杀时间的同时提高命中率。

击杀时间越短、命中率越高，自瞄效果越好。前者依赖良好的瞄准位置决策，后者则依赖良好的开火决策。我们从轨迹视角出发，用自瞄轨迹规划器替代传统自瞄决策器：通过考虑云台控制的能力，使得决策后的瞄准位置的更具可行性；通过考虑射击轨迹和规划后轨迹的偏差，结合开火延迟时间，使得开火决策更加简洁清晰，无需额外的分段参数，调试时更加方便。

### 4.4 轨迹规划器
轨迹突变问题和提前减速策略。小陀螺时装甲板会发生切换，导致目标轨迹和射击轨迹突变，不连续点处的瞬时速度和加速度未定义，强行让控制算法跟随这样的轨迹会有明显的超调或滞后现象。针对轨迹突变问题，轨迹规划器采取“提前减速策略”：若未来一段时间后装甲板会发生切换，则提前减速向下一个装甲板过渡，使规划后轨迹的加速度小于云台最大加速度，如图4.2所示。
![提前减速策略](https://github.com/user-attachments/assets/f3c60424-5ece-4eff-a14e-f30fd9ddd584)
图4.2 提前减速策略示意图

该策略的实现方式不止一种，我们尝试了两种方案：
1. 隐式搜索，以规划后轨迹加速度序列为变量，构造代价函数（重合度尽可能高）和约束条件（不超过云台最大加速度），将该问题转换为二次规划问题，调用第三方库TinyMPC[7]求解。该库针对MPC问题的求解进行了加速优化，实测求解耗时小于1ms。这里MPC并非闭环控制器，而是作为求解器寻找可行轨迹解，其输入不包含云台的实际状态。我们也尝试过使用MPC直接作为闭环控制器，通过下位机转发MPC求解后的力矩指令给电机，当自身小陀螺时，控制效果不如下位机方案。
2. 显式搜索，以过渡时间为变量，确定切换点前后的起点和终点，使用五次多项式生成过渡段轨迹，调整时间长度，直到满足加速度限制。该方案确定性高，“跟随段”不会参与优化，和射击轨迹重合，在高转速下更加稳定。同时实现更加轻量，无需依赖第三方库和QR矩阵等额外参数。

由于时间有限，方案二仅进行了仿真验证，未上车测试，国赛上场采用了方案一。

开火延迟问题和提前开火决策。从开火命令发送到子弹出膛（摩擦轮掉速）存在时间延迟t_fire，仅考虑当前位置误差进行开火判断并不严谨。轨迹规划器会生成一段时间内的轨迹序列，通过查询t_fire时刻射击轨迹和对应规划后轨迹的误差，判断经过t_fire时间后子弹是否应该出膛，从而实现更加精细的开火决策。该方案的前提是开火延迟无波动，对机械和电控的要求较高。

预测时间偏移量。射击轨迹提前于目标轨迹的时间称为预测时间，预测时间除了子弹飞行时间外，还需考虑各个环节引入的时间延迟，包括：
- 图像传输延迟：图像时间戳对应接收完成的时刻，需要图像传输至小电脑的时间。
- 图像处理延迟：神经网络推理耗时无法忽略，需要考虑图像开始处理到形成决策命令的时间。
- 下位机通信延迟：向下位机传输决策命令所需的时间。
- 下位机控制延迟：控制云台到目标位置所需的时间。

这里不需要考虑开火延迟，因为在射击轨迹上任意时间发射均会命中[3]。在实际代码实现中，除了图像处理延迟可以直接计算外，我们把其余延迟时间的总和作为一个调试参数[8]，通过拍摄慢动作视频、比较击杀时间等方式进行调整，约15ms。


## 5 未来优化方向
将轨迹规划器部署到更多兵种上。由于时间有限，国赛中仅步兵使用了轨迹规划器。我们非常期待其在英雄（射频低）、哨兵（惯量大）、无人机（射频高）上的表现。

边跑边打。目前的自瞄并没有考虑自身的移动，无法边跑边打。在本赛季，步兵经常需要从狗洞冲下去杀对面静止的英雄，而此时自瞄会认为对面在移动，导致前几发打不准，限制了操作手的发挥。我们计划引入轮式里程计信息，改善在该场景下的表现。

## 推理后端选择

工程默认使用系统 OpenVINO。通过 CMake 参数选择推理后端，无需修改源码：

```bash
cmake -B build -DIMCA_INFERENCE_BACKEND=OPENVINO
```

Jetson Orin Nano Super 使用 TensorRT 时，使用独立构建目录配置：

```bash
cmake -B build-tensorrt -DIMCA_INFERENCE_BACKEND=TENSORRT
```

`YOLOV5`、`YOLOV8`、`YOLO11`、`Classifier`、`YOLO11_BUFF` 和多线程检测器都会使用当前编译时选择的后端。一次编译只包含一种后端，不通过 YAML 在运行时切换。

TensorRT 的 `.engine` 文件必须与 Jetson Orin Nano Super 的 GPU、CUDA 和 TensorRT 版本匹配，建议在目标设备上生成。
TensorRT 使用的模型路径字段如下：

```yaml
yolov5_engine_path: model/yolov5_orin_fp16.engine
yolov8_engine_path: assets/yolov8.engine
yolo11_engine_path: assets/yolo11.engine
classify_engine_path: model/tiny_resnet.engine
yolo11_buff_engine_path: model/yolo11_buff.engine
buff_yolo_confidence_threshold: 0.50
buff_yolo_keypoint_threshold: 0.50
buff_yolo_nms_iou_threshold: 0.40
buff_loss_max_frames: 20
buff_secondary_hold_max_frames: 20
```

Buff 的 CPU/OpenVINO 配置使用同名配对的 `assets/yolo11_buff_int8.xml` 与
`assets/yolo11_buff_int8.bin`；默认 TensorRT 配置使用 `model/yolo11_buff.engine`。模型输入为
`[1,3,640,640]`。旧模型输出为 `[1,23,8400]`；加入激活类别后输出为
`[1,24,8400]`，包含 `buff` 与 `buf1` 两个类别分数和 6 个 Pose 关键点通道。
`buff` 类保留关键点输出，`buf1` 类只表示对应目标已激活，后处理不读取其关键点。
六点顺序固定为：`kpt0` 是离 R 中心最远的 target 角点，`kpt1~kpt3` 是从
`kpt0` 开始逆时针排列的其余 target 角点，`kpt4` 是 R 标中心，`kpt5` 是流水灯中心。
流水灯中心的 Buff 坐标是 `(0,0,0.344)`；target/扇叶中心由四个 target 角点的
对角线交点计算，其 Buff 坐标仍是 `(0,0,0.700)`，两者不能混用。
当前 PnP 使用 `kpt0~kpt3` 四个 target 角点和 `kpt5` 流水灯中心这五个稳定观测点；
`kpt4` R 标中心只用于诊断，不参与 PnP，禁止在后处理中按图像位置重新排序。`auto_buff_debug_yolo` 保留真实云台
发送与开火逻辑，完成六点重投影和姿态连续性验证前只能在断开拨弹/摩擦轮或录像环境中运行。

Buff YOLO 的三个阈值由 YAML 读取：`buff_yolo_confidence_threshold` 控制目标框置信度，
`buff_yolo_keypoint_threshold` 控制 `kpt0~kpt3`、`kpt5` 的关键点置信度，
`buff_yolo_nms_iou_threshold` 控制重复候选框抑制。未启用 RMCS 周期时，`fire_gap_time` 是两次实际发射的最小间隔，
`barrel_delay` 只用于第一发出膛后切向大符副目标；显式槽位切换不会重新开始 `fire_gap_time` 计时。
`rune_idle_duration` 和 `rune_shoot_duration` 控制 RMCS 风格的能量机关空闲/开火周期；
`buff_use_rmcs_fire_cycle` 为真时，实际开火窗口由该周期决定，`fire_gap_time` 仅作为兼容回退。

### Buff 颜色约定

串口 `my_color` 表示我方颜色：`0` 为蓝方，`1` 为红方。Buff 与我方同色，而自瞄识别敌方颜色，因此两条识别链的颜色关系固定如下：

| `my_color` | 自瞄识别颜色 | Buff / rm-core 识别颜色 |
|---:|---|---|
| `0` | RED | BLUE |
| `1` | BLUE | RED |

`RmCoreAdapter::updateColorConfig()` 当前的 `0→BLUE、1→RED` 是正确映射，不要反转。离线回放没有串口颜色时，YAML 的 `enemy_color` 表示自瞄敌方颜色，Buff 取其相反颜色。`color_threshold_red` 和 `color_threshold_blue` 只控制各自的二值化阈值，不改变上述颜色关系。
