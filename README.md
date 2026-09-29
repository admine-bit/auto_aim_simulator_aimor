感谢河北科技大学仿真开源，该仓库提供给战队内部成员学习使用。

## 本机编译与运行（Ubuntu 24.04）

环境已配置为 Rust 1.98.1、Bevy 0.19、OpenVINO 2024.6.0 CPU 推理，仿真与视觉通过
Talos 共享内存通信，无需安装 ROS2。Rust 位于 `~/.cargo` 和 `~/.rustup`，补充依赖位于
仓库内被 Git 忽略的 `.local-deps/`，不修改系统库。

在仓库根目录重新编译仿真器及两个视觉入口：

```bash
./build_simulation.sh
```

默认使用 3 个编译任务，可用 `BUILD_JOBS=2 ./build_simulation.sh` 调整。
此脚本依赖已配置的本机 `.local-deps/`，不是新机器的自动安装脚本。

一键启动仿真器与视觉程序：

```bash
./bevy_robomaster_simulator/start_simulation.sh
```

运行 `/home/ad/code_game/imca_vision_26aim` 中保持原有海康相机和串口接口的代码：

```bash
./bevy_robomaster_simulator/start_legacy_vision.sh
```

该入口从目标目录复制源码并在仿真器目录内编译和运行，目标目录保持只读。

只运行仿真器：

```bash
source simulation_env.sh
cd bevy_robomaster_simulator
cargo run --release --locked --bin daedalus
```

`F3` 切换视角，`F5` 切换自瞄，`Esc` 释放鼠标。一键启动默认关闭视觉调试窗口；
需要时使用 `IMCA_VISION_DEBUG_WINDOWS=1 ./bevy_robomaster_simulator/start_simulation.sh`。
终端按 `Ctrl+C` 停止联调程序。

本地补充依赖包括 ALSA/udev 开发库、spdlog、nlohmann-json、Ceres、SuiteSparse、
glog/gflags、libunwind，以及 [OpenVINO 2024.6 Ubuntu 24.04 官方 C++ SDK](https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/)。
`simulation_env.sh` 配置这些库的头文件、构建搜索路径和运行搜索路径；启动脚本会自动加载它。
