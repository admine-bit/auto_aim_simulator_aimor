# model/ —— 推理模型文件

> 当前 Buff 部署模型已经替换为 `model/yolo11_buff.engine`（输入 `[1,3,640,640]`，
> 输出 `[1,23,8400]`，6 个关键点均为 `(x,y,confidence)`）。OpenVINO 对应文件位于
> `assets/yolo11_buff_int8.xml/.bin`。下文关于旧 `[1,17,8400]` 模型及传统检测器的内容
> 仅作为历史迁移记录，不代表当前运行链路。

本目录打包了自瞄(yolov5)、打符(yolo11_buff) 与数字分类(tiny_resnet) 三个模型。

| 文件 | 用途 | 输入 | 输出 | 来源 |
|------|------|------|------|------|
| `yolov5.onnx` | 装甲板检测 | `[1,3,640,640]` | `[1,25200,22]` | 由 `assets/yolov5.xml` 转换，数值已比对一致 |
| `yolov5.engine` | 装甲板检测(TRT) | 同上 | 同上 | 由 `yolov5.onnx` 构建，**FP16** |
| `yolo11_buff.onnx` | 打符检测(pose) | `[1,3,640,640]` | `[1,17,8400]` | 由 `yolo11_buff.pt` 用 ultralytics 导出(opset12) |
| `yolo11_buff.engine` | 打符检测(TRT) | 同上 | 同上 | 由 `yolo11_buff.onnx` 构建，**FP16** |
| `tiny_resnet.onnx` | 数字分类 | `[1,1,32,32]` | 9 类 | 原样保留，由 cv::dnn 加载（不经 TensorRT） |

## ⚠️ engine 文件不可跨设备

`.engine` 与 **GPU 架构 + TensorRT 版本**绑定。本目录当前的 `.engine` 已是在
**Jetson Orin Nano (aarch64) + TensorRT 10.16** 上重新构建的 **FP16** 引擎
（旧的 x86_64/TRT10.10 引擎已备份为 `*.engine.x86.bak`）。

换到其它设备（不同 GPU 架构或 TensorRT 版本）部署时，请在目标机上用其自带的
TensorRT 由 `.onnx` 重新构建：

```bash
trtexec --onnx=model/yolov5.onnx      --fp16 --saveEngine=model/yolov5.engine
trtexec --onnx=model/yolo11_buff.onnx --fp16 --saveEngine=model/yolo11_buff.engine
```

构建/验证流程与本次改动详见 [`TRT_MIGRATION.md`](TRT_MIGRATION.md)。

## 备注

- 打符模型 `yolo11_buff` 的输出为 `[1,17,8400]`，其中
  `17 = 4(框 cx,cy,w,h) + 1(类别分数, names={0:'buff'}) + 6 关键点 × 2(x,y)`。
  **已由 onnx 内嵌元数据 `kpt_shape=[6,2]` 实测确认**（无关键点可见度通道）。
  C++ 后处理已按 6×2 适配，详见 [`TRT_MIGRATION.md`](TRT_MIGRATION.md)。
  > 注：本文件早先写的"4关键点×3"为笔误，实际为 6 关键点 × 2，已更正。
# TensorRT 迁移与打符模型适配说明

> 记录本次 OpenVINO → TensorRT 迁移中：ONNX→engine 的构建、`yolo11_buff` 打符模型的
> 后端迁移与输出解析适配，以及对应的代码改动。
>
> 环境：Jetson Orin Nano（aarch64）· TensorRT **10.16.2**（CUDA 13.2）· OpenCV-CUDA 4.13（`/opt/opencv-cuda`）

---

## 1. ONNX → TensorRT engine 构建

`model/` 下需要走 TensorRT 的 onnx 已全部（重新）构建为 **FP16** 引擎。

| onnx | engine | 输入 | 输出 | 说明 |
|------|--------|------|------|------|
| `yolov5.onnx` | `yolov5.engine` | `1×3×640×640` | `1×25200×22` | 装甲板检测，FP16 |
| `yolo11_buff.onnx` | `yolo11_buff.engine` | `1×3×640×640` | `1×17×8400` | 打符 pose，FP16 |
| `tiny_resnet.onnx` | —（不转） | `1×1×32×32` | 9 类 | 数字分类，由 `cv::dnn::readNetFromONNX` 加载，**不走 TensorRT** |

构建命令：

```bash
trtexec --onnx=model/yolov5.onnx      --fp16 --saveEngine=model/yolov5.engine
trtexec --onnx=model/yolo11_buff.onnx --fp16 --saveEngine=model/yolo11_buff.engine
```

**为什么要重新构建**：`.engine` 与 GPU 架构 + TensorRT 版本绑定。原 `.engine` 是
x86_64 / TRT10.10 构建的，在本机反序列化直接报错：

```
Serialization assertion ... Platform specific tag mismatch detected.
TensorRT plan files are only supported on the target runtime platform they were created on.
```

旧引擎已备份：`model/yolov5.engine.x86.bak`、`model/yolo11_buff.engine.x86.bak`
（源 onnx 保留，需要时可在 x86 上再生成）。

**验证**（两者均 `PASSED` 且 I/O 形状与下游代码一致）：

```bash
trtexec --loadEngine=model/yolov5.engine      --noDataTransfers --iterations=1 --duration=0
trtexec --loadEngine=model/yolo11_buff.engine --noDataTransfers --iterations=1 --duration=0
```

---

## 2. 打符模型 `yolo11_buff` 的后端迁移 + 解析适配

### 2.1 背景

- `auto_aim` 的 yolov5/yolov8/yolo11 已通过 `make_infer_engine`（`use_rt` 开关）支持
  OpenVINO / TensorRT 两种后端；而 `auto_buff` 的 `YOLO11_BUFF` 类仍是**纯 OpenVINO**，
  且**未被工程任何地方调用**（活跃打符流水线走的是 `rm_vision_core` 的传统 CV 检测器
  `RuneDetector`）。它也是 `auto_buff` 还在链接 `openvino::runtime` 的唯一原因。
- 本次按"**迁移到 TRT 后端 + 修正输出解析**"的范围处理：把该类改用 TensorRT
  （复用 `auto_aim::TRTEngine`），并把输出解析对齐到引擎真实格式。**不改动现有流水线**
  （仍跑传统 CV，无运行时行为变化），为后续接入做好准备。

### 2.2 ⚠️ 关键修正：输出是 **6 关键点 × 2**，不是 4×3

`model/README.md` 早先写的 "4 关键点 × 3" 经核实**有误**。引擎输出 `[1,17,8400]` 的真实结构：

```
17 = 4(框 cx,cy,w,h) + 1(类别分数) + 6 关键点 × 2(x,y)
```

证据（两条独立验证）：

1. **onnx 内嵌元数据**：`kpt_shape=[6, 2]`、`kpt_names={0:['0'..'5']}`、`names={0:'buff'}`
   （ultralytics 导出时写入；本模型训练时关键点维度为 2，**无可见度/conf 通道**）。
2. **运行时原始输出**：取分数最高 anchor 的 17 个通道值，ch5–ch16 这 12 个值全部在
   坐标量级（~400），**无任何一个落在 sigmoid 可见度应有的 [0,1] 区间** → 排除"×3 含 conf"。

> 这恰好是旧 `get_onecandidatebox` 里 `NUM_POINTS=6`、步长 2 的假设。因此本次把
> `NUM_POINTS` 定为 **6**、`KPT_DIM` 定为 **2**。

### 2.3 改动文件

#### `tasks/auto_buff/yolo11_buff.hpp`（重写）

- 移除全部 OpenVINO 成员（`ov::Core / Model / CompiledModel / InferRequest / Tensor`），
  改为持有推理后端抽象：

  ```cpp
  #include "tasks/auto_aim/inference/infer_engine.hpp"
  ...
  std::unique_ptr<auto_aim::InferEngine> engine_;  // TensorRT 推理后端
  static constexpr int INPUT_W = 640, INPUT_H = 640;
  static constexpr int NUM_POINTS = 6;  // onnx kpt_shape=[6,2]
  static constexpr int KPT_DIM = 2;     // 每点 (x,y)，无可见度通道
  ```

- 新增 `cv::Mat letterbox(const cv::Mat&, double& scale) const` 助手；删除 OpenVINO 专用的
  `convert / fill_tensor_data_image / printInputAndOutputsInfo` 声明。公共接口
  （`get_multicandidateboxes / get_onecandidatebox / Object`）保持不变。

#### `tasks/auto_buff/yolo11_buff.cpp`（重写）

- 构造：从配置读取引擎路径（键 `yolo11_buff_engine_path`，缺省回退
  `model/yolo11_buff.engine`），构造 `auto_aim::TRTEngine`：

  ```cpp
  const std::string engine_path = yaml["yolo11_buff_engine_path"]
      ? yaml["yolo11_buff_engine_path"].as<std::string>()
      : std::string("model/yolo11_buff.engine");
  engine_ = std::make_unique<auto_aim::TRTEngine>(engine_path, INPUT_H, INPUT_W);
  ```

- 预处理：`letterbox` 等比缩放到 640×640、左上角对齐（与 `auto_aim/yolos/yolo11.cpp` 一致），
  `engine_->infer(input)` 返回 `cv::Mat(17, 8400)`（按 `(channel, anchor)` 访问）。
  `TRTEngine` 内部完成 BGR→RGB / /255 / NCHW 的 GPU 预处理。

- 解析（两个函数统一为 6×2，坐标除以前向 `scale` 还原到原图）：

  ```cpp
  const float score = det_output.at<float>(4, i);           // ch4 = 类别分数
  // 框：ch0..3
  box.x = (cx - 0.5f*ow) / scale;  ...
  // 关键点：ch5..16 = 6 × (x,y)
  for (int j = 0; j < NUM_POINTS; ++j) {
    float x = det_output.at<float>(5 + j*KPT_DIM + 0, i) / scale;
    float y = det_output.at<float>(5 + j*KPT_DIM + 1, i) / scale;
    kpts.emplace_back(x, y);
  }
  ```

  `get_multicandidateboxes` 用 `cv::dnn::NMSBoxes` 取多框；`get_onecandidatebox` 取
  ch4 最高的单框。保留原有绘制逻辑（已适配 6 点）。

#### `tasks/auto_buff/CMakeLists.txt`

- `auto_buff` 不再直接依赖 OpenVINO（TRT/CUDA 经 `auto_aim` 间接提供）：
  删除 `set(OpenVINO_DIR ...)`、`find_package(OpenVINO REQUIRED)`，
  链接行去掉 `openvino::runtime`：

  ```cmake
  target_link_libraries(auto_buff auto_aim VisCore_rune_detector ${CERES_LIBRARIES})
  ```

#### `CMakeLists.txt`（顶层）

- `yolo11_buff` 现引用 `auto_aim::TRTEngine`，故链接 `auto_buff` 的可执行必须同时链接
  `auto_aim`。给原先漏链的 3 个目标补上 `auto_aim`（与 `standard` 等其它目标一致）：

  ```
  auto_buff_debug   ：+ auto_aim
  auto_buff_debug_mpc：+ auto_aim
  auto_buff_test    ：+ auto_aim
  ```

---

## 3. 配置说明

- 打符引擎路径键：`yolo11_buff_engine_path`（可选，缺省 `model/yolo11_buff.engine`）。
  原配置里的 `model: assets/yolo11_buff_int8.xml`（OpenVINO 路径）现已不再被读取。
- **本次按要求搁置的改动**（属于把 `auto_aim` 也切到 TRT 的范畴，未做）：
  - 把各 config 的 `use_rt` 改为 `true`；
  - 把 `yolov5_engine_path` 由 `assets/yolov5.engine`（不存在）改为 `model/yolov5.engine`。
  需要让 `auto_aim` 走 TensorRT 时再处理这两处。

---

## 4. 验证情况

- ✅ 两个 engine 在本机 `trtexec --loadEngine` 反序列化通过，I/O 形状与下游代码匹配。
- ✅ `auto_buff_debug / auto_buff_debug_mpc / auto_buff_test` 在 `build_cuda` 增量编译 + 链接通过。
- ✅ 临时冒烟程序直接用 `TRTEngine` 跑 `yolo11_buff.engine`：输出 `17×8400`，
  通道结构与 6×2 假设一致（用于验证后已删除）。
- ⚠️ 未在"真实打符整帧"上验证检测精度——仓库内无打符场景素材（`assets/standard_fanblade.jpg`
  为单扇叶裁切图，模型分数≈0 属正常）。接入流水线/拿到打符视频后建议复测。

---

## 5. 后续 TODO（本次未做）

1. 将 TRT 版 `YOLO11_BUFF` 真正接入打符流水线（替换/补充传统 CV `RuneDetector`）——
   涉及关键点顺序、PnP 解算、`rm_vision_core` 特征节点/tracker 体系，改动较大。
2. 用真实打符数据验证 6 关键点的语义与顺序（对应扇叶角点 / R 标）。
