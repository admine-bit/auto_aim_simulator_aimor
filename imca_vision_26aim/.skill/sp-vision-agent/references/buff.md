# auto_buff 与相机问题参考

## 当前运行链路

修改前必须重新确认实际调用链。编写本参考文档时，`standard_mpc` 使用的链路为：

```text
standard_mpc
  -> Buff_Detector
  -> RmCoreAdapter
  -> rm_vision_core RuneDetector
  -> PowerRune
  -> auto_buff::Solver PnP
  -> BigTarget/SmallTarget
  -> Aimer
```

`YOLO11_BUFF` 当时虽然参与编译，但没有运行时调用者。不能默认认为替换它的模型就会改变比赛中的实际行为，必须重新搜索当前调用关系。

推理后端在 `cmake/inference_backend.cmake` 中于编译期选择。TensorRT 读取 `yolo11_buff_engine_path`，OpenVINO 读取 `model`。TensorRT 构建不会直接使用 CPU INT8 ONNX 文件。修改全局推理后端前，先检查它对自瞄和分类器的影响。

## 离线视频回放

- 设置 start-index 或执行跳转后，必须始终让 AVI 第 `N` 帧与 TXT 第 `N` 行姿态对应。
- 不要完全相信 `CAP_PROP_FRAME_COUNT`：有效的 MJPEG 文件也可能返回 0。可以用解析出的姿态行数确定进度条上限，但视频是否提前结束仍以实际读取结果为准。
- 反复使用 `CAP_PROP_POS_FRAMES` 跳转可能造成卡死、解码错误或黑屏。更稳妥的方法是释放并重新打开 AVI、将 TXT 回到开头，然后依次 `grab()` 并消费姿态行，直到目标索引。
- 进度条回调必须保持轻量。采用“拖动只选择位置、自动暂停、按空格应用跳转”，不要在拖动期间实时解码或重建检测器。
- 创建 OpenCV 进度条时传入空值指针，并显式读取当前位置。
- 将录制时间的秒数转换为时间点时，不要先缩窄成 32 位微秒整数：

```cpp
t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
       std::chrono::duration<double>(t));
```

- 结果窗口缩放只用于显示。已经接受的中间显示比例是 `0.75`，不要为了改变窗口大小而修改推理分辨率。

## 生命周期与内存

不要给 `auto_buff::Solver` 重新赋值；它持有 const 物体点，因此赋值运算符已被删除。需要更新时，只调整它的坐标变换。

视频跳转时避免反复构造 `Buff_Detector`。新的 rm-core 特征图可能与旧对象同时存在，造成较大的瞬时内存峰值。如果日志刚输出一次新的 `[RmCoreAdapter]` 初始化，随后 shell 显示“已杀死”，更符合 Linux OOM 终止，而不只是普通黑屏。

应在原对象上重置检测器历史：

```cpp
detector.getAdapter().reset();
```

该重置接口在复用检测器实例的同时，清空 `feature_nodes_` 和 `last_powerrune_`。

## 跨局旋转方向

当 `abs(clockwise_) > 50` 后，`Voter::vote()` 不再改变方向。如果 `BigTarget` 生命周期与进程相同，它会一直保留第一局判断出的方向，除非显式重置状态。

进入新一局打符时，应同时重置相互关联的状态：

- adapter 的特征历史和上一帧 PowerRune；
- `SmallTarget` 和 `BigTarget`，包括 voter、EKF 和正弦拟合器；
- `Aimer` 的目标切换和发射状态；
- 连续丢符计数。

主要边界应使用进入 `SMALL_BUFF`/`BIG_BUFF` 的时刻。如果两场比赛之间控制器可能一直保持 `BIG_BUFF`，还应在连续超过 50 个打符帧没有有效 `PowerRune` 后重置。

## 新 YOLO11 Pose 模型约定

已检查的新 ONNX 模型元数据为：

```text
input:  images  float32 [1, 3, 640, 640]
output: output0 float32 [1, 23, 8400]
task: pose
class names: {0: buff}
kpt_shape: [6, 3]
```

每个候选位置包含：

```text
[cx, cy, w, h, class_conf,
 kpt0_x, kpt0_y, kpt0_conf,
 ...,
 kpt5_x, kpt5_y, kpt5_conf]
```

关键点 `j` 对应通道 `5 + 3*j`、`6 + 3*j` 和 `7 + 3*j`。旧 `[1,17,8400]` 解析器使用六组 `(x,y)` 和步长 2，与新输出结构不兼容。

必须严格保留模型关键点顺序：

- `kpt0`：距离 R 标中心最远的 target 角点；
- `kpt1..kpt3`：从 `kpt0` 开始逆时针排列的其余 target 角点；
- `kpt4`：R 标中心；
- `kpt5`：流水灯中心。

适配 `PowerRune` 时，`FanBlade.points` 按原顺序保留完整 `kpt0..kpt5`，将
`r_center` 映射到 `kpt4`。`kpt5` 的流水灯中心对应 Buff 坐标 `(0,0,0.344)`；
`FanBlade.center` 则由 `kpt0-kpt2` 与 `kpt1-kpt3` 两条对角线的交点计算，对应
`(0,0,0.700)`，两者不能混用。PnP 的二维点顺序为 target 四角、R 中心、流水灯中心，
与对应的六个三维物理点一一对应；六点全部参与
`SOLVEPNP_IPPE`。启用发射前，必须使用真实标注图像确认物理三维点顺序与二维关键点顺序一致。

现有大符链路使用 `detect_dual()`。通常只输出一个 buff 目标的模型无法隐式保留主目标/副目标语义。接入前必须明确选择以下方案之一：只返回主目标 `{result, nullopt}`、定义多检测框的主副目标语义，或者采用 rm-core 混合回退。

## 阈值

新解析器使用的初始值为：

```cpp
ConfidenceThreshold = 0.5;
KeypointThreshold = 0.5;
IouThreshold = 0.4;
```

- 提高类别置信度可以减少误检；远距离或模糊目标漏检时可以降低。
- 当前关键点规则要求六个点全部通过阈值。提高阈值能改善 PnP 可靠性，但会丢弃更多帧。
- 降低 NMS IoU 会更强地抑制重复框；提高它会保留相互靠近或重叠的候选框。只取最高分且不执行 NMS 的路径不受该参数影响。
- 每次只调整一个参数。画面上能看到检测框但程序没有返回目标时，可能是关键点置信度筛选造成的。

## 相关配置与相机现象

decider YAML 中的 `mode` 只影响自瞄优先级，不影响打符。合法值是数字 `1` 或 `2`；`mode: 1ss` 无法通过 `yaml["mode"].as<double>()` 转换。`standard_mpc` 不会构造 `omniperception::Decider`。

即使只有一个相机并且使用连续采集，反复出现 HikRobot `MV_CC_GetImageBuffer failed: 0x80000007` 仍表示取图缓冲区超时。Trigger Off 和 Continuous Acquisition 是必要条件，但不能排除其他问题。修改检测器代码前，依次检查曝光时间与超时设置、USB 控制器/带宽/供电/线材、像素格式、竞争进程和 SDK 日志。
