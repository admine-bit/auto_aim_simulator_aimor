#ifndef AUTO_BUFF__TARGET_HPP
#define AUTO_BUFF__TARGET_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "buff_detector.hpp"
#include "buff_type.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{
// 槽间相位固定 72°（规则常量，不进配置）。
constexpr double SLOT_ANGLE = 2.0 * CV_PI / 5.0;

// ================= 共享滤波器：一个 Buff 整体只有一份 =================
//
// 设计不变式（详见重构方案）：
// - 换槽（主↔副、激活 target 随机换位）只改“当前 slot 下标”，三个滤波器的
//   状态与协方差零变化；
// - 每个滤波器每帧至多一次 update，观测全部来自五点 PnP 的共享观测；
// - kpt4 永不进入任何滤波器。

/// R 的世界系三维位置：随机游走 KF + 创新门控。
/// 场地上 R 固定，过程噪声只吸收 IMU 慢漂/底盘移动。
class RCenterFilter
{
public:
  void reset();
  bool initialized() const { return initialized_; }
  void predict(double dt);
  void update(const Eigen::Vector3d & xyz_obs, bool good_quality);
  Eigen::Vector3d xyz() const { return x_; }

  // 可调参数（默认值可直接跑；yaml 有键时由 Target 构造注入）
  double measurement_std_m{0.05};
  double process_std_m_sqrt_s{0.08};
  // 7 m 处的创新门限；实际门限随距离线性缩放并夹在 [0.5×, 3×] 之间。
  double innovation_gate_m{0.30};
  // 重定位要求连续一致的帧数，以及收敛所需的最少更新次数。
  // 曾经还有第三条「连续拒绝强制逃生」，已删除：打符场景下连续拒绝的成因是
  // 观测坏而不是模型漂，强制接受会把 R 瞬移数米、让车疯转。
  int relocate_min_streak{3};
  int relocate_min_updates{30};

  // —— 只读诊断量：不参与任何判决，仅供回放归因 ——
  // 径向分量是判别 IPPE 镜像解的关键：镜像分支会让创新成串同号，
  // 而纯 PnP 噪声应当零均值抖动。
  double last_innovation_m() const { return last_innovation_m_; }
  double last_innovation_radial_m() const { return last_innovation_radial_m_; }
  bool last_gated() const { return last_gated_; }
  bool last_relocated() const { return last_relocated_; }
  int gate_streak() const { return gate_streak_; }
  int update_count() const { return update_count_; }

private:
  bool initialized_{false};
  Eigen::Vector3d x_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d P_{Eigen::Matrix3d::Zero()};
  // 上一帧被门控拒绝的观测；快速重定位要求连续两帧大创新且彼此一致，
  // 防止两支互相矛盾的 IPPE 镜像解轮流出现时把 R 拖来拖去。
  std::optional<Eigen::Vector3d> last_gated_obs_;

  double last_innovation_m_{0.0};
  double last_innovation_radial_m_{0.0};
  bool last_gated_{false};
  bool last_relocated_{false};
  int gate_streak_{0};
  int update_count_{0};
};

/// 符面 yaw。v1 锁 pitch=0，与 point_buff2world/reproject_buff 模型全链路一致。
class PlanePoseFilter
{
public:
  void reset();
  bool initialized() const { return initialized_; }
  void predict(double dt);
  void update(double yaw_obs, bool good_quality);
  double yaw() const { return yaw_; }

  double measurement_std_rad{0.03};
  double process_std_rad_sqrt_s{0.05};
  double innovation_gate_rad{0.35};

private:
  bool initialized_{false};
  double yaw_{0.0};
  double p_{0.0};
  int gate_streak_{0};
};

/// 整个 Buff 唯一的旋转相位模型：连续基础相位 θ + 每槽固定 72° 偏移。
/// 小符：|dθ/dt| = π/3（规则常量，不估计大小，只投方向票）。
/// 大符：spd = a·sin(ω·t+φ0)+2.09−a，a∈[0.780,1.045]、ω∈[1.884,2.000] 硬夹紧，
///       φ0 由分阶段最小二乘对解卷绕连续 θ 拟合（不拟合逐帧差分速度）。
///
/// 槽偏移是**刚体推导**而不是逐槽学习的：五片扇叶是同一个刚体上固定相隔 72° 的五个位置，
/// 所以 slot_offset(k) = sign × (k − base_slot) × 72°，只需要一个基准槽和一个方向符号。
/// 旧实现给每个槽独立"学一次、此后永不更新"，一旦 detector 重新洗牌槽号
/// （丢失超时后 last_powerrune_ 被清空），旧偏移就会指向另一片物理扇叶而不自知。
class PhaseModel
{
public:
  enum class Mode { SMALL, BIG };

  explicit PhaseModel(Mode mode) : mode_(mode) { reset(); }
  void reset();
  bool initialized() const { return initialized_; }

  /// 输入某槽的原始 PnP roll。内部完成：槽位偏移推导、θ 观测解算、
  /// 创新门控（单帧拒绝/连续重锚）、方向投票、大符样本与拟合。
  /// slot_epoch 变化表示 detector 重新自举了槽号体系，本模型据此重设基准槽。
  void observe(
    std::size_t slot, double roll_obs, double t, bool good_quality,
    std::uint64_t slot_epoch = 0);

  /// 连续模型外推：任意时刻的基础相位 / 任意槽的 roll / 带符号速度。
  double theta_at(double t) const;
  double roll_at(std::size_t slot, double t) const;
  double speed_at(double t) const;
  /// t0 → t1 的相位增量（含方向，未取模），供 roll 覆写外推使用。
  double advance(double t0, double t1) const;

  int direction() const { return dir_votes_ >= 0 ? 1 : -1; }
  bool slot_known(std::size_t slot) const;
  /// 该槽相对基准槽的固定偏移（刚体推导，不是学出来的）。未初始化时返回 nullopt。
  std::optional<double> slot_offset(std::size_t slot) const;
  /// 槽差符号：图像槽号递增方向对应 roll 的正负。由 Solver 的整符解算注入。
  /// 它一变，所有槽的偏移就整体换向，因此必须把基准槽和相位一起重置——
  /// 否则会出现"偏移已经换向、θ 还按旧向累计"的半新半旧状态。
  void set_slot_sign(int sign);
  int slot_sign() const { return slot_sign_; }

  double a() const { return a_; }
  double w() const { return w_; }
  double phi0() const { return phi0_; }

  // —— 只读诊断量：不参与任何判决，仅供回放归因 ——
  // slot_offset 错位 72° 时，本帧 innovation 会跳到 ~72° 而 slot 号可能完全没变；
  // theta_unwrapped 的阶跃会直接污染大符拟合历史。
  double last_theta_obs() const { return diag_theta_obs_; }
  double last_theta_pred() const { return diag_theta_pred_; }
  double last_innovation() const { return diag_innovation_; }
  double last_slot_offset() const { return diag_slot_offset_; }
  bool last_gated() const { return diag_gated_; }
  bool last_reanchored() const { return diag_reanchored_; }
  bool last_slot_learned() const { return diag_slot_learned_; }
  int gate_streak() const { return gate_streak_; }
  int dir_votes() const { return dir_votes_; }
  double theta_unwrapped() const { return theta_unwrapped_; }
  /// 本帧与上一次被接受的相位观测之间的间隔（s）。判断换组黑屏实际多长、
  /// 有没有进入解卷绕会混叠的区间。
  double last_obs_dt() const { return diag_obs_dt_; }
  /// 本帧解卷绕增量（rad）。72° = 槽偏移错位；任意大跳 = 解卷绕混叠；
  /// 正常应当约等于单帧转角。
  double last_unwrap_step() const { return diag_unwrap_step_; }
  /// 本帧解卷绕增量与模型推进量之差（rad）。假阶跃的直接判据。
  double last_unwrap_residual() const { return diag_unwrap_residual_; }
  /// 本帧是否因为观测间隔过长/出现假阶跃而清空了拟合历史。
  bool last_history_reset() const { return diag_history_reset_; }
  /// 本帧把多少个整槽（±72°）折叠进了基准槽。非零 = 槽号记账与物理扇叶错位了这么多格。
  /// 偶发几次属正常（换目标瞬间）；持续非零说明 detector 的槽号关联本身在抖，
  /// 折叠只是让相位模型不受其害，根因仍在 assign_slots。
  int last_fold_steps() const { return diag_fold_steps_; }
  /// 累计折叠次数。单调增，用来判断"偶发"还是"持续"。
  int fold_count() const { return diag_fold_count_; }
  // 最近一次大符拟合的残差 RMS（rad）与参与拟合的样本数。
  double last_fit_rms() const { return diag_fit_rms_; }
  std::size_t last_fit_samples() const { return diag_fit_samples_; }
  double fit_span() const { return diag_fit_span_; }

  double phase_innovation_gate_rad{25.0 / 57.3};  // 必须 < 半槽 36°
  // 解卷绕可信的最大观测间隔（s）。超过它模型外推本身不可靠，宁可清空拟合历史重开，
  // 也不能把一个猜出来的大步长累进 theta_unwrapped_ 污染整个拟合窗口。
  double unwrap_max_dt_s{0.6};
  // 大符拟合窗口（s）与开拟所需的最小样本跨度（s）。
  // 正弦周期 3.14~3.33 s，窗口 4 s 约覆盖 1.2 个周期。
  double fit_window_s{4.0};
  double fit_min_span_s{1.5};

private:
  double unsigned_phase(double t) const;  // Φ(t)：速度大小的积分（单调增）
  void fit_big_params(double now);

  Mode mode_;
  bool initialized_{false};
  double theta_ref_{0.0};
  double t_ref_{0.0};
  double p_{0.0};  // θ 方差
  int dir_votes_{0};
  int gate_streak_{0};
  // 槽偏移由刚体关系推导：offset(k) = slot_sign_ × (k − base_slot_) × 72°。
  // base_slot_ 是当前纪元里第一个被观测到的槽；纪元变化时重设。
  std::size_t base_slot_{PowerRune::SLOT_COUNT};
  std::uint64_t slot_epoch_{0};
  bool has_slot_epoch_{false};
  int slot_sign_{1};
  std::optional<double> last_theta_obs_;
  std::optional<double> last_theta_obs_t_;

  // 大符正弦参数（越界=夹紧，永不因此重建）
  double a_{0.9125};
  double w_{1.942};
  double phi0_{0.0};
  double t0_{0.0};
  double last_fit_t_{-1.0};
  double theta_unwrapped_{0.0};
  std::deque<std::pair<double, double>> history_;  // (t, 解卷绕连续 θ)

  // 只读诊断量，reset() 时一并清零。
  double diag_theta_obs_{0.0};
  double diag_theta_pred_{0.0};
  double diag_innovation_{0.0};
  double diag_slot_offset_{0.0};
  bool diag_gated_{false};
  bool diag_reanchored_{false};
  bool diag_slot_learned_{false};
  double diag_fit_rms_{0.0};
  std::size_t diag_fit_samples_{0};
  double diag_fit_span_{0.0};
  double diag_obs_dt_{0.0};
  double diag_unwrap_step_{0.0};
  double diag_unwrap_residual_{0.0};
  bool diag_history_reset_{false};
  int diag_fold_steps_{0};
  int diag_fold_count_{0};
};

// ================= Target 门面（保留旧公共接口） =================

class Target
{
public:
  explicit Target(PhaseModel::Mode mode) : phase_(mode) {}
  /// 从 yaml 读取可选滤波参数（键缺失时用默认值），供各主程序传入 config_path。
  Target(PhaseModel::Mode mode, const std::string & config_path);
  virtual ~Target() = default;

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp);

  /// 大符副片的 roll 也喂进相位模型（观测率翻倍），并用"同帧两片 roll 差必须是
  /// 72° 整数倍"做在线自洽性检查。必须在 get_target() 之后调用。
  /// 只喂 roll：位置与符面姿态是整符一次解算的同一份结果，重复喂会人为压小协方差。
  void observe_secondary(const std::optional<PowerRune> & secondary);

  /// 只推进相位模型的预测时基；连续调用会继续累计（与旧 EKF 语义一致）。
  void predict(double dt) { pred_offset_ += dt; }

  Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;

  bool is_unsolve() const { return unsolvable_; }

  /// 兼容旧布局的组合状态视图，供调试绘图与 Aimer 使用：
  /// [R_yaw, 0, R_pitch, R_dis, plane_yaw, roll(当前槽), spd, (a, w, phi0)]
  virtual Eigen::VectorXd ekf_x() const = 0;

  std::optional<std::size_t> target_slot() const { return last_target_slot_; }

  /// 指定物理槽在当前模型时刻的 roll。整符可视化用：五个槽全部画出来，
  /// 才能一眼看出某个槽的 72° 偏移是不是学错了（学错时两条臂会重叠、另一条空着）。
  /// 该槽偏移尚未学习时返回 nullopt，调用方据此区分"模型认为它在这"和"模型还不知道"。
  std::optional<double> roll_of_slot(std::size_t slot) const;

  /// 指定槽在当前模型时刻**相对 R 的图像圆周角**，供 Detector 的关联参考角使用。
  /// 与 roll 的关系由一次成功关联标定（image_angle ≈ angle_sign×roll + angle_bias）。
  /// 未标定或模型未初始化时返回 nullopt，Detector 自动退回上一帧像素角。
  std::optional<double> image_angle_of_slot(std::size_t slot) const;

  /// 由 Detector 在一次可信关联后回灌：把该槽的 roll 与其图像圆周角对应起来。
  void calibrate_image_angle(std::size_t slot, double image_angle);

  /// 由 Solver 的整符解算回灌槽差符号（图像槽号递增方向对应 roll 的正负）。
  /// 槽偏移是刚体推导的，这个符号是推导里唯一无法读码断言的量。
  void set_slot_sign(int sign) { phase_.set_slot_sign(sign); }

  // —— 只读诊断入口：阶段 0 归因用，不改变任何判决路径 ——
  const PhaseModel & phase() const { return phase_; }
  const RCenterFilter & r_filter() const { return r_filter_; }
  const PlanePoseFilter & plane_filter() const { return plane_filter_; }
  double obs_time() const { return obs_time_; }
  // 本帧喂给三个滤波器的 PnP 质量标志。镜像解的 RMSE 与正确解常常分不开，
  // 因此这个标志为 true 并不等于本帧姿态可信——回放时要和法向跳变一起看。
  bool last_good_quality() const { return last_good_quality_; }
  std::uint64_t last_slot_epoch() const { return last_slot_epoch_; }
  /// 同帧主副 roll 差偏离 72° 整数倍的量（rad）。非零说明槽号体系或位姿有问题。
  double last_dual_roll_mismatch() const { return last_dual_roll_mismatch_; }
  bool last_dual_roll_consistent() const { return last_dual_roll_consistent_; }

  double spd = 0;  // 调试用

  // 长时间完全丢失后的整体重置阈值（秒）。规则换组黑屏 ≤200ms，绝不能按帧数清。
  double full_reset_s{3.0};
  // 同帧主副 roll 差偏离 72° 整数倍多少就判定副片观测不可信（rad）。
  double dual_roll_tolerance_rad{12.0 / 57.3};

protected:
  Eigen::VectorXd compose_common_state() const;  // 前 6 维
  double model_time() const { return obs_time_ + pred_offset_; }
  double current_roll() const;
  void reset_all();

  PhaseModel phase_;
  RCenterFilter r_filter_;
  PlanePoseFilter plane_filter_;

  bool unsolvable_{true};
  std::optional<std::size_t> last_target_slot_;
  // clone_with_roll 兼容：以给定观测 roll 为基准、按共享相位模型外推。
  std::optional<double> roll_override_;
  double obs_time_{0.0};
  double pred_offset_{0.0};
  std::chrono::steady_clock::time_point start_timestamp_{};
  bool has_start_timestamp_{false};
  std::optional<std::chrono::steady_clock::time_point> last_seen_timestamp_;
  // 只读诊断量。
  bool last_good_quality_{false};
  std::uint64_t last_slot_epoch_{0};
  double last_primary_roll_obs_{0.0};
  double last_dual_roll_mismatch_{0.0};
  bool last_dual_roll_consistent_{true};
  // roll → 图像圆周角的标定：image_angle ≈ angle_sign×roll + angle_bias。
  // 由 Detector 在一次可信关联后回灌，用于换组黑屏期间外推关联参考角。
  std::optional<double> image_angle_bias_;
  int image_angle_sign_{1};
};

/// SmallTarget：固定 π/3 速度（规则常量），只投方向票。
class SmallTarget : public Target
{
public:
  SmallTarget() : Target(PhaseModel::Mode::SMALL) {}
  explicit SmallTarget(const std::string & config_path)
  : Target(PhaseModel::Mode::SMALL, config_path)
  {
  }

  Eigen::VectorXd ekf_x() const override;

  static constexpr double SMALL_W = CV_PI / 3;
};

/// BigTarget：规则受限正弦相位模型。
class BigTarget : public Target
{
public:
  BigTarget() : Target(PhaseModel::Mode::BIG) {}
  explicit BigTarget(const std::string & config_path)
  : Target(PhaseModel::Mode::BIG, config_path)
  {
  }

  Eigen::VectorXd ekf_x() const override;

  /// 副目标预测优先用槽位（共享相位 + 固定 72° 偏移）。
  BigTarget clone_for_slot(std::size_t slot) const
  {
    BigTarget copy = *this;
    copy.last_target_slot_ = slot;
    copy.roll_override_.reset();
    if (!copy.phase_.slot_known(slot)) copy.unsolvable_ = true;
    return copy;
  }

  /// 兼容旧调用：以副目标观测 roll 为基准、沿共享相位模型外推。
  BigTarget clone_with_roll(double new_roll) const
  {
    BigTarget copy = *this;
    copy.roll_override_ = new_roll;
    return copy;
  }
};

}  // namespace auto_buff
#endif
