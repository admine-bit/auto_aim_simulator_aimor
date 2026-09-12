#include "buff_target.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace auto_buff
{
namespace
{
double read_optional(const YAML::Node & yaml, const char * key, double fallback)
{
  if (!yaml[key]) return fallback;
  const double value = yaml[key].as<double>();
  return std::isfinite(value) ? value : fallback;
}
}  // namespace

Target::Target(PhaseModel::Mode mode, const std::string & config_path) : phase_(mode)
{
  try {
    const auto yaml = YAML::LoadFile(config_path);
    r_filter_.measurement_std_m =
      read_optional(yaml, "buff_r_measurement_std_m", r_filter_.measurement_std_m);
    r_filter_.process_std_m_sqrt_s =
      read_optional(yaml, "buff_r_process_std_m_sqrt_s", r_filter_.process_std_m_sqrt_s);
    r_filter_.innovation_gate_m =
      read_optional(yaml, "buff_r_innovation_gate_m", r_filter_.innovation_gate_m);
    r_filter_.relocate_min_streak = static_cast<int>(read_optional(
      yaml, "buff_r_relocate_min_streak", r_filter_.relocate_min_streak));
    r_filter_.relocate_min_updates = static_cast<int>(read_optional(
      yaml, "buff_r_relocate_min_updates", r_filter_.relocate_min_updates));
    phase_.phase_innovation_gate_rad =
      read_optional(yaml, "buff_phase_innovation_gate_deg", 25.0) / 57.3;
    phase_.unwrap_max_dt_s =
      read_optional(yaml, "buff_unwrap_max_dt_s", phase_.unwrap_max_dt_s);
    phase_.fit_window_s = read_optional(yaml, "buff_fit_window_s", phase_.fit_window_s);
    phase_.fit_min_span_s =
      read_optional(yaml, "buff_fit_min_span_s", phase_.fit_min_span_s);
    dual_roll_tolerance_rad =
      read_optional(yaml, "buff_dual_roll_tolerance_deg", 12.0) / 57.3;
    full_reset_s = read_optional(yaml, "buff_full_reset_s", full_reset_s);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Target] 读取 buff 滤波配置失败，使用默认值: {}", e.what());
  }
}

// ================= RCenterFilter =================

void RCenterFilter::reset()
{
  initialized_ = false;
  last_gated_obs_.reset();
  last_innovation_m_ = 0.0;
  last_innovation_radial_m_ = 0.0;
  last_gated_ = false;
  last_relocated_ = false;
  gate_streak_ = 0;
  update_count_ = 0;
}

void RCenterFilter::predict(double dt)
{
  if (!initialized_ || dt <= 0.0) return;
  P_ += process_std_m_sqrt_s * process_std_m_sqrt_s * dt * Eigen::Matrix3d::Identity();
}

void RCenterFilter::update(const Eigen::Vector3d & xyz_obs, bool good_quality)
{
  last_relocated_ = false;
  if (!xyz_obs.allFinite()) return;
  const double r_std = good_quality ? measurement_std_m : measurement_std_m * 2.0;
  const Eigen::Matrix3d R = r_std * r_std * Eigen::Matrix3d::Identity();

  // 首帧有效观测立即初始化立即输出，不等待收敛。
  if (!initialized_) {
    x_ = xyz_obs;
    P_ = R;
    initialized_ = true;
    last_innovation_m_ = 0.0;
    last_innovation_radial_m_ = 0.0;
    last_gated_ = false;
    gate_streak_ = 0;
    update_count_ = 1;
    return;
  }

  ++update_count_;
  const Eigen::Vector3d innovation = xyz_obs - x_;
  last_innovation_m_ = innovation.norm();
  // 世界系原点在云台，x_ 方向即视线方向：径向分量成串同号 = 深度被系统性拉偏。
  const double range = x_.norm();
  last_innovation_radial_m_ = range > 1e-6 ? innovation.dot(x_) / range : 0.0;

  // 门限随距离缩放：同样的角度误差在远处对应更大的位置偏差，固定 0.30 m 在近处
  // 太松、在远处太紧。以 7 m 为基准线性缩放，并夹在合理区间内。
  const double gate = std::clamp(
    innovation_gate_m * std::max(range, 1.0) / 7.0, innovation_gate_m * 0.5,
    innovation_gate_m * 3.0);

  if (innovation.norm() > gate) {
    last_gated_ = true;
    ++gate_streak_;
    // 单帧大创新（PnP 毛刺）拒绝。快速重定位要求：连续多帧一致的大创新、质量良好、
    // 且滤波器已经收敛过一段时间。
    //
    // 「收敛前禁用」这一条是必需的：捕获初期状态本来就不可信，此时放行重定位
    // 等于让第一个坏观测直接定义状态，之后所有正常观测反而被门控当成异常
    // ——一次坏初始化被锁成持续故障（见 w速度不稳的问题.md 第三节第 2 条）。
    const bool converged = update_count_ >= relocate_min_updates;
    const bool consistent =
      last_gated_obs_.has_value() && (xyz_obs - *last_gated_obs_).norm() < gate * 0.5;
    last_gated_obs_ = xyz_obs;

    if (good_quality && consistent && converged && gate_streak_ >= relocate_min_streak) {
      x_ = xyz_obs;
      P_ = R;
      last_gated_obs_.reset();
      gate_streak_ = 0;
      last_relocated_ = true;
      tools::logger()->debug(
        "[RCenterFilter] 连续 {} 帧一致大创新（{:.2f} m > {:.2f} m），重定位 R",
        relocate_min_streak, innovation.norm(), gate);
      return;
    }

    // 这里曾经有一个「连续拒绝逃生」：拒绝够多帧就强制接受一次观测。它是错的。
    // 连续拒绝的成因在打符场景里是**观测坏**（IPPE 镜像解成串出现），不是模型漂；
    // 强制接受等于在一串坏观测里专挑第 30 帧信，实测直接把 R 瞬移 7.2 m，
    // 瞄准点跟着跳、车疯转。宁可让 R 停在旧值等观测恢复，也不能瞬移。
    // 代价：门控理论上可能长期拒绝。但 R 停在旧值只是瞄偏，不会疯转，
    // 而且 Target::full_reset_s 在真正长时间丢失时会整体重置。
    // gate_streak_ 仍然累计，只作诊断输出，不再触发任何写入。
    return;
  }
  last_gated_ = false;
  gate_streak_ = 0;
  last_gated_obs_.reset();

  const Eigen::Matrix3d S = P_ + R;
  const Eigen::Matrix3d K = P_ * S.inverse();
  x_ += K * innovation;
  P_ = (Eigen::Matrix3d::Identity() - K) * P_;
}

// ================= PlanePoseFilter =================

void PlanePoseFilter::reset()
{
  initialized_ = false;
  gate_streak_ = 0;
}

void PlanePoseFilter::predict(double dt)
{
  if (!initialized_ || dt <= 0.0) return;
  p_ += process_std_rad_sqrt_s * process_std_rad_sqrt_s * dt;
}

void PlanePoseFilter::update(double yaw_obs, bool good_quality)
{
  if (!std::isfinite(yaw_obs)) return;
  const double r_std = good_quality ? measurement_std_rad : measurement_std_rad * 2.0;
  const double r = r_std * r_std;

  if (!initialized_) {
    yaw_ = tools::limit_rad(yaw_obs);
    p_ = r;
    initialized_ = true;
    return;
  }

  const double innovation = tools::limit_rad(yaw_obs - yaw_);
  if (std::abs(innovation) > innovation_gate_rad) {
    if (good_quality && ++gate_streak_ >= 2) {
      yaw_ = tools::limit_rad(yaw_obs);
      p_ = r;
      gate_streak_ = 0;
    }
    return;
  }
  gate_streak_ = 0;

  const double k = p_ / (p_ + r);
  yaw_ = tools::limit_rad(yaw_ + k * innovation);
  p_ *= (1.0 - k);
}

// ================= PhaseModel =================

void PhaseModel::reset()
{
  initialized_ = false;
  theta_ref_ = 0.0;
  t_ref_ = 0.0;
  p_ = 0.0;
  dir_votes_ = 0;
  gate_streak_ = 0;
  base_slot_ = PowerRune::SLOT_COUNT;
  slot_epoch_ = 0;
  has_slot_epoch_ = false;
  slot_sign_ = 1;
  last_theta_obs_.reset();
  last_theta_obs_t_.reset();
  a_ = 0.9125;
  w_ = 1.942;
  phi0_ = 0.0;
  t0_ = 0.0;
  last_fit_t_ = -1.0;
  theta_unwrapped_ = 0.0;
  history_.clear();
  diag_theta_obs_ = 0.0;
  diag_theta_pred_ = 0.0;
  diag_innovation_ = 0.0;
  diag_slot_offset_ = 0.0;
  diag_gated_ = false;
  diag_reanchored_ = false;
  diag_slot_learned_ = false;
  diag_fit_rms_ = 0.0;
  diag_fit_samples_ = 0;
  diag_fit_span_ = 0.0;
  diag_obs_dt_ = 0.0;
  diag_unwrap_step_ = 0.0;
  diag_unwrap_residual_ = 0.0;
  diag_history_reset_ = false;
  diag_fold_steps_ = 0;
  // diag_fold_count_ 是累计量，故意不在 reset() 里清零：模式切换后仍想看到总次数。
}

std::optional<double> PhaseModel::slot_offset(std::size_t slot) const
{
  if (!initialized_ || slot >= PowerRune::SLOT_COUNT || base_slot_ >= PowerRune::SLOT_COUNT)
    return std::nullopt;
  // 刚体关系：五片扇叶固定相隔 72°，所以偏移完全由槽号差决定，不需要逐槽学习。
  const int steps = static_cast<int>(slot) - static_cast<int>(base_slot_);
  return tools::limit_rad(slot_sign_ * steps * SLOT_ANGLE);
}

double PhaseModel::unsigned_phase(double t) const
{
  // Φ(t) = ∫ |spd| dt。小符 |spd|=π/3 常量；大符 |spd|=a·sin(ωt+φ0)+2.09−a。
  if (mode_ == Mode::SMALL) return (CV_PI / 3.0) * t;
  return (2.09 - a_) * t - (a_ / w_) * std::cos(w_ * t + phi0_);
}

double PhaseModel::advance(double t0, double t1) const
{
  return direction() * (unsigned_phase(t1) - unsigned_phase(t0));
}

double PhaseModel::theta_at(double t) const
{
  return tools::limit_rad(theta_ref_ + advance(t_ref_, t));
}

double PhaseModel::roll_at(std::size_t slot, double t) const
{
  const auto offset = slot_offset(slot);
  return tools::limit_rad(theta_at(t) + offset.value_or(0.0));
}

double PhaseModel::speed_at(double t) const
{
  if (mode_ == Mode::SMALL) return direction() * (CV_PI / 3.0);
  return direction() * (a_ * std::sin(w_ * t + phi0_) + 2.09 - a_);
}

bool PhaseModel::slot_known(std::size_t slot) const
{
  // 刚体推导之后，只要模型已初始化、基准槽已定，任何槽的偏移都是已知的。
  return initialized_ && slot < PowerRune::SLOT_COUNT && base_slot_ < PowerRune::SLOT_COUNT;
}

void PhaseModel::set_slot_sign(int sign)
{
  if (sign != 1 && sign != -1) return;
  if (sign == slot_sign_) return;
  slot_sign_ = sign;
  if (!initialized_) return;
  // 符号一变，slot_offset() 对每个非基准槽的返回值都换向。此前累计的 θ 与解卷绕
  // 样本是按旧向解算出来的，混用会得到半新半旧的状态。清掉重来，代价是几帧收敛，
  // 但避免了一个很难查的静默错位。基准槽由下一帧的 observe() 重新确立。
  base_slot_ = PowerRune::SLOT_COUNT;
  initialized_ = false;
  history_.clear();
  theta_unwrapped_ = 0.0;
  last_theta_obs_.reset();
  last_theta_obs_t_.reset();
  tools::logger()->debug("[PhaseModel] 槽差符号变为 {}，重置相位模型", sign);
}

void PhaseModel::observe(
  std::size_t slot, double roll_obs, double t, bool good_quality, std::uint64_t slot_epoch)
{
  if (slot >= PowerRune::SLOT_COUNT || !std::isfinite(roll_obs) || !std::isfinite(t)) return;

  diag_reanchored_ = false;
  diag_slot_learned_ = false;
  diag_history_reset_ = false;
  diag_obs_dt_ = 0.0;
  diag_unwrap_step_ = 0.0;
  diag_unwrap_residual_ = 0.0;
  diag_fold_steps_ = 0;

  // Detector 重新自举了槽号体系：同一个槽号可能已经指向另一片物理扇叶。
  // 必须把基准槽重设到本帧这一片，否则整套偏移会整体错位 72° 的倍数。
  const bool epoch_changed = has_slot_epoch_ && slot_epoch != slot_epoch_;
  if (epoch_changed) {
    // 重设基准槽 = 让本帧这一片重新定义偏移零点。θ 本身是连续的物理量，
    // 不受槽号重编号影响，所以只改基准槽、不动 theta_ref_。
    base_slot_ = slot;
    theta_ref_ = tools::limit_rad(roll_obs);
    t_ref_ = t;
    last_theta_obs_ = theta_ref_;
    last_theta_obs_t_ = t;
    diag_slot_learned_ = true;
    tools::logger()->debug("[PhaseModel] 槽号纪元变化，重设基准槽为 {}", slot);
  }
  slot_epoch_ = slot_epoch;
  has_slot_epoch_ = true;

  if (!initialized_) {
    // 首帧：当前槽定义基础相位零点，立即可输出预测（大符用规则中值参数）。
    base_slot_ = slot;
    theta_ref_ = tools::limit_rad(roll_obs);
    t_ref_ = t;
    p_ = 0.04;
    t0_ = t;
    theta_unwrapped_ = 0.0;
    last_theta_obs_ = theta_ref_;
    last_theta_obs_t_ = t;
    history_.clear();
    history_.emplace_back(t, 0.0);
    initialized_ = true;
    diag_theta_obs_ = theta_ref_;
    diag_theta_pred_ = theta_ref_;
    diag_innovation_ = 0.0;
    diag_slot_offset_ = 0.0;
    diag_gated_ = false;
    diag_slot_learned_ = true;
    return;
  }

  const double theta_pred = theta_at(t);
  double offset = slot_offset(slot).value_or(0.0);
  double theta_obs = tools::limit_rad(roll_obs - offset);
  double innovation = tools::limit_rad(theta_obs - theta_pred);

  // ============ 整槽折叠：把 72° 错位吸收进基准槽，而不是重锚 θ ============
  // 五重对称下，θ 观测与预测相差 ≈ k×72° 不代表模型漂了，而是槽号记账与物理扇叶
  // 错位了 k 格（detector 重新编号、换目标时槽号与扇叶短暂不同步、关联抖一格）。
  // 判据：模型漂移是连续量，要走到 72° 必先穿过 25° 的门限被抓住；帧间**突然**出现的
  // 整槽跳变在物理上不可能来自漂移。
  //
  // 旧做法把它当漂移处理：连续 3 帧后重锚 θ 并 history_.clear()。输出确实在 3 帧内
  // 恢复自洽（所以肉眼与弹道都正常），但每次都把大符拟合窗口清空，样本永远攒不够
  // fit_min_span_s，a/ω/φ0 就一直停在规则中值。
  //
  // 正确响应：base_new = base_old − slot_sign×k，则 offset 增加 k×72°、theta_obs
  // 减少 k×72°，创新恰好抵消。θ、theta_unwrapped_、history_ 全部保持连续。
  diag_fold_steps_ = 0;
  const int fold_steps = static_cast<int>(std::llround(innovation / SLOT_ANGLE));
  if (
    fold_steps != 0 &&
    std::abs(innovation - fold_steps * SLOT_ANGLE) < phase_innovation_gate_rad) {
    // 折叠后残差仍超门限时不折叠——那说明它既不是干净的整槽错位，也不是小漂移，
    // 交给下面的门控按真正的异常处理。
    constexpr int kCount = static_cast<int>(PowerRune::SLOT_COUNT);
    const int shifted = (static_cast<int>(base_slot_) - slot_sign_ * fold_steps) % kCount;
    base_slot_ = static_cast<std::size_t>((shifted + kCount) % kCount);
    offset = slot_offset(slot).value_or(offset);
    theta_obs = tools::limit_rad(roll_obs - offset);
    innovation = tools::limit_rad(theta_obs - theta_pred);
    diag_fold_steps_ = fold_steps;
    ++diag_fold_count_;
    tools::logger()->debug(
      "[PhaseModel] 槽号记账错位 {} 格（θ 差 {:.1f}°），基准槽 → {}，θ 与拟合历史保持连续",
      fold_steps, fold_steps * SLOT_ANGLE * 57.3, base_slot_);
  }

  diag_theta_obs_ = theta_obs;
  diag_theta_pred_ = theta_pred;
  diag_innovation_ = innovation;
  diag_slot_offset_ = offset;

  if (std::abs(innovation) > phase_innovation_gate_rad) {
    diag_gated_ = true;
    // 单帧超门限拒绝；连续多帧且质量良好说明模型漂了，直接重锚到观测。
    if (good_quality && ++gate_streak_ >= 3) {
      theta_ref_ = theta_obs;
      t_ref_ = t;
      p_ = 0.04;
      gate_streak_ = 0;
      diag_reanchored_ = true;
      // 重锚意味着模型与观测已经脱节，此前累计的解卷绕样本不能再和新基准混用。
      history_.clear();
      theta_unwrapped_ = 0.0;
      t0_ = t;
      last_theta_obs_ = theta_obs;
      last_theta_obs_t_ = t;
      diag_history_reset_ = true;
      tools::logger()->debug(
        "[PhaseModel] 相位创新连续超限({:.1f}deg)，重锚 θ 并清空拟合历史", innovation * 57.3);
    }
    return;
  }
  gate_streak_ = 0;
  diag_gated_ = false;

  // ============ 解卷绕：以模型推进量为参考，按半槽折叠 ============
  // 旧实现是 theta_unwrapped_ += limit_rad(theta_obs − last_theta_obs_)，两个毛病：
  //   1) limit_rad 按 ±180° 折叠，只在真实转动 < π 时正确。大符 2.09 rad/s 下
  //      π 只需 1.50 s，而 full_reset_s = 3.0 —— 中断 1.5~3.0 s 会静默混叠。
  //      紧邻的方向投票有 obs_dt < 0.3 的守卫，这一行却没有，明显是漏了。
  //   2) 参考点是上一帧观测，间隔越长越不可靠。
  // 改成"模型推进量 + 残差"：只要模型在这段间隔内的累计误差 < π 就正确，
  // 与间隔长短无关；间隔很短时自然退化成原来的式子。
  if (last_theta_obs_.has_value() && last_theta_obs_t_.has_value()) {
    const double obs_dt = t - *last_theta_obs_t_;
    diag_obs_dt_ = obs_dt;

    const double model_step = advance(*last_theta_obs_t_, t);
    const double residual = tools::limit_rad(theta_obs - *last_theta_obs_ - model_step);
    const double d = model_step + residual;
    diag_unwrap_step_ = d;
    diag_unwrap_residual_ = residual;

    // 方向投票只用短间隔的干净差分（换槽帧天然连续，不会投 ±72° 废票）。
    if (obs_dt > 1e-3 && obs_dt < 0.3 && std::abs(dir_votes_) < 100)
      dir_votes_ += d > 0.0 ? 1 : (d < 0.0 ? -1 : 0);

    // 假阶跃防线：间隔过长时模型外推本身不可信；残差接近半槽说明多半是槽偏移错位。
    // 两种情况都不能把这一步累进 theta_unwrapped_，否则 4 秒拟合窗口会连续吃它。
    const bool interval_too_long = obs_dt > unwrap_max_dt_s;
    const bool residual_too_large = std::abs(residual) > SLOT_ANGLE * 0.5;
    if (interval_too_long || residual_too_large) {
      history_.clear();
      theta_unwrapped_ = 0.0;
      t0_ = t;
      diag_history_reset_ = true;
      tools::logger()->debug(
        "[PhaseModel] 解卷绕不可信(dt={:.2f}s, 残差={:.1f}deg)，清空拟合历史",
        obs_dt, residual * 57.3);
    } else {
      theta_unwrapped_ += d;
    }
  }
  last_theta_obs_ = theta_obs;
  last_theta_obs_t_ = t;

  // θ 一维 KF：过程噪声随观测间隔增长；质量差的帧权重降低。
  const double dt_pred = std::max(t - t_ref_, 0.0);
  p_ += 0.02 * dt_pred;
  const double r = good_quality ? 0.01 : 0.04;
  const double k = p_ / (p_ + r);
  theta_ref_ = tools::limit_rad(theta_pred + k * innovation);
  t_ref_ = t;
  p_ *= (1.0 - k);

  // 大符：累计解卷绕 θ 样本，分阶段拟合 a/ω/φ0。
  if (mode_ == Mode::BIG && good_quality) {
    history_.emplace_back(t, theta_unwrapped_);
    while (!history_.empty() && t - history_.front().first > fit_window_s)
      history_.pop_front();
    // 开拟门槛：正弦周期 2π/ω ∈ [3.14, 3.33] s，用 0.33 s 数据去拟是不可辨识的
    // （旧实现 t-t0_>0.25 且 20 样本就开拟，60fps 下正好 0.33 s）。
    // 要求样本跨度覆盖有意义的一段弧，参数才有分辨力。
    const double span = t - history_.front().first;
    if (span > fit_min_span_s && (last_fit_t_ < 0.0 || t - last_fit_t_ > 0.25)) {
      fit_big_params(t);
      last_fit_t_ = t;
    }
  }
}

void PhaseModel::fit_big_params(double now)
{
  if (history_.size() < 20) return;

  // 下采样至 ≤60 点，控制每 0.25s 一次的拟合开销。
  std::vector<std::pair<double, double>> samples;
  const std::size_t stride = std::max<std::size_t>(1, history_.size() / 60);
  for (std::size_t i = 0; i < history_.size(); i += stride) samples.push_back(history_[i]);

  const int dir = direction();
  // 用真实样本跨度而不是 now - t0_：history_ 会因为假阶跃被清空重开，
  // t0_ 也随之更新，但两者在窗口滑动期间会脱节。
  const double span = now - history_.front().first;

  // 分阶段：<1s 时 ω 固定规则中值只拟合 a/φ0；≥1s 后在规则范围内放开 ω。
  // 参数网格天然落在规则区间内=硬夹紧；拟合永不触发重建。
  std::vector<double> a_grid{0.780, 0.846, 0.913, 0.979, 1.045};
  std::vector<double> w_grid;
  if (span < 1.0)
    w_grid = {1.942};
  else
    w_grid = {1.884, 1.913, 1.942, 1.971, 2.000};
  constexpr int kPhiSteps = 24;

  double best_sse = std::numeric_limits<double>::infinity();
  double best_a = a_;
  double best_w = w_;
  double best_phi = phi0_;

  auto evaluate = [&](double a, double w, double phi) {
    // θ_u(t)·dir ≈ C + Φ(t)；C 取残差均值（线性最优）。
    double mean_residual = 0.0;
    for (const auto & [t, theta_u] : samples) {
      const double phase = (2.09 - a) * t - (a / w) * std::cos(w * t + phi);
      mean_residual += dir * theta_u - phase;
    }
    mean_residual /= static_cast<double>(samples.size());
    double sse = 0.0;
    for (const auto & [t, theta_u] : samples) {
      const double phase = (2.09 - a) * t - (a / w) * std::cos(w * t + phi);
      const double e = dir * theta_u - phase - mean_residual;
      sse += e * e;
    }
    return sse;
  };

  // 当前参数作为基准候选，避免网格分辨率不足时解来回跳。
  best_sse = evaluate(a_, w_, phi0_);
  best_a = a_;
  best_w = w_;
  best_phi = phi0_;

  for (const double a : a_grid) {
    for (const double w : w_grid) {
      for (int i = 0; i < kPhiSteps; ++i) {
        const double phi = tools::limit_rad(-CV_PI + (2.0 * CV_PI * i) / kPhiSteps);
        const double sse = evaluate(a, w, phi);
        if (sse < best_sse) {
          best_sse = sse;
          best_a = a;
          best_w = w;
          best_phi = phi;
        }
      }
    }
  }

  // φ0 局部细化（±半格，三分逼近两轮）。
  double lo = best_phi - CV_PI / kPhiSteps, hi = best_phi + CV_PI / kPhiSteps;
  for (int iter = 0; iter < 8; ++iter) {
    const double m1 = lo + (hi - lo) / 3.0, m2 = hi - (hi - lo) / 3.0;
    if (evaluate(best_a, best_w, m1) < evaluate(best_a, best_w, m2))
      hi = m2;
    else
      lo = m1;
  }
  best_phi = tools::limit_rad(0.5 * (lo + hi));

  a_ = std::clamp(best_a, 0.780, 1.045);
  w_ = std::clamp(best_w, 1.884, 2.000);
  phi0_ = best_phi;

  // 诊断：残差 RMS 与样本跨度。θ_unwrapped 里混进 72° 阶跃时残差会明显抬高，
  // 拟合跨度太短（<一个正弦周期）则参数本身不可辨识。
  const double final_sse = evaluate(a_, w_, phi0_);
  diag_fit_rms_ = std::sqrt(final_sse / static_cast<double>(samples.size()));
  diag_fit_samples_ = samples.size();
  diag_fit_span_ = span;
}

// ================= Target 门面 =================

void Target::reset_all()
{
  r_filter_.reset();
  plane_filter_.reset();
  phase_.reset();
  last_target_slot_.reset();
  roll_override_.reset();
  last_seen_timestamp_.reset();
  has_start_timestamp_ = false;
  obs_time_ = 0.0;
  pred_offset_ = 0.0;
  unsolvable_ = true;
  last_good_quality_ = false;
  last_slot_epoch_ = 0;
}

void Target::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  if (!p.has_value()) {
    unsolvable_ = true;
    // 规则：换组黑屏 ≤200ms，短时丢失绝不重置共享滤波器，只按时间判定长丢失。
    if (
      last_seen_timestamp_.has_value() &&
      tools::delta_time(timestamp, *last_seen_timestamp_) > full_reset_s) {
      tools::logger()->debug("[Target] 长时间丢失，重置共享滤波器");
      reset_all();
    }
    return;
  }

  if (
    last_seen_timestamp_.has_value() &&
    tools::delta_time(timestamp, *last_seen_timestamp_) > full_reset_s) {
    tools::logger()->debug("[Target] 长时间丢失后重捕获，当前帧重新初始化");
    reset_all();
  }

  if (!has_start_timestamp_) {
    start_timestamp_ = timestamp;
    has_start_timestamp_ = true;
    obs_time_ = 0.0;
  }
  const double now = tools::delta_time(timestamp, start_timestamp_);
  const double dt = std::clamp(now - obs_time_, 0.0, 1.0);

  // PnP 质量标志：回退帧或重投影误差偏大的帧降权，且此类帧不允许触发 R 快速重定位。
  const auto & blade = p->target();
  const bool good_quality = !blade.raw_pnp_fallback && !blade.raw_rm_center_fallback &&
                            std::isfinite(blade.raw_reprojection_rmse) &&
                            blade.raw_reprojection_rmse < 8.0;
  last_good_quality_ = good_quality;
  last_slot_epoch_ = p->slot_epoch;

  r_filter_.predict(dt);
  r_filter_.update(p->xyz_in_world, good_quality);
  plane_filter_.predict(dt);
  plane_filter_.update(p->ypr_in_world[0], good_quality);
  phase_.observe(p->target_slot(), p->ypr_in_world[2], now, good_quality, p->slot_epoch);

  last_target_slot_ = p->target_slot();
  last_primary_roll_obs_ = p->ypr_in_world[2];
  roll_override_.reset();
  obs_time_ = now;
  pred_offset_ = 0.0;
  last_seen_timestamp_ = timestamp;

  unsolvable_ =
    !(r_filter_.initialized() && plane_filter_.initialized() && phase_.initialized());
  // Detector 在本帧 PnP 之前回灌的图像角使用的是旧相位基准。若 observe() 刚完成
  // 整槽 fold，base_slot_ 已变化而 image_angle_bias_ 仍对应旧基准，下一帧预测角会
  // 再偏一整槽并形成自持轮转。用本帧真实图像角在 fold 后重新同步即可闭合该状态更新。
  if (!unsolvable_ && blade.observed)
    calibrate_image_angle(p->target_slot(), blade.angle);
  spd = phase_.speed_at(now);
}

void Target::observe_secondary(const std::optional<PowerRune> & secondary)
{
  // 大符恒有 2 块点亮（规则 5588）。只喂主槽等于扔掉一半相位观测率，
  // 也丢掉了"同帧两片 roll 差必须是 72° 整数倍"这个免费的自洽性检查。
  //
  // 注意：副片的位置/符面姿态与主片完全相同（整符一次解算的结果写回两个视图），
  // 所以这里只喂 roll 给相位模型，绝不重复喂 r_filter_ / plane_filter_——
  // 那会把同一个观测当两次独立测量，人为压小协方差。
  last_dual_roll_consistent_ = true;
  if (!secondary.has_value() || secondary->is_unsolve() || unsolvable_) return;
  if (!secondary->target().observed) return;  // 推断视图没有当帧二维点
  if (!last_target_slot_.has_value() || secondary->target_slot() == *last_target_slot_) return;
  if (!secondary->ypr_in_world.allFinite()) return;

  // 自洽性检查：两片 roll 差必须是 72° 的整数倍。偏离说明槽号体系或位姿有问题，
  // 这一帧的副片观测不可信，丢弃而不是硬喂。
  const auto primary_offset = phase_.slot_offset(*last_target_slot_);
  const auto secondary_offset = phase_.slot_offset(secondary->target_slot());
  if (!primary_offset.has_value() || !secondary_offset.has_value()) return;
  const double expected = tools::limit_rad(*secondary_offset - *primary_offset);
  const double measured = tools::limit_rad(
    secondary->ypr_in_world[2] - last_primary_roll_obs_);
  const double mismatch = std::abs(tools::limit_rad(measured - expected));
  last_dual_roll_mismatch_ = mismatch;
  if (mismatch > dual_roll_tolerance_rad) {
    last_dual_roll_consistent_ = false;
    return;
  }

  phase_.observe(
    secondary->target_slot(), secondary->ypr_in_world[2], obs_time_, last_good_quality_,
    secondary->slot_epoch);
  // 正常副片观测不覆盖主片标定；只有它确实触发了 fold，才按 fold 后的槽位基准同步。
  if (phase_.last_fold_steps() != 0)
    calibrate_image_angle(secondary->target_slot(), secondary->target().angle);
}

double Target::current_roll() const
{
  const double t = model_time();
  if (roll_override_.has_value())
    return tools::limit_rad(*roll_override_ + phase_.advance(obs_time_, t));
  const std::size_t slot = last_target_slot_.value_or(0);
  return phase_.roll_at(slot, t);
}

std::optional<double> Target::roll_of_slot(std::size_t slot) const
{
  if (unsolvable_) return std::nullopt;
  // 当前槽直接复用 current_roll()，保证整符可视化里的这一臂与 ekf_x()[5] 完全一致
  // （含 clone_with_roll 的 roll 覆写）。
  if (last_target_slot_.has_value() && *last_target_slot_ == slot) return current_roll();
  if (!phase_.slot_known(slot)) return std::nullopt;
  return phase_.roll_at(slot, model_time());
}

std::optional<double> Target::image_angle_of_slot(std::size_t slot) const
{
  if (unsolvable_ || !image_angle_bias_.has_value()) return std::nullopt;
  const auto roll = roll_of_slot(slot);
  if (!roll.has_value()) return std::nullopt;
  // 图像圆周角用 [0, 2π) 表示（与 Detector 的 point_angle 一致）。
  double angle = image_angle_sign_ * (*roll) + *image_angle_bias_;
  angle = std::fmod(angle, CV_2PI);
  return angle < 0.0 ? angle + CV_2PI : angle;
}

void Target::calibrate_image_angle(std::size_t slot, double image_angle)
{
  if (!std::isfinite(image_angle)) return;
  const auto roll = roll_of_slot(slot);
  if (!roll.has_value()) return;
  // 图像圆周角与 roll 的方向关系恰好就是槽差符号，推导：
  //   assign_slots 按 atan2 递增给格位编号 ⇒ 槽号 +1 对应图像圆周角 +72°；
  //   槽偏移定义 offset(k) = slot_sign×(k−base)×72°  ⇒ 槽号 +1 对应 roll +slot_sign×72°。
  //   两式相除 ⇒ d(image_angle)/d(roll) = slot_sign。
  // 整符连续转动时所有片一起动，方向关系与离散情形一致，所以这个符号也适用于外推。
  image_angle_sign_ = phase_.slot_sign();
  image_angle_bias_ = tools::limit_rad(image_angle - image_angle_sign_ * (*roll));
}

Eigen::Vector3d Target::point_buff2world(const Eigen::Vector3d & point_in_buff) const
{
  if (unsolvable_ || !r_filter_.initialized()) return Eigen::Vector3d::Zero();
  const Eigen::Matrix3d R_buff2world = tools::rotation_matrix(
    Eigen::Vector3d(plane_filter_.yaw(), 0.0, current_roll()));
  return R_buff2world * point_in_buff + r_filter_.xyz();
}

Eigen::VectorXd Target::compose_common_state() const
{
  const Eigen::Vector3d ypd = tools::xyz2ypd(r_filter_.xyz());
  Eigen::VectorXd x(6);
  x << ypd[0], 0.0, ypd[1], ypd[2], plane_filter_.yaw(), current_roll();
  return x;
}

Eigen::VectorXd SmallTarget::ekf_x() const
{
  Eigen::VectorXd x(7);
  x.head<6>() = compose_common_state();
  x[6] = phase_.direction() * SMALL_W;
  return x;
}

Eigen::VectorXd BigTarget::ekf_x() const
{
  Eigen::VectorXd x(10);
  x.head<6>() = compose_common_state();
  x[6] = std::abs(phase_.speed_at(model_time()));
  x[7] = phase_.a();
  x[8] = phase_.w();
  x[9] = phase_.phi0();
  return x;
}

}  // namespace auto_buff
