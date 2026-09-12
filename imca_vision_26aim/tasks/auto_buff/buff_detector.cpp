#include "buff_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{
namespace
{
constexpr std::size_t kRCenterIndex = 4;
constexpr std::size_t kFlowCenterIndex = 5;
constexpr std::size_t kKeypointCount = 6;
constexpr double kSlotAngle = CV_2PI / 5.0;
constexpr double kTargetRadius = 0.700;
constexpr double kFlowRadius = 0.344;

bool finite_point(const cv::Point2f & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

float cross(const cv::Point2f & lhs, const cv::Point2f & rhs)
{
  return lhs.x * rhs.y - lhs.y * rhs.x;
}

cv::Point2f target_center(const YOLO11_BUFF::Object & object)
{
  // target 的物理中心是两条对角线的交点，而不是 kpt4(R) 或 kpt5(流水灯)。
  const cv::Point2f diagonal_02 = object.kpt[2] - object.kpt[0];
  const cv::Point2f diagonal_13 = object.kpt[3] - object.kpt[1];
  const float denominator = cross(diagonal_02, diagonal_13);
  if (std::abs(denominator) < 1e-6F) {
    return (object.kpt[0] + object.kpt[1] + object.kpt[2] + object.kpt[3]) * 0.25F;
  }

  const float ratio = cross(object.kpt[1] - object.kpt[0], diagonal_13) / denominator;
  return object.kpt[0] + diagonal_02 * ratio;
}

double point_angle(const cv::Point2f & point, const cv::Point2f & center)
{
  const cv::Point2f direction = point - center;
  const double angle = std::atan2(direction.y, direction.x);
  return angle >= 0.0 ? angle : angle + CV_2PI;
}

double wrapped_delta(double from, double to)
{
  double delta = to - from;
  while (delta < 0.0) delta += CV_2PI;
  while (delta >= CV_2PI) delta -= CV_2PI;
  return delta;
}

double circular_distance(double lhs, double rhs)
{
  const double delta = std::abs(lhs - rhs);
  return std::min(delta, CV_2PI - delta);
}

double target_diagonal(const FanBlade & blade)
{
  if (blade.points.size() < 4) return 0.0;
  return std::max(
    cv::norm(blade.points[0] - blade.points[2]),
    cv::norm(blade.points[1] - blade.points[3]));
}

double blade_weight(const FanBlade & blade)
{
  double weight = std::clamp(static_cast<double>(blade.object_confidence), 0.01, 1.0);
  if (blade.keypoint_confidences.size() != kKeypointCount) return weight;
  for (const auto index : {0U, 1U, 2U, 3U, 5U}) {
    weight = std::min(
      weight, std::clamp(static_cast<double>(blade.keypoint_confidences[index]), 0.01, 1.0));
  }
  return weight;
}

// ============ R 二维融合的常数 ============
// 粗筛只拦"跳到隔壁扇叶"这种粗离群，宽度按本帧扇叶间距的一半定（相邻扇叶物理弦长
// 0.823 m，跳隔壁就是这个尺度）。下限按 kpt4 自身噪声的倍数兜住，宁可放行不误杀。
constexpr double kCoarseGateBladeFraction = 0.5;
constexpr double kCoarseGateKpt4Sigmas = 6.0;
constexpr double kCoarseGateFloorPx = 25.0;

// ============ kpt4 门控（apply_r_center 当前使用的路径）的常数 ============
// 门限随目标像素尺寸缩放：target 对角线像素长度反映了距离，取其一半作半径，
// 拦得住"跳到隔壁扇叶"（约 1.9 倍对角线），也容得下融合 kpt4 的噪声（σ≈12 px）。
constexpr double kRGateDiagonalRatio = 0.5;
// 物理钳位：远距离下门圆不至于小到误杀正常抖动，近距离下不至于大到套住半个符。
constexpr double kRGateMinPx = 25.0;
constexpr double kRGateMaxPx = 120.0;
// 连续被拦这么多帧就放行一帧 rm R（逃生）。与调试程序判读注释的"逃生阈值（8）"一致。
constexpr int kRGateEscapeStreak = 8;

// rm-core 轮廓 R 的测量标准差。它是两个源里"准"的那个，给小值。
// TODO(回放标定): 取 r_src=RmCore 帧上 rm 测量相对滤波输出的残差标准差。
constexpr double kRmSigmaPx = 1.5;
// 单片时估不出 kpt4 样本散布，用这个先验兜底。
constexpr double kKpt4SigmaFallbackPx = 12.0;
constexpr double kKpt4SigmaFloorPx = 2.0;

// 恒速度模型的过程噪声：由云台角加速度驱动。0.7 m 半径 / 7 m 距离 / 符周期 2.5~8 s
// 下云台角加速度对应像素加速度约 400 px/s² 量级，取同阶。
constexpr double kAccelSigmaPxPerS2 = 400.0;
constexpr double kInitPosSigmaPx = 50.0;
constexpr double kInitVelSigmaPxPerS = 200.0;
// Huber 鲁棒阈值（以标准差为单位）：创新超过它就按比例膨胀该测量的协方差。
// 针对的是"单帧抖动"——软降权而非硬拒绝，避免再引入一个硬门限。
constexpr double kHuberSigmas = 3.0;
// 空滑过久就重新初始化：dt 太大时恒速度外推已无意义。
constexpr double kMaxCoastSeconds = 0.30;

// 置信度加权融合各片的 kpt4。单片 kpt4 精度不够，但跨片融合后噪声被平均，
// 足以充当"rm R 大致该在哪"的粗参照，也可在 rm R 缺失时作回退来源。
struct FusedKpt4
{
  cv::Point2f center{};
  int count{0};
  double mean_diagonal{0.0};
  // 融合值自身的标准差。用户确认 kpt4 误差以随机噪声为主，因此可按
  // 样本散布/√n 收缩——"片数越多越可信"由此自然成立，不用额外的经验公式。
  double sigma_px{kKpt4SigmaFallbackPx};
};

FusedKpt4 fused_kpt4_center(const std::vector<FanBlade> & fanblades)
{
  FusedKpt4 result;
  std::vector<cv::Point2f> points;
  std::vector<double> weights;
  cv::Point2d accumulated(0.0, 0.0);
  double total_weight = 0.0;
  double total_diagonal = 0.0;
  for (const auto & blade : fanblades) {
    // observed 排除继承的历史槽（它们的 raw_r_valid 已被构造函数清掉，这里是双保险）。
    if (!blade.observed || !blade.raw_r_valid || !finite_point(blade.raw_r_center)) continue;
    const double weight = std::clamp(static_cast<double>(blade.raw_r_confidence), 0.01, 1.0);
    accumulated.x += weight * blade.raw_r_center.x;
    accumulated.y += weight * blade.raw_r_center.y;
    total_weight += weight;
    total_diagonal += target_diagonal(blade);
    points.push_back(blade.raw_r_center);
    weights.push_back(weight);
    ++result.count;
  }
  if (result.count == 0 || total_weight <= 0.0) return {};

  result.center = cv::Point2f(
    static_cast<float>(accumulated.x / total_weight),
    static_cast<float>(accumulated.y / total_weight));
  if (!finite_point(result.center)) return {};
  result.mean_diagonal = total_diagonal / result.count;

  // n≥2 才能估散布；n==1 用先验兜底（单片估不出自己的噪声）。
  if (result.count >= 2) {
    double sum_squared = 0.0;
    for (const auto & point : points) {
      const cv::Point2f delta = point - result.center;
      sum_squared += static_cast<double>(delta.x) * delta.x +
                     static_cast<double>(delta.y) * delta.y;
    }
    // 每点两个自由度，除以 (n-1) 得单点方差；再除 n 得均值的方差。
    const double per_point_variance = sum_squared / (2.0 * (result.count - 1));
    result.sigma_px = std::sqrt(std::max(0.0, per_point_variance / result.count));
  }
  result.sigma_px = std::max(result.sigma_px, kKpt4SigmaFloorPx);
  if (!std::isfinite(result.sigma_px)) result.sigma_px = kKpt4SigmaFallbackPx;
  return result;
}

// 粗筛门限：只拦"跳到隔壁扇叶"这种粗离群。按本帧观测到的扇叶中心最小间距的一半定，
// 于是它随距离自动缩放、无需知道内参。看不到两片时退到 kpt4 噪声的倍数。
double coarse_gate_threshold(const std::vector<FanBlade> & fanblades, double kpt4_sigma_px)
{
  double min_gap = std::numeric_limits<double>::infinity();
  std::vector<cv::Point2f> centers;
  for (const auto & blade : fanblades) {
    if (!blade.observed || !finite_point(blade.center)) continue;
    centers.push_back(blade.center);
  }
  for (std::size_t i = 0; i + 1 < centers.size(); ++i)
    for (std::size_t j = i + 1; j < centers.size(); ++j)
      min_gap = std::min(min_gap, cv::norm(centers[i] - centers[j]));

  // 只看到不相邻的两片时 min_gap 更大，门限随之更宽——偏向放行，符合"定宽"的取向。
  const double from_blades =
    std::isfinite(min_gap) ? kCoarseGateBladeFraction * min_gap
                           : std::numeric_limits<double>::infinity();
  const double from_noise = kCoarseGateKpt4Sigmas * kpt4_sigma_px;
  const double threshold = std::isfinite(from_blades) ? std::max(from_blades, from_noise)
                                                     : from_noise;
  return std::max(threshold, kCoarseGateFloorPx);
}

}  // namespace

// ============ RPixelFusion ============
// 类声明在头文件的 auto_buff 命名空间里，所以成员定义不能放进上面的匿名命名空间。

void RPixelFusion::reset()
{
  initialized_ = false;
  last_timestamp_s_ = 0.0;
  last_rm_inflation_ = 1.0;
  x_ = cv::Matx41d(0.0, 0.0, 0.0, 0.0);
  P_ = cv::Matx44d::eye();
}

double RPixelFusion::position_sigma_px() const
{
  if (!initialized_) return std::numeric_limits<double>::infinity();
  return std::sqrt(std::max(P_(0, 0), P_(1, 1)));
}

void RPixelFusion::predict(double dt)
{
  // 恒速度：x' = x + v·dt。
  cv::Matx44d F = cv::Matx44d::eye();
  F(0, 2) = dt;
  F(1, 3) = dt;

  // 过程噪声由加速度驱动的标准离散化。
  const double a2 = kAccelSigmaPxPerS2 * kAccelSigmaPxPerS2;
  const double dt2 = dt * dt;
  const double q_pos = a2 * dt2 * dt2 / 4.0;
  const double q_vel = a2 * dt2;
  const double q_cross = a2 * dt2 * dt / 2.0;
  cv::Matx44d Q = cv::Matx44d::zeros();
  Q(0, 0) = q_pos;
  Q(1, 1) = q_pos;
  Q(2, 2) = q_vel;
  Q(3, 3) = q_vel;
  Q(0, 2) = Q(2, 0) = q_cross;
  Q(1, 3) = Q(3, 1) = q_cross;

  x_ = F * x_;
  P_ = F * P_ * F.t() + Q;
}

void RPixelFusion::correct(const cv::Point2f & z, double sigma_px, double * inflation_out)
{
  const double variance = std::max(sigma_px * sigma_px, 1e-6);
  // 创新与其协方差（H 只取位置两维，所以 S 就是 P 的位置块加 R）。
  const cv::Matx21d innovation(
    static_cast<double>(z.x) - x_(0), static_cast<double>(z.y) - x_(1));

  double inflation = 1.0;
  {
    // Huber 软降权：只针对单帧抖动，创新超过 kHuberSigmas 个标准差就按比例膨胀 R。
    const double s_xx = P_(0, 0) + variance;
    const double s_yy = P_(1, 1) + variance;
    const double normalized = std::sqrt(
      innovation(0) * innovation(0) / std::max(s_xx, 1e-9) +
      innovation(1) * innovation(1) / std::max(s_yy, 1e-9));
    if (normalized > kHuberSigmas) inflation = normalized / kHuberSigmas;
  }
  if (inflation_out != nullptr) *inflation_out = inflation;
  const double effective_variance = variance * inflation * inflation;

  cv::Matx22d S(
    P_(0, 0) + effective_variance, P_(0, 1), P_(1, 0), P_(1, 1) + effective_variance);
  cv::Matx22d S_inv;
  if (!cv::invert(S, S_inv, cv::DECOMP_CHOLESKY)) return;

  // K = P·Hᵀ·S⁻¹，H = [I 0]，所以 P·Hᵀ 就是 P 的前两列。
  // OpenCV 没有预定义 Matx42d 这个 typedef，只能写模板原型。
  cv::Matx<double, 4, 2> PHt;
  for (int r = 0; r < 4; ++r) {
    PHt(r, 0) = P_(r, 0);
    PHt(r, 1) = P_(r, 1);
  }
  const cv::Matx<double, 4, 2> K = PHt * S_inv;

  x_ += K * innovation;
  // P = (I - K·H)·P
  cv::Matx44d KH = cv::Matx44d::zeros();
  for (int r = 0; r < 4; ++r) {
    KH(r, 0) = K(r, 0);
    KH(r, 1) = K(r, 1);
  }
  P_ = (cv::Matx44d::eye() - KH) * P_;
}

std::optional<cv::Point2f> RPixelFusion::update(
  const std::optional<Measurement> & rm, const std::optional<Measurement> & kpt4,
  double timestamp_s)
{
  const bool has_measurement = rm.has_value() || kpt4.has_value();

  if (initialized_) {
    const double dt = timestamp_s - last_timestamp_s_;
    // dt 非正（时间戳回退/重复）或空滑过久，恒速度外推都已无意义。
    if (dt <= 0.0 || dt > kMaxCoastSeconds) {
      if (!has_measurement) {
        reset();
        return std::nullopt;
      }
      reset();
    } else {
      predict(dt);
    }
  }

  if (!initialized_) {
    if (!has_measurement) return std::nullopt;
    // 优先用准的那个源初始化，避免开局就被 kpt4 的偏差带跑。
    const Measurement & seed = rm.has_value() ? *rm : *kpt4;
    x_ = cv::Matx41d(seed.point.x, seed.point.y, 0.0, 0.0);
    P_ = cv::Matx44d::zeros();
    const double pos_var = std::max(seed.sigma_px * seed.sigma_px, kInitPosSigmaPx);
    P_(0, 0) = P_(1, 1) = pos_var;
    P_(2, 2) = P_(3, 3) = kInitVelSigmaPxPerS * kInitVelSigmaPxPerS;
    initialized_ = true;
    last_timestamp_s_ = timestamp_s;
    last_rm_inflation_ = 1.0;
    // 用另一个源（若有）再校正一次，让首帧就吃到两个测量。
    if (rm.has_value() && kpt4.has_value()) correct(kpt4->point, kpt4->sigma_px, nullptr);
    return cv::Point2f(static_cast<float>(x_(0)), static_cast<float>(x_(1)));
  }

  last_timestamp_s_ = timestamp_s;
  last_rm_inflation_ = 1.0;
  // 顺序更新两个测量，等价于联合更新（它们的噪声互不相关）。
  if (rm.has_value()) correct(rm->point, rm->sigma_px, &last_rm_inflation_);
  if (kpt4.has_value()) correct(kpt4->point, kpt4->sigma_px, nullptr);

  if (!std::isfinite(x_(0)) || !std::isfinite(x_(1))) {
    reset();
    return std::nullopt;
  }
  return cv::Point2f(static_cast<float>(x_(0)), static_cast<float>(x_(1)));
}

// 重新进入匿名命名空间：下面这些仍是本 TU 私有的辅助工具。
namespace
{

struct RCenterSeed
{
  cv::Point2f center;
  double weight;
  double target_size;
};

std::optional<cv::Point2f> robust_r_center(const std::vector<RCenterSeed> & seeds)
{
  if (seeds.empty()) return std::nullopt;

  cv::Point2d initial(0.0, 0.0);
  double initial_weight = 0.0;
  for (const auto & seed : seeds) {
    initial.x += seed.weight * seed.center.x;
    initial.y += seed.weight * seed.center.y;
    initial_weight += seed.weight;
  }
  if (initial_weight <= 0.0) return std::nullopt;
  initial *= 1.0 / initial_weight;

  // 两个目标会分别给出一个 R 几何估计。Huber 权重把偶发抖动的那一个压低，
  // 最终 PowerRune 仍只保存一个共享中心。
  cv::Point2d robust_sum(0.0, 0.0);
  double robust_weight_sum = 0.0;
  const cv::Point2f initial_point(
    static_cast<float>(initial.x), static_cast<float>(initial.y));
  for (const auto & seed : seeds) {
    const double threshold = std::max(8.0, seed.target_size * 0.25);
    const double residual = cv::norm(seed.center - initial_point);
    const double huber = residual > threshold ? threshold / residual : 1.0;
    const double weight = seed.weight * huber;
    robust_sum.x += weight * seed.center.x;
    robust_sum.y += weight * seed.center.y;
    robust_weight_sum += weight;
  }
  if (robust_weight_sum <= 0.0) return std::nullopt;
  const cv::Point2f center(
    static_cast<float>(robust_sum.x / robust_weight_sum),
    static_cast<float>(robust_sum.y / robust_weight_sum));
  return finite_point(center) ? std::optional<cv::Point2f>(center) : std::nullopt;
}
}  // namespace

Buff_Detector::Buff_Detector(const std::string & config)
: model_(config), rm_center_detector_(config)
{
  const auto yaml = YAML::LoadFile(config);
  if (!yaml["buff_loss_max_frames"] || !yaml["buff_secondary_hold_max_frames"]) {
    throw std::runtime_error(
      "Buff config must define buff_loss_max_frames and buff_secondary_hold_max_frames");
  }

  loss_max_frames_ = yaml["buff_loss_max_frames"].as<int>();
  secondary_hold_max_frames_ = yaml["buff_secondary_hold_max_frames"].as<int>();
  // 旧配置缺失时保持原 YOLO-only 行为，只有比赛配置显式开启才运行 rm-core。
  rm_center_enabled_ =
    yaml["buff_rm_center_enabled"] && yaml["buff_rm_center_enabled"].as<bool>();
  if (loss_max_frames_ < 1 || secondary_hold_max_frames_ < 0) {
    throw std::runtime_error(
      "Buff frame limits are invalid: loss must be >= 1 and secondary hold must be >= 0");
  }
}

void Buff_Detector::setMyColor(uint8_t my_color)
{
  rm_center_detector_.setMyColor(my_color);
}

std::optional<FanBlade> Buff_Detector::make_fanblade(
  const YOLO11_BUFF::Object & object) const
{
  if (object.kpt.size() != kKeypointCount || object.kpt_prob.size() != kKeypointCount ||
      !std::isfinite(object.prob))
    return std::nullopt;

  // 五点 PnP 和几何 R 都依赖 kpt0~3 + kpt5。关键点置信度已经在 YOLO
  // 解析阶段按 YAML 门控；这里不再复制一份隐藏阈值，避免配置与 Detector 不一致。
  for (const auto index : {0U, 1U, 2U, 3U, 5U}) {
    if (!finite_point(object.kpt[index]) || !std::isfinite(object.kpt_prob[index]))
      return std::nullopt;
  }

  FanBlade blade(object.kpt, target_center(object), _light);
  blade.rect = object.rect;
  blade.keypoint_confidences = object.kpt_prob;
  blade.object_confidence = object.prob;
  blade.raw_r_center = object.kpt[kRCenterIndex];
  blade.raw_r_confidence = object.kpt_prob[kRCenterIndex];
  blade.raw_r_valid = finite_point(blade.raw_r_center) &&
                      std::isfinite(blade.raw_r_confidence) && blade.raw_r_confidence > 0.0F;
  blade.observed = true;
  return blade;
}

cv::Point2f Buff_Detector::shared_r_center(const std::vector<FanBlade> & fanblades)
{
  std::vector<RCenterSeed> seeds;
  seeds.reserve(fanblades.size());
  constexpr float physical_ratio =
    static_cast<float>(kTargetRadius / (kTargetRadius - kFlowRadius));
  for (const auto & blade : fanblades) {
    if (blade.points.size() <= kFlowCenterIndex || !finite_point(blade.center) ||
        !finite_point(blade.points[kFlowCenterIndex]))
      continue;

    const cv::Point2f flow = blade.points[kFlowCenterIndex];
    if (cv::norm(flow - blade.center) < 3.0) continue;
    // T、F、R 在同一条径向线上，TR=0.700 m、FR=0.344 m：
    // R = T + (F-T) * 0.700 / (0.700-0.344)。这条链路完全不使用 kpt4 或颜色轮廓。
    const cv::Point2f center = blade.center + (flow - blade.center) * physical_ratio;
    if (!finite_point(center)) continue;
    seeds.push_back({center, blade_weight(blade), target_diagonal(blade)});
  }

  const auto current = robust_r_center(seeds);
  if (!current.has_value()) {
    if (last_powerrune_.has_value() && finite_point(last_powerrune_->r_center))
      return last_powerrune_->r_center;
    return {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  }

  if (!last_powerrune_.has_value() || !finite_point(last_powerrune_->r_center)) return *current;

  double mean_target_size = 0.0;
  for (const auto & seed : seeds) mean_target_size += seed.target_size;
  mean_target_size /= std::max<std::size_t>(1, seeds.size());
  const double smoothing_gate = std::max(10.0, mean_target_size * 0.35);
  const double innovation = cv::norm(*current - last_powerrune_->r_center);
  if (innovation > smoothing_gate) return *current;

  // 只平滑小范围像素噪声；相机或符真实移动超过门限时立即跟随，避免历史中心拖尾。
  constexpr float kCurrentWeight = 0.65F;
  return last_powerrune_->r_center * (1.0F - kCurrentWeight) + *current * kCurrentWeight;
}

std::size_t Buff_Detector::assign_slots(
  std::vector<FanBlade> & fanblades, const cv::Point2f & r_center) const
{
  if (fanblades.empty()) return 0;

  // ============ 全局 72° 网格对齐 ============
  // 旧做法是"跟上一帧主槽最近邻 + 像素兜底"，两个毛病：
  //   1) 参考角取自上一有效帧的像素角，换组黑屏 200ms 内不推进，转速 1~2 rad/s 时
  //      已陈旧 11°~24°，吃掉 36° 角门限的大半预算；
  //   2) 像素兜底用 `||` 覆盖掉正确的角门限拒绝，把相邻扇叶认成同一片。
  // 新做法参考 rm_vision_core 的 getRuneDeviation：先把本帧所有可见片对齐到一个
  // 72° 网格（全局最小二乘，不依赖上一帧），再决定这个网格整体对应哪些槽号。

  std::vector<double> angles(fanblades.size());
  for (std::size_t i = 0; i < fanblades.size(); ++i)
    angles[i] = point_angle(fanblades[i].center, r_center);

  // 网格相位：让所有可见片同时最接近 72° 整数倍的那个整体偏移。
  // 单片时任何偏移都完美贴合，此时 grid_phase 就等于该片自身的角度（模 72°）。
  double grid_phase = 0.0;
  {
    double best_sse = std::numeric_limits<double>::infinity();
    constexpr int kSteps = 720;  // 半度分辨率
    for (int s = 0; s < kSteps; ++s) {
      const double phase = kSlotAngle * s / kSteps;
      double sse = 0.0;
      for (const double angle : angles) {
        const double residual =
          tools::limit_rad(angle - phase - std::round((angle - phase) / kSlotAngle) * kSlotAngle);
        sse += residual * residual;
      }
      if (sse < best_sse) {
        best_sse = sse;
        grid_phase = phase;
      }
    }
  }

  // 每片在网格上的格位（0~4），彼此的差就是真实槽差，这一步完全不依赖历史。
  std::vector<int> grid_index(fanblades.size());
  for (std::size_t i = 0; i < fanblades.size(); ++i) {
    const int k = static_cast<int>(std::llround((angles[i] - grid_phase) / kSlotAngle));
    grid_index[i] = ((k % static_cast<int>(PowerRune::SLOT_COUNT)) +
                     static_cast<int>(PowerRune::SLOT_COUNT)) %
                    static_cast<int>(PowerRune::SLOT_COUNT);
  }

  // ============ 决定网格到槽号的整体偏移 ============
  // 只有这一步用历史：把网格钉到上一帧的槽号体系上，保持槽号跨帧稳定。
  // 参考角用相位模型外推到当前时刻的预测角（若可用），而不是上一帧的陈旧像素角。
  std::size_t primary_slot = 0;
  int slot_shift = 0;
  std::optional<std::size_t> primary_index;

  if (
    last_powerrune_.has_value() &&
    last_powerrune_->target_slot() < last_powerrune_->fanblades.size()) {
    primary_slot = last_powerrune_->target_slot();
    const FanBlade & previous = last_powerrune_->fanblades[primary_slot];

    double reference_angle = std::numeric_limits<double>::quiet_NaN();
    if (predicted_slot_angle_) {
      // 相位模型外推：黑屏期间也在推进，不会陈旧。
      const auto predicted = predicted_slot_angle_(primary_slot);
      if (predicted.has_value()) reference_angle = *predicted;
    }
    if (!std::isfinite(reference_angle) && finite_point(previous.center))
      reference_angle = point_angle(previous.center, last_powerrune_->r_center);

    if (std::isfinite(reference_angle)) {
      // 找与参考角最接近的那片，把它认作主槽；角距必须在半槽内，
      // 不再用像素距离兜底（那正是把相邻扇叶认成同一片的原因）。
      double nearest_angle_distance = std::numeric_limits<double>::infinity();
      std::size_t nearest_index = 0;
      for (std::size_t i = 0; i < fanblades.size(); ++i) {
        const double d = circular_distance(reference_angle, angles[i]);
        if (d < nearest_angle_distance) {
          nearest_angle_distance = d;
          nearest_index = i;
        }
      }
      if (nearest_angle_distance <= kSlotAngle * 0.5) {
        primary_index = nearest_index;
        slot_shift = static_cast<int>(primary_slot) - grid_index[nearest_index];
      } else {
        // 主槽本帧没被观测到（换组换到了别处）。仍要保持槽号体系连续：
        // 用参考角自身在网格上的格位来钉，这样其余片的槽号仍与历史一致。
        const int k = static_cast<int>(
          std::llround((reference_angle - grid_phase) / kSlotAngle));
        const int reference_grid =
          ((k % static_cast<int>(PowerRune::SLOT_COUNT)) +
           static_cast<int>(PowerRune::SLOT_COUNT)) %
          static_cast<int>(PowerRune::SLOT_COUNT);
        slot_shift = static_cast<int>(primary_slot) - reference_grid;
      }
    }
  } else {
    // 冷启动：置信度最高的片定义为 slot 0。此后槽号体系由 slot_epoch 标记。
    const auto best = std::max_element(
      fanblades.begin(), fanblades.end(), [](const FanBlade & lhs, const FanBlade & rhs) {
        return lhs.object_confidence < rhs.object_confidence;
      });
    primary_index = static_cast<std::size_t>(std::distance(fanblades.begin(), best));
    slot_shift = -grid_index[*primary_index];
  }

  for (std::size_t i = 0; i < fanblades.size(); ++i) {
    const int slot =
      ((grid_index[i] + slot_shift) % static_cast<int>(PowerRune::SLOT_COUNT) +
       static_cast<int>(PowerRune::SLOT_COUNT)) %
      static_cast<int>(PowerRune::SLOT_COUNT);
    fanblades[i].slot_index = static_cast<std::size_t>(slot);
    fanblades[i].angle = angles[i];
  }

  // 同一格位撞车（几何异常）时，保留置信度更高的那片，另一片标为无效槽。
  std::array<int, PowerRune::SLOT_COUNT> owner{};
  owner.fill(-1);
  for (std::size_t i = 0; i < fanblades.size(); ++i) {
    const std::size_t slot = fanblades[i].slot_index;
    if (owner[slot] < 0) {
      owner[slot] = static_cast<int>(i);
      continue;
    }
    const std::size_t incumbent = static_cast<std::size_t>(owner[slot]);
    if (fanblades[i].object_confidence > fanblades[incumbent].object_confidence) {
      fanblades[incumbent].slot_index = PowerRune::SLOT_COUNT;
      owner[slot] = static_cast<int>(i);
    } else {
      fanblades[i].slot_index = PowerRune::SLOT_COUNT;
    }
  }

  if (primary_index.has_value() && *primary_index < fanblades.size() &&
      fanblades[*primary_index].slot_index < PowerRune::SLOT_COUNT)
    primary_slot = fanblades[*primary_index].slot_index;

  // 回灌 roll → 图像圆周角的标定，供下一帧（尤其是换组黑屏期间）外推参考角。
  if (
    calibrate_image_angle_ && primary_index.has_value() && *primary_index < fanblades.size() &&
    fanblades[*primary_index].slot_index < PowerRune::SLOT_COUNT)
    calibrate_image_angle_(primary_slot, angles[*primary_index]);

  return primary_slot;
}

std::optional<PowerRune> Buff_Detector::make_power_rune(
  const std::vector<YOLO11_BUFF::Object> & objects, std::uint64_t frame_id,
  const std::optional<cv::Point2f> & rm_center)
{
  std::vector<FanBlade> fanblades;
  std::vector<YOLO11_BUFF::Object> active_markers;
  fanblades.reserve(objects.size());
  for (const auto & object : objects) {
    if (object.is_active()) {
      active_markers.push_back(object);
      continue;
    }
    auto blade = make_fanblade(object);
    if (blade.has_value()) fanblades.push_back(std::move(*blade));
  }
  if (fanblades.empty()) return std::nullopt;

  if (fanblades.size() > PowerRune::SLOT_COUNT) {
    std::partial_sort(
      fanblades.begin(), fanblades.begin() + PowerRune::SLOT_COUNT, fanblades.end(),
      [](const FanBlade & lhs, const FanBlade & rhs) {
        return lhs.object_confidence > rhs.object_confidence;
      });
    fanblades.resize(PowerRune::SLOT_COUNT);
  }

  // buf1 没有关键点，只作为“该 buff 已激活”的状态观测。
  // 先按框重叠关联到带关键点的 buff，再把状态带入五槽模型；buf1 绝不能进入 PnP。
  for (const auto & marker : active_markers) {
    const cv::Point2f marker_center(
      marker.rect.x + marker.rect.width * 0.5F,
      marker.rect.y + marker.rect.height * 0.5F);
    std::optional<std::size_t> best_index;
    double best_score = -1.0;
    for (std::size_t index = 0; index < fanblades.size(); ++index) {
      const auto & blade = fanblades[index];
      const auto intersection = marker.rect & blade.rect;
      const double intersection_area = static_cast<double>(intersection.area());
      const double union_area = static_cast<double>(marker.rect.area()) +
                                static_cast<double>(blade.rect.area()) - intersection_area;
      const double iou = union_area > 0.0 ? intersection_area / union_area : 0.0;
      const double diagonal = std::hypot(blade.rect.width, blade.rect.height);
      const double distance = cv::norm(marker_center - blade.center);
      const double distance_gate = std::max(12.0, 0.8 * diagonal);
      if (iou < 0.05 && distance > distance_gate) continue;

      const double score = iou + 1.0 / (1.0 + distance / std::max(1.0, diagonal));
      if (score > best_score) {
        best_score = score;
        best_index = index;
      }
    }
    if (!best_index.has_value()) continue;

    auto & blade = fanblades[*best_index];
    if (!blade.active_observed || marker.prob > blade.active_confidence) {
      blade.active = true;
      blade.active_observed = true;
      blade.active_confidence = marker.prob;
    }
  }

  // 关联层**始终**用 kpt5 几何外推中心，不跟随 rm-core 的命中/漏检切换。
  //
  // 关联要的是帧间连续，不是绝对精确：assign_slots 的全部输入就是各片相对该中心的
  // 圆周角与由它拟合的 72° 网格。rm-core R 是逐帧独立的，而且它内部的 checkPoseDiff
  // / gimbal lock 一旦拒绝就会重建整个 RuneGroup，形成 2 帧周期的命中振荡；用它驱动
  // 关联会让五片的圆周角随之整体平移、网格错一格，再经 calibrate_image_angle 回灌
  // 形成"槽号每帧 +1"的自持棘轮（表现为 fold_steps 恒 −1、大符 target 在相邻两片
  // 之间来回跳）。shared_r_center 自带跨帧平滑与创新门限，正是关联需要的性质。
  //
  // rm-core R 只进 r_center_raw，也就是只影响 PnP——那才是它要修的东西。
  cv::Point2f center = shared_r_center(fanblades);
  // 外推完全不可用（无 kpt5 且无历史）时才退回 rm-core 中心，至少保住这一帧。
  if (!finite_point(center) && rm_center.has_value()) center = *rm_center;
  if (!finite_point(center)) return std::nullopt;
  std::size_t selected_slot = assign_slots(fanblades, center);
  // 优先选择 buf1 标记的激活目标；没有 buf1 时保留原有历史槽回退逻辑。
  std::optional<std::size_t> active_slot;
  float active_confidence = 0.0F;
  for (const auto & blade : fanblades) {
    if (!blade.active || !blade.active_observed || blade.slot_index >= PowerRune::SLOT_COUNT)
      continue;
    if (
      !active_slot.has_value() ||
      blade.active_confidence > active_confidence) {
      active_slot = blade.slot_index;
      active_confidence = blade.active_confidence;
    }
  }
  if (active_slot.has_value()) selected_slot = *active_slot;
  PowerRune power_rune(fanblades, center, last_powerrune_, frame_id, selected_slot);
  power_rune.rm_center_enabled = rm_center_enabled_;
  // 读 power_rune.fanblades 而非局部 fanblades：前者已完成槽位分配，
  // 撞槽被丢弃的重复片不在其中，继承的历史槽也已标 observed=false。
  apply_r_center(power_rune, rm_center);
  power_rune.slot_epoch = slot_epoch_;
  return power_rune.is_unsolve() ? std::nullopt : std::optional<PowerRune>(power_rune);
}

// 决定本帧 PnP 用哪个 R，并记录门控结论。
//
// 为什么需要门控：rm-core 的 R 现在已与它自己的 PnP/姿态阶段解耦（见 rm_core_adapter），
// 好 R 不再被姿态域失败误杀——但代价是误识别的 R 也不再被那些检查顺带挡掉。
// 门控参照物用 YOLO kpt4：单片 kpt4 精度不足以进 PnP，但判断"rm R 是不是落到了完全
// 错误的位置"只需要粗精度，跨片融合后足够胜任。
void Buff_Detector::apply_r_center(
  PowerRune & power_rune, const std::optional<cv::Point2f> & rm_center)
{
  // buff_rm_center_enabled=false 的语义是"保持纯 YOLO 旧路径"。融合 kpt4 回退属于
  // rm-center 这一整套功能，关掉时不能偷偷启用它，否则这个开关就名不副实了。
  // 此时 r_center_observed 保持 false，解算照旧走 kpt0~3+kpt5。
  if (!rm_center_enabled_) {
    power_rune.r_center_source = RCenterSource::None;
    return;
  }

  const FusedKpt4 fused = fused_kpt4_center(power_rune.fanblades);
  power_rune.r_kpt4_fused_count = fused.count;
  if (fused.count > 0) power_rune.r_kpt4_fused_center = fused.center;

  const auto accept = [&](const cv::Point2f & point, RCenterSource source) {
    power_rune.r_center_raw = point;
    power_rune.r_center_observed = true;
    power_rune.r_center_source = source;
  };

  // 情况 1：无 rm R。退到融合 kpt4，**不再退 kpt5**（实测 kpt5 会把符中心拉偏）。
  if (!rm_center.has_value() || !finite_point(*rm_center)) {
    r_gate_reject_streak_ = 0;
    if (fused.count > 0)
      accept(fused.center, RCenterSource::FusedKpt4);
    else
      power_rune.r_center_source = RCenterSource::None;
    return;
  }

  // 情况 2：有 rm R 但无可信 kpt4 参照 —— 收敛前禁用，直接放行不做拒绝。
  if (fused.count == 0) {
    r_gate_reject_streak_ = 0;
    accept(*rm_center, RCenterSource::RmCoreUngated);
    return;
  }

  // 情况 3：两者都有，做门控。门限带物理钳位，不随目标尺寸无界伸缩。
  const double threshold =
    std::clamp(kRGateDiagonalRatio * fused.mean_diagonal, kRGateMinPx, kRGateMaxPx);
  const double distance = cv::norm(*rm_center - fused.center);
  power_rune.r_gate_distance = static_cast<float>(distance);
  power_rune.r_gate_threshold = static_cast<float>(threshold);

  if (distance <= threshold) {
    r_gate_reject_streak_ = 0;
    accept(*rm_center, RCenterSource::RmCore);
    return;
  }

  // 超门限。连续拒绝逃生：融合 kpt4 自己跑偏（例如多片 kpt4 系统性偏移）时，
  // 不能把持续正确的 rm R 永久锁在门外，否则门控本身变成新的故障源。
  ++r_gate_reject_streak_;
  if (r_gate_reject_streak_ >= kRGateEscapeStreak) {
    tools::logger()->debug(
      "Buff R gate escaped after {} rejects (dist {:.1f}px > thresh {:.1f}px)",
      r_gate_reject_streak_, distance, threshold);
    r_gate_reject_streak_ = 0;
    accept(*rm_center, RCenterSource::RmCoreEscaped);
    return;
  }
  accept(fused.center, RCenterSource::FusedKpt4);
}

void Buff_Detector::handle_loss()
{
  ++lose_count_;
  if (lose_count_ < loss_max_frames_) {
    status_ = TEM_LOSE;
    return;
  }

  // 清空历史 = 下一帧五槽编号必须重新自举，同一个槽号可能换成另一片物理扇叶。
  // 纪元只在"从有到无"的那一次 +1；越过阈值后 lose_count_ 不复位，
  // 若每帧都自增会把纪元变成"丢失态帧数"，下游无法用它判断编号是否真的换过。
  if (last_powerrune_.has_value()) ++slot_epoch_;
  last_powerrune_.reset();
  secondary_slot_.reset();
  secondary_miss_count_ = 0;
  status_ = LOSE;
}

std::optional<PowerRune> Buff_Detector::detect_group(
  cv::Mat & bgr_img, const Eigen::Quaterniond & q_gimbal2world,
  std::chrono::steady_clock::time_point timestamp)
{
  last_rm_center_.reset();
  last_rm_center_ms_ = 0.0;
  if (rm_center_enabled_) {
    const auto rm_begin = std::chrono::steady_clock::now();
    // YOLO 会在 bgr_img 上画框；rm-core 的颜色二值化必须先看到未污染的原图。
    last_rm_center_ = rm_center_detector_.detect_center(bgr_img, q_gimbal2world, timestamp);
    last_rm_center_ms_ = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - rm_begin)
                           .count();
  }

  // 一帧只推理和 NMS 一次，所有目标共同构成一个五槽 PowerRune。
  const auto yolo_begin = std::chrono::steady_clock::now();
  auto objects = model_.get_multicandidateboxes(bgr_img);
  last_yolo_buff_ms_ = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - yolo_begin)
                         .count();
  if (objects.empty()) {
    handle_loss();
    return std::nullopt;
  }

  auto group = make_power_rune(objects, next_frame_id_++, last_rm_center_);
  if (!group.has_value()) {
    handle_loss();
    return std::nullopt;
  }

  if (group->slot_observed(group->target_slot())) {
    lose_count_ = 0;
    status_ = TRACK;
    last_powerrune_ = group;
    return group;
  }

  // 当前帧仍然有有效候选，只是历史主槽没有被观测到。它不是“整帧丢失”，
  // 不能进入 buff_loss_max_frames 的等待；detect()/detect_dual() 会立即从当前观测中接管主槽。
  status_ = TEM_LOSE;
  last_powerrune_ = group;
  return group;
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  return detect(bgr_img, Eigen::Quaterniond::Identity(), std::chrono::steady_clock::now());
}

std::optional<PowerRune> Buff_Detector::detect(
  cv::Mat & bgr_img, const Eigen::Quaterniond & q_gimbal2world,
  std::chrono::steady_clock::time_point timestamp)
{
  auto group = detect_group(bgr_img, q_gimbal2world, timestamp);
  if (!group.has_value()) return std::nullopt;

  if (group->slot_observed(group->target_slot()))
    return group->with_target(group->target_slot());

  // 小符只有一个当前击打目标。旧主槽消失但本帧已经检测到其他有效槽时，
  // 不能把“有效观测”当成 buff_loss_max_frames 的丢失帧，否则新 target 会被延迟。
  // 大符走独立的 detect_dual() 双目标选择逻辑，不会重复调用这里的单目标接口。
  std::optional<std::size_t> replacement_slot;
  for (std::size_t slot = 0; slot < group->fanblades.size(); ++slot) {
    if (!group->slot_observed(slot)) continue;
    if (
      !replacement_slot.has_value() ||
      group->fanblades[slot].object_confidence >
        group->fanblades[*replacement_slot].object_confidence)
      replacement_slot = slot;
  }
  if (!replacement_slot.has_value()) return std::nullopt;

  auto switched = group->with_target(*replacement_slot);
  if (switched.is_unsolve()) return std::nullopt;

  // 把新槽写回关联历史，下一帧从新 target 继续跟踪；不再继续累计旧主槽丢失帧。
  last_powerrune_ = switched;
  secondary_slot_.reset();
  secondary_miss_count_ = 0;
  lose_count_ = 0;
  status_ = TRACK;
  return switched;
}

std::pair<std::optional<PowerRune>, std::optional<PowerRune>>
Buff_Detector::detect_dual(
  cv::Mat & bgr_img, const Eigen::Quaterniond & q_gimbal2world,
  std::chrono::steady_clock::time_point timestamp)
{
  auto group = detect_group(bgr_img, q_gimbal2world, timestamp);
  if (!group.has_value()) return {std::nullopt, std::nullopt};

  std::size_t primary_slot = group->target_slot();
  if (!group->slot_observed(primary_slot)) {
    // 大符开火后的下一块 target 可能与上一主槽相隔任意位置。
    // 只要当前帧有有效观测，立即接管置信度最高的槽，不能等待 buff_loss_max_frames 帧。
    std::optional<std::size_t> replacement_slot;
    for (std::size_t slot = 0; slot < group->fanblades.size(); ++slot) {
      if (!group->slot_observed(slot)) continue;
      if (
        !replacement_slot.has_value() ||
        group->fanblades[slot].object_confidence >
          group->fanblades[*replacement_slot].object_confidence)
        replacement_slot = slot;
    }
    if (!replacement_slot.has_value()) return {std::nullopt, std::nullopt};

    primary_slot = *replacement_slot;
    auto switched = group->with_target(primary_slot);
    if (switched.is_unsolve()) return {std::nullopt, std::nullopt};
    last_powerrune_ = switched;
    lose_count_ = 0;
    status_ = TRACK;
    // 原副槽身份依赖旧主槽，主槽立即接管后必须在当前帧重新选择副槽。
    secondary_slot_.reset();
    secondary_miss_count_ = 0;
  }

  auto primary = group->with_target(primary_slot);

  if (secondary_slot_.has_value() && *secondary_slot_ == primary_slot) {
    secondary_slot_.reset();
    secondary_miss_count_ = 0;
  }

  auto select_observed_secondary = [&]() -> std::optional<std::size_t> {
    const double primary_angle = point_angle(primary.target().center, group->r_center);
    std::optional<std::size_t> selected;
    double nearest_delta = CV_2PI;
    bool has_active_marker = false;
    for (std::size_t slot = 0; slot < group->fanblades.size(); ++slot) {
      const auto & blade = group->fanblades[slot];
      if (slot != primary_slot && blade.active && blade.active_observed) {
        has_active_marker = true;
        break;
      }
    }
    for (std::size_t slot = 0; slot < group->fanblades.size(); ++slot) {
      if (slot == primary_slot || !group->slot_observed(slot)) continue;
      if (has_active_marker &&
          !(group->fanblades[slot].active && group->fanblades[slot].active_observed))
        continue;
      const double angle = point_angle(group->fanblades[slot].center, group->r_center);
      const double delta = wrapped_delta(primary_angle, angle);
      if (delta < nearest_delta) {
        nearest_delta = delta;
        selected = slot;
      }
    }
    return selected;
  };

  if (!secondary_slot_.has_value()) secondary_slot_ = select_observed_secondary();
  if (!secondary_slot_.has_value()) return {primary, std::nullopt};

  if (group->slot_observed(*secondary_slot_)) {
    secondary_miss_count_ = 0;
    return {primary, group->with_target(*secondary_slot_)};
  }

  // 已建立身份的副槽短时漏检时，保留槽位而不是让副目标闪烁或立刻换成另一片。
  // Solver 看到 observed=false 后只使用主目标姿态和固定 72° 相位补全，不读取历史二维点。
  if (++secondary_miss_count_ <= secondary_hold_max_frames_) {
    auto inferred = group->with_inferred_target(*secondary_slot_);
    if (!inferred.is_unsolve()) return {primary, inferred};
  }

  secondary_slot_.reset();
  secondary_miss_count_ = 0;
  secondary_slot_ = select_observed_secondary();
  if (!secondary_slot_.has_value()) return {primary, std::nullopt};
  return {primary, group->with_target(*secondary_slot_)};
}

void Buff_Detector::set_last_powerrune(const PowerRune & pr)
{
  // 只有开火状态机可以显式改变 selected_slot；普通检测帧绝不自动主副互换。
  last_powerrune_ = pr;
  secondary_slot_.reset();
  secondary_miss_count_ = 0;
  lose_count_ = 0;
  status_ = TRACK;
}

void Buff_Detector::reset()
{
  if (last_powerrune_.has_value()) ++slot_epoch_;
  last_powerrune_.reset();
  secondary_slot_.reset();
  secondary_miss_count_ = 0;
  lose_count_ = 0;
  last_rm_center_.reset();
  last_rm_center_ms_ = 0.0;
  last_yolo_buff_ms_ = 0.0;
  r_gate_reject_streak_ = 0;
  rm_center_detector_.reset();
  status_ = LOSE;
}

}  // namespace auto_buff
