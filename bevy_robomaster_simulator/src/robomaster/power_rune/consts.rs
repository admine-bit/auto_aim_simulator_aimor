pub(super) const ACTIVATION_PRIMARY_TIMEOUT: f32 = 2.5;
pub(super) const LARGE_SECONDARY_TIMEOUT: f32 = 1.0;
pub(super) const INACTIVE_WAIT: f32 = 1.0;
pub(super) const ROUND_BLACKOUT: f32 = 0.2; // 仅大符换组熄灯，最长 200 ms；小符正常换靶持续亮灯
pub(super) const FAILURE_RECOVER: f32 = 0.2; // 失败熄灯后在同一次 20 秒练习窗口内重置
pub(super) const ACTIVATED_HOLD: f32 = 6.0; // 仅用于独立练习的成功展示，不模拟比赛增益持续时间
pub(super) const ACTIVATION_GLOBAL_TIMEOUT: f32 = 20.0; // 20秒全局激活超时
pub(super) const ROTATION_BASELINE_SMALL: f32 = std::f32::consts::PI / 3.0; // 小机关固定角速度
