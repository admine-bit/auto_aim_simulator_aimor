mod camera;
mod ballistic_hud;
mod ballistics;
mod chassis_observation;
mod controller;
mod debug;
mod gimbal_pid;
mod input;
mod projectile;
mod uav;
mod vision_mode;
pub use camera::*;
pub use ballistic_hud::BallisticHudPlugin;
pub use chassis_observation::*;
pub use controller::*;
pub use debug::*;
pub use gimbal_pid::*;
pub use input::*;
pub use projectile::*;
pub use uav::*;
pub use vision_mode::*;

use bevy::prelude::*;

#[derive(SystemSet, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GameplaySystems {
    Input,
    GameLogic,
    Camera,
    Cleanup,
}
