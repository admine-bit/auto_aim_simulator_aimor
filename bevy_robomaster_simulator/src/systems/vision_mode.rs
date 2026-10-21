use crate::robomaster::prelude::RuneStartupModes;
use bevy::prelude::*;

/// 与视觉端 GimbalToVision::mode 保持一致。
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionMode(pub u8);

impl Default for VisionMode {
    fn default() -> Self {
        Self(RuneStartupModes::default().red.initial_vision_mode())
    }
}

pub fn select_vision_mode(keyboard: Res<ButtonInput<KeyCode>>, mut mode: ResMut<VisionMode>) {
    let selected = [
        KeyCode::Digit0,
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
    ]
    .iter()
    .position(|key| keyboard.just_pressed(*key));

    if let Some(selected) = selected {
        let selected = selected as u8;
        if mode.0 != selected {
            mode.0 = selected;
            info!("Vision mode: {}", selected);
        }
    }
}
