use bevy::prelude::Resource;

use crate::robomaster::prelude::Team;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum RuneMode {
    Small,
    Large,
}

impl RuneMode {
    pub fn initial_vision_mode(self) -> u8 {
        match self {
            Self::Small => 2,
            Self::Large => 3,
        }
    }

    fn from_launch_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "small" => Some(Self::Small),
            "large" => Some(Self::Large),
            _ => None,
        }
    }
}

/// Startup selection only. The existing state machine and rotation model own all motion.
#[derive(Resource, Debug, Copy, Clone, PartialEq, Eq)]
pub struct RuneStartupModes {
    pub red: RuneMode,
    pub blue: RuneMode,
}

impl RuneStartupModes {
    fn from_names(red: Option<&str>, blue: Option<&str>) -> Result<Self, String> {
        let parse = |name, value: Option<&str>| {
            let value = value.unwrap_or("large");
            RuneMode::from_launch_name(value)
                .ok_or_else(|| format!("{name} 必须是 small 或 large，实际为 {value:?}"))
        };
        Ok(Self {
            red: parse("DAEDALUS_RED_RUNE_MODE", red)?,
            blue: parse("DAEDALUS_BLUE_RUNE_MODE", blue)?,
        })
    }

    pub fn for_team(self, team: Team) -> RuneMode {
        match team {
            Team::Red => self.red,
            Team::Blue => self.blue,
        }
    }
}

impl Default for RuneStartupModes {
    fn default() -> Self {
        let red = std::env::var("DAEDALUS_RED_RUNE_MODE").ok();
        let blue = std::env::var("DAEDALUS_BLUE_RUNE_MODE").ok();
        Self::from_names(red.as_deref(), blue.as_deref()).expect("无效的能量机关启动模式")
    }
}

pub const RUNE_TARGET_COUNT: usize = 5;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RuneHitOutcome {
    Ignored,
    WrongTarget,
    PrimaryHit,
    SecondaryHit,
    Activated,
}

impl RuneHitOutcome {
    pub const fn is_accurate(self) -> bool {
        matches!(
            self,
            Self::PrimaryHit | Self::SecondaryHit | Self::Activated
        )
    }

    pub const fn activates_rune(self) -> bool {
        matches!(self, Self::Activated)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RuneTransition {
    None,
    Started,
    Advanced,
    Failed,
    Activated,
    ResetToInactive,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_modes_select_both_teams_independently() {
        for (red_name, red) in [("small", RuneMode::Small), ("large", RuneMode::Large)] {
            for (blue_name, blue) in [("small", RuneMode::Small), ("large", RuneMode::Large)] {
                let modes = RuneStartupModes::from_names(Some(red_name), Some(blue_name)).unwrap();
                assert_eq!(modes.for_team(Team::Red), red);
                assert_eq!(modes.for_team(Team::Blue), blue);
                assert_eq!(
                    modes.red.initial_vision_mode(),
                    if red_name == "small" { 2 } else { 3 }
                );
            }
        }
    }

    #[test]
    fn startup_defaults_preserve_both_large_and_reject_invalid_modes() {
        assert_eq!(
            RuneStartupModes::from_names(None, None).unwrap(),
            RuneStartupModes {
                red: RuneMode::Large,
                blue: RuneMode::Large,
            }
        );
        assert!(
            RuneStartupModes::from_names(Some("largeish"), None)
                .unwrap_err()
                .contains("RED")
        );
        assert!(
            RuneStartupModes::from_names(None, Some("2"))
                .unwrap_err()
                .contains("BLUE")
        );
    }
}
