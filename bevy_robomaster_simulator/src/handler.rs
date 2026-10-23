use bevy::{
    asset::AssetServer,
    audio::AudioPlayer,
    ecs::{
        observer::On,
        system::{Commands, Query, Res, ResMut},
    },
    transform::components::Transform,
};

use crate::{
    robomaster::prelude::{PowerRune, RuneActivated, RuneHit, RuneHitFeedback, RuneImpactFeedback},
    statistic::ProjectileStatistics,
};

pub fn on_activate(
    ev: On<RuneActivated>,
    mut commands: Commands,
    query: Query<&PowerRune>,
    asset_server: Res<AssetServer>,
) {
    let Ok(_rune) = query.get(ev.rune) else {
        return;
    };
    commands.spawn(AudioPlayer::new(asset_server.load("rune_activated.ogg")));
}

pub fn on_hit(
    ev: On<RuneHit>,
    mut stats: ResMut<ProjectileStatistics>,
    _commands: Commands,
    query: Query<(&Transform, &PowerRune)>,
    mut feedback: ResMut<RuneHitFeedback>,
) {
    let Ok((_transform, rune)) = query.get(ev.rune) else {
        return;
    };
    feedback.last = Some(RuneImpactFeedback {
        team: rune.team(),
        result: ev.result,
    });
    bevy::log::info!(
        "rune impact: team={:?} target={} ring={} radius_mm={} outcome={:?}",
        rune.team(),
        ev.result.target + 1,
        ev.result.ring,
        ev.result.radial_mm,
        ev.result.outcome
    );
    if ev.result.accurate() {
        stats.increase_accurate();
        //commands.spawn(AudioPlayer::new(asset_server.load("rune_activated.ogg")));
    }
}
