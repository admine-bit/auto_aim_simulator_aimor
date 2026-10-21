use crate::robomaster::power_rune::collision::RuneIndex;
use crate::robomaster::power_rune::common::{RUNE_TARGET_COUNT, RuneStartupModes};
use crate::robomaster::power_rune::rotation::PowerRuneRotation;
use crate::robomaster::power_rune::rune::{PowerRune, PowerRuneMechanism};
use crate::robomaster::power_rune::visual::{PowerRuneVisuals, RuneVisual};
use crate::robomaster::prelude::Team;
use crate::robomaster::visibility::{Controller, StatefulAppearanceCreator};
use crate::util::bevy::{drain_entities_by, insert_all_child};
use crate::{material, visibility};
use avian3d::prelude::CollisionEventsEnabled;
use bevy::ecs::system::SystemParam;
use bevy::prelude::{
    ChildOf, Children, Commands, Component, Entity, In, Name, Quat, Query, Res, Transform, Vec3, With,
};
use bevy::world_serialization::{InstanceId, WorldInstanceSpawner};
use std::collections::HashMap;

#[derive(Component)]
pub struct PowerRuneRoot;

fn build_targets(
    face_index: usize,
    face_entity: Entity,
    name_map: &mut HashMap<&str, Entity>,
    param: &mut PowerRuneParam,
    creator: &mut StatefulAppearanceCreator,
) -> Option<[RuneVisual; RUNE_TARGET_COUNT]> {
    let mut targets = Vec::new();
    for target_idx in 1..=5 {
        let prefix = format!("FACE_{}_TARGET_{}", face_index, target_idx);

        let padding_segments = creator.create_controller(
            drain_entities_by(name_map, |name| {
                name.starts_with(&format!("{}_PADDING", prefix))
            }),
            material!(on = { completed }),
        );
        let progress_segments = creator.create_controller(
            drain_entities_by(name_map, |name| {
                name.starts_with(&format!("{}_LEGGING_PROGRESSING", prefix))
            }),
            visibility!(activating),
        );

        let ad = format!("{}_ACTIVATED", prefix);
        let at = format!("{}_ACTIVE", prefix);
        let d = format!("{}_DISABLED", prefix);
        let c = format!("{}_COMPLETED", prefix);
        let activated = ad.as_str();
        let active = at.as_str();
        let deactivated = d.as_str();
        let completed = c.as_str();

        let activated = name_map.remove(activated);
        let activating = name_map.remove(active);
        let deactivated = name_map.remove(deactivated);
        let completed = name_map.remove(completed);

        let logical_index = targets.len();
        for entity in [deactivated, activating, activated, completed]
            .into_iter()
            .flatten()
        {
            insert_all_child(&mut param.commands, entity, &param.children, || {
                (
                    RuneIndex {
                        target: logical_index,
                        rune: face_entity,
                    },
                    CollisionEventsEnabled,
                )
            });
        }

        let mut legging_segments: [Controller; 3] = [
            Controller::new_combined(vec![]),
            Controller::new_combined(vec![]),
            Controller::new_combined(vec![]),
        ];
        for legging_idx in 1..=3 {
            legging_segments[legging_idx - 1] = creator.create_controller(
                drain_entities_by(name_map, |name| {
                    name.starts_with(&format!("{}_LEGGING_{}", prefix, legging_idx))
                        && !name.contains("PROGRESSING")
                }),
                material!(on = {activated, completed}),
            )
        }

        targets.push(RuneVisual::new(
            Controller::new_visibility(deactivated, activating, activated, completed),
            legging_segments,
            padding_segments,
            progress_segments,
        ));
    }
    targets.try_into().ok()
}

#[derive(SystemParam)]
pub(crate) struct PowerRuneParam<'w, 's> {
    commands: Commands<'w, 's>,
    scene_spawner: Res<'w, WorldInstanceSpawner>,
    startup_modes: Res<'w, RuneStartupModes>,

    power_query: Query<'w, 's, (), With<PowerRuneRoot>>,
    names: Query<'w, 's, &'static Name>,
    children: Query<'w, 's, &'static Children>,
    child_of: Query<'w, 's, &'static ChildOf>,
    transforms: Query<'w, 's, &'static Transform>,
}

pub(crate) fn setup_power_rune(
    In((_root, instance)): In<(Entity, InstanceId)>,
    mut param: PowerRuneParam,
    mut creator: StatefulAppearanceCreator,
) {
    let names = param.names;
    let mut name_map = param
        .scene_spawner
        .iter_instance_entities(instance)
        .filter_map(|entity| names.get(entity).map(|n| (n.as_str(), entity)).ok())
        .fold(HashMap::new(), |mut m, (name, entity)| {
            m.insert(name, entity);
            m
        });

    if name_map.is_empty() {
        return;
    }

    let mut faces: Vec<(usize, Entity)> = name_map
        .iter()
        .filter_map(|(name, &entity)| {
            let rest = name.strip_prefix("FACE_")?;
            if rest.contains('_') {
                return None;
            }
            let index = rest.parse::<usize>().ok()?;
            Some((index, entity))
        })
        .collect();

    faces.sort_by_key(|(idx, _)| *idx);
    if faces.is_empty() {
        return;
    }

    // Keep the established rotation direction across simulator restarts.
    let red_clockwise = false;

    for (index, face_entity) in faces {
        let team = if (index & 1) > 0 {
            Team::Red
        } else {
            Team::Blue
        };
        let mode = param.startup_modes.for_team(team);

        let deactivated = name_map.remove(format!("FACE_{}_R_UNPOWERED", index).as_str());
        let activated = name_map.remove(format!("FACE_{}_R_POWERED", index).as_str());
        let padding = name_map.get(format!("FACE_{}_R_PADDING", index).as_str()).copied();

        // Apply the fixed letter roll for each GLB front. Their front normals
        // point in opposite directions, so the in-plane angle signs differ.
        let marker_roll_degrees: f32 = match index {
            1 => -19.0264,
            2 => 298.5325,
            _ => 0.0,
        };
        let marker_roll = Quat::from_axis_angle(
            Vec3::new(1.0, 0.0, 1.0).normalize(),
            marker_roll_degrees.to_radians(),
        );

        // Keep R fixed to the frame and rotate around the rune centre, since
        // the imported mesh origins are offset from the letter itself.
        if let (Ok(parent), Ok(face_transform)) = (
            param.child_of.get(face_entity),
            param.transforms.get(face_entity),
        ) {
            for marker in [padding, deactivated, activated].into_iter().flatten() {
                if let Ok(marker_transform) = param.transforms.get(marker) {
                    let mut fixed_transform = face_transform.mul_transform(*marker_transform);
                    fixed_transform.translation = face_transform.translation
                        + marker_roll * (fixed_transform.translation - face_transform.translation);
                    fixed_transform.rotation = marker_roll * fixed_transform.rotation;
                    param.commands.entity(marker).insert((
                        ChildOf(parent.parent()),
                        fixed_transform,
                    ));
                }
            }
        }

        let Some(targets) =
            build_targets(index, face_entity, &mut name_map, &mut param, &mut creator)
        else {
            continue;
        };

        let clockwise = match team {
            Team::Red => red_clockwise,
            Team::Blue => !red_clockwise,
        };
        let mut visuals = PowerRuneVisuals::new(
            Controller::new_visibility(deactivated, activated, activated, activated),
            targets,
        );
        let mechanism = PowerRuneMechanism::new(mode);
        visuals.apply(mode, mechanism.state(), &mut creator.appearance);

        param.commands.entity(face_entity).insert((
            PowerRune::new(team, mode),
            mechanism,
            PowerRuneRotation::new(clockwise),
            visuals,
        ));
    }
}

#[derive(Default)]
pub(super) struct PowerRuneConstructorPlugin;

impl bevy::app::Plugin for PowerRuneConstructorPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.init_resource::<RuneStartupModes>();
        let modes = app.world().resource::<RuneStartupModes>();
        bevy::log::info!("Rune startup modes: red={:?}, blue={:?}", modes.red, modes.blue);
        // `setup_power_rune` is invoked by the scene load sequence, not by an event.
    }
}
