use crate::components::GameLayer;
use crate::robomaster::power_rune::collision::{
    RuneColliderProxy, RuneHitSurface, RuneIndex, RuneSolid,
};
use crate::robomaster::power_rune::common::{
    RUNE_TARGET_COUNT, RuneRotationDirection, RuneStartupModes,
};
use crate::robomaster::power_rune::rotation::PowerRuneRotation;
use crate::robomaster::power_rune::rune::{PowerRune, PowerRuneMechanism};
use crate::robomaster::power_rune::visual::{PowerRuneVisuals, RuneVisual};
use crate::robomaster::prelude::Team;
use crate::robomaster::visibility::{Controller, StatefulAppearanceCreator};
use crate::util::bevy::drain_entities_by;
use crate::{material, visibility};
use avian3d::prelude::{
    Collider, CollisionEventsEnabled, CollisionMargin, Restitution, RigidBody, TrimeshFlags,
};
use bevy::ecs::system::SystemParam;
use bevy::prelude::{
    Assets, ChildOf, Children, Commands, Component, Entity, In, Mat4, Mesh, Mesh3d, Name, Quat,
    Query, Res, Transform, Vec3, With,
};
use bevy::world_serialization::{InstanceId, WorldInstanceSpawner};
use std::collections::HashMap;

#[derive(Component)]
pub struct PowerRuneRoot;

fn build_targets(
    face_index: usize,
    face_entity: Entity,
    face_normal: Vec3,
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

        // One permanent physical disc per slot. Display variants have different origins,
        // depths and geometry, and hiding a mesh does not disable its collision.
        let reference = activated?;
        let points = target_points(reference, face_entity, param);
        let center = target_geometry(&points, face_normal)?;
        param.commands.spawn(RuneSolid).insert((
            Name::new(format!("{prefix}_HIT_DISC")),
            Transform::IDENTITY,
            RuneColliderProxy {
                source: face_entity,
                local: Transform::from_translation(center - face_normal * 0.0005)
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, face_normal)),
            },
            RigidBody::Static,
            Collider::cylinder(0.150, 0.001),
            CollisionMargin(0.0),
            Restitution::ZERO,
            GameLayer::environment_collision_layers(),
            CollisionEventsEnabled,
            RuneIndex {
                target: targets.len(),
                rune: face_entity,
            },
            RuneHitSurface {
                center: Vec3::Y * 0.0005,
                normal: Vec3::Y,
            },
        ));

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
    rotation_direction: Res<'w, RuneRotationDirection>,

    power_query: Query<'w, 's, (), With<PowerRuneRoot>>,
    names: Query<'w, 's, &'static Name>,
    children: Query<'w, 's, &'static Children>,
    child_of: Query<'w, 's, &'static ChildOf>,
    transforms: Query<'w, 's, &'static Transform>,
    mesh_handles: Query<'w, 's, &'static Mesh3d>,
    meshes: Res<'w, Assets<Mesh>>,
}

/// Mesh vertices expressed in the rotating face's coordinates. Only the circular
/// reference variant contributes; visible state changes never move the scoring origin.
fn target_points(root: Entity, face: Entity, param: &PowerRuneParam) -> Vec<Vec3> {
    fn visit(e: Entity, matrix: Mat4, p: &PowerRuneParam, points: &mut Vec<Vec3>) {
        if let Ok(handle) = p.mesh_handles.get(e) {
            if let Some(mesh) = p.meshes.get(&handle.0) {
                if let Some(bevy::mesh::VertexAttributeValues::Float32x3(vertices)) =
                    mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                {
                    points.extend(
                        vertices
                            .iter()
                            .map(|v| matrix.transform_point3(Vec3::from_array(*v))),
                    );
                }
            }
        }
        if let Ok(children) = p.children.get(e) {
            for child in children {
                let local = p
                    .transforms
                    .get(*child)
                    .map(Transform::to_matrix)
                    .unwrap_or(Mat4::IDENTITY);
                visit(*child, matrix * local, p, points);
            }
        }
    }
    let mut matrix = Mat4::IDENTITY;
    let mut current = root;
    while current != face {
        let Ok(local) = param.transforms.get(current) else {
            return vec![];
        };
        matrix = local.to_matrix() * matrix;
        let Ok(parent) = param.child_of.get(current) else {
            return vec![];
        };
        current = parent.parent();
    }
    let mut points = vec![];
    visit(root, matrix, param, &mut points);
    points
}

/// Fit the reference disc's outer circle in its plane, rather than using a
/// polygon's axis-aligned bounds (whose midpoint drifts with vertex sampling).
/// Returns the front surface centre in the same coordinate system as `points`.
pub(super) fn target_geometry(points: &[Vec3], normal: Vec3) -> Option<Vec3> {
    if points.len() < 8 {
        return None;
    }
    let normal = normal.try_normalize()?;
    let u = normal.any_orthonormal_vector();
    let v = normal.cross(u);
    let origin = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let projected: Vec<_> = points
        .iter()
        .map(|p| {
            let d = *p - origin;
            (d.dot(u) as f64, d.dot(v) as f64)
        })
        .collect();
    let mut lo = bevy::math::DVec2::splat(f64::INFINITY);
    let mut hi = bevy::math::DVec2::splat(f64::NEG_INFINITY);
    for &(x, y) in &projected {
        lo = lo.min(bevy::math::DVec2::new(x, y));
        hi = hi.max(bevy::math::DVec2::new(x, y));
    }
    let coarse = (lo + hi) * 0.5;
    let outer = projected
        .iter()
        .map(|&(x, y)| (bevy::math::DVec2::new(x, y) - coarse).length())
        .fold(0.0_f64, f64::max);
    let rim: Vec<_> = projected
        .into_iter()
        .filter(|&(x, y)| (bevy::math::DVec2::new(x, y) - coarse).length() >= outer - 0.002)
        .collect();
    if rim.len() < 8 {
        return None;
    }
    // Least squares: 2*cx*x + 2*cy*y + c = x*x+y*y.
    let mut a = [[0.0_f64; 4]; 3];
    for &(x, y) in &rim {
        let row = [2.0 * x, 2.0 * y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += row[i] * row[j];
            }
            a[i][3] += row[i] * (x * x + y * y);
        }
    }
    for i in 0..3 {
        let pivot = (i..3).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs()))?;
        a.swap(i, pivot);
        if a[i][i].abs() < 1e-12 {
            return None;
        }
        let divisor = a[i][i];
        for j in i..4 {
            a[i][j] /= divisor;
        }
        for k in 0..3 {
            if k == i {
                continue;
            }
            let factor = a[k][i];
            for j in i..4 {
                a[k][j] -= factor * a[i][j];
            }
        }
    }
    let cx = a[0][3];
    let cy = a[1][3];
    let radius = (a[2][3] + cx * cx + cy * cy).sqrt();
    if !radius.is_finite() || (radius - 0.15).abs() > 0.001 {
        return None;
    }
    if rim
        .iter()
        .any(|&(x, y)| (((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - radius).abs() > 0.001)
    {
        return None;
    }
    let front = points
        .iter()
        .map(|p| (*p - origin).dot(normal))
        .fold(f32::NEG_INFINITY, f32::max);
    Some(origin + u * cx as f32 + v * cy as f32 + normal * front)
}

/// Include the solid frame, rotating arm skeletons and target rims. Thin glowing
/// overlays are decoration; they must not obscure or duplicate the scoring disc.
pub(super) fn structural_rune_node(name: &str) -> bool {
    name == "BASE"
        || name.starts_with("FACE_") && (name.contains("_BASE") || name.ends_with("_PADDING"))
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

    for (name, &entity) in &name_map {
        if !structural_rune_node(name) {
            continue;
        }
        for source in std::iter::once(entity).chain(param.children.iter_descendants(entity)) {
            let Ok(handle) = param.mesh_handles.get(source) else {
                continue;
            };
            let mesh = param
                .meshes
                .get(&handle.0)
                .expect("rune structure mesh must be loaded");
            let collider = Collider::trimesh_from_mesh_with_config(
                mesh,
                TrimeshFlags::FIX_INTERNAL_EDGES
                    | TrimeshFlags::DELETE_DEGENERATE_TRIANGLES
                    | TrimeshFlags::DELETE_DUPLICATE_TRIANGLES,
            )
            .expect("rune structure mesh must have valid triangle collision geometry");
            // Physics writes only this independent proxy. It must never write a
            // visual node's local transform under the rotating glTF face.
            param.commands.spawn(RuneSolid).insert((
                Name::new(format!("{name}_PHYSICS")),
                RigidBody::Static,
                collider,
                CollisionMargin(0.0),
                Restitution::ZERO,
                GameLayer::environment_collision_layers(),
                Transform::IDENTITY,
                RuneColliderProxy {
                    source,
                    local: Transform::IDENTITY,
                },
            ));
        }
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

    for (index, face_entity) in faces {
        let team = if (index & 1) > 0 {
            Team::Red
        } else {
            Team::Blue
        };
        let mode = param.startup_modes.for_team(team);

        let deactivated = name_map.remove(format!("FACE_{}_R_UNPOWERED", index).as_str());
        let activated = name_map.remove(format!("FACE_{}_R_POWERED", index).as_str());
        let padding = name_map
            .get(format!("FACE_{}_R_PADDING", index).as_str())
            .copied();

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
                    param
                        .commands
                        .entity(marker)
                        .insert((ChildOf(parent.parent()), fixed_transform));
                }
            }
        }

        let clockwise = param.rotation_direction.for_team(team);
        let initial_rotation = param
            .transforms
            .get(face_entity)
            .map(|t| t.rotation)
            .unwrap_or_default();
        let rotation = PowerRuneRotation::for_face(clockwise, team, initial_rotation);
        let face_normal = *rotation.controller().axis();
        let Some(targets) = build_targets(
            index,
            face_entity,
            face_normal,
            &mut name_map,
            &mut param,
            &mut creator,
        ) else {
            panic!(
                "FACE_{index}: cannot construct five precise rune hit discs from reference geometry"
            );
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
            rotation,
            visuals,
        ));
    }
}

#[derive(Default)]
pub(super) struct PowerRuneConstructorPlugin;

impl bevy::app::Plugin for PowerRuneConstructorPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        super::assembly_audit::install(app);
        app.init_resource::<RuneStartupModes>()
            .init_resource::<RuneRotationDirection>();
        let modes = app.world().resource::<RuneStartupModes>();
        bevy::log::info!(
            "Rune startup modes: red={:?}, blue={:?}",
            modes.red,
            modes.blue
        );
        bevy::log::info!(
            "Rune direction facing the red side: clockwise={}",
            app.world()
                .resource::<RuneRotationDirection>()
                .red_clockwise
        );
        // `setup_power_rune` is invoked by the scene load sequence, not by an event.
    }
}

#[cfg(test)]
#[path = "asset_collision_tests.rs"]
mod asset_collision_tests;
