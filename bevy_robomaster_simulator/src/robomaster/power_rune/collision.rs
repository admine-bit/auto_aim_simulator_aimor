use super::common::RuneHitOutcome;
use super::rotation::PowerRuneRotation;
use super::rune::PowerRuneMechanism;
use avian3d::prelude::{CollisionEventsEnabled, CollisionStart, Collisions};
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Component)]
#[require(CollisionEventsEnabled)]
pub struct Projectile;
/// Only normal 17 mm rounds can operate the mechanism. Darts deliberately have no RuneBullet.
#[derive(Component)]
pub struct RuneBullet {
    pub diameter_m: f32,
}
#[derive(Component, Debug, Clone, Copy)]
pub struct RuneHitSurface {
    pub center: Vec3,
}
#[derive(Resource, Default)]
struct ConsumedRuneProjectiles {
    projectiles: HashSet<Entity>,
    last_detection: HashMap<(Entity, usize), f64>,
}
#[derive(Component, Debug, Copy, Clone)]
pub struct RuneIndex {
    pub target: usize,
    pub rune: Entity,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HitResult {
    pub outcome: RuneHitOutcome,
    pub target: usize,
    pub ring: u8,
    pub radial_mm: u16,
}
impl HitResult {
    pub const fn accurate(self) -> bool {
        self.outcome.is_accurate()
    }
}
#[derive(EntityEvent)]
pub struct RuneActivated {
    #[event_target]
    pub rune: Entity,
}
#[derive(EntityEvent)]
pub struct RuneHit {
    #[event_target]
    pub rune: Entity,
    pub result: HitResult,
}

#[derive(Debug, Clone, Copy)]
pub struct RuneImpactFeedback {
    pub team: crate::robomaster::prelude::Team,
    pub result: HitResult,
}
#[derive(Resource, Default)]
pub struct RuneHitFeedback {
    pub last: Option<RuneImpactFeedback>,
}

/// Fig. 5-17: 30, 60, ..., 300 mm diameters. Report radius at the specified 1 mm
/// radial resolution; an exact boundary belongs to the inner (higher) ring.
fn ring_from_radius(radius_m: f32) -> Option<(u8, u16)> {
    if !radius_m.is_finite() || !(0.0..=0.150001).contains(&radius_m) {
        return None;
    }
    let mm = (radius_m * 1000.0).round().clamp(0.0, 150.0) as u16;
    let ring = 10_u16.saturating_sub(mm.saturating_sub(1) / 15) as u8;
    Some((ring, mm))
}

fn valid_detection(
    diameter: f32,
    closing_speed: f32,
    outward_dot: f32,
    offset: Vec3,
    normal: Vec3,
) -> bool {
    let radial = offset - normal * offset.dot(normal);
    (diameter - 0.017).abs() < 0.0005
        && closing_speed > 12.0
        && outward_dot > 0.9
        && radial.length_squared() <= 0.150_f32.powi(2) + 1e-7
}

fn handle_rune_collision(
    event: On<CollisionStart>,
    mut commands: Commands,
    mut consumed: ResMut<ConsumedRuneProjectiles>,
    mut runes: Query<(
        &mut PowerRuneMechanism,
        &mut PowerRuneRotation,
        &GlobalTransform,
    )>,
    targets: Query<&RuneIndex>,
    surfaces: Query<(&RuneHitSurface, &GlobalTransform)>,
    projectiles: Query<(Entity, &RuneBullet), With<Projectile>>,
    child_of: Query<&ChildOf>,
    collisions: Collisions,
    time: Res<Time>,
) {
    let a = event.body1.and_then(|e| projectiles.get(e).ok());
    let b = event.body2.and_then(|e| projectiles.get(e).ok());
    let (projectile, bullet, collider) = match (a, b) {
        (Some((e, p)), _) => (e, p, event.collider2),
        (_, Some((e, p))) => (e, p, event.collider1),
        _ => return,
    };
    let chain: Vec<Entity> = std::iter::once(collider)
        .chain(child_of.iter_ancestors(collider))
        .collect();
    let Some(target) = chain.iter().find_map(|e| targets.get(*e).ok()).copied() else {
        return;
    };
    let Some((surface, transform)) = chain.iter().find_map(|e| surfaces.get(*e).ok()) else {
        return;
    };
    let Ok((mut mechanism, mut rotation, rune_transform)) = runes.get_mut(target.rune) else {
        return;
    };
    let Some(pair) = collisions.get(event.collider1, event.collider2) else {
        return;
    };
    let normal = rune_transform.rotation() * *rotation.controller().axis();
    let center = transform.transform_point(surface.center);
    let detected = pair.manifolds.iter().find_map(|m| {
        let outward = if pair.collider1 == collider {
            m.normal
        } else {
            -m.normal
        };
        m.points.iter().find_map(|p| {
            if !valid_detection(
                bullet.diameter_m,
                -p.normal_speed,
                outward.dot(normal),
                p.point - center,
                normal,
            ) {
                return None;
            }
            let offset = p.point - center;
            ring_from_radius((offset - normal * offset.dot(normal)).length())
        })
    });
    let Some((ring, radial_mm)) = detected else {
        return;
    };
    if !consumed.projectiles.insert(projectile) {
        return;
    }
    // A physical hit can only operate one module, including overlapping visual-state colliders.
    commands
        .entity(projectile)
        .remove::<CollisionEventsEnabled>();
    let now = time.elapsed_secs_f64();
    let previous = consumed
        .last_detection
        .entry((target.rune, target.target))
        .or_insert(f64::NEG_INFINITY);
    if now - *previous < 0.050 {
        return;
    }
    *previous = now;
    let mut rng = rand::rng();
    let outcome = mechanism.state_mut().hit(target.target, &mut rng);
    rotation.sync_activation(
        mechanism.state().mode(),
        mechanism.state().is_activating(),
        &mut rng,
    );
    commands.trigger(RuneHit {
        rune: target.rune,
        result: HitResult {
            outcome,
            target: target.target,
            ring,
            radial_mm,
        },
    });
    if outcome.activates_rune() {
        commands.trigger(RuneActivated { rune: target.rune });
    }
}
fn cleanup_consumed(
    mut consumed: ResMut<ConsumedRuneProjectiles>,
    projectiles: Query<(), With<Projectile>>,
) {
    consumed.projectiles.retain(|e| projectiles.contains(*e));
}
#[derive(Default)]
pub(super) struct PowerRuneCollisionPlugin;
impl Plugin for PowerRuneCollisionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConsumedRuneProjectiles>()
            .init_resource::<RuneHitFeedback>()
            .add_systems(Update, cleanup_consumed)
            .add_observer(handle_rune_collision);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ten_rings_use_fifteen_mm_radial_steps_with_one_mm_readout() {
        assert_eq!(ring_from_radius(0.0), Some((10, 0)));
        for ring in 1..=10_u8 {
            let outer = (11 - ring) as f32 * 0.015;
            assert_eq!(
                ring_from_radius(outer),
                Some((ring, ((11 - ring) as u16) * 15))
            );
            assert_eq!(
                ring_from_radius(outer - 0.007),
                Some((ring, ((11 - ring) as u16) * 15 - 7))
            );
        }
        assert_eq!(ring_from_radius(0.0156), Some((9, 16)));
        assert_eq!(ring_from_radius(0.151), None);
        assert_eq!(ring_from_radius(f32::NAN), None);
    }
    #[test]
    fn detection_requires_seventeen_mm_front_impact_above_twelve_mps_within_300_mm_disc() {
        let valid = |d, v, n, x| valid_detection(d, v, n, Vec3::new(x, 0.0, 0.0), Vec3::Z);
        assert!(valid(0.017, 12.01, 1.0, 0.149));
        assert!(!valid(0.042, 25.0, 1.0, 0.0));
        assert!(!valid(0.017, 12.0, 1.0, 0.0));
        assert!(!valid(0.017, 25.0, -1.0, 0.0));
        assert!(!valid(0.017, 25.0, 0.5, 0.0));
        assert!(!valid(0.017, 25.0, 1.0, 0.151));
    }

    #[test]
    fn real_physics_uses_incoming_contact_speed_and_counts_one_front_hit() {
        use crate::robomaster::prelude::{Activation, RuneMode, Team};
        use avian3d::prelude::*;
        use std::time::Duration;
        #[derive(Resource, Default)]
        struct Log(Vec<HitResult>);
        for (speed, diameter, rear, offset, expected) in [
            (25.0, 0.017, false, 0.0, 1),
            (25.0, 0.017, false, 0.075, 1),
            (25.0, 0.017, false, 0.149, 1),
            (10.0, 0.017, false, 0.0, 0),
            (25.0, 0.042, false, 0.0, 0),
            (25.0, 0.017, true, 0.0, 0),
            (25.0, 0.017, false, 0.2, 0),
        ] {
            let mut app = App::new();
            app.add_plugins((
                MinimalPlugins,
                TransformPlugin,
                AssetPlugin::default(),
                bevy::mesh::MeshPlugin,
                PhysicsPlugins::default(),
                PowerRuneCollisionPlugin,
            ))
            .insert_resource(Gravity(Vec3::ZERO))
            .insert_resource(Time::<Fixed>::from_hz(120.0))
            .insert_resource(SubstepCount(8))
            .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                Duration::from_secs_f64(1.0 / 120.0),
            ))
            .init_resource::<Log>()
            .add_observer(|event: On<RuneHit>, mut log: ResMut<Log>| {
                log.0.push(event.result);
            });
            app.finish();
            app.cleanup();
            let mut mechanism = PowerRuneMechanism::new(RuneMode::Small);
            *mechanism.state_mut() =
                super::super::state::MechanismState::start(RuneMode::Small, &mut rand::rng());
            let active = mechanism
                .state()
                .target_states()
                .iter()
                .position(|a| *a == Activation::Activating)
                .unwrap();
            let rune = app
                .world_mut()
                .spawn((
                    mechanism,
                    PowerRuneRotation::for_face(false, Team::Red, Quat::IDENTITY),
                    Transform::IDENTITY,
                ))
                .id();
            app.world_mut().spawn((
                RigidBody::Static,
                Collider::cuboid(0.6, 0.6, 0.02),
                Transform::from_rotation(Quat::from_rotation_y(-3.0 * std::f32::consts::PI / 4.0)),
                RuneIndex {
                    rune,
                    target: active,
                },
                RuneHitSurface { center: Vec3::ZERO },
                ChildOf(rune),
            ));
            let n = Vec3::new(-1.0, 0.0, -1.0).normalize() * if rear { -1.0 } else { 1.0 };
            app.world_mut().spawn((
                RigidBody::Dynamic,
                Collider::sphere(0.0085),
                Transform::from_translation(n * 0.6 + Vec3::Y * offset),
                LinearVelocity(-n * speed),
                Mass(0.017),
                Restitution::new(0.3),
                Projectile,
                RuneBullet {
                    diameter_m: diameter,
                },
            ));
            for _ in 0..200 {
                app.update();
            }
            assert_eq!(
                app.world().resource::<Log>().0.len(),
                expected,
                "speed={speed} diameter={diameter} rear={rear} offset={offset}"
            );
            if expected == 1 {
                let hit = app.world().resource::<Log>().0[0];
                let (ring, mm) = ring_from_radius(offset).unwrap();
                assert_eq!((hit.ring, hit.radial_mm), (ring, mm));
            }
        }
    }
}
