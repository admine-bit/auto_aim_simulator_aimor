use super::common::{RuneHitOutcome, RuneMode};
use super::rotation::PowerRuneRotation;
use super::rune::PowerRuneMechanism;
use crate::config::SimulationConfig;
use avian3d::collider_tree::{
    ColliderTreeProxyKey, ColliderTreeSystems, ColliderTrees, MovedProxies,
};
use avian3d::collision::collider::EnlargedAabb;
use avian3d::dynamics::integrator::{IntegrationSystems, VelocityIntegrationData};
use avian3d::physics_transform::PhysicsTransformSystems;
use avian3d::prelude::{
    ActiveCollisionHooks, AngularVelocity, BroadPhaseSystems, Collider, ColliderAabb,
    ColliderDisabled, CollisionEventsEnabled, CollisionHooks, CollisionMargin, Collisions,
    ContactPair, LinearVelocity, NarrowPhaseConfig, PhysicsLengthUnit, PhysicsSchedule,
    PhysicsStepSystems, PhysicsSystems, Position, RigidBody, Rotation, SimpleCollider,
    SolverSystems, SubstepCount, SweptCcd,
};
use bevy::ecs::{entity::Entities, system::SystemParam};
use bevy::prelude::*;
use bevy::transform::{TransformSystems, helper::TransformHelper};
use std::collections::{HashMap, HashSet};

#[derive(Component)]
#[require(CollisionEventsEnabled)]
pub struct Projectile;
/// Only normal 17 mm rounds can operate the mechanism. Darts deliberately have no RuneBullet.
#[derive(Component)]
pub struct RuneBullet {
    pub diameter_m: f32,
}

/// Exact rune surfaces use swept CCD for arrival, rather than inflated nearby
/// triangle planes. Insert this before the collider is registered with Avian.
#[derive(Component)]
#[require(ActiveCollisionHooks::MODIFY_CONTACTS)]
pub struct RuneSolid;

/// A separate world-space physics body follows an authored visual source. Keeping
/// it outside the visual hierarchy prevents physics writeback from changing the
/// glTF mesh's local transform while its face rotates.
#[derive(Component)]
pub struct RuneColliderProxy {
    pub source: Entity,
    pub local: Transform,
}

pub(super) fn sync_rune_collider_proxies(
    source_transforms: TransformHelper,
    mut proxies: Query<(&RuneColliderProxy, &mut Position, &mut Rotation)>,
) {
    for (proxy, mut position, mut rotation) in &mut proxies {
        let Ok(source) = source_transforms.compute_global_transform(proxy.source) else {
            continue;
        };
        // Always derive the absolute pose from the original visual hierarchy;
        // never accumulate a delta from the proxy's previous physics pose.
        let world = source.mul_transform(proxy.local).compute_transform();
        if position.0 != world.translation {
            position.0 = world.translation;
        }
        if rotation.0 != world.rotation {
            rotation.0 = world.rotation;
        }
    }
}

fn cleanup_rune_collider_proxies(
    mut commands: Commands,
    entities: &Entities,
    proxies: Query<(Entity, &RuneColliderProxy)>,
) {
    for (entity, proxy) in &proxies {
        // Use entity existence, not a filtered component query: temporarily
        // hidden or disabled visual sources still own their physical proxy.
        if !entities.contains(proxy.source) {
            commands.entity(entity).despawn();
        }
    }
}

#[derive(SystemParam)]
pub struct RuneContactHooks<'w, 's> {
    solids: Query<'w, 's, (), With<RuneSolid>>,
    bullets: Query<'w, 's, (), With<RuneBullet>>,
}

impl CollisionHooks for RuneContactHooks<'_, '_> {
    fn modify_contacts(&self, contacts: &mut ContactPair, _commands: &mut Commands) -> bool {
        let first_bullet = self.bullets.contains(contacts.collider1)
            || contacts
                .body1
                .is_some_and(|body| self.bullets.contains(body));
        let second_bullet = self.bullets.contains(contacts.collider2)
            || contacts
                .body2
                .is_some_and(|body| self.bullets.contains(body));
        if !(first_bullet && self.solids.contains(contacts.collider2)
            || second_bullet && self.solids.contains(contacts.collider1))
        {
            return true;
        }
        // The default 5 mm contact tolerance can produce an impulse while a
        // 17 mm sphere still clears a target rim. Keep only real contact (plus
        // 0.1 mm numerical tolerance); swept CCD handles the remaining travel.
        contacts.manifolds.retain_mut(|manifold| {
            manifold.points.retain(|point| point.penetration >= -0.0001);
            !manifold.points.is_empty()
        });
        !contacts.manifolds.is_empty()
    }
}
#[derive(Component, Debug, Clone, Copy)]
pub struct RuneHitSurface {
    /// A point on the front detection plane, in this entity's local coordinates.
    pub center: Vec3,
    /// Outward normal of the front face, in this entity's local coordinates.
    pub normal: Vec3,
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
    pub recorded_at: f64,
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
    // Quantizing the readout before scoring used to expand every inner ring by 0.5 mm.
    // The tiny tolerance only absorbs f32 roundoff at an exact 15 mm boundary.
    let band = (radius_m / 0.015 - 1e-4).ceil().clamp(1.0, 10.0) as u8;
    let ring = 11 - band;
    Some((ring, mm))
}

#[derive(Clone, Copy)]
struct IncomingProjectile {
    position: Vec3,
    velocity: Vec3,
}

#[derive(Resource, Default)]
struct IncomingRuneProjectiles(HashMap<Entity, IncomingProjectile>);

#[derive(Clone, Copy)]
struct ProjectileIntegration {
    increment: Vec3,
    damping_rhs: f32,
    substeps: u32,
}

impl Default for ProjectileIntegration {
    fn default() -> Self {
        Self {
            increment: Vec3::ZERO,
            damping_rhs: 1.0,
            substeps: 1,
        }
    }
}

#[derive(Resource, Default)]
struct IncomingRuneIntegration(HashMap<Entity, ProjectileIntegration>);

/// Avian 0.7 expands moving bounds at the end of a step, so a newly spawned fast
/// body otherwise has only tight bounds on its first step. CCD then has no broad
/// phase pair to sweep against. Prime that first sweep before collecting pairs.
fn prime_new_swept_bodies(
    mut bodies: Query<
        (
            &Collider,
            &Position,
            &Rotation,
            &LinearVelocity,
            &AngularVelocity,
            Option<&CollisionMargin>,
            &mut ColliderAabb,
            &mut EnlargedAabb,
            &ColliderTreeProxyKey,
        ),
        (Added<SweptCcd>, With<RigidBody>, Without<ColliderDisabled>),
    >,
    mut trees: ResMut<ColliderTrees>,
    mut moved_proxies: ResMut<MovedProxies>,
    config: Res<NarrowPhaseConfig>,
    length_unit: Res<PhysicsLengthUnit>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    for (
        shape,
        position,
        rotation,
        velocity,
        angular,
        collision_margin,
        mut aabb,
        mut enlarged,
        key,
    ) in &mut bodies
    {
        if *key == ColliderTreeProxyKey::PLACEHOLDER {
            continue;
        }
        let tree = trees.tree_for_type_mut(key.tree_type());
        if tree.get_proxy(key.id()).is_none() {
            continue;
        }
        let end_rotation = (Quat::from_scaled_axis(angular.0 * dt) * rotation.0).normalize();
        let growth = Vec3::splat(
            config.contact_tolerance * length_unit.0
                + collision_margin.map_or(0.0, |margin| margin.0),
        );
        *aabb = shape
            .swept_aabb(
                position.0,
                *rotation,
                position.0 + velocity.0 * dt,
                end_rotation,
            )
            .grow(growth);
        if enlarged.update(&aabb, 0.05 * length_unit.0) {
            tree.bvh.init_primitives_to_nodes_if_uninit();
            tree.resize_proxy_aabb(key.id(), enlarged.get().into());
            moved_proxies.insert(*key);
        }
    }
}

fn snapshot_incoming_projectiles(
    mut incoming: ResMut<IncomingRuneProjectiles>,
    projectiles: Query<(Entity, &Position, &LinearVelocity), (With<Projectile>, With<RuneBullet>)>,
) {
    incoming.0.clear();
    incoming
        .0
        .extend(projectiles.iter().map(|(entity, position, velocity)| {
            (
                entity,
                IncomingProjectile {
                    position: position.0,
                    velocity: velocity.0,
                },
            )
        }));
}

/// Capture Avian's own force/gravity/damping increments after preparation, before
/// any contact impulses or velocity integration. This also covers configured wind.
fn snapshot_projectile_integration(
    mut integration: ResMut<IncomingRuneIntegration>,
    projectiles: Query<(Entity, &VelocityIntegrationData), (With<Projectile>, With<RuneBullet>)>,
    substeps: Res<SubstepCount>,
) {
    integration.0.clear();
    integration
        .0
        .extend(projectiles.iter().map(|(entity, data)| {
            (
                entity,
                ProjectileIntegration {
                    increment: data.linear_increment,
                    damping_rhs: data.linear_damping_rhs,
                    substeps: substeps.0,
                },
            )
        }));
}

/// Project the incoming sphere onto the real front plane at its first face contact.
/// Avian contacts may be speculative, and their midpoint is neither the impact point
/// nor the eventual impact radius for an oblique shot.
fn face_impact(
    incoming: IncomingProjectile,
    integration: ProjectileIntegration,
    diameter: f32,
    center: Vec3,
    normal: Vec3,
    step_seconds: f32,
) -> Option<(f32, u8, u16)> {
    if (diameter - 0.017).abs() >= 0.0005
        || !diameter.is_finite()
        || !incoming.position.is_finite()
        || !incoming.velocity.is_finite()
        || !normal.is_normalized()
    {
        return None;
    }
    let mut offset = incoming.position - center;
    if offset.dot(normal) < 0.0 {
        return None;
    }
    let mut velocity = incoming.velocity;
    let substeps = integration.substeps.max(1);
    let substep_seconds = step_seconds / substeps as f32;
    for substep in 0..substeps {
        // Match Avian's semi-implicit Euler order, including the real applied forces.
        velocity = velocity * integration.damping_rhs + integration.increment;
        let closing_speed = -velocity.dot(normal);
        let distance = offset.dot(normal);
        if closing_speed > 0.0 {
            let travel = ((distance - diameter * 0.5) / closing_speed).max(0.0);
            if travel <= substep_seconds + 1e-6 {
                if closing_speed <= 12.0 {
                    return None;
                }
                let impact = offset + velocity * travel;
                let radial = impact - normal * impact.dot(normal);
                let (ring, mm) = ring_from_radius(radial.length())?;
                return Some((substep as f32 * substep_seconds + travel, ring, mm));
            }
        }
        // Accumulate small displacements relative to the plane, like SolverBody's
        // delta_position, rather than repeatedly rounding large world coordinates.
        offset += velocity * substep_seconds;
    }
    // A speculative contact whose sphere does not reach the face this step waits.
    None
}

fn detect_rune_impacts(
    mut commands: Commands,
    mut consumed: ResMut<ConsumedRuneProjectiles>,
    incoming: Res<IncomingRuneProjectiles>,
    integration: Res<IncomingRuneIntegration>,
    config: Option<Res<SimulationConfig>>,
    mut runes: Query<(&mut PowerRuneMechanism, &mut PowerRuneRotation)>,
    targets: Query<&RuneIndex>,
    surfaces: Query<(&RuneHitSurface, &Position, &Rotation)>,
    projectiles: Query<
        (Entity, &RuneBullet, &Position),
        (With<Projectile>, With<CollisionEventsEnabled>),
    >,
    child_of: Query<&ChildOf>,
    collisions: Collisions,
    time: Res<Time>,
) {
    let mut detected = Vec::new();
    // CCD can stop a new sphere at the disc while its broad-phase pair still has
    // no narrow-phase manifold. Record that first arrival before another step
    // slides or pushes the sphere to a different position.
    for pair in collisions.graph().iter_active() {
        let a = pair.body1.and_then(|e| projectiles.get(e).ok());
        let b = pair.body2.and_then(|e| projectiles.get(e).ok());
        let (projectile, bullet, end_position, collider) = match (a, b) {
            (Some((e, p, position)), _) => (e, p, position.0, pair.collider2),
            (_, Some((e, p, position))) => (e, p, position.0, pair.collider1),
            _ => continue,
        };
        if consumed.projectiles.contains(&projectile) {
            continue;
        }
        let Some(&flight) = incoming.0.get(&projectile) else {
            continue;
        };
        let chain: Vec<Entity> = std::iter::once(collider)
            .chain(child_of.iter_ancestors(collider))
            .collect();
        let Some(target) = chain.iter().find_map(|e| targets.get(*e).ok()).copied() else {
            continue;
        };
        let Some((surface, position, rotation)) = chain.iter().find_map(|e| surfaces.get(*e).ok())
        else {
            continue;
        };
        // Use the same physical pose as the collider, including its f32 rounding.
        let normal = (rotation.0 * surface.normal).normalize_or_zero();
        let center = position.0 + rotation.0 * surface.center;
        let Some((travel, ring, radial_mm)) = face_impact(
            flight,
            integration.0.get(&projectile).copied().unwrap_or_default(),
            bullet.diameter_m,
            center,
            normal,
            time.delta_secs(),
        ) else {
            continue;
        };
        let end_offset = end_position - center;
        let end_distance = end_offset.dot(normal);
        let end_radial = end_offset - normal * end_distance;
        // CCD may stop at the face before generating an impulse, and an adjacent
        // backing can share the impulse. Actual arrival at the disc is also valid.
        // Micrometre tolerance accounts for world-coordinate float precision.
        let reached_face = (0.0..=bullet.diameter_m * 0.5 + 0.000003).contains(&end_distance)
            && end_radial.length_squared() <= 0.150003_f32.powi(2);
        // Actual arrival is geometric evidence even when CCD has not produced a
        // manifold yet. A sphere already bounced away needs a front-face impulse.
        let front_impulse = pair.manifolds.iter().any(|manifold| {
            let outward = if pair.collider1 == collider {
                manifold.normal
            } else {
                -manifold.normal
            };
            outward.dot(normal) > 0.9
                && manifold
                    .points
                    .iter()
                    .any(|point| point.normal_impulse > 0.0)
        });
        if reached_face {
            // CCD uses final velocity when clipping displacement after force
            // integration. Score where the sphere physically arrived, which can
            // differ from the free substep trajectory under gravity or wind.
            let (ring, radial_mm) = ring_from_radius(end_radial.length().min(0.150)).unwrap();
            detected.push((travel, projectile, target, ring, radial_mm));
        } else if front_impulse {
            detected.push((travel, projectile, target, ring, radial_mm));
        }
    }
    // If a fast sphere reaches several surfaces in one step, the first physical
    // intersection wins, independently of contact iteration or event delivery order.
    detected.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut rng = rand::rng();
    for (_, projectile, target, ring, radial_mm) in detected {
        if !consumed.projectiles.insert(projectile) {
            continue;
        }
        commands
            .entity(projectile)
            .remove::<CollisionEventsEnabled>();
        let Ok((mut mechanism, mut rotation)) = runes.get_mut(target.rune) else {
            continue;
        };
        // This is a manually selected practice range. Score the final physical
        // impact ring, after CCD/ricochet handling, and never accumulate success
        // count here. Rejected outer hits still consume their physical round but
        // do not reset a wrong target or occupy the valid-hit debounce interval.
        let minimum_ring = if mechanism.state().mode() == RuneMode::Large {
            config
                .as_ref()
                .map_or(1, |c| c.power_rune.minimum_large_ring())
        } else {
            1
        };
        if ring < minimum_ring {
            continue;
        }
        let now = time.elapsed_secs_f64();
        let previous = consumed
            .last_detection
            .entry((target.rune, target.target))
            .or_insert(f64::NEG_INFINITY);
        if now - *previous < 0.050 {
            continue;
        }
        *previous = now;
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
            .init_resource::<IncomingRuneProjectiles>()
            .init_resource::<IncomingRuneIntegration>()
            .init_resource::<RuneHitFeedback>()
            .add_systems(Update, (cleanup_consumed, cleanup_rune_collider_proxies))
            .add_systems(
                FixedPostUpdate,
                sync_rune_collider_proxies
                    .in_set(PhysicsSystems::Prepare)
                    .after(PhysicsTransformSystems::TransformToPosition)
                    .before(PhysicsSystems::StepSimulation),
            )
            .add_systems(
                PostUpdate,
                sync_rune_collider_proxies.after(TransformSystems::Propagate),
            )
            .add_systems(
                PhysicsSchedule,
                prime_new_swept_bodies
                    .after(ColliderTreeSystems::UpdateAabbs)
                    .before(BroadPhaseSystems::CollectCollisions),
            )
            .add_systems(
                PhysicsSchedule,
                snapshot_incoming_projectiles
                    .after(PhysicsStepSystems::First)
                    .before(PhysicsStepSystems::BroadPhase),
            )
            .add_systems(
                PhysicsSchedule,
                snapshot_projectile_integration
                    .in_set(SolverSystems::PreSubstep)
                    .after(IntegrationSystems::UpdateVelocityIncrements),
            )
            .add_systems(
                PhysicsSchedule,
                detect_rune_impacts.in_set(PhysicsStepSystems::Last),
            );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::robomaster::prelude::{Activation, RuneMode, Team};
    use avian3d::prelude::*;
    use std::time::Duration;

    #[test]
    fn ten_rings_use_continuous_radius_before_rounding_the_mm_readout() {
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
            if ring > 1 {
                // A 0.1 mm excursion changes the ring even though the readout rounds down.
                assert_eq!(
                    ring_from_radius(outer + 0.0001),
                    Some((ring - 1, ((11 - ring) as u16) * 15))
                );
            }
        }
        assert_eq!(ring_from_radius(0.0156), Some((9, 16)));
        assert_eq!(ring_from_radius(0.151), None);
        assert_eq!(ring_from_radius(f32::NAN), None);
    }

    #[test]
    fn front_detection_uses_normal_speed_and_waits_for_the_sphere_to_reach_the_plane() {
        let impact = |diameter, speed, distance, radial| {
            face_impact(
                IncomingProjectile {
                    position: Vec3::new(radial, distance, 0.0),
                    velocity: Vec3::NEG_Y * speed,
                },
                ProjectileIntegration::default(),
                diameter,
                Vec3::ZERO,
                Vec3::Y,
                1.0 / 120.0,
            )
        };
        assert!(impact(0.017, 12.01, 0.05, 0.149).is_some());
        assert!(impact(0.017, 12.0, 0.05, 0.0).is_none());
        assert!(impact(0.042, 25.0, 0.05, 0.0).is_none());
        assert!(impact(0.017, -25.0, -0.05, 0.0).is_none());
        assert!(impact(0.017, 25.0, -0.05, 0.0).is_none());
        assert!(impact(0.017, 25.0, 0.05, 0.151).is_none());
        assert!(impact(0.017, 25.0, 0.3, 0.0).is_none());
        assert!(impact(f32::NAN, 25.0, 0.05, 0.0).is_none());
        // Fast sideways motion cannot satisfy the 12 m/s normal-speed gate.
        assert!(
            face_impact(
                IncomingProjectile {
                    position: Vec3::Y * 0.02,
                    velocity: Vec3::new(100.0, -11.99, 0.0)
                },
                ProjectileIntegration::default(),
                0.017,
                Vec3::ZERO,
                Vec3::Y,
                1.0
            )
            .is_none()
        );
    }

    #[derive(Resource, Default)]
    struct HitLog(Vec<HitResult>);

    fn test_app() -> App {
        let mut app = unfinished_test_app();
        app.finish();
        app.cleanup();
        app
    }

    fn unfinished_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            AssetPlugin::default(),
            bevy::mesh::MeshPlugin,
            PhysicsPlugins::default().with_collision_hooks::<RuneContactHooks>(),
            PowerRuneCollisionPlugin,
        ))
        .insert_resource(Gravity(Vec3::ZERO))
        .insert_resource(Time::<Fixed>::from_hz(120.0))
        .insert_resource(SubstepCount(8))
        .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f64(1.0 / 120.0),
        ))
        .init_resource::<HitLog>()
        .add_observer(|event: On<RuneHit>, mut log: ResMut<HitLog>| log.0.push(event.result));
        app
    }

    fn mechanism(app: &mut App, mode: RuneMode, team: Team) -> Entity {
        let mut mechanism = PowerRuneMechanism::new(mode);
        *mechanism.state_mut() = super::super::state::MechanismState::start(mode, &mut rand::rng());
        app.world_mut()
            .spawn((
                mechanism,
                PowerRuneRotation::for_face(false, team, Quat::IDENTITY),
                Transform::IDENTITY,
            ))
            .id()
    }

    fn active_targets(app: &App, rune: Entity) -> Vec<usize> {
        app.world()
            .get::<PowerRuneMechanism>(rune)
            .unwrap()
            .state()
            .target_states()
            .into_iter()
            .enumerate()
            .filter_map(|(i, state)| (state == Activation::Activating).then_some(i))
            .collect()
    }

    fn disc(app: &mut App, rune: Entity, target: usize, normal: Vec3) -> Entity {
        app.world_mut()
            .spawn((
                RuneSolid,
                RigidBody::Static,
                Collider::cylinder(0.15, 0.001),
                Transform::from_rotation(Quat::from_rotation_arc(Vec3::Y, normal)),
                RuneIndex { rune, target },
                RuneHitSurface {
                    center: Vec3::Y * 0.0005,
                    normal: Vec3::Y,
                },
                ChildOf(rune),
            ))
            .id()
    }

    fn shot(
        app: &mut App,
        normal: Vec3,
        radius: f32,
        closing_speed: f32,
        tangential_speed: f32,
        diameter: f32,
        rear: bool,
    ) -> Entity {
        let tangent = normal.any_orthonormal_vector();
        let facing = if rear { -normal } else { normal };
        let velocity = -facing * closing_speed + tangent * tangential_speed;
        let travel = (0.6 - diameter * 0.5) / closing_speed;
        let impact = normal * 0.0005 + tangent * radius;
        let origin = impact + facing * 0.6 - tangent * tangential_speed * travel;
        app.world_mut()
            .spawn((
                RigidBody::Dynamic,
                Collider::sphere(diameter * 0.5),
                Transform::from_translation(origin),
                LinearVelocity(velocity),
                Mass(0.017),
                Restitution::new(0.3),
                SweptCcd::default(),
                SpeculativeMargin(0.001),
                crate::components::GameLayer::projectile_collision_layers(true),
                Projectile,
                RuneBullet {
                    diameter_m: diameter,
                },
            ))
            .id()
    }

    fn advance(app: &mut App, frames: usize) {
        for _ in 0..frames {
            app.update();
        }
    }

    fn manual_large_setting(app: &mut App, count: u8) {
        let mut config = SimulationConfig::default();
        config.power_rune.large_success_count = count;
        app.insert_resource(config);
    }

    fn wait_for_round_consumption(app: &mut App, projectile: Entity) -> f64 {
        for _ in 0..16 {
            app.update();
            if app
                .world()
                .get::<CollisionEventsEnabled>(projectile)
                .is_none()
            {
                return app.world().resource::<Time<Physics>>().elapsed_secs_f64();
            }
        }
        panic!("the round did not reach a physical rune disc");
    }

    #[test]
    fn fifth_large_secondary_collision_lights_all_blades_in_the_same_frame_and_activates_once() {
        use super::super::rune::{PowerRune, PowerRuneUpdatePlugin};
        use super::super::state::MechanismState;
        use super::super::visual::{PowerRuneVisuals, RuneVisual};
        use crate::robomaster::visibility::{Controller, StatefulAppearancePlugin};

        #[derive(Resource, Default)]
        struct ActivationLog(Vec<(Entity, f64)>);

        fn lamp(app: &mut App) -> (Controller, [Entity; 4]) {
            let states = std::array::from_fn(|_| app.world_mut().spawn(Visibility::Hidden).id());
            (
                Controller::new_visibility(
                    Some(states[0]),
                    Some(states[1]),
                    Some(states[2]),
                    Some(states[3]),
                ),
                states,
            )
        }

        fn assert_completed_lamps(app: &App, lamps: &[[Entity; 4]]) {
            for states in lamps {
                for (index, entity) in states.iter().enumerate() {
                    assert_eq!(
                        app.world().get::<Visibility>(*entity),
                        Some(if index == Activation::Completed as usize {
                            &Visibility::Visible
                        } else {
                            &Visibility::Hidden
                        }),
                        "all five blades and their light segments must show Completed immediately"
                    );
                }
            }
        }

        // Each of the first four groups may score its optional secondary. None
        // of those combinations may defer success past the fifth secondary impact.
        for secondary_hits in 0..=4 {
            let mut app = unfinished_test_app();
            app.add_plugins((StatefulAppearancePlugin, PowerRuneUpdatePlugin))
                .init_asset::<StandardMaterial>()
                .init_resource::<ActivationLog>()
                .add_observer(
                    |event: On<RuneActivated>,
                     time: Res<Time<Physics>>,
                     mut log: ResMut<ActivationLog>| {
                        log.0.push((event.rune, time.elapsed_secs_f64()));
                    },
                );
            app.finish();
            app.cleanup();
            let rune = mechanism(&mut app, RuneMode::Large, Team::Red);
            let (root, root_states) = lamp(&mut app);
            let mut lamps = Vec::new();
            let targets = std::array::from_fn(|_| {
                let mut controller = || {
                    let (controller, states) = lamp(&mut app);
                    lamps.push(states);
                    controller
                };
                RuneVisual::new(
                    controller(),
                    std::array::from_fn(|_| controller()),
                    controller(),
                    controller(),
                )
            });
            app.world_mut().entity_mut(rune).insert((
                PowerRune::new(Team::Red, RuneMode::Large),
                PowerRuneVisuals::new(root, targets),
            ));
            let first_target = active_targets(&app, rune)[0];
            let plate = disc(&mut app, rune, first_target, Vec3::Y);
            // The scoring fixture stays in world space while the real update
            // plugin still runs the mechanism, display and rotation systems.
            app.world_mut().entity_mut(plate).remove::<ChildOf>();
            advance(&mut app, 2);

            for group in 0..5 {
                let active = active_targets(&app, rune);
                assert_eq!(
                    active.len(),
                    2,
                    "group={group} secondary_hits={secondary_hits}"
                );
                app.world_mut().get_mut::<RuneIndex>(plate).unwrap().target = active[0];
                let projectile = shot(&mut app, Vec3::Y, 0.0, 25.0, 0.0, 0.017, false);
                let primary_impact_time = wait_for_round_consumption(&mut app, projectile);
                app.world_mut().despawn(projectile);

                if group == 4 {
                    let hits = &app.world().resource::<HitLog>().0;
                    assert_eq!(hits.len(), 5 + secondary_hits);
                    assert_eq!(hits.last().unwrap().outcome, RuneHitOutcome::PrimaryHit);
                    assert!(app.world().resource::<ActivationLog>().0.is_empty());
                    let mechanism = app.world().get::<PowerRuneMechanism>(rune).unwrap();
                    assert!(mechanism.state().is_activating());
                    assert_eq!(active_targets(&app, rune), [active[1]]);
                    assert_ne!(
                        mechanism.state().target_states(),
                        [Activation::Completed; 5],
                        "the fifth primary keeps its optional secondary window open"
                    );
                    for states in &lamps {
                        assert_eq!(
                            app.world()
                                .get::<Visibility>(states[Activation::Completed as usize]),
                            Some(&Visibility::Hidden),
                            "the fifth primary alone must not show Completed"
                        );
                    }

                    advance(&mut app, 8);
                    app.world_mut().get_mut::<RuneIndex>(plate).unwrap().target = active[1];
                    let secondary = shot(&mut app, Vec3::Y, 0.0, 25.0, 0.0, 0.017, false);
                    let impact_time = wait_for_round_consumption(&mut app, secondary);
                    app.world_mut().despawn(secondary);
                    assert!(impact_time - primary_impact_time < 1.0);
                    let hits = &app.world().resource::<HitLog>().0;
                    assert_eq!(hits.len(), 6 + secondary_hits);
                    assert_eq!(hits.last().unwrap().outcome, RuneHitOutcome::Activated);
                    assert_eq!(
                        app.world().resource::<ActivationLog>().0,
                        [(rune, impact_time)],
                        "success must be delivered in the same physical step as the fifth secondary"
                    );
                    assert!(
                        impact_time < 10.0,
                        "the global 20 s deadline is not involved"
                    );
                    let mechanism = app.world().get::<PowerRuneMechanism>(rune).unwrap();
                    assert!(matches!(
                        mechanism.state(),
                        MechanismState::Activated { hits, remaining, .. }
                            if *hits == (6 + secondary_hits) as u8 && *remaining > 5.98
                    ));
                    assert_eq!(
                        mechanism.state().target_states(),
                        [Activation::Completed; 5]
                    );
                    assert_completed_lamps(&app, &lamps);
                    assert_eq!(
                        app.world()
                            .get::<Visibility>(root_states[Activation::Activated as usize]),
                        Some(&Visibility::Visible),
                        "the fixed centre R marker stays illuminated"
                    );

                    // A later arrival on a completed final target cannot add
                    // score, restart the six-second display or emit success again.
                    advance(&mut app, 8);
                    app.world_mut().get_mut::<RuneIndex>(plate).unwrap().target = active[1];
                    let late = shot(&mut app, Vec3::Y, 0.0, 25.0, 0.0, 0.017, false);
                    wait_for_round_consumption(&mut app, late);
                    app.world_mut().despawn(late);
                    assert_eq!(
                        app.world().resource::<HitLog>().0.last().unwrap().outcome,
                        RuneHitOutcome::Ignored
                    );
                    advance(&mut app, 150);
                    assert_eq!(app.world().resource::<ActivationLog>().0.len(), 1);
                    assert_completed_lamps(&app, &lamps);
                    break;
                }

                assert_eq!(
                    app.world().resource::<HitLog>().0.last().unwrap().outcome,
                    RuneHitOutcome::PrimaryHit
                );
                if group < secondary_hits {
                    app.world_mut().get_mut::<RuneIndex>(plate).unwrap().target = active[1];
                    let secondary = shot(&mut app, Vec3::Y, 0.0, 25.0, 0.0, 0.017, false);
                    wait_for_round_consumption(&mut app, secondary);
                    app.world_mut().despawn(secondary);
                    assert_eq!(
                        app.world().resource::<HitLog>().0.last().unwrap().outcome,
                        RuneHitOutcome::SecondaryHit
                    );
                }
                let mut blackout_seen = false;
                for _ in 0..160 {
                    let mechanism = app.world().get::<PowerRuneMechanism>(rune).unwrap();
                    blackout_seen |=
                        mechanism.state().target_states() == [Activation::Deactivated; 5];
                    assert!(app.world().resource::<ActivationLog>().0.is_empty());
                    if active_targets(&app, rune).len() == 2 {
                        break;
                    }
                    app.update();
                }
                assert!(
                    blackout_seen,
                    "ordinary group changes retain their 200 ms blackout"
                );
            }
        }
    }

    #[test]
    fn real_disc_physics_scores_front_oblique_and_edge_impacts_on_both_faces_once() {
        for (team, normal) in [
            (Team::Red, Vec3::new(-1.0, 0.0, -1.0).normalize()),
            (Team::Blue, Vec3::new(1.0, 0.0, 1.0).normalize()),
        ] {
            for (speed, diameter, rear, radius, tangent, expected) in [
                (25.0, 0.017, false, 0.0, 0.0, Some((10, 0))),
                (25.0, 0.017, false, 0.075, 0.0, Some((6, 75))),
                (60.0, 0.017, false, 0.075, 0.0, Some((6, 75))),
                (100.0, 0.017, false, 0.0151, 0.0, Some((9, 15))),
                (25.0, 0.017, false, 0.149, 0.0, Some((1, 149))),
                (25.0, 0.017, false, 0.0601, 10.0, Some((6, 60))),
                (25.0, 0.017, false, 0.0151, -10.0, Some((9, 15))),
                (12.01, 0.017, false, 0.03, 0.0, Some((9, 30))),
                (10.0, 0.017, false, 0.0, 0.0, None),
                (25.0, 0.042, false, 0.0, 0.0, None),
                (25.0, 0.017, true, 0.0, 0.0, None),
                (25.0, 0.017, false, 0.151, 0.0, None),
                (25.0, 0.017, false, 0.2, 0.0, None),
            ] {
                let mut app = test_app();
                let rune = mechanism(&mut app, RuneMode::Small, team);
                let target = active_targets(&app, rune)[0];
                disc(&mut app, rune, target, normal);
                shot(&mut app, normal, radius, speed, tangent, diameter, rear);
                advance(&mut app, 100);
                let hits = &app.world().resource::<HitLog>().0;
                assert_eq!(
                    hits.len(),
                    usize::from(expected.is_some()),
                    "team={team:?} speed={speed} diameter={diameter} rear={rear} radius={radius} tangent={tangent}"
                );
                if let Some(expected) = expected {
                    assert_eq!(
                        (hits[0].ring, hits[0].radial_mm),
                        expected,
                        "team={team:?} radius={radius} tangent={tangent}"
                    );
                    assert_eq!(hits[0].outcome, RuneHitOutcome::PrimaryHit);
                }
            }
        }
    }

    #[test]
    fn real_disc_physics_enforces_fifty_ms_per_target_and_scores_the_new_small_target() {
        let mut app = test_app();
        let rune = mechanism(&mut app, RuneMode::Small, Team::Red);
        let first = active_targets(&app, rune)[0];
        let first_disc = disc(&mut app, rune, first, Vec3::Y);
        shot(&mut app, Vec3::Y, 0.0, 25.0, 0.0, 0.017, false);
        advance(&mut app, 8);
        assert_eq!(app.world().resource::<HitLog>().0.len(), 1);
        let next = active_targets(&app, rune)[0];
        assert_ne!(next, first);
        // A second round hits the completed disc less than 50 ms after the first;
        // it must be consumed without resetting the mechanism or emitting a second hit.
        let second = shot(&mut app, Vec3::Y, 0.0, 25.0, 0.0, 0.017, false);
        app.world_mut().get_mut::<Position>(second).unwrap().0 = Vec3::Y * 0.06;
        app.world_mut()
            .get_mut::<Transform>(second)
            .unwrap()
            .translation = Vec3::Y * 0.06;
        advance(&mut app, 2);
        assert_eq!(app.world().resource::<HitLog>().0.len(), 1);
        assert_eq!(active_targets(&app, rune), [next]);
        app.world_mut().despawn(first_disc);
        disc(&mut app, rune, next, Vec3::Y);
        shot(&mut app, Vec3::Y, 0.075, 25.0, 0.0, 0.017, false);
        advance(&mut app, 15);
        let hits = &app.world().resource::<HitLog>().0;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].target, next);
        assert_eq!(hits[1].outcome, RuneHitOutcome::PrimaryHit);
    }

    #[test]
    fn a_projectile_consumed_by_another_hit_module_cannot_operate_the_rune() {
        let mut app = test_app();
        let rune = mechanism(&mut app, RuneMode::Small, Team::Red);
        let target = active_targets(&app, rune)[0];
        disc(&mut app, rune, target, Vec3::Y);
        let projectile = shot(&mut app, Vec3::Y, 0.0, 25.0, 0.0, 0.017, false);
        app.world_mut()
            .entity_mut(projectile)
            .remove::<CollisionEventsEnabled>();
        advance(&mut app, 20);
        assert!(app.world().resource::<HitLog>().0.is_empty());
        assert_eq!(active_targets(&app, rune), [target]);
    }

    #[test]
    fn manual_large_ranges_use_continuous_ring_boundaries_in_all_three_settings() {
        for (count, limit) in [(0, 0.150), (1, 0.105), (2, 0.060)] {
            for offset in [-0.0001, 0.0, 0.0001] {
                let mut app = test_app();
                manual_large_setting(&mut app, count);
                let rune = mechanism(&mut app, RuneMode::Large, Team::Red);
                let target = active_targets(&app, rune)[0];
                disc(&mut app, rune, target, Vec3::Y);
                let before = app
                    .world()
                    .get::<PowerRuneMechanism>(rune)
                    .unwrap()
                    .state()
                    .clone();
                shot(&mut app, Vec3::Y, limit + offset, 25.0, 0.0, 0.017, false);
                advance(&mut app, 30);
                let hits = &app.world().resource::<HitLog>().0;
                assert_eq!(
                    hits.len(),
                    usize::from(offset <= 0.0),
                    "manual count={count} radius={}",
                    limit + offset
                );
                if offset <= 0.0 {
                    assert_eq!(hits[0].outcome, RuneHitOutcome::PrimaryHit);
                    assert!(
                        hits[0].ring
                            >= app
                                .world()
                                .resource::<SimulationConfig>()
                                .power_rune
                                .minimum_large_ring()
                    );
                } else {
                    assert_eq!(
                        app.world().get::<PowerRuneMechanism>(rune).unwrap().state(),
                        &before
                    );
                }
                assert_eq!(
                    app.world()
                        .resource::<SimulationConfig>()
                        .power_rune
                        .large_success_count,
                    count
                );
            }
        }
    }

    #[test]
    fn reduced_outer_hits_neither_advance_lit_targets_nor_fail_unlit_targets_and_small_is_full_disc()
     {
        for count in [1, 2] {
            for lit in [false, true] {
                let mut app = test_app();
                manual_large_setting(&mut app, count);
                let rune = mechanism(&mut app, RuneMode::Large, Team::Red);
                let active = active_targets(&app, rune);
                let target = if lit {
                    active[0]
                } else {
                    (0..5).find(|i| !active.contains(i)).unwrap()
                };
                disc(&mut app, rune, target, Vec3::Y);
                let before = app
                    .world()
                    .get::<PowerRuneMechanism>(rune)
                    .unwrap()
                    .state()
                    .clone();
                let projectile = shot(&mut app, Vec3::Y, 0.120, 25.0, 0.0, 0.017, false);
                wait_for_round_consumption(&mut app, projectile);
                assert!(
                    app.world().resource::<HitLog>().0.is_empty(),
                    "count={count} lit={lit}"
                );
                assert_eq!(
                    app.world().get::<PowerRuneMechanism>(rune).unwrap().state(),
                    &before
                );
                // The rejected outer area remains solid and consumes this round.
                let position = app.world().get::<Position>(projectile).unwrap().0;
                assert!((0.0..=0.008503).contains(&(position.y - 0.0005)));
            }
        }
        let mut app = test_app();
        manual_large_setting(&mut app, 2);
        let rune = mechanism(&mut app, RuneMode::Small, Team::Red);
        let target = active_targets(&app, rune)[0];
        disc(&mut app, rune, target, Vec3::Y);
        shot(&mut app, Vec3::Y, 0.149, 25.0, 0.0, 0.017, false);
        advance(&mut app, 20);
        let hits = &app.world().resource::<HitLog>().0;
        assert_eq!(hits.len(), 1);
        assert_eq!(
            (hits[0].ring, hits[0].outcome),
            (1, RuneHitOutcome::PrimaryHit)
        );
        assert_eq!(
            app.world()
                .resource::<SimulationConfig>()
                .power_rune
                .large_success_count,
            2
        );
    }

    #[test]
    fn outer_rejection_does_not_debounce_an_inner_hit_and_hot_manual_setting_applies() {
        let mut app = test_app();
        manual_large_setting(&mut app, 2);
        let rune = mechanism(&mut app, RuneMode::Large, Team::Red);
        let target = active_targets(&app, rune)[0];
        disc(&mut app, rune, target, Vec3::Y);
        let outer = shot(&mut app, Vec3::Y, 0.075, 25.0, 0.0, 0.017, false);
        let outer_time = wait_for_round_consumption(&mut app, outer);
        assert!(app.world().resource::<HitLog>().0.is_empty());
        // The same radius becomes valid immediately when the user changes from
        // seven-ring practice to four-ring practice, without restarting the app.
        app.world_mut()
            .resource_mut::<SimulationConfig>()
            .power_rune
            .large_success_count = 1;
        let inner = shot(&mut app, Vec3::Y, 0.075, 25.0, 0.0, 0.017, false);
        let inner_time = wait_for_round_consumption(&mut app, inner);
        assert!(inner_time - outer_time < 0.050);
        let hits = &app.world().resource::<HitLog>().0;
        assert_eq!(hits.len(), 1);
        assert_eq!(
            (hits[0].ring, hits[0].outcome),
            (6, RuneHitOutcome::PrimaryHit)
        );
        assert_eq!(
            app.world()
                .resource::<SimulationConfig>()
                .power_rune
                .large_success_count,
            1
        );
    }

    #[test]
    fn a_rejected_outer_round_cannot_score_when_it_bounces_back_into_the_inner_area() {
        let mut app = test_app();
        manual_large_setting(&mut app, 1);
        app.world_mut()
            .resource_mut::<avian3d::dynamics::solver::SolverConfig>()
            .restitution_threshold = 0.0;
        let rune = mechanism(&mut app, RuneMode::Large, Team::Red);
        let target = active_targets(&app, rune)[0];
        let plate = disc(&mut app, rune, target, Vec3::Y);
        app.world_mut()
            .entity_mut(plate)
            .insert((Restitution::new(1.0), Friction::ZERO));
        app.world_mut().spawn(RuneSolid).insert((
            RigidBody::Static,
            Collider::cuboid(0.5, 0.01, 0.5),
            Transform::from_xyz(0.0, 0.25, 0.0),
            Restitution::new(1.0),
            Friction::ZERO,
        ));
        let before = app
            .world()
            .get::<PowerRuneMechanism>(rune)
            .unwrap()
            .state()
            .clone();
        let projectile = shot(&mut app, Vec3::Y, 0.120, 25.0, 0.0, 0.017, false);
        let origin = Vec3::new(0.120 + 2.0 * ((0.150 - 0.009) / 25.0), 0.150, 0.0);
        let mut entity = app.world_mut().entity_mut(projectile);
        entity.get_mut::<Transform>().unwrap().translation = origin;
        entity.get_mut::<Position>().unwrap().0 = origin;
        entity.get_mut::<LinearVelocity>().unwrap().0 = Vec3::new(-2.0, -25.0, 0.0);
        entity.insert((
            Restitution::new(1.0),
            Friction::ZERO,
            SpeculativeMargin(0.5),
        ));
        let mut bounced_out = false;
        let mut returned_inside = false;
        for _ in 0..120 {
            app.update();
            let position = app.world().get::<Position>(projectile).unwrap().0;
            let velocity = app.world().get::<LinearVelocity>(projectile).unwrap().0;
            bounced_out |= velocity.y > 1.0 && position.y > 0.02;
            returned_inside |= bounced_out
                && (0.0..=0.008503).contains(&(position.y - 0.0005))
                && Vec2::new(position.x, position.z).length() < 0.105;
        }
        assert!(
            bounced_out && returned_inside,
            "the physical round must bounce and revisit the reduced disc"
        );
        assert!(app.world().resource::<HitLog>().0.is_empty());
        assert_eq!(
            app.world().get::<PowerRuneMechanism>(rune).unwrap().state(),
            &before
        );
        assert!(
            app.world()
                .get::<CollisionEventsEnabled>(projectile)
                .is_none()
        );
    }

    #[test]
    fn real_disc_scoring_includes_gravity_and_lateral_force_during_the_impact_step() {
        #[derive(Resource, Default)]
        struct FirstPhysicalContact(Option<Vec3>);
        // Sample the solver's actual first arrival independently of the scoring
        // helper. Avian CCD clips displacement using the end-of-step velocity,
        // rather than the ideal acceleration arc, and the readout must follow it.
        for (gravity, force, origin, expected_ring, manual_count) in [
            (
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::new(0.0, 0.0601, 0.1905),
                6,
                None,
            ),
            (
                Vec3::NEG_Y * 9.81,
                Vec3::ZERO,
                Vec3::new(0.0, 0.0601, 0.1905),
                7,
                None,
            ),
            (
                Vec3::ZERO,
                Vec3::NEG_X * (0.0032 * 40.0),
                Vec3::new(0.0601, 0.0, 0.1905),
                7,
                Some(2),
            ),
            (
                Vec3::ZERO,
                Vec3::X * (0.0032 * 40.0),
                Vec3::new(0.0582, 0.0, 0.1905),
                6,
                Some(2),
            ),
        ] {
            let mut app = test_app();
            app.insert_resource(Gravity(gravity))
                .init_resource::<FirstPhysicalContact>()
                .add_systems(
                    PhysicsSchedule,
                    (|positions: Query<&Position, With<Projectile>>,
                      mut first: ResMut<FirstPhysicalContact>| {
                        if first.0.is_none() {
                            for position in &positions {
                                let distance = position.z - 0.0005;
                                if (0.0..=0.008503).contains(&distance) {
                                    first.0 = Some(position.0);
                                    break;
                                }
                            }
                        }
                    })
                    .after(PhysicsStepSystems::Finalize)
                    .before(PhysicsStepSystems::Last),
                );
            let mode = if let Some(count) = manual_count {
                manual_large_setting(&mut app, count);
                RuneMode::Large
            } else {
                RuneMode::Small
            };
            let rune = mechanism(&mut app, mode, Team::Red);
            let target = active_targets(&app, rune)[0];
            disc(&mut app, rune, target, Vec3::Z);
            let before = app
                .world()
                .get::<PowerRuneMechanism>(rune)
                .unwrap()
                .state()
                .clone();
            let projectile = shot(&mut app, Vec3::Z, 0.0, 25.0, 0.0, 0.017, false);
            let mut entity = app.world_mut().entity_mut(projectile);
            entity.get_mut::<Transform>().unwrap().translation = origin;
            entity.get_mut::<Position>().unwrap().0 = origin;
            entity.get_mut::<LinearVelocity>().unwrap().0 = Vec3::NEG_Z * 25.0;
            entity.insert((
                Mass(0.0032),
                ConstantForce(force),
                Friction::ZERO,
                Restitution::ZERO,
            ));
            advance(&mut app, 10);
            let first = app.world().resource::<FirstPhysicalContact>().0.unwrap();
            let measured_radius = first.truncate().length();
            // Independent CCD formula: final tangential velocity times first TOI
            // (including Avian's documented 1.0001 overshoot). The 40 m/s² case
            // reaches approximately 57.68 mm, hence an actual 58 mm readout.
            let acceleration = gravity + force / 0.0032;
            let toi = (0.19 - 0.0085) / 25.0 * 1.0001;
            let ccd_position = origin + acceleration * (1.0 / 120.0) * toi;
            let expected_radius = ccd_position.truncate().length();
            assert!(
                (measured_radius - expected_radius).abs() < 0.00001,
                "gravity={gravity} force={force} actual={measured_radius} ccd={expected_radius}"
            );
            assert_eq!(ring_from_radius(measured_radius).unwrap().0, expected_ring);
            if manual_count.is_some() {
                assert_eq!(
                    app.world()
                        .resource::<SimulationConfig>()
                        .power_rune
                        .large_success_count,
                    2
                );
                // The inward case starts outside the seven-ring area, yet its
                // real CCD arrival is valid. The outward case's free substep
                // replay stays inside, yet its real CCD arrival is invalid.
                let incoming = IncomingProjectile {
                    position: origin,
                    velocity: Vec3::NEG_Z * 25.0,
                };
                let replay = ProjectileIntegration {
                    increment: acceleration * (1.0 / 120.0 / 8.0),
                    damping_rhs: 1.0,
                    substeps: 8,
                };
                let (_, predicted_ring, _) = face_impact(
                    incoming,
                    replay,
                    0.017,
                    Vec3::Z * 0.0005,
                    Vec3::Z,
                    1.0 / 120.0,
                )
                .unwrap();
                if force.x < 0.0 {
                    assert_eq!(ring_from_radius(origin.truncate().length()).unwrap().0, 6);
                } else {
                    assert_eq!(predicted_ring, 7);
                }
            }
            let hits = &app.world().resource::<HitLog>().0;
            if manual_count.is_some() && expected_ring < 7 {
                assert!(
                    hits.is_empty(),
                    "actual six-ring CCD arrival must be rejected"
                );
                assert_eq!(
                    app.world().get::<PowerRuneMechanism>(rune).unwrap().state(),
                    &before
                );
                assert!(
                    app.world()
                        .get::<CollisionEventsEnabled>(projectile)
                        .is_none()
                );
                continue;
            }
            assert_eq!(hits.len(), 1, "gravity={gravity} force={force}");
            assert_eq!(
                (hits[0].ring, hits[0].radial_mm),
                (expected_ring, (measured_radius * 1000.0).round() as u16),
                "gravity={gravity} force={force} first={first}"
            );
        }
    }
}
