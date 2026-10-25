//! Keep the road above each tunnel driveable without blocking vehicles inside it.
use avian3d::prelude::*;
use bevy::{ecs::system::SystemParam, prelude::*};

use crate::components::Infantry;

// Covers chassis tilt on the shallow road slope and the solver's contact tolerance.
// This is much smaller than the height between the tunnel floor and its road surface.
const DECK_SUPPORT_TOLERANCE: f32 = 0.03;

#[derive(Component)]
#[require(ActiveCollisionHooks::MODIFY_CONTACTS)]
pub struct TunnelDeck;

/// The glTF node owns mesh children, while Avian attaches colliders to those children.
/// Call before collider construction; only the two named deck subtrees are marked.
/// Avian 0.7 caches MODIFY_CONTACTS when adding the collider to its broad-phase tree;
/// adding ActiveCollisionHooks afterwards does not refresh that cached flag.
pub fn tag_tunnel_decks(world: &mut World, root: Entity) -> usize {
    let mut pending = vec![(root, false)];
    let mut marked = 0;
    while let Some((entity, inherited)) = pending.pop() {
        let is_deck = inherited
            || world
                .get::<Name>(entity)
                .is_some_and(|name| name.as_str().starts_with("tunnel_deck_"));
        if let Some(children) = world.get::<Children>(entity) {
            pending.extend(children.iter().map(|child| (child, is_deck)));
        }
        if is_deck && world.get::<Mesh3d>(entity).is_some() {
            world.entity_mut(entity).insert(TunnelDeck);
            marked += 1;
        }
    }
    marked
}

#[derive(SystemParam)]
pub struct TerrainCollisionHooks<'w, 's> {
    decks: Query<'w, 's, (), With<TunnelDeck>>,
    vehicles:
        Query<'w, 's, (&'static Collider, &'static Position, &'static Rotation), With<Infantry>>,
    body_poses: Query<
        'w,
        's,
        (
            &'static Position,
            &'static Rotation,
            &'static ComputedCenterOfMass,
        ),
    >,
}

impl CollisionHooks for TerrainCollisionHooks<'_, '_> {
    fn modify_contacts(&self, contacts: &mut ContactPair, _commands: &mut Commands) -> bool {
        let (deck_is_first, vehicle_body, deck_body) = if self.decks.contains(contacts.collider1) {
            (true, contacts.body2, contacts.body1)
        } else if self.decks.contains(contacts.collider2) {
            (false, contacts.body1, contacts.body2)
        } else {
            return true;
        };
        let Some(vehicle_body) = vehicle_body else {
            return true;
        };
        let Ok((chassis, position, rotation)) = self.vehicles.get(vehicle_body) else {
            return true;
        };
        let Some((deck_position, deck_rotation, deck_com)) =
            deck_body.and_then(|body| self.body_poses.get(body).ok())
        else {
            return true;
        };
        // Use the rigid-body root's chassis, including for armor colliders above the roof.
        // The broad-phase AABB is expanded for predicted movement, so calculate the actual one.
        let deck_world_com = deck_position.0 + deck_rotation.0 * deck_com.0;
        contacts.manifolds.retain_mut(|manifold| {
            let upward_normal = if deck_is_first {
                manifold.normal
            } else {
                -manifold.normal
            };
            if upward_normal.y < 0.5 {
                return false;
            }
            // Measure support along the surface normal, so the deck's steep end bevels
            // do not compare their uphill contact to the vehicle's downhill world-Y minimum.
            let to_surface = Quat::from_rotation_arc(upward_normal, Vec3::Y);
            let bottom = chassis
                .aabb(to_surface * position.0, to_surface * rotation.0)
                .min
                .y;
            manifold.points.retain(|point| {
                // Anchors are world-space offsets from the rigid body's center of mass.
                // Unlike a deck-wide min/max height, this also follows its sloped surface.
                let deck_point = deck_world_com
                    + if deck_is_first {
                        point.anchor1
                    } else {
                        point.anchor2
                    };
                bottom >= deck_point.dot(upward_normal) - DECK_SUPPORT_TOLERANCE
            });
            !manifold.points.is_empty()
        });
        !contacts.manifolds.is_empty()
    }
}

/// Compose the existing tunnel support rules with precise rune projectile contacts.
/// Keep both filters active for the default simulator physics pipeline.
#[derive(SystemParam)]
pub struct SimulationCollisionHooks<'w, 's> {
    terrain: TerrainCollisionHooks<'w, 's>,
    rune: crate::robomaster::prelude::RuneContactHooks<'w, 's>,
}

impl CollisionHooks for SimulationCollisionHooks<'_, '_> {
    fn modify_contacts(&self, contacts: &mut ContactPair, commands: &mut Commands) -> bool {
        self.terrain.modify_contacts(contacts, commands)
            && self.rune.modify_contacts(contacts, commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robomaster::prelude::{INFANTRY_THREE_CONFIG, Team};
    use std::time::Duration;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            AssetPlugin::default(),
            bevy::mesh::MeshPlugin,
            PhysicsPlugins::default().with_collision_hooks::<SimulationCollisionHooks>(),
        ))
        .insert_resource(Time::<Fixed>::from_hz(120.0))
        .insert_resource(SubstepCount(8))
        .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f64(1.0 / 120.0),
        ));
        app.finish();
        app.cleanup();
        app
    }

    fn deck(app: &mut App, height: f32, marked: bool) -> Entity {
        let collider = Collider::trimesh(
            vec![
                Vec3::new(-0.18, height, -1.0),
                Vec3::new(0.18, height, -1.0),
                Vec3::new(0.18, height, 1.0),
                Vec3::new(-0.18, height, 1.0),
            ],
            vec![[0, 2, 1], [0, 3, 2]],
        );
        let mut entity = app.world_mut().spawn_empty();
        if marked {
            entity.insert(TunnelDeck);
        }
        entity.insert((
            RigidBody::Static,
            collider,
            Friction::ZERO,
            Transform::IDENTITY,
        ));
        entity.id()
    }

    fn car(app: &mut App, position: Vec3, with_armor: bool) -> Entity {
        let root = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Infantry::new(Team::Red, INFANTRY_THREE_CONFIG),
                Collider::compound(vec![(
                    Vec3::new(0.0, -0.115649, 0.0),
                    Quat::IDENTITY,
                    Collider::cylinder(0.2593615, 0.231298),
                )]),
                CollisionMargin(0.005),
                Mass(15.0),
                Friction::ZERO,
                LockedAxes::ROTATION_LOCKED,
                Transform::from_translation(position),
            ))
            .id();
        if with_armor {
            app.world_mut().spawn((
                ChildOf(root),
                Collider::cuboid(0.15, 0.2, 0.3),
                Friction::ZERO,
                Transform::from_xyz(0.0, 0.15, 0.0),
            ));
        }
        root
    }

    fn run(app: &mut App, frames: usize) {
        for _ in 0..frames {
            app.update();
        }
    }

    #[test]
    fn tunnel_ignores_armor_above_roof_but_real_wall_still_stops_vehicle() {
        for (marked, wall) in [(false, false), (true, false), (true, true)] {
            let mut app = app();
            app.world_mut().spawn((
                RigidBody::Static,
                Collider::cuboid(10.0, 0.2, 4.0),
                Friction::ZERO,
                Transform::from_xyz(0.0, 0.0, 0.0),
            ));
            deck(&mut app, 0.49, marked);
            if wall {
                app.world_mut().spawn((
                    RigidBody::Static,
                    Collider::cuboid(0.1, 2.0, 2.0),
                    Friction::ZERO,
                    Transform::from_xyz(1.0, 0.5, 0.0),
                ));
            }
            let vehicle = car(&mut app, Vec3::new(-1.0, 0.34, 0.0), true);
            run(&mut app, 120);
            app.world_mut()
                .entity_mut(vehicle)
                .insert(LinearVelocity(Vec3::X));
            run(&mut app, 360);
            let position = app.world().get::<Position>(vehicle).unwrap().0;
            assert!(
                (position.y - 0.3363).abs() < 0.015,
                "marked={marked} wall={wall}: {position}"
            );
            if !marked {
                assert!(
                    position.x < 0.0,
                    "unfiltered armor should reproduce the obstruction: {position}"
                );
            } else if wall {
                assert!(
                    position.x > 0.3 && position.x < 0.8,
                    "real wall must still stop the car: {position}"
                );
            } else {
                assert!(
                    position.x > 1.7,
                    "whole vehicle must cross the tunnel: {position}"
                );
            }
        }
    }

    #[test]
    fn tunnel_deck_supports_landing_from_above_in_both_pair_orders() {
        for vehicle_first in [false, true] {
            let mut app = app();
            let vehicle = if vehicle_first {
                let car = car(&mut app, Vec3::new(0.0, 1.0, 0.0), true);
                deck(&mut app, 0.49, true);
                car
            } else {
                deck(&mut app, 0.49, true);
                car(&mut app, Vec3::new(0.0, 1.0, 0.0), true)
            };
            run(&mut app, 360);
            let position = app.world().get::<Position>(vehicle).unwrap().0;
            assert!(
                (position.y - 0.7263).abs() < 0.015,
                "vehicle_first={vehicle_first}: {position}"
            );
        }
    }

    #[test]
    fn sloped_deck_supports_vehicle_with_shared_transformed_ground() {
        let mut app = app();
        let rotation = Quat::from_rotation_z(30.0_f32.to_radians());
        let translation = Vec3::new(2.0, 0.3, -1.0);
        let normal = rotation * Vec3::Y;
        let center = translation + rotation * Vec3::new(0.0, 0.49, 0.0);
        let ground = app
            .world_mut()
            .spawn((
                RigidBody::Static,
                Friction::new(0.8),
                Transform::from_rotation(rotation).with_translation(translation),
            ))
            .id();
        let deck = app.world_mut().spawn((TunnelDeck, ChildOf(ground))).id();
        app.world_mut().entity_mut(deck).insert((
            Collider::trimesh(
                vec![
                    Vec3::new(-3.0, 0.49, -1.0),
                    Vec3::new(3.0, 0.49, -1.0),
                    Vec3::new(3.0, 0.49, 1.0),
                    Vec3::new(-3.0, 0.49, 1.0),
                ],
                vec![[0, 2, 1], [0, 3, 2]],
            ),
            Transform::IDENTITY,
        ));
        // Another terrain collider shifts the shared body's COM away from this deck.
        app.world_mut().spawn((
            ChildOf(ground),
            Collider::cuboid(1.0, 1.0, 1.0),
            Transform::from_xyz(8.0, -2.0, 0.0),
        ));
        let vehicle = car(&mut app, center + normal * 0.6, true);
        app.world_mut().entity_mut(vehicle).insert((
            Transform::from_rotation(rotation).with_translation(center + normal * 0.6),
            Friction::new(0.8),
        ));
        run(&mut app, 360);
        let position = app.world().get::<Position>(vehicle).unwrap().0;
        let clearance = (position - center).dot(normal);
        assert!(
            (clearance - 0.2363).abs() < 0.02,
            "sloped deck lost support: position={position}, clearance={clearance}"
        );
        assert!(
            app.world()
                .get::<ComputedCenterOfMass>(ground)
                .unwrap()
                .0
                .length()
                > 1.0
        );
    }

    #[test]
    fn deck_tagging_marks_only_colliders_under_named_node() {
        let mut world = World::new();
        let root = world.spawn_empty().id();
        let named = world
            .spawn((Name::new("tunnel_deck_positive_z"), ChildOf(root)))
            .id();
        let deck = world.spawn((Mesh3d::default(), ChildOf(named))).id();
        let wall = world.spawn((Mesh3d::default(), ChildOf(root))).id();
        assert_eq!(tag_tunnel_decks(&mut world, root), 1);
        assert!(world.get::<TunnelDeck>(deck).is_some());
        assert_eq!(
            world.get::<ActiveCollisionHooks>(deck),
            Some(&ActiveCollisionHooks::MODIFY_CONTACTS)
        );
        assert!(world.get::<TunnelDeck>(wall).is_none());
    }
}
