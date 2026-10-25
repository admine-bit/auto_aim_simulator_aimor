//! Acceptance against the shipped buffers, including their non-zero glTF origins.
use super::super::collision::{
    HitResult, PowerRuneCollisionPlugin, Projectile, RuneBullet, RuneColliderProxy, RuneHit,
    RuneHitSurface, RuneIndex, RuneSolid,
};
use super::super::common::RuneMode;
use super::super::rotation::PowerRuneRotation;
use super::super::rune::PowerRuneMechanism;
use super::{structural_rune_node, target_geometry};
use crate::robomaster::prelude::Team;
use avian3d::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

const ASSETS: [&str; 2] = ["POWER.glb", "maps/rmuc2027/power.glb"];

#[derive(Resource, Default)]
struct Hits(Vec<(Entity, HitResult)>);

#[derive(Clone)]
struct Primitive {
    points: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
}
struct Node {
    name: String,
    local: Transform,
    parent: Option<usize>,
    primitives: Vec<Primitive>,
}
struct Asset {
    nodes: Vec<Node>,
}

impl Asset {
    fn load(path: &str) -> Self {
        let bytes = std::fs::read(format!("{}/assets/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap();
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json: Value = serde_json::from_slice(&bytes[20..20 + json_len]).unwrap();
        let binary = &bytes[28 + json_len..];
        let accessor = |index: usize| {
            let a = &json["accessors"][index];
            let view = &json["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
            let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize
                + a["byteOffset"].as_u64().unwrap_or(0) as usize;
            (a, offset)
        };
        let mut nodes: Vec<Node> = json["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| {
                let vector = |name: &str, fallback: Vec3| {
                    node[name]
                        .as_array()
                        .map(|v| {
                            Vec3::from_array(std::array::from_fn(|i| v[i].as_f64().unwrap() as f32))
                        })
                        .unwrap_or(fallback)
                };
                let local = if let Some(matrix) = node["matrix"].as_array() {
                    Transform::from_matrix(Mat4::from_cols_array(&std::array::from_fn(|i| {
                        matrix[i].as_f64().unwrap() as f32
                    })))
                } else {
                    Transform {
                        translation: vector("translation", Vec3::ZERO),
                        rotation: node["rotation"]
                            .as_array()
                            .map(|q| {
                                Quat::from_array(std::array::from_fn(|i| {
                                    q[i].as_f64().unwrap() as f32
                                }))
                            })
                            .unwrap_or(Quat::IDENTITY),
                        scale: vector("scale", Vec3::ONE),
                    }
                };
                let primitives = node["mesh"]
                    .as_u64()
                    .map(|mesh| {
                        json["meshes"][mesh as usize]["primitives"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|primitive| {
                                let (a, offset) =
                                    accessor(primitive["attributes"]["POSITION"].as_u64().unwrap()
                                        as usize);
                                let count = a["count"].as_u64().unwrap() as usize;
                                let points = binary[offset..offset + count * 12]
                                    .chunks_exact(12)
                                    .map(|v| {
                                        Vec3::from_array(std::array::from_fn(|i| {
                                            f32::from_le_bytes(
                                                v[i * 4..i * 4 + 4].try_into().unwrap(),
                                            )
                                        }))
                                    })
                                    .collect();
                                let (a, offset) =
                                    accessor(primitive["indices"].as_u64().unwrap() as usize);
                                let count = a["count"].as_u64().unwrap() as usize;
                                let stride = match a["componentType"].as_u64().unwrap() {
                                    5123 => 2,
                                    5125 => 4,
                                    other => panic!("unsupported index type {other}"),
                                };
                                let indices: Vec<u32> = binary[offset..offset + count * stride]
                                    .chunks_exact(stride)
                                    .map(|v| {
                                        if stride == 2 {
                                            u16::from_le_bytes(v.try_into().unwrap()) as u32
                                        } else {
                                            u32::from_le_bytes(v.try_into().unwrap())
                                        }
                                    })
                                    .collect();
                                Primitive {
                                    points,
                                    triangles: indices
                                        .chunks_exact(3)
                                        .map(|t| [t[0], t[1], t[2]])
                                        .collect(),
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Node {
                    name: node["name"].as_str().unwrap_or("").to_owned(),
                    local,
                    parent: None,
                    primitives,
                }
            })
            .collect();
        for (parent, node) in json["nodes"].as_array().unwrap().iter().enumerate() {
            if let Some(children) = node["children"].as_array() {
                for child in children {
                    nodes[child.as_u64().unwrap() as usize].parent = Some(parent);
                }
            }
        }
        Self { nodes }
    }

    fn matrix(&self, index: usize) -> Mat4 {
        self.nodes[index]
            .parent
            .map_or(Mat4::IDENTITY, |p| self.matrix(p))
            * self.nodes[index].local.to_matrix()
    }

    fn target_nodes(&self) -> impl Iterator<Item = (usize, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.name.ends_with("_ACTIVATED"))
    }

    fn normal(name: &str) -> Vec3 {
        Vec3::new(1.0, 0.0, 1.0).normalize()
            * if name.starts_with("FACE_1_") {
                -1.0
            } else {
                1.0
            }
    }

    fn front(&self, index: usize) -> (Vec3, Vec3) {
        let node = &self.nodes[index];
        let matrix = self.matrix(index);
        let normal = Self::normal(&node.name);
        let local_normal = matrix.inverse().transform_vector3(normal).normalize();
        let points: Vec<Vec3> = node
            .primitives
            .iter()
            .flat_map(|p| p.points.iter().copied())
            .collect();
        let front = target_geometry(&points, local_normal).unwrap();
        (matrix.transform_point3(front), normal)
    }
}

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        AssetPlugin::default(),
        bevy::mesh::MeshPlugin,
        PhysicsPlugins::default()
            .with_collision_hooks::<crate::terrain_collision::SimulationCollisionHooks>(),
        PowerRuneCollisionPlugin,
    ))
    .insert_resource(Gravity(Vec3::ZERO))
    .insert_resource(Time::<Fixed>::from_hz(120.0))
    .insert_resource(SubstepCount(8))
    .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(1.0 / 120.0),
    ))
    .init_resource::<Hits>()
    .add_observer(|event: On<RuneHit>, mut hits: ResMut<Hits>| {
        hits.0.push((event.rune, event.result))
    });
    app.finish();
    app.cleanup();
    app
}

fn spawn_asset(
    app: &mut App,
    asset: &Asset,
    structures: bool,
) -> (Vec<Entity>, HashMap<String, Entity>) {
    let root = app.world_mut().spawn(Transform::IDENTITY).id();
    let entities: Vec<Entity> = asset
        .nodes
        .iter()
        .map(|_| app.world_mut().spawn_empty().id())
        .collect();
    for (index, node) in asset.nodes.iter().enumerate() {
        let parent = node.parent.map_or(root, |p| entities[p]);
        app.world_mut()
            .entity_mut(entities[index])
            .insert((node.local, ChildOf(parent)));
        if node.name == "FACE_1" || node.name == "FACE_2" {
            let team = if node.name == "FACE_1" {
                Team::Red
            } else {
                Team::Blue
            };
            app.world_mut().entity_mut(entities[index]).insert((
                PowerRuneMechanism::new(RuneMode::Small),
                PowerRuneRotation::for_face(false, team, node.local.rotation),
            ));
        }
        if structures && structural_rune_node(&node.name) {
            for primitive in &node.primitives {
                let mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                )
                .with_inserted_attribute(
                    Mesh::ATTRIBUTE_POSITION,
                    primitive
                        .points
                        .iter()
                        .map(|p| p.to_array())
                        .collect::<Vec<_>>(),
                )
                .with_inserted_indices(Indices::U32(
                    primitive
                        .triangles
                        .iter()
                        .flat_map(|t| t.iter().copied())
                        .collect(),
                ));
                let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
                let source = app
                    .world_mut()
                    .spawn((
                        Mesh3d(mesh),
                        Transform::IDENTITY,
                        ChildOf(entities[index]),
                        Name::new(format!("{} visual primitive", node.name)),
                    ))
                    .id();
                app.world_mut().spawn(RuneSolid).insert((
                    RigidBody::Static,
                    Collider::trimesh(primitive.points.clone(), primitive.triangles.clone()),
                    CollisionMargin(0.0),
                    Transform::from_matrix(asset.matrix(index)),
                    RuneColliderProxy {
                        source,
                        local: Transform::IDENTITY,
                    },
                ));
            }
        }
    }
    let mut targets = HashMap::new();
    for (index, node) in asset.target_nodes() {
        let (front, normal) = asset.front(index);
        let face = node.parent.unwrap();
        let face_inverse = asset.matrix(face).inverse();
        let front = face_inverse.transform_point3(front);
        let normal = face_inverse.transform_vector3(normal).normalize();
        let local = Transform::from_translation(front - normal * 0.0005)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, normal));
        let collider = app
            .world_mut()
            .spawn(RuneSolid)
            .insert((
                RigidBody::Static,
                Collider::cylinder(0.150, 0.001),
                CollisionMargin(0.0),
                RuneIndex {
                    rune: entities[face],
                    target: node
                        .name
                        .split('_')
                        .nth(3)
                        .unwrap()
                        .parse::<usize>()
                        .unwrap()
                        - 1,
                },
                RuneHitSurface {
                    center: Vec3::Y * 0.0005,
                    normal: Vec3::Y,
                },
                Transform::from_matrix(asset.matrix(face) * local.to_matrix()),
                RuneColliderProxy {
                    source: entities[face],
                    local,
                },
            ))
            .id();
        targets.insert(node.name.clone(), collider);
    }
    for _ in 0..3 {
        app.update();
    }
    (entities, targets)
}

fn ray(app: &mut App, origin: Vec3, normal: Vec3, distance: f32) -> Option<RayHitData> {
    app.world_mut()
        .run_system_once(move |spatial: SpatialQuery| {
            spatial.cast_ray(
                origin,
                Dir3::new(normal).unwrap(),
                distance,
                true,
                &SpatialQueryFilter::default(),
            )
        })
        .unwrap()
}

#[test]
fn both_real_assets_have_ten_concentric_300_mm_front_detection_discs() {
    for path in ASSETS {
        let asset = Asset::load(path);
        assert_eq!(asset.target_nodes().count(), 10, "{path}");
        for (index, node) in asset.target_nodes() {
            let (front, normal) = asset.front(index);
            let matrix = asset.matrix(index);
            let mut max_radius = 0.0_f32;
            let mut min_depth = f32::INFINITY;
            let mut max_depth = f32::NEG_INFINITY;
            for point in node.primitives.iter().flat_map(|p| &p.points) {
                let delta = matrix.transform_point3(*point) - front;
                let depth = delta.dot(normal);
                min_depth = min_depth.min(depth);
                max_depth = max_depth.max(depth);
                max_radius = max_radius.max((delta - normal * depth).length());
            }
            assert!(
                (max_radius - 0.150).abs() < 0.00001,
                "{path} {} radius={max_radius}",
                node.name
            );
            assert!(
                max_depth.abs() < 0.00001 && (min_depth + 0.001).abs() < 0.00001,
                "{path} {} depth={min_depth}..{max_depth}",
                node.name
            );
            assert!(structural_rune_node("FACE_2_BASE.578"));
            assert!(!structural_rune_node(&node.name));
        }
    }
}

#[test]
fn complete_real_asset_structure_and_production_hit_detector_report_all_ten_targets_once() {
    for path in ASSETS {
        let asset = Asset::load(path);
        for (speed, radius, tangent_ratio, expected_ring, expected_mm) in [
            (25.0, 0.0, 0.0, 10, 0),
            (60.0, 0.075, 0.0, 6, 75),
            (100.0, 0.0151, 0.0, 9, 15),
        ] {
            let mut app = app();
            let (entities, _) = spawn_asset(&mut app, &asset, true);
            let mut expected = Vec::new();
            for (index, node) in asset.target_nodes() {
                let (front, normal) = asset.front(index);
                let origin =
                    front + Vec3::Y * (radius - tangent_ratio * (0.35 - 0.0085)) + normal * 0.35;
                app.world_mut().spawn((
                    RigidBody::Dynamic,
                    Collider::sphere(0.0085),
                    SweptCcd::default(),
                    SpeculativeMargin(0.001),
                    CollisionMargin(0.0),
                    Mass(0.0032),
                    Restitution::ZERO,
                    Friction::ZERO,
                    CollisionLayers::new(2, 1),
                    Transform::from_translation(origin),
                    LinearVelocity((-normal + Vec3::Y * tangent_ratio) * speed),
                    Projectile,
                    RuneBullet { diameter_m: 0.017 },
                ));
                expected.push((
                    entities[node.parent.unwrap()],
                    node.name
                        .split('_')
                        .nth(3)
                        .unwrap()
                        .parse::<usize>()
                        .unwrap()
                        - 1,
                ));
            }
            for _ in 0..35 {
                app.update();
            }
            let hits = &app.world().resource::<Hits>().0;
            assert_eq!(
                hits.len(),
                10,
                "{path} speed={speed} radius={radius}: {hits:?}"
            );
            for (face, target) in expected {
                let matching: Vec<_> = hits
                    .iter()
                    .filter(|(rune, hit)| *rune == face && hit.target == target)
                    .collect();
                assert_eq!(matching.len(), 1, "{path} target={target} speed={speed}");
                assert_eq!(
                    (matching[0].1.ring, matching[0].1.radial_mm),
                    (expected_ring, expected_mm),
                    "{path} target={target} speed={speed}"
                );
            }
        }
    }
}

#[test]
fn real_structural_deflections_only_score_at_actual_disc_arrival_and_report_its_radius() {
    #[derive(Resource, Default)]
    struct FrameContacts(Vec<Entity>);
    #[derive(Resource, Default)]
    struct ObservedDiscHits(Vec<(Entity, usize, u16)>);

    for path in ASSETS {
        let asset = Asset::load(path);
        for (radius, tangent_ratio, expected_ring, expected_mm) in [
            (0.140, 0.0, 1, 140),
            (0.149, 0.0, 1, 149),
            // Oblique paths can meet the backing before the disc. Use a ball
            // sweep and actual arrival as the oracle, just as for rim deflection.
            (0.067, 0.6, 6, 67),
        ] {
            for structures in [true, false] {
                let mut app = app();
                app.init_resource::<ObservedDiscHits>().add_observer(
                |event: On<RuneHit>,
                 targets: Query<(&RuneIndex, &RuneHitSurface, &Position, &Rotation)>,
                 bullets: Query<(&Position, &RuneBullet), With<Projectile>>,
                 mut observed: ResMut<ObservedDiscHits>| {
                    let (_, surface, position, rotation) = targets.iter().find(|(index, _, _, _)| {
                        index.rune == event.rune && index.target == event.result.target
                    }).unwrap();
                    let normal = (rotation.0 * surface.normal).normalize();
                    let center = position.0 + rotation.0 * surface.center;
                    let radius = bullets.iter().filter_map(|(position, bullet)| {
                        let offset = position.0 - center;
                        let depth = offset.dot(normal);
                        let radius = (offset - normal * depth).length();
                        ((0.0..=bullet.diameter_m * 0.5 + 0.000003).contains(&depth)
                            && radius <= 0.150003).then_some(radius)
                    }).next().expect("a reported grazing hit must have a real ball at the disc");
                    let radial_mm = (radius * 1000.0).round() as u16;
                    assert_eq!(event.result.radial_mm, radial_mm,
                        "grazing feedback must use the actual first disc arrival after frame deflection");
                    observed.0.push((event.rune, event.result.target, radial_mm));
                },
            );
                app.init_resource::<FrameContacts>().add_observer(
                    |event: On<CollisionStart>,
                     projectiles: Query<(), With<Projectile>>,
                     targets: Query<(), With<RuneIndex>>,
                     collisions: Collisions,
                     mut contacts: ResMut<FrameContacts>| {
                        let (projectile, other) =
                            if event.body1.is_some_and(|e| projectiles.contains(e)) {
                                (event.body1.unwrap(), event.collider2)
                            } else if event.body2.is_some_and(|e| projectiles.contains(e)) {
                                (event.body2.unwrap(), event.collider1)
                            } else {
                                return;
                            };
                        if !targets.contains(other)
                            && collisions
                                .get(event.collider1, event.collider2)
                                .is_some_and(|pair| {
                                    pair.manifolds
                                        .iter()
                                        .flat_map(|m| &m.points)
                                        .any(|point| point.normal_impulse > 0.0)
                                })
                        {
                            contacts.0.push(projectile);
                        }
                    },
                );
                let (entities, target_colliders) = spawn_asset(&mut app, &asset, structures);
                let bullets: Vec<(Entity, Entity, usize, bool)> = asset
                    .target_nodes()
                    .map(|(index, node)| {
                        let (front, normal) = asset.front(index);
                        let origin = front
                            + Vec3::Y * (radius - tangent_ratio * (0.35 - 0.0085))
                            + normal * 0.35;
                        let direction = (-normal + Vec3::Y * tangent_ratio).normalize();
                        let disc = target_colliders[&node.name];
                        // A zero-distance sphere cast is independent of speculative
                        // contact settings and proves genuine structural occlusion.
                        let blocked = app
                            .world_mut()
                            .run_system_once(move |spatial: SpatialQuery| {
                                spatial
                                    .cast_shape(
                                        &Collider::sphere(0.0085),
                                        origin,
                                        Quat::IDENTITY,
                                        Dir3::new(direction).unwrap(),
                                        &ShapeCastConfig::from_max_distance(0.6)
                                            .with_target_distance(0.0),
                                        &SpatialQueryFilter::default(),
                                    )
                                    .is_some_and(|hit| hit.entity != disc)
                            })
                            .unwrap();
                        let bullet = app
                            .world_mut()
                            .spawn((
                                RigidBody::Dynamic,
                                Collider::sphere(0.0085),
                                SweptCcd::default(),
                                SpeculativeMargin(0.001),
                                CollisionMargin(0.0),
                                Mass(0.0032),
                                Restitution::ZERO,
                                Friction::ZERO,
                                CollisionLayers::new(2, 1),
                                Transform::from_translation(origin),
                                LinearVelocity((-normal + Vec3::Y * tangent_ratio) * 25.0),
                                Projectile,
                                RuneBullet { diameter_m: 0.017 },
                            ))
                            .id();
                        let target = node
                            .name
                            .split('_')
                            .nth(3)
                            .unwrap()
                            .parse::<usize>()
                            .unwrap()
                            - 1;
                        (bullet, entities[node.parent.unwrap()], target, blocked)
                    })
                    .collect();
                for _ in 0..35 {
                    app.update();
                }
                let hits = &app.world().resource::<Hits>().0;
                if structures {
                    // Real backing details and rims can contact a 17 mm ball before
                    // its centre reaches the plate, even when a point ray is clear.
                    // Deflections onto the plate score at their actual arrival radius.
                    if tangent_ratio == 0.0 {
                        assert!(
                            hits.len() < 10,
                            "{path} the frame must still block some grazing shots"
                        );
                    }
                    let observed = &app.world().resource::<ObservedDiscHits>().0;
                    assert_eq!(hits.len(), observed.len());
                    for (rune, hit) in hits {
                        assert!(
                            observed.iter().any(|&(face, target, mm)| {
                                face == *rune && target == hit.target && mm == hit.radial_mm
                            }),
                            "{path} a frame-only contact was reported as a disc hit"
                        );
                    }
                    let contacts = &app.world().resource::<FrameContacts>().0;
                    for (bullet, face, target, blocked) in bullets {
                        if !hits
                            .iter()
                            .any(|(rune, hit)| *rune == face && hit.target == target)
                        {
                            assert!(
                                blocked || contacts.contains(&bullet),
                                "{path} radius={radius} target={target}: an unobstructed ball was missed"
                            );
                        }
                    }
                } else {
                    assert_eq!(
                        hits.len(),
                        10,
                        "{path} unobstructed radius={radius} disc impacts"
                    );
                    assert!(
                        hits.iter()
                            .all(|(_, hit)| (hit.ring, hit.radial_mm)
                                == (expected_ring, expected_mm)),
                        "{path} {hits:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn real_asset_discs_block_fast_bullets_from_both_faces_and_back_sides() {
    for path in ASSETS {
        let asset = Asset::load(path);
        for sign in [-1.0, 1.0] {
            for speed in [25.0, 60.0, 100.0] {
                let mut app = app();
                spawn_asset(&mut app, &asset, false);
                let bullets: Vec<_> = asset
                    .target_nodes()
                    .map(|(index, node)| {
                        let (front, normal) = asset.front(index);
                        let origin = front + normal * sign * 0.35;
                        let entity = app
                            .world_mut()
                            .spawn((
                                RigidBody::Dynamic,
                                Collider::sphere(0.0085),
                                SweptCcd::default(),
                                SpeculativeMargin(0.001),
                                Mass(0.0032),
                                Restitution::ZERO,
                                Friction::ZERO,
                                CollisionLayers::new(2, 1),
                                Transform::from_translation(origin),
                                LinearVelocity(-normal * sign * speed),
                            ))
                            .id();
                        (entity, front, normal, node.name.clone())
                    })
                    .collect();
                for _ in 0..35 {
                    app.update();
                }
                for (entity, front, normal, name) in bullets {
                    let position = app.world().get::<Position>(entity).unwrap().0;
                    let signed = (position - front).dot(normal) * sign;
                    assert!(
                        signed >= -0.002,
                        "{path} {name} side={sign} speed={speed} passed disc: {signed}"
                    );
                }
            }
        }
    }
}

#[test]
fn real_asset_colliders_follow_each_rotated_face_without_leaving_old_ghost_discs() {
    for path in ASSETS {
        let asset = Asset::load(path);
        let mut app = app();
        let (entities, targets) = spawn_asset(&mut app, &asset, false);
        for (index, node) in asset.nodes.iter().enumerate() {
            if node.name == "FACE_1" || node.name == "FACE_2" {
                let outward = Asset::normal(&format!("{}_", node.name));
                let axis = asset
                    .matrix(index)
                    .inverse()
                    .transform_vector3(outward)
                    .normalize();
                app.world_mut()
                    .get_mut::<Transform>(entities[index])
                    .unwrap()
                    .rotate_local_axis(Dir3::new(axis).unwrap(), 0.417);
            }
        }
        for _ in 0..3 {
            app.update();
        }
        for (index, node) in asset.target_nodes() {
            let disc = targets[&node.name];
            let global = *app.world().get::<GlobalTransform>(disc).unwrap();
            let center = global.transform_point(Vec3::Y * 0.0005);
            let normal = global.rotation() * Vec3::Y;
            let hit = ray(&mut app, center + normal * 0.1, -normal, 0.2).unwrap();
            assert_eq!(hit.entity, disc, "{path} {} moved disc", node.name);
            let (old_front, old_normal) = asset.front(index);
            assert!((center - old_front).length() > 0.20);
            assert!(
                ray(&mut app, old_front + old_normal * 0.1, -old_normal, 0.2).is_none(),
                "{path} {} ghost at original pose",
                node.name
            );
        }
    }
}

#[test]
fn real_asset_primitives_and_hit_discs_remain_rigid_during_one_thousand_rotating_frames() {
    #[derive(Clone)]
    struct Tracked {
        entity: Entity,
        source: Entity,
        source_is_primitive: bool,
        face: Entity,
        label: String,
        local: Transform,
        proxy_local: Transform,
        relative_to_face: Mat4,
    }
    #[derive(Resource, Default)]
    struct Rigidity {
        tracked: Vec<Tracked>,
        physics_steps: usize,
        max_position_error: f32,
        max_axis_error: f32,
        worst_physics: String,
    }
    fn axis_error(a: Quat, b: Quat) -> f32 {
        [Vec3::X, Vec3::Y, Vec3::Z]
            .into_iter()
            .map(|axis| (a * axis).distance(b * axis))
            .fold(0.0, f32::max)
    }

    for path in ASSETS {
        for mode in [RuneMode::Small, RuneMode::Large] {
            let asset = Asset::load(path);
            let mut app = app();
            let (entities, _) = spawn_asset(&mut app, &asset, true);
            let faces: HashMap<Entity, &str> = asset
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.name == "FACE_1" || n.name == "FACE_2")
                .map(|(i, n)| (entities[i], n.name.as_str()))
                .collect();
            // Exercise the existing motion implementation with the fixed, opposite
            // apparent directions used by the two fronts. No motion code is copied.
            for (&face, &name) in &faces {
                let team = if name == "FACE_1" {
                    Team::Red
                } else {
                    Team::Blue
                };
                let initial = app.world().get::<Transform>(face).unwrap().rotation;
                let mut rotation = PowerRuneRotation::for_face(team == Team::Blue, team, initial);
                if mode == RuneMode::Large {
                    use rand::SeedableRng;
                    let mut rng =
                        rand::rngs::StdRng::seed_from_u64(if team == Team::Red { 42 } else { 43 });
                    rotation.sync_activation(mode, true, &mut rng);
                }
                app.world_mut().entity_mut(face).insert(rotation);
            }
            let candidates: Vec<Entity> = {
                let mut q = app.world_mut().query_filtered::<Entity, With<Collider>>();
                q.iter(app.world()).collect()
            };
            let mut tracked = Vec::new();
            for entity in candidates {
                let proxy = app.world().get::<RuneColliderProxy>(entity).unwrap();
                let source = proxy.source;
                let proxy_local = proxy.local;
                assert!(
                    app.world().get::<ChildOf>(entity).is_none(),
                    "a physics proxy must be outside the visual tree"
                );
                assert!(
                    app.world().get::<RigidBody>(source).is_none()
                        && app.world().get::<Collider>(source).is_none(),
                    "a visual source must not own physics components"
                );
                let mut ancestor = source;
                let mut label = format!("hit disc {entity:?}");
                let mut marker = false;
                let face = loop {
                    if let Some(&name) = faces.get(&ancestor) {
                        label = format!("{name}/{label}");
                        break Some(ancestor);
                    }
                    if let Some(index) = entities.iter().position(|&e| e == ancestor) {
                        let name = &asset.nodes[index].name;
                        marker |= name.contains("_R_");
                        label = format!("{name}/{label}");
                    }
                    let Some(parent) = app.world().get::<ChildOf>(ancestor) else {
                        break None;
                    };
                    ancestor = parent.parent();
                };
                // R is fixed separately by construction and is not a rotating part.
                let Some(face) = face.filter(|_| !marker) else {
                    continue;
                };
                let local = *app.world().get::<Transform>(source).unwrap();
                let face_world = app
                    .world()
                    .get::<GlobalTransform>(face)
                    .unwrap()
                    .to_matrix();
                let primitive_world = app
                    .world()
                    .get::<GlobalTransform>(source)
                    .unwrap()
                    .to_matrix()
                    * proxy_local.to_matrix();
                tracked.push(Tracked {
                    entity,
                    source,
                    source_is_primitive: app.world().get::<Mesh3d>(source).is_some(),
                    face,
                    label,
                    local,
                    proxy_local,
                    relative_to_face: face_world.inverse() * primitive_world,
                });
            }
            assert!(
                tracked.len() >= 22,
                "{path}: all ten discs, arms and target rims must be tracked"
            );
            app.insert_resource(Rigidity {
                tracked: tracked.clone(),
                ..default()
            })
            .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                Duration::from_secs_f64(1.0 / 100.0),
            ))
            .add_systems(
                Update,
                move |time: Res<Time>,
                      mut faces: Query<(&mut PowerRuneRotation, &mut Transform)>| {
                    for (mut rotation, mut transform) in &mut faces {
                        rotation.rotate(mode, &mut transform, time.delta_secs());
                    }
                },
            )
            .add_systems(
                PhysicsSchedule,
                (|mut rigidity: ResMut<Rigidity>,
                  faces: Query<&GlobalTransform>,
                  poses: Query<(&Position, &Rotation)>| {
                    rigidity.physics_steps += 1;
                    for part in rigidity.tracked.clone() {
                        let expected =
                            faces.get(part.face).unwrap().to_matrix() * part.relative_to_face;
                        let (_, expected_rotation, expected_position) =
                            expected.to_scale_rotation_translation();
                        let (position, rotation) = poses.get(part.entity).unwrap();
                        let position_error = position.0.distance(expected_position);
                        if position_error > rigidity.max_position_error {
                            rigidity.max_position_error = position_error;
                            rigidity.worst_physics = part.label.clone();
                        }
                        rigidity.max_axis_error = rigidity
                            .max_axis_error
                            .max(axis_error(rotation.0, expected_rotation));
                    }
                })
                .after(PhysicsStepSystems::Finalize)
                .before(PhysicsStepSystems::Last),
            );

            let mut max_local_translation = 0.0_f32;
            let mut max_local_axes = 0.0_f32;
            let mut max_local_scale = 0.0_f32;
            let mut worst_local = String::new();
            for frame in 0..1000 {
                app.update();
                for part in &tracked {
                    let proxy = app.world().get::<RuneColliderProxy>(part.entity).unwrap();
                    let mut check_local = |current: &Transform, original: &Transform| {
                        let translation_error = current.translation.distance(original.translation);
                        if translation_error > max_local_translation {
                            max_local_translation = translation_error;
                            worst_local = format!("frame={frame} {}", part.label);
                        }
                        max_local_axes =
                            max_local_axes.max(axis_error(current.rotation, original.rotation));
                        max_local_scale =
                            max_local_scale.max(current.scale.distance(original.scale));
                    };
                    check_local(&proxy.local, &part.proxy_local);
                    if part.source_is_primitive {
                        check_local(
                            app.world().get::<Transform>(part.source).unwrap(),
                            &part.local,
                        );
                    }
                }
            }
            let rigidity = app.world().resource::<Rigidity>();
            eprintln!(
                "RUNE_RIGIDITY {path} {mode:?}: frames=1000 fixed_steps={} local_translation={}m local_axes={} local_scale={} physics_position={}m physics_axes={} local_worst={} physics_worst={}",
                rigidity.physics_steps,
                max_local_translation,
                max_local_axes,
                max_local_scale,
                rigidity.max_position_error,
                rigidity.max_axis_error,
                worst_local,
                rigidity.worst_physics
            );
            assert!(
                (1190..=1210).contains(&rigidity.physics_steps),
                "{path}: expected 120 Hz physics during 100 Hz updates"
            );
            assert!(
                max_local_translation < 0.00002
                    && max_local_axes < 0.00002
                    && max_local_scale < 0.00002,
                "{path}: rotating visual primitive/disc local transforms drifted: {worst_local}"
            );
            assert!(
                rigidity.max_position_error < 0.00003 && rigidity.max_axis_error < 0.00003,
                "{path}: collider lost its rigid pose relative to the rotating face: {}",
                rigidity.worst_physics
            );
        }
    }
}

#[test]
fn real_structural_meshes_preserve_frames_and_open_arm_gaps() {
    for path in ASSETS {
        let asset = Asset::load(path);
        let mut app = app();
        let (_, targets) = spawn_asset(&mut app, &asset, true);
        for (index, node) in asset.target_nodes() {
            let (front, normal) = asset.front(index);
            let hit = ray(&mut app, front + normal * 0.1, -normal, 0.2).unwrap();
            assert_eq!(
                hit.entity, targets[&node.name],
                "{path} {} scoring centre occluded by a structural mesh",
                node.name
            );
        }
        // Pick actual triangle interiors, independently of the replacement detection discs.
        for name in [
            "BASE",
            "FACE_1_BASE",
            "FACE_2_BASE.578",
            "FACE_1_TARGET_1_PADDING",
            "FACE_2_TARGET_1_PADDING",
        ] {
            let (index, node) = asset
                .nodes
                .iter()
                .enumerate()
                .find(|(_, n)| n.name == name)
                .unwrap();
            let matrix = asset.matrix(index);
            let normal = Asset::normal(&node.name);
            let centroid = node
                .primitives
                .iter()
                .flat_map(|p| {
                    p.triangles.iter().map(move |t| {
                        let a = matrix.transform_point3(p.points[t[0] as usize]);
                        let b = matrix.transform_point3(p.points[t[1] as usize]);
                        let c = matrix.transform_point3(p.points[t[2] as usize]);
                        (((b - a).cross(c - a)).dot(normal).abs(), (a + b + c) / 3.0)
                    })
                })
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .unwrap()
                .1;
            assert!(
                ray(&mut app, centroid + normal * 0.08, -normal, 0.16).is_some(),
                "{path} {name} frame missing"
            );
        }
        let (face_index, _) = asset
            .nodes
            .iter()
            .enumerate()
            .find(|(_, n)| n.name == "FACE_1")
            .unwrap();
        let pivot = asset.matrix(face_index).transform_point3(Vec3::ZERO);
        let normal = Asset::normal("FACE_1_");
        let tangent = Vec3::new(1.0, 0.0, -1.0).normalize();
        let clear = (0..180)
            .filter(|angle| {
                let angle = *angle as f32 * std::f32::consts::TAU / 180.0;
                let point = pivot + (tangent * angle.cos() + Vec3::Y * angle.sin()) * 0.45;
                ray(&mut app, point + normal * 0.5, -normal, 1.0).is_none()
            })
            .count();
        assert!(
            clear >= 15 && clear <= 165,
            "{path} expected open and obstructed arm paths, clear={clear}/180"
        );
    }
}
