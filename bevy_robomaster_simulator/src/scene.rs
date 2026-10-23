//! Scene load order.
//!
//! glTF worlds become ready on whatever frame their asset happens to finish loading, so spawning
//! everything at once lets `setup_vehicle` turn a robot into a dynamic rigid body before the
//! environment has any colliders — and the robot falls through the floor. Loading is therefore one
//! linear task: each `spawn` resolves when that world's instance is ready, and the robots are
//! spawned last.
//!
//! Per-world setup (naming, collision layers, gimbal wiring) stays in the `WorldInstanceReady`
//! observers, because worlds spawned at runtime — projectiles, UAVs — need it too. What used to be
//! implicit ordering *between* those observers is now explicit here.

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::world_serialization::{InstanceId, WorldAssetRoot, WorldInstance};
use std::collections::HashMap;
use std::path::Path;

use crate::components::{
    ActiveSlapper, Controlled, GameLayer, GroundRoot, Infantry, PreciousCollision, SlapperInfantry,
};
use crate::config::{SceneMap, SimulationConfig};
use crate::robomaster::power_rune::construct::setup_power_rune;
use crate::robomaster::prelude::{
    HERO_ROBOT_CONFIG, INFANTRY_THREE_CONFIG, PowerRuneRoot, Team, TechCoreRoot,
};
use crate::robomaster::tech_core::construct::setup_tech_core;
use crate::setup::{
    ScanOutpost, setup_collision, setup_dart_launch, setup_outposts, setup_vehicle,
};
use crate::util::async_world::{AsyncWorld, AsyncWorldTask, drive_async_world};

/// Resolves once Avian has built every collider `setup_collision` queued under `root`.
///
/// `setup_collision` puts a [`ColliderConstructorHierarchy`] on the named descendants of a world as
/// soon as its instance is ready, so by the time this runs the queue for `root` is complete.
/// Selecting and observing happen in one world job, so constructors finishing early are simply not
/// selected rather than awaited forever.
async fn colliders_ready(w: &AsyncWorld, root: Entity) {
    w.observe_all::<ColliderConstructorHierarchyReady, _>(move |world| {
        let mut queued = vec![root];
        collect_descendants(world, root, &mut queued);
        queued.retain(|entity| world.get::<ColliderConstructorHierarchy>(*entity).is_some());
        queued
    })
    .await;
}

/// The instance id a loaded world spawned under, needed to enumerate its entities by name.
async fn instance_of(w: &AsyncWorld, root: Entity) -> InstanceId {
    w.with_world(move |world| {
        **world
            .get::<WorldInstance>(root)
            .expect("world instance is ready, so its id must exist")
    })
    .await
}

fn collect_descendants(world: &World, entity: Entity, out: &mut Vec<Entity>) {
    let Some(children) = world.get::<Children>(entity) else {
        return;
    };
    let children: Vec<Entity> = children.iter().collect();
    for child in children {
        out.push(child);
        collect_descendants(world, child, out);
    }
}

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AsyncWorldTask::new(load_scene))
            .add_systems(
                PreUpdate,
                drive_async_world.run_if(resource_exists::<AsyncWorldTask>),
            );
    }
}

async fn load_scene(w: AsyncWorld) {
    let assets = w
        .with_world(|world| world.resource::<AssetServer>().clone())
        .await;
    let player_start = w
        .with_world(|world| {
            world
                .resource::<SimulationConfig>()
                .scene
                .player_start
                .clone()
        })
        .await;
    let map = w
        .with_world(|world| world.resource::<SimulationConfig>().scene.map)
        .await;
    let scene =
        |path: &'static str| WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset(path)));

    let layers = GameLayer::environment_collision_layers();
    let trimesh = || {
        ColliderConstructorHierarchy::new(ColliderConstructor::TrimeshFromMeshWithConfig(
            TrimeshFlags::all(),
        ))
        .with_default_layers(layers)
    };
    let static_trimesh = || {
        (
            trimesh(),
            layers,
            Visibility::Visible,
            Some(RigidBody::Static),
        )
    };

    // Environment first. Every robot below lands on what these build.
    if map == SceneMap::Rmuc2027 {
        // CAD is already converted from Z-up metres to Y-up metres. Do not scale the terrain.
        let report: serde_json::Value =
            serde_json::from_str(include_str!("../assets/maps/rmuc2027/import_report.json"))
                .expect("generated RMUC2027 import report must be valid");
        for file in report["render_files"].as_array().expect("render files") {
            let path = format!("maps/rmuc2027/{}", file.as_str().expect("render file name"));
            w.spawn(
                WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset(path))),
                GroundRoot,
            )
            .await;
        }
        let ground = w
            .spawn(
                scene("maps/rmuc2027/collision.glb"),
                (
                    GroundRoot,
                    RigidBody::Static,
                    Friction::new(0.5),
                    Visibility::Hidden,
                ),
            )
            .await;
        w.with_world(move |world| {
            world.entity_mut(ground).insert(
                ColliderConstructorHierarchy::new(ColliderConstructor::TrimeshFromMesh)
                    .with_default_layers(layers),
            );
        })
        .await;
        colliders_ready(&w, ground).await;
        let markers = w
            .spawn(scene("maps/rmuc2027/markers.glb"), GroundRoot)
            .await;
        w.run(setup_dart_launch, markers).await;
        info!("RMUC2027 CAD terrain and concave triangle-mesh collisions ready");
    } else {
        let ground = w
            .spawn(scene("GROUND.glb"), (GroundRoot, Friction::new(0.5)))
            .await;
        w.run(setup_dart_launch, ground).await;
        w.run(
            setup_collision,
            (
                ground,
                PreciousCollision(HashMap::from([(
                    "GROUND_DENSE".to_string(),
                    static_trimesh(),
                )])),
            ),
        )
        .await;
        colliders_ready(&w, ground).await;
    }

    let outpost = w
        .spawn(
            scene(if map == SceneMap::Rmuc2027 {
                "maps/rmuc2027/outposts.glb"
            } else {
                "OUTPOST.glb"
            }),
            (RigidBody::Static, ScanOutpost),
        )
        .await;
    w.run(setup_outposts, outpost).await;

    let tech_core = w
        .spawn(
            scene(if map == SceneMap::Rmuc2027 {
                "maps/rmuc2027/tech_core.glb"
            } else {
                "TECH_CORE.glb"
            }),
            TechCoreRoot,
        )
        .await;
    w.run(
        setup_tech_core,
        (tech_core, instance_of(&w, tech_core).await),
    )
    .await;
    if map == SceneMap::Rmuc2027 {
        w.with_world(move |world| {
            world.entity_mut(tech_core).insert((
                RigidBody::Static,
                ColliderConstructorHierarchy::new(ColliderConstructor::TrimeshFromMesh)
                    .with_default_layers(layers),
            ));
        })
        .await;
    } else {
        w.run(
            setup_collision,
            (
                tech_core,
                PreciousCollision(HashMap::from([("GROUND".to_string(), static_trimesh())])),
            ),
        )
        .await;
    }
    colliders_ready(&w, tech_core).await;

    let power_rune = w
        .spawn(
            scene(if map == SceneMap::Rmuc2027 {
                "maps/rmuc2027/power.glb"
            } else {
                "POWER.glb"
            }),
            (
                RigidBody::Static,
                CollisionMargin(0.001),
                Restitution::ZERO,
                PowerRuneRoot,
            ),
        )
        .await;
    w.run(
        setup_power_rune,
        (power_rune, instance_of(&w, power_rune).await),
    )
    .await;
    w.run(
        setup_collision,
        (power_rune, PreciousCollision(power_rune_collision(layers))),
    )
    .await;
    colliders_ready(&w, power_rune).await;

    // User-configured static scene models are loaded before robots so their colliders are ready
    // before any dynamic body enters the world.
    let custom_models = w
        .with_world(|world| world.resource::<SimulationConfig>().scene.models.clone())
        .await;
    for model in custom_models {
        if !model.enabled {
            continue;
        }
        if let Err(reason) = model.validate() {
            warn!("Skipping custom scene model {:?}: {reason}", model.path);
            continue;
        }

        let path = model.path.trim().to_string();
        if !Path::new("assets").join(&path).is_file() {
            warn!(
                "Skipping custom scene model {path:?}: file was not found below the assets directory"
            );
            continue;
        }

        let root = w
            .spawn(
                WorldAssetRoot(
                    assets.load(GltfAssetLabel::Scene(model.scene_index).from_asset(path.clone())),
                ),
                model.transform(),
            )
            .await;

        if model.collision {
            if model.collision_nodes.is_empty() {
                let collider = trimesh();
                w.with_world(move |world| {
                    world.entity_mut(root).insert((RigidBody::Static, collider));
                })
                .await;
            } else {
                let collision_map = model
                    .collision_nodes
                    .iter()
                    .map(|name| (name.clone(), static_trimesh()))
                    .collect::<HashMap<_, _>>();
                let configured_count = collision_map.len();
                let matched_count = w
                    .run(setup_collision, (root, PreciousCollision(collision_map)))
                    .await;
                if matched_count != configured_count {
                    warn!(
                        "Custom scene model {path:?} matched {matched_count}/{configured_count} collision nodes; configured names: {:?}",
                        model.collision_nodes
                    );
                }
            }
            colliders_ready(&w, root).await;
        }

        info!(
            "Loaded custom scene model {path:?} (scene {}, collision: {})",
            model.scene_index, model.collision
        );
    }

    // Robots last, so they can never become dynamic over an empty world.
    let player = w
        .spawn(
            scene("vehicle.glb"),
            (
                player_start.transform(),
                Infantry::new(Team::Red, INFANTRY_THREE_CONFIG),
                Controlled,
            ),
        )
        .await;
    w.run(setup_vehicle, player).await;

    let slapper = w
        .spawn(
            scene("vehicle.glb"),
            (
                Transform::from_xyz(1.0, 1.0, 1.0),
                Infantry::new(Team::Blue, INFANTRY_THREE_CONFIG),
                SlapperInfantry,
            ),
        )
        .await;
    w.run(setup_vehicle, slapper).await;

    let hero = w
        .spawn(
            scene("HERO.glb"),
            (
                Transform::from_xyz(2.0, 1.0, 1.0),
                Infantry::new(Team::Blue, HERO_ROBOT_CONFIG),
                SlapperInfantry,
                ActiveSlapper,
            ),
        )
        .await;
    w.run(setup_vehicle, hero).await;

    info!("scene loaded");
}

/// Every rune target gets a voxelized collider so hits register on the spinning arms.
fn power_rune_collision(
    layers: CollisionLayers,
) -> HashMap<
    String,
    (
        ColliderConstructorHierarchy,
        CollisionLayers,
        Visibility,
        Option<RigidBody>,
    ),
> {
    let voxel = |size| {
        ColliderConstructorHierarchy::new(ColliderConstructor::VoxelizedTrimeshFromMesh {
            voxel_size: size,
            fill_mode: FillMode::FloodFill {
                detect_cavities: true,
            },
        })
        .with_default_layers(layers)
    };

    let mut collision = HashMap::from([(
        "BASE".to_string(),
        (
            ColliderConstructorHierarchy::new(ColliderConstructor::TrimeshFromMeshWithConfig(
                TrimeshFlags::all(),
            ))
            .with_default_layers(layers),
            layers,
            Visibility::Visible,
            Some(RigidBody::Static),
        ),
    )]);
    for face in 1..=2 {
        for target in 1..=5 {
            for state in ["ACTIVATED", "ACTIVE", "COMPLETED", "DISABLED"] {
                collision.insert(
                    format!("FACE_{face}_TARGET_{target}_{state}"),
                    (voxel(0.015), layers, Visibility::Visible, None),
                );
            }
        }
    }
    collision
}

#[cfg(test)]
mod map_tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};
    use serde_json::Value;

    /// Exercise Avian against the shipped collision buffers, including tunnel voids.
    /// Requiring NORMAL guards against Bevy's glTF loader removing the index buffer.
    #[test]
    fn rmuc2027_colliders_preserve_slopes_and_tunnel_openings() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/maps/rmuc2027/collision.glb"
        ))
        .unwrap();
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json: Value = serde_json::from_slice(&bytes[20..20 + json_len]).unwrap();
        let binary = &bytes[28 + json_len..];
        let buffer = |index: usize| {
            let accessor = &json["accessors"][index];
            let view = &json["bufferViews"][accessor["bufferView"].as_u64().unwrap() as usize];
            let start = view["byteOffset"].as_u64().unwrap_or(0) as usize
                + accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
            let count = accessor["count"].as_u64().unwrap() as usize;
            let stride = if accessor["type"] == "VEC3" { 12 } else { 4 };
            &binary[start..start + count * stride]
        };
        let mut colliders = Vec::new();
        for mesh in json["meshes"].as_array().unwrap() {
            for primitive in mesh["primitives"].as_array().unwrap() {
                assert!(primitive["attributes"]["NORMAL"].is_number());
                let positions: Vec<[f32; 3]> =
                    buffer(primitive["attributes"]["POSITION"].as_u64().unwrap() as usize)
                        .chunks_exact(12)
                        .map(|v| {
                            std::array::from_fn(|i| {
                                f32::from_le_bytes(v[4 * i..4 * i + 4].try_into().unwrap())
                            })
                        })
                        .collect();
                let indices: Vec<u32> = buffer(primitive["indices"].as_u64().unwrap() as usize)
                    .chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                    .collect();
                let mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
                .with_inserted_indices(Indices::U32(indices));
                colliders.push(Collider::trimesh_from_mesh(&mesh).expect("CAD collider"));
            }
        }
        assert_eq!(colliders.len(), 396);
        let hit = |origin: Vec3, direction: Vec3, distance| {
            colliders
                .iter()
                .filter_map(|c| {
                    c.cast_ray(
                        Vec3::ZERO,
                        Quat::IDENTITY,
                        origin,
                        direction,
                        distance,
                        false,
                    )
                    .map(|(distance, _)| distance)
                })
                .min_by(f32::total_cmp)
        };
        // The saved horizontal spawn remains over the original CAD floor, now at +89 mm.
        let floor = 4.0 - hit(Vec3::new(-4.1976643, 4.0, -4.590342), Vec3::NEG_Y, 5.0).unwrap();
        assert!((floor - 0.0893).abs() < 0.001, "spawn floor {floor}");
        // This lane has the source slope, rather than an axis-aligned flat replacement.
        let a = 4.0 - hit(Vec3::new(3.0, 4.0, 0.0), Vec3::NEG_Y, 5.0).unwrap();
        let b = 4.0 - hit(Vec3::new(3.0, 4.0, 1.0), Vec3::NEG_Y, 5.0).unwrap();
        assert!((a - 0.5095).abs() < 0.001);
        assert!((b - 0.4833).abs() < 0.001);
        for z in [-7.0, 7.0] {
            assert!(hit(Vec3::new(-1.5, 0.35, z), Vec3::X, 3.0).is_none());
            assert!(hit(Vec3::new(-1.5, 0.49, z), Vec3::X, 3.0).is_some());
            let ceiling = hit(Vec3::new(0.0, 0.35, z), Vec3::Y, 1.0).unwrap();
            assert!((ceiling - 0.1264).abs() < 0.001);
        }
    }
}
