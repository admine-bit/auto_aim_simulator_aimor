//! Run the real chassis and drive law over the shipped map collision buffers.
use avian3d::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use serde_json::Value;
use std::time::Duration;

use crate::components::{GameLayer, Infantry};
use crate::robomaster::prelude::{INFANTRY_THREE_CONFIG, Team};
use crate::robomaster::vehicle::movement::VehicleDynamic;
use crate::terrain_collision::{TerrainCollisionHooks, TunnelDeck};

struct MapPart {
    name: String,
    mesh: Mesh,
}

fn map_parts() -> Vec<MapPart> {
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
    json["meshes"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|mesh| {
            mesh["primitives"]
                .as_array()
                .unwrap()
                .iter()
                .map(|primitive| {
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
                    MapPart {
                        name: mesh["name"].as_str().unwrap().to_owned(),
                        mesh: Mesh::new(
                            PrimitiveTopology::TriangleList,
                            RenderAssetUsages::default(),
                        )
                        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
                        .with_inserted_indices(Indices::U32(indices)),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

// Armor is a child collider in the real vehicle. Keep its exact glTF geometry and transforms,
// so terrain acceptance does not silently test a smaller cylinder-only robot.
fn vehicle_armor_colliders() -> Vec<Collider> {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/vehicle.glb")).unwrap();
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json: Value = serde_json::from_slice(&bytes[20..20 + json_len]).unwrap();
    let binary = &bytes[28 + json_len..];
    fn visit(json: &Value, binary: &[u8], index: usize, parent: Mat4, output: &mut Vec<Collider>) {
        let node = &json["nodes"][index];
        let vector = |key: &str, fallback: Vec3| -> Vec3 {
            node[key]
                .as_array()
                .map(|v| {
                    Vec3::new(
                        v[0].as_f64().unwrap() as f32,
                        v[1].as_f64().unwrap() as f32,
                        v[2].as_f64().unwrap() as f32,
                    )
                })
                .unwrap_or(fallback)
        };
        let local = if let Some(values) = node["matrix"].as_array() {
            Mat4::from_cols_array(&std::array::from_fn(|i| values[i].as_f64().unwrap() as f32))
        } else {
            let rotation = node["rotation"]
                .as_array()
                .map(|v| Quat::from_array(std::array::from_fn(|i| v[i].as_f64().unwrap() as f32)))
                .unwrap_or(Quat::IDENTITY);
            Mat4::from_scale_rotation_translation(
                vector("scale", Vec3::ONE),
                rotation,
                vector("translation", Vec3::ZERO),
            )
        };
        let transform = parent * local;
        if node["name"]
            .as_str()
            .is_some_and(|name| name.ends_with("_ARMOR"))
        {
            let primitive =
                &json["meshes"][node["mesh"].as_u64().unwrap() as usize]["primitives"][0];
            let accessor = |index: usize| {
                let a = &json["accessors"][index];
                let view = &json["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
                let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize
                    + a["byteOffset"].as_u64().unwrap_or(0) as usize;
                (a, offset)
            };
            let (a, offset) =
                accessor(primitive["attributes"]["POSITION"].as_u64().unwrap() as usize);
            let positions: Vec<[f32; 3]> = binary
                [offset..offset + a["count"].as_u64().unwrap() as usize * 12]
                .chunks_exact(12)
                .map(|v| {
                    transform
                        .transform_point3(Vec3::from_array(std::array::from_fn(|i| {
                            f32::from_le_bytes(v[4 * i..4 * i + 4].try_into().unwrap())
                        })))
                        .to_array()
                })
                .collect();
            let (a, offset) = accessor(primitive["indices"].as_u64().unwrap() as usize);
            let stride = match a["componentType"].as_u64().unwrap() {
                5123 => 2,
                5125 => 4,
                other => panic!("unsupported index type {other}"),
            };
            let indices: Vec<u32> = binary
                [offset..offset + a["count"].as_u64().unwrap() as usize * stride]
                .chunks_exact(stride)
                .map(|v| {
                    if stride == 2 {
                        u16::from_le_bytes(v.try_into().unwrap()) as u32
                    } else {
                        u32::from_le_bytes(v.try_into().unwrap())
                    }
                })
                .collect();
            let mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_indices(Indices::U32(indices));
            output.push(Collider::trimesh_from_mesh(&mesh).unwrap());
        }
        if let Some(children) = node["children"].as_array() {
            for child in children {
                visit(
                    json,
                    binary,
                    child.as_u64().unwrap() as usize,
                    transform,
                    output,
                );
            }
        }
    }
    let mut result = Vec::new();
    for root in json["scenes"][0]["nodes"].as_array().unwrap() {
        visit(
            &json,
            binary,
            root.as_u64().unwrap() as usize,
            Mat4::IDENTITY,
            &mut result,
        );
    }
    assert_eq!(result.len(), 4);
    result
}

#[derive(Resource)]
struct Drive {
    enabled: bool,
    frame: GlobalTransform,
}

fn drive(
    time: Res<Time>,
    drive: Res<Drive>,
    mut cars: Query<(Forces, &Mass, &mut Friction, &mut VehicleDynamic)>,
) {
    for (mut forces, mass, mut friction, mut dynamics) in &mut cars {
        let input = if drive.enabled { Vec2::Y } else { Vec2::ZERO };
        dynamics.linear(
            &mut forces,
            &mut friction,
            mass.0,
            &drive.frame,
            input,
            time.delta_secs(),
            1.0,
        );
    }
}

#[derive(Clone, Copy)]
struct Route {
    name: &'static str,
    start: Vec3,
    end: Vec3,
    seconds: f32,
}

#[derive(Debug)]
struct Result {
    progress: f32,
    max_progress: f32,
    final_position: Vec3,
    final_speed: f32,
    min_height: f32,
    max_height: f32,
    lateral_displacement: f32,
    stopped_speed: f32,
    last_second_drift: f32,
    held_position: Vec3,
}

fn traverse(colliders: &[(String, Collider)], route: Route) -> Result {
    let direction = (route.end - route.start).with_y(0.0).normalize();
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        AssetPlugin::default(),
        bevy::mesh::MeshPlugin,
        PhysicsPlugins::default().with_collision_hooks::<TerrainCollisionHooks>(),
    ))
    .insert_resource(Gravity(Vec3::NEG_Y * 9.81))
    .insert_resource(Time::<Fixed>::from_hz(120.0))
    .insert_resource(SubstepCount(8))
    .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(0.01),
    ))
    .insert_resource(Drive {
        enabled: false,
        frame: GlobalTransform::from(Transform::IDENTITY.looking_to(direction, Vec3::Y)),
    })
    .add_systems(Update, drive);
    app.finish();
    app.cleanup();
    let ground = app
        .world_mut()
        .spawn((RigidBody::Static, Friction::new(0.5), Transform::IDENTITY))
        .id();
    for (name, collider) in colliders {
        // Avian 0.7 reads MODIFY_CONTACTS when the Collider is inserted.
        let mut entity = app.world_mut().spawn_empty();
        if name.starts_with("tunnel_deck_") {
            entity.insert(TunnelDeck);
        }
        entity.insert((
            Name::new(name.clone()),
            collider.clone(),
            ChildOf(ground),
            GameLayer::environment_collision_layers(),
            Transform::IDENTITY,
        ));
    }
    let car = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Infantry::new(Team::Red, INFANTRY_THREE_CONFIG),
            VehicleDynamic::new(6.0, 6.0, 2.5),
            Collider::compound(vec![(
                Vec3::new(0.0, -0.115649, 0.0),
                Quat::IDENTITY,
                Collider::cylinder(0.2593615, 0.231298),
            )]),
            CollisionMargin(0.005),
            GameLayer::vehicle_body_collision_layers(true),
            Mass(15.0),
            Friction::default(),
            Restitution::new(0.01),
            AngularDamping(50.0),
            Transform::from_translation(route.start),
        ))
        .id();
    for collider in vehicle_armor_colliders() {
        app.world_mut().spawn((
            collider,
            ChildOf(car),
            Transform::IDENTITY,
            GameLayer::vehicle_armor_collision_layers(true),
        ));
    }
    for _ in 0..100 {
        app.update();
    }
    let settled = app.world().get::<Position>(car).unwrap().0;
    app.world_mut().resource_mut::<Drive>().enabled = true;
    let mut max_progress: f32 = 0.0;
    let mut min_height = settled.y;
    let mut max_height = settled.y;
    for _ in 0..(route.seconds * 100.0) as usize {
        app.update();
        let position = app.world().get::<Position>(car).unwrap().0;
        max_progress = max_progress.max((position - settled).dot(direction));
        min_height = min_height.min(position.y);
        max_height = max_height.max(position.y);
        if max_progress >= (route.end - route.start).with_y(0.0).length() {
            break;
        }
    }
    let position = app.world().get::<Position>(car).unwrap().0;
    let final_speed = app.world().get::<LinearVelocity>(car).unwrap().length();
    app.world_mut().resource_mut::<Drive>().enabled = false;
    let mut before_last_second = position;
    for step in 0..400 {
        app.update();
        if step == 299 {
            before_last_second = app.world().get::<Position>(car).unwrap().0;
        }
    }
    let held_position = app.world().get::<Position>(car).unwrap().0;
    assert_eq!(
        *app.world().get::<Friction>(car).unwrap(),
        Friction::default()
    );
    Result {
        progress: (position - settled).dot(direction),
        max_progress,
        final_position: position,
        final_speed,
        min_height,
        max_height,
        lateral_displacement: ((position - settled).with_y(0.0)
            - direction * (position - settled).dot(direction))
        .length(),
        stopped_speed: app.world().get::<LinearVelocity>(car).unwrap().length(),
        last_second_drift: (held_position - before_last_second).length(),
        held_position,
    }
}

fn colliders(parts: &[MapPart], flags: TrimeshFlags) -> Vec<(String, Collider)> {
    parts
        .iter()
        .map(|p| {
            (
                p.name.clone(),
                Collider::trimesh_from_mesh_with_config(&p.mesh, flags).unwrap(),
            )
        })
        .collect()
}

fn route(name: &'static str, start: [f32; 3], end: [f32; 3]) -> Route {
    Route {
        name,
        start: Vec3::from_array(start),
        end: Vec3::from_array(end),
        seconds: 8.0,
    }
}

fn assert_crossed(route: Route, result: &Result) {
    let distance = (route.end - route.start).with_y(0.0).length();
    assert!(
        result.progress >= distance,
        "{} did not cross: {result:?}",
        route.name
    );
    assert!(
        result.lateral_displacement < 0.15,
        "{} slid sideways: {result:?}",
        route.name
    );
    assert!(
        result.min_height > 0.2 && result.max_height < 1.0,
        "{} left the terrain: {result:?}",
        route.name
    );
}

#[test]
fn vehicle_crosses_flat_seams_and_both_ramps() {
    let colliders = colliders(&map_parts(), super::terrain_trimesh_flags());
    let routes = [
        route("flat-east", [-8.0, 0.4, -2.8], [-4.0, 0.4, -2.8]),
        route("flat-west", [-4.0, 0.4, -2.8], [-8.0, 0.4, -2.8]),
        route("west-ramp-up", [-4.5, 0.36, -4.6], [-6.5, 0.6, -4.6]),
        route("west-ramp-down", [-6.5, 0.6, -4.6], [-4.5, 0.36, -4.6]),
        route("east-ramp-up", [4.5, 0.36, 4.6], [6.5, 0.6, 4.6]),
        route("east-ramp-down", [6.5, 0.6, 4.6], [4.5, 0.36, 4.6]),
        route("millimeter-strip-east", [2.9, 0.8, 0.0], [3.4, 0.8, 0.0]),
        route("millimeter-strip-west", [3.4, 0.8, 0.0], [2.9, 0.8, 0.0]),
        route("center-north", [3.0, 0.8, -2.0], [3.0, 0.8, 2.0]),
        route("center-south", [-3.0, 0.8, 2.0], [-3.0, 0.8, -2.0]),
    ];
    for route in routes {
        let result = traverse(&colliders, route);
        eprintln!("TRAVERSE {}: {result:?}", route.name);
        assert_crossed(route, &result);
        assert!(
            result.stopped_speed < 0.02 && result.last_second_drift < 0.005,
            "{} failed to stop after release: {result:?}",
            route.name
        );
    }
}

#[test]
fn complete_vehicle_crosses_both_tunnels_and_remains_supported_on_decks() {
    let colliders = colliders(&map_parts(), super::terrain_trimesh_flags());
    for z in [-7.0, 7.0] {
        for (from, to) in [(-1.8, 1.8), (1.8, -1.8)] {
            let route = route("tunnel", [from, 0.36, z], [to, 0.36, z]);
            let result = traverse(&colliders, route);
            eprintln!("TUNNEL z={z} from={from}: {result:?}");
            assert_crossed(route, &result);
        }
        let mut deck = route("deck-support", [0.0, 1.3, z], [0.1, 1.3, z]);
        deck.seconds = 0.0;
        let result = traverse(&colliders, deck);
        eprintln!("DECK z={z}: {result:?}");
        assert!(
            result.held_position.y > 0.7 && result.held_position.y < 0.9,
            "deck lost support: {result:?}"
        );
        assert!(result.stopped_speed < 0.02 && result.last_second_drift < 0.005);
    }
}

#[test]
fn vehicle_still_hits_real_walls_and_holds_on_slopes_when_released() {
    let colliders = colliders(&map_parts(), super::terrain_trimesh_flags());
    let wall = route("highland-barrier", [2.8, 0.8, 0.0], [4.2, 0.8, 0.0]);
    let result = traverse(&colliders, wall);
    eprintln!("WALL: {result:?}");
    assert!(
        result.final_position.x < 3.7 && result.progress < 1.0,
        "vehicle passed through the real barrier: {result:?}"
    );
    for sign in [-1.0, 1.0] {
        let mut slope = route(
            "park-on-ramp",
            [sign * 5.25, 0.5, sign * 4.6],
            [sign * 5.3, 0.5, sign * 4.6],
        );
        slope.seconds = 0.0;
        let result = traverse(&colliders, slope);
        eprintln!("PARK sign={sign}: {result:?}");
        assert!(
            result.stopped_speed < 0.02 && result.last_second_drift < 0.005,
            "parked chassis slid on the ramp: {result:?}"
        );
        assert!(
            (result.held_position.x - sign * 5.25).abs() < 0.2,
            "parked chassis left the ramp: {result:?}"
        );
    }
}
