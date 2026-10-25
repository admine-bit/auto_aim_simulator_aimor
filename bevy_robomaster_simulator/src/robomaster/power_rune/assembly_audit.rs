//! Optional runtime evidence for the visual assembly regression. Disabled in normal runs.
use super::collision::{RuneColliderProxy, sync_rune_collider_proxies};
use super::construct::PowerRuneRoot;
use super::rotation::PowerRuneRotation;
use super::rune::PowerRuneMechanism;
use avian3d::prelude::{Collider, Position, RigidBody, Rotation};
use bevy::{prelude::*, transform::helper::TransformHelper};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
};

#[derive(Resource)]
struct AssemblyAudit {
    output: BufWriter<File>,
    reference: HashMap<Entity, (Transform, Option<Entity>)>,
    frames: u64,
    next_log: f64,
    max_translation: f32,
    max_axes: f32,
    max_scale: f32,
    parent_changes: u64,
    max_proxy_position: f32,
    max_proxy_axes: f32,
    max_proxy_scale: f32,
}

pub(super) fn install(app: &mut App) {
    // 需要长时验证显示装配时设置输出 CSV；日常不设置，不采样也不写日志。
    let Some(path) = std::env::var_os("DAEDALUS_RUNE_ASSEMBLY_AUDIT") else {
        return;
    };
    let mut output = BufWriter::new(File::create(path).expect("open rune assembly audit CSV"));
    writeln!(output, "elapsed_s,frames,visual_nodes,mesh_count,proxy_count,max_local_translation_m,max_local_axis_error,max_local_scale_error,parent_changes,visual_physics_nodes,max_proxy_position_m,max_proxy_axis_error,max_proxy_scale_error,missing_sources").unwrap();
    app.insert_resource(AssemblyAudit {
        output,
        reference: HashMap::new(),
        frames: 0,
        next_log: 0.0,
        max_translation: 0.0,
        max_axes: 0.0,
        max_scale: 0.0,
        parent_changes: 0,
        max_proxy_position: 0.0,
        max_proxy_axes: 0.0,
        max_proxy_scale: 0.0,
    })
    .add_systems(
        PostUpdate,
        audit
            .after(TransformSystems::Propagate)
            .after(sync_rune_collider_proxies),
    );
}

fn audit(
    time: Res<Time>,
    mut audit: ResMut<AssemblyAudit>,
    roots: Query<Entity, With<PowerRuneRoot>>,
    runes: Query<(), With<PowerRuneMechanism>>,
    children: Query<&Children>,
    nodes: Query<(
        &Transform,
        Option<&ChildOf>,
        Has<Mesh3d>,
        Has<PowerRuneRotation>,
        Has<RigidBody>,
        Has<Collider>,
    )>,
    proxies: Query<(&RuneColliderProxy, &Position, &Rotation)>,
    source_transforms: TransformHelper,
) {
    // Both mechanisms exist only after the fixed R-marker parenting is complete.
    if runes.iter().count() < 2 {
        return;
    }
    let mut node_count = 0;
    let mut mesh_count = 0;
    let mut visual_physics_nodes = 0;
    for root in &roots {
        for entity in std::iter::once(root).chain(children.iter_descendants(root)) {
            let Ok((current, parent, mesh, rotating, rigid_body, collider)) = nodes.get(entity)
            else {
                continue;
            };
            node_count += 1;
            mesh_count += usize::from(mesh);
            visual_physics_nodes += usize::from(rigid_body || collider);
            let parent = parent.map(ChildOf::parent);
            let (reference, original_parent) =
                *audit.reference.entry(entity).or_insert((*current, parent));
            audit.parent_changes += u64::from(parent != original_parent);
            audit.max_translation = audit
                .max_translation
                .max((current.translation - reference.translation).length());
            audit.max_scale = audit
                .max_scale
                .max((current.scale - reference.scale).abs().max_element());
            // A rotating face may change its rotation, but not its origin, scale or parent.
            if !rotating {
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    audit.max_axes = audit
                        .max_axes
                        .max((current.rotation * axis - reference.rotation * axis).length());
                }
            }
        }
    }
    if mesh_count == 0 {
        return;
    }
    let mut proxy_count = 0;
    let mut missing_sources = 0;
    for (proxy, position, rotation) in &proxies {
        proxy_count += 1;
        let Ok(source) = source_transforms.compute_global_transform(proxy.source) else {
            missing_sources += 1;
            continue;
        };
        let expected = source.mul_transform(proxy.local).compute_transform();
        audit.max_proxy_position = audit
            .max_proxy_position
            .max((position.0 - expected.translation).length());
        audit.max_proxy_scale = audit
            .max_proxy_scale
            .max((expected.scale - Vec3::ONE).abs().max_element());
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            audit.max_proxy_axes = audit
                .max_proxy_axes
                .max((rotation.0 * axis - expected.rotation * axis).length());
        }
    }
    audit.frames += 1;
    let now = time.elapsed_secs_f64();
    if now >= audit.next_log {
        let line = format!(
            "{now:.6},{},{node_count},{mesh_count},{proxy_count},{:.9},{:.9},{:.9},{},{visual_physics_nodes},{:.9},{:.9},{:.9},{missing_sources}\n",
            audit.frames,
            audit.max_translation,
            audit.max_axes,
            audit.max_scale,
            audit.parent_changes,
            audit.max_proxy_position,
            audit.max_proxy_axes,
            audit.max_proxy_scale,
        );
        audit.output.write_all(line.as_bytes()).unwrap();
        audit.output.flush().unwrap();
        audit.next_log = now + 1.0;
    }
}
