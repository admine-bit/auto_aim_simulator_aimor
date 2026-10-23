use avian3d::prelude::{Gravity, LinearVelocity, SpatialQuery, SpatialQueryFilter};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bevy::window::PrimaryWindow;

use super::VisionMode;
use super::ballistics::{BallisticModel, project_to_preview};
use super::debug::HelpText;
use super::projectile_launch_geometry;
use crate::capture::{CaptureCamera, PreviewCamera};
use crate::components::{
    CameraMode, Controlled, FollowingType, GameLayer, Infantry, InfantryGimbal,
    InfantryLaunchOffset, MainCamera, SubscribeAutoAim,
};
use crate::config::{BallisticHudConfig, SimulationConfig};
use crate::robomaster::prelude::{RuneHitFeedback, RuneHitOutcome, Team};

pub struct BallisticHudPlugin;

impl Plugin for BallisticHudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialPlugin::<ScopeMaterial>::default())
            .init_resource::<HudSettings>()
            .add_systems(Startup, spawn_hud)
            .add_systems(Update, hud_controls)
            .add_systems(PostUpdate, update_hud.after(TransformSystems::Propagate));
    }
}

#[derive(Resource, Default)]
struct HudSettings {
    visible: Option<bool>,
    automatic: Option<bool>,
    manual_range: Option<f32>,
}

impl HudSettings {
    fn range(&self, config: &BallisticHudConfig) -> f32 {
        self.manual_range
            .unwrap_or(config.manual_range_m)
            .clamp(0.5, config.max_range_m.max(0.5))
    }

    fn automatic(&self, config: &BallisticHudConfig) -> bool {
        self.automatic.unwrap_or(config.automatic_range)
    }
}

// Analytic shading keeps the circular aperture, etched lines and anti-aliasing crisp
// at any window size. This material is attached only to the local preview camera.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct ScopeMaterial {
    #[uniform(0)]
    optics: Vec4, // focal lengths / window height, aperture radius / window height
    #[uniform(1)]
    impact: Vec4, // normalized preview position, valid flag
}

impl UiMaterial for ScopeMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/optical_scope.wgsl".into()
    }
}

#[derive(Resource)]
struct ScopeMaterialHandle(Handle<ScopeMaterial>);

#[derive(Component)]
struct HudRoot;

#[derive(Component, Clone, Copy)]
enum HudText {
    Range,
    Raise,
    Details,
    Status,
    SelectedRange,
    Ring,
}

#[derive(Component, Clone, Copy)]
enum ScopeDecoration {
    Angular(f32),
    Range(f32),
    Selected,
}

const ETCHED: Color = Color::srgb(0.64, 0.62, 0.54);
const AMBER: Color = Color::srgb(0.77, 0.66, 0.45);
const QUIET: Color = Color::srgb(0.42, 0.43, 0.40);

fn font(handle: &Handle<Font>, size: f32) -> TextFont {
    TextFont {
        font: handle.clone().into(),
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn spawn_hud(
    mut commands: Commands,
    assets: Res<AssetServer>,
    mut materials: ResMut<Assets<ScopeMaterial>>,
) {
    let face = assets.load("fonts/ballistic-hud.otf");
    let material = materials.add(ScopeMaterial {
        optics: Vec4::new(1.3, 1.3, 0.46, 0.0),
        impact: Vec4::ZERO,
    });
    commands.insert_resource(ScopeMaterialHandle(material.clone()));
    commands
        .spawn((
            HudRoot,
            Node {
                position_type: PositionType::Absolute,
                width: percent(100),
                height: percent(100),
                display: Display::None,
                ..default()
            },
            GlobalZIndex(10),
            Pickable::IGNORE,
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    width: percent(100),
                    height: percent(100),
                    ..default()
                },
                MaterialNode(material),
            ));

            // Compact instrument readout on the dark eye relief, outside the sight picture.
            root.spawn(Node {
                position_type: PositionType::Absolute,
                left: px(26),
                bottom: px(43),
                width: px(270),
                flex_direction: FlexDirection::Column,
                row_gap: px(6),
                ..default()
            })
            .with_children(|readout| {
                readout.spawn((
                    Text::new("弹道 / 测距"),
                    font(&face, 12.0),
                    TextColor(QUIET),
                ));
                readout.spawn((
                    HudText::Range,
                    Text::new(""),
                    font(&face, 15.0),
                    TextColor(ETCHED),
                ));
                readout.spawn((
                    HudText::Raise,
                    Text::new(""),
                    font(&face, 21.0),
                    TextColor(AMBER),
                ));
                readout.spawn((
                    HudText::Details,
                    Text::new(""),
                    font(&face, 12.0),
                    TextColor(ETCHED),
                ));
            });
            root.spawn((
                Text::new("10 mrad / 格"),
                font(&face, 11.0),
                TextColor(QUIET),
                Node {
                    position_type: PositionType::Absolute,
                    left: percent(50),
                    top: percent(50),
                    margin: UiRect {
                        left: px(16),
                        top: Val::VMin(-22.0),
                        ..default()
                    },
                    ..default()
                },
            ));
            root.spawn((
                Text::new("H 收镜   T 测距模式   PgUp/PgDn 距离   Shift 精调"),
                font(&face, 11.0),
                TextColor(QUIET),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(26),
                    bottom: px(16),
                    ..default()
                },
            ));
            root.spawn((
                HudText::Status,
                Text::new(""),
                font(&face, 12.0),
                TextColor(QUIET),
                Node {
                    position_type: PositionType::Absolute,
                    right: px(26),
                    bottom: px(44),
                    ..default()
                },
            ));
            root.spawn((
                HudText::Ring,
                Text::new("环数 —"),
                font(&face, 16.0),
                TextColor(AMBER),
                Node {
                    position_type: PositionType::Absolute,
                    right: px(26),
                    bottom: px(76),
                    ..default()
                },
            ));

            // Numbers on the horizontal angular scale; positions use the real preview lens.
            for mrad in [-100_i32, -80, -60, -40, -20, 20, 40, 60, 80, 100] {
                root.spawn((
                    ScopeDecoration::Angular(mrad as f32),
                    Text::new(mrad.unsigned_abs().to_string()),
                    font(&face, 10.0),
                    TextColor(ETCHED),
                    TextShadow::default(),
                    Node {
                        position_type: PositionType::Absolute,
                        width: px(24),
                        margin: UiRect {
                            left: px(-12),
                            top: px(10),
                            ..default()
                        },
                        ..default()
                    },
                    TextLayout::justify(Justify::Center),
                ));
            }
            // Ballistic drop marks are computed from actual flight, rather than decorative spacing.
            for distance in [5.0, 10.0, 15.0, 20.0, 30.0] {
                root.spawn((
                    ScopeDecoration::Range(distance),
                    Node {
                        position_type: PositionType::Absolute,
                        width: px(0),
                        height: px(0),
                        display: Display::None,
                        ..default()
                    },
                ))
                .with_children(|mark| {
                    mark.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(-7),
                            top: px(-0.5),
                            width: px(14),
                            height: px(1),
                            ..default()
                        },
                        BackgroundColor(QUIET),
                    ));
                    mark.spawn((
                        Text::new(format!("{distance:.0}")),
                        font(&face, 11.0),
                        TextColor(ETCHED),
                        TextShadow::default(),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(13),
                            top: px(-8),
                            ..default()
                        },
                    ));
                });
            }
            root.spawn((
                ScopeDecoration::Selected,
                HudText::SelectedRange,
                Text::new(""),
                font(&face, 12.0),
                TextColor(AMBER),
                TextShadow::default(),
                Node {
                    position_type: PositionType::Absolute,
                    margin: UiRect {
                        left: px(17),
                        top: px(-9),
                        ..default()
                    },
                    display: Display::None,
                    ..default()
                },
            ));
        });
}

fn hud_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    config: Res<SimulationConfig>,
    mode: Res<CameraMode>,
    mut settings: ResMut<HudSettings>,
) {
    if mode.0 != FollowingType::Robot {
        return;
    }
    let cfg = &config.ballistic_hud;
    if keyboard.just_pressed(KeyCode::KeyH) {
        settings.visible = Some(!settings.visible.unwrap_or(cfg.enabled));
    }
    if keyboard.just_pressed(KeyCode::KeyT) {
        settings.automatic = Some(!settings.automatic(cfg));
    }
    let adjustment = i32::from(keyboard.just_pressed(KeyCode::PageUp))
        - i32::from(keyboard.just_pressed(KeyCode::PageDown));
    if adjustment != 0 {
        let fine = keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);
        let step = cfg.range_step_m.max(0.01) * if fine { 0.1 } else { 1.0 };
        settings.manual_range = Some(
            (settings.range(cfg) + adjustment as f32 * step).clamp(0.5, cfg.max_range_m.max(0.5)),
        );
        settings.automatic = Some(false);
    }
}

fn update_hud(
    mut commands: Commands,
    context: (
        Res<Time>,
        Res<SimulationConfig>,
        Res<HudSettings>,
        Res<CameraMode>,
        Res<Gravity>,
    ),
    spatial: SpatialQuery,
    preview: Query<Entity, With<PreviewCamera>>,
    capture: Query<&Projection, With<CaptureCamera>>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(Entity, &Transform, &Projection), With<MainCamera>>,
    infantry: Query<(&Transform, &LinearVelocity), (With<Infantry>, With<Controlled>)>,
    gimbal: Query<&GlobalTransform, (With<InfantryGimbal>, With<Controlled>)>,
    launch: Query<&Transform, (With<InfantryLaunchOffset>, With<Controlled>)>,
    mut root: Single<(Entity, &mut Node, Option<&UiTargetCamera>), With<HudRoot>>,
    mut decorations: Query<(&ScopeDecoration, &mut Node), (Without<HudRoot>, Without<HelpText>)>,
    mut labels: Query<(&HudText, &mut Text)>,
    mut help: Query<&mut Node, (With<HelpText>, Without<HudRoot>)>,
    hit_feedback: Option<Res<RuneHitFeedback>>,
    state: (
        (Res<ScopeMaterialHandle>, ResMut<Assets<ScopeMaterial>>),
        (Res<SubscribeAutoAim>, Res<VisionMode>),
        Local<f64>,
    ),
) {
    let (time, config, settings, mode, gravity) = context;
    let (material, status, mut last_update) = state;
    let cfg = &config.ballistic_hud;
    let visible = settings.visible.unwrap_or(cfg.enabled)
        && config.preview.enabled
        && mode.0 == FollowingType::Robot;
    let display = if visible {
        Display::Flex
    } else {
        Display::None
    };
    if root.1.display != display {
        root.1.display = display;
    }
    for mut help in &mut help {
        let display = if visible {
            Display::None
        } else {
            Display::Flex
        };
        if help.display != display {
            help.display = display;
        }
    }
    if !visible {
        return;
    }
    let target_camera = preview.single().unwrap_or(camera.0);
    if root.2.is_none_or(|current| current.0 != target_camera) {
        commands
            .entity(root.0)
            .insert(UiTargetCamera(target_camera));
    }
    if time.elapsed_secs_f64() - *last_update < 0.05
        && !settings.is_changed()
        && !config.is_changed()
    {
        return;
    }
    *last_update = time.elapsed_secs_f64();
    let (handle, mut materials) = material;
    let Some(mut material) = materials.get_mut(&handle.0) else {
        return;
    };
    material.impact = Vec4::ZERO;
    for (_, mut node) in &mut decorations {
        node.display = Display::None;
    }
    let (Ok((body, velocity)), Ok(gimbal), Ok(launch)) =
        (infantry.single(), gimbal.single(), launch.single())
    else {
        return;
    };
    let camera_transform = camera.1;
    let forward = camera_transform.forward();
    let filter = SpatialQueryFilter::from_mask([
        GameLayer::Default,
        GameLayer::VehicleOther,
        GameLayer::Environment,
    ]);
    let hit = settings
        .automatic(cfg)
        .then(|| {
            spatial.cast_ray(
                camera_transform.translation,
                forward,
                cfg.max_range_m.max(0.5),
                true,
                &filter,
            )
        })
        .flatten()
        .filter(|hit| hit.distance >= 0.5);
    let range = hit.map_or_else(|| settings.range(cfg), |hit| hit.distance);
    let source = if hit.is_some() {
        "自动"
    } else if settings.automatic(cfg) {
        "无回波 / 参考"
    } else {
        "手动"
    };
    let target = camera_transform.translation + forward * range;
    let (origin, direction) = projectile_launch_geometry(body, gimbal, launch);
    let model = BallisticModel::new(&config, gravity.0, velocity.0);
    let aim = model.solve(origin, target);
    let projection = capture.single().unwrap_or(camera.2);
    let (fov, aspect) = match projection {
        Projection::Perspective(p) => (p.fov, p.aspect_ratio),
        _ => return,
    };
    let window_aspect = window.width() / window.height().max(1.0);
    let radius = window_aspect.min(1.0) * 0.46;
    let fy = 0.5 / (fov * 0.5).tan();
    let fx = fy * window_aspect / aspect;
    material.optics = Vec4::new(fx, fy, radius, 0.0);
    let project = |point| {
        project_to_preview(camera_transform, point, fov, aspect)
            .filter(|p| ((p.x - 0.5) * window_aspect).hypot(p.y - 0.5) < radius * 0.93)
    };
    let flight_at = |range: f32| {
        let target = camera_transform.translation + forward * range;
        let horizontal = Vec3::new(target.x - origin.x, 0.0, target.z - origin.z);
        model.flight_to_plane(
            origin,
            direction,
            horizontal.normalize_or_zero(),
            horizontal.length(),
        )
    };
    let impact = flight_at(range).and_then(|flight| project(flight.position));
    if let Some(p) = impact {
        material.impact = Vec4::new(p.x, p.y, 1.0, 0.0);
    }
    let mut used_rows: Vec<f32> = impact.map(|p| p.y).into_iter().collect();
    for (decoration, mut node) in &mut decorations {
        let position = match *decoration {
            ScopeDecoration::Angular(mrad) => {
                let x = fx * (mrad * 0.001).tan();
                (x.abs() < radius * 0.65).then_some(Vec2::new(0.5 + x / window_aspect, 0.5))
            }
            ScopeDecoration::Selected => impact,
            ScopeDecoration::Range(distance) => {
                let point = flight_at(distance).and_then(|flight| project(flight.position));
                point.filter(|p| {
                    if (distance - range).abs() < 0.5
                        || used_rows
                            .iter()
                            .any(|y| (p.y - y).abs() * window.height() < 18.0)
                    {
                        return false;
                    }
                    used_rows.push(p.y);
                    true
                })
            }
        };
        if let Some(p) = position {
            node.display = Display::Flex;
            node.left = percent(p.x * 100.0);
            node.top = percent(p.y * 100.0);
        }
    }
    let auto = status.0.load(std::sync::atomic::Ordering::Acquire);
    for (kind, mut text) in &mut labels {
        let value = match kind {
            HudText::Range => format!("距离 {range:05.2} m  /  {source}"),
            HudText::Raise => match &aim {
                Some(aim) => format!("预抬 {:+.1} cm", aim.raise_m * 100.0),
                None => "无可用低弹道解".to_owned(),
            },
            HudText::Details => match &aim {
                Some(aim) => format!(
                    "补偿 {:+.2}°  /  {:.1} mrad\n初速 {:.1} m/s  ·  飞行 {:.0} ms",
                    aim.elevation_delta.to_degrees(),
                    aim.elevation_delta * 1000.0,
                    model.speed,
                    aim.flight.time * 1000.0
                ),
                None => "距离超出弹道或寿命范围".to_owned(),
            },
            HudText::Status => format!(
                "{}  /  视觉模式 {}",
                if auto { "自动跟随" } else { "手动瞄准" },
                status.1.0
            ),
            HudText::SelectedRange => format!("{range:.1} m"),
            HudText::Ring => hit_feedback
                .as_ref()
                .and_then(|f| f.last.as_ref())
                .map_or_else(
                    || "环数 —  /  等待有效靶面命中".to_owned(),
                    |hit| {
                        format!(
                            "{}符 · 第 {} 靶  /  {} 环\n径向 {} mm · {}",
                            if hit.team == Team::Red { "红" } else { "蓝" },
                            hit.result.target + 1,
                            hit.result.ring,
                            hit.result.radial_mm,
                            match hit.result.outcome {
                                RuneHitOutcome::WrongTarget => "错靶重置",
                                RuneHitOutcome::Ignored => "当前未亮靶",
                                _ => "有效命中",
                            }
                        )
                    },
                ),
        };
        if text.0 != value {
            text.0 = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controls_adjust_distance_toggle_modes_and_ignore_other_views() {
        let mut app = App::new();
        app.insert_resource(SimulationConfig::default())
            .init_resource::<HudSettings>()
            .init_resource::<CameraMode>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, hud_controls);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::PageUp);
        app.update();
        let cfg = &app.world().resource::<SimulationConfig>().ballistic_hud;
        let state = app.world().resource::<HudSettings>();
        assert!(!state.automatic(cfg));
        assert!((state.range(cfg) - (cfg.manual_range_m + cfg.range_step_m)).abs() < 1e-5);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyT);
        app.update();
        assert_eq!(app.world().resource::<HudSettings>().automatic, Some(true));
        app.world_mut().resource_mut::<CameraMode>().0 = FollowingType::Free;
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyH);
        app.update();
        assert!(app.world().resource::<HudSettings>().visible.is_none());
    }
}
