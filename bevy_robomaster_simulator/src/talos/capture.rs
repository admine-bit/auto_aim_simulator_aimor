use crate::capture::{
    CameraFov, CaptureBundle, CaptureCamera, ImageHandle,
    driver::{
        CaptureConfig, CaptureFrameId, CapturedFrame, CapturedFrameKind, GpuCaptureHandler,
        SnapshotAsync, SnapshotSync,
    },
    setup_capture_camera, setup_preview_window, sync_capture_camera,
};
use crate::components::{Controlled, InfantryGimbal, InfantryLaunchOffset, SubscribeAutoAim};
use crate::systems::{ChassisObservationFrame, GameplaySystems};
use crate::talos::plugin::{M_ALIGN_MAT3, to_ros_quat, to_ros_translation};
use bevy::ecs::world::DeferredWorld;
use bevy::prelude::*;
use bevy::render::{Extract, ExtractSchedule, RenderApp, RenderSystems};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use talos_ipc::*;

static FRAME_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct TalosFrameStamp {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
}

pub fn advance_talos_frame_stamp(mut stamp: ResMut<TalosFrameStamp>) {
    stamp.frame_seq = FRAME_SEQ.fetch_add(1, Ordering::Relaxed);
    stamp.timestamp_ns = now_ns();
}

/// Extracted pose data from MainApp to RenderApp for synchronized publishing
#[derive(Resource, Clone, Default)]
pub struct ExtractedPoseData {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pose: Option<CapturedPoseData>,
    camera_info: Option<CameraInfo>,
    pub valid: bool,
}

/// Pose data captured at frame snapshot time
#[derive(Clone)]
struct CapturedPoseData {
    gimbal_ros: [f32; 3],
    gimbal_quat: [f32; 4],
    muzzle_rel: [f32; 3],
    camera_rel: [f32; 3],
    camera_quat: [f32; 4],
    chassis_observation: ChassisObservation,
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

struct TalosSnapshotSync {
    frame_seq: u64,
    timestamp_ns: u64,
    pose: CapturedPoseData,
    camera_info: CameraInfo,
}

impl SnapshotSync for TalosSnapshotSync {
    fn captured(
        self: Box<Self>,
        world: &mut DeferredWorld,
        _config: &CaptureConfig,
    ) -> Box<dyn SnapshotAsync> {
        let ctx = world.resource::<TalosCaptureContextShared>().0.clone();

        Box::new(TalosSnapshot {
            ctx,
            frame_seq: self.frame_seq,
            timestamp_ns: self.timestamp_ns,
            pose: self.pose,
            camera_info: self.camera_info,
        })
    }
}

struct TalosSnapshot {
    ctx: Arc<Mutex<ShmPublisher>>,
    frame_seq: u64,
    timestamp_ns: u64,
    pose: CapturedPoseData,
    camera_info: CameraInfo,
}

impl SnapshotAsync for TalosSnapshot {
    fn captured(&mut self, frame: CapturedFrame<'_>) {
        if frame.kind != CapturedFrameKind::Rgb8 {
            return;
        }

        let expected_size = (frame.width * frame.height * 3) as usize;
        if frame.data.len() != expected_size {
            warn!(
                "图像大小不匹配: expected {} bytes, got {} bytes",
                expected_size,
                frame.data.len()
            );
            return;
        }

        if frame.width != IMAGE_WIDTH || frame.height != IMAGE_HEIGHT {
            warn!(
                "image reesolution mismatched: expected {}x{}, got {}x{}",
                IMAGE_WIDTH, IMAGE_HEIGHT, frame.width, frame.height
            );
            return;
        }

        if let Ok(mut publisher) = self.ctx.lock() {
            let _ = publisher.try_publish_synchronized_image(
                frame.data,
                self.frame_seq,
                self.timestamp_ns,
                |publisher| {
                    publisher.set_camera_info(self.camera_info);
                    publish_pose_data(publisher, self.frame_seq, self.timestamp_ns, &self.pose);
                },
            );
        }
    }
}

#[derive(Default)]
struct TalosSnapshotCreator {}

impl GpuCaptureHandler for TalosSnapshotCreator {
    fn captured(
        &self,
        world: &World,
        _frame_id: Option<CaptureFrameId>,
    ) -> Option<Box<dyn SnapshotSync>> {
        // Timestamp, frame sequence and pose must come from the same ExtractSchedule snapshot.
        let extracted = world.get_resource::<ExtractedPoseData>()?;
        if !extracted.valid {
            return None;
        }
        let pose = extracted.pose.clone()?;
        let camera_info = extracted.camera_info?;

        Some(Box::new(TalosSnapshotSync {
            frame_seq: extracted.frame_seq,
            timestamp_ns: extracted.timestamp_ns,
            pose,
            camera_info,
        }))
    }
}

#[derive(Resource, Clone, Deref, DerefMut)]
pub struct TalosCaptureContextShared(pub Arc<Mutex<ShmPublisher>>);

#[derive(Resource, Clone)]
pub struct TalosCaptureContext {
    pub publisher: Arc<Mutex<ShmPublisher>>,
    pub fov_y: f32,
}

pub struct TalosCapturePlugin {
    pub config: CaptureConfig,
    pub context: TalosCaptureContext,
}

pub fn publish_talos_runtime_state_system(
    context: Option<Res<TalosCaptureContext>>,
    frame_stamp: Res<TalosFrameStamp>,
    following: Res<SubscribeAutoAim>,
) {
    let Some(ctx) = context else {
        return;
    };

    if let Ok(mut publisher) = ctx.publisher.lock() {
        publisher.publish_runtime_state(RuntimeState {
            timestamp_ns: frame_stamp.timestamp_ns,
            following: u8::from(following.load(Ordering::Acquire)),
            _pad: [0; 55],
        });
    }
}

impl Plugin for TalosCapturePlugin {
    fn build(&self, app: &mut App) {
        let capture = CaptureBundle::color(
            app,
            self.config.clone(),
            vec![Box::new(TalosSnapshotCreator::default())],
        );
        let render_target_handle = capture.color_target().unwrap().clone();

        app.add_plugins(capture)
            .insert_resource(ImageHandle(render_target_handle))
            .insert_resource(CameraFov(self.context.fov_y))
            .insert_resource(self.context.clone())
            .add_systems(Startup, setup_capture_camera)
            .add_systems(Startup, setup_preview_window)
            .add_systems(
                Update,
                sync_capture_camera
                    .after(GameplaySystems::Camera)
                    .before(RenderSystems::Render),
            );

        app.sub_app_mut(RenderApp)
            .insert_resource(TalosCaptureContextShared(self.context.publisher.clone()))
            .insert_resource(self.context.clone())
            .insert_resource(ExtractedPoseData::default())
            .add_systems(ExtractSchedule, extract_pose_data);
    }
}

/// Extract pose data from MainApp to RenderApp
fn extract_pose_data(
    mut pose_data: ResMut<ExtractedPoseData>,
    frame_stamp: Extract<Res<TalosFrameStamp>>,
    camera: Extract<Query<(&GlobalTransform, &Camera), With<CaptureCamera>>>,
    gimbal: Extract<Query<&GlobalTransform, (With<Controlled>, With<InfantryGimbal>)>>,
    muzzle_offset: Extract<Query<&GlobalTransform, (With<InfantryLaunchOffset>, With<Controlled>)>>,
    chassis_obs: Extract<Res<ChassisObservationFrame>>,
) {
    pose_data.frame_seq = frame_stamp.frame_seq;
    pose_data.timestamp_ns = frame_stamp.timestamp_ns;
    pose_data.camera_info = None;
    pose_data.valid = false;

    let Ok((cam_transform, capture_camera)) = camera.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };
    let Ok(gimbal_transform) = gimbal.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };
    let Ok(muzzle_global) = muzzle_offset.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };

    // Measure the camera that renders the IPC image, not the window/preview camera.
    let Some(size) = capture_camera.physical_viewport_size() else {
        return;
    };
    if size != UVec2::new(IMAGE_WIDTH, IMAGE_HEIGHT) {
        return;
    }
    let Some(camera_info) = camera_info_from_projection(
        capture_camera.clip_from_view(),
        size.x,
        size.y,
        frame_stamp.timestamp_ns,
    ) else {
        return;
    };
    pose_data.camera_info = Some(camera_info);

    pose_data.pose = Some(captured_pose_data(
        cam_transform,
        gimbal_transform,
        muzzle_global,
        &chassis_obs,
        pose_data.frame_seq,
        pose_data.timestamp_ns,
    ));
    pose_data.valid = true;
}

/// Intrinsics from the exact GPU projection, with OpenCV integer pixel centres.
/// No FOV, camera mounting, or rendering state is changed by this function.
fn camera_info_from_projection(
    clip: Mat4,
    width: u32,
    height: u32,
    timestamp_ns: u64,
) -> Option<CameraInfo> {
    if width == 0
        || height == 0
        || !clip.is_finite()
        || clip.w_axis.w.abs() > 1e-6
        || (clip.z_axis.w + 1.0).abs() > 1e-6
        || clip.x_axis.x <= 0.0
        || clip.y_axis.y <= 0.0
    {
        return None;
    }
    Some(CameraInfo {
        timestamp_ns,
        fx: width as f64 * clip.x_axis.x as f64 / 2.0,
        fy: height as f64 * clip.y_axis.y as f64 / 2.0,
        cx: width as f64 * (1.0 - clip.z_axis.x as f64) / 2.0 - 0.5,
        cy: height as f64 * (1.0 + clip.z_axis.y as f64) / 2.0 - 0.5,
        distortion: [0.0; 5],
        width,
        height,
        _pad: [0; 24],
    })
}

fn captured_pose_data(
    cam_transform: &GlobalTransform,
    gimbal_transform: &GlobalTransform,
    muzzle_global: &GlobalTransform,
    chassis_obs: &ChassisObservationFrame,
    frame_seq: u64,
    timestamp_ns: u64,
) -> CapturedPoseData {
    // Keep the existing wire convention: gimbal +X points along the actual barrel.
    // SHOT_DIRECTION launches along its +Y. Its GLOBAL rotation includes every
    // model parent/mount transform; no camera pose or commanded angle enters here.
    let gimbal_rot = (muzzle_global.rotation() * Quat::from_rotation_x(PI / 2.0)).normalize();
    let origin = gimbal_transform.translation();
    let world_to_gimbal = gimbal_rot.inverse();
    // Use world-space metres, not reparented_to's inverse scale/model units.
    let muzzle = to_ros_translation(world_to_gimbal * (muzzle_global.translation() - origin));
    let camera = to_ros_translation(world_to_gimbal * (cam_transform.translation() - origin));
    // OpenCV optical +X right/+Y down/+Z forward -> Bevy camera +X/-Y/-Z.
    let optical_to_bevy = Mat3::from_cols(Vec3::X, Vec3::NEG_Y, Vec3::NEG_Z);
    let camera_rotation = Quat::from_mat3(
        &(M_ALIGN_MAT3
            * Mat3::from_quat(world_to_gimbal * cam_transform.rotation())
            * optical_to_bevy),
    )
    .normalize();

    let gimbal_ros = to_ros_translation(gimbal_transform.translation());
    let gimbal_rot = to_ros_quat(gimbal_rot).normalize();

    CapturedPoseData {
        gimbal_ros: [gimbal_ros.x, gimbal_ros.y, gimbal_ros.z],
        gimbal_quat: [gimbal_rot.w, gimbal_rot.x, gimbal_rot.y, gimbal_rot.z],
        muzzle_rel: [muzzle.x, muzzle.y, muzzle.z],
        camera_rel: [camera.x, camera.y, camera.z],
        camera_quat: [
            camera_rotation.w,
            camera_rotation.x,
            camera_rotation.y,
            camera_rotation.z,
        ],
        chassis_observation: ChassisObservation {
            frame_seq,
            timestamp_ns,
            dt_s: chassis_obs.dt_s,
            v_body: [chassis_obs.v_body.x, chassis_obs.v_body.y],
            wz_radps: chassis_obs.wz_radps,
            wheel_linear_mps: chassis_obs.wheel_linear_mps,
            wheel_angular_radps: chassis_obs.wheel_angular_radps,
            a_body: [chassis_obs.a_body.x, chassis_obs.a_body.y],
            alpha_z_radps2: chassis_obs.alpha_z_radps2,
            rpy_rad: [
                chassis_obs.rpy_rad.x,
                chassis_obs.rpy_rad.y,
                chassis_obs.rpy_rad.z,
            ],
            gyro_xyz_radps: [
                chassis_obs.gyro_xyz_radps.x,
                chassis_obs.gyro_xyz_radps.y,
                chassis_obs.gyro_xyz_radps.z,
            ],
            accel_xyz_mps2: [
                chassis_obs.accel_xyz_mps2.x,
                chassis_obs.accel_xyz_mps2.y,
                chassis_obs.accel_xyz_mps2.z,
            ],
            _pad: [0; 16],
        },
    }
}

#[cfg(test)]
mod pose_tests {
    use super::*;

    fn quat(wxyz: [f32; 4]) -> Quat {
        Quat::from_xyzw(wxyz[1], wxyz[2], wxyz[3], wxyz[0])
    }

    fn sample(
        camera: &GlobalTransform,
        gimbal: &GlobalTransform,
        muzzle: &GlobalTransform,
    ) -> CapturedPoseData {
        captured_pose_data(
            camera,
            gimbal,
            muzzle,
            &ChassisObservationFrame::default(),
            17,
            1234,
        )
    }

    #[test]
    fn gimbal_forward_is_actual_barrel_not_camera_or_requested_angle() {
        let gimbal = GlobalTransform::from(Transform::from_xyz(1.0, 2.0, 3.0));
        for yaw in [-3.1, -1.7, 0.0, 1.7, 3.1] {
            for elevation in [-0.4, 0.0, 0.3] {
                let reference = Quat::from_rotation_y(-yaw)
                    * Quat::from_rotation_x(elevation)
                    * Quat::from_rotation_z(0.2);
                let muzzle = GlobalTransform::from(
                    Transform::from_xyz(1.1, 2.1, 3.1)
                        .with_rotation(reference * Quat::from_rotation_x(-PI / 2.0)),
                );
                let camera = GlobalTransform::from(
                    Transform::from_xyz(7.0, 8.0, 9.0).with_rotation(Quat::from_rotation_z(1.0)),
                );
                let pose = sample(&camera, &gimbal, &muzzle);
                let forward = quat(pose.gimbal_quat) * Vec3::X;
                let real_forward = to_ros_translation(muzzle.rotation() * Vec3::Y);
                assert!((forward - real_forward).length() < 1e-5);
                let other_camera = GlobalTransform::IDENTITY;
                let other = sample(&other_camera, &gimbal, &muzzle);
                assert_eq!(pose.gimbal_ros, other.gimbal_ros);
                assert_eq!(pose.gimbal_quat, other.gimbal_quat);
                assert!((quat(pose.gimbal_quat).length() - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn nonzero_camera_pose_reconstructs_world_points_and_distances() {
        let optical_to_bevy = Mat3::from_cols(Vec3::X, Vec3::NEG_Y, Vec3::NEG_Z);
        for yaw in [-2.8, 0.0, 2.8] {
            let body = GlobalTransform::from(
                Transform::from_xyz(2.0, 0.7, -3.0).with_rotation(Quat::from_euler(
                    EulerRot::YXZ,
                    yaw,
                    0.2,
                    -0.3,
                )),
            );
            let gimbal = body.mul_transform(
                Transform::from_xyz(0.1, 0.2, -0.03).with_rotation(Quat::from_euler(
                    EulerRot::YXZ,
                    0.4,
                    -0.1,
                    0.0,
                )),
            );
            // Include a nonidentity intermediate parent before the shot marker.
            let mount_parent = gimbal.mul_transform(
                Transform::from_xyz(0.02, 0.01, 0.0).with_rotation(Quat::from_rotation_z(0.06)),
            );
            let muzzle = mount_parent.mul_transform(
                Transform::from_xyz(0.0, 0.11, -0.01)
                    .with_rotation(Quat::from_rotation_x(-65.0_f32.to_radians())),
            );
            let camera = gimbal.mul_transform(
                Transform::from_xyz(-0.002, 0.196, -0.060).with_rotation(Quat::from_euler(
                    EulerRot::YXZ,
                    0.03,
                    0.4,
                    -0.02,
                )),
            );
            let before = (camera, gimbal, muzzle);
            let pose = sample(&camera, &gimbal, &muzzle);
            let r_gw = quat(pose.gimbal_quat);
            let r_cg = quat(pose.camera_quat);
            let origin = Vec3::from_array(pose.gimbal_ros);
            let t_cg = Vec3::from_array(pose.camera_rel);
            let t_mg = Vec3::from_array(pose.muzzle_rel);
            assert!(t_cg.length() > 0.1);
            assert!(
                (origin + r_gw * t_cg - to_ros_translation(camera.translation())).length() < 1e-5
            );
            assert!(
                (origin + r_gw * t_mg - to_ros_translation(muzzle.translation())).length() < 1e-5
            );
            for p_camera in [Vec3::new(0.2, -0.1, 2.0), Vec3::new(-0.7, 0.4, 7.0)] {
                let p_gimbal = r_cg * p_camera + t_cg;
                let recovered = origin + r_gw * p_gimbal;
                let expected = to_ros_translation(
                    camera.translation() + camera.rotation() * (optical_to_bevy * p_camera),
                );
                assert!((recovered - expected).length() < 2e-5);
                assert!((p_gimbal.length() - (expected - origin).length()).abs() < 2e-5);
            }
            assert_eq!(before, (camera, gimbal, muzzle));
            assert_eq!(pose.chassis_observation.frame_seq, 17);
            assert_eq!(pose.chassis_observation.timestamp_ns, 1234);
        }
    }

    #[test]
    fn offsets_are_world_metres_even_when_model_is_scaled() {
        let rotation = Quat::from_euler(EulerRot::YXZ, 0.9, 0.2, -0.1);
        let gimbal = GlobalTransform::from(
            Transform::from_xyz(1.0, 2.0, 3.0)
                .with_rotation(rotation)
                .with_scale(Vec3::splat(2.0)),
        );
        let muzzle = gimbal.mul_transform(
            Transform::from_xyz(0.0, 0.1, -0.01).with_rotation(Quat::from_rotation_x(-PI / 2.0)),
        );
        let camera = gimbal.mul_transform(Transform::from_xyz(0.0, 0.2, -0.05));
        let pose = sample(&camera, &gimbal, &muzzle);
        let expected = (camera.translation() - gimbal.translation()).length();
        assert!((Vec3::from_array(pose.camera_rel).length() - expected).abs() < 1e-6);
        assert!((expected - 2.0 * Vec3::new(0.0, 0.2, -0.05).length()).abs() < 1e-6);
    }

    #[test]
    fn published_intrinsics_match_render_projection_in_pixel_centres() {
        for fov in [30.0_f32, 45.0, 70.0] {
            let mut clip =
                Mat4::perspective_infinite_reverse_rh(fov.to_radians(), 1440.0 / 1080.0, 0.1);
            clip.z_axis.x = 0.03;
            clip.z_axis.y = -0.02;
            let info = camera_info_from_projection(clip, 1440, 1080, 1234).unwrap();
            for p in [Vec3::new(0.1, -0.2, -2.0), Vec3::new(-0.5, 0.3, -4.0)] {
                let ndc = clip.project_point3(p);
                let render_u = (ndc.x as f64 + 1.0) * 1440.0 / 2.0 - 0.5;
                let render_v = (1.0 - ndc.y as f64) * 1080.0 / 2.0 - 0.5;
                let cv_u = info.fx * p.x as f64 / -p.z as f64 + info.cx;
                let cv_v = info.fy * -p.y as f64 / -p.z as f64 + info.cy;
                assert!((render_u - cv_u).abs() < 1e-4);
                assert!((render_v - cv_v).abs() < 1e-4);
            }
            assert_eq!(info.timestamp_ns, 1234);
        }
    }

    #[test]
    fn invalid_projection_is_not_published_as_valid_calibration() {
        assert!(camera_info_from_projection(Mat4::IDENTITY, 1440, 1080, 0).is_none());
        let clip = Mat4::perspective_infinite_reverse_rh(PI / 4.0, 4.0 / 3.0, 0.1);
        assert!(camera_info_from_projection(clip, 0, 1080, 0).is_none());
        let mut invalid = clip;
        invalid.x_axis.x = f32::NAN;
        assert!(camera_info_from_projection(invalid, 1440, 1080, 0).is_none());
    }
}

fn publish_pose_data(
    publisher: &mut ShmPublisher,
    frame_seq: u64,
    timestamp_ns: u64,
    pose: &CapturedPoseData,
) {
    publisher.publish_pose(
        PoseIndex::Odom,
        pose.gimbal_ros,
        [1.0, 0.0, 0.0, 0.0],
        frame_seq,
        timestamp_ns,
    );

    publisher.publish_pose(
        PoseIndex::Gimbal,
        [0.0, 0.0, 0.0],
        pose.gimbal_quat,
        frame_seq,
        timestamp_ns,
    );

    publisher.publish_pose(
        PoseIndex::Muzzle,
        pose.muzzle_rel,
        [1.0, 0.0, 0.0, 0.0],
        frame_seq,
        timestamp_ns,
    );

    publisher.publish_pose(
        PoseIndex::Camera,
        pose.camera_rel,
        pose.camera_quat,
        frame_seq,
        timestamp_ns,
    );

    let mut observation = pose.chassis_observation;
    observation.frame_seq = frame_seq;
    observation.timestamp_ns = timestamp_ns;
    publisher.publish_chassis_observation(observation);

    // Legacy compatibility path for consumers still reading pose slot 4.
    publisher.publish_pose_with_aux(
        PoseIndex::ChassisObservation,
        [
            observation.v_body[0],
            observation.v_body[1],
            observation.wz_radps,
        ],
        observation.wheel_angular_radps,
        [
            observation.a_body[0],
            observation.a_body[1],
            observation.alpha_z_radps2,
            observation.dt_s,
        ],
        frame_seq,
        timestamp_ns,
    );
}
