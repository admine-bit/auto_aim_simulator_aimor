//! Local, collision-free trajectory estimates. Nothing here controls the gimbal or projectile.
use bevy::prelude::*;
use std::f32::consts::PI;

use crate::config::SimulationConfig;

pub(super) struct BallisticModel {
    pub speed: f32,
    pub gravity: Vec3,
    pub inherited_velocity: Vec3,
    lifetime: f32,
    dt: f32,
    substeps: usize,
    damping: f32,
    drag_per_mass: f32,
    wind: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Flight {
    pub position: Vec3,
    pub time: f32,
}

#[derive(Debug)]
pub(super) struct AimSolution {
    pub direction: Vec3,
    pub flight: Flight,
    pub raise_m: f32,
    pub elevation_delta: f32,
}

impl BallisticModel {
    pub fn new(config: &SimulationConfig, gravity: Vec3, inherited_velocity: Vec3) -> Self {
        let projectile = &config.projectile;
        let aero = &projectile.aerodynamics;
        let substeps = config.physics.substep_count.clamp(1, 64) as usize;
        let lifetime = projectile.lifetime.clamp(0.0, 30.0);
        Self {
            speed: projectile.speed,
            gravity,
            inherited_velocity,
            lifetime,
            // Match Avian's substep integration; bound work for extreme user configurations.
            dt: (1.0 / config.physics.fixed_hz.max(1.0) as f32 / substeps as f32)
                .max(lifetime / 20_000.0),
            substeps,
            damping: projectile.linear_damping.max(0.0),
            drag_per_mass: if aero.enabled && projectile.mass > 0.0 {
                0.5 * aero.air_density.max(0.0)
                    * aero.drag_coefficient.max(0.0)
                    * PI
                    * (projectile.diameter.max(0.0) * 0.5).powi(2)
                    / projectile.mass
            } else {
                0.0
            },
            wind: Vec3::from_array(aero.wind),
        }
    }

    /// Intersect a vertical target plane. Linear damping precedes acceleration in Avian;
    /// aerodynamic force is sampled once per fixed tick, as in projectile_aerodynamics.
    pub fn flight_to_plane(
        &self,
        origin: Vec3,
        direction: Vec3,
        normal: Vec3,
        distance: f32,
    ) -> Option<Flight> {
        if self.speed <= 0.0 || distance <= 0.0 || !distance.is_finite() {
            return None;
        }
        let mut position = origin;
        let mut velocity = direction * self.speed + self.inherited_velocity;
        let mut drag = Vec3::ZERO;
        let mut time = 0.0;
        let mut step = 0;
        while time < self.lifetime {
            if step % self.substeps == 0 {
                let relative = velocity - self.wind;
                drag = -self.drag_per_mass * relative.length() * relative;
            }
            let dt = self.dt.min(self.lifetime - time);
            let before = position;
            velocity = velocity / (1.0 + dt * self.damping) + (self.gravity + drag) * dt;
            position += velocity * dt;
            let progress = (position - origin).dot(normal);
            if !position.is_finite() || !velocity.is_finite() {
                return None;
            }
            if progress >= distance {
                let previous = (before - origin).dot(normal);
                let fraction = (distance - previous) / (progress - previous);
                return Some(Flight {
                    position: before.lerp(position, fraction),
                    time: time + fraction * dt,
                });
            }
            time += dt;
            step += 1;
        }
        None
    }

    /// Solve both elevation and yaw for a stationary point, including camera/muzzle parallax,
    /// inherited chassis velocity and wind. Newton derivatives come from the same flight model.
    pub fn solve(&self, origin: Vec3, target: Vec3) -> Option<AimSolution> {
        let delta = target - origin;
        let horizontal = Vec3::new(delta.x, 0.0, delta.z);
        let distance = horizontal.length();
        if distance < 0.05 || !distance.is_finite() {
            return None;
        }
        let normal = horizontal / distance;
        let right = normal.cross(Vec3::Y);
        let direct_elevation = delta.y.atan2(distance);
        let mut elevation = direct_elevation;
        let mut yaw = 0.0;
        let direction = |elevation: f32, yaw: f32| {
            (normal * yaw.cos() + right * yaw.sin()) * elevation.cos() + Vec3::Y * elevation.sin()
        };
        let evaluate = |elevation: f32, yaw: f32| -> Option<(Flight, Vec2)> {
            let flight =
                self.flight_to_plane(origin, direction(elevation, yaw), normal, distance)?;
            let error = flight.position - target;
            Some((flight, Vec2::new(error.dot(right), error.y)))
        };
        for _ in 0..12 {
            let (flight, error) = evaluate(elevation, yaw)?;
            if error.length() < 0.001 {
                return Some(AimSolution {
                    direction: direction(elevation, yaw),
                    flight,
                    raise_m: distance * (elevation.tan() / yaw.cos() - direct_elevation.tan()),
                    elevation_delta: elevation - direct_elevation,
                });
            }
            const EPS: f32 = 0.001;
            let dp = (evaluate(elevation + EPS, yaw)?.1 - error) / EPS;
            let dy = (evaluate(elevation, yaw + EPS)?.1 - error) / EPS;
            let determinant = dp.x * dy.y - dy.x * dp.y;
            if !determinant.is_finite() || determinant.abs() < 1e-5 {
                return None;
            }
            elevation -= ((error.x * dy.y - dy.x * error.y) / determinant).clamp(-0.2, 0.2);
            yaw -= ((dp.x * error.y - error.x * dp.y) / determinant).clamp(-0.2, 0.2);
            if elevation.abs() > 85.0_f32.to_radians() || yaw.abs() > PI / 3.0 {
                return None;
            }
        }
        None
    }
}

/// Project through the actual preview lens. Captured textures stretch to fill the window,
/// so use the capture aspect ratio rather than assuming that it equals the window aspect.
pub(super) fn project_to_preview(
    camera: &Transform,
    point: Vec3,
    fov_y: f32,
    image_aspect: f32,
) -> Option<Vec2> {
    let local = camera.rotation.inverse() * (point - camera.translation);
    if local.z >= -0.01 || !local.is_finite() {
        return None;
    }
    let half_height = -local.z * (fov_y * 0.5).tan();
    let p = Vec2::new(
        0.5 + local.x / (2.0 * half_height * image_aspect),
        0.5 - local.y / (2.0 * half_height),
    );
    (p.is_finite() && p.cmpge(Vec2::ZERO).all() && p.cmple(Vec2::ONE).all()).then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SimulationConfig {
        let mut config = SimulationConfig::default();
        config.projectile.speed = 25.0;
        config.projectile.lifetime = 5.0;
        config.projectile.linear_damping = 0.0;
        config.projectile.aerodynamics.enabled = false;
        config.physics.fixed_hz = 120.0;
        config.physics.substep_count = 8;
        config
    }

    #[test]
    fn level_shot_matches_gravity_and_low_arc_solution() {
        let model = BallisticModel::new(&config(), Vec3::NEG_Y * 9.81, Vec3::ZERO);
        for distance in [3.0, 5.0, 10.0, 20.0] {
            let hit = model
                .flight_to_plane(Vec3::ZERO, Vec3::NEG_Z, Vec3::NEG_Z, distance)
                .unwrap();
            let time = distance / 25.0;
            assert!((hit.time - time).abs() < 0.0001);
            // Semi-implicit Euler differs from the continuous parabola by g*t*dt/2.
            assert!((hit.position.y + 0.5 * 9.81 * time * time).abs() < 0.005);
            let aim = model.solve(Vec3::ZERO, Vec3::NEG_Z * distance).unwrap();
            let expected = 0.5 * (9.81 * distance / 625.0).asin();
            assert!((aim.elevation_delta - expected).abs() < 0.0005);
            assert!((aim.flight.position - Vec3::NEG_Z * distance).length() < 0.001);
            assert!(aim.raise_m > 0.0);
        }
    }

    #[test]
    fn drag_damping_and_crosswind_are_included() {
        let mut config = config();
        config.projectile.mass = 0.0032;
        config.projectile.diameter = 0.017;
        let plain = BallisticModel::new(&config, Vec3::NEG_Y * 9.81, Vec3::ZERO)
            .solve(Vec3::ZERO, Vec3::NEG_Z * 10.0)
            .unwrap();
        config.projectile.aerodynamics.enabled = true;
        config.projectile.aerodynamics.wind = [5.0, 0.0, 0.0];
        config.projectile.linear_damping = 0.1;
        let model = BallisticModel::new(&config, Vec3::NEG_Y * 9.81, Vec3::ZERO);
        let aim = model.solve(Vec3::ZERO, Vec3::NEG_Z * 10.0).unwrap();
        assert!(aim.raise_m > plain.raise_m);
        assert!(aim.flight.time > plain.flight.time);
        assert!(aim.direction.x < 0.0);
        assert!((aim.flight.position - Vec3::NEG_Z * 10.0).length() < 0.001);
    }

    #[test]
    fn camera_parallax_moving_chassis_and_elevated_target_are_included() {
        let model = BallisticModel::new(&config(), Vec3::NEG_Y * 9.81, Vec3::new(2.0, 0.0, 1.0));
        let origin = Vec3::new(0.1, 0.3, 0.2);
        for height in [-1.0, 0.0, 3.0] {
            let target = Vec3::new(0.0, height, -5.0);
            let aim = model.solve(origin, target).unwrap();
            assert!((aim.flight.position - target).length() < 0.001);
            assert!(aim.direction.x < 0.0);
            let plane_normal = Vec3::new(target.x - origin.x, 0.0, target.z - origin.z).normalize();
            let distance = (target - origin).dot(plane_normal);
            let aim_point = origin + aim.direction * distance / aim.direction.dot(plane_normal);
            assert!((aim.raise_m - (aim_point.y - target.y)).abs() < 1e-5);
        }
    }

    #[test]
    fn unreachable_and_invalid_shots_do_not_report_zero_compensation() {
        let model = BallisticModel::new(&config(), Vec3::NEG_Y * 9.81, Vec3::ZERO);
        assert!(model.solve(Vec3::ZERO, Vec3::NEG_Z * 100.0).is_none());
        assert!(model.solve(Vec3::ZERO, Vec3::Y).is_none());
        assert!(
            model
                .flight_to_plane(Vec3::ZERO, Vec3::Z, Vec3::NEG_Z, 5.0)
                .is_none()
        );
        let mut config = config();
        config.projectile.speed = 0.0;
        assert!(
            BallisticModel::new(&config, Vec3::ZERO, Vec3::ZERO)
                .solve(Vec3::ZERO, Vec3::NEG_Z * 5.0)
                .is_none()
        );
    }

    #[test]
    fn preview_projection_handles_capture_aspect_and_points_behind_camera() {
        let camera = Transform::IDENTITY;
        let fov = 45.0_f32.to_radians();
        assert_eq!(
            project_to_preview(&camera, Vec3::NEG_Z, fov, 4.0 / 3.0),
            Some(Vec2::splat(0.5))
        );
        let up = project_to_preview(&camera, Vec3::new(0.0, 0.1, -1.0), fov, 4.0 / 3.0).unwrap();
        assert!(up.y < 0.5);
        let wide = project_to_preview(&camera, Vec3::new(0.1, 0.0, -1.0), fov, 16.0 / 9.0).unwrap();
        let narrow =
            project_to_preview(&camera, Vec3::new(0.1, 0.0, -1.0), fov, 4.0 / 3.0).unwrap();
        assert!(wide.x < narrow.x);
        assert!(project_to_preview(&camera, Vec3::Z, fov, 1.0).is_none());
    }
}
