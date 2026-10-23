use super::common::RuneMode;
use super::consts::ROTATION_BASELINE_SMALL;
use crate::robomaster::prelude::Team;
use bevy::prelude::{Component, Dir3, Quat, Transform, Vec3};
use rand::{Rng, RngExt};

struct VariableRotation {
    a: f32,
    omega: f32,
    t: f64,
}
impl VariableRotation {
    fn random(rng: &mut impl Rng) -> Self {
        Self {
            a: rng.random_range(0.780..=1.045),
            omega: rng.random_range(1.884..=2.000),
            t: 0.0,
        }
    }
    fn speed(&self) -> f32 {
        self.a * (self.omega as f64 * self.t).sin() as f32 + 2.090 - self.a
    }
    fn advance_angle(&mut self, dt: f32) -> f32 {
        let t0 = self.t;
        self.t += dt as f64;
        let w = self.omega as f64;
        // Exact integral of a*sin(w*t)+(2.090-a), independent of render frame size.
        ((2.090 - self.a) as f64 * dt as f64
            + self.a as f64 / w * ((w * t0).cos() - (w * self.t).cos())) as f32
    }
}

pub struct RotationController {
    direction: Dir3,
    initial_rotation: Quat,
    variable: Option<VariableRotation>,
    clockwise: bool,
}
impl RotationController {
    fn for_face(clockwise: bool, team: Team, initial_rotation: Quat) -> Self {
        // Face normals point in opposite directions. Opposite apparent directions therefore
        // require the SAME physical axis rotation, rather than opposite world angular velocity.
        let outward = match team {
            Team::Red => Vec3::new(-1.0, 0.0, -1.0),
            Team::Blue => Vec3::new(1.0, 0.0, 1.0),
        }
        .normalize();
        Self {
            direction: Dir3::new(initial_rotation.inverse() * outward).unwrap(),
            initial_rotation,
            variable: None,
            clockwise,
        }
    }
    pub fn variable_params(&self) -> Option<(f32, f32, f32)> {
        self.variable.as_ref().map(|v| (v.a, v.omega, v.t as f32))
    }
    pub fn is_clockwise(&self) -> bool {
        self.clockwise
    }
    pub fn axis(&self) -> Dir3 {
        self.direction
    }
    pub fn current_speed(&self, mode: RuneMode) -> f32 {
        let magnitude = if mode == RuneMode::Large {
            self.variable
                .as_ref()
                .map_or(ROTATION_BASELINE_SMALL, VariableRotation::speed)
        } else {
            ROTATION_BASELINE_SMALL
        };
        if self.clockwise {
            magnitude
        } else {
            -magnitude
        }
    }
    pub fn current_angle(&self, transform: &Transform) -> f32 {
        let q = self.initial_rotation.inverse() * transform.rotation;
        // Positive means clockwise when looking at this face, as in the protocol direction flag.
        -2.0 * q.xyz().dot(*self.direction).atan2(q.w)
    }
    fn sync_activation(&mut self, mode: RuneMode, activating: bool, rng: &mut impl Rng) {
        match (mode, activating, self.variable.is_some()) {
            (RuneMode::Large, true, false) => self.variable = Some(VariableRotation::random(rng)),
            (RuneMode::Large, true, true) => {}
            _ => self.variable = None,
        }
    }
    fn rotate(&mut self, mode: RuneMode, transform: &mut Transform, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let angle = if mode == RuneMode::Large {
            self.variable
                .as_mut()
                .map_or(ROTATION_BASELINE_SMALL * dt, |v| v.advance_angle(dt))
        } else {
            ROTATION_BASELINE_SMALL * dt
        };
        // Positive right-handed rotation about a normal towards the viewer is counterclockwise.
        transform.rotate_local_axis(self.direction, if self.clockwise { -angle } else { angle });
        transform.rotation = transform.rotation.normalize();
    }
}
#[derive(Component)]
pub struct PowerRuneRotation {
    controller: RotationController,
}
impl PowerRuneRotation {
    pub fn for_face(clockwise: bool, team: Team, initial_rotation: Quat) -> Self {
        Self {
            controller: RotationController::for_face(clockwise, team, initial_rotation),
        }
    }
    pub fn controller(&self) -> &RotationController {
        &self.controller
    }
    pub fn sync_activation(&mut self, mode: RuneMode, activating: bool, rng: &mut impl Rng) {
        self.controller.sync_activation(mode, activating, rng);
    }
    pub fn rotate(&mut self, mode: RuneMode, transform: &mut Transform, dt: f32) {
        self.controller.rotate(mode, transform, dt);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_is_pi_over_three_in_small_and_inactive_large() {
        let c = RotationController::for_face(true, Team::Red, Quat::IDENTITY);
        assert_eq!(c.current_speed(RuneMode::Small), std::f32::consts::PI / 3.0);
        assert_eq!(c.current_speed(RuneMode::Large), std::f32::consts::PI / 3.0);
    }
    #[test]
    fn opposite_apparent_directions_share_world_rotation_for_both_direction_choices() {
        for red_cw in [false, true] {
            let qa = Quat::from_rotation_x(0.008);
            let qb = Quat::from_rotation_x(-0.008);
            let mut red = RotationController::for_face(red_cw, Team::Red, qa);
            let mut blue = RotationController::for_face(!red_cw, Team::Blue, qb);
            let mut a = Transform::from_rotation(qa);
            let mut b = Transform::from_rotation(qb);
            red.rotate(RuneMode::Small, &mut a, 0.4);
            blue.rotate(RuneMode::Small, &mut b, 0.4);
            assert!((a.rotation * qa.inverse()).angle_between(b.rotation * qb.inverse()) < 1e-5);
            assert!((red.current_angle(&a) + blue.current_angle(&b)).abs() < 1e-5);
        }
    }
    #[test]
    fn sine_parameters_reset_at_entry_and_persist_through_active_rounds() {
        let mut rng = rand::rng();
        let mut c = RotationController::for_face(false, Team::Red, Quat::IDENTITY);
        c.sync_activation(RuneMode::Large, true, &mut rng);
        let (a, w, t) = c.variable_params().unwrap();
        assert!((0.780..=1.045).contains(&a));
        assert!((1.884..=2.0).contains(&w));
        assert_eq!(t, 0.0);
        let mut transform = Transform::IDENTITY;
        c.rotate(RuneMode::Large, &mut transform, 1.0);
        c.sync_activation(RuneMode::Large, true, &mut rng);
        assert_eq!(c.variable_params().unwrap(), (a, w, 1.0));
        c.sync_activation(RuneMode::Large, false, &mut rng);
        assert!(c.variable_params().is_none());
        c.sync_activation(RuneMode::Large, true, &mut rng);
        assert_eq!(c.variable_params().unwrap().2, 0.0);
    }
    #[test]
    fn sine_displacement_is_the_integral_and_does_not_depend_on_frame_rate() {
        let mut a = RotationController::for_face(false, Team::Red, Quat::IDENTITY);
        let mut b = RotationController::for_face(false, Team::Red, Quat::IDENTITY);
        a.variable = Some(VariableRotation {
            a: 0.9,
            omega: 1.95,
            t: 0.0,
        });
        b.variable = Some(VariableRotation {
            a: 0.9,
            omega: 1.95,
            t: 0.0,
        });
        let mut x = Transform::IDENTITY;
        let mut y = x;
        a.rotate(RuneMode::Large, &mut x, 2.0);
        for _ in 0..200 {
            b.rotate(RuneMode::Large, &mut y, 0.01);
        }
        assert!(x.rotation.angle_between(y.rotation) < 1e-4);
        let expected = 1.19 * 2.0 + 0.9 / 1.95 * (1.0 - (1.95_f32 * 2.0).cos());
        assert!((a.current_angle(&x).abs() - expected).abs() < 1e-5);
    }
}
