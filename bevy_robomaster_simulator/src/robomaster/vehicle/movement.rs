use avian3d::prelude::forces::ForcesItem;
use avian3d::prelude::*;
use bevy::prelude::*;

// The chassis collider represents a wheeled vehicle, not a block sliding on its belly.
// Retain ordinary static friction when the driver releases the controls.
const DRIVEN_ROLLING_FRICTION: f32 = 0.2;
const LATERAL_GRIP_RESPONSE_SECS: f32 = 0.05;

#[derive(Component, Clone, Debug)]
pub struct VehicleDynamic {
    pub max_speed: f32,           // m/s
    pub linear_acceleration: f32, // m/s^2

    n: f32,
}

impl VehicleDynamic {
    pub fn new(max_speed: f32, linear_acceleration: f32, acceleration_exponent: f32) -> Self {
        Self {
            max_speed,
            linear_acceleration,
            n: acceleration_exponent,
        }
    }
}

impl Default for VehicleDynamic {
    fn default() -> Self {
        Self {
            max_speed: 4.0,
            linear_acceleration: 8.0,
            n: 10.0,
        }
    }
}

impl VehicleDynamic {
    pub fn linear(
        &mut self,
        forces: &mut ForcesItem,
        friction: &mut Friction,
        mass: f32,
        movement_frame: &GlobalTransform,
        input: Vec2,
        dt: f32,
        boost: f32,
    ) {
        *friction = if input.length_squared() > 0.0 {
            Friction::new(DRIVEN_ROLLING_FRICTION).with_combine_rule(CoefficientCombine::Min)
        } else {
            Friction::default()
        };
        let lin_vel = forces.linear_velocity();
        let acceleration = self.linear_accelerate(input, movement_frame, lin_vel, dt, boost);
        forces.apply_linear_impulse(acceleration * mass * dt);
    }

    fn linear_accelerate(
        &mut self,
        input: Vec2,
        movement_frame: &GlobalTransform,
        current_velocity: Vec3,
        dt: f32,
        boost: f32,
    ) -> Vec3 {
        if input.length_squared() == 0.0 {
            return Vec3::ZERO;
        }
        let forward = movement_frame.forward().with_y(0.0);
        let right = movement_frame.right().with_y(0.0);
        let forward_xz = forward.with_y(0.0).normalize_or_zero();
        let right_xz = right.with_y(0.0).normalize_or_zero();
        let dirc = (forward_xz * input.y + right_xz * input.x).normalize_or_zero();
        let max_speed = self.max_speed * boost;
        let accel = self.linear_acceleration * boost;
        let drive = dirc * accel * (1.0 - (current_velocity.length() / max_speed).powf(self.n));
        // A low rolling resistance alone also turns the chassis into a freely sliding puck.
        // Wheel traction resists lateral slip, with a finite force and without teleporting velocity.
        let horizontal_velocity = current_velocity.with_y(0.0);
        let lateral_velocity = horizontal_velocity - dirc * horizontal_velocity.dot(dirc);
        let grip = (-lateral_velocity / LATERAL_GRIP_RESPONSE_SECS.max(dt)).clamp_length_max(accel);
        drive + grip
    }
}
