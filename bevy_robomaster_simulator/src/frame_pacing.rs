use crate::config::SimulationConfig;
use bevy::prelude::*;
use bevy::time::{TimeSystems, TimeUpdateStrategy};
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

pub struct FramePacingPlugin;

impl Plugin for FramePacingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FramePacer>()
            .add_systems(First, pace_frame.before(TimeSystems));

        // Opt-in buffered diagnostics; normal runs do no per-frame file I/O.
        if let Some(path) = std::env::var_os("DAEDALUS_FRAME_TIMING_LOG") {
            match std::fs::File::create(&path) {
                Ok(file) => {
                    app.insert_resource(FrameTimingLog {
                        writer: BufWriter::new(file),
                        last: Instant::now(),
                    })
                    .add_systems(First, trace_frame_timing.after(TimeSystems));
                }
                Err(error) => warn!("Cannot create frame timing log {path:?}: {error}"),
            }
        }
    }
}

#[derive(Resource, Default)]
struct FramePacer {
    deadline: Option<Instant>,
    period: Option<Duration>,
}

fn next_frame_deadline(deadline: Option<Instant>, now: Instant, period: Duration) -> Instant {
    // Absorb small scheduler oversleeps without accumulating drift. After a slow
    // frame, start a fresh period instead of rendering a catch-up burst.
    deadline
        .filter(|deadline| now.saturating_duration_since(*deadline) <= period / 8)
        .map(|deadline| deadline + period)
        .unwrap_or(now + period)
}

fn pace_frame(
    config: Res<SimulationConfig>,
    mut pacer: ResMut<FramePacer>,
    mut time_strategy: ResMut<TimeUpdateStrategy>,
) {
    let period = (config.window.max_fps != 0)
        .then(|| Duration::from_secs_f64(1.0 / f64::from(config.window.max_fps)));
    if pacer.period != period {
        pacer.deadline = None;
        pacer.period = period;
    }
    if let Some(deadline) = pacer.deadline {
        let now = Instant::now();
        if deadline > now {
            std::thread::sleep(deadline - now);
        }
    }
    let now = Instant::now();
    pacer.deadline = period.map(|period| next_frame_deadline(pacer.deadline, now, period));

    // Pipelined rendering sends the previous frame's *completion* time. Using
    // that as simulation time turns variable render cost into alternating long
    // and short motion steps, even when this loop starts at a steady cadence.
    // Sample the real clock after pacing, including when the cap is disabled.
    // Bevy still drains its render time channel and runs its normal fixed loop.
    *time_strategy = TimeUpdateStrategy::ManualInstant(now);
}

#[derive(Resource)]
struct FrameTimingLog {
    writer: BufWriter<std::fs::File>,
    last: Instant,
}

fn trace_frame_timing(time: Res<Time>, mut log: ResMut<FrameTimingLog>) {
    let now = Instant::now();
    let interval_ms = now.duration_since(log.last).as_secs_f64() * 1000.0;
    let _ = writeln!(
        log.writer,
        "{},{},{}",
        time.elapsed_secs_f64(),
        time.delta_secs_f64() * 1000.0,
        interval_ms,
    );
    log.last = now;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::time::{TimePlugin, create_time_channels};

    #[test]
    fn scheduler_oversleep_does_not_accumulate_into_deadline_drift() {
        let origin = Instant::now();
        let period = Duration::from_millis(10);
        let mut deadline = origin + period;
        for tick in 1..=1000 {
            let actual = deadline + Duration::from_micros(100);
            deadline = next_frame_deadline(Some(deadline), actual, period);
            assert_eq!(deadline, origin + period * (tick + 1));
        }
    }

    #[test]
    fn slow_frames_restart_without_short_catch_up_intervals() {
        let deadline = Instant::now();
        let period = Duration::from_millis(10);
        for delay_ms in [2, 9, 35, 100] {
            let now = deadline + Duration::from_millis(delay_ms);
            assert_eq!(next_frame_deadline(Some(deadline), now, period), now + period);
        }
        assert_eq!(next_frame_deadline(None, deadline, period), deadline + period);
    }

    #[test]
    fn simulation_samples_current_clock_despite_stale_render_completion() {
        for max_fps in [0, 1000] {
            let mut app = App::new();
            app.add_plugins(TimePlugin);
            let mut config = SimulationConfig::default();
            config.window.max_fps = max_fps;
            app.insert_resource(config)
                .init_resource::<FramePacer>()
                .add_systems(First, pace_frame.before(TimeSystems));
            let (sender, receiver) = create_time_channels();
            app.insert_resource(receiver);
            let stale_render_time = Instant::now();
            for tick in 0..3 {
                sender.0.send(stale_render_time + Duration::from_nanos(tick)).unwrap();
                let before = Instant::now();
                app.update();
                let after = Instant::now();
                let sampled = app.world().resource::<Time<Real>>().last_update().unwrap();
                assert!(sampled >= before && sampled <= after);
            }
        }
    }

    #[test]
    fn disabling_cap_discards_previous_deadline() {
        let mut app = App::new();
        app.add_plugins(TimePlugin);
        let mut config = SimulationConfig::default();
        config.window.max_fps = 0;
        app.insert_resource(config)
            .insert_resource(FramePacer {
                deadline: Some(Instant::now() + Duration::from_secs(60)),
                period: Some(Duration::from_millis(10)),
            })
            .add_systems(First, pace_frame.before(TimeSystems));
        app.update();
        let pacer = app.world().resource::<FramePacer>();
        assert!(pacer.deadline.is_none() && pacer.period.is_none());
    }
}
