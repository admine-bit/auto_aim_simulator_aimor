use super::common::{RUNE_TARGET_COUNT, RuneHitOutcome, RuneMode, RuneTransition};
use super::consts::{
    ACTIVATED_HOLD, ACTIVATION_GLOBAL_TIMEOUT, ACTIVATION_PRIMARY_TIMEOUT, FAILURE_RECOVER,
    INACTIVE_WAIT, LARGE_SECONDARY_TIMEOUT, ROUND_BLACKOUT,
};
use crate::robomaster::visibility::Activation;
use rand::{Rng, prelude::SliceRandom};

pub type RuneTargetStates = [Activation; RUNE_TARGET_COUNT];
const EPS: f32 = 1e-5;

/// Standalone practice: no match clock, opportunities, stages, buffs or economy.
#[derive(Debug, Clone, PartialEq)]
pub enum MechanismState {
    Inactive {
        mode: RuneMode,
        remaining: f32,
    },
    Activating(ActivationRun),
    Activated {
        mode: RuneMode,
        remaining: f32,
        hits: u8,
    },
    Failed {
        mode: RuneMode,
        remaining: f32,
        activation_remaining: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActivationRun {
    mode: RuneMode,
    global_remaining: f32,
    targets: RuneTargetStates,
    phase: Phase,
    blackout_remaining: f32,
    completed_groups: usize,
    hits: u8,
}

#[derive(Debug, Clone, PartialEq)]
enum Phase {
    Primary { remaining: f32 },
    Secondary { remaining: f32, target: usize },
}

impl MechanismState {
    pub fn inactive(mode: RuneMode) -> Self {
        Self::Inactive {
            mode,
            remaining: INACTIVE_WAIT,
        }
    }
    pub fn start(mode: RuneMode, rng: &mut impl Rng) -> Self {
        Self::Activating(ActivationRun::new(mode, ACTIVATION_GLOBAL_TIMEOUT, rng))
    }
    pub fn mode(&self) -> RuneMode {
        match self {
            Self::Inactive { mode, .. }
            | Self::Activated { mode, .. }
            | Self::Failed { mode, .. } => *mode,
            Self::Activating(run) => run.mode,
        }
    }
    /// Split long frames at local/global deadlines; dark time never steals target time.
    pub fn tick(&mut self, delta_secs: f32, rng: &mut impl Rng) -> RuneTransition {
        if !delta_secs.is_finite() || delta_secs <= 0.0 {
            return RuneTransition::None;
        }
        let mut left = delta_secs;
        let mut result = RuneTransition::None;
        while left > EPS {
            let dt = left.min(self.next_deadline().max(0.0));
            let transition = self.step(dt, rng);
            left -= dt;
            if transition != RuneTransition::None {
                result = transition;
            }
            if dt <= EPS && transition == RuneTransition::None {
                break;
            }
        }
        result
    }
    fn next_deadline(&self) -> f32 {
        match self {
            Self::Inactive { remaining, .. } | Self::Activated { remaining, .. } => *remaining,
            Self::Failed {
                remaining,
                activation_remaining,
                ..
            } => remaining.min(*activation_remaining),
            Self::Activating(run) => {
                let local = if run.blackout_remaining > EPS {
                    run.blackout_remaining
                } else {
                    match run.phase {
                        Phase::Primary { remaining } | Phase::Secondary { remaining, .. } => {
                            remaining
                        }
                    }
                };
                run.global_remaining.min(local)
            }
        }
    }
    fn step(&mut self, dt: f32, rng: &mut impl Rng) -> RuneTransition {
        let mode = self.mode();
        match self {
            Self::Inactive { remaining, .. } => {
                *remaining -= dt;
                if *remaining <= EPS {
                    *self = Self::start(mode, rng);
                    return RuneTransition::Started;
                }
            }
            Self::Activated { remaining, .. } => {
                *remaining -= dt;
                if *remaining <= EPS {
                    *self = Self::inactive(mode);
                    return RuneTransition::ResetToInactive;
                }
            }
            Self::Failed {
                remaining,
                activation_remaining,
                ..
            } => {
                *remaining -= dt;
                *activation_remaining -= dt;
                if *activation_remaining <= EPS {
                    *self = Self::inactive(mode);
                    return RuneTransition::ResetToInactive;
                }
                if *remaining <= EPS {
                    *self = Self::Activating(ActivationRun::new(mode, *activation_remaining, rng));
                    return RuneTransition::Started;
                }
            }
            Self::Activating(run) => {
                run.global_remaining -= dt;
                if run.global_remaining <= EPS {
                    *self = Self::inactive(mode);
                    return RuneTransition::ResetToInactive;
                }
                if run.blackout_remaining > EPS {
                    run.blackout_remaining = (run.blackout_remaining - dt).max(0.0);
                    if run.blackout_remaining <= EPS {
                        return RuneTransition::Advanced;
                    }
                } else {
                    let timer = match &mut run.phase {
                        Phase::Primary { remaining } | Phase::Secondary { remaining, .. } => {
                            remaining
                        }
                    };
                    *timer -= dt;
                    if *timer <= EPS {
                        match run.phase {
                            Phase::Primary { .. } => {
                                *self = Self::Failed {
                                    mode,
                                    remaining: FAILURE_RECOVER,
                                    activation_remaining: run.global_remaining,
                                };
                                return RuneTransition::Failed;
                            }
                            Phase::Secondary { .. }
                                if run.completed_groups == RUNE_TARGET_COUNT =>
                            {
                                *self = Self::Activated {
                                    mode,
                                    remaining: ACTIVATED_HOLD,
                                    hits: run.hits,
                                };
                                return RuneTransition::Activated;
                            }
                            Phase::Secondary { .. } => {
                                run.select_round(rng);
                                run.blackout_remaining = ROUND_BLACKOUT;
                                return RuneTransition::Advanced;
                            }
                        }
                    }
                }
            }
        }
        RuneTransition::None
    }
    pub fn hit(&mut self, index: usize, rng: &mut impl Rng) -> RuneHitOutcome {
        let Self::Activating(run) = self else {
            return RuneHitOutcome::Ignored;
        };
        // The old visible meshes still physically exist during the transition.
        if run.blackout_remaining > EPS {
            return RuneHitOutcome::Ignored;
        }
        if index >= RUNE_TARGET_COUNT || run.targets[index] != Activation::Activating {
            *self = Self::Failed {
                mode: run.mode,
                remaining: FAILURE_RECOVER,
                activation_remaining: run.global_remaining,
            };
            return RuneHitOutcome::WrongTarget;
        }
        run.targets[index] = Activation::Activated;
        run.hits += 1;
        match (run.mode, &run.phase) {
            (RuneMode::Small, _) => {
                run.completed_groups += 1;
                if run.completed_groups == RUNE_TARGET_COUNT {
                    *self = Self::Activated {
                        mode: RuneMode::Small,
                        remaining: ACTIVATED_HOLD,
                        hits: run.hits,
                    };
                    RuneHitOutcome::Activated
                } else {
                    // Small-rune progress stays illuminated while the next target lights up.
                    run.select_round(rng);
                    RuneHitOutcome::PrimaryHit
                }
            }
            (RuneMode::Large, Phase::Primary { .. }) => {
                let other = run
                    .targets
                    .iter()
                    .position(|a| *a == Activation::Activating)
                    .expect("large round has two targets");
                run.completed_groups += 1;
                // The fifth group also gets the full final one-second secondary window.
                run.phase = Phase::Secondary {
                    remaining: LARGE_SECONDARY_TIMEOUT,
                    target: other,
                };
                RuneHitOutcome::PrimaryHit
            }
            (RuneMode::Large, Phase::Secondary { target, .. }) => {
                debug_assert_eq!(*target, index);
                RuneHitOutcome::SecondaryHit
            }
        }
    }
    pub fn is_activating(&self) -> bool {
        matches!(self, Self::Activating(_))
    }
    pub fn is_activating_large(&self) -> bool {
        self.is_activating() && self.mode() == RuneMode::Large
    }
    pub fn large_progress(&self) -> Option<usize> {
        match self {
            Self::Activating(run) if run.mode == RuneMode::Large => Some(match run.phase {
                Phase::Secondary { .. } => run.completed_groups.saturating_sub(1),
                Phase::Primary { .. } => run.completed_groups,
            }),
            _ => None,
        }
    }
    pub fn target_states(&self) -> RuneTargetStates {
        match self {
            Self::Activating(run) if run.blackout_remaining <= EPS => run.targets,
            Self::Activated { .. } => [Activation::Completed; RUNE_TARGET_COUNT],
            _ => [Activation::Deactivated; RUNE_TARGET_COUNT],
        }
    }
}

impl ActivationRun {
    fn new(mode: RuneMode, remaining: f32, rng: &mut impl Rng) -> Self {
        let mut run = Self {
            mode,
            global_remaining: remaining,
            targets: [Activation::Deactivated; RUNE_TARGET_COUNT],
            phase: Phase::Primary {
                remaining: ACTIVATION_PRIMARY_TIMEOUT,
            },
            blackout_remaining: 0.0,
            completed_groups: 0,
            hits: 0,
        };
        run.select_round(rng);
        run
    }
    fn select_round(&mut self, rng: &mut impl Rng) {
        if self.mode == RuneMode::Large {
            self.targets.fill(Activation::Deactivated);
        }
        let mut available: Vec<usize> = self
            .targets
            .iter()
            .enumerate()
            .filter_map(|(i, a)| (*a != Activation::Activated).then_some(i))
            .collect();
        available.shuffle(rng);
        for i in available
            .into_iter()
            .take(if self.mode == RuneMode::Small { 1 } else { 2 })
        {
            self.targets[i] = Activation::Activating;
        }
        self.phase = Phase::Primary {
            remaining: ACTIVATION_PRIMARY_TIMEOUT,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn active(s: &MechanismState) -> Vec<usize> {
        s.target_states()
            .iter()
            .enumerate()
            .filter_map(|(i, a)| (*a == Activation::Activating).then_some(i))
            .collect()
    }
    #[test]
    fn small_keeps_completed_blades_lit_when_the_next_target_lights_up() {
        let mut rng = rand::rng();
        let mut s = MechanismState::start(RuneMode::Small, &mut rng);
        let mut seen = Vec::new();
        for n in 0..5 {
            assert_eq!(active(&s).len(), 1);
            let i = active(&s)[0];
            assert!(!seen.contains(&i));
            seen.push(i);
            let result = s.hit(i, &mut rng);
            if n < 4 {
                assert_eq!(result, RuneHitOutcome::PrimaryHit);
                let next = active(&s);
                assert_eq!(next.len(), 1);
                assert!(!seen.contains(&next[0]));
                // Both the hit blade and the next target stay lit throughout the old blackout.
                for frame in 0..=20 {
                    assert_eq!(active(&s), next);
                    for &completed in &seen {
                        assert_eq!(s.target_states()[completed], Activation::Activated);
                    }
                    if frame < 20 {
                        s.tick(0.01, &mut rng);
                    }
                }
            } else {
                assert_eq!(result, RuneHitOutcome::Activated);
            }
        }
        assert!(matches!(s, MechanismState::Activated { hits: 5, .. }));
        assert_eq!(s.target_states(), [Activation::Completed; RUNE_TARGET_COUNT]);
    }
    #[test]
    fn small_next_target_timeout_starts_at_the_hit_and_clears_progress() {
        let mut rng = rand::rng();
        let mut s = MechanismState::start(RuneMode::Small, &mut rng);
        s.tick(1.0, &mut rng);
        let first = active(&s)[0];
        assert_eq!(s.hit(first, &mut rng), RuneHitOutcome::PrimaryHit);
        s.tick(ACTIVATION_PRIMARY_TIMEOUT - 0.01, &mut rng);
        assert_eq!(active(&s).len(), 1);
        assert_eq!(s.target_states()[first], Activation::Activated);
        assert_eq!(s.tick(0.01, &mut rng), RuneTransition::Failed);
        assert_eq!(s.target_states(), [Activation::Deactivated; RUNE_TARGET_COUNT]);
        s.tick(FAILURE_RECOVER, &mut rng);
        assert_eq!(active(&s).len(), 1);
        assert!(!s.target_states().contains(&Activation::Activated));
    }
    #[test]
    fn wrong_target_and_primary_timeout_restart_without_extending_twenty_seconds() {
        let mut rng = rand::rng();
        for mode in [RuneMode::Small, RuneMode::Large] {
            let mut s = MechanismState::start(mode, &mut rng);
            s.tick(1.0, &mut rng);
            let i = (0..5).find(|i| !active(&s).contains(i)).unwrap();
            assert_eq!(s.hit(i, &mut rng), RuneHitOutcome::WrongTarget);
            assert!(matches!(s, MechanismState::Failed { .. }));
            s.tick(FAILURE_RECOVER, &mut rng);
            assert!(s.is_activating());
            s.tick(ACTIVATION_PRIMARY_TIMEOUT, &mut rng);
            assert!(matches!(s, MechanismState::Failed { .. }));
            s.tick(
                20.0 - 1.0 - FAILURE_RECOVER - ACTIVATION_PRIMARY_TIMEOUT,
                &mut rng,
            );
            assert!(matches!(s, MechanismState::Inactive { .. }));
        }
    }
    #[test]
    fn large_final_group_preserves_secondary_window_and_counts_ten_hits() {
        let mut rng = rand::rng();
        let mut s = MechanismState::start(RuneMode::Large, &mut rng);
        for group in 0..5 {
            let a = active(&s);
            assert_eq!(a.len(), 2);
            assert_eq!(s.hit(a[0], &mut rng), RuneHitOutcome::PrimaryHit);
            assert_eq!(s.large_progress(), Some(group));
            assert_eq!(s.hit(a[1], &mut rng), RuneHitOutcome::SecondaryHit);
            assert!(s.is_activating_large());
            s.tick(0.99, &mut rng);
            assert!(s.is_activating_large());
            let transition = s.tick(0.01, &mut rng);
            if group < 4 {
                assert_eq!(transition, RuneTransition::Advanced);
                assert!(active(&s).is_empty());
                s.tick(ROUND_BLACKOUT, &mut rng);
                assert_eq!(s.large_progress(), Some(group + 1));
            } else {
                assert_eq!(transition, RuneTransition::Activated);
            }
        }
        assert!(matches!(s, MechanismState::Activated { hits: 10, .. }));
    }
    #[test]
    fn missing_secondary_is_allowed_for_all_five_groups() {
        let mut rng = rand::rng();
        let mut s = MechanismState::start(RuneMode::Large, &mut rng);
        for group in 0..5 {
            let i = active(&s)[0];
            s.hit(i, &mut rng);
            s.tick(1.0, &mut rng);
            if group < 4 {
                s.tick(ROUND_BLACKOUT, &mut rng);
            }
        }
        assert!(matches!(s, MechanismState::Activated { hits: 5, .. }));
    }
    #[test]
    fn large_frame_matches_many_small_frames_through_failure_and_retry() {
        use rand::{SeedableRng, rngs::StdRng};
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        let mut x = MechanismState::start(RuneMode::Large, &mut a);
        let mut y = MechanismState::start(RuneMode::Large, &mut b);
        x.tick(9.0, &mut a);
        for _ in 0..900 {
            y.tick(0.01, &mut b);
        }
        assert_eq!(active(&x), active(&y));
        let (MechanismState::Activating(x), MechanismState::Activating(y)) = (x, y) else {
            panic!("expected active retry")
        };
        assert!((x.global_remaining - y.global_remaining).abs() < 0.001);
    }
}
