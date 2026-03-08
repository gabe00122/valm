use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::distr::Uniform;
use rand::prelude::*;
use regex::Regex;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Heading {
    North,
    East,
    South,
    West,
}

impl Heading {
    fn turn_left(self) -> Self {
        match self {
            Heading::North => Heading::West,
            Heading::West => Heading::South,
            Heading::South => Heading::East,
            Heading::East => Heading::North,
        }
    }

    fn turn_right(self) -> Self {
        match self {
            Heading::North => Heading::East,
            Heading::East => Heading::South,
            Heading::South => Heading::West,
            Heading::West => Heading::North,
        }
    }

    fn dx_dy(self) -> (i32, i32) {
        match self {
            Heading::North => (0, 1),
            Heading::East => (1, 0),
            Heading::South => (0, -1),
            Heading::West => (-1, 0),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Heading::North => "North",
            Heading::East => "East",
            Heading::South => "South",
            Heading::West => "West",
        }
    }
}

#[derive(Clone, FromPyObject)]
pub struct SpatialSettings {
    grid_size: i32,
    max_steps: usize,
}

pub struct SpatialShared {
    settings: SpatialSettings,
    walk_re: Regex,
}

impl EnvShared for SpatialShared {
    type Settings = SpatialSettings;

    fn new(settings: Self::Settings) -> Self {
        Self {
            walk_re: Regex::new(r"(?i)(?:walk|move|forward|step)\s+(\d+)").unwrap(),
            settings,
        }
    }
}

enum Action {
    Walk(i32),
    TurnLeft,
    TurnRight,
    TurnAround,
}

struct SpatialInstance {
    shared: Arc<SpatialShared>,
    rng: SmallRng,
    x: i32,
    y: i32,
    heading: Heading,
    target_x: i32,
    target_y: i32,
    steps: usize,
    initial_distance: i32,
}

fn manhattan(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
    (x1 - x2).abs() + (y1 - y2).abs()
}

fn parse_action(walk_re: &Regex, text: &str) -> Option<Action> {
    let lower = text.to_lowercase();

    // Check turns first
    if lower.contains("turn around") {
        return Some(Action::TurnAround);
    }
    if lower.contains("turn left") || lower.contains("left") {
        return Some(Action::TurnLeft);
    }
    if lower.contains("turn right") || lower.contains("right") {
        return Some(Action::TurnRight);
    }

    // Check walk
    if let Some(caps) = walk_re.captures(text) {
        if let Ok(n) = caps[1].parse::<i32>() {
            return Some(Action::Walk(n.max(0)));
        }
    }

    None
}

fn format_state(x: i32, y: i32, heading: Heading, target_x: i32, target_y: i32) -> String {
    format!(
        "You are at ({}, {}) facing {}. Target: ({}, {}).",
        x,
        y,
        heading.name(),
        target_x,
        target_y
    )
}

impl EnvInstance for SpatialInstance {
    type Shared = SpatialShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        SpatialInstance {
            shared,
            rng: SmallRng::seed_from_u64(seed),
            x: 0,
            y: 0,
            heading: Heading::North,
            target_x: 0,
            target_y: 0,
            steps: 0,
            initial_distance: 0,
        }
    }

    fn reset(&mut self) -> String {
        let gs = self.shared.settings.grid_size;
        let dist = Uniform::new(-gs, gs + 1).unwrap();

        self.x = 0;
        self.y = 0;
        self.heading = match self.rng.sample(Uniform::new(0, 4).unwrap()) {
            0 => Heading::North,
            1 => Heading::East,
            2 => Heading::South,
            _ => Heading::West,
        };
        self.steps = 0;

        // Generate target that's not at origin
        loop {
            self.target_x = self.rng.sample(dist);
            self.target_y = self.rng.sample(dist);
            if self.target_x != 0 || self.target_y != 0 {
                break;
            }
        }

        self.initial_distance = manhattan(0, 0, self.target_x, self.target_y);

        format_state(self.x, self.y, self.heading, self.target_x, self.target_y)
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let dist_before = manhattan(self.x, self.y, self.target_x, self.target_y);

        let parsed = parse_action(&self.shared.walk_re, action);

        match parsed {
            Some(Action::Walk(n)) => {
                let (dx, dy) = self.heading.dx_dy();
                self.x += dx * n;
                self.y += dy * n;
            }
            Some(Action::TurnLeft) => self.heading = self.heading.turn_left(),
            Some(Action::TurnRight) => self.heading = self.heading.turn_right(),
            Some(Action::TurnAround) => {
                self.heading = self.heading.turn_right().turn_right();
            }
            None => {} // no valid action parsed, skip
        }

        self.steps += 1;
        let dist_after = manhattan(self.x, self.y, self.target_x, self.target_y);
        let reached = dist_after == 0;
        let out_of_steps = self.steps >= self.shared.settings.max_steps;
        let done = reached || out_of_steps;

        // Reward: small shaping for getting closer + big bonus for reaching target
        let shaping = if dist_after < dist_before {
            0.02
        } else if dist_after > dist_before {
            -0.01
        } else {
            0.0
        };

        let arrival_bonus = if reached { 0.5 } else { 0.0 };
        let efficiency_bonus = if reached {
            0.5 * (1.0 - self.steps as f32 / self.shared.settings.max_steps as f32)
        } else {
            0.0
        };

        let reward = (shaping + arrival_bonus + efficiency_bonus).clamp(0.0, 1.0);

        if done {
            (self.reset(), reward, true)
        } else {
            let obs = format_state(self.x, self.y, self.heading, self.target_x, self.target_y);
            (obs, reward, false)
        }
    }
}

create_env_wrapper!(
    SpatialEnv,
    SpatialInstance,
    SpatialSettings,
    "Navigate to the target coordinates. Each turn, you can do ONE action: 'walk N' (move forward N steps), 'turn left', 'turn right', or 'turn around'. You start at (0, 0). Positive X is East, positive Y is North."
);
