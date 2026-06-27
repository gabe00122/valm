extern crate rand;

use std::collections::HashMap;
use std::sync::Arc;

use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use crate::groups::GroupSequence;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::{distr::Uniform, prelude::*};
use regex::Regex;

#[derive(Clone, Copy)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, FromPyObject)]
pub struct ArithmeticSettings {
    max_x: i32,
    max_y: i32,
}

pub struct ArithmeticShared {
    number_re: Regex,
    settings: ArithmeticSettings,
}

impl EnvShared for ArithmeticShared {
    type Settings = ArithmeticSettings;

    fn new(settings: Self::Settings) -> Self {
        Self {
            number_re: Regex::new(r"-?[\d,]+(?:\.\d*)?").unwrap(),
            settings,
        }
    }
}

struct ArithmeticEnvInstance {
    shared: Arc<ArithmeticShared>,
    group_id: u64,
    op: Operator,
    x: f32,
    y: f32,
    result: f32,
}

fn sample_op(rng: &mut impl Rng) -> Operator {
    let id: u8 = rng.sample(Uniform::new(0, 3).unwrap());
    match id {
        0 => Operator::Add,
        1 => Operator::Sub,
        2 => Operator::Mul,
        _ => Operator::Div,
    }
}

fn op_str(op: Operator) -> &'static str {
    match op {
        Operator::Add => "+",
        Operator::Sub => "-",
        Operator::Mul => "*",
        Operator::Div => "/",
    }
}

fn calc(op: Operator, x: f32, y: f32) -> f32 {
    match op {
        Operator::Add => x + y,
        Operator::Sub => x - y,
        Operator::Mul => x * y,
        Operator::Div => x / y,
    }
}

fn parse_response(re: &Regex, text: &str) -> Option<f32> {
    if let Some(last_match) = re.captures_iter(text).last() {
        let clean_match = last_match[0].replace(",", "");
        clean_match.parse().ok()
    } else {
        None
    }
}

impl ArithmeticEnvInstance {
    fn metrics(&self) -> HashMap<String, f32> {
        HashMap::new()
    }
}

impl EnvInstance for ArithmeticEnvInstance {
    type Shared = ArithmeticShared;

    const MAX_TURNS: usize = 1;

    fn new(group_seq: &mut GroupSequence, shared: Arc<Self::Shared>) -> Self {
        ArithmeticEnvInstance {
            shared,
            group_id: group_seq.take_group_id(),
            op: Operator::Add,
            x: 0.0,
            y: 0.0,
            result: 0.0,
        }
    }

    fn reset(&mut self, group_seq: &mut GroupSequence) -> (String, HashMap<String, f32>) {
        self.group_id = group_seq.take_group_id();
        let mut prng = SmallRng::seed_from_u64(self.group_id);

        let x_dist = Uniform::new(0.0, self.shared.settings.max_x.max(1) as f32).unwrap();
        let y_dist = Uniform::new(0.0, self.shared.settings.max_y.max(1) as f32).unwrap();
        let x: f32 = prng.sample::<f32, _>(x_dist).round();
        let y: f32 = prng.sample::<f32, _>(y_dist).round();
        let op = sample_op(&mut prng);

        self.x = x;
        self.y = y;
        self.op = op;
        self.result = calc(op, x, y);

        let op_prompt = op_str(op);
        let prompt = format!("{x} {op_prompt} {y} = ...");

        (prompt, self.metrics())
    }

    fn step(
        &mut self,
        action: &str,
        group_seq: &mut GroupSequence,
    ) -> (String, f32, bool, HashMap<String, f32>) {
        let parsed = parse_response(&self.shared.number_re, action);

        let corrected = if let Some(p) = parsed {
            (p - self.result).abs() < 0.001
        } else {
            false
        };
        let reward = if corrected { 1.0 } else { 0.0 };
        let done = true;

        let (obs, metrics) = self.reset(group_seq);

        (obs, reward, done, metrics)
    }

    fn group_id(&self) -> u64 {
        self.group_id
    }
}

// Create the wrapper for Arithmetic
create_env_wrapper!(
    ArithmeticEnv,
    ArithmeticEnvInstance,
    ArithmeticSettings,
    "Solve the arithmetic expression using +, -, * or /. Show your work if needed, but end with only the numeric result on its own line. Always output with decimals such as 123.456"
);
