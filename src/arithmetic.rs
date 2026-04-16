extern crate rand;

use std::collections::HashMap;
use std::sync::Arc;

use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
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
    rng: SmallRng,
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

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        ArithmeticEnvInstance {
            shared,
            rng: SmallRng::seed_from_u64(seed),
            op: Operator::Add,
            x: 0.0,
            y: 0.0,
            result: 0.0,
        }
    }

    fn reset(&mut self) -> String {
        let dist = Uniform::new(0.0, 10000.0).unwrap();
        let x: f32 = self.rng.sample::<f32, _>(dist).round();
        let y: f32 = self.rng.sample::<f32, _>(dist).round();
        let op = sample_op(&mut self.rng);

        self.x = x;
        self.y = y;
        self.op = op;
        self.result = calc(op, x, y);

        let op_prompt = op_str(op);
        let prompt = format!("{x} {op_prompt} {y} = ...");

        prompt
    }

    fn step(&mut self, action: &str) -> (String, f32, bool, HashMap<String, f32>) {
        let parsed = parse_response(&self.shared.number_re, action);

        let corrected = if let Some(p) = parsed {
            (p - self.result).abs() < 0.001
        } else {
            false
        };
        let reward = if corrected { 1.0 } else { 0.0 };
        let done = true;

        let metrics = self.metrics();

        (self.reset(), reward, done, metrics)
    }
}

// Create the wrapper for Arithmetic
create_env_wrapper!(
    ArithmeticEnv,
    ArithmeticEnvInstance,
    ArithmeticSettings,
    "Solve the arithmetic expression using +, -, * or /. Show your work if needed, but end with only the numeric result on its own line. Always output with decimals such as 123.456"
);
