extern crate rand;

use std::sync::Arc;

use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::{distr::Uniform, prelude::*};
use regex::Regex;

const BASES: [u32; 4] = [2, 8, 10, 16];

#[derive(Clone, FromPyObject)]
pub struct BaseConversionSettings {
    max_value: i64,
}

pub struct BaseConversionShared {
    hex_re: Regex,
    settings: BaseConversionSettings,
}

impl EnvShared for BaseConversionShared {
    type Settings = BaseConversionSettings;

    fn new(settings: Self::Settings) -> Self {
        Self {
            // Match tokens that look like numbers in any base (hex digits, 0x/0b/0o prefixes, plain digits)
            hex_re: Regex::new(r"(?i)(?:0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+|[0-9a-fA-F]+)").unwrap(),
            settings,
        }
    }
}

struct BaseConversionEnvInstance {
    shared: Arc<BaseConversionShared>,
    rng: SmallRng,
    value: i64,
    target_base: u32,
}

fn format_in_base(value: i64, base: u32) -> String {
    match base {
        2 => format!("0b{:b}", value),
        8 => format!("0o{:o}", value),
        10 => format!("{}", value),
        16 => format!("0x{:X}", value),
        _ => unreachable!(),
    }
}

fn base_name(base: u32) -> &'static str {
    match base {
        2 => "binary (base 2)",
        8 => "octal (base 8)",
        10 => "decimal (base 10)",
        16 => "hexadecimal (base 16)",
        _ => unreachable!(),
    }
}

/// Try to parse a string token as a number in the given target base.
/// Be lenient: accept with or without prefixes, ignore case for hex.
fn try_parse_in_base(token: &str, target_base: u32) -> Option<i64> {
    let lower = token.to_lowercase();

    // Strip known prefixes
    let stripped = if lower.starts_with("0x") {
        &token[2..]
    } else if lower.starts_with("0b") {
        &token[2..]
    } else if lower.starts_with("0o") {
        &token[2..]
    } else {
        token
    };

    i64::from_str_radix(stripped, target_base).ok()
}

fn parse_response(re: &Regex, text: &str, target_base: u32) -> Option<i64> {
    // Find all number-like tokens and try the last one that parses in the target base
    let mut last_valid: Option<i64> = None;
    for cap in re.captures_iter(text) {
        let token = &cap[0];
        if let Some(val) = try_parse_in_base(token, target_base) {
            last_valid = Some(val);
        }
    }
    last_valid
}

impl EnvInstance for BaseConversionEnvInstance {
    type Shared = BaseConversionShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        BaseConversionEnvInstance {
            shared,
            rng: SmallRng::seed_from_u64(seed),
            value: 0,
            target_base: 10,
        }
    }

    fn reset(&mut self) -> String {
        let max_val = self.shared.settings.max_value;
        let dist = Uniform::new(0i64, max_val + 1).unwrap();
        self.value = self.rng.sample(dist);

        // Pick two different bases
        let base_dist = Uniform::new(0usize, BASES.len()).unwrap();
        let src_idx = self.rng.sample(base_dist);
        let mut tgt_idx = self.rng.sample(base_dist);
        while tgt_idx == src_idx {
            tgt_idx = self.rng.sample(base_dist);
        }

        let source_base = BASES[src_idx];
        self.target_base = BASES[tgt_idx];

        let number_str = format_in_base(self.value, source_base);
        let target_name = base_name(self.target_base);

        format!("Convert {} to {}", number_str, target_name)
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let parsed = parse_response(&self.shared.hex_re, action, self.target_base);

        let correct = if let Some(p) = parsed {
            p == self.value
        } else {
            false
        };
        let reward = if correct { 1.0 } else { 0.0 };
        let done = true;

        (self.reset(), reward, done)
    }
}

create_env_wrapper!(
    BaseConversionEnv,
    BaseConversionEnvInstance,
    BaseConversionSettings,
    "Convert the given number to the specified base. Show your work if needed, but end with only the converted number. Use standard prefixes: 0b for binary, 0o for octal, 0x for hexadecimal, or plain digits for decimal."
);
