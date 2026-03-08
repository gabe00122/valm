use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::distr::Uniform;
use std::sync::Arc;

#[derive(Clone, FromPyObject)]
pub struct CountdownSettings {
    num_numbers: usize,
    num_operations: usize,
    max_number: i32,
}

pub struct CountdownShared {
    settings: CountdownSettings,
}

impl EnvShared for CountdownShared {
    type Settings = CountdownSettings;

    fn new(settings: Self::Settings) -> Self {
        Self { settings }
    }
}

struct CountdownInstance {
    shared: Arc<CountdownShared>,
    rng: SmallRng,
    numbers: Vec<i32>,
    target: i32,
}

// Generate a puzzle by building an expression tree forwards, guaranteeing a solution exists.
fn generate_puzzle(rng: &mut impl Rng, num_numbers: usize, num_operations: usize, max_number: i32) -> (Vec<i32>, i32) {
    let dist = Uniform::new(1, max_number + 1).unwrap();

    loop {
        let numbers: Vec<i32> = (0..num_numbers).map(|_| rng.sample(dist)).collect();

        // Pick a subset and chain operations
        let mut pool = numbers.clone();
        pool.shuffle(rng);

        let mut result = pool[0] as f64;
        let ops_to_apply = num_operations.min(pool.len() - 1);

        let mut valid = true;
        for i in 0..ops_to_apply {
            let operand = pool[i + 1] as f64;
            let op: u8 = rng.sample(Uniform::new(0, 4).unwrap());
            result = match op {
                0 => result + operand,
                1 => result - operand,
                2 => result * operand,
                _ => {
                    if operand == 0.0 {
                        valid = false;
                        break;
                    }
                    let div = result / operand;
                    if div.fract() != 0.0 {
                        // skip non-integer divisions, retry
                        valid = false;
                        break;
                    }
                    div
                }
            };
        }

        if !valid || result <= 0.0 || result > 999999.0 || result.fract() != 0.0 {
            continue;
        }

        let target = result as i32;
        // Make sure target isn't trivially one of the numbers
        if numbers.contains(&target) && num_operations > 0 {
            continue;
        }

        return (numbers, target);
    }
}

// Simple expression evaluator supporting +, -, *, / and parentheses.
// Returns (value, list of number literals used).
fn evaluate_expr(expr: &str) -> Option<(f64, Vec<i32>)> {
    let tokens = tokenize(expr)?;
    let mut pos = 0;
    let mut used = Vec::new();
    let result = parse_expr(&tokens, &mut pos, &mut used)?;
    if pos != tokens.len() {
        return None;
    }
    Some((result, used))
}

#[derive(Debug, Clone)]
enum Token {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenize(expr: &str) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\n' | '\t' => { chars.next(); }
            '0'..='9' => {
                let mut num_str = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() {
                        num_str.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Num(num_str.parse().ok()?));
            }
            '+' | '-' | '*' | '/' | 'x' | 'X' | '×' | '÷' => {
                let normalized = match c {
                    'x' | 'X' | '×' => '*',
                    '÷' => '/',
                    other => other,
                };
                tokens.push(Token::Op(normalized));
                chars.next();
            }
            '(' => { tokens.push(Token::LParen); chars.next(); }
            ')' => { tokens.push(Token::RParen); chars.next(); }
            _ => { chars.next(); } // skip unknown chars
        }
    }

    Some(tokens)
}

// Recursive descent: expr -> term ((+|-) term)*
fn parse_expr(tokens: &[Token], pos: &mut usize, used: &mut Vec<i32>) -> Option<f64> {
    let mut left = parse_term(tokens, pos, used)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Op('+') => { *pos += 1; left += parse_term(tokens, pos, used)?; }
            Token::Op('-') => { *pos += 1; left -= parse_term(tokens, pos, used)?; }
            _ => break,
        }
    }
    Some(left)
}

// term -> factor ((*|/) factor)*
fn parse_term(tokens: &[Token], pos: &mut usize, used: &mut Vec<i32>) -> Option<f64> {
    let mut left = parse_factor(tokens, pos, used)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Op('*') => { *pos += 1; left *= parse_factor(tokens, pos, used)?; }
            Token::Op('/') => {
                *pos += 1;
                let right = parse_factor(tokens, pos, used)?;
                if right == 0.0 { return None; }
                left /= right;
            }
            _ => break,
        }
    }
    Some(left)
}

// factor -> number | '(' expr ')'
fn parse_factor(tokens: &[Token], pos: &mut usize, used: &mut Vec<i32>) -> Option<f64> {
    if *pos >= tokens.len() { return None; }
    match &tokens[*pos] {
        Token::Num(n) => {
            let val = *n;
            used.push(val as i32);
            *pos += 1;
            Some(val)
        }
        Token::LParen => {
            *pos += 1;
            let val = parse_expr(tokens, pos, used)?;
            if *pos < tokens.len() && matches!(&tokens[*pos], Token::RParen) {
                *pos += 1;
                Some(val)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn validate_numbers(used: &[i32], available: &[i32]) -> bool {
    let mut remaining: Vec<i32> = available.to_vec();
    for &n in used {
        if let Some(idx) = remaining.iter().position(|&x| x == n) {
            remaining.remove(idx);
        } else {
            return false;
        }
    }
    true
}

// Try to find a math expression in the LLM response (last line that parses).
fn extract_and_eval(response: &str, available: &[i32]) -> Option<f64> {
    // Try each line in reverse, return first valid expression
    for line in response.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        if let Some((value, used)) = evaluate_expr(trimmed) {
            if validate_numbers(&used, available) {
                return Some(value);
            }
        }
    }
    None
}

impl EnvInstance for CountdownInstance {
    type Shared = CountdownShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        CountdownInstance {
            shared,
            rng: SmallRng::seed_from_u64(seed),
            numbers: Vec::new(),
            target: 0,
        }
    }

    fn reset(&mut self) -> String {
        let (numbers, target) = generate_puzzle(
            &mut self.rng,
            self.shared.settings.num_numbers,
            self.shared.settings.num_operations,
            self.shared.settings.max_number,
        );
        self.numbers = numbers;
        self.target = target;

        let nums_str: Vec<String> = self.numbers.iter().map(|n| n.to_string()).collect();
        format!(
            "Numbers: [{}]\nTarget: {}",
            nums_str.join(", "),
            self.target
        )
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let reward = if let Some(value) = extract_and_eval(action, &self.numbers) {
            let diff = (value - self.target as f64).abs();
            if diff < 0.5 {
                1.0
            } else {
                (1.0 - (diff / self.target as f64)).max(0.0) as f32
            }
        } else {
            0.0
        };

        (self.reset(), reward, true)
    }
}

create_env_wrapper!(
    CountdownEnv,
    CountdownInstance,
    CountdownSettings,
    "You are playing Countdown. Given a list of numbers and a target, combine the numbers using +, -, *, / to reach the target. Each number can only be used once. You don't have to use all numbers. Write your final expression on its own line."
);
