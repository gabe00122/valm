use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

#[derive(Clone, FromPyObject)]
pub struct WordleSettings {
    max_guesses: usize,
}

pub struct WordleShared {
    settings: WordleSettings,
    guess_re: Regex,
    words: Vec<String>,
}

impl EnvShared for WordleShared {
    type Settings = WordleSettings;

    fn new(settings: Self::Settings) -> Self {
        let words: Vec<String> = fs::read_to_string("./env_assets/wordle_words.txt")
            .expect("Failed to read wordle_words.txt")
            .split_whitespace()
            .map(|word| word.to_string())
            .collect();

        Self {
            settings,
            guess_re: Regex::new(r"(?i)\b[a-z]{5}\b").unwrap(),
            words,
        }
    }
}

struct WordleInstance {
    shared: Arc<WordleShared>,
    rng: SmallRng,
    secret_word_index: usize,
    guesses: usize,
    got_yellow: [bool; 5],
    got_green: [bool; 5],
}

impl WordleInstance {
    fn generate_feedback(&mut self, guess: &str) -> (String, f32) {
        let secret_word: &str = &self.shared.words[self.secret_word_index];
        let mut output: Vec<String> = Vec::new();
        let mut remaining_letters = HashMap::<char, u8>::new();
        let mut feedback = ['X'; 5];

        for (i, (g, s)) in guess.chars().zip(secret_word.chars()).enumerate() {
            if g == s {
                feedback[i] = 'G';
            } else {
                remaining_letters
                    .entry(s)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }
        for (i, g) in guess.chars().enumerate() {
            if feedback[i] == 'X' && remaining_letters.get(&g).unwrap_or(&0) > &0 {
                remaining_letters.entry(g).and_modify(|count| *count -= 1);
                feedback[i] = 'Y';
            }
        }

        output.push("Your feedback is:\n".to_string());
        for f in feedback {
            output.push(format!("{} ", f));
        }
        output.push("\n".to_string());
        for g in guess.chars() {
            output.push(format!("{} ", g));
        }
        output.push("\n".to_string());

        let mut reward = 0.0;
        for (i, f) in feedback.iter().enumerate() {
            if (f == &'Y' || f == &'G') && !self.got_yellow[i] {
                self.got_yellow[i] = true;
                reward += 0.05;
            }
            if f == &'G' && !self.got_green[i] {
                self.got_green[i] = true;
                reward += 0.05;
            }
        }

        (output.join(""), reward)
    }

    fn metrics(&self) -> HashMap<String, f32> {
        HashMap::new()
    }
}

impl EnvInstance for WordleInstance {
    type Shared = WordleShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        let rng = SmallRng::seed_from_u64(seed);

        WordleInstance {
            shared,
            rng,
            secret_word_index: 0,
            guesses: 0,
            got_yellow: [false; 5],
            got_green: [false; 5],
        }
    }

    fn reset(&mut self) -> String {
        self.guesses = 0;
        self.secret_word_index = self.rng.random_range(0..self.shared.words.len());
        self.got_yellow = [false; 5];
        self.got_green = [false; 5];

        "Make your first guess now".to_string()
    }

    fn step(&mut self, action: &str) -> (String, f32, bool, HashMap<String, f32>) {
        let guess = self
            .shared
            .guess_re
            .find_iter(action)
            .last()
            .map(|m| m.as_str().to_uppercase());

        let (obs, reward, word_found) = if let Some(guess) = guess {
            let (feedback, reward) = self.generate_feedback(&guess);
            let word_found = guess == self.shared.words[self.secret_word_index];

            let reward = reward + if word_found { 0.5 } else { 0.0 }; // big bonus for the complete word correct

            (feedback, reward, word_found)
        } else {
            ("No guess was found".to_string(), 0.0, false)
        };

        let metrics = self.metrics();

        self.guesses += 1;
        if word_found || self.guesses >= self.shared.settings.max_guesses {
            (self.reset(), reward, true, metrics)
        } else {
            (obs, reward, false, metrics)
        }
    }
}

// Create the wrapper for Wordle
create_env_wrapper!(
    WordleEnv,
    WordleInstance,
    WordleSettings,
    "Your job is to play wordle: Guess the 5-letter word in 6 tries. Feedback: G = Green (Correct), Y = Yellow (Wrong spot), X = Grey (Not in word). Your guess must be the last word in your response."
);
