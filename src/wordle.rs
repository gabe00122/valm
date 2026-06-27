use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use crate::groups::GroupSequence;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::fs;

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
    group_id: u64,
    secret_word_index: usize,
    guesses: usize,
    got_yellow: [bool; 5],
    got_green: [bool; 5],
}

impl WordleInstance {
    fn generate_feedback(&mut self, shared: &WordleShared, guess: &str) -> (String, f32) {
        let secret_word = &shared.words[self.secret_word_index];
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
                reward += 0.025;
            }
            if f == &'G' && !self.got_green[i] {
                self.got_green[i] = true;
                reward += 0.025;
            }
        }

        (output.join(""), reward)
    }

    fn metrics(&self, word_found: bool) -> HashMap<String, f32> {
        let mut map = HashMap::new();
        map.insert(
            String::from("word_found"),
            if word_found { 1.0 } else { 0.0 },
        );

        map
    }
}

impl EnvInstance for WordleInstance {
    type Shared = WordleShared;

    const MAX_TURNS: usize = 6;

    fn new(_shared: &WordleShared, group_seq: &mut GroupSequence) -> Self {
        WordleInstance {
            group_id: group_seq.take_group_id(),
            secret_word_index: 0,
            guesses: 0,
            got_yellow: [false; 5],
            got_green: [false; 5],
        }
    }

    fn reset(
        &mut self,
        shared: &WordleShared,
        group_seq: &mut GroupSequence,
    ) -> (String, HashMap<String, f32>) {
        self.group_id = group_seq.take_group_id();
        let mut prng = SmallRng::seed_from_u64(self.group_id);

        self.guesses = 0;
        self.secret_word_index = prng.random_range(0..shared.words.len());
        self.got_yellow = [false; 5];
        self.got_green = [false; 5];

        ("Make your first guess now".to_string(), self.metrics(false))
    }

    fn step(
        &mut self,
        shared: &WordleShared,
        group_seq: &mut GroupSequence,
        action: &str,
    ) -> (String, f32, bool, HashMap<String, f32>) {
        let guess = shared
            .guess_re
            .find_iter(action)
            .last()
            .map(|m| m.as_str().to_uppercase());

        let (obs, reward, word_found) = if let Some(guess) = guess {
            let (feedback, reward) = self.generate_feedback(shared, &guess);
            let word_found = guess == shared.words[self.secret_word_index];

            let reward = reward + if word_found { 0.75 } else { 0.0 }; // big bonus for the complete word correct

            (feedback, reward, word_found)
        } else {
            ("No guess was found".to_string(), 0.0, false)
        };
        self.guesses += 1;

        let remaining = shared.settings.max_guesses - self.guesses;
        let guess_word = if remaining > 1 { "guesses" } else { "guess" };
        let obs = format!("{}\nYou have {} {} left", obs, remaining, guess_word);

        let metrics = self.metrics(word_found);

        if word_found || self.guesses >= shared.settings.max_guesses {
            (self.reset(shared, group_seq).0, reward, true, metrics)
        } else {
            (obs, reward, false, metrics)
        }
    }

    fn group_id(&self) -> u64 {
        self.group_id
    }
}

// Create the wrapper for Wordle
create_env_wrapper!(
    WordleEnv,
    WordleInstance,
    WordleSettings,
    "You are playing Wordle. Guess the secret 5-letter word. After each guess you will see a feedback row above your guess, with one symbol per letter position: G = correct letter in the correct position, Y = letter is in the word but in the wrong position, X = letter is not in the word. Think very briefly (2-3 sentences) then respond with your next guess. The last 5-letter word (a-z) that appears in your response is taken as your guess, so do not write other 5-letter words after it."
);
