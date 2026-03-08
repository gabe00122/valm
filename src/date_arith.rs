extern crate rand;

use std::sync::Arc;

use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::{distr::Uniform, prelude::*};
use regex::Regex;

const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_month(year: i32, month: i32) -> i32 {
    match month {
        1 => 31,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        3 => 31,
        4 => 30,
        5 => 31,
        6 => 30,
        7 => 31,
        8 => 31,
        9 => 30,
        10 => 31,
        11 => 30,
        12 => 31,
        _ => 0,
    }
}

/// Convert a (year, month, day) date to a day count from a fixed epoch (year 0, Jan 1).
fn date_to_days(year: i32, month: i32, day: i32) -> i64 {
    // We use an algorithm based on shifting March to month 0
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let m = if month <= 2 {
        month + 9
    } else {
        month - 3
    } as i64;
    let d = day as i64;

    // Days from years
    let days_from_years = 365 * y + y / 4 - y / 100 + y / 400;
    // Days from months (using the shifted month formula)
    let days_from_months = (153 * m + 2) / 5;

    days_from_years + days_from_months + d - 1 // -1 because day 1 is the start
}

/// Convert a day count back to (year, month, day).
fn days_to_date(day_count: i64) -> (i32, i32, i32) {
    // Inverse of date_to_days
    let d = day_count + 1; // undo the -1
    // Estimate year
    let mut y = (10000 * d + 14780) / 3652425;
    let mut doy = d - (365 * y + y / 4 - y / 100 + y / 400);
    if doy < 0 {
        y -= 1;
        doy = d - (365 * y + y / 4 - y / 100 + y / 400);
    }
    let mi = (100 * doy + 52) / 3060;
    let month = if mi < 10 { mi + 3 } else { mi - 9 };
    let year = y + (if month <= 2 { 1 } else { 0 });
    let day = doy - (mi * 306 + 5) / 10 + 1;

    (year as i32, month as i32, day as i32)
}

fn add_days(year: i32, month: i32, day: i32, delta: i32) -> (i32, i32, i32) {
    let d = date_to_days(year, month, day);
    days_to_date(d + delta as i64)
}

fn month_name(month: i32) -> &'static str {
    MONTH_NAMES[(month - 1) as usize]
}

fn month_from_name(name: &str) -> Option<i32> {
    let lower = name.to_lowercase();
    MONTH_NAMES
        .iter()
        .position(|&m| m.to_lowercase() == lower)
        .map(|i| i as i32 + 1)
}

#[derive(Clone, FromPyObject)]
pub struct DateArithSettings {
    max_days: i32,
}

pub struct DateArithShared {
    // Regex patterns for parsing dates from LLM response
    // Pattern 1: "March 12, 2024" or "12 March 2024"
    month_name_re: Regex,
    // Pattern 2: "2024-03-12"
    iso_re: Regex,
    // Pattern 3: "3/12/2024" or "03/12/2024"
    slash_re: Regex,
    settings: DateArithSettings,
}

impl EnvShared for DateArithShared {
    type Settings = DateArithSettings;

    fn new(settings: Self::Settings) -> Self {
        Self {
            month_name_re: Regex::new(
                r"(?i)(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})|(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})"
            ).unwrap(),
            iso_re: Regex::new(r"(\d{4})-(\d{1,2})-(\d{1,2})").unwrap(),
            slash_re: Regex::new(r"(\d{1,2})/(\d{1,2})/(\d{4})").unwrap(),
            settings,
        }
    }
}

/// Try to parse a date from the LLM response text. Returns the last match found as (year, month, day).
fn parse_date_response(shared: &DateArithShared, text: &str) -> Option<(i32, i32, i32)> {
    let mut last_date: Option<(i32, i32, i32)> = None;

    // Try month name patterns: "March 12, 2024" or "12 March 2024"
    for cap in shared.month_name_re.captures_iter(text) {
        if let (Some(month_str), Some(day_str), Some(year_str)) =
            (cap.get(1), cap.get(2), cap.get(3))
        {
            // "Month Day, Year" form
            if let Some(month) = month_from_name(month_str.as_str()) {
                if let (Ok(day), Ok(year)) =
                    (day_str.as_str().parse::<i32>(), year_str.as_str().parse::<i32>())
                {
                    last_date = Some((year, month, day));
                }
            }
        } else if let (Some(day_str), Some(month_str), Some(year_str)) =
            (cap.get(4), cap.get(5), cap.get(6))
        {
            // "Day Month Year" form
            if let Some(month) = month_from_name(month_str.as_str()) {
                if let (Ok(day), Ok(year)) =
                    (day_str.as_str().parse::<i32>(), year_str.as_str().parse::<i32>())
                {
                    last_date = Some((year, month, day));
                }
            }
        }
    }

    // Try ISO format: "2024-03-12"
    for cap in shared.iso_re.captures_iter(text) {
        if let (Ok(year), Ok(month), Ok(day)) = (
            cap[1].parse::<i32>(),
            cap[2].parse::<i32>(),
            cap[3].parse::<i32>(),
        ) {
            last_date = Some((year, month, day));
        }
    }

    // Try slash format: "3/12/2024"
    for cap in shared.slash_re.captures_iter(text) {
        if let (Ok(month), Ok(day), Ok(year)) = (
            cap[1].parse::<i32>(),
            cap[2].parse::<i32>(),
            cap[3].parse::<i32>(),
        ) {
            last_date = Some((year, month, day));
        }
    }

    last_date
}

struct DateArithEnvInstance {
    shared: Arc<DateArithShared>,
    rng: SmallRng,
    expected_year: i32,
    expected_month: i32,
    expected_day: i32,
}

impl EnvInstance for DateArithEnvInstance {
    type Shared = DateArithShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        DateArithEnvInstance {
            shared,
            rng: SmallRng::seed_from_u64(seed),
            expected_year: 0,
            expected_month: 0,
            expected_day: 0,
        }
    }

    fn reset(&mut self) -> String {
        let max_days = self.shared.settings.max_days;

        // Random year 2000..=2030
        let year: i32 = self.rng.sample(Uniform::new(2000, 2031).unwrap());
        // Random month 1..=12
        let month: i32 = self.rng.sample(Uniform::new(1, 13).unwrap());
        // Random day 1..=days_in_month
        let dim = days_in_month(year, month);
        let day: i32 = self.rng.sample(Uniform::new(1, dim + 1).unwrap());

        // Random delta in [-max_days, max_days], excluding 0
        let mut delta: i32 = self.rng.sample(Uniform::new(1, max_days + 1).unwrap());
        let direction: bool = self.rng.random();
        if !direction {
            delta = -delta;
        }

        let (ey, em, ed) = add_days(year, month, day, delta);
        self.expected_year = ey;
        self.expected_month = em;
        self.expected_day = ed;

        let direction_word = if delta > 0 { "after" } else { "before" };
        let abs_delta = delta.abs();
        let month_str = month_name(month);

        format!(
            "What date is {} days {} {} {}, {}?",
            abs_delta, direction_word, month_str, day, year
        )
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let parsed = parse_date_response(&self.shared, action);

        let correct = if let Some((y, m, d)) = parsed {
            y == self.expected_year && m == self.expected_month && d == self.expected_day
        } else {
            false
        };

        let reward = if correct { 1.0 } else { 0.0 };
        let done = true;

        (self.reset(), reward, done)
    }
}

create_env_wrapper!(
    DateArithEnv,
    DateArithEnvInstance,
    DateArithSettings,
    "Solve the date arithmetic problem. You are given a date and a number of days to add or subtract. Compute the resulting date. Show your work if needed, but make sure your final answer contains the date in a clear format such as 'Month Day, Year' (e.g. April 28, 2024)."
);
