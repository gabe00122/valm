use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use std::sync::Arc;

#[derive(Clone, FromPyObject)]
pub struct SudokuSettings {
    grid_size: usize, // 4 or 9
    num_removed: usize,
}

pub struct SudokuShared {
    settings: SudokuSettings,
    box_size: usize,
}

impl EnvShared for SudokuShared {
    type Settings = SudokuSettings;

    fn new(settings: Self::Settings) -> Self {
        let box_size = match settings.grid_size {
            4 => 2,
            9 => 3,
            _ => panic!("grid_size must be 4 or 9"),
        };
        Self { settings, box_size }
    }
}

struct SudokuInstance {
    shared: Arc<SudokuShared>,
    rng: SmallRng,
    solution: Vec<Vec<u8>>,
    puzzle: Vec<Vec<u8>>, // 0 = empty
}

// Fill the grid using backtracking with randomized value ordering
fn solve(grid: &mut Vec<Vec<u8>>, size: usize, box_size: usize, rng: &mut SmallRng) -> bool {
    if let Some((row, col)) = find_empty(grid, size) {
        let mut values: Vec<u8> = (1..=size as u8).collect();
        values.shuffle(rng);

        for val in values {
            if is_valid_placement(grid, size, box_size, row, col, val) {
                grid[row][col] = val;
                if solve(grid, size, box_size, rng) {
                    return true;
                }
                grid[row][col] = 0;
            }
        }
        false
    } else {
        true // no empty cells, solved
    }
}

fn find_empty(grid: &[Vec<u8>], size: usize) -> Option<(usize, usize)> {
    for r in 0..size {
        for c in 0..size {
            if grid[r][c] == 0 {
                return Some((r, c));
            }
        }
    }
    None
}

fn is_valid_placement(
    grid: &[Vec<u8>],
    size: usize,
    box_size: usize,
    row: usize,
    col: usize,
    val: u8,
) -> bool {
    // Check row
    for c in 0..size {
        if grid[row][c] == val {
            return false;
        }
    }
    // Check column
    for r in 0..size {
        if grid[r][col] == val {
            return false;
        }
    }
    // Check box
    let box_row = (row / box_size) * box_size;
    let box_col = (col / box_size) * box_size;
    for r in box_row..box_row + box_size {
        for c in box_col..box_col + box_size {
            if grid[r][c] == val {
                return false;
            }
        }
    }
    true
}

fn format_grid(grid: &[Vec<u8>], size: usize, box_size: usize) -> String {
    let mut out = String::new();
    for (r, row) in grid.iter().enumerate() {
        if r > 0 && r % box_size == 0 {
            // Separator line
            for b in 0..box_size {
                if b > 0 {
                    out.push_str("-+-");
                }
                for _ in 0..box_size {
                    out.push('-');
                    out.push('-');
                }
                // Remove trailing space equivalent
                out.pop();
            }
            out.push('\n');
        }
        for (c, &val) in row.iter().enumerate() {
            if c > 0 && c % box_size == 0 {
                out.push_str("| ");
            }
            if val == 0 {
                out.push('.');
            } else {
                out.push((b'0' + val) as char);
            }
            if c < size - 1 {
                out.push(' ');
            }
        }
        out.push('\n');
    }
    out
}

fn parse_grid(response: &str, size: usize) -> Option<Vec<Vec<u8>>> {
    let mut grid = Vec::new();

    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Skip separator lines (containing only -, +, |, spaces)
        if trimmed.chars().all(|c| c == '-' || c == '+' || c == '|' || c == ' ') {
            continue;
        }

        // Extract digits from the line
        let digits: Vec<u8> = trimmed
            .chars()
            .filter(|c| c.is_ascii_digit() && *c != '0')
            .map(|c| c as u8 - b'0')
            .collect();

        if digits.len() == size {
            grid.push(digits);
        }
    }

    if grid.len() == size {
        Some(grid)
    } else {
        None
    }
}

fn score_grid(submitted: &[Vec<u8>], solution: &[Vec<u8>], puzzle: &[Vec<u8>], size: usize) -> f32 {
    let mut correct = 0;
    let mut total = 0;

    for r in 0..size {
        for c in 0..size {
            if puzzle[r][c] == 0 {
                // This was an empty cell the LLM needed to fill
                total += 1;
                if submitted[r][c] == solution[r][c] {
                    correct += 1;
                }
            }
        }
    }

    if total == 0 {
        return 1.0;
    }

    correct as f32 / total as f32
}

impl EnvInstance for SudokuInstance {
    type Shared = SudokuShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        SudokuInstance {
            shared,
            rng: SmallRng::seed_from_u64(seed),
            solution: Vec::new(),
            puzzle: Vec::new(),
        }
    }

    fn reset(&mut self) -> String {
        let size = self.shared.settings.grid_size;
        let box_size = self.shared.box_size;

        // Generate a solved grid
        let mut grid = vec![vec![0u8; size]; size];
        solve(&mut grid, size, box_size, &mut self.rng);
        self.solution = grid.clone();

        // Remove cells
        let mut positions: Vec<(usize, usize)> = (0..size)
            .flat_map(|r| (0..size).map(move |c| (r, c)))
            .collect();
        positions.shuffle(&mut self.rng);

        let to_remove = self.shared.settings.num_removed.min(positions.len());
        for &(r, c) in &positions[..to_remove] {
            grid[r][c] = 0;
        }
        self.puzzle = grid;

        let grid_str = format_grid(&self.puzzle, size, box_size);
        format!("Solve this {}x{} Sudoku (replace . with the correct digit):\n{}", size, size, grid_str)
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let size = self.shared.settings.grid_size;

        let reward = if let Some(submitted) = parse_grid(action, size) {
            score_grid(&submitted, &self.solution, &self.puzzle, size)
        } else {
            0.0
        };

        (self.reset(), reward, true)
    }
}

create_env_wrapper!(
    SudokuEnv,
    SudokuInstance,
    SudokuSettings,
    "Solve the Sudoku puzzle. Replace each '.' with the correct digit. Each row, column, and box must contain each digit exactly once. For a 4x4 grid use digits 1-4, for 9x9 use 1-9. Output the complete filled grid with digits separated by spaces, using | between boxes and dashes between box rows, matching the input format."
);
