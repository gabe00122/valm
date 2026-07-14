mod arithmetic;
mod env;
mod groups;
mod wordle;

/// A Python module implemented in Rust.
#[pyo3::pymodule]
mod _envs {
    #[pymodule_export]
    use crate::arithmetic::ArithmeticEnv;

    #[pymodule_export]
    use crate::wordle::WordleEnv;
}
