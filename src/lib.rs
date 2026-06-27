mod arithmetic;
mod env;
mod groups;
mod td_lambda;
mod wordle;

/// A Python module implemented in Rust.
#[pyo3::pymodule]
mod _envs {
    #[pymodule_export]
    use crate::arithmetic::ArithmeticEnv;

    #[pymodule_export]
    use crate::wordle::WordleEnv;

    #[pymodule_export]
    use crate::td_lambda::lambda_returns;
}
