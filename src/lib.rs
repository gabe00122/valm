mod arithmetic;
mod base_conversion;
mod countdown;
mod date_arith;
mod env;
mod graph;
mod spatial;
mod sudoku;
mod td_lambda;
mod wordle;

/// A Python module implemented in Rust.
#[pyo3::pymodule]
mod _envs {
    #[pymodule_export]
    use crate::arithmetic::ArithmeticEnv;

    #[pymodule_export]
    use crate::base_conversion::BaseConversionEnv;

    #[pymodule_export]
    use crate::countdown::CountdownEnv;

    #[pymodule_export]
    use crate::date_arith::DateArithEnv;

    #[pymodule_export]
    use crate::graph::GraphEnv;

    #[pymodule_export]
    use crate::spatial::SpatialEnv;

    #[pymodule_export]
    use crate::sudoku::SudokuEnv;

    #[pymodule_export]
    use crate::wordle::WordleEnv;

    #[pymodule_export]
    use crate::td_lambda::lambda_returns;
}
