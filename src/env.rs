use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

use itertools::Itertools;
use rand::prelude::*;

pub trait EnvShared {
    type Settings;
    fn new(settings: Self::Settings) -> Self;
}

pub trait EnvInstance {
    type Shared: EnvShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self;
    fn reset(&mut self) -> String;
    fn step(&mut self, action: &str) -> (String, f32, bool, HashMap<String, f32>);
}

pub struct Envs<E> {
    envs: Vec<E>,
}

fn mean_metrics(metrics: Vec<HashMap<String, f32>>) -> HashMap<String, f32> {
    let count = metrics.len() as f32;
    let mut combined: HashMap<String, f32> = HashMap::new();

    for m in metrics {
        for (key, value) in m {
            *combined.entry(key).or_insert(0.0) += value;
        }
    }

    for v in combined.values_mut() {
        *v /= count;
    }

    combined
}

impl<E, S> Envs<E>
where
    E: EnvInstance<Shared = S>,
    S: EnvShared,
{
    pub fn new(num: usize, seed: u64, settings: S::Settings) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let shared = Arc::new(S::new(settings));

        let envs = (0..num)
            .map(|_| E::new(rng.next_u64(), shared.clone()))
            .collect();

        Self { envs }
    }

    pub fn reset<I>(&mut self, indices: I) -> Vec<String>
    where
        I: IntoIterator,
        I::Item: Borrow<i32>,
    {
        indices
            .into_iter()
            .map(|i| self.envs.get_mut(*i.borrow() as usize).unwrap().reset())
            .collect()
    }

    pub fn step<I, A>(
        &mut self,
        indices: I,
        actions: A,
    ) -> (Vec<String>, Vec<f32>, Vec<bool>, HashMap<String, f32>)
    where
        I: IntoIterator,
        I::Item: Borrow<i32>,
        A: IntoIterator,
        A::Item: AsRef<str>,
    {
        let (obs, reward, done, metrics): (
            Vec<String>,
            Vec<f32>,
            Vec<bool>,
            Vec<HashMap<String, f32>>,
        ) = indices
            .into_iter()
            .zip(actions)
            .map(|(index, action)| {
                let index = *index.borrow();
                let action = action.as_ref();

                self.envs.get_mut(index as usize).unwrap().step(action)
            })
            .multiunzip();

        (obs, reward, done, mean_metrics(metrics))
    }
}

#[macro_export]
macro_rules! create_env_wrapper {
    ($py_name:ident, $rust_env:ty, $setting_struct:ty, $instr:literal) => {
        #[pyclass]
        pub struct $py_name {
            envs: Envs<$rust_env>,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new(num_agents: usize, seed: u64, settings: $setting_struct) -> Self {
                Self {
                    envs: Envs::new(num_agents, seed, settings),
                }
            }

            fn reset<'py>(
                &mut self,
                batch_indices: PyReadonlyArray1<'py, i32>,
            ) -> PyResult<Vec<String>> {
                let indices = batch_indices.as_array();
                Ok(self.envs.reset(indices))
            }

            fn step<'py>(
                &mut self,
                py: Python<'py>,
                batch_indices: PyReadonlyArray1<'py, i32>,
                actions: Vec<String>,
            ) -> PyResult<(
                Vec<String>,
                Bound<'py, PyArray1<f32>>,
                Bound<'py, PyArray1<bool>>,
                HashMap<String, f32>,
            )> {
                let indices = batch_indices.as_array();

                let (obs, rewards, dones, metrics) = self.envs.step(&indices, &actions);

                let rewards = rewards.into_pyarray(py);
                let dones = dones.into_pyarray(py);

                Ok((obs, rewards, dones, metrics))
            }

            fn instructions(&self) -> PyResult<&'static str> {
                Ok($instr)
            }
        }
    };
}
