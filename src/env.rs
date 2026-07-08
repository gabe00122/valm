use std::borrow::Borrow;
use std::collections::HashMap;

use itertools::Itertools;
use numpy::ndarray::Array1;

use crate::groups::GroupSequence;

pub trait EnvShared {
    type Settings;
    fn new(settings: Self::Settings) -> Self;
}

pub trait EnvInstance {
    type Shared: EnvShared;

    const MAX_TURNS: usize;

    fn new(shared: &Self::Shared, group_seq: &mut GroupSequence) -> Self;
    fn reset(
        &mut self,
        shared: &Self::Shared,
        group_seq: &mut GroupSequence,
    ) -> (String, HashMap<String, f32>);
    fn step(
        &mut self,
        shared: &Self::Shared,
        group_seq: &mut GroupSequence,
        action: &str,
    ) -> (String, f32, bool, HashMap<String, f32>);

    /// The id of the problem the instance is currently solving. Members of the
    /// same GRPO group share the same id for the same problem (see `Envs::new`).
    fn group_id(&self) -> u64;
}

pub struct Envs<E, S> {
    shared: S,
    envs: Vec<E>,
    group_seq: GroupSequence,
}

fn collect_metrics(metrics: Vec<HashMap<String, f32>>) -> HashMap<String, Array1<f32>> {
    let mut collected = HashMap::<String, Array1<f32>>::new();
    let metrics_len = metrics.len();
    let mut i = 0;

    for env_result in metrics {
        for (key, value) in env_result {
            let array = collected
                .entry(key)
                .or_insert_with(|| Array1::zeros(metrics_len));
            array[i] = value;
        }
        i += 1;
    }

    collected
}

impl<E, S> Envs<E, S>
where
    E: EnvInstance<Shared = S>,
    S: EnvShared,
{
    pub fn new(num: usize, group_size: usize, seed: u64, settings: S::Settings) -> Self {
        let shared = S::new(settings);
        let group_size = group_size.max(1);
        let mut group_seq = GroupSequence::new(seed, group_size);

        let envs = (0..num).map(|_| E::new(&shared, &mut group_seq)).collect();

        Self {
            envs,
            group_seq,
            shared,
        }
    }

    pub fn reset<I>(&mut self, indices: I) -> (Vec<String>, Vec<u64>, HashMap<String, Array1<f32>>)
    where
        I: IntoIterator,
        I::Item: Borrow<i32>,
    {
        let (obs, group_ids, metrics): (Vec<String>, Vec<u64>, Vec<HashMap<String, f32>>) = indices
            .into_iter()
            .map(|i| {
                let env = self.envs.get_mut(*i.borrow() as usize).unwrap();
                let (obs, metrics) = env.reset(&self.shared, &mut self.group_seq);
                (obs, env.group_id(), metrics)
            })
            .multiunzip();

        (obs, group_ids, collect_metrics(metrics))
    }

    pub fn step<I, A>(
        &mut self,
        indices: I,
        actions: A,
    ) -> (
        Vec<String>,
        Vec<f32>,
        Vec<bool>,
        Vec<u64>,
        HashMap<String, Array1<f32>>,
    )
    where
        I: IntoIterator,
        I::Item: Borrow<i32>,
        A: IntoIterator,
        A::Item: AsRef<str>,
    {
        let (obs, reward, done, group_ids, metrics): (
            Vec<String>,
            Vec<f32>,
            Vec<bool>,
            Vec<u64>,
            Vec<HashMap<String, f32>>,
        ) = indices
            .into_iter()
            .zip(actions)
            .map(|(index, action)| {
                let index = *index.borrow();
                let action = action.as_ref();

                let env = self.envs.get_mut(index as usize).unwrap();
                // Capture the id *before* stepping: a `done` step resets the
                // instance to the next problem, so this is the id of the episode
                // that just finished.
                let group_id = env.group_id();
                let (obs, reward, done, metrics) =
                    env.step(&self.shared, &mut self.group_seq, action);

                (obs, reward, done, group_id, metrics)
            })
            .multiunzip();

        (obs, reward, done, group_ids, collect_metrics(metrics))
    }
}

#[macro_export]
macro_rules! create_env_wrapper {
    ($py_name:ident, $rust_env:ty, $setting_struct:ty, $instr:literal) => {
        #[pyclass]
        pub struct $py_name {
            envs: Envs<$rust_env, <$rust_env as EnvInstance>::Shared>,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new(
                num_agents: usize,
                group_size: usize,
                seed: u64,
                settings: $setting_struct,
            ) -> Self {
                Self {
                    envs: Envs::new(num_agents, group_size, seed, settings),
                }
            }

            fn reset<'py>(
                &mut self,
                py: Python<'py>,
                batch_indices: PyReadonlyArray1<'py, i32>,
            ) -> PyResult<(
                Vec<String>,
                Bound<'py, PyArray1<u64>>,
                HashMap<String, Bound<'py, PyArray1<f32>>>,
            )> {
                let indices = batch_indices.as_array();
                let (obs, group_ids, metrics) = self.envs.reset(indices);

                let group_ids = group_ids.into_pyarray(py);
                let metrics = metrics
                    .into_iter()
                    .map(|(k, v)| (k, v.into_pyarray(py)))
                    .collect();

                Ok((obs, group_ids, metrics))
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
                Bound<'py, PyArray1<u64>>,
                HashMap<String, Bound<'py, PyArray1<f32>>>,
            )> {
                let indices = batch_indices.as_array();

                let (obs, rewards, dones, group_ids, metrics) = self.envs.step(&indices, &actions);

                let rewards = rewards.into_pyarray(py);
                let dones = dones.into_pyarray(py);
                let group_ids = group_ids.into_pyarray(py);
                let metrics = metrics
                    .into_iter()
                    .map(|(k, v)| (k, v.into_pyarray(py)))
                    .collect();

                Ok((obs, rewards, dones, group_ids, metrics))
            }

            fn instructions(&self) -> PyResult<&'static str> {
                Ok($instr)
            }

            #[getter]
            fn max_turns(&self) -> usize {
                <$rust_env as EnvInstance>::MAX_TURNS
            }
        }
    };
}
