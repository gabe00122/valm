use rand::{RngCore, SeedableRng, rngs::SmallRng};

pub struct GroupSequence {
    rng: SmallRng,
    group_size: usize,
    position_in_group: usize,
    group_id: u64,
}

impl GroupSequence {
    pub fn new(seed: u64, group_size: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let group_id = rng.next_u64();
        Self {
            rng: rng,
            group_size: group_size,
            position_in_group: 0,
            group_id: group_id,
        }
    }

    pub fn take_group_id(&mut self) -> u64 {
        if self.position_in_group >= self.group_size {
            self.position_in_group = 0;
            self.group_id = self.rng.next_u64();
        }
        self.position_in_group += 1;

        self.group_id
    }
}
