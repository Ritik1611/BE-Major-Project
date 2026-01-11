use crate::round::Round;
use dashmap::DashMap;
use std::sync::Arc;

pub type DeviceId = Vec<u8>;

pub struct OrchestratorState {
    pub devices: DashMap<DeviceId, ()>,
    pub rounds: DashMap<u64, Round>,
}

impl OrchestratorState {
    pub fn new() -> Arc<Self> {
        let s = Arc::new(Self {
            devices: DashMap::new(),
            rounds: DashMap::new(),
        });

        s.rounds.insert(1, Round::new(1));
        s
    }
}
