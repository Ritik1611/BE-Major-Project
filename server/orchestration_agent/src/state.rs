use dashmap::DashMap;
use std::sync::Arc;
use crate::round::{Round, RoundState};

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

        s.rounds.insert(1, Round {
            id: 1,
            model_version: "v1".into(),
            epsilon_max: 1.0,
            upload_uri: "objectstore://round-1".into(),
            state: RoundState::Collecting,
            updates: Vec::new(),
            aggregation_receipt: None,
            epsilon_spent: 0.0,
        });

        s
    }
}

