// round.rs — RoundState derives Debug so format!("{:?}", round.state) compiles.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundState {
    Open,
    Collecting,
    Aggregating,
    Complete,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateEntry {
    pub device_id_hex: String,
    pub file_path: String,
    pub payload_hash: String,
    pub epsilon_spent: f64,
    pub verified: bool,
    pub timestamp: String,
}

// FIXED: Single Round struct with NO duplicate fields
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Round {
    pub id:                  u64,
    pub model_version:       String,
    pub epsilon_max:         f64,
    pub epsilon_spent:       f64,
    pub upload_uri:          String,
    pub state:               RoundState,
    pub updates:             Vec<UpdateEntry>,
    pub aggregation_receipt: Option<AggregationReceipt>,
    pub global_model_path:   Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregationReceipt {
    pub round_id: u64,
    pub num_updates: usize,
    pub aggregation_mode: String,
    pub aggregated_uri: String,
}