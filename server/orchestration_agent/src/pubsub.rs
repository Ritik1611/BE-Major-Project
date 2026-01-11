use crate::state::OrchestratorState;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

pub fn start(state: Arc<OrchestratorState>) {
    tokio::spawn(async move {
        loop {
            for r in state.rounds.iter() {
                tracing::info!(
                    "Round {} active (model {})",
                    r.id,
                    r.model_version
                );
            }
            sleep(Duration::from_secs(10)).await;
        }
    });
}
