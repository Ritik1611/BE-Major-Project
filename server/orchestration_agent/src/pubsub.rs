use crate::state::OrchestratorState;
use std::sync::Arc;

pub fn start(state: Arc<OrchestratorState>) {
    tokio::spawn(async move {
        loop {
            {
                // Lock is dropped at end of this block — before the await
                let rounds = state.rounds.read().unwrap();
                for (_, r) in rounds.iter() {
                    tracing::info!("Round {} active (model {})", r.id, r.model_version);
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
        }
    });
}