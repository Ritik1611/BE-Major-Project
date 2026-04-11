// main.rs

mod config;
mod crypto;
mod errors;
mod grpc;
mod identity;
mod ledger;
mod pubsub;
mod receipts;
mod round;
mod state;
mod otp;

use crate::config::Config;
use crate::state::OrchestratorState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── 1. Init logging ───────────────────────────────────────────────────────
    tracing_subscriber::fmt::init();

    // ── 2. Load config ────────────────────────────────────────────────────────
    let cfg = Config::load("config/orchestrator.toml")?;

    // ── 3. Connect to MongoDB ──────────────────────────────────────────────────
    // FIX: MongoDB URI from environment — never hardcoded
    let mongo_uri = std::env::var("MONGO_URI")
        .unwrap_or_else(|_| {
            tracing::warn!("MONGO_URI not set — using localhost default");
            "mongodb://localhost:27017".to_string()
        });

    let mongo = mongodb::Client::with_uri_str(&mongo_uri).await?;

    // Verify connectivity
    mongo.database("federated")
        .run_command(mongodb::bson::doc! { "ping": 1 }, None)
        .await
        .map_err(|e| anyhow::anyhow!("MongoDB connection failed: {}", e))?;

    tracing::info!("MongoDB connected: {}", mongo_uri);

    // Ensure indexes exist
    _ensure_indexes(&mongo).await?;

    // ── 4. Create shared orchestrator state ───────────────────────────────────
    let state = OrchestratorState::new();

    // ── 5. Bootstrap enrollment OTP ───────────────────────────────────────────
    let otp = crate::otp::generate_otp();
    state.enrollment_tokens.insert(otp.clone(), ());
    println!("[DEV] Enrollment OTP: {} (valid for 10 minutes)", otp);
    tracing::info!("Enrollment OTP generated — valid 10 minutes");

    // ── 6. Start background systems ───────────────────────────────────────────
    pubsub::start(state.clone());

    // ── 7. Start gRPC server ──────────────────────────────────────────────────
    grpc::server::serve(cfg, state, mongo).await?;

    Ok(())
}

async fn _ensure_indexes(mongo: &mongodb::Client) -> anyhow::Result<()> {
    use mongodb::bson::doc;
    use mongodb::IndexModel;
    use mongodb::options::IndexOptions;

    let db = mongo.database("federated");

    // devices: unique index on device_id
    db.collection::<mongodb::bson::Document>("devices")
        .create_index(
            IndexModel::builder()
                .keys(doc! { "device_id": 1 })
                .options(IndexOptions::builder().unique(true).build())
                .build(),
            None,
        ).await?;

    // model_updates: compound index for receipt verification lookup
    db.collection::<mongodb::bson::Document>("model_updates")
        .create_index(
            IndexModel::builder()
                .keys(doc! { "file_id": 1, "device_id": 1, "round_id": 1 })
                .build(),
            None,
        ).await?;

    // receipts: index for chain lookup (most recent per round)
    db.collection::<mongodb::bson::Document>("receipts")
        .create_index(
            IndexModel::builder()
                .keys(doc! { "round_id": 1, "_id": -1 })
                .build(),
            None,
        ).await?;

    // global_models: index on round_id
    db.collection::<mongodb::bson::Document>("global_models")
        .create_index(
            IndexModel::builder()
                .keys(doc! { "round_id": 1 })
                .options(IndexOptions::builder().unique(true).build())
                .build(),
            None,
        ).await?;

    tracing::info!("MongoDB indexes ensured");
    Ok(())
}