// server/orchestration_agent/src/main.rs
// Zero-trust orchestrator entrypoint — filesystem/ledger backend (NO MongoDB)

// ── MODULE DECLARATIONS (MUST come before ANY use statements) ────────────────
mod config;
mod crypto;
mod errors;
mod grpc;
mod identity;
mod ledger;
mod otp;
mod pubsub;
mod receipts;
mod round;
mod state;

// ── IMPORTS (after module declarations) ─────────────────────────────────────
use std::path::PathBuf;
use std::sync::Arc;
use std::env;

use anyhow::{Context, Result};
use dirs;
use tokio::signal;
use tracing_subscriber;

use crate::config::Config;
use crate::state::OrchestratorState;
use crate::grpc::server::serve;

// ── MAIN ENTRYPOINT ─────────────────────────────────────────────────────────
#[tokio::main]
async fn main() -> Result<()> {
    // ── 1. Initialize structured logging ───────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,orchestrator=debug,tower=warn".into())
        )
        .with_target(false)
        .with_thread_ids(true)
        .init();

    tracing::info!("🚀 Federated Orchestrator starting (filesystem/ledger backend)");

    // ── 2. Load configuration ──────────────────────────────────────────────
    let config_path = env::var("CONFIG_PATH").unwrap_or_else(|_| "config/orchestrator.toml".into());
    let cfg = Config::load(&config_path)
        .with_context(|| format!("Failed to load config from '{}'", config_path))?;
    tracing::info!("Configuration loaded from: {}", config_path);

    // ── 3. Resolve canonical server root ───────────────────────────────────
    let server_root: PathBuf = env::var("FL_SERVER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            tracing::warn!("FL_SERVER_ROOT not set — using default ~/.federated/server");
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".federated")
                .join("server")
        });

    // ── 4. Validate & create required directories ──────────────────────────
    let dirs_to_verify = [
        server_root.clone(),
        server_root.join("devices"),
        server_root.join("rounds"),
        server_root.join("global_models"),
        server_root.join("logs"),
    ];
    for dir in &dirs_to_verify {
        if !dir.exists() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("Failed to create directory: {:?}", dir))?;
            tracing::info!("Created directory: {:?}", dir);
        }
    }
    tracing::info!("Server root verified: {:?}", server_root);

    // ── 5. Validate TLS certificates & keys ────────────────────────────────
    verify_file_exists(&cfg.tls.ca_cert, "CA certificate")?;
    verify_file_exists(&cfg.tls.ca_key, "CA private key")?;
    verify_file_exists(&cfg.tls.server_cert, "Server certificate")?;
    verify_file_exists(&cfg.tls.server_key, "Server private key")?;
    tracing::info!("TLS certificates verified successfully");

    // ── 6. Validate receipt chaining key ───────────────────────────────────
    if env::var("RECEIPT_CHAIN_KEY").is_err() {
        tracing::warn!(
            "⚠️  RECEIPT_CHAIN_KEY not set. Server will use an EPHEMERAL HMAC key. \
             Receipt chains will break on restart. Set in production."
        );
    }

    // ── 7. Initialize in-memory state ──────────────────────────────────────
    let state = Arc::new(OrchestratorState::new(server_root.clone()));
    tracing::info!("Orchestrator state initialized (Round 1: Collecting)");

    // ── 8. Start dual-port gRPC servers ────────────────────────────────────
    let server_handle = tokio::spawn(async move {
        serve(cfg, state).await
    });

    // ── 9. Graceful shutdown on SIGINT/SIGTERM ─────────────────────────────
    tracing::info!("✅ Orchestrator running. Press Ctrl+C to stop.");
    signal::ctrl_c()
        .await
        .context("Failed to install signal handler")?;

    tracing::info!("🛑 Shutdown signal received. Flushing state & exiting...");
    let _ = server_handle.await?;
    tracing::info!("👋 Orchestrator stopped cleanly.");

    Ok(())
}

// ── Helper: verify file exists and has secure permissions ──────────────────
fn verify_file_exists(path: &str, description: &str) -> Result<()> {
    let p = PathBuf::from(path);
    if !p.exists() {
        return Err(anyhow::anyhow!("{} not found: {:?}", description, p));
    }
    if !p.is_file() {
        return Err(anyhow::anyhow!("{} is not a file: {:?}", description, p));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let meta = std::fs::metadata(&p)
            .with_context(|| format!("Cannot stat {}: {:?}", description, p))?;
        let mode = meta.permissions().mode();
        let octal = mode & 0o777;
        if description.contains("key") && (octal & 0o77 != 0) {
            tracing::warn!(
                "⚠️  {} has insecure permissions ({:o}). Should be 0600. \
                 Run: chmod 600 {}",
                description,
                octal,
                p.display()
            );
        }
    }
    Ok(())
}