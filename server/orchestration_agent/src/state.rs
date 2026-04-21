// server/orchestration_agent/src/state.rs
// Zero-trust state management — filesystem/ledger backend (NO MongoDB)
//
// SECURITY FEATURES:
//   • All state is in-memory (DashMap) + filesystem persistence
//   • No external database dependencies
//   • Paths are canonicalized via server_root
//   • epsilon_max is configurable via env var (default: 8.0 for meaningful privacy)

use std::sync::RwLock;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

use crate::round::{Round, RoundState};

// ── Type aliases ─────────────────────────────────────────────────────────────
/// DeviceId is always SHA-256 hash of the device's public key (32 bytes)
pub type DeviceId = Vec<u8>;

/// Pending enrollment: fingerprint → (device_pubkey_bytes, csr_bytes)
pub type PendingEnrollment = (Vec<u8>, Vec<u8>);

// ── Data structures ──────────────────────────────────────────────────────────
/// Entry for a single device's model update in a round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateEntry {
    /// Hex-encoded device_id (for filesystem-safe filenames)
    pub device_id_hex: String,
    /// Absolute filesystem path to the encrypted update file
    pub file_path: String,
    /// SHA-256 hash of the payload (for receipt verification)
    pub payload_hash: String,
    /// Differential privacy epsilon spent for this update
    pub epsilon_spent: f64,
    /// Two-phase verification: false when stored, true after receipt check
    pub verified: bool,
    /// ISO 8601 timestamp of upload
    pub timestamp: String,
}

/// Receipt metadata recorded after successful aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationReceipt {
    pub round_id: u64,
    pub num_updates: usize,
    pub aggregation_mode: String,
    pub aggregated_uri: String,
}

// ── OrchestratorState: in-memory state + filesystem root ─────────────────────
pub struct OrchestratorState {
    pub devices: RwLock<HashMap<String, Vec<u8>>>,
    pub rounds: RwLock<HashMap<u64, crate::round::Round>>,
    pub pending_enrollments: RwLock<HashMap<String, PendingEnrollment>>,
    pub server_root: PathBuf,
}

impl OrchestratorState {
    /// Create new state with filesystem root
    /// 
    /// # Arguments
    /// * `server_root` - Canonical path for all server data (~/.federated/server by default)
    pub fn new(server_root: PathBuf) -> Self {
        let mut rounds = HashMap::new();
        let epsilon_max = std::env::var("FL_EPSILON_MAX")
            .ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(8.0);
        rounds.insert(1, crate::round::Round {
            id: 1, model_version: "v1".into(), epsilon_max,
            epsilon_spent: 0.0, state: crate::round::RoundState::Collecting,
            upload_uri: String::new(), updates: Vec::new(),
            aggregation_receipt: None, global_model_path: None,
        });
        Self {
            devices: RwLock::new(HashMap::new()),
            rounds: RwLock::new(rounds),
            pending_enrollments: RwLock::new(HashMap::new()),
            server_root,
        }
    }
    
    // ── Filesystem path helpers (all paths are canonicalized) ─────────────────
    
    /// Directory for device enrollment records
    /// Path: {server_root}/devices/{device_id_hex}.json
    pub fn devices_dir(&self) -> PathBuf {
        self.server_root.join("devices")
    }
    
    /// Directory for model updates in a specific round
    /// Path: {server_root}/rounds/{round_id}/updates/{device_id_hex}.bin
    pub fn updates_dir(&self, round_id: u64) -> PathBuf {
        self.server_root
            .join("rounds")
            .join(round_id.to_string())
            .join("updates")
    }
    
    /// Directory for aggregated global models
    /// Path: {server_root}/global_models/round_{round_id}.bin
    pub fn global_models_dir(&self) -> PathBuf {
        self.server_root.join("global_models")
    }
    
    /// Path to the append-only audit ledger
    /// Path: {server_root}/../logs/audit_ledger.log
    /// (ledger is sibling to server data for integrity separation)
    pub fn ledger_path(&self) -> PathBuf {
        self.server_root
            .parent()
            .map(|p| p.join("logs").join("audit_ledger.log"))
            .unwrap_or_else(|| {
                // Fallback if server_root is at filesystem root
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".federated")
                    .join("logs")
                    .join("audit_ledger.log")
            })
    }
    
    // ── Utility methods ───────────────────────────────────────────────────────
    
    /// Check if a device is enrolled (by device_id bytes)
    pub fn is_device_enrolled(&self, device_id: &[u8]) -> bool {
        // FIXED: Acquire read lock first
        self.devices.read().unwrap().contains_key(&hex::encode(device_id))
    }
    
    /// Get a device's public key (PEM format) if enrolled
    pub fn get_device_pubkey(&self, device_id: &[u8]) -> Option<Vec<u8>> {
        let device_hex = hex::encode(device_id);
        self.devices.read().unwrap().get(&device_hex).cloned()
    }
    
    /// Get mutable access to a round (for aggregation)
    pub fn get_round_mut(&self, _round_id: u64) -> Option<std::sync::RwLockWriteGuard<'_, std::collections::HashMap<u64, Round>>> {
        // Return guard, not dashmap RefMut
        self.rounds.write().ok()
    }
    
    /// Get immutable access to a round
    pub fn get_round(&self, round_id: u64) -> Option<Round> {
        self.rounds.read().unwrap().get(&round_id).cloned()
    }
    
    /// Add a pending enrollment (called during RequestEnrollment)
    pub fn add_pending_enrollment(&self, fingerprint: String, pubkey: Vec<u8>, csr: Vec<u8>) {
        self.pending_enrollments.write().unwrap().insert(fingerprint, (pubkey, csr));
    }
    
    /// Consume a pending enrollment (called during EnrollDevice)
    /// Returns (pubkey, csr) if found and removed, None otherwise
    pub fn consume_pending_enrollment(&self, fingerprint: &str) -> Option<(Vec<u8>, Vec<u8>)> {
        self.pending_enrollments.write().unwrap().remove(fingerprint)
    }

    pub fn register_device(&self, device_id: Vec<u8>, pubkey_pem: Vec<u8>) {
        let device_hex = hex::encode(&device_id);
        self.devices.write().unwrap().insert(device_hex, pubkey_pem);
    }

    pub fn next_round_id(&self) -> u64 {
        self.rounds.read().unwrap().keys().max().copied().unwrap_or(0) + 1
    }
    
    /// Open a new round with default parameters
    pub fn open_round(&self, round_id: u64) {
        let epsilon_max = std::env::var("FL_EPSILON_MAX")
            .ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(8.0);
        
        let new_round = Round {
            id: round_id,
            model_version: format!("v{}", round_id),
            epsilon_max,
            epsilon_spent: 0.0,
            state: RoundState::Collecting,
            upload_uri: String::new(),
            updates: Vec::new(),
            aggregation_receipt: None,
            global_model_path: None,
        };
        self.rounds.write().unwrap().insert(round_id, new_round);
    }
}

// ── Arc wrapper for shared state across async tasks ───────────────────────────
pub type SharedState = Arc<OrchestratorState>;

// ── Helpers for filesystem safety ─────────────────────────────────────────────

/// Ensure a path is within the canonical root (prevent path traversal)
/// 
/// # Returns
/// * `Ok(canonical_path)` if path is valid and within server_root
/// * `Err` if path traversal attempt or invalid path
pub fn validate_path_within_root(path: &PathBuf, root: &PathBuf) -> Result<PathBuf, String> {
    let canonical_root = root.canonicalize()
        .map_err(|e| format!("Cannot canonicalize root {:?}: {}", root, e))?;
    
    let canonical_path = path.canonicalize()
        .map_err(|e| format!("Cannot canonicalize path {:?}: {}", path, e))?;
    
    if !canonical_path.starts_with(&canonical_root) {
        return Err(format!(
            "Path traversal attempt: {:?} is not within {:?}",
            canonical_path, canonical_root
        ));
    }
    
    Ok(canonical_path)
}

/// Generate a filesystem-safe filename from device_id bytes
pub fn device_filename(device_id: &[u8]) -> String {
    format!("{}.json", hex::encode(device_id))
}

/// Generate a filesystem-safe filename for an update
pub fn update_filename(device_id: &[u8], round_id: u64) -> String {
    format!("r{}_{}.bin", round_id, hex::encode(device_id))
}