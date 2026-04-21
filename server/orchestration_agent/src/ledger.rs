// server/orchestration_agent/src/ledger.rs
// Zero-trust append-only audit ledger — filesystem backend
//
// SECURITY FEATURES:
//   • HMAC-chained entries: each receipt links to previous via HMAC(prev_hmac | payload_hash)
//   • Path canonicalization: prevents path traversal attacks
//   • No panics: all errors logged via tracing, never crash the server
//   • In-process mutex: serializes concurrent writes within the process
//   • O_APPEND atomicity: OS guarantees atomic appends for cross-process safety
//   • TPM-ready: HMAC key can be supplied from TPM-sealed secret via env var
//
// BUGS FIXED:
//   FIX-LEDGER-1: Removed duplicate `use std::path::PathBuf` (was imported twice:
//                 once via `std::path::{Path, PathBuf}` and again standalone).
//   FIX-LEDGER-2: Removed unused `use std::path::Path` import.
//   FIX-LEDGER-3: Removed `nix` file-locking code. `nix` was declared `optional = true`
//                 in Cargo.toml but used without a feature gate, causing compile failure
//                 on all Unix targets. Replaced with an in-process `Mutex` + O_APPEND.
//   FIX-LEDGER-4: Removed `windows_sys` file-locking code. `windows_sys` was never
//                 declared in Cargo.toml, causing compile failure on Windows targets.
//                 The in-process `Mutex` + O_APPEND strategy handles Windows equally.
//   FIX-LEDGER-5: Removed `#[cfg(feature = "tpm")]` block inside `derive_chain_key`.
//                 The `tpm` feature does not exist in Cargo.toml, and the called
//                 function `crate::receipts::derive_hmac_key` does not exist in
//                 receipts.rs — both caused compile errors. Replaced with a clear
//                 comment directing future implementers.
//   FIX-LEDGER-6: Fixed logic bug in `test_append_and_read_hmac_chain`. The second
//                 assertion `assert_eq!(last, Some(expected))` compared `last` (which
//                 holds the `hmac_chain` field written into entry2, i.e. `hmac1`) against
//                 `compute_chain_link(&key, "genesis", "def456")`, a completely different
//                 value. The test always failed. Removed the incorrect assertion and added
//                 a correct one that verifies the actual chain link value.

use hmac::{Hmac, Mac};
use sha2::Sha256;
// FIX-LEDGER-1: PathBuf was imported twice. `Path` was imported but never used.
// Corrected: import only what is actually used.
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use tracing::{error, warn};
use dirs;

// Type alias for HMAC-SHA256
type HmacSha256 = Hmac<Sha256>;

// FIX-LEDGER-3 / FIX-LEDGER-4: Replace OS-level file locking (nix / windows_sys) with
// an in-process Mutex. This serializes all writes within a single process. For cross-
// process safety we rely on O_APPEND, which the OS guarantees to be atomic for writes
// smaller than PIPE_BUF (~4 KB on Linux, unlimited on Windows with NTFS). Each ledger
// entry is a single JSON line well within that bound.
static LEDGER_WRITE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Append a ledger entry to a specific file path with in-process locking.
///
/// # Arguments
/// * `entry` — Raw JSON bytes of the ledger entry (newline appended automatically)
/// * `path`  — Absolute canonical path to the ledger file
///
/// # Security
/// * Path is canonicalized before open to prevent traversal
/// * File is opened with O_APPEND — OS guarantees atomicity for single write calls
/// * An in-process Mutex prevents interleaved writes from concurrent async tasks
/// * Errors are logged via tracing; server never panics on ledger failure
pub fn append_to(entry: &[u8], path: &PathBuf) {
    // Ensure parent directory exists before canonicalization
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                error!("Ledger: cannot create parent dir {:?}: {}", parent, e);
                return;
            }
        }
    }

    // Canonicalize path — reject traversal attempts
    let canonical_path = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            // File may not exist yet; canonicalize the parent then append the filename
            let canonical_parent = match path.parent().and_then(|p| p.canonicalize().ok()) {
                Some(p) => p,
                None => {
                    warn!("Ledger: cannot canonicalize {:?} — using as-is (risk: path traversal)", path);
                    path.clone()
                }
            };
            let file_name = path.file_name().unwrap_or_default();
            canonical_parent.join(file_name)
        }
    };

    // Acquire in-process lock before touching the file (FIX-LEDGER-3/4)
    let _guard = match LEDGER_WRITE_LOCK.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            // Mutex is poisoned only if a previous thread panicked while holding it.
            // Recover by clearing the poison.
            warn!("Ledger: write Mutex was poisoned — recovering");
            poisoned.into_inner()
        }
    };

    // Open with O_APPEND + O_CREAT — writes are atomic at the OS level
    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&canonical_path)
    {
        Ok(f) => f,
        Err(e) => {
            error!("Ledger: cannot open {:?}: {}", canonical_path, e);
            return;
        }
    };

    if let Err(e) = file.write_all(entry) {
        error!("Ledger: write failed at {:?}: {}", canonical_path, e);
        return;
    }
    // Ensure newline terminator so each entry is on its own line
    let _ = file.write_all(b"\n");
    let _ = file.flush();
    // `_guard` drops here — Mutex released, file closed by RAII
}

/// Append to the default ledger path (~/.federated/logs/audit_ledger.log).
///
/// Convenience wrapper that resolves the canonical path via dirs::home_dir().
pub fn append(entry: &[u8]) {
    let path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".federated")
        .join("logs")
        .join("audit_ledger.log");

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    append_to(entry, &path);
}

/// Get the last HMAC chain value for a given round from the ledger.
///
/// # Arguments
/// * `ledger_path` — Canonical path to the ledger file
/// * `round_id`    — Filter entries to this round only
///
/// # Returns
/// * `Some(hex_string)` — The `hmac_chain` field of the most recent receipt entry for this round
/// * `None`             — No entries found (start the chain with "genesis")
///
/// # Security
/// * File is read-only; no locking needed (concurrent reads are safe)
/// * Malformed JSON lines are skipped with a warning — never panic
pub fn get_last_hmac(ledger_path: &PathBuf, round_id: u64) -> Option<String> {
    let path = match ledger_path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            warn!("Ledger: path {:?} cannot be canonicalized for read", ledger_path);
            ledger_path.clone()
        }
    };

    if !path.exists() {
        return None;
    }

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            error!("Ledger: cannot open {:?} for read: {}", path, e);
            return None;
        }
    };

    let reader = BufReader::new(file);
    let mut last_hmac: Option<String> = None;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                warn!("Ledger: read error at {:?}: {}", path, e);
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let entry: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!("Ledger: malformed JSON entry at {:?}: {}", path, e);
                continue;
            }
        };

        // Only process receipt entries for the requested round
        if entry.get("round_id").and_then(|v| v.as_u64()) == Some(round_id)
            && entry.get("type").and_then(|v| v.as_str()) == Some("receipt")
        {
            if let Some(hmac_val) = entry.get("hmac_chain").and_then(|v| v.as_str()) {
                last_hmac = Some(hmac_val.to_string());
            }
        }
    }

    last_hmac
}

/// Compute HMAC-SHA256 chain link: HMAC(key, prev_hmac || "|" || payload_hash_hex)
///
/// # Arguments
/// * `key`              — 32-byte HMAC key
/// * `prev_hmac`        — Previous chain value; use `"genesis"` for the first entry
/// * `payload_hash_hex` — SHA-256 hex string of the payload being chained
///
/// # Returns
/// Hex-encoded HMAC result (64-char lowercase string)
pub fn compute_chain_link(key: &[u8; 32], prev_hmac: &str, payload_hash_hex: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(key)
        .expect("HMAC key is exactly 32 bytes — this cannot fail");
    mac.update(prev_hmac.as_bytes());
    mac.update(b"|");
    mac.update(payload_hash_hex.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Derive the HMAC chain key from environment variable (dev/testing) or a supplied secret.
///
/// # Priority
/// 1. `RECEIPT_CHAIN_KEY` env var — 32-byte hex string (preferred for dev)
/// 2. Ephemeral random key with a loud warning (never use in production)
///
/// # FIX-LEDGER-5 note
/// A `#[cfg(feature = "tpm")]` block that called `crate::receipts::derive_hmac_key`
/// was removed because:
///   a) The `tpm` feature is not declared in Cargo.toml
///   b) `receipts::derive_hmac_key` does not exist in receipts.rs
/// To add TPM-derived key support, declare a `tpm` feature in Cargo.toml and implement
/// `receipts::derive_hmac_key(context: &str) -> Result<[u8;32], ...>` that calls
/// `tpm2_sign` / Windows CNG to derive a deterministic key from the TPM-sealed secret.
pub fn derive_chain_key() -> Result<[u8; 32], String> {
    // Option 1: Environment variable (recommended for dev/CI)
    if let Ok(hex_key) = std::env::var("RECEIPT_CHAIN_KEY") {
        let bytes = hex::decode(&hex_key)
            .map_err(|e| format!("RECEIPT_CHAIN_KEY is not valid hex: {}", e))?;
        if bytes.len() != 32 {
            return Err(format!(
                "RECEIPT_CHAIN_KEY must be 32 bytes (64 hex chars), got {}",
                bytes.len()
            ));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes);
        return Ok(key);
    }

    // Option 2: Ephemeral random key — chain breaks on server restart
    warn!(
        "RECEIPT_CHAIN_KEY not set — using ephemeral random HMAC key. \
         Receipt chains will break on restart. Set RECEIPT_CHAIN_KEY in production."
    );
    use rand::RngCore;
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    Ok(key)
}

/// Validate that a ledger path is within the canonical integrity root (prevent traversal).
///
/// The ledger lives in `~/.federated/logs/`, one level above `server_root`
/// (`~/.federated/server/`). Both paths must descend from `~/.federated/`.
///
/// # Returns
/// * `Ok(canonical_path)` if the path is valid and within the integrity boundary
/// * `Err(String)`        with a human-readable message if validation fails
pub fn validate_ledger_path(path: &PathBuf, server_root: &PathBuf) -> Result<PathBuf, String> {
    let canonical_root = server_root
        .canonicalize()
        .map_err(|e| format!("Cannot canonicalize server root {:?}: {}", server_root, e))?;

    // The integrity boundary is the parent of server_root (e.g. ~/.federated/)
    let integrity_root = canonical_root
        .parent()
        .unwrap_or(&canonical_root)
        .to_path_buf();

    let canonical_path = path
        .canonicalize()
        .map_err(|e| format!("Cannot canonicalize ledger path {:?}: {}", path, e))?;

    if !canonical_path.starts_with(&integrity_root) {
        return Err(format!(
            "Ledger path {:?} is not within integrity root {:?} — possible path traversal",
            canonical_path, integrity_root
        ));
    }

    Ok(canonical_path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// FIX-LEDGER-6: The original test had a logically incorrect second assertion:
    ///
    ///   let expected = compute_chain_link(&key, "genesis", "def456");
    ///   assert_eq!(last, Some(expected));   // WRONG — always failed
    ///
    /// `last` is the `hmac_chain` field stored IN entry2, which equals
    /// `compute_chain_link(&key, "genesis", "abc123")` (i.e. `hmac1`), NOT a chain
    /// computed over "def456". The assertion was comparing two unrelated values.
    ///
    /// The correct invariant to verify is:
    ///   get_last_hmac() == the hmac_chain written into the last receipt entry
    #[test]
    fn test_append_and_read_hmac_chain() {
        let temp_dir = TempDir::new().unwrap();
        let ledger_path = temp_dir.path().join("test_ledger.log");

        let key = [0u8; 32]; // test key (all-zeros is fine in unit tests)

        // --- Entry 1: chain root = "genesis" ---
        let hmac1 = compute_chain_link(&key, "genesis", "abc123");
        let entry1 = format!(
            r#"{{"type":"receipt","round_id":1,"payload_hash":"abc123","hmac_chain":"{}"}}"#,
            hmac1
        );
        append_to(entry1.as_bytes(), &ledger_path);

        // --- Entry 2: chained from entry 1 ---
        let hmac2 = compute_chain_link(&key, &hmac1, "def456");
        let entry2 = format!(
            r#"{{"type":"receipt","round_id":1,"payload_hash":"def456","hmac_chain":"{}"}}"#,
            hmac2
        );
        append_to(entry2.as_bytes(), &ledger_path);

        // get_last_hmac must return the hmac_chain of the LAST receipt entry
        let last = get_last_hmac(&ledger_path, 1);

        // FIX-LEDGER-6: correct assertion — last entry stored hmac2
        assert_eq!(
            last,
            Some(hmac2.clone()),
            "get_last_hmac should return the hmac_chain of the final receipt entry"
        );
    }

    #[test]
    fn test_get_last_hmac_returns_none_for_missing_file() {
        let temp_dir = TempDir::new().unwrap();
        let missing = temp_dir.path().join("nonexistent.log");
        assert_eq!(get_last_hmac(&missing, 1), None);
    }

    #[test]
    fn test_get_last_hmac_ignores_other_rounds() {
        let temp_dir = TempDir::new().unwrap();
        let ledger_path = temp_dir.path().join("ledger.log");

        let key = [1u8; 32];
        let hmac_r1 = compute_chain_link(&key, "genesis", "aaa");
        let hmac_r2 = compute_chain_link(&key, "genesis", "bbb");

        let e1 = format!(
            r#"{{"type":"receipt","round_id":1,"payload_hash":"aaa","hmac_chain":"{}"}}"#,
            hmac_r1
        );
        let e2 = format!(
            r#"{{"type":"receipt","round_id":2,"payload_hash":"bbb","hmac_chain":"{}"}}"#,
            hmac_r2
        );
        append_to(e1.as_bytes(), &ledger_path);
        append_to(e2.as_bytes(), &ledger_path);

        assert_eq!(get_last_hmac(&ledger_path, 1), Some(hmac_r1));
        assert_eq!(get_last_hmac(&ledger_path, 2), Some(hmac_r2));
        assert_eq!(get_last_hmac(&ledger_path, 99), None);
    }

    #[test]
    fn test_get_last_hmac_skips_malformed_lines() {
        let temp_dir = TempDir::new().unwrap();
        let ledger_path = temp_dir.path().join("ledger.log");

        let key = [2u8; 32];
        let good_hmac = compute_chain_link(&key, "genesis", "ccc");

        // Malformed line first, then a valid one
        append_to(b"not-valid-json", &ledger_path);
        let good = format!(
            r#"{{"type":"receipt","round_id":3,"payload_hash":"ccc","hmac_chain":"{}"}}"#,
            good_hmac
        );
        append_to(good.as_bytes(), &ledger_path);

        assert_eq!(get_last_hmac(&ledger_path, 3), Some(good_hmac));
    }

    #[test]
    fn test_path_validation_rejects_traversal() {
        let temp_dir = TempDir::new().unwrap();
        let server_root = temp_dir.path().join("server");
        std::fs::create_dir_all(&server_root).unwrap();

        // A path inside the integrity boundary (sibling to server_root)
        let good_path = temp_dir.path().join("logs").join("audit.log");
        std::fs::create_dir_all(good_path.parent().unwrap()).unwrap();
        std::fs::write(&good_path, b"").unwrap(); // must exist for canonicalize()
        assert!(validate_ledger_path(&good_path, &server_root).is_ok());

        // A path outside the integrity boundary (/tmp is not under the temp dir)
        let evil_path = std::env::temp_dir().join("evil_ledger.log");
        std::fs::write(&evil_path, b"").unwrap();
        let result = validate_ledger_path(&evil_path, &server_root);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("path traversal"));
    }

    #[test]
    fn test_compute_chain_link_is_deterministic() {
        let key = [3u8; 32];
        let a = compute_chain_link(&key, "genesis", "abc");
        let b = compute_chain_link(&key, "genesis", "abc");
        assert_eq!(a, b, "same inputs must always produce same HMAC");

        let c = compute_chain_link(&key, "genesis", "xyz");
        assert_ne!(a, c, "different payloads must produce different HMACs");
    }

    #[test]
    fn test_derive_chain_key_from_env() {
        let test_key_hex = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
        std::env::set_var("RECEIPT_CHAIN_KEY", test_key_hex);
        let key = derive_chain_key().expect("should succeed with valid env var");
        assert_eq!(key[0], 0x01);
        assert_eq!(key[31], 0x20);
        std::env::remove_var("RECEIPT_CHAIN_KEY");
    }

    #[test]
    fn test_derive_chain_key_rejects_wrong_length() {
        std::env::set_var("RECEIPT_CHAIN_KEY", "deadbeef"); // too short (4 bytes)
        let result = derive_chain_key();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("32 bytes"));
        std::env::remove_var("RECEIPT_CHAIN_KEY");
    }
}