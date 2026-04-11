// otp.rs — Per-device OTP with 10-minute expiry and rate limiting
//
// SECURITY FIX:
//   FIX-OTP-1: OTP_EXPIRY_SECS was 6000 (100 minutes). Comment claimed 60 seconds.
//              Changed to 600 seconds (10 minutes) — short enough to limit
//              exposure, long enough for operator workflow.
//              If you need stricter security, reduce to 300 (5 minutes).

use once_cell::sync::Lazy;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// OTP expires after 10 minutes. Previously was 6000s (100 min) by mistake.
const OTP_EXPIRY_SECS:      u64 = 600;
const MAX_FAILED_ATTEMPTS:  u32 = 5;
const LOCKOUT_DURATION_SECS: u64 = 300; // 5-minute lockout after MAX_FAILED_ATTEMPTS

#[derive(Debug)]
struct OtpEntry {
    token:       String,
    created_at:  Instant,
    device_hint: Option<String>,
    used:        bool,
}

#[derive(Debug, Default)]
struct RateLimitEntry {
    failed_attempts: u32,
    locked_until:    Option<Instant>,
}

static OTP_STORE: Lazy<Mutex<HashMap<String, OtpEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static RATE_LIMITER: Lazy<Mutex<HashMap<String, RateLimitEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Generate a cryptographically random 6-digit OTP.
pub fn generate_otp() -> String {
    generate_otp_for(None)
}

pub fn generate_otp_for(device_hint: Option<String>) -> String {
    let mut rng = thread_rng();
    let otp: u32 = rng.gen_range(100_000..=999_999);
    let token = otp.to_string();

    let mut store = OTP_STORE.lock().unwrap();
    _purge_expired(&mut store);

    store.insert(
        token.clone(),
        OtpEntry {
            token:       token.clone(),
            created_at:  Instant::now(),
            device_hint,
            used:        false,
        },
    );

    token
}

/// Consume an OTP. Returns true only if valid, unexpired, unused, and not rate-limited.
pub fn consume_otp(token: &str) -> bool {
    consume_otp_from(token, "unknown")
}

pub fn consume_otp_from(token: &str, device_id: &str) -> bool {
    // Rate limit check
    {
        let mut rl = RATE_LIMITER.lock().unwrap();
        let entry = rl.entry(device_id.to_string()).or_default();

        if let Some(locked_until) = entry.locked_until {
            if Instant::now() < locked_until {
                tracing::warn!("OTP attempt from locked device '{}'", device_id);
                return false;
            } else {
                entry.failed_attempts = 0;
                entry.locked_until    = None;
            }
        }
    }

    let mut store = OTP_STORE.lock().unwrap();
    _purge_expired(&mut store);

    match store.get_mut(token) {
        None => {
            tracing::warn!("OTP '{}' not found or expired", token);
            _record_failure(device_id);
            false
        }
        Some(entry) if entry.used => {
            tracing::warn!("OTP '{}' already consumed", token);
            _record_failure(device_id);
            false
        }
        Some(entry) if entry.created_at.elapsed() > Duration::from_secs(OTP_EXPIRY_SECS) => {
            tracing::warn!(
                "OTP '{}' expired (age={:.0}s > {}s)",
                token, entry.created_at.elapsed().as_secs_f64(), OTP_EXPIRY_SECS
            );
            store.remove(token);
            _record_failure(device_id);
            false
        }
        Some(entry) => {
            entry.used = true;
            tracing::info!(
                "OTP consumed by '{}' (age={:.0}s)",
                device_id, entry.created_at.elapsed().as_secs_f64()
            );
            // Reset failure count on success
            let mut rl = RATE_LIMITER.lock().unwrap();
            rl.remove(device_id);
            true
        }
    }
}

#[allow(dead_code)]
pub fn list_active_otps() -> Vec<String> {
    let mut store = OTP_STORE.lock().unwrap();
    _purge_expired(&mut store);
    store.keys().cloned().collect()
}

fn _purge_expired(store: &mut HashMap<String, OtpEntry>) {
    let expiry = Duration::from_secs(OTP_EXPIRY_SECS);
    store.retain(|_, v| !v.used && v.created_at.elapsed() <= expiry);
}

fn _record_failure(device_id: &str) {
    let mut rl    = RATE_LIMITER.lock().unwrap();
    let entry     = rl.entry(device_id.to_string()).or_default();
    entry.failed_attempts += 1;
    tracing::warn!(
        "OTP failure #{} from '{}'", entry.failed_attempts, device_id
    );
    if entry.failed_attempts >= MAX_FAILED_ATTEMPTS {
        let until = Instant::now() + Duration::from_secs(LOCKOUT_DURATION_SECS);
        entry.locked_until = Some(until);
        tracing::error!(
            "Device '{}' locked for {}s after {} failures",
            device_id, LOCKOUT_DURATION_SECS, entry.failed_attempts
        );
    }
}