use crate::crypto::hash_bytes;

/// DeviceId is always derived, never accepted blindly
pub fn derive_device_id(device_pubkey: &[u8]) -> Vec<u8> {
    hash_bytes(device_pubkey).to_vec()
}
