use ed25519_dalek::{Signature, VerifyingKey, Verifier};
use crate::errors::OrchestratorError;

pub fn verify(
    device_pubkey: &[u8],
    msg: &[u8],
    sig: &[u8],
) -> Result<(), OrchestratorError> {

    // Enforce exact public key size (Ed25519 = 32 bytes)
    let pubkey: [u8; 32] = device_pubkey
        .try_into()
        .map_err(|_| OrchestratorError::InvalidIdentity)?;

    // Enforce exact signature size (Ed25519 = 64 bytes)
    let signature_bytes: [u8; 64] = sig
        .try_into()
        .map_err(|_| OrchestratorError::CryptoError)?;

    let verifying_key = VerifyingKey::from_bytes(&pubkey)
        .map_err(|_| OrchestratorError::CryptoError)?;

    let signature = Signature::from_bytes(&signature_bytes);

    verifying_key
        .verify(msg, &signature)
        .map_err(|_| OrchestratorError::CryptoError)?;

    Ok(())
}
