// receipts.rs
//
// FIX: VerifyingKey::from_sec1_bytes expects raw SEC1 EC point bytes
//      (65 bytes: 0x04 || X[32] || Y[32]) but device_pubkey arrives as a
//      PEM-encoded SPKI structure produced by windows_signer --pubkey and
//      tpm2_readpublic.  The SPKI DER for P-256 has a fixed 26-byte header
//      before the 65-byte EC point.  We must strip the PEM wrapper, base64-
//      decode to DER, skip the header, then call from_sec1_bytes.

use base64::{engine::general_purpose::STANDARD, Engine};
use p256::ecdsa::{signature::Verifier, Signature, VerifyingKey};
use crate::errors::OrchestratorError;

/// Parse a PEM-encoded P-256 public key (SPKI format) into a VerifyingKey.
///
/// windows_signer and tpm2_readpublic both emit:
///   -----BEGIN PUBLIC KEY-----
///   <base64-encoded DER SPKI>
///   -----END PUBLIC KEY-----
///
/// The SPKI DER for an uncompressed P-256 key is always 91 bytes:
///   26-byte header  (SEQUENCE + AlgorithmIdentifier for id-ecPublicKey + prime256v1)
///   65-byte payload (BIT STRING containing 0x04 || X[32] || Y[32])
///
/// from_sec1_bytes only wants those 65 payload bytes.
fn pem_to_verifying_key(pem_bytes: &[u8]) -> Result<VerifyingKey, OrchestratorError> {
    // 1. Decode UTF-8
    let pem_str = std::str::from_utf8(pem_bytes)
        .map_err(|_| OrchestratorError::CryptoError)?;

    // 2. Strip PEM header/footer lines, concatenate the base64 body
    let b64: String = pem_str
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");

    if b64.trim().is_empty() {
        tracing::error!("receipts::verify — empty base64 after stripping PEM headers");
        return Err(OrchestratorError::CryptoError);
    }

    // 3. Base64-decode to DER bytes
    let der = STANDARD
        .decode(b64.trim())
        .map_err(|_| {
            tracing::error!("receipts::verify — base64 decode failed");
            OrchestratorError::CryptoError
        })?;

    // 4. Validate length: SPKI P-256 uncompressed = 91 bytes
    //    26-byte header + 65-byte EC point
    const SPKI_HEADER: usize = 26;
    const EC_POINT_LEN: usize = 65; // 0x04 || X[32] || Y[32]
    const EXPECTED_LEN: usize = SPKI_HEADER + EC_POINT_LEN; // 91

    if der.len() < EXPECTED_LEN {
        tracing::error!(
            "receipts::verify — DER too short: got {} bytes, expected >= {}",
            der.len(), EXPECTED_LEN
        );
        return Err(OrchestratorError::CryptoError);
    }

    // 5. Extract the raw EC point (skip the 26-byte SPKI prefix)
    let ec_point = &der[SPKI_HEADER..SPKI_HEADER + EC_POINT_LEN];

    if ec_point[0] != 0x04 {
        tracing::error!(
            "receipts::verify — EC point does not start with 0x04 (uncompressed marker), \
             got 0x{:02x}. Key may be compressed or corrupted.",
            ec_point[0]
        );
        return Err(OrchestratorError::CryptoError);
    }

    // 6. Build the VerifyingKey from the raw SEC1 EC point bytes
    VerifyingKey::from_sec1_bytes(ec_point)
        .map_err(|e| {
            tracing::error!("receipts::verify — from_sec1_bytes failed: {:?}", e);
            OrchestratorError::InvalidIdentity
        })
}

pub fn verify(
    device_pubkey: &[u8],
    msg: &[u8],
    sig: &[u8],
) -> Result<(), OrchestratorError> {
    // Parse PEM → DER → EC point → VerifyingKey
    let verifying_key = pem_to_verifying_key(device_pubkey)?;

    // Signature arrives as DER-encoded ECDSA signature from windows_signer --sign
    let signature = Signature::from_der(sig).map_err(|e| {
        tracing::error!("receipts::verify — Signature::from_der failed: {:?}", e);
        OrchestratorError::CryptoError
    })?;

    verifying_key.verify(msg, &signature).map_err(|e| {
        tracing::error!("receipts::verify — signature verification failed: {:?}", e);
        OrchestratorError::CryptoError
    })?;

    Ok(())
}