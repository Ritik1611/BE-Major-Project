use std::io::{Read, Write};
use windows::core::*;
use windows::Win32::Security::Cryptography::*;
use windows::Win32::Foundation::*;
use sha2::{Sha256, Digest};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "--export-pub" {
        export_public_key()?;
        return Ok(());
    }

    sign_stdin()?;
    Ok(())
}

fn open_or_create_key() -> Result<NCRYPT_KEY_HANDLE> {
    unsafe {
        let mut provider = NCRYPT_PROV_HANDLE::default();

        NCryptOpenStorageProvider(
            &mut provider,
            w!("Microsoft Platform Crypto Provider"),
            0,
        )?;

        let mut key = NCRYPT_KEY_HANDLE::default();

        let status = NCryptOpenKey(
            provider,
            &mut key,
            PCWSTR::from_raw(w!("FederatedDeviceKey").as_ptr()),
            0,
            0,
        );

        if status.is_err() {
            // Create new ECC P-256 key
            NCryptCreatePersistedKey(
                provider,
                &mut key,
                w!("ECDSA_P256"),
                PCWSTR::from_raw(w!("FederatedDeviceKey").as_ptr()),
                0,
                NCRYPT_MACHINE_KEY_FLAG,
            )?;

            NCryptFinalizeKey(key, 0)?;
        }

        Ok(key)
    }
}

fn sign_stdin() -> Result<()> {
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input)?;

    let mut hasher = Sha256::new();
    hasher.update(&input);
    let digest = hasher.finalize();

    unsafe {
        let key = open_or_create_key()?;

        let mut sig_len = 0u32;

        NCryptSignHash(
            key,
            None,
            Some(digest.as_slice()),
            None,
            &mut sig_len,
            0,
        )?;

        let mut signature = vec![0u8; sig_len as usize];

        NCryptSignHash(
            key,
            None,
            Some(digest.as_slice()),
            Some(signature.as_mut_slice()),
            &mut sig_len,
            0,
        )?;

        std::io::stdout().write_all(&signature)?;
    }

    Ok(())
}

fn export_public_key() -> Result<()> {
    unsafe {
        let key = open_or_create_key()?;

        let mut len = 0u32;

        NCryptExportKey(
            key,
            None,
            w!("PUBLICBLOB"),
            None,
            None,
            0,
            &mut len,
            0,
        )?;

        let mut buf = vec![0u8; len as usize];

        NCryptExportKey(
            key,
            None,
            w!("PUBLICBLOB"),
            None,
            Some(buf.as_mut_slice()),
            len,
            &mut len,
            0,
        )?;

        // TODO: convert blob to proper PEM
        println!("PUBLIC KEY BLOB LENGTH: {}", len);
    }

    Ok(())
}