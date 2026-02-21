use std::io::{Read, Write};
use windows::core::*;
use windows::Win32::Security::Cryptography::*;
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

        let open_result = NCryptOpenKey(
            provider,
            &mut key,
            w!("FederatedDeviceKey"),
            CERT_KEY_SPEC(0),
            NCRYPT_FLAGS(0),
        );

        if open_result.is_err() {
            NCryptCreatePersistedKey(
                provider,
                &mut key,
                w!("ECDSA_P256"),
                w!("FederatedDeviceKey"),
                CERT_KEY_SPEC(0),
                NCRYPT_MACHINE_KEY_FLAG,
            )?;

            NCryptFinalizeKey(key, NCRYPT_FLAGS(0))?;
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
            &digest,
            None,
            &mut sig_len,
            NCRYPT_FLAGS(0),
        )?;

        let mut signature = vec![0u8; sig_len as usize];

        NCryptSignHash(
            key,
            None,
            &digest,
            Some(&mut signature),
            &mut sig_len,
            NCRYPT_FLAGS(0),
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
            &mut len,
            NCRYPT_FLAGS(0),
        )?;

        let mut buf = vec![0u8; len as usize];

        NCryptExportKey(
            key,
            None,
            w!("PUBLICBLOB"),
            None,
            Some(&mut buf),
            &mut len,
            NCRYPT_FLAGS(0),
        )?;

        println!("PUBLIC KEY BLOB SIZE: {}", len);
    }

    Ok(())
}