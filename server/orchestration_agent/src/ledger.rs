use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

fn ledger_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".federated")
        .join("logs")
        .join("audit_ledger.log")
}

pub fn append(entry: &[u8]) {
    let path = ledger_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(entry).and_then(|_| f.write_all(b"\n")) {
                tracing::error!("Ledger write failed at {:?}: {}", path, e);
            }
        }
        Err(e) => tracing::error!("Ledger open failed at {:?}: {}", path, e),
    }
}