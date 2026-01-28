use thiserror::Error;

#[allow(dead_code)]
#[derive(Error, Debug)]
pub enum OrchestratorError {
    #[error("configuration error")]
    ConfigError,

    #[error("invalid identity")]
    InvalidIdentity,

    #[error("cryptographic verification failed")]
    CryptoError,

    #[error("round not found")]
    RoundNotFound,

    #[error("ledger error")]
    LedgerError,

    #[error("internal error")]
    Internal,
}

#[allow(dead_code)]
pub type Result<T> = std::result::Result<T, OrchestratorError>;
